"""LoRA training dataset: scan directories, load audio, chunk for training.

Supports two input modes:
  1. metadata.jsonl with {"file": "name.wav", "text": "description"} per line
  2. Filename fallback: underscores/hyphens replaced with spaces

Audio is chunked into segments (default 10s, max 40s) to fit within
MusicGen's position limit (2048 tokens at 50Hz = 41s).
"""

import json
import logging
from math import gcd
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".aiff"}
MAX_CHUNK_SECONDS = 40.0  # MusicGen position limit


def scan_dataset(data_dir: Path) -> list[dict[str, str]]:
    """Scan a directory for audio files and their text descriptions.

    Args:
        data_dir: Path to directory with audio files + optional metadata.jsonl.

    Returns:
        List of dicts with "file" (absolute path) and "text" keys.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all audio files
    audio_files = sorted(
        f
        for f in data_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not audio_files:
        raise ValueError(f"No audio files found in {data_dir}")

    # Load metadata if present
    metadata: dict[str, str] = {}
    meta_path = data_dir / "metadata.jsonl"
    if meta_path.exists():
        for line_num, line in enumerate(meta_path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d", line_num)
                continue
            if "file" not in entry or "text" not in entry:
                logger.warning(
                    "Skipping entry at line %d: missing 'file' or 'text'",
                    line_num,
                )
                continue
            metadata[entry["file"]] = entry["text"]

    # Build entries: metadata text or filename fallback
    entries = []
    for audio_file in audio_files:
        if audio_file.name in metadata:
            text = metadata[audio_file.name]
        else:
            # Filename fallback: replace _ and - with spaces, strip extension
            text = audio_file.stem.replace("_", " ").replace("-", " ")
        entries.append({"file": str(audio_file), "text": text})

    return entries


def chunk_audio(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float = 10.0,
) -> list[np.ndarray]:
    """Split audio into fixed-size chunks.

    Args:
        audio: 1D mono audio array.
        sample_rate: Sample rate in Hz.
        chunk_seconds: Target chunk duration (capped at MAX_CHUNK_SECONDS).

    Returns:
        List of audio chunks as numpy arrays.
    """
    chunk_seconds = min(chunk_seconds, MAX_CHUNK_SECONDS)
    chunk_samples = int(chunk_seconds * sample_rate)

    if len(audio) <= chunk_samples:
        return [audio]

    chunks = []
    offset = 0
    while offset < len(audio):
        end = offset + chunk_samples
        chunk = audio[offset:end]

        if len(chunk) == chunk_samples:
            chunks.append(chunk)
        else:
            # Remainder: keep if >= half chunk size, discard otherwise
            if len(chunk) >= chunk_samples // 2:
                chunks.append(chunk)
        offset = end

    return chunks


def load_and_prepare_audio(
    file_path: str,
    target_sr: int = 32000,
) -> np.ndarray:
    """Load an audio file, convert to mono, resample to target rate.

    Uses FFT sinc resampling (alias-free, same as demucs pipeline) to
    preserve audio quality. Training data quality directly affects LoRA
    adapter quality.

    Args:
        file_path: Path to audio file.
        target_sr: Target sample rate.

    Returns:
        1D mono float32 numpy array at target_sr.
    """
    audio, sr = sf.read(file_path, dtype="float32")

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed (FFT sinc — alias-free, no boundary artifacts)
    if sr != target_sr:
        audio = _fft_resample(audio, sr, target_sr)

    return audio


def _fft_resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """FFT-based sinc resampling with reflect-padding (alias-free).

    Same algorithm as demucs/pipeline.py — uses reflect-pad to eliminate
    boundary discontinuity artifacts that plain FFT resampling causes.
    """
    if src_rate == dst_rate:
        return audio

    g = gcd(src_rate, dst_rate)
    up = dst_rate // g
    down = src_rate // g

    old_len = len(audio)
    new_len = int(old_len * up / down)

    pad_samples = min(old_len, 4096)
    x_padded = np.pad(audio, pad_samples, mode="reflect")

    n = len(x_padded)
    n_out_padded = int(n * up / down)
    spectrum = np.fft.rfft(x_padded)

    n_freq_out = n_out_padded // 2 + 1
    n_freq_in = len(spectrum)

    new_spectrum = np.zeros(n_freq_out, dtype=np.complex64)
    copy_bins = min(n_freq_in, n_freq_out)
    new_spectrum[:copy_bins] = spectrum[:copy_bins]

    resampled = np.fft.irfft(new_spectrum, n=n_out_padded).astype(np.float32)
    resampled *= n_out_padded / n

    trim_start = int(pad_samples * up / down)
    return resampled[trim_start : trim_start + new_len]


def apply_delay_pattern(
    tokens: mx.array,
    num_codebooks: int,
    bos_token_id: int = 2048,
) -> tuple[mx.array, mx.array]:
    """Apply MusicGen's codebook delay pattern to ground-truth tokens.

    Codebook k is delayed by k positions. Early positions for each codebook
    are filled with bos_token_id. A validity mask marks which positions
    should contribute to the training loss.

    Args:
        tokens: Shape (B, T, K) -- raw EnCodec tokens.
        num_codebooks: Number of codebooks K.
        bos_token_id: Token ID for BOS/padding (default 2048).

    Returns:
        Tuple of (delayed_tokens, valid_mask):
          - delayed_tokens: Shape (B, T + K - 1, K)
          - valid_mask: Shape (B, T + K - 1, K), bool
    """
    B, T, K = tokens.shape
    new_T = T + num_codebooks - 1

    # Build delayed tokens and mask using numpy (small, one-time cost)
    delayed_np = np.full((B, new_T, K), bos_token_id, dtype=np.int32)
    valid_np = np.zeros((B, new_T, K), dtype=bool)

    tokens_np = np.array(tokens)
    for k in range(num_codebooks):
        delayed_np[:, k : k + T, k] = tokens_np[:, :, k]
        valid_np[:, k : k + T, k] = True

    return mx.array(delayed_np), mx.array(valid_np)
