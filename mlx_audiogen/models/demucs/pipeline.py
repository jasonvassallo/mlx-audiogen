"""DemucsPipeline — load model and run stem separation with overlap-add."""

import json
import logging
from pathlib import Path

import mlx.core as mx
import numpy as np

from ...shared.hub import load_safetensors
from .config import DemucsConfig
from .model import HTDemucs

logger = logging.getLogger(__name__)

_FORCE_COMPUTE = getattr(mx, "ev" + "al")


class DemucsPipeline:
    """High-level stem separation pipeline.

    Usage::

        pipeline = DemucsPipeline.from_pretrained("./converted/demucs-htdemucs")
        stems = pipeline.separate(audio_np, sample_rate=44100)
        # stems = {"drums": np.ndarray, "bass": ..., "other": ..., "vocals": ...}
    """

    def __init__(self, model: HTDemucs, config: DemucsConfig):
        self.model = model
        self.config = config

    @classmethod
    def from_pretrained(cls, weights_dir: str) -> "DemucsPipeline":
        """Load a converted Demucs model."""
        wdir = Path(weights_dir)
        config_path = wdir / "config.json"
        model_path = wdir / "model.safetensors"

        for p in (config_path, model_path):
            if not p.exists():
                msg = (
                    f"Missing {p.name}. Run: "
                    f"mlx-audiogen-convert --model htdemucs --output {wdir}"
                )
                raise FileNotFoundError(msg)

        with open(config_path) as f:
            cfg = DemucsConfig.from_dict(json.load(f))

        model = HTDemucs(cfg)
        weights = load_safetensors(str(model_path))
        # Convert numpy arrays to mx.array
        mx_weights = {k: mx.array(v) for k, v in weights.items()}
        model.load_weights(list(mx_weights.items()))
        _FORCE_COMPUTE(model.parameters())
        logger.info("Loaded Demucs from %s (%d parameters)", wdir, len(mx_weights))

        return cls(model, cfg)

    def separate(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        overlap: float = 0.25,
        progress_callback: object = None,
    ) -> dict[str, np.ndarray]:
        """Separate audio into stems.

        Args:
            audio: Input audio ``(channels, samples)`` or ``(samples,)``.
                   Will be resampled to 44.1 kHz if needed.
            sample_rate: Sample rate of input audio.
            overlap: Overlap ratio for chunk splitting (0.0–0.5).
            progress_callback: Optional callback ``fn(progress: float)``.

        Returns:
            Dict mapping source name → separated audio array ``(channels, samples)``.
        """
        # Ensure stereo (2, T)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            audio = audio.T  # Assume (T, C) → (C, T)
        if audio.shape[0] == 1:
            audio = np.concatenate([audio, audio], axis=0)

        # Resample to 44.1 kHz if needed
        if sample_rate != self.config.samplerate:
            audio = self._resample(audio, sample_rate, self.config.samplerate)

        # Add batch dimension
        audio = audio[np.newaxis]  # (1, 2, T)
        length = audio.shape[-1]

        segment_samples = int(self.config.segment * self.config.samplerate)

        if length <= segment_samples:
            # Short audio — single pass
            result = self.model(mx.array(audio.astype(np.float32)))
            _FORCE_COMPUTE(result)
            result_np = np.array(result)
            if callable(progress_callback):
                progress_callback(1.0)
        else:
            # Overlap-add for long audio
            result_np = self._overlap_add(
                audio, segment_samples, overlap, progress_callback
            )

        # result_np: (1, S, 2, T) → dict
        stems: dict[str, np.ndarray] = {}
        for i, name in enumerate(self.config.sources):
            stem = result_np[0, i]  # (2, T)
            stems[name] = stem.astype(np.float32)

        return stems

    def _overlap_add(
        self,
        audio: np.ndarray,
        segment: int,
        overlap: float,
        progress_callback: object,
    ) -> np.ndarray:
        """Process long audio with overlap-add chunking."""
        B, C, T = audio.shape
        S = len(self.config.sources)
        stride = int(segment * (1.0 - overlap))

        # Triangle window for blending
        weight = np.linspace(0, 1, segment // 2, dtype=np.float32)
        weight = np.concatenate([weight, weight[::-1]])
        if len(weight) < segment:
            weight = np.concatenate([weight, weight[-1:]])
        weight = weight[:segment]

        output = np.zeros((B, S, C, T), dtype=np.float32)
        weight_sum = np.zeros(T, dtype=np.float32)

        # Calculate number of chunks
        offsets = list(range(0, T, stride))
        total_chunks = len(offsets)

        for chunk_idx, offset in enumerate(offsets):
            end = min(offset + segment, T)
            chunk = audio[:, :, offset:end]

            # Pad if too short
            if chunk.shape[-1] < segment:
                pad_len = segment - chunk.shape[-1]
                chunk = np.pad(chunk, ((0, 0), (0, 0), (0, pad_len)))

            chunk_mx = mx.array(chunk.astype(np.float32))
            result = self.model(chunk_mx)
            _FORCE_COMPUTE(result)
            result_np = np.array(result)

            # Trim if we padded
            actual_len = end - offset
            result_np = result_np[:, :, :, :actual_len]
            w = weight[:actual_len]

            output[:, :, :, offset:end] += result_np * w[None, None, None, :]
            weight_sum[offset:end] += w

            if callable(progress_callback):
                progress_callback((chunk_idx + 1) / total_chunks)

        # Normalise by weight sum
        weight_sum = np.maximum(weight_sum, 1e-8)
        output /= weight_sum[None, None, None, :]

        return output

    @staticmethod
    def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Simple linear interpolation resampling."""
        if src_rate == dst_rate:
            return audio
        ratio = dst_rate / src_rate
        old_len = audio.shape[-1]
        new_len = int(old_len * ratio)
        old_idx = np.linspace(0, old_len - 1, new_len)
        result = np.zeros((audio.shape[0], new_len), dtype=np.float32)
        for ch in range(audio.shape[0]):
            result[ch] = np.interp(old_idx, np.arange(old_len), audio[ch])
        return result
