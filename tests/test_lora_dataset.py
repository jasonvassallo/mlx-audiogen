"""Tests for LoRA training dataset loading."""

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_audiogen.lora.dataset import apply_delay_pattern, chunk_audio, scan_dataset


def _make_wav(path: Path, duration_s: float = 2.0, sr: int = 32000):
    """Create a test WAV file with a sine wave."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t) * 0.5
    sf.write(str(path), audio, sr)


def test_scan_dataset_with_metadata():
    """Scan a directory with metadata.jsonl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        _make_wav(p / "track1.wav")
        _make_wav(p / "track2.wav")
        meta = [
            {"file": "track1.wav", "text": "upbeat house"},
            {"file": "track2.wav", "text": "chill ambient"},
        ]
        (p / "metadata.jsonl").write_text("\n".join(json.dumps(m) for m in meta))
        entries = scan_dataset(p)
        assert len(entries) == 2
        assert entries[0]["text"] == "upbeat house"
        assert entries[1]["text"] == "chill ambient"


def test_scan_dataset_filename_fallback():
    """Files without metadata get descriptions from filenames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        _make_wav(p / "deep_bass_groove.wav")
        entries = scan_dataset(p)
        assert len(entries) == 1
        assert entries[0]["text"] == "deep bass groove"


def test_scan_dataset_mixed_metadata():
    """Some files in metadata, others use filename fallback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        _make_wav(p / "track1.wav")
        _make_wav(p / "no_meta.wav")
        meta = [{"file": "track1.wav", "text": "described track"}]
        (p / "metadata.jsonl").write_text(json.dumps(meta[0]))
        entries = scan_dataset(p)
        assert len(entries) == 2
        texts = {e["text"] for e in entries}
        assert "described track" in texts
        assert "no meta" in texts


def test_scan_dataset_skips_bad_jsonl():
    """Malformed JSON lines are skipped with no crash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        _make_wav(p / "track1.wav")
        (p / "metadata.jsonl").write_text(
            '{"file": "track1.wav", "text": "good"}\nnot json at all\n'
        )
        entries = scan_dataset(p)
        assert len(entries) == 1
        assert entries[0]["text"] == "good"


def test_chunk_audio_short():
    """Audio shorter than chunk size is used whole."""
    audio = np.zeros(32000, dtype=np.float32)  # 1 second
    chunks = chunk_audio(audio, sample_rate=32000, chunk_seconds=10.0)
    assert len(chunks) == 1
    assert len(chunks[0]) == 32000


def test_chunk_audio_exact():
    """Audio exactly chunk_seconds produces one chunk."""
    audio = np.zeros(320000, dtype=np.float32)  # 10 seconds
    chunks = chunk_audio(audio, sample_rate=32000, chunk_seconds=10.0)
    assert len(chunks) == 1
    assert len(chunks[0]) == 320000


def test_chunk_audio_multiple():
    """Long audio is split into multiple chunks."""
    audio = np.zeros(640000, dtype=np.float32)  # 20 seconds
    chunks = chunk_audio(audio, sample_rate=32000, chunk_seconds=10.0)
    assert len(chunks) == 2
    assert len(chunks[0]) == 320000
    assert len(chunks[1]) == 320000


def test_chunk_audio_discard_tiny_remainder():
    """Last chunk shorter than half chunk size is discarded."""
    # 11 seconds -> chunk at 10s leaves 1s remainder (< 5s half) -> discard
    audio = np.zeros(352000, dtype=np.float32)  # 11 seconds
    chunks = chunk_audio(audio, sample_rate=32000, chunk_seconds=10.0)
    assert len(chunks) == 1


def test_chunk_audio_keep_large_remainder():
    """Last chunk >= half chunk size is kept."""
    # 16 seconds -> chunk at 10s leaves 6s remainder (>= 5s half) -> keep
    audio = np.zeros(512000, dtype=np.float32)  # 16 seconds
    chunks = chunk_audio(audio, sample_rate=32000, chunk_seconds=10.0)
    assert len(chunks) == 2


def test_chunk_max_40s():
    """Chunk size is capped at 40 seconds."""
    audio = np.zeros(32000 * 120, dtype=np.float32)  # 120 seconds
    chunks = chunk_audio(audio, sample_rate=32000, chunk_seconds=40.0)
    # 120s / 40s = 3 chunks
    assert len(chunks) == 3
    assert len(chunks[0]) == 32000 * 40


def test_delay_pattern_shape():
    """Delay pattern expands sequence by K-1."""
    tokens = mx.zeros((1, 10, 4), dtype=mx.int32)  # B=1, T=10, K=4
    delayed, valid = apply_delay_pattern(tokens, num_codebooks=4, bos_token_id=2048)
    assert delayed.shape == (1, 13, 4)  # T + K - 1 = 13
    assert valid.shape == (1, 13, 4)


def test_delay_pattern_bos_fill():
    """Early positions for later codebooks are filled with BOS."""
    tokens = mx.ones((1, 5, 4), dtype=mx.int32)
    delayed, valid = apply_delay_pattern(tokens, num_codebooks=4, bos_token_id=2048)
    # Codebook 0: no delay, starts at position 0
    assert delayed[0, 0, 0].item() == 1
    # Codebook 1: delayed by 1, position 0 should be BOS
    assert delayed[0, 0, 1].item() == 2048
    # Codebook 1: position 1 should have real data
    assert delayed[0, 1, 1].item() == 1
    # Codebook 3: delayed by 3, positions 0-2 should be BOS
    assert delayed[0, 0, 3].item() == 2048
    assert delayed[0, 1, 3].item() == 2048
    assert delayed[0, 2, 3].item() == 2048
    assert delayed[0, 3, 3].item() == 1


def test_delay_pattern_valid_mask():
    """Valid mask correctly marks non-BOS positions."""
    tokens = mx.ones((1, 5, 4), dtype=mx.int32)
    delayed, valid = apply_delay_pattern(tokens, num_codebooks=4, bos_token_id=2048)
    # Codebook 0: valid from position 0
    assert valid[0, 0, 0].item() == True  # noqa: E712
    # Codebook 1: valid from position 1
    assert valid[0, 0, 1].item() == False  # noqa: E712
    assert valid[0, 1, 1].item() == True  # noqa: E712
    # Codebook 3: valid from position 3
    assert valid[0, 2, 3].item() == False  # noqa: E712
    assert valid[0, 3, 3].item() == True  # noqa: E712
