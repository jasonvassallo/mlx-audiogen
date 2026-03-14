"""Tests for LoRA training loop."""

import json
import tempfile
from pathlib import Path

import mlx.core as mx

from mlx_audiogen.lora.config import LoRAConfig
from mlx_audiogen.lora.trainer import (
    compute_masked_loss,
    list_available_loras,
    load_lora_config,
    save_lora,
)


def test_compute_masked_loss():
    """Masked cross-entropy loss ignores invalid positions."""
    # Logits: (B=1, T=3, vocab=4, K=2)
    logits = mx.zeros((1, 3, 4, 2))
    # Target: (B=1, T=3, K=2) — all zeros (class 0)
    targets = mx.zeros((1, 3, 2), dtype=mx.int32)
    # Valid mask: only first 2 positions valid for codebook 0, first for codebook 1
    valid = mx.array([[[True, True], [True, False], [False, False]]])
    loss = compute_masked_loss(logits, targets, valid)
    assert loss.shape == ()  # scalar
    assert loss.item() > 0  # cross-entropy of uniform logits > 0


def test_compute_masked_loss_all_masked():
    """If all positions are masked, loss should be zero."""
    logits = mx.zeros((1, 3, 4, 2))
    targets = mx.zeros((1, 3, 2), dtype=mx.int32)
    valid = mx.zeros((1, 3, 2), dtype=mx.bool_)
    loss = compute_masked_loss(logits, targets, valid)
    assert loss.item() == 0.0


def test_save_and_load_lora_config():
    """LoRA config round-trips through save/load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = LoRAConfig(
            name="test-style",
            base_model="musicgen-small",
            hidden_size=1024,
            rank=16,
            alpha=32.0,
            targets=["self_attn.q_proj", "self_attn.v_proj"],
            profile="balanced",
            final_loss=2.5,
            best_loss=2.3,
            training_samples=10,
        )
        out_dir = Path(tmpdir) / "test-style"
        out_dir.mkdir()
        # Save config
        with open(out_dir / "config.json", "w") as f:
            json.dump(cfg.to_dict(), f)
        # Load it back
        loaded = load_lora_config(out_dir)
        assert loaded.name == "test-style"
        assert loaded.rank == 16
        assert loaded.hidden_size == 1024
        assert loaded.final_loss == 2.5


def test_save_lora_creates_files():
    """save_lora creates lora.safetensors and config.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "my-lora"
        cfg = LoRAConfig(
            name="my-lora",
            base_model="musicgen-small",
            hidden_size=64,
            rank=8,
            alpha=16.0,
            targets=["self_attn.q_proj"],
        )
        # Fake LoRA params
        params = {
            "layers.0.self_attn.q_proj.lora_a": mx.zeros((64, 8)),
            "layers.0.self_attn.q_proj.lora_b": mx.zeros((8, 64)),
        }
        save_lora(params, cfg, out_dir)
        assert (out_dir / "lora.safetensors").exists()
        assert (out_dir / "config.json").exists()
        # Verify config contents
        with open(out_dir / "config.json") as f:
            data = json.load(f)
        assert data["name"] == "my-lora"
        assert data["rank"] == 8


def test_load_lora_config_missing():
    """Loading from nonexistent directory raises FileNotFoundError."""
    import pytest

    with pytest.raises(FileNotFoundError):
        load_lora_config(Path("/nonexistent/path"))


def test_list_available_loras_empty():
    """Empty directory returns empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = list_available_loras(Path(tmpdir))
        assert result == []


def test_list_available_loras_finds_valid():
    """Discovers valid LoRA directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lora_dir = Path(tmpdir) / "test-lora"
        lora_dir.mkdir()
        cfg = {"name": "test-lora", "base_model": "musicgen-small", "rank": 16}
        (lora_dir / "config.json").write_text(json.dumps(cfg))
        result = list_available_loras(Path(tmpdir))
        assert len(result) == 1
        assert result[0]["name"] == "test-lora"
