"""Tests for LoRA injection: LoRALinear, apply_lora, remove_lora."""

import mlx.core as mx
import mlx.nn as nn

from mlx_audiogen.lora.inject import (
    LoRALinear,
    apply_lora,
    list_lora_params,
    remove_lora,
)


class DummyAttention(nn.Module):
    """Minimal attention module matching MusicGen structure."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)


class DummyBlock(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.self_attn = DummyAttention(dim)
        self.encoder_attn = DummyAttention(dim)


class DummyModel(nn.Module):
    def __init__(self, dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.layers = [DummyBlock(dim) for _ in range(num_layers)]
        self.hidden_size = dim


def test_lora_linear_preserves_output_shape():
    """LoRALinear output shape matches base Linear."""
    base = nn.Linear(64, 64, bias=False)
    lora = LoRALinear(base, rank=8, alpha=16.0)
    x = mx.ones((1, 10, 64))
    out = lora(x)
    assert out.shape == (1, 10, 64)


def test_lora_linear_starts_at_zero():
    """LoRA contribution is zero initially (B is zero-initialized)."""
    base = nn.Linear(64, 64, bias=False)
    lora = LoRALinear(base, rank=8, alpha=16.0)
    x = mx.ones((1, 5, 64))
    base_out = base(x)
    lora_out = lora(x)
    assert mx.allclose(base_out, lora_out, atol=1e-6).item()


def test_lora_linear_base_is_frozen():
    """Base weights should be frozen, only lora_a/lora_b are trainable."""
    base = nn.Linear(64, 64, bias=False)
    lora = LoRALinear(base, rank=8)
    # Flatten trainable params — should only contain lora_a and lora_b
    trainable = nn.utils.tree_flatten(lora.trainable_parameters())
    trainable_keys = [k for k, _ in trainable]
    assert any("lora_a" in k for k in trainable_keys)
    assert any("lora_b" in k for k in trainable_keys)
    # Base weight should NOT be trainable
    assert not any("base" in k for k in trainable_keys)


def test_apply_lora_targets_q_v():
    """apply_lora replaces targeted layers with LoRALinear."""
    model = DummyModel(dim=64, num_layers=2)
    apply_lora(
        model, targets=["self_attn.q_proj", "self_attn.v_proj"], rank=8, alpha=16.0
    )
    # q_proj and v_proj should be LoRALinear
    assert isinstance(model.layers[0].self_attn.q_proj, LoRALinear)
    assert isinstance(model.layers[0].self_attn.v_proj, LoRALinear)
    # k_proj and out_proj should remain nn.Linear
    assert isinstance(model.layers[0].self_attn.k_proj, nn.Linear)
    assert not isinstance(model.layers[0].self_attn.k_proj, LoRALinear)
    # encoder_attn should be untouched
    assert isinstance(model.layers[0].encoder_attn.q_proj, nn.Linear)
    assert not isinstance(model.layers[0].encoder_attn.q_proj, LoRALinear)


def test_apply_lora_all_layers():
    """apply_lora applies to all layers in model.layers."""
    model = DummyModel(dim=64, num_layers=3)
    apply_lora(model, targets=["self_attn.q_proj"], rank=4, alpha=8.0)
    for i in range(3):
        assert isinstance(model.layers[i].self_attn.q_proj, LoRALinear)


def test_remove_lora_restores_base():
    """remove_lora restores original nn.Linear layers."""
    model = DummyModel(dim=64, num_layers=2)
    # Get original output from layer 0 q_proj
    orig_weight = model.layers[0].self_attn.q_proj.weight
    apply_lora(model, targets=["self_attn.q_proj"], rank=8, alpha=16.0)
    assert isinstance(model.layers[0].self_attn.q_proj, LoRALinear)
    remove_lora(model)
    assert isinstance(model.layers[0].self_attn.q_proj, nn.Linear)
    assert not isinstance(model.layers[0].self_attn.q_proj, LoRALinear)
    # Weight should be the same object
    assert model.layers[0].self_attn.q_proj.weight is orig_weight


def test_list_lora_params():
    """list_lora_params returns only LoRA A/B parameter keys."""
    model = DummyModel(dim=64, num_layers=2)
    apply_lora(
        model, targets=["self_attn.q_proj", "self_attn.v_proj"], rank=8, alpha=16.0
    )
    params = list_lora_params(model)
    # Should have entries with lora_a and lora_b
    keys = list(params.keys())
    assert len(keys) > 0
    for k in keys:
        assert "lora_a" in k or "lora_b" in k


def test_musicgen_model_mask_parameter():
    """MusicGenModel.__call__ accepts optional mask parameter."""
    from mlx_audiogen.models.musicgen.config import DecoderConfig, MusicGenConfig
    from mlx_audiogen.models.musicgen.model import MusicGenModel

    # Tiny config for testing
    cfg = MusicGenConfig(
        decoder=DecoderConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            ffn_dim=128,
            num_codebooks=4,
            vocab_size=32,
        )
    )
    model = MusicGenModel(cfg)
    tokens = mx.zeros((1, 5, 4), dtype=mx.int32)  # (B, T, K)
    cond = mx.zeros((1, 3, 64))  # (B, cond_len, hidden)

    # Without mask (existing behavior)
    logits1 = model(tokens, cond)
    assert logits1.shape == (1, 5, 32, 4)  # (B, T, vocab, K)

    # With causal mask
    mask = nn.MultiHeadAttention.create_additive_causal_mask(5)
    logits2 = model(tokens, cond, mask=mask)
    assert logits2.shape == (1, 5, 32, 4)


def test_apply_lora_encoder_attn():
    """apply_lora can target encoder_attn projections."""
    model = DummyModel(dim=64, num_layers=2)
    apply_lora(
        model,
        targets=["encoder_attn.q_proj", "encoder_attn.v_proj"],
        rank=8,
        alpha=16.0,
    )
    assert isinstance(model.layers[0].encoder_attn.q_proj, LoRALinear)
    assert isinstance(model.layers[0].encoder_attn.v_proj, LoRALinear)
    assert isinstance(model.layers[0].self_attn.q_proj, nn.Linear)
    assert not isinstance(model.layers[0].self_attn.q_proj, LoRALinear)
