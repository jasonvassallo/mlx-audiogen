"""Tests for LoRA configuration dataclass and training profiles."""

from mlx_audiogen.lora.config import DEFAULT_LORAS_DIR, PROFILES, LoRAConfig


def test_default_loras_dir():
    """Default LoRA directory is under ~/.mlx-audiogen/loras/."""
    assert str(DEFAULT_LORAS_DIR).endswith(".mlx-audiogen/loras")


def test_profiles_exist():
    """All three training profiles are defined."""
    assert "quick" in PROFILES
    assert "balanced" in PROFILES
    assert "deep" in PROFILES


def test_balanced_is_default():
    """Balanced profile has expected defaults."""
    p = PROFILES["balanced"]
    assert p.rank == 16
    assert p.alpha == 32.0
    assert "self_attn.q_proj" in p.targets
    assert "self_attn.v_proj" in p.targets
    assert "self_attn.out_proj" in p.targets
    assert len(p.targets) == 3


def test_quick_profile():
    """Quick profile targets only q and v in self_attn."""
    p = PROFILES["quick"]
    assert p.rank == 8
    assert p.alpha == 16.0
    assert set(p.targets) == {"self_attn.q_proj", "self_attn.v_proj"}


def test_deep_profile():
    """Deep profile targets all projections in both attention modules."""
    p = PROFILES["deep"]
    assert p.rank == 32
    assert p.alpha == 64.0
    assert len(p.targets) == 8  # 4 projections x 2 attention modules


def test_config_from_dict_roundtrip():
    """Config can be serialized to dict and back."""
    cfg = LoRAConfig(
        name="test",
        base_model="musicgen-small",
        hidden_size=1024,
        rank=16,
        alpha=32.0,
        targets=["self_attn.q_proj", "self_attn.v_proj"],
    )
    d = cfg.to_dict()
    cfg2 = LoRAConfig.from_dict(d)
    assert cfg2.name == cfg.name
    assert cfg2.rank == cfg.rank
    assert cfg2.targets == cfg.targets
    assert cfg2.hidden_size == cfg.hidden_size


def test_config_from_dict_missing_optional():
    """Config handles missing optional fields gracefully."""
    d = {
        "name": "test",
        "base_model": "musicgen-small",
        "hidden_size": 1024,
        "rank": 16,
        "alpha": 32.0,
        "targets": ["self_attn.q_proj"],
    }
    cfg = LoRAConfig.from_dict(d)
    assert cfg.profile is None
    assert cfg.final_loss is None
    assert cfg.best_loss is None
