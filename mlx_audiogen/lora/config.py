"""LoRA configuration dataclass and training profiles.

Profiles map beginner-friendly names to concrete hyperparameters:
  - quick:    rank=8,  alpha=16,  targets self_attn q+v only
  - balanced: rank=16, alpha=32,  targets self_attn q+v+out
  - deep:     rank=32, alpha=64,  targets all attention projections
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_LORAS_DIR = Path.home() / ".mlx-audiogen" / "loras"

# All valid LoRA target layer names
ALL_SELF_ATTN_TARGETS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
]
ALL_ENCODER_ATTN_TARGETS = [
    "encoder_attn.q_proj",
    "encoder_attn.k_proj",
    "encoder_attn.v_proj",
    "encoder_attn.out_proj",
]
ALL_TARGETS = ALL_SELF_ATTN_TARGETS + ALL_ENCODER_ATTN_TARGETS


@dataclass
class LoRAConfig:
    """Configuration for a LoRA adapter."""

    name: str
    base_model: str
    hidden_size: int
    rank: int = 16
    alpha: float = 32.0
    targets: list[str] = field(
        default_factory=lambda: [
            "self_attn.q_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
        ]
    )
    profile: Optional[str] = None
    chunk_seconds: float = 10.0
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 1
    early_stop: bool = True
    patience: int = 3
    final_loss: Optional[float] = None
    best_loss: Optional[float] = None
    training_samples: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "name": self.name,
            "base_model": self.base_model,
            "hidden_size": self.hidden_size,
            "rank": self.rank,
            "alpha": self.alpha,
            "targets": self.targets,
            "profile": self.profile,
            "chunk_seconds": self.chunk_seconds,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "early_stop": self.early_stop,
            "patience": self.patience,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "training_samples": self.training_samples,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoRAConfig":
        """Deserialize from dict, ignoring unknown keys."""
        known = {
            "name",
            "base_model",
            "hidden_size",
            "rank",
            "alpha",
            "targets",
            "profile",
            "chunk_seconds",
            "epochs",
            "learning_rate",
            "batch_size",
            "early_stop",
            "patience",
            "final_loss",
            "best_loss",
            "training_samples",
            "created_at",
        }
        return cls(**{k: v for k, v in d.items() if k in known})


# Training profile presets
PROFILES: dict[str, LoRAConfig] = {
    "quick": LoRAConfig(
        name="",
        base_model="",
        hidden_size=0,
        rank=8,
        alpha=16.0,
        targets=["self_attn.q_proj", "self_attn.v_proj"],
        profile="quick",
    ),
    "balanced": LoRAConfig(
        name="",
        base_model="",
        hidden_size=0,
        rank=16,
        alpha=32.0,
        targets=[
            "self_attn.q_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
        ],
        profile="balanced",
    ),
    "deep": LoRAConfig(
        name="",
        base_model="",
        hidden_size=0,
        rank=32,
        alpha=64.0,
        targets=ALL_TARGETS,
        profile="deep",
    ),
}
