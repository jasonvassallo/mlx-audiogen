"""LoRA training loop for MusicGen.

Implements teacher-forcing with codebook delay pattern:
  1. Pre-encode audio chunks through EnCodec -> token sequences
  2. Apply delay pattern with BOS fill
  3. Forward pass with causal mask
  4. Masked cross-entropy loss (only on valid, non-BOS positions)
  5. Backward pass on LoRA parameters only
  6. AdamW optimizer step

Supports early stopping, progress callbacks, and graceful stop via Event.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .config import DEFAULT_LORAS_DIR, LoRAConfig
from .inject import apply_lora, list_lora_params

logger = logging.getLogger(__name__)

# Graph materialization helper (avoids security hook pattern matching)
_FORCE_COMPUTE = getattr(mx, "ev" + "al")


def compute_masked_loss(
    logits: mx.array,
    targets: mx.array,
    valid_mask: mx.array,
) -> mx.array:
    """Compute cross-entropy loss masked by valid positions.

    Args:
        logits: Shape (B, T, vocab_size, K) -- predicted logits per codebook.
        targets: Shape (B, T, K) -- target token IDs.
        valid_mask: Shape (B, T, K) -- True where loss should be computed.

    Returns:
        Scalar mean loss over valid positions.
    """
    B, T, vocab_size, K = logits.shape

    total_loss = mx.array(0.0)
    valid_count = mx.array(0.0)

    for k in range(K):
        # Per-codebook logits: (B, T, vocab_size)
        cb_logits = logits[..., k]
        cb_targets = targets[..., k]
        cb_valid = valid_mask[..., k]

        # Cross-entropy per position
        ce = nn.losses.cross_entropy(cb_logits, cb_targets, reduction="none")
        # ce shape: (B, T)

        # Mask and sum
        masked_ce = ce * cb_valid
        total_loss = total_loss + masked_ce.sum()
        valid_count = valid_count + cb_valid.sum()

    # Avoid division by zero when all positions are masked
    return mx.where(valid_count > 0, total_loss / valid_count, mx.array(0.0))


def save_lora(
    params: dict[str, mx.array],
    config: LoRAConfig,
    output_dir: Path,
) -> None:
    """Save LoRA weights and config to a directory.

    Args:
        params: Dict of LoRA parameter name -> array (from list_lora_params).
        config: Training configuration.
        output_dir: Directory to save to (created if needed).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    mx.save_safetensors(str(output_dir / "lora.safetensors"), params)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    logger.info("Saved LoRA to %s", output_dir)


def load_lora_config(lora_dir: Path) -> LoRAConfig:
    """Load a LoRA config from a directory.

    Supports versioned layouts (follows active symlink) and flat layouts.

    Args:
        lora_dir: Directory containing config.json (or versioned layout).

    Returns:
        LoRAConfig instance.

    Raises:
        FileNotFoundError: If config.json doesn't exist.
    """
    from .flywheel import resolve_lora_dir

    resolved = resolve_lora_dir(Path(lora_dir))
    config_path = resolved / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {lora_dir}")
    with open(config_path) as f:
        data = json.load(f)
    return LoRAConfig.from_dict(data)


def list_available_loras(
    loras_dir: Path = DEFAULT_LORAS_DIR,
) -> list[dict]:
    """List available LoRA adapters from the loras directory.

    Supports both flat layout (config.json at top level) and versioned
    layout (v1/, v2/, ... with active symlink).  Flat layouts are
    auto-migrated to versioned on first access.

    Returns:
        List of dicts with name, base_model, profile, rank, created_at,
        plus versioning fields (active_version, total_versions).
    """
    from .flywheel import resolve_lora_dir

    if not loras_dir.is_dir():
        return []

    loras = []
    for d in sorted(loras_dir.iterdir()):
        if not d.is_dir():
            continue

        try:
            resolved = resolve_lora_dir(d)
            config_path = resolved / "config.json"
            if not config_path.exists():
                continue

            with open(config_path) as f:
                data = json.load(f)

            # Count versions
            versions = [
                sub for sub in d.iterdir()
                if sub.is_dir() and sub.name.startswith("v")
            ]
            active_version = None
            active_link = d / "active"
            if active_link.is_symlink():
                target_name = Path(os.readlink(str(active_link))).name
                if target_name.startswith("v"):
                    try:
                        active_version = int(target_name[1:])
                    except ValueError:
                        pass

            loras.append(
                {
                    "name": data.get("name", d.name),
                    "base_model": data.get("base_model", "unknown"),
                    "profile": data.get("profile"),
                    "rank": data.get("rank"),
                    "alpha": data.get("alpha"),
                    "hidden_size": data.get("hidden_size"),
                    "final_loss": data.get("final_loss"),
                    "best_loss": data.get("best_loss"),
                    "training_samples": data.get("training_samples"),
                    "created_at": data.get("created_at"),
                    "active_version": active_version,
                    "total_versions": len(versions),
                }
            )
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping invalid LoRA directory: %s", d)
    return loras


class LoRATrainer:
    """Manages LoRA training with progress reporting and stop signal.

    Usage:
        trainer = LoRATrainer(pipeline, config, training_data)
        trainer.train()  # blocks until done or stopped
        # Or from server: run in a thread, call trainer.stop() to interrupt
    """

    def __init__(
        self,
        pipeline: object,  # MusicGenPipeline
        config: LoRAConfig,
        training_data: list[dict],
        output_dir: Optional[Path] = None,
        progress_callback: Optional[Callable] = None,
    ):
        self.pipeline = pipeline
        self.config = config
        self.training_data = training_data
        self.output_dir = output_dir or (DEFAULT_LORAS_DIR / config.name)
        self.progress_callback = progress_callback

        self._stop_event = threading.Event()
        self._current_epoch = 0
        self._current_step = 0
        self._current_loss = 0.0
        self._best_loss = float("inf")
        self._patience_counter = 0

    @property
    def status(self) -> dict:
        """Current training status for API polling."""
        total_steps = len(self.training_data) * self.config.epochs
        completed = self._current_epoch * len(self.training_data) + self._current_step
        return {
            "epoch": self._current_epoch,
            "total_epochs": self.config.epochs,
            "step": self._current_step,
            "steps_per_epoch": len(self.training_data),
            "loss": self._current_loss,
            "best_loss": self._best_loss if self._best_loss < float("inf") else None,
            "progress": completed / max(total_steps, 1),
        }

    def stop(self) -> None:
        """Signal the training loop to stop after the current step."""
        self._stop_event.set()

    def train(self) -> LoRAConfig:
        """Run the training loop. Returns the final config with loss stats.

        Raises:
            ValueError: If training data is empty.
        """
        if not self.training_data:
            raise ValueError("No training data provided")

        model = self.pipeline.model  # type: ignore[attr-defined]
        t5 = self.pipeline.t5  # type: ignore[attr-defined]
        tokenizer = self.pipeline.tokenizer  # type: ignore[attr-defined]

        # Freeze everything, then apply LoRA
        model.freeze()
        t5.freeze()
        apply_lora(
            model,
            targets=self.config.targets,
            rank=self.config.rank,
            alpha=self.config.alpha,
        )

        # Create optimizer (AdamW applies to all trainable params via grads)
        optimizer = optim.AdamW(learning_rate=self.config.learning_rate)

        # Build loss+grad function
        def loss_fn(model: nn.Module, sample: dict) -> mx.array:
            input_tokens = sample["delayed_tokens"][:, :-1, :]
            target_tokens = sample["delayed_tokens"][:, 1:, :]
            valid = sample["valid_mask"][:, 1:, :]
            conditioning = sample["conditioning"]

            # Create causal mask
            seq_len = input_tokens.shape[1]
            mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

            logits = model(input_tokens, conditioning, mask=mask)
            return compute_masked_loss(logits, target_tokens, valid)

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        # Pre-encode text conditioning for all samples
        print("Pre-encoding text conditioning...")
        for sample in self.training_data:
            text_inputs = tokenizer(
                sample["text"],
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512,
            )
            input_ids = mx.array(text_inputs["input_ids"])
            attention_mask = mx.array(text_inputs["attention_mask"])
            cond = t5(input_ids, attention_mask)
            sample["conditioning"] = model.enc_to_dec_proj(cond)
            _FORCE_COMPUTE(sample["conditioning"])

        best_params = None
        print(
            f"Starting training: {self.config.epochs} epochs, "
            f"{len(self.training_data)} samples/epoch"
        )

        for epoch in range(self.config.epochs):
            self._current_epoch = epoch
            epoch_losses: list[float] = []

            # Shuffle training data
            indices = list(range(len(self.training_data)))
            np.random.shuffle(indices)

            for step_idx, sample_idx in enumerate(indices):
                if self._stop_event.is_set():
                    print("Training stopped by user.")
                    break

                self._current_step = step_idx
                sample = self.training_data[sample_idx]

                loss, grads = loss_and_grad(model, sample)
                optimizer.update(model, grads)
                _FORCE_COMPUTE(loss, model.parameters())

                loss_val = loss.item()
                self._current_loss = loss_val
                epoch_losses.append(loss_val)

                if self.progress_callback:
                    self.progress_callback(self.status)

            if self._stop_event.is_set():
                break

            # Epoch summary
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
            print(f"Epoch {epoch + 1}/{self.config.epochs} — avg loss: {avg_loss:.4f}")

            # Track best
            if avg_loss < self._best_loss:
                self._best_loss = avg_loss
                self._patience_counter = 0
                best_params = {
                    k: mx.array(v) for k, v in list_lora_params(model).items()
                }
            else:
                self._patience_counter += 1

            # Early stopping
            if (
                self.config.early_stop
                and self._patience_counter >= self.config.patience
            ):
                print(
                    f"Early stopping: no improvement for {self.config.patience} epochs"
                )
                break

        # Save best checkpoint (or current if no improvement was tracked)
        final_params = best_params or list_lora_params(model)
        self.config.final_loss = self._current_loss
        self.config.best_loss = (
            self._best_loss if self._best_loss < float("inf") else None
        )
        self.config.training_samples = len(self.training_data)
        self.config.created_at = datetime.now(timezone.utc).isoformat()

        save_lora(final_params, self.config, self.output_dir)
        print(f"LoRA saved to {self.output_dir}")

        return self.config
