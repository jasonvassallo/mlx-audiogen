"""Flywheel intelligence — connects LoRA training, taste, and enrichment.

The flywheel cycle:
  1. User generates audio with a LoRA adapter
  2. User stars generations they like → saved to ~/.mlx-audiogen/kept/
  3. When starred count hits threshold, auto-retrain fires
  4. Retrain builds cumulative dataset (library + kept gens at blend ratio)
  5. New adapter version created with changelog
  6. Taste profile refreshed → smarter suggestions
  7. Repeat
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import DEFAULT_LORAS_DIR, LoRAConfig

logger = logging.getLogger(__name__)

DEFAULT_KEPT_DIR = Path.home() / ".mlx-audiogen" / "kept"


@dataclass
class FlywheelConfig:
    """Flywheel configuration — persisted as 'flywheel' key in settings.json."""

    retrain_threshold: int = 10
    blend_ratio: int = 80  # 0-100, percentage for library tracks
    taste_refresh_interval: int = 5
    auto_retrain: bool = True
    base_collection: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrain_threshold": self.retrain_threshold,
            "blend_ratio": self.blend_ratio,
            "taste_refresh_interval": self.taste_refresh_interval,
            "auto_retrain": self.auto_retrain,
            "base_collection": self.base_collection,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FlywheelConfig:
        return cls(
            retrain_threshold=max(1, min(100, d.get("retrain_threshold", 10))),
            blend_ratio=max(0, min(100, d.get("blend_ratio", 80))),
            taste_refresh_interval=max(1, min(50, d.get("taste_refresh_interval", 5))),
            auto_retrain=d.get("auto_retrain", True),
            base_collection=d.get("base_collection"),
        )


@dataclass
class KeptGeneration:
    """Metadata for a starred generation."""

    job_id: str
    prompt: str
    model: str
    adapter_name: str
    adapter_version: Optional[str] = None
    params: dict[str, Any] = field(default_factory=dict)
    starred_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "prompt": self.prompt,
            "model": self.model,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "params": self.params,
            "starred_at": self.starred_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KeptGeneration:
        return cls(
            job_id=d.get("job_id", ""),
            prompt=d.get("prompt", ""),
            model=d.get("model", ""),
            adapter_name=d.get("adapter_name", ""),
            adapter_version=d.get("adapter_version"),
            params=d.get("params", {}),
            starred_at=d.get("starred_at"),
        )


def resolve_lora_dir(lora_dir: Path) -> Path:
    """Resolve a LoRA directory, following the active symlink if versioned.

    For versioned layouts (contains v*/ subdirs + active symlink),
    returns the path the active symlink points to.
    For flat layouts (config.json at top level), auto-migrates to v1/.

    Args:
        lora_dir: Path to a LoRA adapter directory.

    Returns:
        The resolved directory containing lora.safetensors + config.json.
    """
    lora_dir = Path(lora_dir)

    # Check for active symlink first
    active_link = lora_dir / "active"
    if active_link.is_symlink() or active_link.is_dir():
        resolved = active_link.resolve()
        if resolved.is_dir() and (resolved / "config.json").exists():
            return resolved

    # Check for flat layout (config.json at top level)
    config_at_top = lora_dir / "config.json"
    weights_at_top = lora_dir / "lora.safetensors"
    if config_at_top.exists() and weights_at_top.exists():
        # Auto-migrate to versioned layout
        return _migrate_flat_to_versioned(lora_dir)

    # No recognizable layout — return as-is (caller handles error)
    return lora_dir


def _migrate_flat_to_versioned(lora_dir: Path) -> Path:
    """Migrate a flat LoRA directory to versioned layout.

    Moves lora.safetensors + config.json into v1/, creates active symlink.
    Returns the v1/ path.
    """
    v1_dir = lora_dir / "v1"
    if v1_dir.exists():
        # Already migrated (idempotent)
        active_link = lora_dir / "active"
        if not active_link.exists():
            active_link.symlink_to("v1")
        return v1_dir

    v1_dir.mkdir(parents=True)

    # Move files
    for fname in ("lora.safetensors", "config.json"):
        src = lora_dir / fname
        if src.exists():
            shutil.move(str(src), str(v1_dir / fname))

    # Write minimal changelog
    changelog = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_version": None,
        "dataset": {
            "library_tracks": 0,
            "kept_generations": 0,
            "blend_ratio": {"library": 100, "generations": 0},
            "total_training_samples": 0,
        },
        "top_influences": {"genre": [], "mood": [], "instrument": []},
        "new_since_parent": {
            "kept_generations_added": 0,
            "enrichment_tags_added": 0,
            "library_tracks_added": 0,
        },
        "training": {
            "profile": "unknown",
            "epochs": 0,
            "best_loss": 0.0,
            "duration_seconds": 0.0,
        },
    }
    with open(v1_dir / "changelog.json", "w") as f:
        json.dump(changelog, f, indent=2)

    # Create active symlink (relative so it's portable)
    active_link = lora_dir / "active"
    active_link.symlink_to("v1")

    logger.info("Migrated flat LoRA %s to versioned layout (v1)", lora_dir.name)
    return v1_dir


class FlywheelManager:
    """Orchestrates the re-training flywheel.

    Tracks starred generations, triggers re-training at threshold,
    manages adapter versions with changelogs.
    """

    def __init__(
        self,
        config: Optional[FlywheelConfig] = None,
        loras_dir: Optional[Path] = None,
        kept_dir: Optional[Path] = None,
    ) -> None:
        self.config = config or FlywheelConfig()
        self.loras_dir = loras_dir or DEFAULT_LORAS_DIR
        self.kept_dir = kept_dir or DEFAULT_KEPT_DIR

    def _adapter_kept_dir(self, adapter_name: str) -> Path:
        """Return the kept generations directory for an adapter."""
        d = self.kept_dir / adapter_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _stars_since_train(self, adapter_name: str) -> int:
        """Count starred generations for an adapter."""
        kept = self._adapter_kept_dir(adapter_name)
        return len(list(kept.glob("gen_*.json")))

    def record_star(
        self,
        job_id: str,
        audio_data: np.ndarray | bytes,
        sample_rate: int,
        metadata: KeptGeneration,
        adapter_name: str = "default",
    ) -> int:
        """Save a starred generation to disk.

        Args:
            job_id: Job ID of the generation.
            audio_data: Numpy array (float32) or raw WAV bytes.
            sample_rate: Audio sample rate.
            metadata: Generation metadata.
            adapter_name: Which adapter this generation used.

        Returns:
            Current star count for this adapter.
        """
        kept = self._adapter_kept_dir(adapter_name)

        # Save audio
        wav_path = kept / f"gen_{job_id}.wav"
        if isinstance(audio_data, np.ndarray):
            import soundfile as sf

            sf.write(str(wav_path), audio_data, sample_rate)
        else:
            # Raw WAV bytes from client upload
            with open(wav_path, "wb") as f:
                f.write(audio_data)

        # Save metadata
        metadata.starred_at = datetime.now(timezone.utc).isoformat()
        metadata.adapter_name = adapter_name
        meta_path = kept / f"gen_{job_id}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        return self._stars_since_train(adapter_name)

    def remove_star(self, job_id: str, adapter_name: str = "default") -> int:
        """Remove a starred generation.

        Returns:
            Updated star count.
        """
        kept = self._adapter_kept_dir(adapter_name)
        for ext in (".wav", ".json"):
            p = kept / f"gen_{job_id}{ext}"
            if p.exists():
                p.unlink()
        return self._stars_since_train(adapter_name)

    def check_threshold(self, adapter_name: str) -> bool:
        """Check if the star count has reached the retrain threshold.

        Returns True only if auto_retrain is enabled AND stars >= threshold.
        """
        if not self.config.auto_retrain:
            return False
        return self._stars_since_train(adapter_name) >= self.config.retrain_threshold

    def should_refresh_taste(self, adapter_name: str) -> bool:
        """Check if taste should be refreshed based on star count."""
        count = self._stars_since_train(adapter_name)
        if count == 0:
            return False
        return count % self.config.taste_refresh_interval == 0

    def get_kept_generations(
        self, adapter_name: str
    ) -> list[tuple[Path, KeptGeneration]]:
        """List all kept generations for an adapter.

        Returns:
            List of (wav_path, metadata) tuples.
        """
        kept = self._adapter_kept_dir(adapter_name)
        results = []
        for meta_path in sorted(kept.glob("gen_*.json")):
            wav_path = meta_path.with_suffix(".wav")
            if not wav_path.exists():
                continue
            with open(meta_path) as f:
                data = json.load(f)
            results.append((wav_path, KeptGeneration.from_dict(data)))
        return results

    def build_dataset(
        self,
        adapter_name: str,
        library_entries: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Build cumulative training dataset at the configured blend ratio.

        Args:
            adapter_name: Which adapter's kept generations to include.
            library_entries: Training entries from collection_to_training_data().

        Returns:
            Blended list of {"file": path, "text": description} dicts.
        """
        kept_gens = self.get_kept_generations(adapter_name)

        # Build kept generation entries
        kept_entries = []
        for wav_path, meta in kept_gens:
            # Use the original prompt as the training description
            text = meta.prompt if meta.prompt else "instrumental generation"
            kept_entries.append({"file": str(wav_path), "text": text})

        if not library_entries and not kept_entries:
            return []

        # Apply blend ratio
        blend = self.config.blend_ratio  # percentage for library
        total = len(library_entries) + len(kept_entries)

        if blend == 100 or not kept_entries:
            return library_entries
        if blend == 0 or not library_entries:
            return kept_entries

        # Calculate target counts based on ratio
        lib_target = max(1, round(total * blend / 100))
        gen_target = max(1, total - lib_target)

        # Subsample if needed (prefer keeping all, only subsample the larger side)
        if len(library_entries) > lib_target:
            # Randomly subsample library entries
            indices = np.random.choice(
                len(library_entries), size=lib_target, replace=False
            )
            library_entries = [library_entries[i] for i in sorted(indices)]
        if len(kept_entries) > gen_target:
            indices = np.random.choice(
                len(kept_entries), size=gen_target, replace=False
            )
            kept_entries = [kept_entries[i] for i in sorted(indices)]

        return library_entries + kept_entries

    # ------------------------------------------------------------------
    # Adapter versioning
    # ------------------------------------------------------------------

    def _adapter_dir(self, adapter_name: str) -> Path:
        """Return the LoRA adapter directory."""
        return self.loras_dir / adapter_name

    def get_latest_version(self, adapter_name: str) -> int:
        """Return the highest version number, or 0 if none exist."""
        adapter_dir = self._adapter_dir(adapter_name)
        if not adapter_dir.is_dir():
            return 0
        versions = []
        for d in adapter_dir.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                try:
                    versions.append(int(d.name[1:]))
                except ValueError:
                    pass
        return max(versions) if versions else 0

    def create_version(
        self,
        adapter_name: str,
        training_config: LoRAConfig,
        library_count: int,
        kept_count: int,
        top_influences: dict[str, list[dict[str, Any]]],
        parent_version: Optional[int] = None,
        enrichment_tags_added: int = 0,
        duration_seconds: float = 0.0,
    ) -> int:
        """Create a new version directory with changelog.

        The caller is responsible for having already trained the adapter
        and saved weights to the version directory.

        Returns:
            The new version number.
        """
        version = self.get_latest_version(adapter_name) + 1
        version_dir = self._adapter_dir(adapter_name) / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        changelog = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent_version": parent_version,
            "dataset": {
                "library_tracks": library_count,
                "kept_generations": kept_count,
                "blend_ratio": {
                    "library": self.config.blend_ratio,
                    "generations": 100 - self.config.blend_ratio,
                },
                "total_training_samples": library_count + kept_count,
            },
            "top_influences": top_influences,
            "new_since_parent": {
                "kept_generations_added": kept_count
                - (
                    self._parent_kept_count(adapter_name, parent_version)
                    if parent_version
                    else 0
                ),
                "enrichment_tags_added": enrichment_tags_added,
                "library_tracks_added": 0,
            },
            "training": {
                "profile": training_config.profile or "custom",
                "epochs": training_config.epochs,
                "best_loss": training_config.best_loss or 0.0,
                "duration_seconds": duration_seconds,
            },
        }

        with open(version_dir / "changelog.json", "w") as f:
            json.dump(changelog, f, indent=2)

        # Update active symlink
        self._update_active_symlink(adapter_name, version)

        logger.info("Created %s v%d", adapter_name, version)
        return version

    def _parent_kept_count(
        self, adapter_name: str, parent_version: int
    ) -> int:
        """Get kept_generations count from a parent version's changelog."""
        changelog = self.get_changelog(adapter_name, parent_version)
        if changelog:
            dataset = changelog.get("dataset", {})
            return dataset.get("kept_generations", 0)
        return 0

    def _update_active_symlink(
        self, adapter_name: str, version: int
    ) -> None:
        """Update the active symlink to point to the given version."""
        adapter_dir = self._adapter_dir(adapter_name)
        active_link = adapter_dir / "active"

        # Remove existing symlink
        if active_link.is_symlink():
            active_link.unlink()
        elif active_link.exists():
            # Not a symlink but exists — remove it
            if active_link.is_dir():
                shutil.rmtree(str(active_link))
            else:
                active_link.unlink()

        # Create relative symlink
        active_link.symlink_to(f"v{version}")

    def get_versions(self, adapter_name: str) -> list[dict[str, Any]]:
        """List all versions with summary info.

        Returns:
            List of version summary dicts, ordered by version number.
        """
        adapter_dir = self._adapter_dir(adapter_name)
        if not adapter_dir.is_dir():
            return []

        # Determine active version
        active_version = self._get_active_version(adapter_name)

        versions = []
        for d in sorted(adapter_dir.iterdir()):
            if not d.is_dir() or not d.name.startswith("v"):
                continue
            try:
                v_num = int(d.name[1:])
            except ValueError:
                continue

            changelog_path = d / "changelog.json"
            if not changelog_path.exists():
                continue

            with open(changelog_path) as f:
                cl = json.load(f)

            dataset = cl.get("dataset", {})
            training = cl.get("training", {})
            versions.append(
                {
                    "version": v_num,
                    "created_at": cl.get("created_at", ""),
                    "is_active": v_num == active_version,
                    "library_tracks": dataset.get("library_tracks", 0),
                    "kept_generations": dataset.get("kept_generations", 0),
                    "best_loss": training.get("best_loss", 0.0),
                }
            )

        return sorted(versions, key=lambda v: v["version"])

    def _get_active_version(self, adapter_name: str) -> Optional[int]:
        """Return the version number the active symlink points to."""
        active_link = self._adapter_dir(adapter_name) / "active"
        if active_link.is_symlink():
            target = os.readlink(str(active_link))
            name = Path(target).name
            if name.startswith("v"):
                try:
                    return int(name[1:])
                except ValueError:
                    pass
        return None

    def get_changelog(
        self, adapter_name: str, version: int
    ) -> Optional[dict[str, Any]]:
        """Return the full changelog for a specific version."""
        changelog_path = (
            self._adapter_dir(adapter_name) / f"v{version}" / "changelog.json"
        )
        if not changelog_path.exists():
            return None
        with open(changelog_path) as f:
            return json.load(f)

    def revert_version(self, adapter_name: str, version: int) -> bool:
        """Set the active version symlink. Returns True on success."""
        version_dir = self._adapter_dir(adapter_name) / f"v{version}"
        if not version_dir.is_dir():
            return False
        self._update_active_symlink(adapter_name, version)
        logger.info("Reverted %s to v%d", adapter_name, version)
        return True

    def reset_kept_generations(self, adapter_name: str) -> None:
        """Clear all kept generations for an adapter."""
        kept = self._adapter_kept_dir(adapter_name)
        if kept.is_dir():
            shutil.rmtree(str(kept))
            kept.mkdir(parents=True, exist_ok=True)
        logger.info("Reset kept generations for %s", adapter_name)

    def compute_top_influences(
        self,
        dataset: list[dict[str, str]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Analyze training dataset descriptions for top genre/mood/instrument.

        Returns dict with 'genre', 'mood', 'instrument' keys, each a list
        of {'tag': str, 'pct': float} sorted by percentage.
        """
        # Lazy import to avoid circular dependency
        from ..shared.prompt_suggestions import GENRES, INSTRUMENTS, MOODS

        genre_counter: Counter = Counter()
        mood_counter: Counter = Counter()
        instrument_counter: Counter = Counter()

        all_genres = {g.lower() for g in GENRES}
        all_moods = {m.lower() for m in MOODS}
        all_instruments: set[str] = set()
        for group in INSTRUMENTS.values():
            for inst in group:
                all_instruments.add(inst.lower())

        for entry in dataset:
            text = entry.get("text", "").lower()
            words = text.replace(",", " ").split()

            # Check single words and bigrams
            for i, word in enumerate(words):
                if word in all_genres:
                    genre_counter[word] += 1
                if word in all_moods:
                    mood_counter[word] += 1
                if word in all_instruments:
                    instrument_counter[word] += 1

                # Check bigrams
                if i + 1 < len(words):
                    bigram = f"{word} {words[i + 1]}"
                    if bigram in all_genres:
                        genre_counter[bigram] += 1
                    if bigram in all_moods:
                        mood_counter[bigram] += 1
                    if bigram in all_instruments:
                        instrument_counter[bigram] += 1

        def _to_pct(counter: Counter) -> list[dict[str, Any]]:
            total = sum(counter.values())
            if total == 0:
                return []
            return [
                {"tag": tag, "pct": round(count / total * 100, 1)}
                for tag, count in counter.most_common(10)
            ]

        return {
            "genre": _to_pct(genre_counter),
            "mood": _to_pct(mood_counter),
            "instrument": _to_pct(instrument_counter),
        }

    def get_flywheel_status(self, adapter_name: str) -> dict[str, Any]:
        """Return current flywheel state for an adapter."""
        return {
            "stars_since_train": self._stars_since_train(adapter_name),
            "retrain_threshold": self.config.retrain_threshold,
            "auto_retrain": self.config.auto_retrain,
            "active_version": self._get_active_version(adapter_name),
            "total_versions": self.get_latest_version(adapter_name),
        }
