"""Tests for the flywheel intelligence module (Phase 9g-4)."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from mlx_audiogen.lora.flywheel import (
    FlywheelConfig,
    FlywheelManager,
    KeptGeneration,
    resolve_lora_dir,
)


@pytest.fixture
def tmp_flywheel(tmp_path):
    """Create a FlywheelManager with temp dirs."""
    loras_dir = tmp_path / "loras"
    kept_dir = tmp_path / "kept"
    loras_dir.mkdir()
    kept_dir.mkdir()
    config = FlywheelConfig(retrain_threshold=3, blend_ratio=80)
    return FlywheelManager(config=config, loras_dir=loras_dir, kept_dir=kept_dir)


@pytest.fixture
def sample_audio():
    """Generate a short audio array for testing."""
    return np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz


@pytest.fixture
def sample_metadata():
    """Create sample generation metadata."""
    return KeptGeneration(
        job_id="test-job-1",
        prompt="deep house atmospheric",
        model="musicgen",
        adapter_name="my-style",
    )


# ---------------------------------------------------------------------------
# FlywheelConfig tests
# ---------------------------------------------------------------------------


class TestFlywheelConfig:
    def test_defaults(self):
        cfg = FlywheelConfig()
        assert cfg.retrain_threshold == 10
        assert cfg.blend_ratio == 80
        assert cfg.taste_refresh_interval == 5
        assert cfg.auto_retrain is True
        assert cfg.base_collection is None

    def test_to_dict_roundtrip(self):
        cfg = FlywheelConfig(retrain_threshold=15, blend_ratio=60)
        d = cfg.to_dict()
        restored = FlywheelConfig.from_dict(d)
        assert restored.retrain_threshold == 15
        assert restored.blend_ratio == 60

    def test_validation_clamps_values(self):
        cfg = FlywheelConfig.from_dict(
            {"retrain_threshold": -5, "blend_ratio": 200, "taste_refresh_interval": 0}
        )
        assert cfg.retrain_threshold == 1  # clamped to min
        assert cfg.blend_ratio == 100  # clamped to max
        assert cfg.taste_refresh_interval == 1  # clamped to min


# ---------------------------------------------------------------------------
# Star recording tests
# ---------------------------------------------------------------------------


class TestRecordStar:
    def test_record_star_saves_wav_and_json(self, tmp_flywheel, sample_audio, sample_metadata):
        count = tmp_flywheel.record_star(
            "job1", sample_audio, 16000, sample_metadata, adapter_name="my-style"
        )
        assert count == 1

        kept_dir = tmp_flywheel.kept_dir / "my-style"
        assert (kept_dir / "gen_job1.wav").exists()
        assert (kept_dir / "gen_job1.json").exists()

        with open(kept_dir / "gen_job1.json") as f:
            meta = json.load(f)
        assert meta["prompt"] == "deep house atmospheric"
        assert meta["starred_at"] is not None

    def test_record_star_with_bytes(self, tmp_flywheel, sample_metadata):
        wav_bytes = b"RIFF" + b"\x00" * 100  # fake WAV bytes
        count = tmp_flywheel.record_star(
            "job2", wav_bytes, 16000, sample_metadata, adapter_name="my-style"
        )
        assert count == 1
        assert (tmp_flywheel.kept_dir / "my-style" / "gen_job2.wav").exists()

    def test_remove_star_deletes_files(self, tmp_flywheel, sample_audio, sample_metadata):
        tmp_flywheel.record_star(
            "job1", sample_audio, 16000, sample_metadata, adapter_name="my-style"
        )
        count = tmp_flywheel.remove_star("job1", adapter_name="my-style")
        assert count == 0

        kept_dir = tmp_flywheel.kept_dir / "my-style"
        assert not (kept_dir / "gen_job1.wav").exists()
        assert not (kept_dir / "gen_job1.json").exists()

    def test_star_count_increments(self, tmp_flywheel, sample_audio):
        for i in range(5):
            meta = KeptGeneration(
                job_id=f"j{i}", prompt=f"prompt {i}", model="musicgen",
                adapter_name="my-style",
            )
            count = tmp_flywheel.record_star(
                f"j{i}", sample_audio, 16000, meta, adapter_name="my-style"
            )
        assert count == 5


# ---------------------------------------------------------------------------
# Threshold tests
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_below_threshold(self, tmp_flywheel, sample_audio):
        meta = KeptGeneration(
            job_id="j1", prompt="test", model="musicgen", adapter_name="my-style"
        )
        tmp_flywheel.record_star("j1", sample_audio, 16000, meta, adapter_name="my-style")
        assert not tmp_flywheel.check_threshold("my-style")

    def test_at_threshold(self, tmp_flywheel, sample_audio):
        for i in range(3):  # threshold is 3
            meta = KeptGeneration(
                job_id=f"j{i}", prompt="test", model="musicgen",
                adapter_name="my-style",
            )
            tmp_flywheel.record_star(
                f"j{i}", sample_audio, 16000, meta, adapter_name="my-style"
            )
        assert tmp_flywheel.check_threshold("my-style")

    def test_auto_retrain_off_skips(self, tmp_flywheel, sample_audio):
        tmp_flywheel.config.auto_retrain = False
        for i in range(5):
            meta = KeptGeneration(
                job_id=f"j{i}", prompt="test", model="musicgen",
                adapter_name="my-style",
            )
            tmp_flywheel.record_star(
                f"j{i}", sample_audio, 16000, meta, adapter_name="my-style"
            )
        assert not tmp_flywheel.check_threshold("my-style")


# ---------------------------------------------------------------------------
# Taste refresh tests
# ---------------------------------------------------------------------------


class TestTasteRefresh:
    def test_should_refresh_at_interval(self, tmp_flywheel, sample_audio):
        tmp_flywheel.config.taste_refresh_interval = 2
        for i in range(2):
            meta = KeptGeneration(
                job_id=f"j{i}", prompt="test", model="musicgen",
                adapter_name="my-style",
            )
            tmp_flywheel.record_star(
                f"j{i}", sample_audio, 16000, meta, adapter_name="my-style"
            )
        assert tmp_flywheel.should_refresh_taste("my-style")

    def test_should_not_refresh_between_intervals(self, tmp_flywheel, sample_audio):
        tmp_flywheel.config.taste_refresh_interval = 3
        meta = KeptGeneration(
            job_id="j0", prompt="test", model="musicgen", adapter_name="my-style"
        )
        tmp_flywheel.record_star(
            "j0", sample_audio, 16000, meta, adapter_name="my-style"
        )
        assert not tmp_flywheel.should_refresh_taste("my-style")


# ---------------------------------------------------------------------------
# Build dataset tests
# ---------------------------------------------------------------------------


class TestBuildDataset:
    def test_blend_80_20(self, tmp_flywheel, sample_audio):
        # Add 3 kept generations
        for i in range(3):
            meta = KeptGeneration(
                job_id=f"j{i}", prompt=f"kept prompt {i}", model="musicgen",
                adapter_name="test",
            )
            tmp_flywheel.record_star(
                f"j{i}", sample_audio, 16000, meta, adapter_name="test"
            )

        library = [{"file": f"/lib/{i}.wav", "text": f"lib track {i}"} for i in range(7)]

        dataset = tmp_flywheel.build_dataset("test", library)
        assert len(dataset) > 0
        # Check both sources are present
        lib_count = sum(1 for d in dataset if d["file"].startswith("/lib/"))
        gen_count = sum(1 for d in dataset if "gen_" in d["file"])
        assert lib_count > 0
        assert gen_count > 0

    def test_blend_100_0_library_only(self, tmp_flywheel, sample_audio):
        tmp_flywheel.config.blend_ratio = 100
        meta = KeptGeneration(
            job_id="j0", prompt="test", model="musicgen", adapter_name="test"
        )
        tmp_flywheel.record_star("j0", sample_audio, 16000, meta, adapter_name="test")

        library = [{"file": "/lib/0.wav", "text": "lib track"}]
        dataset = tmp_flywheel.build_dataset("test", library)
        # Should return only library entries
        assert all(d["file"].startswith("/lib/") for d in dataset)

    def test_blend_0_100_generations_only(self, tmp_flywheel, sample_audio):
        tmp_flywheel.config.blend_ratio = 0
        meta = KeptGeneration(
            job_id="j0", prompt="gen prompt", model="musicgen", adapter_name="test"
        )
        tmp_flywheel.record_star("j0", sample_audio, 16000, meta, adapter_name="test")

        library = [{"file": "/lib/0.wav", "text": "lib track"}]
        dataset = tmp_flywheel.build_dataset("test", library)
        # Should return only kept generation entries
        assert all("gen_" in d["file"] for d in dataset)

    def test_empty_dataset(self, tmp_flywheel):
        dataset = tmp_flywheel.build_dataset("test", [])
        assert dataset == []


# ---------------------------------------------------------------------------
# Versioning tests
# ---------------------------------------------------------------------------


class TestVersioning:
    def _create_fake_lora(self, loras_dir, name, version):
        """Helper to create a fake versioned LoRA."""
        v_dir = loras_dir / name / f"v{version}"
        v_dir.mkdir(parents=True, exist_ok=True)
        # Minimal files
        with open(v_dir / "config.json", "w") as f:
            json.dump({"name": name, "rank": 16}, f)
        with open(v_dir / "changelog.json", "w") as f:
            json.dump({
                "version": version,
                "created_at": "2026-03-14T00:00:00Z",
                "parent_version": version - 1 if version > 1 else None,
                "dataset": {
                    "library_tracks": 10, "kept_generations": version * 3,
                    "blend_ratio": {"library": 80, "generations": 20},
                    "total_training_samples": 10 + version * 3,
                },
                "top_influences": {"genre": [], "mood": [], "instrument": []},
                "new_since_parent": {
                    "kept_generations_added": 3,
                    "enrichment_tags_added": 0,
                    "library_tracks_added": 0,
                },
                "training": {
                    "profile": "balanced", "epochs": 20,
                    "best_loss": 0.5 - version * 0.05,
                    "duration_seconds": 120.0,
                },
            }, f)
        # Create lora.safetensors placeholder
        (v_dir / "lora.safetensors").touch()
        return v_dir

    def test_get_latest_version(self, tmp_flywheel):
        self._create_fake_lora(tmp_flywheel.loras_dir, "test", 1)
        self._create_fake_lora(tmp_flywheel.loras_dir, "test", 2)
        assert tmp_flywheel.get_latest_version("test") == 2

    def test_get_latest_version_no_versions(self, tmp_flywheel):
        assert tmp_flywheel.get_latest_version("nonexistent") == 0

    def test_get_versions_ordered(self, tmp_flywheel):
        for v in [1, 2, 3]:
            self._create_fake_lora(tmp_flywheel.loras_dir, "test", v)
        # Set active to v3
        active = tmp_flywheel.loras_dir / "test" / "active"
        active.symlink_to("v3")

        versions = tmp_flywheel.get_versions("test")
        assert len(versions) == 3
        assert versions[0]["version"] == 1
        assert versions[2]["version"] == 3
        assert versions[2]["is_active"] is True
        assert versions[0]["is_active"] is False

    def test_get_changelog(self, tmp_flywheel):
        self._create_fake_lora(tmp_flywheel.loras_dir, "test", 1)
        cl = tmp_flywheel.get_changelog("test", 1)
        assert cl is not None
        assert cl["version"] == 1
        assert cl["parent_version"] is None

    def test_get_changelog_nonexistent(self, tmp_flywheel):
        assert tmp_flywheel.get_changelog("test", 99) is None

    def test_revert_version(self, tmp_flywheel):
        self._create_fake_lora(tmp_flywheel.loras_dir, "test", 1)
        self._create_fake_lora(tmp_flywheel.loras_dir, "test", 2)
        # Set active to v2
        active = tmp_flywheel.loras_dir / "test" / "active"
        active.symlink_to("v2")

        # Revert to v1
        assert tmp_flywheel.revert_version("test", 1)
        resolved = (tmp_flywheel.loras_dir / "test" / "active").resolve()
        assert resolved.name == "v1"

    def test_revert_nonexistent_version(self, tmp_flywheel):
        assert not tmp_flywheel.revert_version("test", 99)

    def test_reset_kept_generations(self, tmp_flywheel, sample_audio):
        meta = KeptGeneration(
            job_id="j0", prompt="test", model="musicgen", adapter_name="test"
        )
        tmp_flywheel.record_star("j0", sample_audio, 16000, meta, adapter_name="test")
        assert tmp_flywheel._stars_since_train("test") == 1

        tmp_flywheel.reset_kept_generations("test")
        assert tmp_flywheel._stars_since_train("test") == 0


# ---------------------------------------------------------------------------
# LoRA migration tests
# ---------------------------------------------------------------------------


class TestLoRAMigration:
    def test_flat_layout_migrates_to_v1(self, tmp_path):
        """A flat LoRA dir (config.json + lora.safetensors at top) migrates to v1/."""
        lora_dir = tmp_path / "my-style"
        lora_dir.mkdir()
        (lora_dir / "config.json").write_text('{"name": "my-style", "rank": 16}')
        (lora_dir / "lora.safetensors").touch()

        resolved = resolve_lora_dir(lora_dir)
        assert resolved == lora_dir / "v1"
        assert (lora_dir / "v1" / "config.json").exists()
        assert (lora_dir / "v1" / "lora.safetensors").exists()
        assert (lora_dir / "v1" / "changelog.json").exists()
        assert (lora_dir / "active").is_symlink()

        # Original files moved
        assert not (lora_dir / "config.json").exists()
        assert not (lora_dir / "lora.safetensors").exists()

    def test_already_versioned_not_remigrated(self, tmp_path):
        """A versioned LoRA dir is returned as-is."""
        lora_dir = tmp_path / "my-style"
        v2_dir = lora_dir / "v2"
        v2_dir.mkdir(parents=True)
        (v2_dir / "config.json").write_text('{"name": "my-style", "rank": 16}')
        (v2_dir / "lora.safetensors").touch()
        (lora_dir / "active").symlink_to("v2")

        resolved = resolve_lora_dir(lora_dir)
        assert resolved == v2_dir

    def test_migration_is_idempotent(self, tmp_path):
        """Calling resolve twice on an already-migrated dir doesn't break."""
        lora_dir = tmp_path / "my-style"
        lora_dir.mkdir()
        (lora_dir / "config.json").write_text('{"name": "my-style", "rank": 16}')
        (lora_dir / "lora.safetensors").touch()

        resolve_lora_dir(lora_dir)
        resolved2 = resolve_lora_dir(lora_dir)
        assert resolved2 == lora_dir / "v1"


# ---------------------------------------------------------------------------
# Top influences tests
# ---------------------------------------------------------------------------


class TestTopInfluences:
    def test_compute_influences(self, tmp_flywheel):
        dataset = [
            {"file": "a.wav", "text": "deep house atmospheric ambient"},
            {"file": "b.wav", "text": "deep house minimal techno"},
            {"file": "c.wav", "text": "ambient piano dreamy"},
        ]
        influences = tmp_flywheel.compute_top_influences(dataset)
        assert "genre" in influences
        assert "mood" in influences
        assert "instrument" in influences

    def test_empty_dataset_influences(self, tmp_flywheel):
        influences = tmp_flywheel.compute_top_influences([])
        assert influences == {"genre": [], "mood": [], "instrument": []}


# ---------------------------------------------------------------------------
# Flywheel status tests
# ---------------------------------------------------------------------------


class TestFlywheelStatus:
    def test_status_empty(self, tmp_flywheel):
        status = tmp_flywheel.get_flywheel_status("my-style")
        assert status["stars_since_train"] == 0
        assert status["retrain_threshold"] == 3
        assert status["auto_retrain"] is True

    def test_status_with_stars(self, tmp_flywheel, sample_audio):
        meta = KeptGeneration(
            job_id="j0", prompt="test", model="musicgen", adapter_name="my-style"
        )
        tmp_flywheel.record_star("j0", sample_audio, 16000, meta, adapter_name="my-style")
        status = tmp_flywheel.get_flywheel_status("my-style")
        assert status["stars_since_train"] == 1
