"""Tests for Phase 6 features: prompt suggestions, MIDI-to-prompt,
stem separation, and preset marketplace.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlx_audiogen.server.app import (
    GenerateRequest,
    JobStatus,
    _Job,
    _jobs,
    _weights_dirs,
    app,
)
from mlx_audiogen.shared.midi_to_prompt import midi_to_prompt
from mlx_audiogen.shared.prompt_suggestions import analyze_prompt, suggest_refinements
from mlx_audiogen.shared.stem_separator import separate_basic


@pytest.fixture(autouse=True)
def _clean_state():
    _jobs.clear()
    _weights_dirs.clear()
    yield
    _jobs.clear()
    _weights_dirs.clear()


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 6.1 — Prompt suggestions
# ---------------------------------------------------------------------------


class TestPromptSuggestions:
    def test_suggest_returns_list(self):
        results = suggest_refinements("a drum beat", count=3)
        assert len(results) == 3
        for r in results:
            assert "drum beat" in r.lower()

    def test_analyze_detects_genre(self):
        analysis = analyze_prompt("ambient drone with reverb")
        assert "ambient" in analysis["genres"]

    def test_analyze_detects_mood(self):
        analysis = analyze_prompt("dark industrial noise")
        assert "dark" in analysis["moods"]

    def test_analyze_missing_fields(self):
        analysis = analyze_prompt("a sound")
        assert "genre" in analysis["missing"]

    def test_suggest_endpoint(self, client: TestClient):
        res = client.post("/api/suggest", json={"prompt": "warm pad sound", "count": 2})
        assert res.status_code == 200
        data = res.json()
        assert "suggestions" in data
        assert len(data["suggestions"]) >= 2

    def test_suggest_empty_prompt_rejected(self, client: TestClient):
        res = client.post("/api/suggest", json={"prompt": "", "count": 2})
        assert res.status_code == 422


# ---------------------------------------------------------------------------
# 6.4 — Stem separation
# ---------------------------------------------------------------------------


class TestStemSeparation:
    def test_basic_separation(self):
        # 1 second of white noise
        audio = np.random.randn(44100).astype(np.float32) * 0.5
        stems = separate_basic(audio, 44100)
        assert "bass" in stems
        assert "mid" in stems
        assert "high" in stems
        for name, stem in stems.items():
            assert len(stem) == len(audio)
            assert stem.dtype == np.float32

    def test_separation_endpoint(self, client: TestClient):
        req = GenerateRequest(model="musicgen", prompt="test")
        job = _Job("sep-1", req)
        job.status = JobStatus.DONE
        job.audio = np.random.randn(32000).astype(np.float32) * 0.3
        job.sample_rate = 32000
        job.channels = 1
        _jobs["sep-1"] = job

        res = client.post("/api/separate/sep-1")
        assert res.status_code == 200
        data = res.json()
        assert "stems" in data
        assert len(data["stems"]) >= 3

    def test_separation_missing_job(self, client: TestClient):
        res = client.post("/api/separate/nonexistent")
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# 6.5 — MIDI-to-prompt
# ---------------------------------------------------------------------------


class TestMidiToPrompt:
    def test_empty_midi(self):
        from mlx_audiogen.shared.audio_to_midi import audio_to_midi

        midi_bytes = audio_to_midi(np.zeros(32000, dtype=np.float32), 32000)
        result = midi_to_prompt(midi_bytes)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_bytes(self):
        result = midi_to_prompt(b"not midi data")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 6.8 — Preset marketplace
# ---------------------------------------------------------------------------


class TestPresetMarketplace:
    def test_list_presets(self, client: TestClient):
        res = client.get("/api/presets")
        assert res.status_code == 200
        assert isinstance(res.json(), list)

    def test_save_and_load_preset(self, client: TestClient):
        res = client.post(
            "/api/presets/test_preset",
            json={"model": "musicgen", "prompt": "jazzy piano"},
        )
        assert res.status_code == 200
        assert res.json()["saved"] == "test_preset"

        # Load it back
        res2 = client.get("/api/presets/test_preset")
        assert res2.status_code == 200
        data = res2.json()
        assert data["prompt"] == "jazzy piano"

        # Clean up
        import os

        from mlx_audiogen.server.app import _PRESETS_DIR

        preset_file = _PRESETS_DIR / "test_preset.mlxpreset"
        if preset_file.exists():
            os.remove(preset_file)

    def test_load_nonexistent_preset(self, client: TestClient):
        res = client.get("/api/presets/does_not_exist_xyz")
        assert res.status_code == 404
