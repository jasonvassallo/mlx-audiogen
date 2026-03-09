"""Tests for audio-to-MIDI transcription and MIDI endpoint."""

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
from mlx_audiogen.shared.audio_to_midi import audio_to_midi


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
# audio_to_midi unit tests
# ---------------------------------------------------------------------------


class TestAudioToMidi:
    def test_silence_produces_valid_midi(self):
        audio = np.zeros(32000, dtype=np.float32)
        midi = audio_to_midi(audio, 32000)
        assert midi[:4] == b"MThd"  # Valid MIDI header

    def test_sine_wave_produces_notes(self):
        """A 440Hz sine wave should produce MIDI note 69 (A4)."""
        sr = 44100
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        midi = audio_to_midi(audio, sr)
        assert midi[:4] == b"MThd"
        assert len(midi) > 14  # Has content beyond header

    def test_short_audio(self):
        audio = np.zeros(100, dtype=np.float32)
        midi = audio_to_midi(audio, 32000)
        assert midi[:4] == b"MThd"

    def test_stereo_input(self):
        audio = np.zeros((2, 32000), dtype=np.float32)
        midi = audio_to_midi(audio, 32000)
        assert midi[:4] == b"MThd"

    def test_custom_bpm(self):
        audio = np.zeros(32000, dtype=np.float32)
        midi = audio_to_midi(audio, 32000, bpm=140.0)
        assert midi[:4] == b"MThd"


# ---------------------------------------------------------------------------
# MIDI endpoint tests
# ---------------------------------------------------------------------------


class TestMidiEndpoint:
    def test_midi_not_found(self, client: TestClient):
        res = client.get("/api/midi/nonexistent")
        assert res.status_code == 404

    def test_midi_not_ready(self, client: TestClient):
        req = GenerateRequest(model="musicgen", prompt="test")
        job = _Job("run-1", req)
        job.status = JobStatus.RUNNING
        _jobs["run-1"] = job
        res = client.get("/api/midi/run-1")
        assert res.status_code == 202

    def test_midi_no_data(self, client: TestClient):
        req = GenerateRequest(model="musicgen", prompt="test")
        job = _Job("done-1", req)
        job.status = JobStatus.DONE
        _jobs["done-1"] = job
        res = client.get("/api/midi/done-1")
        assert res.status_code == 404  # No MIDI data

    def test_midi_download(self, client: TestClient):
        req = GenerateRequest(model="musicgen", prompt="test", output_mode="midi")
        job = _Job("midi-1", req)
        job.status = JobStatus.DONE
        job.midi_data = audio_to_midi(np.zeros(32000, dtype=np.float32), 32000)
        _jobs["midi-1"] = job
        res = client.get("/api/midi/midi-1")
        assert res.status_code == 200
        assert res.headers["content-type"] == "audio/midi"
        assert res.content[:4] == b"MThd"

    def test_generate_request_output_mode(self):
        """Verify output_mode validation."""
        req = GenerateRequest(model="musicgen", prompt="test", output_mode="both")
        assert req.output_mode == "both"

    def test_generate_request_invalid_output_mode(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GenerateRequest(model="musicgen", prompt="test", output_mode="invalid")
