"""Server endpoint tests — validates API contract without real model weights.

Uses FastAPI's TestClient (runs in-process, no network needed).
Tests input validation, job lifecycle, error handling, and edge cases.
"""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlx_audiogen.server.app import (
    GenerateRequest,
    JobStatus,
    PipelineCache,
    _cleanup_old_jobs,
    _encode_wav,
    _Job,
    _jobs,
    _trim_to_exact_duration,
    _weights_dirs,
    app,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset global server state between tests."""
    _jobs.clear()
    _weights_dirs.clear()
    yield
    _jobs.clear()
    _weights_dirs.clear()


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


def test_health_check(client: TestClient):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------


def test_models_empty(client: TestClient):
    res = client.get("/api/models")
    assert res.status_code == 200
    assert res.json() == []


def test_models_with_registered_dirs(client: TestClient):
    _weights_dirs["musicgen-small"] = "/fake/path"
    _weights_dirs["stable-audio"] = "/fake/path2"
    res = client.get("/api/models")
    data = res.json()
    assert len(data) == 2
    assert data[0]["name"] == "musicgen-small"
    assert data[0]["model_type"] == "musicgen"
    assert data[0]["is_loaded"] is False
    assert data[1]["name"] == "stable-audio"
    assert data[1]["model_type"] == "stable_audio"


# ---------------------------------------------------------------------------
# Generate endpoint — input validation
# ---------------------------------------------------------------------------


def test_generate_missing_prompt(client: TestClient):
    res = client.post("/api/generate", json={"model": "musicgen"})
    assert res.status_code == 422  # Pydantic validation error


def test_generate_empty_prompt(client: TestClient):
    res = client.post("/api/generate", json={"model": "musicgen", "prompt": ""})
    assert res.status_code == 422


def test_generate_invalid_model(client: TestClient):
    res = client.post(
        "/api/generate", json={"model": "invalid_model", "prompt": "test"}
    )
    assert res.status_code == 422


def test_generate_invalid_sampler(client: TestClient):
    res = client.post(
        "/api/generate",
        json={"model": "stable_audio", "prompt": "test", "sampler": "bad"},
    )
    assert res.status_code == 422


def test_generate_negative_duration(client: TestClient):
    res = client.post(
        "/api/generate",
        json={"model": "musicgen", "prompt": "test", "seconds": -1},
    )
    assert res.status_code == 422


def test_generate_excessive_duration(client: TestClient):
    res = client.post(
        "/api/generate",
        json={"model": "musicgen", "prompt": "test", "seconds": 500},
    )
    assert res.status_code == 422


def test_generate_zero_temperature(client: TestClient):
    res = client.post(
        "/api/generate",
        json={"model": "musicgen", "prompt": "test", "temperature": 0},
    )
    assert res.status_code == 422


def test_generate_negative_top_k(client: TestClient):
    res = client.post(
        "/api/generate",
        json={"model": "musicgen", "prompt": "test", "top_k": 0},
    )
    assert res.status_code == 422


def test_generate_audio_path_traversal(client: TestClient):
    _weights_dirs["musicgen-small"] = "/fake/path"
    res = client.post(
        "/api/generate",
        json={
            "model": "musicgen",
            "prompt": "test",
            "melody_path": "../../../etc/passwd",
        },
    )
    assert res.status_code == 400
    assert ".." in res.text


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------


def test_status_missing_job(client: TestClient):
    res = client.get("/api/status/nonexistent")
    assert res.status_code == 404


def test_status_returns_progress(client: TestClient):
    req = GenerateRequest(model="musicgen", prompt="test")
    job = _Job("test-123", req)
    job.status = JobStatus.RUNNING
    job.progress = 0.42
    _jobs["test-123"] = job

    res = client.get("/api/status/test-123")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "running"
    assert data["progress"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Audio download endpoint
# ---------------------------------------------------------------------------


def test_audio_missing_job(client: TestClient):
    res = client.get("/api/audio/nonexistent")
    assert res.status_code == 404


def test_audio_still_running(client: TestClient):
    req = GenerateRequest(model="musicgen", prompt="test")
    job = _Job("run-123", req)
    job.status = JobStatus.RUNNING
    _jobs["run-123"] = job

    res = client.get("/api/audio/run-123")
    assert res.status_code == 202


def test_audio_error_job(client: TestClient):
    req = GenerateRequest(model="musicgen", prompt="test")
    job = _Job("err-123", req)
    job.status = JobStatus.ERROR
    job.error = "OOM"
    _jobs["err-123"] = job

    res = client.get("/api/audio/err-123")
    assert res.status_code == 500


def test_audio_download_success(client: TestClient):
    req = GenerateRequest(model="musicgen", prompt="test")
    job = _Job("done-123", req)
    job.status = JobStatus.DONE
    job.audio = np.zeros(32000, dtype=np.float32)  # 1 second at 32kHz
    job.sample_rate = 32000
    job.channels = 1
    _jobs["done-123"] = job

    res = client.get("/api/audio/done-123")
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert "done-123.wav" in res.headers["content-disposition"]
    assert len(res.content) > 44  # WAV header is 44 bytes


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------


def test_cache_lru_eviction():
    cache = PipelineCache(max_size=2)
    cache.put("a", "pipeline_a")
    cache.put("b", "pipeline_b")
    cache.put("c", "pipeline_c")  # Should evict "a"

    assert cache.get("a") is None
    assert cache.get("b") == "pipeline_b"
    assert cache.get("c") == "pipeline_c"


def test_cache_promotes_on_get():
    cache = PipelineCache(max_size=2)
    cache.put("a", "pipeline_a")
    cache.put("b", "pipeline_b")
    cache.get("a")  # Promote "a" to most recently used
    cache.put("c", "pipeline_c")  # Should evict "b" (LRU)

    assert cache.get("a") == "pipeline_a"
    assert cache.get("b") is None
    assert cache.get("c") == "pipeline_c"


# ---------------------------------------------------------------------------
# Job cleanup
# ---------------------------------------------------------------------------


def test_cleanup_old_jobs():
    req = GenerateRequest(model="musicgen", prompt="test")

    # Old completed job (should be cleaned)
    old_job = _Job("old", req)
    old_job.status = JobStatus.DONE
    old_job.completed_at = time.time() - 400  # 6+ minutes ago
    _jobs["old"] = old_job

    # Recent completed job (should NOT be cleaned)
    new_job = _Job("new", req)
    new_job.status = JobStatus.DONE
    new_job.completed_at = time.time() - 60  # 1 minute ago
    _jobs["new"] = new_job

    # Running job (should NOT be cleaned regardless of age)
    running = _Job("running", req)
    running.status = JobStatus.RUNNING
    _jobs["running"] = running

    _cleanup_old_jobs()

    assert "old" not in _jobs
    assert "new" in _jobs
    assert "running" in _jobs


# ---------------------------------------------------------------------------
# Audio trimming
# ---------------------------------------------------------------------------


def test_trim_mono_longer():
    audio = np.zeros(48000, dtype=np.float32)  # 1.5 seconds at 32kHz
    trimmed = _trim_to_exact_duration(audio, 1.0, 32000, 1)
    assert len(trimmed) == 32000


def test_trim_mono_shorter():
    audio = np.zeros(16000, dtype=np.float32)  # 0.5 seconds at 32kHz
    trimmed = _trim_to_exact_duration(audio, 1.0, 32000, 1)
    assert len(trimmed) == 32000  # Padded with zeros


def test_trim_stereo_interleaved():
    audio = np.zeros(96000, dtype=np.float32)  # 1.5s stereo interleaved at 32kHz
    trimmed = _trim_to_exact_duration(audio, 1.0, 32000, 2)
    assert len(trimmed) == 64000  # 32000 frames * 2 channels


def test_trim_exact_no_change():
    audio = np.zeros(32000, dtype=np.float32)
    trimmed = _trim_to_exact_duration(audio, 1.0, 32000, 1)
    assert len(trimmed) == 32000


def test_trim_bpm_precision():
    """4 bars at 128 BPM = 7.5 seconds exactly."""
    target = 4 * 4 * (60 / 128)  # = 7.5
    audio = np.zeros(int(8 * 32000), dtype=np.float32)  # 8 seconds
    trimmed = _trim_to_exact_duration(audio, target, 32000, 1)
    assert len(trimmed) == int(round(target * 32000))  # 240000 samples


# ---------------------------------------------------------------------------
# WAV encoding
# ---------------------------------------------------------------------------


def test_encode_wav_mono():
    audio = np.random.randn(32000).astype(np.float32)
    wav_bytes = _encode_wav(audio, 32000, 1)
    assert len(wav_bytes) > 44
    assert wav_bytes[:4] == b"RIFF"


def test_encode_wav_stereo():
    audio = np.random.randn(64000).astype(np.float32)  # interleaved
    wav_bytes = _encode_wav(audio, 44100, 2)
    assert wav_bytes[:4] == b"RIFF"
