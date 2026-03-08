"""FastAPI server for MLX audio generation.

Provides endpoints for generating audio from text prompts using MusicGen
or Stable Audio. Features an LRU pipeline cache to avoid reloading models
on every request, and async generation with status polling.

Endpoints:
    POST /api/generate    — Submit a generation request (returns job ID)
    GET  /api/status/{id} — Poll job status (queued/running/done/error)
    GET  /api/audio/{id}  — Download generated audio as WAV
    GET  /api/models      — List available models/weights

Architecture:
    - Pipeline cache: holds 1-2 loaded model pipelines in an LRU cache
    - Generation runs in a thread pool executor (MLX is CPU-bound)
    - Jobs stored in-memory with configurable max retention
    - CORS enabled for local Max for Live / browser integration
"""

import argparse
import io
import sys
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    from pydantic import BaseModel, Field
except ImportError:
    print(
        "Error: FastAPI not installed. "
        "Install server dependencies: uv sync --extra server"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Models (Pydantic)
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Generation job lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class GenerateRequest(BaseModel):
    """Request body for POST /api/generate."""

    model: str = Field(
        default="musicgen",
        pattern=r"^(musicgen|stable_audio)$",
        description="Model type: 'musicgen' or 'stable_audio'",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text description of desired audio",
    )
    negative_prompt: str = Field(
        default="", max_length=5000, description="Negative prompt (stable_audio only)"
    )
    seconds: float = Field(
        default=5.0, ge=0.1, le=300, description="Duration in seconds"
    )
    # MusicGen params
    temperature: float = Field(default=1.0, gt=0)
    top_k: int = Field(default=250, ge=1)
    guidance_coef: float = Field(default=3.0, ge=0)
    # Stable Audio params
    steps: int = Field(default=8, ge=1, le=1000)
    cfg_scale: float = Field(default=6.0, ge=0)
    sampler: str = Field(default="euler", pattern=r"^(euler|rk4)$")
    # General
    seed: Optional[int] = Field(default=None)
    # Style/melody: paths are validated in _validate_audio_path() before use
    melody_path: Optional[str] = Field(default=None, max_length=1024)
    style_audio_path: Optional[str] = Field(default=None, max_length=1024)
    style_coef: float = Field(default=5.0, ge=0)


class JobInfo(BaseModel):
    """Response for GET /api/status/{id}."""

    id: str
    status: JobStatus
    model: str
    prompt: str
    seconds: float
    created_at: float
    completed_at: Optional[float] = None
    error: Optional[str] = None
    sample_rate: Optional[int] = None


class GenerateResponse(BaseModel):
    """Response for POST /api/generate."""

    id: str
    status: JobStatus


class ModelInfo(BaseModel):
    """Info about an available model."""

    name: str
    model_type: str
    is_loaded: bool


# ---------------------------------------------------------------------------
# Pipeline Cache
# ---------------------------------------------------------------------------


class PipelineCache:
    """LRU cache for loaded model pipelines.

    Keeps up to ``max_size`` pipelines loaded in memory. When the cache
    is full, the least-recently-used pipeline is evicted. This avoids
    reloading large models (1-5 GB) on every request while bounding
    memory usage.
    """

    def __init__(self, max_size: int = 2):
        self._cache: OrderedDict[str, object] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> object | None:
        """Get a pipeline, promoting it to most-recently-used."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, pipeline: object) -> None:
        """Add a pipeline, evicting LRU if at capacity."""
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = pipeline
        else:
            if len(self._cache) >= self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                print(f"Evicted pipeline: {evicted_key}")
            self._cache[key] = pipeline

    def keys(self) -> list[str]:
        """Return list of cached pipeline keys."""
        return list(self._cache.keys())


# ---------------------------------------------------------------------------
# Job Store
# ---------------------------------------------------------------------------


class _Job:
    """Internal job tracking object."""

    __slots__ = (
        "id",
        "status",
        "request",
        "created_at",
        "completed_at",
        "audio",
        "sample_rate",
        "channels",
        "error",
    )

    def __init__(self, job_id: str, request: GenerateRequest):
        self.id = job_id
        self.status = JobStatus.QUEUED
        self.request = request
        self.created_at = time.time()
        self.completed_at: float | None = None
        self.audio: np.ndarray | None = None
        self.sample_rate: int | None = None
        self.channels: int = 1
        self.error: str | None = None


# ---------------------------------------------------------------------------
# App State
# ---------------------------------------------------------------------------

# Global state — initialized in startup
_pipeline_cache = PipelineCache(max_size=2)
_executor = ThreadPoolExecutor(max_workers=1)  # Single worker — MLX is GPU-bound
_jobs: dict[str, _Job] = {}
_weights_dirs: dict[str, str] = {}  # name -> path
_max_jobs = 100

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MLX Audio Generation Server",
    version="0.1.0",
    description="Generate music and audio on Apple Silicon via MLX",
)

# CORS: allow local connections (Max for Live, browser, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static Files (Web UI)
# ---------------------------------------------------------------------------

# Serve the built React SPA from web/dist/ if it exists.
# This is mounted AFTER the API routes so /api/* takes priority.
_WEB_DIST = Path(__file__).resolve().parent.parent.parent / "web" / "dist"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/models")
def list_models() -> list[ModelInfo]:
    """List available models and their loading status."""
    result = []
    loaded_keys = _pipeline_cache.keys()
    for name in _weights_dirs:
        # Determine model type from name
        if "stable" in name.lower():
            model_type = "stable_audio"
        else:
            model_type = "musicgen"
        result.append(
            ModelInfo(
                name=name,
                model_type=model_type,
                is_loaded=name in loaded_keys,
            )
        )
    return result


@app.post("/api/generate")
def submit_generation(req: GenerateRequest) -> GenerateResponse:
    """Submit a generation request. Returns immediately with a job ID."""
    # Validate audio file paths (rejects traversal, missing files, bad extensions)
    req.melody_path = _validate_audio_path(req.melody_path, "melody")
    req.style_audio_path = _validate_audio_path(req.style_audio_path, "style_audio")

    # Enforce job limit
    if len(_jobs) >= _max_jobs:
        _cleanup_old_jobs()
    if len(_jobs) >= _max_jobs:
        raise HTTPException(429, "Too many jobs. Wait for existing jobs to complete.")

    job_id = str(uuid.uuid4())[:8]
    job = _Job(job_id, req)
    _jobs[job_id] = job

    # Submit to thread pool
    _executor.submit(_run_generation, job)

    return GenerateResponse(id=job_id, status=JobStatus.QUEUED)


@app.get("/api/status/{job_id}")
def get_status(job_id: str) -> JobInfo:
    """Poll the status of a generation job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    return JobInfo(
        id=job.id,
        status=job.status,
        model=job.request.model,
        prompt=job.request.prompt,
        seconds=job.request.seconds,
        created_at=job.created_at,
        completed_at=job.completed_at,
        error=job.error,
        sample_rate=job.sample_rate,
    )


@app.get("/api/audio/{job_id}")
def get_audio(job_id: str) -> Response:
    """Download generated audio as WAV.

    Returns 404 if job not found, 202 if still running, or audio/wav if done.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status == JobStatus.ERROR:
        raise HTTPException(500, f"Generation failed: {job.error}")
    if job.status != JobStatus.DONE:
        raise HTTPException(
            202, "Generation still in progress. Poll /api/status first."
        )
    if job.audio is None:
        raise HTTPException(500, "Audio data missing from completed job.")

    # Encode as WAV in memory
    wav_bytes = _encode_wav(job.audio, job.sample_rate or 32000, job.channels)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{job_id}.wav"',
        },
    )


# ---------------------------------------------------------------------------
# Generation Worker
# ---------------------------------------------------------------------------


def _run_generation(job: _Job) -> None:
    """Execute generation in the thread pool worker."""
    try:
        job.status = JobStatus.RUNNING
        req = job.request

        if req.model == "stable_audio":
            audio, sr, channels = _generate_stable_audio(req)
        elif req.model == "musicgen":
            audio, sr, channels = _generate_musicgen(req)
        else:
            raise ValueError(f"Unknown model: {req.model}")

        job.audio = audio
        job.sample_rate = sr
        job.channels = channels
        job.status = JobStatus.DONE
        job.completed_at = time.time()

    except Exception as e:
        job.status = JobStatus.ERROR
        job.error = str(e)
        job.completed_at = time.time()
        print(f"Job {job.id} failed: {e}")


def _get_pipeline(model_type: str) -> object:
    """Get or load a pipeline from the cache.

    Looks for a weights directory matching the model type. The first
    registered weights dir containing the model type name is used.
    """
    # Find weights dir
    weights_dir: str | None = None
    cache_key: str = ""
    for name, path in _weights_dirs.items():
        if model_type in name.lower():
            weights_dir = path
            cache_key = name
            break

    if weights_dir is None:
        # Use first available if only one is registered
        if len(_weights_dirs) == 1:
            cache_key, weights_dir = next(iter(_weights_dirs.items()))
        else:
            raise ValueError(
                f"No weights directory found for model '{model_type}'. "
                f"Available: {list(_weights_dirs.keys())}"
            )

    # Check cache
    pipeline = _pipeline_cache.get(cache_key)
    if pipeline is not None:
        return pipeline

    # Load pipeline
    print(f"Loading {model_type} pipeline from {weights_dir}...")
    if model_type == "stable_audio":
        from mlx_audiogen.models.stable_audio import StableAudioPipeline

        pipeline = StableAudioPipeline.from_pretrained(weights_dir)
    else:
        from mlx_audiogen.models.musicgen import MusicGenPipeline

        pipeline = MusicGenPipeline.from_pretrained(weights_dir)

    _pipeline_cache.put(cache_key, pipeline)
    return pipeline


def _trim_to_exact_duration(
    audio: np.ndarray, target_seconds: float, sample_rate: int, channels: int
) -> np.ndarray:
    """Trim or pad audio to exactly the requested duration.

    Models often produce slightly more or less audio than requested due to
    codec frame boundaries and token-to-sample alignment. This ensures the
    output is sample-exact for BPM-synced loop playback.
    """
    target_samples = int(round(target_seconds * sample_rate))

    if channels > 1 and audio.ndim == 1:
        # Interleaved stereo: total samples = frames * channels
        target_total = target_samples * channels
        if len(audio) > target_total:
            audio = audio[:target_total]
        elif len(audio) < target_total:
            audio = np.pad(audio, (0, target_total - len(audio)))
    else:
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        elif len(audio) < target_samples:
            audio = np.pad(audio, (0, target_samples - len(audio)))

    return audio


def _generate_musicgen(
    req: GenerateRequest,
) -> tuple[np.ndarray, int, int]:
    """Generate audio using MusicGen pipeline."""
    from mlx_audiogen.models.musicgen import MusicGenPipeline

    pipe: MusicGenPipeline = _get_pipeline("musicgen")  # type: ignore[assignment]

    # Request extra time to compensate for codec frame boundary losses,
    # then trim to the exact requested duration.
    buffer_seconds = req.seconds + 0.5
    audio = pipe.generate(
        prompt=req.prompt,
        seconds=buffer_seconds,
        temperature=req.temperature,
        top_k=req.top_k,
        guidance_coef=req.guidance_coef,
        seed=req.seed,
        melody_path=req.melody_path,
        style_audio_path=req.style_audio_path,
        style_coef=req.style_coef,
    )
    # MusicGen outputs mono by default; stereo variants output interleaved
    channels = 1
    if pipe.config.decoder.num_codebooks > 4:
        channels = 2

    audio = _trim_to_exact_duration(audio, req.seconds, pipe.sample_rate, channels)  # type: ignore[arg-type, assignment]
    return audio, pipe.sample_rate, channels  # type: ignore[return-value]


def _generate_stable_audio(
    req: GenerateRequest,
) -> tuple[np.ndarray, int, int]:
    """Generate audio using Stable Audio pipeline."""
    from mlx_audiogen.models.stable_audio import StableAudioPipeline

    pipe: StableAudioPipeline = _get_pipeline("stable_audio")  # type: ignore[assignment]
    audio = pipe.generate(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        seconds_total=req.seconds,
        steps=req.steps,
        cfg_scale=req.cfg_scale,
        seed=req.seed,
        sampler=req.sampler,
    )

    audio = _trim_to_exact_duration(audio, req.seconds, 44100, 2)  # type: ignore[arg-type, assignment]
    return audio, 44100, 2  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_audio_path(path_str: Optional[str], label: str) -> Optional[str]:
    """Validate an audio file path from a client request.

    Rejects path traversal, non-existent files, and non-file paths.
    Returns the resolved path string if valid, None if input is None.

    Raises HTTPException on invalid paths.
    """
    if path_str is None:
        return None
    p = Path(path_str)
    # Reject path traversal
    if ".." in p.parts:
        raise HTTPException(400, f"Invalid {label} path: must not contain '..'")
    resolved = p.resolve()
    if not resolved.is_file():
        raise HTTPException(400, f"Invalid {label} path: file not found")
    # Restrict to common audio extensions
    allowed_exts = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".aiff"}
    if resolved.suffix.lower() not in allowed_exts:
        raise HTTPException(
            400, f"Invalid {label} path: unsupported file type '{resolved.suffix}'"
        )
    return str(resolved)


def _encode_wav(audio: np.ndarray, sample_rate: int, channels: int) -> bytes:
    """Encode audio array as WAV bytes in memory."""
    import soundfile as sf

    buf = io.BytesIO()
    # Reshape for multi-channel if needed
    if channels > 1 and audio.ndim == 1:
        # Interleaved stereo -> (samples, 2)
        audio = audio.reshape(-1, channels)
    sf.write(buf, audio, sample_rate, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def _cleanup_old_jobs() -> None:
    """Remove completed/errored jobs older than 5 minutes."""
    cutoff = time.time() - 300
    to_remove = [
        jid
        for jid, job in _jobs.items()
        if job.status in (JobStatus.DONE, JobStatus.ERROR)
        and (job.completed_at or 0) < cutoff
    ]
    for jid in to_remove:
        del _jobs[jid]


# ---------------------------------------------------------------------------
# Web UI Static Files
# ---------------------------------------------------------------------------

if _WEB_DIST.is_dir():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    @app.get("/{full_path:path}")
    def spa_catch_all(full_path: str) -> Response:
        """Serve static files or fall back to index.html for SPA routing."""
        file_path = _WEB_DIST / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        index = _WEB_DIST / "index.html"
        if index.is_file():
            return FileResponse(index)
        raise HTTPException(404, "Web UI not built. Run: cd web && npm run build")

    # Mount static assets directory for Vite's hashed files
    if (_WEB_DIST / "assets").is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=str(_WEB_DIST / "assets")),
            name="static-assets",
        )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for mlx-audiogen-server."""
    parser = argparse.ArgumentParser(description="MLX Audio Generation HTTP Server")
    parser.add_argument(
        "--weights-dir",
        type=str,
        required=True,
        action="append",
        help="Path to converted weights directory "
        "(can be specified multiple times for multiple models)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8420,
        help="Server port (default: 8420)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--max-cache",
        type=int,
        default=2,
        help="Max pipelines to keep in memory (default: 2)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open Web UI in browser after server starts",
    )

    args = parser.parse_args()

    # Register weights directories
    for wd in args.weights_dir:
        path = Path(wd).resolve()
        if not path.is_dir():
            print(f"Error: Weights directory not found: {wd}")
            sys.exit(1)
        name = path.name
        _weights_dirs[name] = str(path)
        print(f"Registered model: {name} -> {path}")

    # Configure cache
    global _pipeline_cache
    _pipeline_cache = PipelineCache(max_size=args.max_cache)

    # Pre-load first model
    if _weights_dirs:
        first_name, first_path = next(iter(_weights_dirs.items()))
        print(f"\nPre-loading {first_name}...")
        try:
            if "stable" in first_name.lower():
                from mlx_audiogen.models.stable_audio import (
                    StableAudioPipeline,
                )

                pipe = StableAudioPipeline.from_pretrained(first_path)
            else:
                from mlx_audiogen.models.musicgen import (
                    MusicGenPipeline,
                )

                pipe = MusicGenPipeline.from_pretrained(first_path)
            _pipeline_cache.put(first_name, pipe)
        except Exception as e:
            print(f"Warning: Failed to pre-load {first_name}: {e}")

    url = f"http://{args.host}:{args.port}"
    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"API docs: {url}/docs")
    if _WEB_DIST.is_dir():
        print(f"Web UI:   {url}/")

    if args.open and _WEB_DIST.is_dir():
        import threading
        import webbrowser

        # Open browser after a short delay to let uvicorn start
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
