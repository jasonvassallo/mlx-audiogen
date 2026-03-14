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
import json as json_mod
import sys
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, Response
    from pydantic import BaseModel, Field
    from starlette.middleware.base import BaseHTTPMiddleware
except ImportError:
    print(
        "Error: FastAPI not installed. "
        "Install server dependencies: uv sync --extra server"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """In-memory sliding-window rate limiter per client IP.

    Two tiers:
      - Generation endpoints (POST /api/generate): strict limit
      - All other API endpoints: generous limit
    Health checks are exempt (used for heartbeat polling).
    """

    def __init__(
        self,
        generate_rpm: int = 10,
        general_rpm: int = 60,
    ) -> None:
        self._generate_rpm = generate_rpm
        self._general_rpm = general_rpm
        self._lock = Lock()
        # {ip: [timestamps]} — separate buckets per tier
        self._generate_hits: dict[str, list[float]] = {}
        self._general_hits: dict[str, list[float]] = {}

    def _check(self, bucket: dict[str, list[float]], ip: str, limit: int) -> bool:
        """Return True if request is allowed, False if rate-limited."""
        now = time.monotonic()
        window = 60.0  # 1-minute sliding window
        with self._lock:
            hits = bucket.get(ip, [])
            # Prune expired entries
            hits = [t for t in hits if now - t < window]
            if len(hits) >= limit:
                bucket[ip] = hits
                return False
            hits.append(now)
            bucket[ip] = hits
            return True

    def allow_generate(self, ip: str) -> bool:
        return self._check(self._generate_hits, ip, self._generate_rpm)

    def allow_general(self, ip: str) -> bool:
        return self._check(self._general_hits, ip, self._general_rpm)


_rate_limiter = RateLimiter()

# Paths exempt from rate limiting (health check for heartbeat polling)
_RATE_LIMIT_EXEMPT = {"/api/health"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply per-IP rate limiting to API endpoints."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        path = request.url.path
        # Only rate-limit /api/* endpoints
        if not path.startswith("/api/") or path in _RATE_LIMIT_EXEMPT:
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"

        if path == "/api/generate" and request.method == "POST":
            if not _rate_limiter.allow_generate(ip):
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded: max 10 generations/minute"
                    },
                )
        else:
            if not _rate_limiter.allow_general(ip):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded: max 60 requests/minute"},
                )

        return await call_next(request)


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
    # Output mode: audio, midi, or both
    output_mode: str = Field(default="audio", pattern=r"^(audio|midi|both)$")
    # Style/melody: paths are validated in _validate_audio_path() before use
    melody_path: Optional[str] = Field(default=None, max_length=1024)
    style_audio_path: Optional[str] = Field(default=None, max_length=1024)
    style_coef: float = Field(default=5.0, ge=0)
    # Audio-to-audio: reference audio path + strength (Phase 9f)
    reference_audio_path: Optional[str] = Field(default=None, max_length=1024)
    reference_strength: float = Field(default=0.7, ge=0.0, le=1.0)
    # LoRA adapter name or path (Phase 9g)
    lora: Optional[str] = Field(
        default=None, max_length=200, description="LoRA adapter name or path"
    )


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
    progress: float = 0.0  # 0.0 to 1.0
    has_midi: bool = False


class GenerateResponse(BaseModel):
    """Response for POST /api/generate."""

    id: str
    status: JobStatus


class ModelInfo(BaseModel):
    """Info about an available model."""

    name: str
    model_type: str
    is_loaded: bool


class EnhanceRequest(BaseModel):
    """Request for POST /api/enhance."""

    prompt: str = Field(..., min_length=1, max_length=5000)
    memory_context: str = Field(default="", max_length=10000)


class SettingsData(BaseModel):
    """Server-side settings."""

    llm_model: Optional[str] = None
    ai_enhance: bool = True
    history_context_count: int = Field(default=50, ge=0, le=200)


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
        "progress",
        "midi_data",
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
        self.progress: float = 0.0
        self.midi_data: bytes | None = None


# ---------------------------------------------------------------------------
# App State
# ---------------------------------------------------------------------------

# Global state — initialized in startup
_pipeline_cache = PipelineCache(max_size=2)
_executor = ThreadPoolExecutor(max_workers=1)  # Single worker — MLX is GPU-bound
_jobs: dict[str, _Job] = {}
_weights_dirs: dict[str, str] = {}  # name -> path
_max_jobs = 100

# Phase 7b: LLM + Memory + Settings state
_llm_model_path: str | None = None
_llm_model_id: str | None = None
_SETTINGS_PATH = Path.home() / ".mlx-audiogen" / "settings.json"
_MEMORY_PATH = Path.home() / ".mlx-audiogen" / "prompt_memory.json"
_prompt_memory: object | None = None  # Lazy-loaded PromptMemory
_server_settings: dict = {
    "llm_model": None,
    "ai_enhance": True,
    "history_context_count": 50,
}


def _load_settings() -> None:
    """Load settings from disk if they exist."""
    global _server_settings
    if _SETTINGS_PATH.is_file():
        try:
            _server_settings.update(json_mod.loads(_SETTINGS_PATH.read_text()))
        except (json_mod.JSONDecodeError, OSError):
            pass


def _save_settings() -> None:
    """Persist settings to disk."""
    _SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SETTINGS_PATH.write_text(json_mod.dumps(_server_settings, indent=2))


def _get_memory():
    """Get or create the PromptMemory singleton."""
    global _prompt_memory
    if _prompt_memory is None:
        from mlx_audiogen.shared.prompt_suggestions import PromptMemory

        _prompt_memory = PromptMemory(_MEMORY_PATH)
    return _prompt_memory


_load_settings()

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

# Rate limiting: protect public deployments from abuse
app.add_middleware(RateLimitMiddleware)

# ---------------------------------------------------------------------------
# Static Files (Web UI)
# ---------------------------------------------------------------------------

# Serve the built React SPA from web/dist/ if it exists.
# This is mounted AFTER the API routes so /api/* takes priority.
_WEB_DIST = Path(__file__).resolve().parent.parent.parent / "web" / "dist"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health_check() -> dict[str, str]:
    """Health check endpoint for browser heartbeat."""
    return {"status": "ok"}


@app.get("/api/jobs")
def list_jobs() -> list[JobInfo]:
    """List all active and recent jobs (for multi-instance monitoring)."""
    return [
        JobInfo(
            id=job.id,
            status=job.status,
            model=job.request.model,
            prompt=job.request.prompt,
            seconds=job.request.seconds,
            created_at=job.created_at,
            completed_at=job.completed_at,
            error=job.error,
            sample_rate=job.sample_rate,
            progress=job.progress,
            has_midi=job.midi_data is not None,
        )
        for job in _jobs.values()
    ]


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


# ---------------------------------------------------------------------------
# Prompt AI Endpoints (Phase 6.1 + 6.5)
# ---------------------------------------------------------------------------


class PromptSuggestRequest(BaseModel):
    """Request for prompt suggestions."""

    prompt: str = Field(..., min_length=1, max_length=5000)
    count: int = Field(default=4, ge=1, le=10)


class MidiToPromptRequest(BaseModel):
    """Request for MIDI-to-prompt conversion."""

    midi_path: str = Field(..., max_length=1024)


@app.post("/api/suggest")
def suggest_prompts(req: PromptSuggestRequest) -> dict:
    """Generate refined prompt suggestions from a base prompt."""
    from mlx_audiogen.shared.prompt_suggestions import analyze_prompt

    analysis = analyze_prompt(req.prompt, count=req.count)
    return analysis


@app.post("/api/midi-to-prompt")
def midi_to_prompt(req: MidiToPromptRequest) -> dict:
    """Analyze a MIDI file and generate a descriptive prompt."""
    from mlx_audiogen.shared.midi_to_prompt import midi_to_prompt as m2p

    path = Path(req.midi_path)
    if not path.is_file():
        raise HTTPException(400, f"MIDI file not found: {req.midi_path}")
    if path.suffix.lower() not in (".mid", ".midi"):
        raise HTTPException(400, "File must be .mid or .midi")

    midi_bytes = path.read_bytes()
    prompt = m2p(midi_bytes)
    return {"prompt": prompt, "path": str(path)}


@app.post("/api/separate/{job_id}")
def separate_stems(job_id: str) -> dict:
    """Separate a completed job's audio into stems.

    Returns download URLs for each stem (bass, mid, high — or
    drums, bass, vocals, other if Demucs is available).
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status != JobStatus.DONE or job.audio is None:
        raise HTTPException(400, "Job not complete or no audio")

    from mlx_audiogen.shared.stem_separator import encode_stems_wav, separate

    stems = separate(job.audio, job.sample_rate or 32000)
    wav_stems = encode_stems_wav(stems, job.sample_rate or 32000)

    # Store stems as sub-jobs for download
    stem_ids = {}
    for stem_name, wav_bytes in wav_stems.items():
        stem_id = f"{job_id}_stem_{stem_name}"
        stem_job = _Job(stem_id, job.request)
        stem_job.status = JobStatus.DONE
        stem_job.completed_at = time.time()
        stem_job.sample_rate = job.sample_rate
        stem_job.channels = 1
        # Store WAV bytes directly — decode back to numpy for the job
        import soundfile as sf

        audio_data, _sr = sf.read(io.BytesIO(wav_bytes))
        stem_job.audio = np.array(audio_data, dtype=np.float32)
        _jobs[stem_id] = stem_job
        stem_ids[stem_name] = stem_id

    return {"stems": stem_ids}


# ---------------------------------------------------------------------------
# Preset Marketplace (Phase 6.8)
# ---------------------------------------------------------------------------

_PRESETS_DIR = Path.home() / ".mlx-audiogen" / "presets"


@app.get("/api/presets")
def list_presets() -> list[dict]:
    """List shared presets from ~/.mlx-audiogen/presets/."""
    _PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    presets = []
    for f in sorted(_PRESETS_DIR.glob("*.mlxpreset")):
        import json as json_mod

        try:
            data = json_mod.loads(f.read_text())
            presets.append(
                {
                    "name": f.stem,
                    "filename": f.name,
                    "prompt": data.get("prompt", ""),
                    "model": data.get("model", ""),
                }
            )
        except (json_mod.JSONDecodeError, OSError):
            continue
    return presets


@app.post("/api/presets/{name}")
def save_shared_preset(name: str, req: GenerateRequest) -> dict:
    """Save a preset to the shared marketplace directory."""
    import json as json_mod

    _PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    # Sanitize name
    safe_name = "".join(c for c in name if c.isalnum() or c in "._- ")[:100]
    if not safe_name:
        raise HTTPException(400, "Invalid preset name")

    preset_file = _PRESETS_DIR / f"{safe_name}.mlxpreset"
    preset_data = req.model_dump()
    preset_file.write_text(json_mod.dumps(preset_data, indent=2))
    return {"saved": safe_name, "path": str(preset_file)}


@app.get("/api/presets/{name}")
def load_shared_preset(name: str) -> dict:
    """Load a preset from the shared marketplace directory."""
    import json as json_mod

    safe_name = "".join(c for c in name if c.isalnum() or c in "._- ")[:100]
    preset_file = _PRESETS_DIR / f"{safe_name}.mlxpreset"
    if not preset_file.is_file():
        raise HTTPException(404, f"Preset not found: {safe_name}")

    data = json_mod.loads(preset_file.read_text())
    return data


# ---------------------------------------------------------------------------
# Phase 7b: LLM Enhancement, Tags, Memory, Settings
# ---------------------------------------------------------------------------


@app.post("/api/enhance")
def enhance_prompt(req: EnhanceRequest) -> dict:
    """Enhance a prompt using local LLM or template fallback."""
    from mlx_audiogen.shared.prompt_suggestions import enhance_with_llm

    result = enhance_with_llm(
        prompt=req.prompt,
        model_path=_llm_model_path,
        memory_context=req.memory_context,
    )
    return result


@app.get("/api/tags")
def get_tags() -> dict:
    """Return the full tag database for autocomplete."""
    from mlx_audiogen.shared.prompt_suggestions import TAG_DATABASE

    return TAG_DATABASE


@app.get("/api/llm/models")
def list_llm_models() -> list[dict]:
    """List locally available MLX LLM models."""
    from mlx_audiogen.shared.prompt_suggestions import discover_mlx_models

    return discover_mlx_models()


@app.post("/api/llm/select")
def select_llm_model(req: dict) -> dict:
    """Select the active LLM model by ID."""
    global _llm_model_path, _llm_model_id
    model_id = req.get("model_id")
    if model_id is None:
        # Deselect
        _llm_model_path = None
        _llm_model_id = None
        return {"model_id": None, "status": "deselected"}

    from mlx_audiogen.shared.prompt_suggestions import discover_mlx_models

    models = discover_mlx_models()
    for m in models:
        if m["id"] == model_id:
            # Model found — store the ID (mlx-lm resolves HF cache paths)
            _llm_model_id = model_id
            _llm_model_path = model_id
            return {"model_id": model_id, "status": "selected"}

    raise HTTPException(404, f"Model not found: {model_id}")


@app.get("/api/llm/status")
def get_llm_status() -> dict:
    """Return current LLM status."""
    return {
        "model_id": _llm_model_id,
        "loaded": _llm_model_path is not None,
        "idle_seconds": 0,
        "memory_mb": 0,
    }


@app.get("/api/memory")
def get_memory() -> dict:
    """Return prompt memory data."""
    mem = _get_memory()
    return mem.to_dict()


@app.delete("/api/memory")
def clear_memory() -> dict:
    """Clear all prompt memory."""
    mem = _get_memory()
    mem.clear()
    return {"status": "cleared"}


@app.get("/api/memory/export")
def export_memory() -> Response:
    """Export prompt memory as downloadable JSON."""
    mem = _get_memory()
    data = json_mod.dumps(mem.to_dict(), indent=2)
    return Response(
        content=data,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="prompt_memory.json"'},
    )


@app.post("/api/memory/import")
async def import_memory(file: UploadFile) -> dict:
    """Import prompt memory from uploaded JSON file."""
    content = await file.read()
    try:
        data = json_mod.loads(content)
    except json_mod.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON file")

    if not isinstance(data, dict) or "history" not in data:
        raise HTTPException(400, "Invalid memory format: missing 'history' key")

    mem = _get_memory()
    mem.history = data["history"]
    mem._derive_profile()
    mem.save()
    return {"status": "imported", "entries": len(mem.history)}


@app.get("/api/settings")
def get_settings() -> dict:
    """Return current server settings."""
    return dict(_server_settings)


@app.post("/api/settings")
def update_settings(req: SettingsData) -> dict:
    """Update server settings."""
    global _server_settings, _llm_model_path, _llm_model_id
    updates = req.model_dump(exclude_unset=True)
    _server_settings.update(updates)
    _save_settings()

    # If LLM model changed, update the path
    if "llm_model" in updates:
        _llm_model_id = updates["llm_model"]
        _llm_model_path = updates["llm_model"]

    return dict(_server_settings)


# ---------------------------------------------------------------------------
# LoRA Endpoints (Phase 9g)
# ---------------------------------------------------------------------------

# Track which LoRA is currently applied to each pipeline (by model name)
_active_loras: dict[str, str | None] = {}
# Training state
_active_trainer: dict[str, object] = {}  # job_id -> LoRATrainer
_training_lock = Lock()


class TrainRequest(BaseModel):
    """Request body for POST /api/train."""

    data_dir: str = Field(
        ..., min_length=1, max_length=1024, description="Path to training data"
    )
    base_model: str = Field(
        ..., min_length=1, max_length=200, description="Base model name"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9_-]+$",
        description="LoRA adapter name",
    )
    profile: str = Field(default="balanced", pattern=r"^(quick|balanced|deep)$")
    rank: Optional[int] = Field(default=None, ge=1, le=256)
    alpha: Optional[float] = Field(default=None, gt=0)
    targets: Optional[list[str]] = Field(default=None)
    epochs: int = Field(default=10, ge=1, le=100)
    learning_rate: float = Field(default=1e-4, gt=0, le=1.0)
    batch_size: int = Field(default=1, ge=1, le=8)
    chunk_seconds: float = Field(default=10.0, ge=5.0, le=40.0)
    early_stop: bool = Field(default=True)
    patience: int = Field(default=3, ge=1, le=20)


@app.get("/api/loras")
def list_loras() -> list[dict]:
    """List available LoRA adapters."""
    from mlx_audiogen.lora.trainer import list_available_loras

    return list_available_loras()


@app.get("/api/loras/{name}")
def get_lora(name: str) -> dict:
    """Get config for a specific LoRA adapter."""
    from mlx_audiogen.lora.config import DEFAULT_LORAS_DIR
    from mlx_audiogen.lora.trainer import load_lora_config

    lora_dir = DEFAULT_LORAS_DIR / name
    if not lora_dir.is_dir():
        raise HTTPException(404, f"LoRA not found: {name}")
    try:
        cfg = load_lora_config(lora_dir)
        return cfg.to_dict()
    except FileNotFoundError:
        raise HTTPException(404, f"LoRA config not found: {name}")


@app.delete("/api/loras/{name}")
def delete_lora(name: str) -> dict:
    """Delete a LoRA adapter."""
    import shutil

    from mlx_audiogen.lora.config import DEFAULT_LORAS_DIR

    # Validate name (prevent path traversal)
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(400, "Invalid LoRA name")

    lora_dir = DEFAULT_LORAS_DIR / name
    if not lora_dir.is_dir():
        raise HTTPException(404, f"LoRA not found: {name}")

    shutil.rmtree(lora_dir)
    return {"deleted": name}


@app.post("/api/train")
def start_training(req: TrainRequest) -> dict:
    """Start LoRA training in a background thread."""
    with _training_lock:
        if _active_trainer:
            raise HTTPException(409, "Training already in progress")

    import mlx.core as mx

    from mlx_audiogen.lora.config import PROFILES, LoRAConfig
    from mlx_audiogen.lora.dataset import (
        apply_delay_pattern,
        chunk_audio,
        load_and_prepare_audio,
        scan_dataset,
    )
    from mlx_audiogen.lora.trainer import LoRATrainer
    from mlx_audiogen.models.musicgen import MusicGenPipeline

    # Validate data_dir
    data_dir = Path(req.data_dir)
    if ".." in data_dir.parts:
        raise HTTPException(400, "data_dir must not contain '..'")
    if not data_dir.is_dir():
        raise HTTPException(400, f"Data directory not found: {req.data_dir}")

    # Load pipeline
    pipe: MusicGenPipeline = _get_pipeline("musicgen")  # type: ignore[assignment]
    hidden_size = pipe.config.decoder.hidden_size

    # Build config from profile + overrides
    profile = PROFILES[req.profile]
    rank = req.rank if req.rank is not None else profile.rank
    alpha = req.alpha if req.alpha is not None else profile.alpha
    targets = req.targets if req.targets else profile.targets

    config = LoRAConfig(
        name=req.name,
        base_model=req.base_model,
        hidden_size=hidden_size,
        rank=rank,
        alpha=alpha,
        targets=targets,
        profile=req.profile,
        chunk_seconds=req.chunk_seconds,
        epochs=req.epochs,
        learning_rate=req.learning_rate,
        batch_size=req.batch_size,
        early_stop=req.early_stop,
        patience=req.patience,
    )

    # Scan and prepare data
    try:
        entries = scan_dataset(data_dir)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))

    training_data = []
    for entry in entries:
        audio = load_and_prepare_audio(entry["file"], target_sr=pipe.sample_rate)
        chunks = chunk_audio(audio, pipe.sample_rate, req.chunk_seconds)
        for chunk in chunks:
            chunk_mx = mx.array(chunk)[mx.newaxis, :, mx.newaxis]
            codes, _scales = pipe.encodec.encode(chunk_mx)
            tokens = mx.swapaxes(codes, 1, 2)
            delayed, valid = apply_delay_pattern(
                tokens,
                pipe.config.decoder.num_codebooks,
                pipe.config.decoder.bos_token_id,
            )
            training_data.append(
                {
                    "delayed_tokens": delayed,
                    "valid_mask": valid,
                    "text": entry["text"],
                }
            )

    if not training_data:
        raise HTTPException(400, "No valid training samples produced")

    # Create trainer
    job_id = str(uuid.uuid4())[:8]
    trainer = LoRATrainer(
        pipeline=pipe,
        config=config,
        training_data=training_data,
    )

    with _training_lock:
        _active_trainer[job_id] = trainer

    # Run in thread pool
    def _run_training() -> None:
        try:
            trainer.train()
        except Exception as e:
            print(f"Training {job_id} failed: {e}")
        finally:
            with _training_lock:
                _active_trainer.pop(job_id, None)

    _executor.submit(_run_training)
    return {"id": job_id, "status": "running"}


@app.get("/api/train/status/{job_id}")
def get_train_status(job_id: str) -> dict:
    """Get training progress."""
    with _training_lock:
        trainer = _active_trainer.get(job_id)
    if trainer is None:
        raise HTTPException(404, f"Training job not found: {job_id}")
    return trainer.status  # type: ignore[attr-defined]


@app.post("/api/train/stop/{job_id}")
def stop_training(job_id: str) -> dict:
    """Stop an active training job."""
    with _training_lock:
        trainer = _active_trainer.get(job_id)
    if trainer is None:
        raise HTTPException(404, f"Training job not found: {job_id}")
    trainer.stop()  # type: ignore[attr-defined]
    return {"stopped": job_id}


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
        progress=job.progress,
        has_midi=job.midi_data is not None,
    )


@app.get("/api/audio/{job_id}")
def get_audio(job_id: str, fmt: str = "aiff") -> Response:
    """Download generated audio in the requested format.

    Args:
        job_id: The job ID.
        fmt: Audio format — aiff (default), wav, flac.

    Returns 404 if job not found, 202 if still running, or audio file if done.
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

    fmt = fmt.lower()
    if fmt not in _AUDIO_FORMATS:
        raise HTTPException(
            400, f"Unsupported format: {fmt}. Use: {', '.join(_AUDIO_FORMATS)}"
        )

    audio_bytes = _encode_audio(job.audio, job.sample_rate or 32000, job.channels, fmt)
    ext = fmt
    media_type = _AUDIO_MEDIA_TYPES.get(fmt, "application/octet-stream")
    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{job_id}.{ext}"',
        },
    )


@app.get("/api/midi/{job_id}")
def get_midi(job_id: str) -> Response:
    """Download generated MIDI file.

    Available when output_mode is 'midi' or 'both'.
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status != JobStatus.DONE:
        raise HTTPException(202, "Generation still in progress.")
    if job.midi_data is None:
        raise HTTPException(404, "No MIDI data for this job.")

    return Response(
        content=job.midi_data,
        media_type="audio/midi",
        headers={
            "Content-Disposition": f'attachment; filename="{job_id}.mid"',
        },
    )


# ---------------------------------------------------------------------------
# Generation Worker
# ---------------------------------------------------------------------------


def _run_generation(job: _Job) -> None:
    """Execute generation in the thread pool worker."""
    try:
        job.status = JobStatus.RUNNING

        def on_progress(frac: float) -> None:
            """Callback: update job progress (0.0 to 1.0)."""
            job.progress = max(0.0, min(1.0, frac))

        req = job.request

        if req.model == "stable_audio":
            audio, sr, channels = _generate_stable_audio(req, on_progress)
        elif req.model == "musicgen":
            audio, sr, channels = _generate_musicgen(req, on_progress)
        else:
            raise ValueError(f"Unknown model: {req.model}")

        job.audio = audio
        job.sample_rate = sr
        job.channels = channels

        # Generate MIDI if requested
        output_mode = req.output_mode
        if output_mode in ("midi", "both") and audio is not None:
            from mlx_audiogen.shared.audio_to_midi import audio_to_midi

            job.midi_data = audio_to_midi(audio, sr, bpm=120.0)

        # In MIDI-only mode, we still keep the audio for the transcription
        # but the client knows to fetch /api/midi instead of /api/audio

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


def _apply_lora_to_pipeline(
    pipe: object,
    lora_name: str | None,
) -> None:
    """Apply or swap a LoRA adapter on a MusicGen pipeline.

    Tracks which LoRA is currently applied via _active_loras dict.
    If the requested LoRA is already applied, does nothing.
    If a different LoRA is applied, removes it first.
    """
    import mlx.core as mx

    from mlx_audiogen.lora.config import DEFAULT_LORAS_DIR
    from mlx_audiogen.lora.inject import apply_lora, remove_lora
    from mlx_audiogen.lora.trainer import load_lora_config
    from mlx_audiogen.shared.hub import load_safetensors

    model = pipe.model  # type: ignore[attr-defined]
    pipe_id = id(pipe)
    current = _active_loras.get(str(pipe_id))

    if lora_name == current:
        return  # Already applied (or both None)

    # Remove current LoRA if any
    if current is not None:
        remove_lora(model)
        _active_loras[str(pipe_id)] = None

    if lora_name is None:
        return

    # Load and apply new LoRA
    lora_dir = DEFAULT_LORAS_DIR / lora_name
    if not lora_dir.is_dir():
        raise ValueError(f"LoRA not found: {lora_name}")

    lora_config = load_lora_config(lora_dir)
    apply_lora(
        model,
        targets=lora_config.targets,
        rank=lora_config.rank,
        alpha=lora_config.alpha,
    )
    lora_weights = load_safetensors(lora_dir / "lora.safetensors")
    model.load_weights(
        [(k, mx.array(v)) for k, v in lora_weights.items()],
        strict=False,
    )
    _active_loras[str(pipe_id)] = lora_name
    print(f"LoRA applied: {lora_name}")


def _generate_musicgen(
    req: GenerateRequest,
    on_progress: object = None,
) -> tuple[np.ndarray, int, int]:
    """Generate audio using MusicGen pipeline."""
    from mlx_audiogen.models.musicgen import MusicGenPipeline

    pipe: MusicGenPipeline = _get_pipeline("musicgen")  # type: ignore[assignment]

    # Apply/swap LoRA if requested
    _apply_lora_to_pipeline(pipe, req.lora)

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
        progress_callback=on_progress,
    )
    # MusicGen outputs mono by default; stereo variants output interleaved
    channels = 1
    if pipe.config.decoder.num_codebooks > 4:
        channels = 2

    audio = _trim_to_exact_duration(audio, req.seconds, pipe.sample_rate, channels)  # type: ignore[arg-type, assignment]
    return audio, pipe.sample_rate, channels  # type: ignore[return-value]


def _generate_stable_audio(
    req: GenerateRequest,
    on_progress: object = None,
) -> tuple[np.ndarray, int, int]:
    """Generate audio using Stable Audio pipeline."""
    from mlx_audiogen.models.stable_audio import StableAudioPipeline

    pipe: StableAudioPipeline = _get_pipeline("stable_audio")  # type: ignore[assignment]

    # Load reference audio for audio-to-audio if provided
    ref_audio_mx = None
    if req.reference_audio_path:
        validated = _validate_audio_path(req.reference_audio_path, "reference_audio")
        if validated:
            import mlx.core as mx

            from mlx_audiogen.shared.audio_io import load_wav

            ref_np, ref_sr = load_wav(validated)
            # Resample to 44.1kHz if needed (simple linear for now)
            if ref_sr != 44100:
                import numpy as np_ref

                target_len = int(len(ref_np) * 44100 / ref_sr)
                ref_np = np_ref.interp(
                    np_ref.linspace(0, len(ref_np) - 1, target_len),
                    np_ref.arange(len(ref_np)),
                    ref_np,
                )
            # Shape: (1, 1, samples) for mono, (1, 2, samples) for stereo
            ref_audio_mx = mx.array(ref_np).reshape(1, 1, -1)

    audio = pipe.generate(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        seconds_total=req.seconds,
        steps=req.steps,
        cfg_scale=req.cfg_scale,
        seed=req.seed,
        sampler=req.sampler,
        progress_callback=on_progress,
        reference_audio=ref_audio_mx,
        reference_strength=req.reference_strength,
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


_AUDIO_FORMATS = {"wav", "aiff", "flac"}
_AUDIO_MEDIA_TYPES = {
    "wav": "audio/wav",
    "aiff": "audio/aiff",
    "flac": "audio/flac",
}


def _encode_audio(
    audio: np.ndarray, sample_rate: int, channels: int, fmt: str = "aiff"
) -> bytes:
    """Encode audio array in the specified format (wav, aiff, flac)."""
    import soundfile as sf

    buf = io.BytesIO()
    if channels > 1 and audio.ndim == 1:
        audio = audio.reshape(-1, channels)

    sf_format = fmt.upper()
    subtype = "FLOAT" if fmt != "flac" else "PCM_24"
    sf.write(buf, audio, sample_rate, format=sf_format, subtype=subtype)
    return buf.getvalue()


def _encode_wav(audio: np.ndarray, sample_rate: int, channels: int) -> bytes:
    """Encode audio array as WAV bytes in memory (backward compat)."""
    return _encode_audio(audio, sample_rate, channels, "wav")


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


def launch_app():
    """One-command launcher: auto-discovers all converted models and opens the Web UI.

    Usage:
        mlx-audiogen-app
        mlx-audiogen-app --port 8420 --max-cache 3
    """
    parser = argparse.ArgumentParser(
        description="Launch MLX AudioGen with all available models"
    )
    parser.add_argument(
        "--converted-dir",
        type=str,
        default=None,
        help="Directory containing converted model folders (default: ./converted)",
    )
    parser.add_argument(
        "--port", type=int, default=8420, help="Server port (default: 8420)"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--max-cache",
        type=int,
        default=3,
        help="Max pipelines to keep in memory (default: 3)",
    )

    args = parser.parse_args()

    # Auto-discover converted model directories
    converted_root = Path(args.converted_dir) if args.converted_dir else None

    # Search common locations
    search_paths = []
    if converted_root:
        search_paths.append(converted_root.resolve())
    else:
        # Try relative to CWD, then relative to package
        search_paths.append(Path.cwd() / "converted")
        search_paths.append(Path(__file__).resolve().parent.parent.parent / "converted")

    # Also check the auto-download models directory
    from mlx_audiogen.shared.model_registry import DEFAULT_MODELS_DIR

    search_paths.append(DEFAULT_MODELS_DIR)

    found_models = []
    for root in search_paths:
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if child.is_dir() and child.name not in (".", ".."):
                    # Follow symlinks (auto-download creates symlinks to HF cache)
                    target = child.resolve() if child.is_symlink() else child
                    # Quick check: does it look like a model dir?
                    has_config = (target / "config.json").exists()
                    has_weights = any(target.glob("*.safetensors"))
                    if has_config or has_weights:
                        if child.name not in [m.name for m in found_models]:
                            found_models.append(child)

    if not found_models:
        print("No converted models found.")
        print("Expected locations: ./converted/, ~/.mlx-audiogen/models/")
        print("Run mlx-audiogen-convert first, or pass --converted-dir")
        print("Models can also be auto-downloaded: use --model-name <name>")
        sys.exit(1)

    # Register all found models
    print(f"Found {len(found_models)} model(s):")
    for model_path in found_models:
        name = model_path.name
        _weights_dirs[name] = str(model_path)
        print(f"  {name}")

    # Configure cache
    global _pipeline_cache
    _pipeline_cache = PipelineCache(max_size=args.max_cache)

    # Pre-load first model
    first_name, first_path = next(iter(_weights_dirs.items()))
    print(f"\nPre-loading {first_name}...")
    try:
        if "stable" in first_name.lower():
            from mlx_audiogen.models.stable_audio import StableAudioPipeline

            pipe = StableAudioPipeline.from_pretrained(first_path)
        else:
            from mlx_audiogen.models.musicgen import MusicGenPipeline

            pipe = MusicGenPipeline.from_pretrained(first_path)
        _pipeline_cache.put(first_name, pipe)
    except Exception as e:
        print(f"Warning: Failed to pre-load {first_name}: {e}")

    url = f"http://{args.host}:{args.port}"
    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"API docs: {url}/docs")
    if _WEB_DIST.is_dir():
        print(f"Web UI:   {url}/")

    import threading
    import webbrowser

    if _WEB_DIST.is_dir():
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
