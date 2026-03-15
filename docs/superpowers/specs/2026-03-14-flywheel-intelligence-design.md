# Phase 9g-4: Re-training Loop + Flywheel Intelligence

**Date:** 2026-03-14
**Status:** Approved
**Approach:** B — Smart Flywheel (orchestrator + intelligence layers)

## Overview

Phase 9g-4 is the capstone that connects LoRA training (9g), library scanning (9g-2), and enrichment/taste (9g-3) into a self-improving cycle. The flywheel: library metadata + audio -> taste profile + LoRA -> better suggestions + generations -> user feedback (stars) -> back into both loops.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Trigger mechanism | Semi-automatic (threshold-based) | Trains when N starred gens accumulate; threshold configurable via CLI + UI |
| Dataset strategy | Cumulative + versioned snapshots | Each retrain uses ALL accumulated data; prior versions preserved independently |
| Blend ratio | User-controlled slider, default 80/20 library/generations | Library-anchored prevents feedback-loop drift; slider gives full control (0-100%) |
| Quality signal | Explicit star rating | Star button on history entries; user clicks when they like a generation |
| Enrichment role | Descriptions + taste signals | Enrichment tags merged into training descriptions AND fed into taste profile |
| Flywheel surfacing | Smarter suggestions + adapter changelog | No nudge banners; auto-retrain is silent; changelog shows version evolution |
| Taste refresh | Trigger-based (every N stars + post-retrain + post-enrichment) | More responsive than scheduled; default refresh every 5 stars |
| Future exploration | Approach C — automated quality metrics | Self-correcting feedback loop with spectral comparison; deferred to post-Phase 10 |

## Data Model

### Kept Generations

Starred generations persist to disk with their audio and metadata:

```
~/.mlx-audiogen/kept/{adapter_name}/
  gen_{job_id}.wav        # the starred audio
  gen_{job_id}.json       # metadata (see schema below)
```

**Audio lifecycle:** The server's in-memory job store cleans up completed jobs after 5 minutes. To handle starring after cleanup, the star endpoint accepts an optional WAV file upload from the client (the web UI holds the audio as a blob URL). Flow:
1. If `job.audio` is still in memory (< 5 min), save directly from server memory.
2. If job has been cleaned up, the client sends the audio blob in the star request body.
3. If neither source is available, return HTTP 410 Gone with a message that the audio has expired.

The web UI eagerly downloads generation audio as blob URLs (same pattern used for stem separation), so audio is always available client-side for starring even after server cleanup.

Kept generation metadata schema:
```json
{
  "job_id": "abc123",
  "prompt": "deep house atmospheric pads",
  "model": "musicgen",
  "adapter_name": "my-style",
  "adapter_version": "v2",
  "params": {
    "temperature": 1.0,
    "top_k": 250,
    "guidance_scale": 3.0,
    "seconds": 10
  },
  "starred_at": "2026-03-14T15:30:00Z"
}
```

### Adapter Versioning

Each retrain creates an independent version directory:

```
~/.mlx-audiogen/loras/{adapter_name}/
  v1/
    lora.safetensors      # adapter weights
    config.json           # LoRA hyperparams (rank, targets, alpha, etc.)
    changelog.json        # training composition metadata
  v2/
    lora.safetensors
    config.json
    changelog.json
  v3/ ...
  active -> v3/           # symlink to current version
```

**Backward compatibility with existing flat-layout LoRAs:**
Existing adapters stored as `~/.mlx-audiogen/loras/{name}/lora.safetensors` (no version subdirectories) are auto-migrated on first access:
1. `list_available_loras()` detects flat layout (config.json at top level, no `v*/` dirs)
2. Creates `v1/` subdirectory, moves `lora.safetensors` + `config.json` into it
3. Creates `active -> v1/` symlink
4. Writes a minimal `changelog.json` with `parent_version: null` and `version: 1`
5. Migration is idempotent — already-versioned adapters are left unchanged

**LoRA loading path changes:**
- `list_available_loras()` updated to follow `active` symlink (or detect flat layout + migrate)
- `get_lora()` endpoint reads from `{name}/active/config.json` instead of `{name}/config.json`
- `delete_lora()` deletes the entire `{name}/` directory (all versions)
- `load_lora_config()` follows `active` symlink
- Flywheel's `start_retrain()` passes `output_dir = DEFAULT_LORAS_DIR / name / f"v{N+1}"` to `LoRATrainer`
- `trainer.py`'s `save_lora()` is unchanged — it writes to whatever `output_dir` is provided

Changelog schema:
```json
{
  "version": 3,
  "created_at": "2026-03-14T15:30:00Z",
  "parent_version": 2,
  "dataset": {
    "library_tracks": 45,
    "kept_generations": 12,
    "blend_ratio": {"library": 80, "generations": 20},
    "total_training_samples": 57
  },
  "top_influences": {
    "genre": [{"tag": "deep house", "pct": 38}, {"tag": "minimal techno", "pct": 22}],
    "mood": [{"tag": "atmospheric", "pct": 30}],
    "instrument": [{"tag": "808", "pct": 15}]
  },
  "new_since_parent": {
    "kept_generations_added": 5,
    "enrichment_tags_added": 30,
    "library_tracks_added": 0
  },
  "training": {
    "profile": "balanced",
    "epochs": 20,
    "best_loss": 0.42,
    "duration_seconds": 180
  }
}
```

Note: `parent_version` is `null` for v1 (no parent).

Each version is fully independent — deleting v2 does not affect v3.

## Architecture

### Core Module: `mlx_audiogen/lora/flywheel.py`

**FlywheelConfig** dataclass (persisted in `~/.mlx-audiogen/settings.json`):
- `retrain_threshold: int` — starred generations before auto-retrain (default: 10)
- `blend_ratio: int` — library percentage 0-100 (default: 80)
- `taste_refresh_interval: int` — refresh taste every N stars (default: 5)
- `auto_retrain: bool` — enable/disable auto-retrain (default: True)
- `base_collection: str | None` — collection providing library tracks

**FlywheelManager** class:

| Method | Purpose |
|--------|---------|
| `record_star(job_id, audio_data, metadata)` | Save WAV + metadata to kept dir, increment counter, check threshold. `audio_data` is numpy array (from job memory) or raw WAV bytes (from client upload) |
| `remove_star(job_id)` | Remove WAV + metadata from kept dir, decrement counter |
| `check_threshold(adapter_name)` | If stars >= threshold AND auto_retrain: trigger `start_retrain()` |
| `start_retrain(adapter_name)` | Build cumulative dataset, call trainer, create new version |
| `build_dataset(adapter_name)` | Merge library tracks + kept gens at blend ratio, enrich descriptions |
| `create_version(adapter_name, training_result)` | Create version dir, write changelog, update symlink |
| `get_versions(adapter_name)` | List all versions with changelog summaries |
| `get_changelog(adapter_name, version)` | Full changelog for specific version |
| `revert_version(adapter_name, version)` | Update active symlink |
| `reset_cache(adapter_name)` | Clear kept dir, reset star count |
| `compute_top_influences(dataset)` | Analyze dataset to extract genre/mood/instrument percentages |

### Data Flow

```
User stars generation in History
        |
        v
FlywheelManager.record_star()
  -> saves WAV + metadata to ~/.mlx-audiogen/kept/
  -> increments counter
        |
        v
Stars count % taste_refresh_interval == 0?
  -> YES: refresh taste profile (collect_flywheel_signals + collect_library_signals with enrichment_tags)
  -> NO: continue
        |
        v
check_threshold() -- stars >= retrain_threshold?
  -> NO: done, wait for more
  -> YES: start_retrain()
        |
        v
build_dataset():
  +-- Library tracks via collection_to_training_data()
  |     enrichment tags merged into descriptions via description_gen
  +-- Kept generations from kept/ dir
  |     description = prompt text + model/params context (LLM-enriched if available)
  +-- Apply blend ratio (default 80% library / 20% generations)
  +-- Output: list[dict] with {"file": path, "text": description} format
        |
        v
check _training_lock -- if training already in progress, skip (log warning)
        |
        v
trainer.py runs (output_dir = loras/{name}/v{N+1})
        |
        v
create_version(): loras/{name}/v{N+1}/
  -> lora.safetensors + config.json + changelog.json
  -> active symlink updated
        |
        v
Taste profile refreshed (enrichment_tags now populated + flywheel signals)
  -> prompt suggestions improve
```

### Enrichment Integration

**Enrichment -> Training Descriptions:**
- `description_gen.generate_description()` gains optional `enrichment` parameter
- When building training dataset, library tracks with enrichment data in SQLite get richer descriptions
- Example: `"house, 124 BPM, A minor"` becomes `"deep house minimal atmospheric late night 124 BPM A minor reverb-heavy layered synths Innervisions label"`

**Enrichment -> Taste Signals:**
- `collect_library_signals()` has existing empty `enrichment_tags` parameter — now populated
- Enrichment tags weighted by play count (consistent with existing genre/BPM signals)
- Taste profile gains: label affinity, production style, crowd-sourced mood tags

**Kept Generations -> Taste Signals:**
- New `collect_flywheel_signals(adapter_name)` in `signals.py`
- Reads all `gen_*.json` files from `~/.mlx-audiogen/kept/{adapter_name}/`
- Returns a dict matching the existing signal format used by `TasteEngine`:
  - `genres`: Counter of genre-related words extracted from prompts
  - `moods`: Counter of mood-related words extracted from prompts
  - `instruments`: Counter of instrument-related words extracted from prompts
  - `bpms`: list of BPM values from params (if present)
  - `keys`: list of key values from params (if present)
- Uses the same keyword lists from `prompt_suggestions.py` (GENRES, MOODS, INSTRUMENTS) for extraction
- `TasteEngine` gains `update_flywheel_signals(signals)` method that merges these into the profile
- Flywheel signals are weighted at 1.5x vs library signals (user explicitly approved these)

**Taste Auto-Refresh Triggers:**
1. Every N starred generations (default: 5, configurable)
2. After a retrain completes
3. After a batch enrichment finishes
4. On demand via existing refresh button

**Kept Generation Descriptions for Training:**
When `build_dataset()` processes kept generations, it creates training descriptions:
1. If LLM is available: sends the original prompt + model/params as context to the LLM to generate a rich description
2. If LLM unavailable: uses the original prompt directly as the training text (the user already wrote a good description — it's what generated the audio they liked)
3. The kept WAV file path is used as the `"file"` key in the training data dict

**Concurrency:**
If `start_retrain()` is triggered (by threshold or manual) while a training job is already running (`_training_lock` held), the retrain is skipped and a warning is logged. The star count is NOT reset — the next star after the current training completes will re-check the threshold.

**Rate Limiting:**
`POST /api/star/{job_id}` falls under the general API rate limit (60 req/min). Since it writes a WAV to disk only when starring (not unstarring), and the audio was already generated, disk abuse is bounded by the generation rate limit.

## Server API

### New Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/star/{job_id}` | Star a generation (saves audio + metadata). Accepts optional WAV upload for expired jobs. Returns `{starred: bool}` |
| `DELETE` | `/api/star/{job_id}` | Unstar a generation (removes kept WAV + metadata). Returns `{starred: bool}` |
| `GET` | `/api/flywheel/status` | Stars since last train, threshold, auto-retrain state |
| `GET` | `/api/flywheel/config` | Get flywheel settings |
| `PUT` | `/api/flywheel/config` | Update flywheel settings |
| `GET` | `/api/loras/{name}/versions` | List all versions with changelog summaries |
| `GET` | `/api/loras/{name}/versions/{v}` | Full changelog for specific version |
| `PUT` | `/api/loras/{name}/active/{v}` | Set active version (revert) |
| `POST` | `/api/flywheel/retrain/{name}` | Manual re-train now |
| `POST` | `/api/flywheel/reset/{name}` | Reset kept generations cache |

### Modified Endpoints

| Endpoint | Change |
|----------|--------|
| `GET /api/loras` | Adds `active_version`, `total_versions`, `stars_since_train` per adapter |
| `GET /api/status/{id}` | Adds `starred: bool` field |
| `GET /api/jobs` | Adds `starred: bool` field |

### Pydantic Models

```python
class FlywheelConfigModel(BaseModel):
    retrain_threshold: int = Field(default=10, ge=1, le=100)
    blend_ratio: int = Field(default=80, ge=0, le=100)
    taste_refresh_interval: int = Field(default=5, ge=1, le=50)
    auto_retrain: bool = Field(default=True)
    base_collection: str | None = Field(default=None, max_length=64)

class StarResponse(BaseModel):
    starred: bool
    stars_since_train: int

class DatasetComposition(BaseModel):
    library_tracks: int
    kept_generations: int
    blend_ratio: dict[str, int]  # {"library": 80, "generations": 20}
    total_training_samples: int

class TagInfluence(BaseModel):
    tag: str
    pct: float

class TopInfluences(BaseModel):
    genre: list[TagInfluence]
    mood: list[TagInfluence]
    instrument: list[TagInfluence]

class NewSinceParent(BaseModel):
    kept_generations_added: int
    enrichment_tags_added: int
    library_tracks_added: int

class TrainingStats(BaseModel):
    profile: str
    epochs: int
    best_loss: float
    duration_seconds: float

class VersionSummary(BaseModel):
    version: int
    created_at: str
    is_active: bool
    library_tracks: int
    kept_generations: int
    best_loss: float

class VersionChangelog(BaseModel):
    version: int
    created_at: str
    parent_version: int | None
    dataset: DatasetComposition
    top_influences: TopInfluences
    new_since_parent: NewSinceParent
    training: TrainingStats
```

**Settings persistence:** FlywheelConfig fields are stored as a nested `flywheel` key in `~/.mlx-audiogen/settings.json` (alongside existing top-level keys like `llm_model`, `ai_enhance`). The existing `_load_settings()` / `_save_settings()` functions handle the file I/O. `FlywheelManager` reads/writes the `flywheel` sub-dict on init and config updates.

## Web UI Changes

### History Panel — Star Button
- Yellow ★ button on each completed generation entry
- Click to star (POST) or unstar (DELETE) — explicit actions, not toggle
- Filled yellow when starred, outline when not
- When starring, client sends audio blob if job has been cleaned from server memory
- Star state persisted server-side (WAV + metadata in kept/ dir)

### LoRA Selector — Version Dropdown
- Shows `my-style (v3)` with small dropdown arrow
- Dropdown lists all versions with one-line summary
- Active version highlighted with checkmark
- Selecting a version calls `PUT /api/loras/{name}/active/{v}`

### Settings Tab — Flywheel Section
```
Flywheel Intelligence
+-- Auto-retrain:        [ON/OFF toggle]
+-- Retrain after:       [10] starred generations
+-- Taste refresh after: [5]  starred generations
+-- Dataset blend:       [========..] 80% Library <-> Generations
+-- [Re-train Now]       [Reset Cache]
+-- Adapter Changelog
    +-- my-style
        +-- v3 (active) -- 45 library + 12 kept, deep house 38%, minimal 22%
        +-- v2 -- 45 library + 7 kept, deep house 40%, techno 18%
        +-- v1 -- 45 library, deep house 42%
```

### Sidebar Layout Improvements
- **Collapsible parameter groups**: Disclosure triangles with summary lines
  - "Model & Prompt" section: always expanded
  - "Generation Parameters" section: collapsed by default, shows summary like `temp 1.0 | top-k 250 | cfg 3.0`
  - Click to expand/collapse
- **Drag-to-resize sidebar**: Thin drag handle on right edge
  - Min: 280px, Max: 480px, Default: 320px
  - Width persisted to localStorage
- Pattern matches DAW sidebar behavior (Ableton, Logic)

## Bug Fixes (Bundled)

These existing bugs will be fixed alongside the Phase 9g-4 implementation:

1. **Sidebar overflow** — Settings tab content bleeds into main area. Fix: proper overflow handling + collapsible sections
2. **Generate button occlusion** — pinned button cuts off parameter sliders. Fix: collapsible sections ensure all params visible above button
3. **Library path tilde expansion** — `~/Music/...` not resolved server-side. Fix: `os.path.expanduser()` in library source handler + file picker in UI
4. **LLM template fallback** — Qwen3.5-9B-6bit not loading, falling back to template. Fix: diagnose model discovery/loading issue
5. **Instrument tag recognition** — TR-909, 727, 606, Korg M1 not recognized as instruments. Fix: expand instrument keyword list in tag analyzer
6. **Stable Audio naming** — `stable_audio` (underscore) vs `stable-audio` (hyphen) mismatch. Fix: normalize model name lookup
7. **502 Mac Mini server** — diagnose and fix server crash on production Mac Mini
8. **Download all 12 models** — use model registry to download all converted models locally
9. **Automated model tests** — integration tests that verify each model can load and generate

## Testing Strategy

### Unit Tests (~35 tests)

**FlywheelManager (12 tests):**
- record_star saves WAV + metadata
- record_star/remove_star toggle
- check_threshold below/met/auto-off
- build_dataset blend ratios (80/20, 100/0, 0/100, 50/50)
- enrichment merged into descriptions
- kept gen LLM description + template fallback

**Versioning (8 tests):**
- create_version directory structure
- changelog.json content correctness (including parent_version: null for v1)
- active symlink created and updated
- revert_version changes symlink
- get_versions returns ordered list
- reset_cache clears kept dir and resets star count
- flat-layout LoRA auto-migration to v1/ + active symlink
- already-versioned LoRA not re-migrated (idempotent)

**FlywheelConfig (3 tests):**
- persistence across restarts
- defaults (threshold=10, blend=80, taste_refresh=5, auto_retrain=true)
- validation (threshold >= 1, blend 0-100, taste_refresh >= 1)

**Taste Integration (4 tests):**
- flywheel signals collected from kept gens
- taste refresh after N stars
- taste refresh after retrain
- enrichment tags flow into taste profile

**Server Endpoints (9 tests):**
- star POST endpoint (with in-memory audio)
- star POST endpoint (with client-uploaded WAV)
- unstar DELETE endpoint
- flywheel config GET/PUT
- versions list
- version changelog detail
- revert endpoint
- manual retrain endpoint
- reset endpoint
- starred field on job status

**Top Influences (2 tests):**
- compute_top_influences genre/mood/instrument percentages
- changelog reflects correct composition

### Integration Tests (2-3 tests, `@pytest.mark.integration`)
- Full flywheel cycle: star N gens -> auto-retrain -> new version + changelog
- Enrichment -> taste -> suggestions pipeline

All unit tests use temp directories and mocked training (no real weights needed).

## Future Work

**Approach C — Automated Quality Metrics (post-Phase 10):**
After retraining, the system would generate a test batch with the new adapter and compare spectral similarity to kept generations. If the new version scores worse, it flags it and keeps the old version active. This creates a self-correcting feedback loop. Deferred because it requires a new spectral comparison subsystem and enough adapter versions to be meaningful.

## Files Changed

### New Files
- `mlx_audiogen/lora/flywheel.py` — FlywheelManager + FlywheelConfig (~400-500 lines)
- `tests/test_flywheel.py` — unit tests (~500 lines)

### Modified Files
- `mlx_audiogen/lora/__init__.py` — export `FlywheelManager`, `FlywheelConfig`
- `mlx_audiogen/lora/trainer.py` — `list_available_loras()` updated to follow `active` symlink + auto-migrate flat layout
- `mlx_audiogen/cli/generate.py` — LoRA resolution follows `active` symlink (shared resolver function)
- `mlx_audiogen/library/taste/signals.py` — add `collect_flywheel_signals()`
- `mlx_audiogen/library/taste/engine.py` — add `update_flywheel_signals()`, call on refresh
- `mlx_audiogen/library/description_gen.py` — add `enrichment` parameter to `generate_description()`
- `mlx_audiogen/server/app.py` — 10 new endpoints (star POST/DELETE split) + starred field on jobs + flywheel config in settings
- `mlx_audiogen/shared/prompt_suggestions.py` — instrument keyword expansion (TR-909, 727, 606, Korg M1, etc.)
- `web/src/store/useStore.ts` — flywheel state (stars, config, versions)
- `web/src/api/client.ts` — 10 new API client functions
- `web/src/types/api.ts` — flywheel TypeScript types (properly typed, not `Record<string, any>`)
- `web/src/components/HistoryPanel.tsx` — star button
- `web/src/components/ParameterPanel.tsx` — collapsible sections
- `web/src/components/App.tsx` — resizable sidebar
- `web/src/components/SettingsPanel.tsx` (or new FlywheelSettings) — flywheel config UI + changelog viewer
- `web/src/components/LoRASelector.tsx` — version dropdown
