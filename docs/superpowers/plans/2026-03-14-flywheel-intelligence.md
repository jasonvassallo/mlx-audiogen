# Phase 9g-4: Flywheel Intelligence + Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect LoRA training, library scanning, and enrichment/taste into a self-improving flywheel with star ratings, auto-retrain, adapter versioning, and changelog. Also fix 9 reported UI/server bugs.

**Architecture:** New `flywheel.py` orchestrator module in `mlx_audiogen/lora/` connects existing systems. FlywheelManager tracks starred generations, triggers cumulative re-training at configurable threshold, creates versioned adapter snapshots with changelog metadata. Enrichment tags flow into training descriptions and taste signals. Frontend gains star buttons, flywheel settings, version selector, collapsible sidebar, and resizable panel.

**Tech Stack:** Python 3.11+ (FastAPI, MLX, Pydantic), React 19, TypeScript, Zustand 5, Tailwind CSS v4, Vite 6

**Spec:** `docs/superpowers/specs/2026-03-14-flywheel-intelligence-design.md`

---

## Chunk 1: Bug Fixes (Independent)

### Task 1: Fix tilde expansion in library path + file input

**Files:**
- Modify: `mlx_audiogen/server/app.py` — library source endpoints
- Modify: `web/src/components/LibraryPanel.tsx` — add file input for path selection

- [ ] **Step 1:** In `app.py`, find the library source add/update endpoints and add `os.path.expanduser()` to resolve `~` in paths before validation.
- [ ] **Step 2:** In `LibraryPanel.tsx`, add a hidden `<input type="file" accept=".xml" webkitdirectory={false}>` and a "Browse" button next to the path text input. On file select, read the file path (note: browsers only expose filename, not full path — use the `webkitRelativePath` or show instructions). Alternative: add a `/api/library/browse` endpoint that uses `tkinter.filedialog` server-side.
- [ ] **Step 3:** Test: add a library source with `~/Music/Media/Library.xml` and verify it resolves correctly.
- [ ] **Step 4:** Commit.

### Task 2: Fix stable_audio naming mismatch

**Files:**
- Modify: `mlx_audiogen/server/app.py` — model name normalization

- [ ] **Step 1:** Find where the server matches model names to weights directories. Add normalization: `model_name.replace("_", "-")` when looking up weights, so `stable_audio` finds `stable-audio` directory.
- [ ] **Step 2:** Test: verify `stable_audio` model name resolves to `stable-audio` weights.
- [ ] **Step 3:** Commit.

### Task 3: Fix instrument tag recognition

**Files:**
- Modify: `mlx_audiogen/shared/prompt_suggestions.py` — INSTRUMENTS dict

- [ ] **Step 1:** Add missing instruments to the INSTRUMENTS dict: `"drum machines": ["tr-909", "909", "tr-808", "808", "tr-707", "707", "tr-727", "727", "tr-606", "606", "cr-78", "linndrum", "linn drum", "drum machine", "mpc", "sp-404", "sp404", "maschine"]` and synths: `"synthesizers": [..., "korg m1", "m1", "korg m2", "korg wavestation", "roland juno", "juno-106", "juno 106", "jupiter-8", "jupiter 8", "sh-101", "tb-303", "303", "minimoog", "moog", "prophet-5", "prophet 5", "dx7", "dx-7", "nord lead", "virus ti", "ms-20", "arp 2600"]`.
- [ ] **Step 2:** Run existing tests to verify no regressions.
- [ ] **Step 3:** Commit.

### Task 4: Fix sidebar overflow + collapsible parameter sections

**Files:**
- Modify: `web/src/components/ParameterPanel.tsx` — collapsible sections with disclosure triangles
- Modify: `web/src/components/App.tsx` — resizable sidebar

- [ ] **Step 1:** In `ParameterPanel.tsx`, wrap generation parameters in a collapsible `<details>` element. Default: collapsed. Show summary line with current values (e.g., `temp 1.0 | top-k 250 | cfg 3.0`). Keep model selector and prompt always visible.
- [ ] **Step 2:** In `App.tsx`, add a drag-to-resize handle on the right edge of the sidebar. Store width in localStorage key `sidebar_width`. Min 280px, max 480px, default 320px. Use `onMouseDown` → `onMouseMove` → `onMouseUp` pattern.
- [ ] **Step 3:** Verify Settings tab no longer bleeds, parameter sliders are accessible via expand, Generate button is always visible.
- [ ] **Step 4:** Commit.

### Task 5: Diagnose and fix LLM template fallback

**Files:**
- Modify: `mlx_audiogen/server/app.py` — enhance endpoint / LLM loading

- [ ] **Step 1:** Check the `/api/enhance` endpoint and LLM model discovery. The Qwen3.5-9B-6bit model may not be auto-discovered if the HF cache path doesn't match expected patterns. Add debug logging to the model discovery path.
- [ ] **Step 2:** Check if `mlx-lm` is importable and if the model path resolves. Common issue: the model directory name in HF cache includes a hash suffix. Fix discovery to handle this.
- [ ] **Step 3:** Test: call `/api/enhance` with a prompt and verify LLM is used (not template fallback).
- [ ] **Step 4:** Commit.

### Task 6: Diagnose 502 / Mac Mini server

**Files:**
- Check: Mac Mini server logs at `~/Library/Logs/mlx-audiogen-server.log`

- [ ] **Step 1:** SSH to Mac Mini (`ssh macmini`) and check server status: `launchctl list | grep mlx-audiogen`, check logs. If server crashed, identify cause from log.
- [ ] **Step 2:** Restart if needed: `launchctl unload/load ~/Library/LaunchAgents/com.jasonvassallo.mlx-audiogen-server.plist`.
- [ ] **Step 3:** This is a runtime issue, not a code fix. Document findings.

---

## Chunk 2: Flywheel Backend (Python)

### Task 7: Create FlywheelConfig and FlywheelManager core

**Files:**
- Create: `mlx_audiogen/lora/flywheel.py`
- Test: `tests/test_flywheel.py`

- [ ] **Step 1:** Write tests for FlywheelConfig: defaults, validation, persistence to/from settings dict.
- [ ] **Step 2:** Write tests for FlywheelManager: record_star saves WAV + metadata, remove_star deletes them, check_threshold returns correct bool.
- [ ] **Step 3:** Run tests to verify they fail.
- [ ] **Step 4:** Implement `FlywheelConfig` dataclass and `FlywheelManager.__init__`, `record_star`, `remove_star`, `check_threshold`.
- [ ] **Step 5:** Run tests to verify they pass.
- [ ] **Step 6:** Commit.

### Task 8: Add adapter versioning to FlywheelManager

**Files:**
- Modify: `mlx_audiogen/lora/flywheel.py`
- Modify: `tests/test_flywheel.py`

- [ ] **Step 1:** Write tests for: create_version creates directory + changelog + symlink, get_versions returns ordered list, get_changelog returns correct data, revert_version changes symlink, reset_kept_generations clears kept dir.
- [ ] **Step 2:** Run tests to verify they fail.
- [ ] **Step 3:** Implement `create_version`, `get_versions`, `get_changelog`, `revert_version`, `reset_kept_generations`, `compute_top_influences`.
- [ ] **Step 4:** Run tests to verify they pass.
- [ ] **Step 5:** Commit.

### Task 9: Add flat-layout LoRA migration + update list_available_loras

**Files:**
- Modify: `mlx_audiogen/lora/trainer.py` — `list_available_loras()`, add `resolve_lora_dir()`
- Modify: `mlx_audiogen/cli/generate.py` — use `resolve_lora_dir()`
- Modify: `tests/test_flywheel.py` — migration tests

- [ ] **Step 1:** Write tests: flat-layout LoRA auto-migrates to v1/ + active symlink, already-versioned LoRA not re-migrated.
- [ ] **Step 2:** Run tests to verify they fail.
- [ ] **Step 3:** Add `resolve_lora_dir(lora_dir)` function that follows `active` symlink if present, or auto-migrates flat layout. Update `list_available_loras()` to use it. Update `cli/generate.py` to use it.
- [ ] **Step 4:** Run tests to verify they pass.
- [ ] **Step 5:** Commit.

### Task 10: Add build_dataset + start_retrain to FlywheelManager

**Files:**
- Modify: `mlx_audiogen/lora/flywheel.py`
- Modify: `tests/test_flywheel.py`

- [ ] **Step 1:** Write tests for build_dataset: blend ratios (80/20, 100/0, 0/100), enrichment merged into descriptions, kept gen descriptions.
- [ ] **Step 2:** Write tests for start_retrain: calls trainer with correct output_dir, creates version, skips if training lock held.
- [ ] **Step 3:** Run tests to verify they fail.
- [ ] **Step 4:** Implement `build_dataset` (merge library collection + kept gens at blend ratio, enrich descriptions) and `start_retrain` (build dataset, call trainer, create version, refresh taste).
- [ ] **Step 5:** Run tests to verify they pass.
- [ ] **Step 6:** Commit.

### Task 11: Add enrichment parameter to description_gen

**Files:**
- Modify: `mlx_audiogen/library/description_gen.py` — `generate_description()` gains `enrichment` param
- Modify: existing description_gen tests

- [ ] **Step 1:** Write test: `generate_description(track, enrichment={"tags": ["atmospheric", "reverb"]})` includes enrichment tags in output.
- [ ] **Step 2:** Run test to verify it fails.
- [ ] **Step 3:** Add optional `enrichment: dict | None = None` parameter. If present, append enrichment tags to description.
- [ ] **Step 4:** Run test to verify it passes.
- [ ] **Step 5:** Commit.

### Task 12: Add collect_flywheel_signals + update TasteEngine

**Files:**
- Modify: `mlx_audiogen/library/taste/signals.py` — add `collect_flywheel_signals()`
- Modify: `mlx_audiogen/library/taste/engine.py` — add `update_flywheel_signals()`, call on refresh
- Modify: existing taste tests

- [ ] **Step 1:** Write tests: collect_flywheel_signals extracts genres/moods/instruments from kept gen prompts, TasteEngine.update_flywheel_signals merges into profile, taste refresh triggers flywheel signal collection.
- [ ] **Step 2:** Run tests to verify they fail.
- [ ] **Step 3:** Implement `collect_flywheel_signals(adapter_name, kept_dir)` — reads gen_*.json, extracts keywords using GENRES/MOODS/INSTRUMENTS lists, returns signal dict. Add `update_flywheel_signals()` to TasteEngine.
- [ ] **Step 4:** Run tests to verify they pass.
- [ ] **Step 5:** Commit.

### Task 13: Update lora/__init__.py exports

**Files:**
- Modify: `mlx_audiogen/lora/__init__.py`

- [ ] **Step 1:** Add exports: `FlywheelManager`, `FlywheelConfig` from `flywheel.py`, `resolve_lora_dir` from `trainer.py`.
- [ ] **Step 2:** Commit.

### Task 14: Add flywheel server endpoints

**Files:**
- Modify: `mlx_audiogen/server/app.py` — 10 new endpoints + starred field on jobs
- Modify: existing server tests or create new test file

- [ ] **Step 1:** Write tests for: POST /api/star/{id} (star with audio), DELETE /api/star/{id} (unstar), GET /api/flywheel/config, PUT /api/flywheel/config, GET /api/loras/{name}/versions, GET /api/loras/{name}/versions/{v}, PUT /api/loras/{name}/active/{v}, POST /api/flywheel/retrain/{name}, POST /api/flywheel/reset/{name}, GET /api/flywheel/status.
- [ ] **Step 2:** Run tests to verify they fail.
- [ ] **Step 3:** Implement all endpoints. Add `starred: bool` field to job status/list responses. Add FlywheelManager instance as server-level state. Add Pydantic models (FlywheelConfigModel, StarResponse, VersionSummary, VersionChangelog, DatasetComposition, TagInfluence, TopInfluences, NewSinceParent, TrainingStats).
- [ ] **Step 4:** Run tests to verify they pass.
- [ ] **Step 5:** Run full test suite: `uv run pytest`
- [ ] **Step 6:** Commit.

---

## Chunk 3: Flywheel Frontend (TypeScript)

### Task 15: Add flywheel TypeScript types

**Files:**
- Modify: `web/src/types/api.ts`

- [ ] **Step 1:** Add interfaces: `FlywheelConfig`, `StarResponse`, `FlywheelStatus`, `VersionSummary`, `VersionChangelog`, `DatasetComposition`, `TagInfluence`, `TopInfluences`, `NewSinceParent`, `TrainingStats`. Extend `JobInfo` with `starred?: boolean`. Extend `LoRAInfo` with `active_version?: number`, `total_versions?: number`, `stars_since_train?: number`.
- [ ] **Step 2:** Commit.

### Task 16: Add flywheel API client functions

**Files:**
- Modify: `web/src/api/client.ts`

- [ ] **Step 1:** Add functions: `starGeneration(jobId, audioBlob?)`, `unstarGeneration(jobId)`, `getFlywheelConfig()`, `updateFlywheelConfig(config)`, `getFlywheelStatus()`, `getLoraVersions(name)`, `getLoraChangelog(name, version)`, `setActiveLoraVersion(name, version)`, `triggerRetrain(name)`, `resetKeptGenerations(name)`.
- [ ] **Step 2:** Commit.

### Task 17: Add flywheel state to Zustand store

**Files:**
- Modify: `web/src/store/useStore.ts`

- [ ] **Step 1:** Add to AppState: `flywheelConfig`, `flywheelStatus`, `loraVersions`, `loadFlywheelConfig()`, `updateFlywheelConfig()`, `starGeneration()`, `unstarGeneration()`, `loadLoraVersions()`, `setActiveVersion()`, `triggerRetrain()`, `resetKeptGenerations()`.
- [ ] **Step 2:** Initialize flywheel config on store creation.
- [ ] **Step 3:** Commit.

### Task 18: Add star button to HistoryPanel

**Files:**
- Modify: `web/src/components/HistoryPanel.tsx`

- [ ] **Step 1:** Add a ★ button next to each completed job entry. Yellow filled when `job.starred`, outline when not. On click: if not starred, POST with audio blob; if starred, DELETE.
- [ ] **Step 2:** Eagerly download audio blob URL on generation complete (existing pattern from stems).
- [ ] **Step 3:** Verify star state persists across page reloads.
- [ ] **Step 4:** Commit.

### Task 19: Add version selector to LoRASelector

**Files:**
- Modify: `web/src/components/LoRASelector.tsx` (or wherever LoRA dropdown lives)

- [ ] **Step 1:** Show `{name} (v{N})` in dropdown. Add small version sub-dropdown or expandable list showing all versions with one-line changelog summary. Active version gets checkmark. Selecting a version calls `setActiveLoraVersion()`.
- [ ] **Step 2:** Commit.

### Task 20: Add Flywheel Settings section

**Files:**
- Modify: `web/src/components/SettingsPanel.tsx` or create `web/src/components/FlywheelSettings.tsx`

- [ ] **Step 1:** Add "Flywheel Intelligence" section to Settings tab with:
  - Auto-retrain toggle (ON/OFF)
  - Retrain threshold input (number, default 10)
  - Taste refresh interval input (number, default 5)
  - Dataset blend slider (0-100%, default 80% library, shows "80% Library / 20% Generations")
  - [Re-train Now] button
  - [Reset Cache] button
  - Adapter changelog viewer: expandable list of versions with summary
- [ ] **Step 2:** Connect all controls to `updateFlywheelConfig()` store action.
- [ ] **Step 3:** Commit.

### Task 21: Build web UI and verify

- [ ] **Step 1:** Run `cd web && npm run build` — fix any TypeScript errors.
- [ ] **Step 2:** Commit.

---

## Chunk 4: Model Downloads + Integration Tests + Final QA

### Task 22: Download all 12 models

**Files:**
- Use: `mlx_audiogen/shared/model_registry.py`

- [ ] **Step 1:** Run model downloads for all 12 models using the auto-download mechanism. Check which models are already in `~/.mlx-audiogen/models/` or `./converted/`.
- [ ] **Step 2:** Download missing models. This may take a while (~30GB total).
- [ ] **Step 3:** Verify all 12 models are accessible.

### Task 23: Add model load integration tests

**Files:**
- Create: `tests/test_model_load.py`

- [ ] **Step 1:** Write integration tests (marked `@pytest.mark.integration`) that verify each model can be loaded via `from_pretrained()`. One test per model variant. Tests should: resolve weights, create pipeline, verify config loaded correctly. Do NOT generate audio (too slow for CI).
- [ ] **Step 2:** Run: `uv run pytest tests/test_model_load.py -m integration -v`
- [ ] **Step 3:** Commit.

### Task 24: Full QA pass

- [ ] **Step 1:** Run full check suite: `uv run ruff format . && uv run ruff check . && uv run mypy mlx_audiogen/ && uv run bandit -r mlx_audiogen/ -c pyproject.toml && uv run pip-audit && uv run pytest && cd web && npm run build`
- [ ] **Step 2:** Fix any issues found.
- [ ] **Step 3:** Update CLAUDE.md with new endpoints, flywheel architecture, test count.
- [ ] **Step 4:** Update memory files (MEMORY.md, phase plan).
- [ ] **Step 5:** Commit and push to main.
