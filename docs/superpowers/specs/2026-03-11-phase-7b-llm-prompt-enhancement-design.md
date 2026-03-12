# Phase 7b: LLM Prompt Enhancement + Prompt Memory + Tag Autocomplete

**Date:** 2026-03-11
**Status:** Design approved, pending implementation
**Revision:** 2 (addresses 21 spec review issues)

## Overview

Replace the template-based `prompt_suggestions.py` with local LLM inference via `mlx-lm`, add prompt history memory for style learning, inline tag autocomplete for the prompt input, a Settings tab with LLM model selection, and prompt memory management.

## Goals

1. **LLM-powered prompt enhancement** — every prompt optionally refined by a local MLX LLM before generation (with user approval)
2. **Prompt memory** — history of past prompts + auto-derived style profile, injected into LLM context for personalized suggestions
3. **Inline tag autocomplete** — color-coded descriptive tags (genre, mood, instrument, era, production) appear as the user types
4. **Settings tab** — LLM model dropdown, AI Enhance toggle, prompt memory management (export/clear/import)
5. **Future: Theme selector** — placeholder in Settings for user-selectable UI color themes (deferred to later phase)

## Architecture

### LLM Integration

**Dependency:** `mlx-lm` added as optional extra (`uv sync --extra llm`)

**Model discovery** scans three paths (follows symlinks, resolves HF snapshot dirs):
- `~/.cache/huggingface/hub/` — XDG/Linux HF cache convention (resolves `models--org--name/snapshots/<hash>/` structure)
- `~/Library/Caches/huggingface/hub/` — macOS native HF cache (may use flat `org/name/` structure from LM Studio downloads, or HF `models--org--name/snapshots/<hash>/`)
- `~/.lmstudio/hub/models/` — LM Studio (often symlinks into HF cache)

A valid MLX LLM model directory must contain:
1. `config.json` with a recognized model architecture (checked against `mlx-lm` supported list)
2. At least one `*.safetensors` file
3. A tokenizer file (`tokenizer_config.json` or `tokenizer.json`) — this distinguishes LLMs from non-LLM models like EnCodec or MERT

The scanner runs synchronously during `main()` after argument parsing (before `uvicorn.run()`), same pattern as audio model registration. Each discovered model is assigned an opaque identifier (e.g., `mlx-community/Qwen3.5-9B-6bit`) derived from the directory name. The server maintains an internal `Dict[str, Path]` mapping identifiers to filesystem paths — **paths are never exposed to clients**.

**Default model:** `mlx-community/Qwen3.5-9B-6bit` (7.7 GB)

**Server integration:**
- `--llm-model` CLI flag sets the default model by identifier (overridable from UI)
- LLM pipeline loaded lazily on first `/api/enhance` call via `mlx_lm.load()`
- Pipeline cached in server memory (similar to audio gen pipeline cache)
- Model can be switched at runtime via `POST /api/llm/select` (accepts model identifier, not path)
- If a `/api/enhance` call is in-flight when `/api/llm/select` is called, the select returns `409 Conflict` with message "LLM is currently busy"
- If no LLM model available, `/api/enhance` falls back to the existing template engine
- **Memory management:** LLM is auto-unloaded after 5 minutes of idle time (configurable via `--llm-idle-timeout`). This reclaims ~7-8 GB of RAM for audio generation. Re-loaded lazily on next enhance call.

**LLM inference timeout:** 30 seconds. If exceeded, falls back to template engine and returns `used_llm: false` with a warning in the response.

**RAM budget consideration:** On 24GB M4 Pro, expected usage:
- OS + apps: ~8 GB
- Audio gen model (e.g., MusicGen-small): ~1.5 GB
- LLM (Qwen3.5-9B-6bit): ~7.7 GB
- Headroom: ~6.8 GB
- The 5-minute idle unload ensures LLM RAM is reclaimed when not actively enhancing prompts.

**LLM system prompt:**
```
You are a music prompt engineer for AI audio generation models (MusicGen, Stable Audio).
Given a user's prompt, enhance it with rich musical descriptors including genre, mood,
instrumentation, tempo, production style, and era details.
Keep the user's core intent and artistic direction. Output ONLY the enhanced prompt
as a single line, nothing else. Do not add explanations or formatting.

{memory_context}
```

Where `{memory_context}` is dynamically injected:
```
Recent prompts from this user (newest first):
- dark ambient pad, warm analog, slow tempo
- synthwave arpeggio, 80s, neon
- melancholic piano, reverb, minimal
... (up to 50 most recent)

User's style profile: Prefers ambient and electronic genres, dark/atmospheric moods,
synth and pad instruments. Average duration: 10 seconds. 42 total generations.

Honor the user's style preferences but introduce creative variety and fresh ideas.
```

### Generation Flow

**AI Enhance toggle ON (default):**
```
User types prompt
  -> clicks Generate
  -> POST /api/enhance {prompt, include_memory: true}
  -> Server: load LLM (lazy) -> build system prompt with memory context (50 recent prompts)
  -> LLM generates enhanced prompt (30s timeout, template fallback on failure)
  -> Response: {original, enhanced, analysis_tags, used_llm}
  -> UI shows inline preview card below prompt input:
    +---------------------------------------------+
    | Enhanced: "melancholic piano, warm analog,   |
    | slow tempo, reverb-drenched, intimate, solo  |
    | performance, minor key, sparse arrangement"  |
    |                                              |
    | [Accept & Generate]  [Edit]  [Use Original]  |
    +---------------------------------------------+
  -> If LLM failed (used_llm: false), show warning: "Using template (LLM unavailable)"
  -> User clicks "Accept & Generate" (or edits, or skips)
  -> POST /api/generate with chosen prompt
  -> On success -> server auto-saves prompt to memory (inside generate handler)
```

**AI Enhance toggle OFF:**
```
User types prompt
  -> clicks Generate
  -> POST /api/generate directly
  -> On success -> server auto-saves prompt to memory
```

**Fallback** (no LLM model available or LLM error):
```
POST /api/enhance falls back to template engine (existing suggest_refinements)
-> Returns template-enhanced prompt with used_llm: false
-> UI shows warning toast: "LLM enhancement unavailable, using template suggestions"
-> UI flow is otherwise identical (user still sees preview and approves)
```

**Memory saving:** The server auto-saves the prompt inside the `POST /api/generate` handler after successful generation. Both the original prompt and the enhanced prompt (if used) are stored. No separate `/api/memory/save` endpoint needed — this avoids an extra round-trip and ensures history is only populated with prompts that actually produced audio.

### Inline Tag Autocomplete

**Tag database** — extended from existing `prompt_suggestions.py` lists:

| Category | Color | Count | Examples |
|----------|-------|-------|---------|
| Genre | amber/orange | ~40 | ambient, synthwave, hip hop, jazz, lo-fi, drum and bass, house |
| Mood | emerald/green | ~25 | dark, melancholic, upbeat, dreamy, euphoric, aggressive, peaceful |
| Instrument | sky/blue | ~35 | 808, 909, piano, strings, synth pad, guitar, bass, choir, organ |
| Era/Style | violet/purple | ~15 | 80s, 90s, vintage, modern, retro, futuristic, classic, Y2K |
| Production | rose/pink | ~20 | warm analog, crisp digital, lo-fi tape, reverb-drenched, distorted |

**Behavior:**
- Triggers after 2+ characters typed (debounced 150ms)
- Matches against tag names (case-insensitive substring match)
- Shows max 8 suggestions in a dropdown **below the textarea element** (not cursor-positioned — avoids the complexity of pixel-accurate cursor tracking in a textarea)
- Each suggestion shows: colored dot + tag name + category label (dimmed)
- Click or Tab inserts the tag at cursor position with a comma separator
- Escape or clicking outside dismisses the dropdown
- **Keyboard priority:** If autocomplete dropdown is open, Tab inserts the selected tag and Enter dismisses the dropdown. Cmd+Enter only triggers generation when autocomplete is closed.
- Tags served from `GET /api/tags` (static, cached client-side on first fetch)

### Prompt Memory

**Storage:** `~/.mlx-audiogen/prompt_memory.json`

**Max history size:** 2000 entries. When exceeded, oldest entries are evicted first. This provides months of history for most users while keeping file size manageable (~1-2 MB) and profile derivation fast.

```json
{
  "history": [
    {
      "prompt": "dark ambient pad",
      "enhanced_prompt": "dark ambient pad, warm analog, slow tempo, reverb",
      "model": "musicgen",
      "params": {"seconds": 10, "temperature": 0.9, "top_k": 250},
      "timestamp": "2026-03-11T15:30:00Z"
    }
  ],
  "style_profile": {
    "top_genres": ["ambient", "electronic"],
    "top_moods": ["dark", "atmospheric"],
    "top_instruments": ["synth", "pad"],
    "preferred_duration": 10,
    "generation_count": 42
  }
}
```

**Style profile derivation:**
- Rebuilt on every write (append to history -> recompute profile)
- `top_genres/moods/instruments`: frequency count across all history prompts using `analyze_prompt()`, top 5 each
- `preferred_duration`: median of all `params.seconds` values
- `generation_count`: length of history array

**LLM context injection:**
- Number of history entries injected into LLM system prompt is **user-configurable** via a "History Context" slider in Settings
- **Default: 50** prompts (newest first, prompt text only)
- **Range: 0 to unlimited** — slider goes 0–100 with a text input for typing any number (including values > 100). Value of 0 means "all history" (no limit).
- Rationale for 50 default: Qwen3.5-9B has 32K context window. 50 prompts is approx 1500-2500 tokens — well within budget, gives rich style signal without diminishing returns.
- **Warning threshold:** If the user sets a value that would exceed ~8000 tokens (~160+ prompts), show a warning: "Large history context may slow down enhancement and reduce output quality"
- Setting persisted to server-side `settings.json` as `history_context_count: int` (0 = unlimited)
- Full style profile summary always included (regardless of history count setting)
- System prompt instructs LLM to honor preferences but add creative variety

**Memory import validation:**
- File size: max 10 MB
- Max history entries: 5000 (excess silently truncated to most recent 2000)
- Required top-level keys: `history` (array), `style_profile` (object)
- Required fields per history entry: `prompt` (string, max 5000 chars), `timestamp` (string)
- Optional fields per entry: `enhanced_prompt`, `model`, `params`
- Invalid entries (missing `prompt` or `timestamp`) are silently skipped
- Style profile is always re-derived from imported history (ignoring any imported profile)

### Settings Tab and UI Layout

**Layout change:** The left panel gets a third tab: **Generate | Suggest | Settings**

The `activeTab` type in the Zustand store expands to `"generate" | "suggest" | "settings"`.

**Existing SettingsPanel integration:** The current `SettingsPanel.tsx` component (master BPM, pitch mode, history retention, audio device selector) is rendered at the bottom of the left panel in a `border-t` section that is **always visible** regardless of which tab is active. The new Settings tab adds **LLM-specific and memory-specific settings only** — it does not move the existing always-visible settings.

**New LLMSettingsPanel.tsx** (rendered inside the Settings tab content area):

1. **LLM Model**
   - Dropdown of auto-detected MLX models (scanned on server start + refreshable)
   - Shows model identifier + size (e.g., "Qwen3.5-9B-6bit (7.7 GB)")
   - Current selection persisted to server-side `~/.mlx-audiogen/settings.json`
   - "Refresh" button to rescan for newly downloaded models
   - Status indicator: loaded / not loaded / loading / error

2. **AI Enhance**
   - Toggle switch (default: ON)
   - Also shown as a compact toggle near the Generate button for quick access
   - State persisted to server-side `settings.json` (shared across all browser sessions)

3. **History Context**
   - Slider (0–100) + editable text input for typing any number
   - Default: 50. Value of 0 = "All history" (no limit)
   - Label shows current value: "50 recent prompts" or "All history"
   - Warning icon + tooltip at 160+: "Large context may slow enhancement"
   - State persisted to server-side `settings.json`

4. **Prompt Memory**
   - Style profile summary display (top genres, moods, instruments as colored tags)
   - Generation count + history size
   - Buttons: **Export** (download JSON), **Clear** (with confirmation dialog), **Import** (file upload)

4. **Theme** (future phase — placeholder only)
   - Disabled dropdown showing "Dark (Default)"
   - Tooltip: "More themes coming soon"

**Settings persistence strategy:**
- **Server-side** (`~/.mlx-audiogen/settings.json`): LLM model selection, AI Enhance toggle, history context count — shared across all browser sessions connecting to this server
- **Client-side** (IndexedDB, existing): master BPM, pitch mode, history retention, audio device — per-browser settings that don't need server persistence

### API Endpoints (New)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/enhance` | Enhance a prompt via LLM (or template fallback). Body: `{prompt, include_memory?}`. Returns: `{original, enhanced, analysis_tags, used_llm, warning?}`. Timeout: 30s. |
| `GET` | `/api/tags` | Return full tag database grouped by category |
| `GET` | `/api/llm/models` | List auto-detected MLX models. Returns: `[{id, name, size_gb, source}]` — **no filesystem paths** |
| `POST` | `/api/llm/select` | Set active LLM model. Body: `{model_id}`. Returns 409 if LLM busy. Reloads pipeline. |
| `GET` | `/api/llm/status` | Current LLM status: `{model_id, loaded, idle_seconds, memory_mb}` |
| `GET` | `/api/memory` | Return current prompt_memory.json contents |
| `DELETE` | `/api/memory` | Clear all history + reset style profile |
| `POST` | `/api/memory/import` | Upload/restore a prompt_memory.json file (max 10 MB, validated schema) |
| `GET` | `/api/memory/export` | Download prompt_memory.json as file attachment |
| `GET` | `/api/settings` | Return current settings.json |
| `POST` | `/api/settings` | Update settings (llm_model, ai_enhance) |

**Existing endpoints unchanged:** `/api/suggest` remains for the Suggest tab (template-based multi-suggestion). `/api/enhance` is the new endpoint for the Generate tab's single-prompt LLM enhancement flow. They serve different purposes: suggest returns multiple template-based options for browsing; enhance returns one LLM-refined result for immediate use.

### Pydantic Models (New)

```python
class EnhanceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    include_memory: bool = Field(default=True)

class AnalysisTags(BaseModel):
    genres: list[str]
    moods: list[str]
    instruments: list[str]
    missing: list[str]

class EnhanceResponse(BaseModel):
    original: str
    enhanced: str
    analysis_tags: AnalysisTags
    used_llm: bool  # True if LLM was used, False if template fallback
    warning: str | None = None  # e.g., "LLM timed out, used template"

class LLMModelInfo(BaseModel):
    id: str  # opaque identifier, e.g., "mlx-community/Qwen3.5-9B-6bit"
    name: str  # display name, e.g., "Qwen3.5-9B-6bit"
    size_gb: float  # approximate size in GB
    source: str  # "huggingface" | "lmstudio"
    # NOTE: no filesystem path exposed to client

class LLMSelectRequest(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=200, pattern=r"^[a-zA-Z0-9/_.-]+$")
    # Must match an id from GET /api/llm/models

class SettingsData(BaseModel):
    llm_model: str | None = None  # model identifier (not path)
    ai_enhance: bool = True
    history_context_count: int = Field(default=50, ge=0)  # 0 = unlimited
```

## Files to Modify

### Backend (Python)

| File | Changes |
|------|---------|
| `mlx_audiogen/shared/prompt_suggestions.py` | Add `PromptMemory` class (load/save/derive profile, max 2000 entries with eviction), `enhance_with_llm()` with 30s timeout, `TAG_DATABASE` dict with all categories, `discover_mlx_models()` scanner (resolves HF snapshots + symlinks) |
| `mlx_audiogen/server/app.py` | Add `/api/enhance`, `/api/tags`, `/api/llm/*`, `/api/memory/*`, `/api/settings` endpoints. Lazy LLM pipeline loading with 5-min idle unload. Auto-save prompt to memory inside generate handler. 409 on model switch during inference. |
| `pyproject.toml` | Add `llm` optional extra with `mlx-lm>=0.22.0` dependency |

### Frontend (TypeScript/React)

| File | Changes |
|------|---------|
| `web/src/api/client.ts` | Add typed wrappers for all new endpoints (enhance, tags, llm/*, memory/*, settings) |
| `web/src/types/api.ts` | Add `EnhanceResponse`, `AnalysisTags`, `LLMModelInfo`, `SettingsData`, `TagDatabase`, `PromptMemory` types |
| `web/src/store/useStore.ts` | Add settings state (server-synced), enhance flow state, tag database cache, memory state. Expand `activeTab` type to include `"settings"`. |
| `web/src/components/App.tsx` | Add Settings tab to TabBar, render LLMSettingsPanel in settings tab content area |
| **NEW** `web/src/components/LLMSettingsPanel.tsx` | LLM model dropdown, AI Enhance toggle, prompt memory management |
| **NEW** `web/src/components/TagAutocomplete.tsx` | Inline autocomplete dropdown below textarea |
| **NEW** `web/src/components/EnhancePreview.tsx` | Inline card showing enhanced prompt with Accept/Edit/Use Original buttons, warning display |
| `web/src/components/PromptInput.tsx` | Integrate TagAutocomplete component |
| `web/src/components/GenerateButton.tsx` | Integrate AI Enhance toggle + enhance flow before generation |
| `web/src/components/SettingsPanel.tsx` | **No changes** — existing always-visible bottom panel stays as-is |

## Security Considerations

- **No filesystem path exposure:** `GET /api/llm/models` returns opaque identifiers only. `POST /api/llm/select` accepts identifiers that must match a pre-scanned model. The server maps identifiers to paths internally.
- **Model selection restricted:** Only models returned by the discovery scanner can be selected. Arbitrary filesystem paths are rejected.
- **Prompt memory file:** Written with `json.dump()` to `~/.mlx-audiogen/` only (no arbitrary paths)
- **Memory import validation:** Max 10 MB file size, max 5000 history entries (truncated to 2000), required schema keys, individual prompt max 5000 chars. Invalid entries silently skipped. Style profile always re-derived.
- **LLM output sanitization:** Server truncates to 2000 characters. Rendered as plain text in React JSX (standard React text interpolation, never raw HTML injection). Output is never used to construct URLs, file paths, or executable code.
- **Settings file:** Only accepts known keys (`llm_model`, `ai_enhance`); unknown keys silently dropped
- **Tag database:** Static server-side list; no user-submitted tags (prevents XSS in autocomplete)

## Testing Plan

- Unit tests for `PromptMemory` (load, save, derive profile, history append, eviction at 2000 entries)
- Unit tests for `discover_mlx_models()` (mock filesystem: flat dirs, HF snapshot dirs, symlinks, non-LLM models filtered)
- Unit tests for `enhance_with_llm()` (mock mlx-lm, test fallback on error, test 30s timeout)
- Unit tests for tag database completeness and category assignment
- Unit tests for memory import validation (valid file, oversized, malformed, missing fields, excess entries)
- Integration test for `/api/enhance` endpoint (with and without LLM, timeout scenario)
- Integration test for `/api/memory/*` endpoints (CRUD operations, import validation)
- Integration test for `/api/settings` endpoint
- Integration test for `/api/llm/select` 409 conflict behavior
- Web UI: manual verification of autocomplete, enhance preview, settings panel, keyboard interactions

## Future Work (Not in This Phase)

- **Theme selector:** User-selectable UI color themes in Settings tab (Phase 7e or later)
- **Prompt templates library:** Pre-built prompt templates for common genres/moods
- **Multi-model enhance:** Use different LLMs for different enhancement styles
- **Streaming LLM output:** Stream enhanced prompt token-by-token for perceived speed
