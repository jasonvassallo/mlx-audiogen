# Phase 7a: Web UI Catch-Up — Design Spec

Date: 2026-03-10

## Goal

Add missing Web UI components so the React app matches the server's full API surface. Four feature areas, a server-side fix, plus a Support link.

## Architecture Decision: Tabbed Left Panel

The left panel gets a two-tab layout:
- **Generate** — all current controls (ModelSelector, PromptInput, ParameterPanel, GenerateButton, Settings, AudioDeviceSelector)
- **Suggest** — prompt suggestions + presets (new)

Implementation: `useState<"generate" | "suggest">` tab selector. No routing library.

### New Files
- `web/src/components/TabBar.tsx` — reusable tab header
- `web/src/components/SuggestPanel.tsx` — container for Suggest tab

### Modified Files
- `web/src/App.tsx` — wrap left panel content in tab system

## Feature 1: Prompt Suggestions (Suggest Tab — Top Section)

**Behavior:**
- Reads current prompt from Zustand store (`params.prompt`)
- "Analyze" button calls existing `suggestPrompts(prompt)` → server `POST /api/suggest`
- Displays analysis: detected genres/moods/instruments as colored tags, "missing" items as dimmed tags
- 3-4 suggestion cards displayed as plain text (no color highlighting — descriptors aren't tagged by category in the API response)
- Each card: "Use" button (replaces `params.prompt`, switches to Generate tab) + "Copy" button (clipboard)
- Auto-analyzes when Suggest tab opened if prompt is non-empty; caches result until prompt changes (compare `params.prompt` against a `lastAnalyzedPrompt` ref)

**Store additions:**
- `suggestions: PromptAnalysis | null`
- `suggestionsLoading: boolean`
- `fetchSuggestions(): Promise<void>`

**No new API client code** — `suggestPrompts()` and `PromptAnalysis` type already exist.

**Server fix:** `server/app.py` `suggest_prompts()` must forward `req.count` to `analyze_prompt()`. Currently it ignores the count parameter. Fix: `analyze_prompt(req.prompt, count=req.count)` and update `prompt_suggestions.py` `analyze_prompt` to accept and forward `count`.

## Feature 2: Presets (Suggest Tab — Bottom Section)

**Behavior:**
- "Save Current" button → inline text input for name → `POST /api/presets/{name}` with current params
- Preset name validated client-side: alphanumeric + hyphens + underscores only, 1-64 chars, whitespace trimmed
- Preset list fetched on tab open via `GET /api/presets`
- Each row: preset name, model type badge, truncated prompt preview
- Click preset → `GET /api/presets/{name}` → loads params into store, switches to Generate tab
- `applyPreset` validates loaded data before applying: checks `model` is in available models list, numeric params are within valid ranges; falls back to current defaults for any invalid fields
- Empty state: "No presets saved yet."

**New API client functions:**
- `fetchPresets()` → `GET /api/presets` returns `PresetInfo[]`
- `savePreset(name: string, params: GenerateRequest)` → `POST /api/presets/{name}`
- `loadPreset(name: string)` → `GET /api/presets/{name}` returns `GenerateRequest`

**New type:**
- `PresetInfo { name: string; filename: string; prompt: string; model: string }`

**Store additions:**
- `presets: PresetInfo[]`
- `presetsLoading: boolean`
- `loadPresets()`, `savePreset(name: string)`, `applyPreset(name: string)`

## Feature 3: Enhanced History Entries (MIDI + Stems + Metadata)

### Metadata Header
Each history card shows below the prompt: `musicgen · 5.0s · 32kHz mono`
- Source: `job.model`, `job.seconds`, `job.sample_rate`
- Mono/stereo inferred from model name containing "stereo"

### MIDI Download Button + Output Mode Selector
- Add `output_mode` dropdown to `ParameterPanel` (options: "audio", "midi", "both"; default: "audio")
- MIDI download button always visible on every history entry
- If `job.has_midi === true`: click downloads via `getMidiUrl(id)` (already in client.ts)
- If `job.has_midi === false`: greyed out, title tooltip "Set output mode to 'midi' or 'both' before generating"

### Stem Separation
- "Stems" button on each entry
- Click calls `separateStems(jobId)` (already in client.ts) → expands inline section
- Server returns dynamic stem names: Demucs gives `drums/bass/vocals/other`, FFT fallback gives `bass/mid/high`
- UI renders whatever stems the server returns — uses `Object.entries(response.stems)` to iterate
- Color mapping: known stems get assigned colors (drums=green, bass=red, vocals=purple, other=blue, mid=amber, high=cyan); unknown stems get a default grey
- Each stem: play button + download link via `getAudioUrl(stemJobId)`
- On successful separation, immediately fetch each stem's audio blob and cache in store (same pattern as main audio download) to avoid relying on server's 5-minute job cleanup
- Spinner while separating

**Type update:**
- `JobInfo` in `api.ts`: add `has_midi?: boolean` (optional — older jobs from before the feature won't have it; treat `undefined` as `false`)

**Store additions:**
- `stemResults: Record<string, Record<string, string>>` — jobId → {stemName → stemJobId}
- `stemsLoading: Record<string, boolean>`
- `requestStemSeparation(jobId: string): Promise<void>`

## Feature 4: Support Link

- Small heart icon + "Support" text link in `Header.tsx`, right-aligned
- Links to `https://paypal.me/jasonvassallo`
- Opens in new tab (`target="_blank" rel="noopener noreferrer"`)
- Subtle styling — does not interrupt creative workflow

## File Change Summary

| File | Change |
|------|--------|
| `web/src/components/TabBar.tsx` | **New** — reusable tab header |
| `web/src/components/SuggestPanel.tsx` | **New** — suggestions + presets container |
| `web/src/App.tsx` | Add tab system wrapping left panel |
| `web/src/components/Header.tsx` | Add Support link |
| `web/src/components/HistoryPanel.tsx` | Add metadata, MIDI button, stems section |
| `web/src/components/ParameterPanel.tsx` | Add output_mode dropdown |
| `web/src/store/useStore.ts` | Add suggestions, presets, stems state + actions |
| `web/src/api/client.ts` | Add preset API functions |
| `web/src/types/api.ts` | Add `has_midi` to JobInfo (optional), add `PresetInfo` |
| `mlx_audiogen/server/app.py` | Forward `count` param to `analyze_prompt()` |
| `mlx_audiogen/shared/prompt_suggestions.py` | Accept `count` param in `analyze_prompt()` |

## Out of Scope
- LLM-powered suggestions (Phase 7b)
- Cloud rendering UI (Phase 7d)
- Demucs quality testing (Phase 7c)
