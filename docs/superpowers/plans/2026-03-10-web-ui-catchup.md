# Phase 7a: Web UI Catch-Up — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add prompt suggestions, presets, MIDI output, stem separation, and a support link to the React web UI so it matches the server's full API surface.

**Architecture:** Tabbed left panel (Generate / Suggest). New Zustand state slices for suggestions, presets, and stems. Server-side fix to forward suggestion count parameter. All new UI components use existing Tailwind CSS v4 DAW theme tokens.

**Tech Stack:** React 19, TypeScript, Zustand 5, Tailwind CSS v4, Vite 6. Server: FastAPI + Python.

**Spec:** `docs/superpowers/specs/2026-03-10-web-ui-catchup-design.md`

---

## File Structure

| File | Responsibility | Status |
|------|---------------|--------|
| `mlx_audiogen/shared/prompt_suggestions.py` | Accept `count` param in `analyze_prompt()` | Modify |
| `mlx_audiogen/server/app.py` | Forward `count` to `analyze_prompt()` | Modify |
| `web/src/types/api.ts` | Add `has_midi?` to `JobInfo`, add `PresetInfo` | Modify |
| `web/src/api/client.ts` | Add `fetchPresets`, `savePreset`, `loadPreset` | Modify |
| `web/src/store/useStore.ts` | Add suggestions/presets/stems/output_mode state | Modify |
| `web/src/components/TabBar.tsx` | Reusable tab header component | Create |
| `web/src/components/SuggestPanel.tsx` | Suggestions + presets container | Create |
| `web/src/App.tsx` | Wrap left panel in tab system | Modify |
| `web/src/components/ParameterPanel.tsx` | Add output_mode dropdown | Modify |
| `web/src/components/HistoryPanel.tsx` | Add metadata, MIDI button, stems section | Modify |
| `web/src/components/Header.tsx` | Add support link | Modify |

---

## Chunk 1: Server-Side Fix + Types + API Client

### Task 1: Fix server count parameter forwarding

**Files:**
- Modify: `mlx_audiogen/shared/prompt_suggestions.py:190`
- Modify: `mlx_audiogen/server/app.py:316-321`

- [ ] **Step 1: Update `analyze_prompt` signature to accept `count`**

In `mlx_audiogen/shared/prompt_suggestions.py`, change the function signature at line 190 and the `suggest_refinements` call at line 219:

```python
# Line 190: change signature
def analyze_prompt(prompt: str, count: int = 3) -> dict:

# Line 219: forward count
        "suggestions": suggest_refinements(prompt, count=count),
```

- [ ] **Step 2: Update server endpoint to forward `count`**

In `mlx_audiogen/server/app.py`, change line 320:

```python
# Before:
    analysis = analyze_prompt(req.prompt)
# After:
    analysis = analyze_prompt(req.prompt, count=req.count)
```

- [ ] **Step 3: Run Python tests and type checks**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen && uv run pytest tests/ -x -q && uv run mypy mlx_audiogen/ && uv run ruff check .`

Expected: All pass with no errors.

- [ ] **Step 4: Commit**

```bash
git add mlx_audiogen/shared/prompt_suggestions.py mlx_audiogen/server/app.py
git commit -m "fix: forward count parameter to analyze_prompt in suggest endpoint"
```

### Task 2: Add TypeScript types

**Files:**
- Modify: `web/src/types/api.ts`

- [ ] **Step 1: Add `has_midi` to `JobInfo` and add `PresetInfo` type**

In `web/src/types/api.ts`, add `has_midi` as optional to `JobInfo` (line 49-60) and add `PresetInfo` after `StemResult`:

```typescript
// In JobInfo interface, after the `progress` field (line 59):
  has_midi?: boolean;

// After the StemResult interface (after line 40):

/** Preset info from /api/presets listing. */
export interface PresetInfo {
  name: string;
  filename: string;
  prompt: string;
  model: string;
}
```

- [ ] **Step 2: Commit**

```bash
git add web/src/types/api.ts
git commit -m "feat(web): add has_midi to JobInfo, add PresetInfo type"
```

### Task 3: Add preset API client functions

**Files:**
- Modify: `web/src/api/client.ts`

- [ ] **Step 1: Add preset functions and import PresetInfo**

At the top of `web/src/api/client.ts`, add `PresetInfo` to the import (line 1-8):

```typescript
import type {
  GenerateRequest,
  GenerateResponse,
  JobInfo,
  ModelInfo,
  PresetInfo,
  PromptAnalysis,
  StemResult,
} from "../types/api";
```

At the end of the file (after `separateStems`), add:

```typescript
/** List all saved presets. */
export function fetchPresets(): Promise<PresetInfo[]> {
  return request<PresetInfo[]>("/presets");
}

/** Save current params as a named preset. */
export function savePreset(
  name: string,
  params: GenerateRequest,
): Promise<{ saved: string }> {
  return request<{ saved: string }>(`/presets/${encodeURIComponent(name)}`, {
    method: "POST",
    body: JSON.stringify(params),
  });
}

/** Load a preset by name. */
export function loadPreset(name: string): Promise<GenerateRequest> {
  return request<GenerateRequest>(`/presets/${encodeURIComponent(name)}`);
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen/web && npx tsc --noEmit`

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
git add web/src/api/client.ts
git commit -m "feat(web): add preset API client functions"
```

---

## Chunk 2: Zustand Store Additions

### Task 4: Add suggestions, presets, stems, and output_mode to store

**Files:**
- Modify: `web/src/store/useStore.ts`

- [ ] **Step 1: Add imports**

At the top of `useStore.ts`, add to the existing imports:

```typescript
// Add to the type import from "../types/api":
import type { GenerateRequest, JobInfo, ModelInfo, PresetInfo, PromptAnalysis } from "../types/api";

// Add to the import from "../api/client":
import {
  fetchModels,
  submitGeneration,
  fetchJobStatus,
  getAudioUrl,
  suggestPrompts,
  separateStems,
  fetchPresets as apiFetchPresets,
  savePreset as apiSavePreset,
  loadPreset as apiLoadPreset,
} from "../api/client";
```

- [ ] **Step 2: Add `output_mode` to DEFAULT_PARAMS**

In the `DEFAULT_PARAMS` object (line 69-83), add after `style_coef`:

```typescript
  output_mode: "audio" as const,
```

- [ ] **Step 3: Extend `AppState` interface with new slices**

After the Settings section in the `AppState` interface (after line 66), add:

```typescript
  // --- Active Tab ---
  activeTab: "generate" | "suggest";
  setActiveTab: (tab: "generate" | "suggest") => void;

  // --- Prompt Suggestions ---
  suggestions: PromptAnalysis | null;
  suggestionsLoading: boolean;
  lastAnalyzedPrompt: string;
  fetchSuggestions: () => Promise<void>;

  // --- Presets ---
  presets: PresetInfo[];
  presetsLoading: boolean;
  loadPresets: () => Promise<void>;
  saveCurrentPreset: (name: string) => Promise<void>;
  applyPreset: (name: string) => Promise<void>;

  // --- Stem Separation ---
  stemResults: Record<string, Record<string, string>>; // jobId → {stemName → stemJobId}
  stemsLoading: Record<string, boolean>;
  stemAudioUrls: Record<string, Record<string, string>>; // jobId → {stemName → blobUrl}
  requestStemSeparation: (jobId: string) => Promise<void>;
```

- [ ] **Step 4: Implement the new state slices in `create<AppState>`**

After the `updateSettings` implementation (before the closing `}));`), add all new state and actions:

```typescript
  // --- Active Tab ---
  activeTab: "generate",
  setActiveTab: (tab) => set({ activeTab: tab }),

  // --- Prompt Suggestions ---
  suggestions: null,
  suggestionsLoading: false,
  lastAnalyzedPrompt: "",
  fetchSuggestions: async () => {
    const { params, lastAnalyzedPrompt, suggestionsLoading } = get();
    const prompt = params.prompt.trim();
    if (!prompt || suggestionsLoading) return;
    if (prompt === lastAnalyzedPrompt) return; // cached
    set({ suggestionsLoading: true });
    try {
      const analysis = await suggestPrompts(prompt);
      set({ suggestions: analysis, lastAnalyzedPrompt: prompt, suggestionsLoading: false });
    } catch (e) {
      console.error("Failed to fetch suggestions:", e);
      set({ suggestionsLoading: false });
    }
  },

  // --- Presets ---
  presets: [],
  presetsLoading: false,
  loadPresets: async () => {
    set({ presetsLoading: true });
    try {
      const presets = await apiFetchPresets();
      set({ presets, presetsLoading: false });
    } catch (e) {
      console.error("Failed to load presets:", e);
      set({ presetsLoading: false });
    }
  },

  saveCurrentPreset: async (name: string) => {
    const { params } = get();
    try {
      await apiSavePreset(name, params);
      await get().loadPresets(); // refresh list
    } catch (e) {
      console.error("Failed to save preset:", e);
    }
  },

  applyPreset: async (name: string) => {
    try {
      const loaded = await apiLoadPreset(name);
      // Validate model is available
      const { models } = get();
      const validModel = models.some((m) => m.model_type === loaded.model);
      set((s) => ({
        params: {
          ...s.params,
          ...loaded,
          model: validModel ? loaded.model : s.params.model,
        },
        activeTab: "generate",
      }));
    } catch (e) {
      console.error("Failed to apply preset:", e);
    }
  },

  // --- Stem Separation ---
  stemResults: {},
  stemsLoading: {},
  stemAudioUrls: {},
  requestStemSeparation: async (jobId: string) => {
    const { stemsLoading } = get();
    if (stemsLoading[jobId]) return;
    set((s) => ({ stemsLoading: { ...s.stemsLoading, [jobId]: true } }));
    try {
      const result = await separateStems(jobId);
      set((s) => ({
        stemResults: { ...s.stemResults, [jobId]: result.stems },
        stemsLoading: { ...s.stemsLoading, [jobId]: false },
      }));
      // Eagerly download stem audio blobs before server cleanup (5 min)
      const urls: Record<string, string> = {};
      for (const [stemName, stemId] of Object.entries(result.stems)) {
        try {
          const res = await fetch(getAudioUrl(stemId));
          const blob = await res.blob();
          urls[stemName] = URL.createObjectURL(blob);
        } catch {
          // Fall back to direct server URL
          urls[stemName] = getAudioUrl(stemId);
        }
      }
      set((s) => ({
        stemAudioUrls: { ...s.stemAudioUrls, [jobId]: urls },
      }));
    } catch (e) {
      console.error("Failed to separate stems:", e);
      set((s) => ({ stemsLoading: { ...s.stemsLoading, [jobId]: false } }));
    }
  },
```

- [ ] **Step 5: Verify TypeScript compiles**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen/web && npx tsc --noEmit`

Expected: No errors.

- [ ] **Step 6: Commit**

```bash
git add web/src/store/useStore.ts
git commit -m "feat(web): add suggestions, presets, stems, output_mode to Zustand store"
```

---

## Chunk 3: Tab System + Suggest Panel

### Task 5: Create TabBar component

**Files:**
- Create: `web/src/components/TabBar.tsx`

- [ ] **Step 1: Write TabBar component**

Create `web/src/components/TabBar.tsx`:

```typescript
interface TabBarProps {
  active: string;
  tabs: { id: string; label: string }[];
  onChange: (id: string) => void;
}

export default function TabBar({ active, tabs, onChange }: TabBarProps) {
  return (
    <div className="flex border-b border-border">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`
            flex-1 px-3 py-2 text-xs font-medium uppercase tracking-wider
            transition-colors
            ${
              active === tab.id
                ? "border-b-2 border-accent text-text-primary"
                : "text-text-muted hover:text-text-secondary"
            }
          `}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add web/src/components/TabBar.tsx
git commit -m "feat(web): add TabBar component"
```

### Task 6: Create SuggestPanel component

**Files:**
- Create: `web/src/components/SuggestPanel.tsx`

- [ ] **Step 1: Write SuggestPanel component**

Create `web/src/components/SuggestPanel.tsx`:

```typescript
import { useEffect, useState } from "react";
import { useStore } from "../store/useStore";

const PRESET_NAME_RE = /^[a-zA-Z0-9_-]{1,64}$/;

export default function SuggestPanel() {
  const prompt = useStore((s) => s.params.prompt);
  const suggestions = useStore((s) => s.suggestions);
  const suggestionsLoading = useStore((s) => s.suggestionsLoading);
  const fetchSuggestions = useStore((s) => s.fetchSuggestions);
  const setParam = useStore((s) => s.setParam);
  const setActiveTab = useStore((s) => s.setActiveTab);

  const presets = useStore((s) => s.presets);
  const presetsLoading = useStore((s) => s.presetsLoading);
  const loadPresets = useStore((s) => s.loadPresets);
  const saveCurrentPreset = useStore((s) => s.saveCurrentPreset);
  const applyPreset = useStore((s) => s.applyPreset);

  const [saveName, setSaveName] = useState("");
  const [showSaveInput, setShowSaveInput] = useState(false);

  // Auto-fetch suggestions when tab opens with a prompt
  useEffect(() => {
    if (prompt.trim()) {
      fetchSuggestions();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Load presets on mount
  useEffect(() => {
    loadPresets();
  }, [loadPresets]);

  const handleUseSuggestion = (text: string) => {
    setParam("prompt", text);
    setActiveTab("generate");
  };

  const handleCopy = async (text: string) => {
    await navigator.clipboard.writeText(text);
  };

  const handleSavePreset = async () => {
    const name = saveName.trim();
    if (!PRESET_NAME_RE.test(name)) return;
    await saveCurrentPreset(name);
    setSaveName("");
    setShowSaveInput(false);
  };

  return (
    <div className="flex flex-col gap-5 overflow-y-auto">
      {/* --- Prompt Suggestions --- */}
      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
            Suggestions
          </h3>
          <button
            onClick={fetchSuggestions}
            disabled={suggestionsLoading || !prompt.trim()}
            className="text-xs text-accent hover:text-accent-hover disabled:opacity-50"
          >
            {suggestionsLoading ? "Analyzing..." : "Analyze"}
          </button>
        </div>

        {!prompt.trim() && (
          <p className="text-xs text-text-muted">
            Enter a prompt in the Generate tab first.
          </p>
        )}

        {suggestions && (
          <>
            {/* Analysis tags */}
            <div className="space-y-1.5">
              {suggestions.genres.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {suggestions.genres.map((g) => (
                    <span
                      key={g}
                      className="rounded bg-warning/20 px-1.5 py-0.5 text-xs text-warning"
                    >
                      {g}
                    </span>
                  ))}
                </div>
              )}
              {suggestions.moods.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {suggestions.moods.map((m) => (
                    <span
                      key={m}
                      className="rounded bg-success/20 px-1.5 py-0.5 text-xs text-success"
                    >
                      {m}
                    </span>
                  ))}
                </div>
              )}
              {suggestions.instruments.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {suggestions.instruments.map((i) => (
                    <span
                      key={i}
                      className="rounded bg-info/20 px-1.5 py-0.5 text-xs text-info"
                    >
                      {i}
                    </span>
                  ))}
                </div>
              )}
              {suggestions.missing.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {suggestions.missing.map((m) => (
                    <span
                      key={m}
                      className="rounded bg-surface-3 px-1.5 py-0.5 text-xs text-text-muted"
                    >
                      + {m}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Suggestion cards */}
            <div className="space-y-2">
              {suggestions.suggestions.map((s, idx) => (
                <div
                  key={idx}
                  className="rounded border border-border bg-surface-2 p-2.5 space-y-2"
                >
                  <p className="text-xs text-text-primary leading-relaxed">
                    {s}
                  </p>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleUseSuggestion(s)}
                      className="rounded bg-accent/20 px-2 py-0.5 text-xs text-accent hover:bg-accent/30"
                    >
                      Use
                    </button>
                    <button
                      onClick={() => handleCopy(s)}
                      className="rounded bg-surface-3 px-2 py-0.5 text-xs text-text-muted hover:text-text-secondary"
                    >
                      Copy
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </section>

      {/* --- Presets --- */}
      <section className="space-y-3 border-t border-border pt-4">
        <div className="flex items-center justify-between">
          <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
            Presets
          </h3>
          <button
            onClick={() => setShowSaveInput(!showSaveInput)}
            className="text-xs text-accent hover:text-accent-hover"
          >
            {showSaveInput ? "Cancel" : "Save Current"}
          </button>
        </div>

        {showSaveInput && (
          <div className="flex gap-2">
            <input
              type="text"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
              placeholder="preset-name"
              maxLength={64}
              className="
                flex-1 rounded border border-border bg-surface-2 px-2 py-1
                text-xs text-text-primary placeholder-text-muted
                focus:border-accent focus:outline-none
              "
            />
            <button
              onClick={handleSavePreset}
              disabled={!PRESET_NAME_RE.test(saveName.trim())}
              className="rounded bg-accent px-3 py-1 text-xs font-medium text-surface-0 disabled:opacity-50"
            >
              Save
            </button>
          </div>
        )}

        {presetsLoading && (
          <p className="text-xs text-text-muted">Loading presets...</p>
        )}

        {!presetsLoading && presets.length === 0 && (
          <p className="text-xs text-text-muted">No presets saved yet.</p>
        )}

        <div className="space-y-1.5">
          {presets.map((p) => (
            <button
              key={p.name}
              onClick={() => applyPreset(p.name)}
              className="
                w-full rounded border border-border bg-surface-2 p-2 text-left
                hover:border-accent/40 transition-colors
              "
            >
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-text-primary truncate">
                  {p.name}
                </span>
                <span className="shrink-0 rounded bg-surface-3 px-1.5 py-0.5 text-xs text-text-muted">
                  {p.model === "musicgen" ? "MG" : "SA"}
                </span>
              </div>
              {p.prompt && (
                <p className="mt-0.5 text-xs text-text-muted truncate">
                  {p.prompt}
                </p>
              )}
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen/web && npx tsc --noEmit`

- [ ] **Step 3: Commit**

```bash
git add web/src/components/SuggestPanel.tsx
git commit -m "feat(web): add SuggestPanel with prompt suggestions and presets"
```

### Task 7: Add tab system to App.tsx

**Files:**
- Modify: `web/src/App.tsx`

- [ ] **Step 1: Update App.tsx with tab system**

Replace the entire `App.tsx` with:

```typescript
import { useEffect } from "react";
import { useStore } from "./store/useStore";
import { useServerHeartbeat } from "./hooks/useServerHeartbeat";
import Header from "./components/Header";
import ModelSelector from "./components/ModelSelector";
import PromptInput from "./components/PromptInput";
import ParameterPanel from "./components/ParameterPanel";
import GenerateButton from "./components/GenerateButton";
import HistoryPanel from "./components/HistoryPanel";
import AudioDeviceSelector from "./components/AudioDeviceSelector";
import SettingsPanel from "./components/SettingsPanel";
import TabBar from "./components/TabBar";
import SuggestPanel from "./components/SuggestPanel";

const TABS = [
  { id: "generate", label: "Generate" },
  { id: "suggest", label: "Suggest" },
];

export default function App() {
  const loadModels = useStore((s) => s.loadModels);
  const loadHistory = useStore((s) => s.loadHistory);
  const loadSettings = useStore((s) => s.loadSettings);
  const modelsLoading = useStore((s) => s.modelsLoading);
  const modelsError = useStore((s) => s.modelsError);
  const activeTab = useStore((s) => s.activeTab);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const connected = useServerHeartbeat();

  useEffect(() => {
    loadModels();
    loadHistory();
    loadSettings();
  }, [loadModels, loadHistory, loadSettings]);

  return (
    <div className="flex h-screen flex-col bg-surface-0">
      {/* Server disconnected banner */}
      {!connected && (
        <div className="bg-error/90 text-surface-0 px-4 py-2 text-center text-sm font-medium">
          Server disconnected — restart with{" "}
          <code className="bg-surface-0/20 px-1 rounded">mlx-audiogen-app</code>
        </div>
      )}

      <Header />

      <main className="flex flex-1 overflow-hidden">
        {/* Left panel: Controls */}
        <div className="flex w-80 shrink-0 flex-col border-r border-border bg-surface-1">
          <TabBar
            active={activeTab}
            tabs={TABS}
            onChange={(id) => setActiveTab(id as "generate" | "suggest")}
          />

          <div className="flex flex-1 flex-col gap-5 overflow-y-auto p-5">
            {activeTab === "generate" && (
              <>
                {modelsLoading && (
                  <div className="text-xs text-text-muted">Loading models...</div>
                )}
                {modelsError && (
                  <div className="rounded border border-error/30 bg-error/10 px-3 py-2 text-xs text-error">
                    Failed to connect to server: {modelsError}
                  </div>
                )}

                <ModelSelector />
                <PromptInput />
                <ParameterPanel />
                <GenerateButton />
              </>
            )}

            {activeTab === "suggest" && <SuggestPanel />}
          </div>

          {/* Bottom section: always visible */}
          <div className="space-y-4 border-t border-border p-5">
            <SettingsPanel />
            <AudioDeviceSelector />
          </div>
        </div>

        {/* Right panel: History / Output */}
        <div className="flex flex-1 flex-col overflow-y-auto p-5">
          <HistoryPanel />
        </div>
      </main>
    </div>
  );
}
```

- [ ] **Step 2: Verify TypeScript compiles and dev server renders**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen/web && npx tsc --noEmit`

- [ ] **Step 3: Commit**

```bash
git add web/src/App.tsx
git commit -m "feat(web): add tabbed left panel with Generate/Suggest tabs"
```

---

## Chunk 4: Parameter Panel + History Enhancements

### Task 8: Add output_mode dropdown to ParameterPanel

**Files:**
- Modify: `web/src/components/ParameterPanel.tsx`

- [ ] **Step 1: Add output_mode dropdown after the Seed control**

In `ParameterPanel.tsx`, after the seed control closing `</div>` (line 56) and before the model-specific params `<div>` (line 58), add:

```typescript
      {/* Output mode */}
      <div className="space-y-1">
        <label className="text-xs font-medium text-text-secondary">
          Output
        </label>
        <select
          value={params.output_mode ?? "audio"}
          onChange={(e) =>
            setParam(
              "output_mode",
              e.target.value as "audio" | "midi" | "both",
            )
          }
          disabled={isGenerating}
          className="
            w-full rounded border border-border bg-surface-2 px-2 py-1.5
            text-xs text-text-primary
            focus:border-accent focus:outline-none
            disabled:opacity-50
          "
        >
          <option value="audio">Audio only</option>
          <option value="midi">MIDI only</option>
          <option value="both">Audio + MIDI</option>
        </select>
      </div>
```

- [ ] **Step 2: Commit**

```bash
git add web/src/components/ParameterPanel.tsx
git commit -m "feat(web): add output_mode dropdown to ParameterPanel"
```

### Task 9: Enhance HistoryPanel with metadata, MIDI, and stems

**Files:**
- Modify: `web/src/components/HistoryPanel.tsx`

- [ ] **Step 1: Add store selectors and stem color map**

At the top of `HistoryPanel.tsx`, update imports and add constants:

```typescript
import { useStore } from "../store/useStore";
import { getMidiUrl } from "../api/client";
import AudioPlayer from "./AudioPlayer";

const STEM_COLORS: Record<string, { bg: string; text: string }> = {
  drums: { bg: "bg-success/20", text: "text-success" },
  bass: { bg: "bg-error/20", text: "text-error" },
  vocals: { bg: "bg-purple-400/20", text: "text-purple-400" },
  other: { bg: "bg-info/20", text: "text-info" },
  mid: { bg: "bg-warning/20", text: "text-warning" },
  high: { bg: "bg-cyan-400/20", text: "text-cyan-400" },
};
const DEFAULT_STEM_COLOR = { bg: "bg-surface-3", text: "text-text-secondary" };
```

Add new store selectors inside the component function body (after existing selectors):

```typescript
  const stemResults = useStore((s) => s.stemResults);
  const stemsLoading = useStore((s) => s.stemsLoading);
  const stemAudioUrls = useStore((s) => s.stemAudioUrls);
  const requestStemSeparation = useStore((s) => s.requestStemSeparation);
```

- [ ] **Step 2: Add metadata line and action buttons to each history card**

Replace the metadata `<div>` (lines 56-74 — the one with MusicGen/Stable Audio, seconds, gen time, and timestamp) with an enhanced version that also includes sample rate info:

```typescript
                <div className="flex flex-wrap gap-2 mt-1 text-xs text-text-muted">
                  <span>
                    {entry.job.model === "musicgen"
                      ? "MusicGen"
                      : "Stable Audio"}
                  </span>
                  <span>{entry.job.seconds}s</span>
                  {entry.job.sample_rate && (
                    <span>
                      {(entry.job.sample_rate / 1000).toFixed(0)}kHz{" "}
                      {entry.job.model?.includes("stereo") ? "stereo" : "mono"}
                    </span>
                  )}
                  {entry.job.completed_at && entry.job.created_at && (
                    <span>
                      {(
                        entry.job.completed_at - entry.job.created_at
                      ).toFixed(1)}
                      s gen
                    </span>
                  )}
                </div>
```

- [ ] **Step 3: Add MIDI download and Stems buttons after the AudioPlayer**

After the `<AudioPlayer />` component (line 151), add:

```typescript
            {/* Action bar: MIDI + Stems */}
            <div className="flex items-center gap-2">
              {/* MIDI download */}
              {entry.job.has_midi ? (
                <a
                  href={getMidiUrl(entry.id)}
                  download
                  className="rounded bg-surface-2 px-2 py-1 text-xs text-accent hover:bg-surface-3"
                >
                  MIDI
                </a>
              ) : (
                <span
                  className="rounded bg-surface-2 px-2 py-1 text-xs text-text-muted opacity-40 cursor-not-allowed"
                  title="Set output to 'midi' or 'both' before generating"
                >
                  MIDI
                </span>
              )}

              {/* Stem separation */}
              <button
                onClick={() => requestStemSeparation(entry.id)}
                disabled={!!stemsLoading[entry.id] || !!stemResults[entry.id]}
                className="rounded bg-surface-2 px-2 py-1 text-xs text-accent hover:bg-surface-3 disabled:opacity-40"
              >
                {stemsLoading[entry.id]
                  ? "Separating..."
                  : stemResults[entry.id]
                    ? "Stems"
                    : "Separate"}
              </button>

              {/* WAV download */}
              <a
                href={entry.audioUrl}
                download={`${entry.job.model}_${entry.id}.wav`}
                className="rounded bg-surface-2 px-2 py-1 text-xs text-text-secondary hover:bg-surface-3"
              >
                WAV
              </a>
            </div>

            {/* Expanded stems section */}
            {stemResults[entry.id] && (
              <div className="space-y-1.5 rounded border border-border bg-surface-0 p-2">
                {Object.entries(stemResults[entry.id]!).map(
                  ([stemName, stemId]) => {
                    const color =
                      STEM_COLORS[stemName] ?? DEFAULT_STEM_COLOR;
                    const audioUrl =
                      stemAudioUrls[entry.id]?.[stemName];
                    return (
                      <div
                        key={stemName}
                        className={`flex items-center gap-2 rounded p-1.5 ${color.bg}`}
                      >
                        <span
                          className={`w-14 text-xs font-medium capitalize ${color.text}`}
                        >
                          {stemName}
                        </span>
                        {audioUrl ? (
                          <>
                            <audio
                              src={audioUrl}
                              controls
                              className="h-6 flex-1"
                              style={{ minWidth: 0 }}
                            />
                            <a
                              href={audioUrl}
                              download={`${entry.id}_${stemName}.wav`}
                              className="shrink-0 text-xs text-text-muted hover:text-text-secondary"
                            >
                              DL
                            </a>
                          </>
                        ) : (
                          <span className="text-xs text-text-muted">
                            Loading...
                          </span>
                        )}
                      </div>
                    );
                  },
                )}
              </div>
            )}
```

- [ ] **Step 4: Verify TypeScript compiles**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen/web && npx tsc --noEmit`

- [ ] **Step 5: Commit**

```bash
git add web/src/components/HistoryPanel.tsx
git commit -m "feat(web): add metadata, MIDI download, and stem separation to history entries"
```

---

## Chunk 5: Header Support Link + Build + Verify

### Task 10: Add support link to Header

**Files:**
- Modify: `web/src/components/Header.tsx`

- [ ] **Step 1: Add support link in header**

In `Header.tsx`, add the support link inside the right-side `<div>` (line 15), after the loaded count span:

```typescript
        <a
          href="https://paypal.me/jasonvassallo"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-xs text-text-muted hover:text-accent transition-colors"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
          </svg>
          Support
        </a>
```

- [ ] **Step 2: Commit**

```bash
git add web/src/components/Header.tsx
git commit -m "feat(web): add PayPal support link to header"
```

### Task 11: Full build and verification

- [ ] **Step 1: Run full Python quality suite**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen && uv run ruff format . && uv run ruff check . && uv run mypy mlx_audiogen/ && uv run bandit -r mlx_audiogen/ -c pyproject.toml && uv run pytest -x -q`

Expected: All pass.

- [ ] **Step 2: Run full web build**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen/web && npm run build`

Expected: Build succeeds with no TypeScript errors.

- [ ] **Step 3: Fix any issues**

If any step fails, fix the issue and re-run.

- [ ] **Step 4: Final commit with all fixes**

```bash
git add -A
git commit -m "chore: Phase 7a Web UI Catch-Up — full build passes"
```

- [ ] **Step 5: Push to remote**

```bash
git push origin main
```
