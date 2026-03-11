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
