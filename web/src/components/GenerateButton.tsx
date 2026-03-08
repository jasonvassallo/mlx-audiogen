import { useStore } from "../store/useStore";

export default function GenerateButton() {
  const generate = useStore((s) => s.generate);
  const isGenerating = useStore((s) => s.isGenerating);
  const activeJob = useStore((s) => s.activeJob);
  const generateError = useStore((s) => s.generateError);

  // Use real server-reported progress (0.0 to 1.0)
  const progress = activeJob?.progress ?? 0;
  const progressPct = Math.round(progress * 100);

  return (
    <div className="space-y-2">
      <button
        onClick={generate}
        disabled={isGenerating}
        className="
          w-full rounded py-3 text-sm font-bold uppercase tracking-wider
          transition-all duration-150
          bg-accent text-surface-0 hover:bg-accent-hover
          active:scale-[0.98]
          disabled:opacity-70 disabled:cursor-not-allowed disabled:active:scale-100
        "
      >
        {isGenerating ? `Generating ${progressPct}%` : "Generate"}
      </button>

      {/* Progress bar */}
      {isGenerating && (
        <div className="space-y-1">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
            <div
              className="h-full rounded-full bg-accent transition-all duration-300"
              style={{ width: `${Math.max(2, progressPct)}%` }}
            />
          </div>
          <div className="text-center text-xs text-text-muted">
            {activeJob?.status === "queued" && "Queued — waiting for model to load..."}
            {activeJob?.status === "running" &&
              `Step ${Math.round(progress * 100)}%`}
          </div>
        </div>
      )}

      {/* Error display */}
      {generateError && !isGenerating && (
        <div className="rounded border border-error/30 bg-error/10 px-3 py-2 text-xs text-error">
          {generateError}
        </div>
      )}
    </div>
  );
}
