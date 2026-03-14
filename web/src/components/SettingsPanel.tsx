import { useEffect } from "react";
import { useStore } from "../store/useStore";
import { deleteLora } from "../api/client";

const RETENTION_OPTIONS = [
  { label: "Keep forever", value: 0 },
  { label: "1 hour", value: 1 },
  { label: "6 hours", value: 6 },
  { label: "24 hours", value: 24 },
  { label: "7 days", value: 168 },
  { label: "30 days", value: 720 },
  { label: "90 days", value: 2160 },
  { label: "180 days", value: 4320 },
  { label: "1 year", value: 8760 },
  { label: "2 years", value: 17520 },
  { label: "3 years", value: 26280 },
  { label: "4 years", value: 35040 },
  { label: "5 years", value: 43800 },
];

export default function SettingsPanel() {
  const settings = useStore((s) => s.settings);
  const updateSettings = useStore((s) => s.updateSettings);
  const history = useStore((s) => s.history);
  const favoriteCount = history.filter((h) => h.favorite).length;

  return (
    <div className="space-y-3">
      <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
        History
      </h3>

      {/* Auto-delete retention */}
      <div className="space-y-1">
        <label className="text-xs font-medium text-text-secondary">
          Auto-delete after
        </label>
        <select
          value={settings.retentionHours}
          onChange={(e) =>
            updateSettings({ retentionHours: parseInt(e.target.value) })
          }
          className="
            w-full rounded border border-border bg-surface-2 px-2 py-1.5
            text-xs text-text-primary
            focus:border-accent focus:outline-none
          "
        >
          {RETENTION_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
        <p className="text-xs text-text-muted">
          Favorites are never auto-deleted.
          {favoriteCount > 0 && (
            <span className="text-accent"> {favoriteCount} favorited.</span>
          )}
        </p>
      </div>

      {/* LoRA Adapters section */}
      <LoRASection />
    </div>
  );
}

function LoRASection() {
  const loras = useStore((s) => s.loras);
  const fetchLoras = useStore((s) => s.fetchLoras);

  useEffect(() => {
    fetchLoras();
  }, [fetchLoras]);

  const handleDelete = async (name: string) => {
    try {
      await deleteLora(name);
      fetchLoras();
    } catch {
      // Ignore — may already be deleted
    }
  };

  return (
    <div className="space-y-2 border-t border-border pt-3">
      <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
        LoRA Adapters
      </h3>
      <p className="text-[10px] text-text-muted">
        ~/.mlx-audiogen/loras/
      </p>

      {loras.length === 0 ? (
        <p className="text-xs text-text-muted">
          No LoRA adapters installed. Train one in the Train tab.
        </p>
      ) : (
        <div className="space-y-1.5">
          {loras.map((l) => (
            <div
              key={l.name}
              className="flex items-center justify-between rounded border border-border bg-surface-2 px-3 py-2"
            >
              <div>
                <div className="text-xs font-medium text-text-primary">
                  {l.name}
                </div>
                <div className="text-[10px] text-text-muted">
                  {l.base_model} — rank {l.rank}
                  {l.profile && ` — ${l.profile}`}
                </div>
              </div>
              <button
                onClick={() => handleDelete(l.name)}
                className="text-xs text-red-400 hover:text-red-300"
                title="Delete LoRA"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
