import { useStore } from "../store/useStore";

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
    </div>
  );
}
