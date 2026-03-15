import { useCallback, useEffect, useState } from "react";
import { useStore } from "../store/useStore";
import {
  deleteLora,
  getCredentialStatus,
  setCredential,
  deleteCredential,
  getTasteProfile,
  refreshTasteProfile,
  setTasteOverrides,
  getFlywheelConfig,
  updateFlywheelConfig,
  getFlywheelStatus,
  triggerRetrain,
  resetKeptGenerations,
} from "../api/client";
import type {
  CredentialStatus,
  FlywheelConfig,
  FlywheelStatus,
  TasteProfile,
} from "../types/api";

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

      {/* API Keys section */}
      <APIKeysSection />

      {/* Taste Profile section */}
      <TasteProfileSection />

      {/* Flywheel Intelligence section */}
      <FlywheelSection />
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

/** Credential row for a single service. */
function CredentialRow({
  label,
  configured,
  needsKey,
  onSave,
  onClear,
}: {
  label: string;
  configured: boolean;
  needsKey: boolean;
  onSave: (key: string) => void;
  onClear: () => void;
}) {
  const [value, setValue] = useState("");

  return (
    <div className="flex items-center gap-2">
      <span
        className={`inline-block h-2 w-2 shrink-0 rounded-full ${
          configured ? "bg-emerald-500" : "bg-zinc-600"
        }`}
      />
      <span className="w-24 text-xs text-text-primary">{label}</span>
      {needsKey ? (
        <>
          <input
            type="password"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder="API key..."
            className="flex-1 rounded border border-border bg-surface-2 px-2 py-1 text-xs text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
          />
          <button
            onClick={() => { if (value.trim()) { onSave(value.trim()); setValue(""); } }}
            disabled={!value.trim()}
            className="rounded bg-sky-600 px-2 py-1 text-[10px] font-medium text-white hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Save
          </button>
          {configured && (
            <button
              onClick={onClear}
              className="text-[10px] text-red-400 hover:text-red-300"
            >
              Clear
            </button>
          )}
        </>
      ) : (
        <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-[10px] text-emerald-400">
          Ready
        </span>
      )}
    </div>
  );
}

function APIKeysSection() {
  const [creds, setCreds] = useState<CredentialStatus | null>(null);
  const [autoEnrich, setAutoEnrich] = useState(
    () => localStorage.getItem("mlx_auto_enrich") === "true",
  );

  const loadCreds = useCallback(async () => {
    try {
      const status = await getCredentialStatus();
      setCreds(status);
    } catch {
      // Server may not support this endpoint yet
    }
  }, []);

  useEffect(() => {
    loadCreds();
  }, [loadCreds]);

  const handleSave = useCallback(
    async (service: string, apiKey: string) => {
      try {
        await setCredential(service, apiKey);
        loadCreds();
      } catch {
        // Ignore
      }
    },
    [loadCreds],
  );

  const handleClear = useCallback(
    async (service: string) => {
      try {
        await deleteCredential(service);
        loadCreds();
      } catch {
        // Ignore
      }
    },
    [loadCreds],
  );

  const toggleAutoEnrich = useCallback(() => {
    setAutoEnrich((prev) => {
      const next = !prev;
      localStorage.setItem("mlx_auto_enrich", String(next));
      return next;
    });
  }, []);

  return (
    <div className="space-y-2 border-t border-border pt-3">
      <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
        Web Enrichment API Keys
      </h3>
      <p className="text-[10px] text-text-muted">
        Keys are stored securely in macOS Keychain
      </p>

      <div className="space-y-2 rounded border border-border bg-surface-2 p-3">
        <CredentialRow
          label="MusicBrainz"
          configured={creds?.musicbrainz ?? true}
          needsKey={false}
          onSave={() => {}}
          onClear={() => {}}
        />
        <CredentialRow
          label="Last.fm"
          configured={creds?.lastfm ?? false}
          needsKey
          onSave={(key) => handleSave("lastfm", key)}
          onClear={() => handleClear("lastfm")}
        />
        <CredentialRow
          label="Discogs"
          configured={creds?.discogs ?? false}
          needsKey
          onSave={(key) => handleSave("discogs", key)}
          onClear={() => handleClear("discogs")}
        />
      </div>

      <label className="flex items-center gap-2 text-xs text-text-primary cursor-pointer">
        <input
          type="checkbox"
          checked={autoEnrich}
          onChange={toggleAutoEnrich}
          className="accent-sky-500"
        />
        Auto-enrich on browse
      </label>
    </div>
  );
}

function TasteProfileSection() {
  const [profile, setProfile] = useState<TasteProfile | null>(null);
  const [overrideText, setOverrideText] = useState("");
  const [loading, setLoading] = useState(false);

  const loadProfile = useCallback(async () => {
    try {
      const p = await getTasteProfile();
      setProfile(p);
      setOverrideText(p.overrides ?? "");
    } catch {
      // Server may not support this endpoint yet
    }
  }, []);

  useEffect(() => {
    loadProfile();
  }, [loadProfile]);

  const handleRefresh = useCallback(async () => {
    setLoading(true);
    try {
      const p = await refreshTasteProfile();
      setProfile(p);
      setOverrideText(p.overrides ?? "");
    } catch {
      // Ignore
    }
    setLoading(false);
  }, []);

  const handleSaveOverride = useCallback(async () => {
    try {
      const p = await setTasteOverrides(overrideText);
      setProfile(p);
    } catch {
      // Ignore
    }
  }, [overrideText]);

  const handleReset = useCallback(async () => {
    try {
      const p = await setTasteOverrides("");
      setProfile(p);
      setOverrideText("");
    } catch {
      // Ignore
    }
  }, []);

  const hasData = profile && profile.library_track_count > 0;

  return (
    <div className="space-y-2 border-t border-border pt-3">
      <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
        Your Taste Profile
      </h3>

      {!hasData ? (
        <p className="text-xs text-text-muted">
          No taste data yet. Scan your library and generate some audio to build
          your profile.
        </p>
      ) : (
        <div className="space-y-3 rounded border border-border bg-surface-2 p-3">
          {/* Top Genres */}
          {profile.top_genres.length > 0 && (
            <div>
              <p className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Top Genres
              </p>
              <div className="flex flex-wrap gap-1">
                {profile.top_genres.map((g) => (
                  <span
                    key={g.name}
                    className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-400"
                  >
                    {g.name}
                    <span className="ml-0.5 text-amber-400/50">
                      {Math.round(g.weight * 100)}%
                    </span>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* BPM Range */}
          {profile.bpm_range && (
            <div>
              <p className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                BPM Range
              </p>
              <span className="text-lg font-bold tabular-nums text-text-primary">
                {Math.round(profile.bpm_range[0])}-{Math.round(profile.bpm_range[1])}
              </span>
            </div>
          )}

          {/* Preferred Keys */}
          {profile.key_preferences.length > 0 && (
            <div>
              <p className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Preferred Keys
              </p>
              <div className="flex flex-wrap gap-1">
                {profile.key_preferences.map((k) => (
                  <span
                    key={k.name}
                    className="rounded bg-rose-500/10 px-1.5 py-0.5 text-[10px] text-rose-400"
                  >
                    {k.name}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Mood Profile */}
          {profile.mood_profile.length > 0 && (
            <div>
              <p className="mb-1 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
                Mood
              </p>
              <div className="flex flex-wrap gap-1">
                {profile.mood_profile.map((m) => (
                  <span
                    key={m.name}
                    className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-[10px] text-emerald-400"
                  >
                    {m.name}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Creative Intent */}
          {(profile.gen_genres.length > 0 ||
            profile.gen_moods.length > 0 ||
            profile.gen_instruments.length > 0) && (
            <div className="rounded bg-fuchsia-500/5 p-2">
              <p className="mb-1 text-[10px] font-medium uppercase tracking-wider text-fuchsia-400">
                Creative Intent
              </p>
              <div className="flex flex-wrap gap-1">
                {profile.gen_genres.map((g) => (
                  <span
                    key={g.name}
                    className="rounded bg-fuchsia-500/10 px-1.5 py-0.5 text-[10px] text-fuchsia-300"
                  >
                    {g.name}
                  </span>
                ))}
                {profile.gen_moods.map((m) => (
                  <span
                    key={m.name}
                    className="rounded bg-fuchsia-500/10 px-1.5 py-0.5 text-[10px] text-fuchsia-300"
                  >
                    {m.name}
                  </span>
                ))}
                {profile.gen_instruments.map((i) => (
                  <span
                    key={i.name}
                    className="rounded bg-fuchsia-500/10 px-1.5 py-0.5 text-[10px] text-fuchsia-300"
                  >
                    {i.name}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Stats line */}
          <p className="text-[10px] text-text-muted">
            Learned from {profile.library_track_count} library tracks +{" "}
            {profile.generation_count} generations
          </p>

          {/* Manual override */}
          <div className="space-y-1">
            <label className="text-[10px] font-medium text-zinc-500">
              Manual Override
            </label>
            <input
              type="text"
              value={overrideText}
              onChange={(e) => setOverrideText(e.target.value)}
              placeholder="e.g. prefer ambient over EDM, more minor keys..."
              className="w-full rounded border border-border bg-zinc-900 px-2 py-1 text-xs text-text-primary placeholder:text-text-muted focus:border-accent focus:outline-none"
            />
            <button
              onClick={handleSaveOverride}
              className="rounded bg-sky-600 px-2 py-1 text-[10px] font-medium text-white hover:bg-sky-500"
            >
              Save Override
            </button>
          </div>

          {/* Refresh + Reset */}
          <div className="flex gap-2">
            <button
              onClick={handleRefresh}
              disabled={loading}
              className="rounded border border-zinc-600 px-2 py-1 text-[10px] text-zinc-300 hover:border-zinc-500 disabled:opacity-50"
            >
              {loading ? "Refreshing..." : "Refresh"}
            </button>
            <button
              onClick={handleReset}
              className="text-[10px] text-red-400 hover:text-red-300"
            >
              Reset
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function FlywheelSection() {
  const [config, setConfig] = useState<FlywheelConfig | null>(null);
  const [status, setStatus] = useState<FlywheelStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [actionMsg, setActionMsg] = useState<string | null>(null);
  const loras = useStore((s) => s.loras);

  const loadData = useCallback(async () => {
    try {
      const [c, s] = await Promise.all([
        getFlywheelConfig(),
        getFlywheelStatus(),
      ]);
      setConfig(c);
      setStatus(s);
    } catch {
      // Server may not support flywheel endpoints yet
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleToggleAutoRetrain = useCallback(async () => {
    if (!config) return;
    try {
      const updated = await updateFlywheelConfig({
        auto_retrain: !config.auto_retrain,
      });
      setConfig(updated);
    } catch {
      // Ignore
    }
  }, [config]);

  const handleThresholdChange = useCallback(
    async (value: number) => {
      if (!config) return;
      try {
        const updated = await updateFlywheelConfig({
          retrain_threshold: value,
        });
        setConfig(updated);
      } catch {
        // Ignore
      }
    },
    [config],
  );

  const handleTasteIntervalChange = useCallback(
    async (value: number) => {
      if (!config) return;
      try {
        const updated = await updateFlywheelConfig({
          taste_refresh_interval: value,
        });
        setConfig(updated);
      } catch {
        // Ignore
      }
    },
    [config],
  );

  const handleBlendChange = useCallback(
    async (value: number) => {
      if (!config) return;
      try {
        const updated = await updateFlywheelConfig({
          blend_ratio: value / 100,
        });
        setConfig(updated);
      } catch {
        // Ignore
      }
    },
    [config],
  );

  const handleRetrain = useCallback(async () => {
    const first = loras[0];
    if (!first) return;
    setLoading(true);
    setActionMsg(null);
    try {
      const result = await triggerRetrain(first.name);
      setActionMsg(`Re-train started: ${result.adapter}`);
      loadData();
    } catch (err) {
      setActionMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
    }
    setLoading(false);
  }, [loras, loadData]);

  const handleReset = useCallback(async () => {
    const first = loras[0];
    if (!first) return;
    setLoading(true);
    setActionMsg(null);
    try {
      const result = await resetKeptGenerations(first.name);
      setActionMsg(`Cache reset: ${result.adapter}`);
      loadData();
    } catch (err) {
      setActionMsg(`Error: ${err instanceof Error ? err.message : String(err)}`);
    }
    setLoading(false);
  }, [loras, loadData]);

  const blendPct = config ? Math.round(config.blend_ratio * 100) : 80;

  return (
    <div className="space-y-2 border-t border-border pt-3">
      <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
        Flywheel Intelligence
      </h3>
      <p className="text-[10px] text-text-muted">
        Auto-retrain LoRA adapters from starred generations
      </p>

      <div className="space-y-3 rounded border border-border bg-surface-2 p-3">
        {/* Auto-retrain toggle */}
        <label className="flex items-center gap-2 text-xs text-text-primary cursor-pointer">
          <input
            type="checkbox"
            checked={config?.auto_retrain ?? false}
            onChange={handleToggleAutoRetrain}
            className="accent-sky-500"
          />
          Auto-retrain when threshold reached
        </label>

        {/* Retrain threshold */}
        <div className="space-y-1">
          <label className="text-[10px] font-medium text-zinc-500">
            Retrain Threshold (starred generations)
          </label>
          <input
            type="number"
            min={1}
            max={100}
            value={config?.retrain_threshold ?? 10}
            onChange={(e) => {
              const v = parseInt(e.target.value);
              if (!isNaN(v) && v >= 1 && v <= 100) handleThresholdChange(v);
            }}
            className="w-full rounded border border-border bg-zinc-900 px-2 py-1 text-xs text-text-primary tabular-nums focus:border-accent focus:outline-none"
          />
        </div>

        {/* Taste refresh interval */}
        <div className="space-y-1">
          <label className="text-[10px] font-medium text-zinc-500">
            Taste Refresh Interval (generations)
          </label>
          <input
            type="number"
            min={1}
            max={50}
            value={config?.taste_refresh_interval ?? 5}
            onChange={(e) => {
              const v = parseInt(e.target.value);
              if (!isNaN(v) && v >= 1 && v <= 50) handleTasteIntervalChange(v);
            }}
            className="w-full rounded border border-border bg-zinc-900 px-2 py-1 text-xs text-text-primary tabular-nums focus:border-accent focus:outline-none"
          />
        </div>

        {/* Dataset blend slider */}
        <div className="space-y-1">
          <label className="text-[10px] font-medium text-zinc-500">
            Dataset Blend
          </label>
          <input
            type="range"
            min={0}
            max={100}
            value={blendPct}
            onChange={(e) => handleBlendChange(parseInt(e.target.value))}
            className="w-full accent-sky-500"
          />
          <p className="text-[10px] text-text-muted tabular-nums">
            {blendPct}% Library / {100 - blendPct}% Generations
          </p>
        </div>

        {/* Status line */}
        {status && (
          <div className="rounded bg-zinc-900 p-2 text-[10px] text-text-muted space-y-0.5">
            <p>
              Stars since train:{" "}
              <span className="text-yellow-400 font-medium">
                {status.stars_since_train}
              </span>
              {" / "}
              <span className="text-text-secondary">
                {status.retrain_threshold}
              </span>
            </p>
            {status.active_version != null && (
              <p>
                Active version:{" "}
                <span className="text-text-primary font-medium">
                  v{status.active_version}
                </span>
                {" of "}
                {status.total_versions}
              </p>
            )}
          </div>
        )}

        {/* Action buttons */}
        <div className="flex gap-2">
          <button
            onClick={handleRetrain}
            disabled={loading || loras.length === 0}
            className="rounded bg-sky-600 px-3 py-1.5 text-[10px] font-medium text-white hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Working..." : "Re-train Now"}
          </button>
          <button
            onClick={handleReset}
            disabled={loading || loras.length === 0}
            className="rounded border border-zinc-600 px-3 py-1.5 text-[10px] text-zinc-300 hover:border-zinc-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Reset Cache
          </button>
        </div>

        {/* Action feedback */}
        {actionMsg && (
          <p
            className={`text-[10px] ${
              actionMsg.startsWith("Error")
                ? "text-red-400"
                : "text-emerald-400"
            }`}
          >
            {actionMsg}
          </p>
        )}

        {loras.length === 0 && (
          <p className="text-[10px] text-text-muted">
            Train a LoRA adapter first to enable flywheel features.
          </p>
        )}
      </div>
    </div>
  );
}
