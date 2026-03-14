import { useStore } from "../store/useStore";
import ParamSlider from "./ParamSlider";
import DurationControl from "./DurationControl";
import LoRASelector from "./LoRASelector";

export default function ParameterPanel() {
  const params = useStore((s) => s.params);
  const setParam = useStore((s) => s.setParam);
  const isGenerating = useStore((s) => s.isGenerating);

  return (
    <div className="space-y-4">
      <h3 className="text-xs font-medium uppercase tracking-wider text-text-secondary">
        Parameters
      </h3>

      {/* Duration — supports both seconds and BPM-based bars */}
      <DurationControl />

      {/* Seed control */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <label className="text-xs font-medium text-text-secondary">
            Seed
          </label>
          <button
            onClick={() =>
              setParam(
                "seed",
                params.seed === null
                  ? Math.floor(Math.random() * 100000)
                  : null,
              )
            }
            disabled={isGenerating}
            className="text-xs text-accent hover:text-accent-hover disabled:opacity-50"
          >
            {params.seed === null ? "Pin seed" : "Randomize"}
          </button>
        </div>
        {params.seed !== null && (
          <input
            type="number"
            value={params.seed}
            onChange={(e) =>
              setParam("seed", parseInt(e.target.value) || 0)
            }
            disabled={isGenerating}
            className="
              w-full rounded border border-border bg-surface-2 px-2 py-1
              text-xs tabular-nums text-text-primary
              focus:border-accent focus:outline-none
              disabled:opacity-50
            "
          />
        )}
      </div>

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

      {/* LoRA selector (MusicGen only) */}
      {params.model === "musicgen" && <LoRASelector />}

      <div className="border-t border-border pt-3">
        {params.model === "musicgen" ? (
          <MusicGenParams />
        ) : (
          <StableAudioParams />
        )}
      </div>
    </div>
  );
}

function MusicGenParams() {
  const params = useStore((s) => s.params);
  const setParam = useStore((s) => s.setParam);
  const isGenerating = useStore((s) => s.isGenerating);

  return (
    <div className="space-y-3">
      <div className="text-xs font-medium text-text-muted mb-2">MusicGen</div>

      <ParamSlider
        label="Temperature"
        value={params.temperature ?? 1.0}
        onChange={(v) => setParam("temperature", v)}
        min={0.1}
        max={2.0}
        step={0.05}
        disabled={isGenerating}
      />

      <ParamSlider
        label="Top K"
        value={params.top_k ?? 250}
        onChange={(v) => setParam("top_k", Math.round(v))}
        min={1}
        max={500}
        step={1}
        disabled={isGenerating}
      />

      <ParamSlider
        label="Guidance"
        value={params.guidance_coef ?? 3.0}
        onChange={(v) => setParam("guidance_coef", v)}
        min={0}
        max={10}
        step={0.1}
        disabled={isGenerating}
      />

      <ParamSlider
        label="Style Coef"
        value={params.style_coef ?? 5.0}
        onChange={(v) => setParam("style_coef", v)}
        min={0}
        max={20}
        step={0.5}
        disabled={isGenerating}
      />
    </div>
  );
}

function StableAudioParams() {
  const params = useStore((s) => s.params);
  const setParam = useStore((s) => s.setParam);
  const isGenerating = useStore((s) => s.isGenerating);

  return (
    <div className="space-y-3">
      <div className="text-xs font-medium text-text-muted mb-2">
        Stable Audio
      </div>

      <ParamSlider
        label="Steps"
        value={params.steps ?? 8}
        onChange={(v) => setParam("steps", Math.round(v))}
        min={1}
        max={100}
        step={1}
        disabled={isGenerating}
      />

      <ParamSlider
        label="CFG Scale"
        value={params.cfg_scale ?? 6.0}
        onChange={(v) => setParam("cfg_scale", v)}
        min={0}
        max={15}
        step={0.1}
        disabled={isGenerating}
      />

      {/* Reference Strength (audio-to-audio) */}
      <ParamSlider
        label="Ref. Strength"
        value={params.reference_strength ?? 0.7}
        onChange={(v) => setParam("reference_strength", v)}
        min={0}
        max={1}
        step={0.05}
        disabled={isGenerating}
      />

      {/* Sampler dropdown */}
      <div className="space-y-1">
        <label className="text-xs font-medium text-text-secondary">
          Sampler
        </label>
        <select
          value={params.sampler ?? "euler"}
          onChange={(e) =>
            setParam("sampler", e.target.value as "euler" | "rk4")
          }
          disabled={isGenerating}
          className="
            w-full rounded border border-border bg-surface-2 px-2 py-1.5
            text-xs text-text-primary
            focus:border-accent focus:outline-none
            disabled:opacity-50
          "
        >
          <option value="euler">Euler (fast)</option>
          <option value="rk4">RK4 (accurate)</option>
        </select>
      </div>
    </div>
  );
}
