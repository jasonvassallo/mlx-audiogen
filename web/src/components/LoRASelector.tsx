import { useEffect } from "react";
import { useStore } from "../store/useStore";

/** Dropdown for selecting a LoRA adapter in the Generate tab. */
export default function LoRASelector() {
  const loras = useStore((s) => s.loras);
  const selectedLora = useStore((s) => s.selectedLora);
  const fetchLoras = useStore((s) => s.fetchLoras);
  const setSelectedLora = useStore((s) => s.setSelectedLora);
  const params = useStore((s) => s.params);

  useEffect(() => {
    fetchLoras();
  }, [fetchLoras]);

  // Warn if selected LoRA's base model doesn't match current model
  const selectedInfo = loras.find((l) => l.name === selectedLora);
  const mismatch =
    selectedInfo &&
    params.model === "musicgen" &&
    selectedInfo.base_model &&
    !selectedInfo.base_model.includes("musicgen");

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-xs text-zinc-400">LoRA Adapter</label>
        <button
          onClick={fetchLoras}
          className="text-xs text-zinc-500 hover:text-zinc-300"
          title="Refresh LoRA list"
        >
          ↻
        </button>
      </div>
      <select
        value={selectedLora ?? ""}
        onChange={(e) =>
          setSelectedLora(e.target.value === "" ? null : e.target.value)
        }
        className="w-full rounded bg-zinc-800 px-2 py-1.5 text-sm text-zinc-200 border border-zinc-700 focus:border-sky-500 focus:outline-none"
      >
        <option value="">None</option>
        {loras.map((l) => (
          <option key={l.name} value={l.name}>
            {l.name} (rank {l.rank})
          </option>
        ))}
      </select>
      {mismatch && (
        <p className="text-xs text-amber-400">
          ⚠ This LoRA was trained on {selectedInfo.base_model}, which may not
          match the current model.
        </p>
      )}
      {params.model === "stable_audio" && selectedLora && (
        <p className="text-xs text-amber-400">
          ⚠ LoRA is only supported with MusicGen models.
        </p>
      )}
    </div>
  );
}
