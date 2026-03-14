import { useCallback, useEffect, useRef, useState } from "react";
import { useStore } from "../store/useStore";
import {
  startTraining,
  fetchTrainStatus,
  stopTraining,
} from "../api/client";
import type { TrainStatus } from "../types/api";

type Profile = "quick" | "balanced" | "deep";

const PROFILES: Record<Profile, { label: string; desc: string }> = {
  quick: { label: "Quick & Light", desc: "rank 8, q+v only" },
  balanced: { label: "Balanced", desc: "rank 16, q+v+out" },
  deep: { label: "Deep", desc: "rank 32, all projections" },
};

export default function TrainPanel() {
  const models = useStore((s) => s.models);
  const fetchLoras = useStore((s) => s.fetchLoras);

  // Form state
  const [dataDir, setDataDir] = useState("");
  const [name, setName] = useState("");
  const [baseModel, setBaseModel] = useState("musicgen-small");
  const [profile, setProfile] = useState<Profile>("balanced");
  const [chunkSeconds, setChunkSeconds] = useState(10);
  const [epochs, setEpochs] = useState(10);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Training state
  const [trainId, setTrainId] = useState<string | null>(null);
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Filter to MusicGen models only
  const musicgenModels = models.filter((m) => m.model_type === "musicgen");

  // Validate name
  const nameValid = /^[a-zA-Z0-9_-]{1,64}$/.test(name);

  // Poll training status
  useEffect(() => {
    if (!trainId) return;

    const poll = async () => {
      try {
        const s = await fetchTrainStatus(trainId);
        setStatus(s);
        if (s.loss > 0) {
          setLossHistory((prev) => [...prev, s.loss]);
        }
        // Check if done (progress >= 1 or no longer found)
        if (s.progress >= 1.0) {
          clearInterval(pollRef.current!);
          pollRef.current = null;
          setTrainId(null);
          fetchLoras(); // Refresh LoRA list
        }
      } catch {
        // Training finished or stopped
        clearInterval(pollRef.current!);
        pollRef.current = null;
        setTrainId(null);
        fetchLoras();
      }
    };

    pollRef.current = setInterval(poll, 1000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [trainId, fetchLoras]);

  // Draw loss chart
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || lossHistory.length < 2) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, w, h);

    const maxLoss = Math.max(...lossHistory);
    const minLoss = Math.min(...lossHistory);
    const range = maxLoss - minLoss || 1;

    ctx.beginPath();
    ctx.strokeStyle = "#38bdf8"; // sky-400
    ctx.lineWidth = 1.5;

    lossHistory.forEach((loss, i) => {
      const x = (i / (lossHistory.length - 1)) * w;
      const y = h - ((loss - minLoss) / range) * (h - 8) - 4;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [lossHistory]);

  const handleStart = useCallback(async () => {
    if (!dataDir || !name || !nameValid) return;

    setError(null);
    setLossHistory([]);
    setStatus(null);

    try {
      const res = await startTraining({
        data_dir: dataDir,
        base_model: baseModel,
        name,
        profile,
        chunk_seconds: chunkSeconds,
        epochs,
      });
      setTrainId(res.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start training");
    }
  }, [dataDir, name, nameValid, baseModel, profile, chunkSeconds, epochs]);

  const handleStop = useCallback(async () => {
    if (!trainId) return;
    try {
      await stopTraining(trainId);
    } catch {
      // Already stopped
    }
  }, [trainId]);

  const isTraining = trainId !== null;

  return (
    <div className="space-y-4">
      <h3 className="text-xs font-medium uppercase tracking-wider text-zinc-400">
        LoRA Training
      </h3>

      {/* Data directory */}
      <div className="space-y-1">
        <label className="text-xs text-zinc-400">Data Directory</label>
        <input
          type="text"
          value={dataDir}
          onChange={(e) => setDataDir(e.target.value)}
          placeholder="/path/to/audio/files/"
          disabled={isTraining}
          className="w-full rounded bg-zinc-800 px-2 py-1.5 text-sm text-zinc-200 border border-zinc-700 focus:border-sky-500 focus:outline-none placeholder:text-zinc-600 disabled:opacity-50"
        />
        <p className="text-[10px] text-zinc-500">
          WAV/MP3/FLAC files + optional metadata.jsonl
        </p>
      </div>

      {/* Name */}
      <div className="space-y-1">
        <label className="text-xs text-zinc-400">Adapter Name</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="my-style"
          disabled={isTraining}
          className={`w-full rounded bg-zinc-800 px-2 py-1.5 text-sm text-zinc-200 border focus:outline-none placeholder:text-zinc-600 disabled:opacity-50 ${
            name && !nameValid
              ? "border-red-500"
              : "border-zinc-700 focus:border-sky-500"
          }`}
        />
        {name && !nameValid && (
          <p className="text-[10px] text-red-400">
            Alphanumeric, hyphens, underscores only (1-64 chars)
          </p>
        )}
      </div>

      {/* Base model */}
      <div className="space-y-1">
        <label className="text-xs text-zinc-400">Base Model</label>
        <select
          value={baseModel}
          onChange={(e) => setBaseModel(e.target.value)}
          disabled={isTraining}
          className="w-full rounded bg-zinc-800 px-2 py-1.5 text-sm text-zinc-200 border border-zinc-700 focus:border-sky-500 focus:outline-none disabled:opacity-50"
        >
          {musicgenModels.length > 0 ? (
            musicgenModels.map((m) => (
              <option key={m.name} value={m.name}>
                {m.name}
              </option>
            ))
          ) : (
            <option value="musicgen-small">musicgen-small</option>
          )}
        </select>
      </div>

      {/* Profile cards */}
      <div className="space-y-1">
        <label className="text-xs text-zinc-400">Training Profile</label>
        <div className="grid grid-cols-3 gap-1.5">
          {(Object.entries(PROFILES) as [Profile, { label: string; desc: string }][]).map(
            ([key, { label, desc }]) => (
              <button
                key={key}
                onClick={() => setProfile(key)}
                disabled={isTraining}
                className={`rounded border px-2 py-2 text-center transition-colors disabled:opacity-50 ${
                  profile === key
                    ? "border-sky-500 bg-sky-500/10 text-sky-400"
                    : "border-zinc-700 bg-zinc-800 text-zinc-400 hover:border-zinc-600"
                }`}
              >
                <div className="text-xs font-medium">{label}</div>
                <div className="text-[10px] text-zinc-500">{desc}</div>
              </button>
            ),
          )}
        </div>
      </div>

      {/* Chunk duration */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <label className="text-xs text-zinc-400">Chunk Duration</label>
          <span className="text-xs tabular-nums text-zinc-500">{chunkSeconds}s</span>
        </div>
        <input
          type="range"
          min={5}
          max={40}
          step={1}
          value={chunkSeconds}
          onChange={(e) => setChunkSeconds(Number(e.target.value))}
          disabled={isTraining}
          className="w-full accent-sky-500"
        />
      </div>

      {/* Epochs */}
      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <label className="text-xs text-zinc-400">Epochs</label>
          <span className="text-xs tabular-nums text-zinc-500">{epochs}</span>
        </div>
        <input
          type="range"
          min={1}
          max={50}
          step={1}
          value={epochs}
          onChange={(e) => setEpochs(Number(e.target.value))}
          disabled={isTraining}
          className="w-full accent-sky-500"
        />
      </div>

      {/* Advanced toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="text-xs text-zinc-500 hover:text-zinc-300"
      >
        {showAdvanced ? "▾ Hide advanced" : "▸ Show advanced"}
      </button>

      {showAdvanced && (
        <div className="space-y-2 rounded border border-zinc-700/50 bg-zinc-800/50 p-3 text-[10px] text-zinc-500">
          <p>
            Advanced settings (rank, alpha, targets, learning rate) are
            available via the CLI:
          </p>
          <code className="block rounded bg-zinc-900 p-2 text-[10px] text-zinc-400">
            mlx-audiogen-train --rank 32 --alpha 64 --targets q_proj,v_proj
          </code>
        </div>
      )}

      {/* Start / Stop */}
      <div className="pt-2">
        {isTraining ? (
          <button
            onClick={handleStop}
            className="w-full rounded bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-500"
          >
            Stop Training
          </button>
        ) : (
          <button
            onClick={handleStart}
            disabled={!dataDir || !name || !nameValid}
            className="w-full rounded bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Start Training
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <p className="text-xs text-red-400">{error}</p>
      )}

      {/* Progress */}
      {status && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-zinc-400">
            <span>
              Epoch {status.epoch + 1}/{status.total_epochs}
            </span>
            <span>Loss: {status.loss.toFixed(4)}</span>
          </div>
          <div className="h-2 rounded-full bg-zinc-800">
            <div
              className="h-full rounded-full bg-sky-500 transition-all"
              style={{ width: `${Math.min(status.progress * 100, 100)}%` }}
            />
          </div>
          {status.best_loss !== null && (
            <p className="text-[10px] text-zinc-500">
              Best loss: {status.best_loss.toFixed(4)}
            </p>
          )}

          {/* Loss chart */}
          {lossHistory.length >= 2 && (
            <div className="space-y-1">
              <label className="text-[10px] text-zinc-500">Loss curve</label>
              <canvas
                ref={canvasRef}
                className="h-16 w-full rounded border border-zinc-700/50 bg-zinc-900"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
