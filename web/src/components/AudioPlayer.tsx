import { useRef, useState, useEffect, useCallback } from "react";
import { getGlobalSinkId, onSinkIdChange } from "./AudioDeviceSelector";
import { useStore } from "../store/useStore";

interface AudioPlayerProps {
  src: string;
  title: string;
  autoPlay?: boolean;
  entryId: string;
  sourceBpm: number;
}

/** Decode audio blob URL into a Float32Array of mono samples for waveform display. */
async function decodeAudioBuffer(
  src: string,
): Promise<{ samples: Float32Array; sampleRate: number } | null> {
  try {
    const res = await fetch(src);
    const arrayBuf = await res.arrayBuffer();
    const ctx = new OfflineAudioContext(1, 1, 44100);
    const decoded = await ctx.decodeAudioData(arrayBuf);
    // Mix down to mono
    const mono = decoded.getChannelData(0);
    return { samples: mono, sampleRate: decoded.sampleRate };
  } catch {
    return null;
  }
}

/** Draw a static waveform on a canvas. */
function drawStaticWaveform(
  canvas: HTMLCanvasElement,
  samples: Float32Array,
  playbackProgress: number,
  zoom: number,
  scrollOffset: number,
  selectionStart: number | null,
  selectionEnd: number | null,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.scale(dpr, dpr);

  // Background
  ctx.fillStyle = "#111111";
  ctx.fillRect(0, 0, w, h);

  const totalSamples = samples.length;
  const visibleFrac = 1 / zoom;
  const startSample = Math.floor(scrollOffset * totalSamples);
  const visibleSamples = Math.min(
    Math.floor(visibleFrac * totalSamples),
    totalSamples - startSample,
  );

  if (visibleSamples <= 0) return;

  // Selection highlight
  if (selectionStart !== null && selectionEnd !== null) {
    const selStartFrac =
      (selectionStart * totalSamples - startSample) / visibleSamples;
    const selEndFrac =
      (selectionEnd * totalSamples - startSample) / visibleSamples;
    const x1 = Math.max(0, selStartFrac * w);
    const x2 = Math.min(w, selEndFrac * w);
    if (x2 > x1) {
      ctx.fillStyle = "rgba(255, 107, 53, 0.12)";
      ctx.fillRect(x1, 0, x2 - x1, h);
      // Selection borders
      ctx.strokeStyle = "rgba(255, 107, 53, 0.4)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x1, 0);
      ctx.lineTo(x1, h);
      ctx.moveTo(x2, 0);
      ctx.lineTo(x2, h);
      ctx.stroke();
    }
  }

  // Waveform — peak-based rendering
  const midY = h / 2;
  ctx.fillStyle = "rgba(255, 107, 53, 0.7)";
  for (let x = 0; x < w; x++) {
    const s0 = startSample + Math.floor((x / w) * visibleSamples);
    const s1 = startSample + Math.floor(((x + 1) / w) * visibleSamples);
    let peak = 0;
    for (let s = s0; s < s1 && s < totalSamples; s++) {
      peak = Math.max(peak, Math.abs(samples[s]!));
    }
    const barH = peak * h * 0.9;
    ctx.fillRect(x, midY - barH / 2, 1, barH || 0.5);
  }

  // Playback position line
  if (playbackProgress >= 0) {
    const posSample = playbackProgress * totalSamples;
    const posFrac = (posSample - startSample) / visibleSamples;
    if (posFrac >= 0 && posFrac <= 1) {
      ctx.strokeStyle = "#e8e8e8";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(posFrac * w, 0);
      ctx.lineTo(posFrac * w, h);
      ctx.stroke();
    }
  }
}

export default function AudioPlayer({
  src,
  title,
  autoPlay = false,
  entryId,
  sourceBpm,
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [loadError, setLoadError] = useState(false);

  // Waveform state
  const [waveformData, setWaveformData] = useState<Float32Array | null>(null);
  const [zoom, setZoom] = useState(1);
  const [scrollOffset, setScrollOffset] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [selectionStart, setSelectionStart] = useState<number | null>(null);
  const [selectionEnd, setSelectionEnd] = useState<number | null>(null);

  const settings = useStore((s) => s.settings);
  const setSourceBpm = useStore((s) => s.setSourceBpm);

  const playbackRate =
    sourceBpm > 0 && settings.masterBpm > 0
      ? settings.masterBpm / sourceBpm
      : 1.0;

  // Decode waveform on src change
  useEffect(() => {
    setWaveformData(null);
    setZoom(1);
    setScrollOffset(0);
    setSelectionStart(null);
    setSelectionEnd(null);
    let cancelled = false;
    decodeAudioBuffer(src).then((result) => {
      if (!cancelled && result) {
        setWaveformData(result.samples);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [src]);

  // Apply playback rate and pitch
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.playbackRate = playbackRate;
    type PitchAudio = HTMLAudioElement & {
      preservesPitch?: boolean;
      webkitPreservesPitch?: boolean;
    };
    const el = audio as PitchAudio;
    if (typeof el.preservesPitch !== "undefined") {
      el.preservesPitch = settings.preservePitch;
    } else if (typeof el.webkitPreservesPitch !== "undefined") {
      el.webkitPreservesPitch = settings.preservePitch;
    }
  }, [playbackRate, settings.preservePitch]);

  useEffect(() => {
    const audio = audioRef.current;
    if (audio) audio.loop = isLooping;
  }, [isLooping]);

  // Audio device
  const applySinkId = useCallback((sinkId: string) => {
    const audio = audioRef.current;
    if (!audio) return;
    if ("setSinkId" in audio) {
      (audio as HTMLAudioElement & { setSinkId: (id: string) => Promise<void> })
        .setSinkId(sinkId)
        .catch(() => {});
    }
  }, []);

  useEffect(() => {
    applySinkId(getGlobalSinkId());
    return onSinkIdChange(applySinkId);
  }, [applySinkId]);

  // Redraw waveform on state changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !waveformData) return;

    const progress = duration > 0 ? currentTime / duration : -1;
    drawStaticWaveform(
      canvas,
      waveformData,
      progress,
      zoom,
      scrollOffset,
      selectionStart,
      selectionEnd,
    );
  }, [
    waveformData,
    currentTime,
    duration,
    zoom,
    scrollOffset,
    selectionStart,
    selectionEnd,
  ]);

  // Animate during playback
  useEffect(() => {
    if (!isPlaying) return;
    const tick = () => {
      setCurrentTime(audioRef.current?.currentTime ?? 0);
      animFrameRef.current = requestAnimationFrame(tick);
    };
    animFrameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animFrameRef.current);
  }, [isPlaying]);

  // Cleanup
  useEffect(() => {
    return () => cancelAnimationFrame(animFrameRef.current);
  }, []);

  const ensureAudioContext = useCallback(() => {
    const audio = audioRef.current;
    if (!audio || audioCtxRef.current) return;
    const audioCtx = new AudioContext();
    audioCtxRef.current = audioCtx;
    const source = audioCtx.createMediaElementSource(audio);
    source.connect(audioCtx.destination);
    sourceRef.current = source;
  }, []);

  const togglePlay = async () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
    } else {
      ensureAudioContext();
      if (audioCtxRef.current?.state === "suspended") {
        await audioCtxRef.current.resume();
      }
      try {
        await audio.play();
      } catch (e) {
        console.error("Playback failed:", e);
        setLoadError(true);
      }
    }
  };

  // Click-to-seek on canvas
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) return;
    const canvas = canvasRef.current;
    const audio = audioRef.current;
    if (!canvas || !audio || duration <= 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const visibleFrac = 1 / zoom;
    const seekFrac = scrollOffset + x * visibleFrac;
    audio.currentTime = Math.max(0, Math.min(duration, seekFrac * duration));
    setCurrentTime(audio.currentTime);
    // Clear selection on single click
    setSelectionStart(null);
    setSelectionEnd(null);
  };

  // Drag-to-select on canvas
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || duration <= 0) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const visibleFrac = 1 / zoom;
    const frac = scrollOffset + x * visibleFrac;
    setSelectionStart(frac);
    setSelectionEnd(frac);
    setIsDragging(true);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDragging || selectionStart === null) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const visibleFrac = 1 / zoom;
    const frac = Math.max(0, Math.min(1, scrollOffset + x * visibleFrac));
    setSelectionEnd(frac);
  };

  const handleMouseUp = () => {
    if (isDragging && selectionStart !== null && selectionEnd !== null) {
      const diff = Math.abs(selectionEnd - selectionStart);
      if (diff < 0.005) {
        // Too small — treat as click
        setSelectionStart(null);
        setSelectionEnd(null);
      }
    }
    setIsDragging(false);
  };

  // Scroll-to-zoom (Cmd+scroll), scroll-to-pan
  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    if (e.metaKey || e.ctrlKey) {
      // Zoom
      const newZoom = Math.max(1, Math.min(32, zoom + e.deltaY * -0.02));
      setZoom(newZoom);
      // Clamp scroll offset
      const maxScroll = Math.max(0, 1 - 1 / newZoom);
      setScrollOffset((prev) => Math.min(prev, maxScroll));
    } else if (zoom > 1) {
      // Pan
      const maxScroll = 1 - 1 / zoom;
      setScrollOffset((prev) =>
        Math.max(0, Math.min(maxScroll, prev + e.deltaY * 0.001)),
      );
    }
  };

  const handleDownload = () => {
    const a = document.createElement("a");
    a.href = src;
    a.download = `${title.replace(/\s+/g, "_")}.wav`;
    a.click();
  };

  const formatTime = (t: number) => {
    const mins = Math.floor(t / 60);
    const secs = Math.floor(t % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div className="rounded border border-border bg-surface-2 p-3 space-y-2">
      <audio
        ref={audioRef}
        src={src}
        autoPlay={autoPlay}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
        onTimeUpdate={() =>
          setCurrentTime(audioRef.current?.currentTime ?? 0)
        }
        onLoadedMetadata={() =>
          setDuration(audioRef.current?.duration ?? 0)
        }
        onError={() => setLoadError(true)}
      />

      {/* Waveform canvas — click to seek, drag to select, scroll to zoom/pan */}
      <canvas
        ref={canvasRef}
        className="w-full h-14 rounded bg-surface-1 cursor-crosshair"
        onClick={handleCanvasClick}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />

      {/* Zoom indicator */}
      {zoom > 1 && (
        <div className="flex items-center justify-between text-[10px] text-text-muted">
          <span>
            {zoom.toFixed(1)}x zoom — ⌘+scroll to zoom, scroll to pan
          </span>
          <button
            onClick={() => {
              setZoom(1);
              setScrollOffset(0);
            }}
            className="text-accent hover:text-accent-hover"
          >
            Reset
          </button>
        </div>
      )}

      {loadError && (
        <div className="text-xs text-error">
          Failed to load audio. The file may be corrupted.
        </div>
      )}

      {/* Transport controls */}
      <div className="flex items-center gap-2">
        {/* Play/Pause */}
        <button
          onClick={togglePlay}
          disabled={loadError}
          className="
            flex h-8 w-8 items-center justify-center rounded-full
            bg-accent text-surface-0 hover:bg-accent-hover
            transition-colors shrink-0
            disabled:opacity-50 disabled:cursor-not-allowed
          "
        >
          {isPlaying ? (
            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
              <rect x="1" y="1" width="4" height="10" rx="1" />
              <rect x="7" y="1" width="4" height="10" rx="1" />
            </svg>
          ) : (
            <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
              <polygon points="2,0 12,6 2,12" />
            </svg>
          )}
        </button>

        {/* Loop toggle */}
        <button
          onClick={() => setIsLooping(!isLooping)}
          className={`
            flex h-7 w-7 items-center justify-center rounded
            transition-colors text-xs shrink-0
            ${
              isLooping
                ? "bg-accent/20 text-accent border border-accent/40"
                : "text-text-muted hover:text-text-secondary border border-transparent"
            }
          `}
          title={isLooping ? "Loop ON" : "Loop OFF"}
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M17 2l4 4-4 4" />
            <path d="M3 11V9a4 4 0 014-4h14" />
            <path d="M7 22l-4-4 4-4" />
            <path d="M21 13v2a4 4 0 01-4 4H3" />
          </svg>
        </button>

        {/* Time display */}
        <div className="flex-1 text-xs tabular-nums text-text-secondary">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>

        {/* Source BPM input */}
        <div className="flex items-center gap-1">
          <label className="text-xs text-text-muted">BPM</label>
          <input
            type="number"
            value={sourceBpm || ""}
            onChange={(e) => {
              const val = parseInt(e.target.value) || 0;
              setSourceBpm(entryId, Math.max(0, Math.min(300, val)));
            }}
            placeholder="?"
            className="
              w-12 rounded border border-border bg-surface-1 px-1 py-0.5
              text-xs tabular-nums text-text-primary text-center
              focus:border-accent focus:outline-none
              placeholder:text-text-muted
            "
          />
        </div>

        {/* Rate display */}
        {sourceBpm > 0 && settings.masterBpm > 0 && (
          <span className="text-xs tabular-nums text-accent">
            {playbackRate.toFixed(2)}x
          </span>
        )}

        {/* Download */}
        <button
          onClick={handleDownload}
          className="text-text-secondary hover:text-accent transition-colors shrink-0"
          title="Download WAV"
        >
          <svg
            width="14"
            height="14"
            viewBox="0 0 14 14"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
          >
            <path d="M7 1v8M3 6l4 4 4-4M2 12h10" />
          </svg>
        </button>
      </div>
    </div>
  );
}
