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
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [isLooping, setIsLooping] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [loadError, setLoadError] = useState(false);

  const settings = useStore((s) => s.settings);
  const setSourceBpm = useStore((s) => s.setSourceBpm);

  // Calculate playback rate from BPM ratio
  const playbackRate =
    sourceBpm > 0 && settings.masterBpm > 0
      ? settings.masterBpm / sourceBpm
      : 1.0;

  // Apply playback rate and preservesPitch whenever they change
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

  // Apply loop state
  useEffect(() => {
    const audio = audioRef.current;
    if (audio) audio.loop = isLooping;
  }, [isLooping]);

  // Apply audio output device (setSinkId)
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

  /**
   * Lazily initialize Web Audio API on first user interaction.
   * This avoids the browser's autoplay policy blocking the AudioContext.
   */
  const ensureAudioContext = useCallback(() => {
    const audio = audioRef.current;
    if (!audio || audioCtxRef.current) return;

    const audioCtx = new AudioContext();
    audioCtxRef.current = audioCtx;

    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    analyserRef.current = analyser;

    const source = audioCtx.createMediaElementSource(audio);
    source.connect(analyser);
    analyser.connect(audioCtx.destination);
    sourceRef.current = source;
  }, []);

  const drawWaveform = useCallback(() => {
    const analyser = analyserRef.current;
    const canvas = canvasRef.current;
    if (!analyser || !canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animFrameRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      const { width, height } = canvas;
      ctx.fillStyle = "#111111";
      ctx.fillRect(0, 0, width, height);

      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#ff6b35";
      ctx.beginPath();

      const sliceWidth = width / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i]! / 128.0;
        const y = (v * height) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(width, height / 2);
      ctx.stroke();
    };

    draw();
  }, []);

  // Cleanup animation frame on unmount
  useEffect(() => {
    return () => {
      cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  useEffect(() => {
    if (isPlaying) {
      drawWaveform();
    } else {
      cancelAnimationFrame(animFrameRef.current);
    }
  }, [isPlaying, drawWaveform]);

  const togglePlay = async () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      // Initialize audio context on first play (requires user gesture)
      ensureAudioContext();

      // Resume AudioContext if suspended (browser autoplay policy)
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

      {/* Waveform canvas */}
      <canvas
        ref={canvasRef}
        width={400}
        height={48}
        className="w-full h-12 rounded bg-surface-1"
      />

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
