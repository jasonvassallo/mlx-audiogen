/** Matches server's JobStatus enum. */
export type JobStatus = "queued" | "running" | "done" | "error";

/** Matches server's GenerateRequest Pydantic model. */
export interface GenerateRequest {
  model: "musicgen" | "stable_audio";
  prompt: string;
  negative_prompt?: string;
  seconds: number;
  // MusicGen params
  temperature?: number;
  top_k?: number;
  guidance_coef?: number;
  // Stable Audio params
  steps?: number;
  cfg_scale?: number;
  sampler?: "euler" | "rk4";
  // General
  seed?: number | null;
  // Output mode
  output_mode?: "audio" | "midi" | "both";
  // Conditioning paths (MusicGen only)
  melody_path?: string | null;
  style_audio_path?: string | null;
  style_coef?: number;
}

/** Prompt analysis result from /api/suggest. */
export interface PromptAnalysis {
  genres: string[];
  moods: string[];
  instruments: string[];
  missing: string[];
  suggestions: string[];
}

/** Stem separation result from /api/separate/{id}. */
export interface StemResult {
  stems: Record<string, string>; // stem_name -> job_id
}

/** Matches server's GenerateResponse. */
export interface GenerateResponse {
  id: string;
  status: JobStatus;
}

/** Matches server's JobInfo. */
export interface JobInfo {
  id: string;
  status: JobStatus;
  model: string;
  prompt: string;
  seconds: number;
  created_at: number;
  completed_at: number | null;
  error: string | null;
  sample_rate: number | null;
  progress: number; // 0.0 to 1.0
  has_midi?: boolean;
}

/** Preset info from /api/presets listing. */
export interface PresetInfo {
  name: string;
  filename: string;
  prompt: string;
  model: string;
}

/** Matches server's ModelInfo. */
export interface ModelInfo {
  name: string;
  model_type: "musicgen" | "stable_audio";
  is_loaded: boolean;
}
