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
  // Conditioning paths (MusicGen only)
  melody_path?: string | null;
  style_audio_path?: string | null;
  style_coef?: number;
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
}

/** Matches server's ModelInfo. */
export interface ModelInfo {
  name: string;
  model_type: "musicgen" | "stable_audio";
  is_loaded: boolean;
}
