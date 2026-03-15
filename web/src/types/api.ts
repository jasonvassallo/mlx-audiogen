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
  // Audio-to-audio (Stable Audio only)
  reference_audio_path?: string | null;
  reference_strength?: number;
  // LoRA adapter (MusicGen only)
  lora?: string | null;
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
  starred?: boolean;
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

// ---------------------------------------------------------------------------
// Phase 7b: LLM Enhancement, Memory, Settings
// ---------------------------------------------------------------------------

/** Analysis tags from prompt analysis. */
export interface AnalysisTags {
  genres: string[];
  moods: string[];
  instruments: string[];
  missing: string[];
}

/** Response from POST /api/enhance. */
export interface EnhanceResponse {
  original: string;
  enhanced: string;
  analysis_tags: AnalysisTags;
  used_llm: boolean;
  warning: string | null;
}

/** LLM model info from GET /api/llm/models. */
export interface LLMModelInfo {
  id: string;
  name: string;
  size_gb: number;
  source: "huggingface" | "lmstudio";
}

/** Server-side settings from GET /api/settings. */
export interface ServerSettings {
  llm_model: string | null;
  ai_enhance: boolean;
  history_context_count: number;
}

/** LLM status from GET /api/llm/status. */
export interface LLMStatus {
  model_id: string | null;
  loaded: boolean;
  idle_seconds: number;
  memory_mb: number;
}

/** Tag database from GET /api/tags. */
export type TagDatabase = Record<string, string[]>;

/** Prompt memory from GET /api/memory. */
export interface PromptMemoryData {
  history: Array<{
    prompt: string;
    enhanced_prompt?: string;
    model: string;
    params: Record<string, unknown>;
    timestamp: string;
  }>;
  style_profile: {
    top_genres: string[];
    top_moods: string[];
    top_instruments: string[];
    preferred_duration: number;
    generation_count: number;
  };
}

// ---------------------------------------------------------------------------
// Phase 9g: LoRA Fine-Tuning
// ---------------------------------------------------------------------------

/** LoRA adapter info from GET /api/loras. */
export interface LoRAInfo {
  name: string;
  base_model: string;
  profile: string | null;
  rank: number;
  alpha: number;
  hidden_size: number;
  final_loss: number | null;
  best_loss: number | null;
  training_samples: number | null;
  created_at: string | null;
  active_version?: number;
  total_versions?: number;
  stars_since_train?: number;
}

/** LoRA training request for POST /api/train. */
export interface TrainRequest {
  data_dir?: string;
  collection?: string;
  base_model: string;
  name: string;
  profile?: string;
  rank?: number;
  alpha?: number;
  targets?: string[];
  epochs?: number;
  learning_rate?: number;
  batch_size?: number;
  chunk_seconds?: number;
  early_stop?: boolean;
  patience?: number;
}

/** Training status from GET /api/train/status/{id}. */
export interface TrainStatus {
  epoch: number;
  total_epochs: number;
  step: number;
  steps_per_epoch: number;
  loss: number;
  best_loss: number | null;
  progress: number;
}

// ---------------------------------------------------------------------------
// Phase 9g-4: Flywheel Intelligence
// ---------------------------------------------------------------------------

/** Flywheel configuration. */
export interface FlywheelConfig {
  retrain_threshold: number;
  blend_ratio: number;
  taste_refresh_interval: number;
  auto_retrain: boolean;
  base_collection: string | null;
}

/** Response from star/unstar generation. */
export interface StarResponse {
  starred: boolean;
  stars_since_train: number;
}

/** Flywheel status overview. */
export interface FlywheelStatus {
  stars_since_train: number;
  retrain_threshold: number;
  auto_retrain: boolean;
  active_version: number | null;
  total_versions: number;
}

/** A single tag influence entry. */
export interface TagInfluence {
  tag: string;
  pct: number;
}

/** Top influences by category. */
export interface TopInfluences {
  genre: TagInfluence[];
  mood: TagInfluence[];
  instrument: TagInfluence[];
}

/** Dataset composition for a version. */
export interface DatasetComposition {
  library_tracks: number;
  kept_generations: number;
  blend_ratio: { library: number; generations: number };
  total_training_samples: number;
}

/** Changes since parent version. */
export interface NewSinceParent {
  kept_generations_added: number;
  enrichment_tags_added: number;
  library_tracks_added: number;
}

/** Training stats for a version. */
export interface TrainingStats {
  profile: string;
  epochs: number;
  best_loss: number;
  duration_seconds: number;
}

/** Summary of a LoRA version. */
export interface VersionSummary {
  version: number;
  created_at: string;
  is_active: boolean;
  library_tracks: number;
  kept_generations: number;
  best_loss: number;
}

/** Full changelog for a LoRA version. */
export interface VersionChangelog {
  version: number;
  created_at: string;
  parent_version: number | null;
  dataset: DatasetComposition;
  top_influences: TopInfluences;
  new_since_parent: NewSinceParent;
  training: TrainingStats;
}

// ---------------------------------------------------------------------------
// Phase 9g-2: Library Scanner
// ---------------------------------------------------------------------------

/** Track metadata from a music library source. */
export interface LibraryTrackInfo {
  track_id: string;
  title: string;
  artist: string;
  album: string;
  genre: string;
  bpm: number | null;
  key: string | null;
  year: number | null;
  rating: number | null;
  play_count: number;
  duration_seconds: number;
  comments: string;
  file_path: string | null;
  file_available: boolean;
  source: string;
  loved: boolean;
  description: string;
  description_edited: boolean;
  enrichment_status?: EnrichmentStatus;
}

/** Playlist from a library source. */
export interface PlaylistInfo {
  id: string;
  name: string;
  track_count: number;
  track_ids: string[];
  source: string;
}

/** A configured music library source (Apple Music or rekordbox XML). */
export interface LibrarySource {
  id: string;
  type: "apple_music" | "rekordbox";
  path: string;
  label: string;
  track_count: number | null;
  playlist_count: number | null;
  last_loaded: string | null;
}

/** Summary of a saved collection (returned from list endpoint). */
export interface CollectionSummary {
  name: string;
  track_count: number;
  source: string;
  playlist: string;
  created_at: string | null;
  updated_at: string | null;
}

/** Full collection with track data. */
export interface CollectionFull {
  name: string;
  created_at: string;
  updated_at: string;
  source: string;
  playlist: string;
  tracks: LibraryTrackInfo[];
}

/** Paginated track search result from GET /api/library/tracks/{id}. */
export interface TrackSearchResult {
  tracks: LibraryTrackInfo[];
  count: number;
  offset: number;
  limit: number;
}

/** Analysis + generated prompt from POST /api/library/generate-prompt. */
export interface PlaylistAnalysis {
  bpm_median: number | null;
  bpm_range: [number, number] | null;
  top_keys: string[];
  top_genres: string[];
  top_artists: string[];
  year_range: [number, number] | null;
  track_count: number;
  available_count: number;
  prompt: string;
}

/** Track search/filter query parameters. */
export interface LibrarySearchParams {
  q?: string;
  artist?: string;
  album?: string;
  genre?: string;
  key?: string;
  bpm_min?: number;
  bpm_max?: number;
  year_min?: number;
  year_max?: number;
  rating_min?: number;
  loved?: boolean;
  available?: boolean;
  sort?: string;
  order?: "asc" | "desc";
  offset?: number;
  limit?: number;
}

// ---------------------------------------------------------------------------
// Phase 9g-3: Enrichment + Taste
// ---------------------------------------------------------------------------

/** Enrichment status for a track. */
export type EnrichmentStatus = "none" | "partial" | "complete" | "stale";

/** Credential service status. */
export interface CredentialStatus {
  musicbrainz: boolean;
  lastfm: boolean;
  discogs: boolean;
}

/** Enrichment job progress. */
export interface EnrichmentJobStatus {
  job_id?: string;
  status: "idle" | "running" | "done" | "error" | "cancelled";
  total: number;
  completed: number;
  errors: number;
  current_track?: string | null;
}

/** Enrichment statistics. */
export interface EnrichmentStats {
  total_tracks: number;
  musicbrainz_count: number;
  lastfm_count: number;
  discogs_count: number;
}

/** Weighted tag for taste profile. */
export interface WeightedTag {
  name: string;
  weight: number;
}

/** User taste profile. */
export interface TasteProfile {
  top_genres: WeightedTag[];
  top_artists: WeightedTag[];
  bpm_range: [number, number];
  key_preferences: WeightedTag[];
  era_distribution: Record<string, number>;
  mood_profile: WeightedTag[];
  style_tags: WeightedTag[];
  gen_genres: WeightedTag[];
  gen_moods: WeightedTag[];
  gen_instruments: WeightedTag[];
  kept_ratio: number;
  avg_duration: number;
  preferred_models: string[];
  library_track_count: number;
  generation_count: number;
  last_updated: string;
  version: number;
  overrides: string;
}
