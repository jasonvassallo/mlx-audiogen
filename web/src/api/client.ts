import type {
  CollectionFull,
  CollectionSummary,
  CredentialStatus,
  EnhanceResponse,
  EnrichmentJobStatus,
  EnrichmentStats,
  FlywheelConfig,
  FlywheelStatus,
  GenerateRequest,
  GenerateResponse,
  JobInfo,
  LibrarySearchParams,
  LibrarySource,
  LLMModelInfo,
  LLMStatus,
  LoRAInfo,
  ModelInfo,
  PlaylistAnalysis,
  PlaylistInfo,
  PresetInfo,
  PromptAnalysis,
  PromptMemoryData,
  ServerSettings,
  StarResponse,
  StemResult,
  TagDatabase,
  TasteProfile,
  TrackSearchResult,
  TrainRequest,
  TrainStatus,
  VersionChangelog,
  VersionSummary,
} from "../types/api";

/**
 * API base URL management.
 *
 * Default: "/api" (works with Vite proxy in dev and FastAPI static serving in prod).
 * Remote: "http://host:port/api" when the user configures a remote server URL.
 *
 * The server URL (without /api suffix) is persisted in localStorage so it
 * survives page refreshes. The module-level `_base` variable is used for
 * fast access by all fetch functions.
 */
const SERVER_URL_KEY = "mlx_audiogen_server_url";

function resolveBase(serverUrl: string): string {
  if (!serverUrl) return "/api";
  return serverUrl.replace(/\/+$/, "") + "/api";
}

let _serverUrl = localStorage.getItem(SERVER_URL_KEY) || "";
let _base = resolveBase(_serverUrl);

/** Set the remote server URL. Pass empty string to revert to local. */
export function setServerUrl(url: string): void {
  _serverUrl = url;
  _base = resolveBase(url);
  if (url) {
    localStorage.setItem(SERVER_URL_KEY, url);
  } else {
    localStorage.removeItem(SERVER_URL_KEY);
  }
}

/** Get the current server URL (without /api suffix). Empty = local. */
export function getServerUrl(): string {
  return _serverUrl;
}

/** Get the resolved API base URL (e.g. "/api" or "http://host:port/api"). */
export function getApiBase(): string {
  return _base;
}

async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const res = await fetch(`${_base}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

/** List available models and their loading status. */
export function fetchModels(): Promise<ModelInfo[]> {
  return request<ModelInfo[]>("/models");
}

/** Submit a generation request. Returns immediately with a job ID. */
export function submitGeneration(
  req: GenerateRequest,
): Promise<GenerateResponse> {
  return request<GenerateResponse>("/generate", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

/** Poll job status. */
export function fetchJobStatus(jobId: string): Promise<JobInfo> {
  return request<JobInfo>(`/status/${jobId}`);
}

/** Get the URL for downloading generated audio in the specified format. */
export function getAudioUrl(jobId: string, fmt: string = "aiff"): string {
  return `${_base}/audio/${jobId}?fmt=${fmt}`;
}

/** Get the URL for downloading generated MIDI. */
export function getMidiUrl(jobId: string): string {
  return `${_base}/midi/${jobId}`;
}

/** Get AI prompt suggestions. */
export function suggestPrompts(
  prompt: string,
  count = 4,
): Promise<PromptAnalysis> {
  return request<PromptAnalysis>("/suggest", {
    method: "POST",
    body: JSON.stringify({ prompt, count }),
  });
}

/** Separate a job's audio into stems. */
export function separateStems(jobId: string): Promise<StemResult> {
  return request<StemResult>(`/separate/${jobId}`, {
    method: "POST",
  });
}

/** List all saved presets. */
export function fetchPresets(): Promise<PresetInfo[]> {
  return request<PresetInfo[]>("/presets");
}

/** Save current params as a named preset. */
export function savePreset(
  name: string,
  params: GenerateRequest,
): Promise<{ saved: string }> {
  return request<{ saved: string }>(`/presets/${encodeURIComponent(name)}`, {
    method: "POST",
    body: JSON.stringify(params),
  });
}

/** Load a preset by name. */
export function loadPreset(name: string): Promise<GenerateRequest> {
  return request<GenerateRequest>(`/presets/${encodeURIComponent(name)}`);
}

// ---------------------------------------------------------------------------
// Phase 7b: LLM Enhancement, Memory, Settings
// ---------------------------------------------------------------------------

/** Enhance a prompt via LLM or template fallback. */
export function enhancePrompt(
  prompt: string,
  includeMemory = true,
): Promise<EnhanceResponse> {
  return request<EnhanceResponse>("/enhance", {
    method: "POST",
    body: JSON.stringify({ prompt, include_memory: includeMemory }),
  });
}

/** Get the tag database for autocomplete. */
export function fetchTags(): Promise<TagDatabase> {
  return request<TagDatabase>("/tags");
}

/** List discovered LLM models. */
export function fetchLLMModels(): Promise<LLMModelInfo[]> {
  return request<LLMModelInfo[]>("/llm/models");
}

/** Select an LLM model. */
export function selectLLMModel(
  modelId: string,
): Promise<{ status: string }> {
  return request<{ status: string }>("/llm/select", {
    method: "POST",
    body: JSON.stringify({ model_id: modelId }),
  });
}

/** Get LLM status. */
export function fetchLLMStatus(): Promise<LLMStatus> {
  return request<LLMStatus>("/llm/status");
}

/** Get prompt memory. */
export function fetchMemory(): Promise<PromptMemoryData> {
  return request<PromptMemoryData>("/memory");
}

/** Clear prompt memory. */
export function clearMemory(): Promise<{ status: string }> {
  return request<{ status: string }>("/memory", { method: "DELETE" });
}

/** Export prompt memory as downloadable JSON. */
export function getMemoryExportUrl(): string {
  return `${_base}/memory/export`;
}

/** Import prompt memory from file. */
export async function importMemory(
  file: File,
): Promise<{ status: string }> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${_base}/memory/import`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

/** Get server settings. */
export function fetchServerSettings(): Promise<ServerSettings> {
  return request<ServerSettings>("/settings");
}

/** Update server settings. */
export function updateServerSettings(
  settings: Partial<ServerSettings>,
): Promise<ServerSettings> {
  return request<ServerSettings>("/settings", {
    method: "POST",
    body: JSON.stringify(settings),
  });
}

// ---------------------------------------------------------------------------
// Phase 9g: LoRA Fine-Tuning
// ---------------------------------------------------------------------------

/** List available LoRA adapters. */
export function fetchLoras(): Promise<LoRAInfo[]> {
  return request<LoRAInfo[]>("/loras");
}

/** Delete a LoRA adapter. */
export function deleteLora(name: string): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(
    `/loras/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  );
}

/** Start LoRA training. */
export function startTraining(
  req: TrainRequest,
): Promise<{ id: string }> {
  return request<{ id: string }>("/train", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

/** Get training status. */
export function fetchTrainStatus(id: string): Promise<TrainStatus> {
  return request<TrainStatus>(`/train/status/${id}`);
}

/** Stop an active training job. */
export function stopTraining(id: string): Promise<{ stopped: string }> {
  return request<{ stopped: string }>(`/train/stop/${id}`, {
    method: "POST",
  });
}

// ---------------------------------------------------------------------------
// Phase 9g-2: Library Scanner
// ---------------------------------------------------------------------------

/** List configured library sources. */
export function fetchLibrarySources(): Promise<LibrarySource[]> {
  return request<LibrarySource[]>("/library/sources");
}

/** Add a new library source. */
export function addLibrarySource(
  type: "apple_music" | "rekordbox",
  path: string,
  label: string,
): Promise<LibrarySource> {
  return request<LibrarySource>("/library/sources", {
    method: "POST",
    body: JSON.stringify({ type, path, label }),
  });
}

/** Update a library source. */
export function updateLibrarySource(
  id: string,
  updates: { path?: string; label?: string },
): Promise<LibrarySource> {
  return request<LibrarySource>(`/library/sources/${encodeURIComponent(id)}`, {
    method: "PUT",
    body: JSON.stringify(updates),
  });
}

/** Delete a library source. */
export function deleteLibrarySource(
  id: string,
): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(
    `/library/sources/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
}

/** Parse/refresh a library source XML. */
export function scanLibrarySource(id: string): Promise<LibrarySource> {
  return request<LibrarySource>(
    `/library/scan/${encodeURIComponent(id)}`,
    { method: "POST" },
  );
}

/** List playlists for a library source. */
export function fetchPlaylists(sourceId: string): Promise<PlaylistInfo[]> {
  return request<PlaylistInfo[]>(
    `/library/playlists/${encodeURIComponent(sourceId)}`,
  );
}

/** Search/filter/sort/paginate tracks from a library source. */
export function searchLibraryTracks(
  sourceId: string,
  params: LibrarySearchParams = {},
): Promise<TrackSearchResult> {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      qs.set(key, String(value));
    }
  }
  const query = qs.toString();
  return request<TrackSearchResult>(
    `/library/tracks/${encodeURIComponent(sourceId)}${query ? `?${query}` : ""}`,
  );
}

/** Get tracks in a specific playlist. */
export function fetchPlaylistTracks(
  sourceId: string,
  playlistId: string,
): Promise<{ tracks: import("../types/api").LibraryTrackInfo[]; count: number }> {
  return request(
    `/library/playlist-tracks/${encodeURIComponent(sourceId)}/${encodeURIComponent(playlistId)}`,
  );
}

/** Generate text descriptions for selected tracks. */
export function describeLibraryTracks(
  sourceId: string,
  trackIds: string[],
  mode: "template" | "llm" = "template",
): Promise<{ descriptions: Record<string, string>; mode: string }> {
  return request("/library/describe", {
    method: "POST",
    body: JSON.stringify({ source_id: sourceId, track_ids: trackIds, mode }),
  });
}

/** Suggest a LoRA adapter name from track metadata. */
export function suggestAdapterName(
  sourceId: string,
  trackIds: string[],
): Promise<{ name: string; analysis: Record<string, unknown> }> {
  return request("/library/suggest-name", {
    method: "POST",
    body: JSON.stringify({ source_id: sourceId, track_ids: trackIds }),
  });
}

/** Analyze tracks and generate a prompt capturing their vibe. */
export function generatePlaylistPrompt(
  sourceId: string,
  trackIds: string[],
): Promise<PlaylistAnalysis> {
  return request<PlaylistAnalysis>("/library/generate-prompt", {
    method: "POST",
    body: JSON.stringify({ source_id: sourceId, track_ids: trackIds }),
  });
}

// ---------------------------------------------------------------------------
// Collections
// ---------------------------------------------------------------------------

/** List all saved collections. */
export function fetchCollections(): Promise<CollectionSummary[]> {
  return request<CollectionSummary[]>("/collections");
}

/** Get a collection by name. */
export function getCollection(name: string): Promise<CollectionFull> {
  return request<CollectionFull>(
    `/collections/${encodeURIComponent(name)}`,
  );
}

/** Create a new collection. */
export function createCollection(data: {
  name: string;
  source?: string;
  playlist?: string;
  tracks?: Record<string, unknown>[];
}): Promise<CollectionFull> {
  return request<CollectionFull>("/collections", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

/** Update a collection. */
export function updateCollection(
  name: string,
  updates: {
    tracks?: Record<string, unknown>[];
    source?: string;
    playlist?: string;
  },
): Promise<CollectionFull> {
  return request<CollectionFull>(
    `/collections/${encodeURIComponent(name)}`,
    { method: "PUT", body: JSON.stringify(updates) },
  );
}

/** Delete a collection. */
export function deleteCollection(
  name: string,
): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(
    `/collections/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  );
}

/** Export a collection as JSON download URL. */
export function getCollectionExportUrl(name: string): string {
  return `${_base}/collections/${encodeURIComponent(name)}/export`;
}

/** Import a collection from a JSON file. */
export async function importCollection(
  file: File,
): Promise<CollectionFull> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${_base}/collections/import`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API error ${res.status}: ${body}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Phase 9g-3: Credentials
// ---------------------------------------------------------------------------

/** Get credential configuration status for each enrichment service. */
export function getCredentialStatus(): Promise<CredentialStatus> {
  return request<CredentialStatus>("/credentials/status");
}

/** Set an API key for an enrichment service. */
export function setCredential(
  service: string,
  apiKey: string,
): Promise<void> {
  return request("/credentials/" + encodeURIComponent(service), {
    method: "POST",
    body: JSON.stringify({ api_key: apiKey }),
  });
}

/** Delete an API key for an enrichment service. */
export function deleteCredential(service: string): Promise<void> {
  return request("/credentials/" + encodeURIComponent(service), {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// Phase 9g-3: Enrichment
// ---------------------------------------------------------------------------

/** Enrich tracks with metadata from external services. */
export function enrichTracks(body: {
  track_ids?: string[];
  source_id?: string;
  tracks?: { artist: string; title: string }[];
}): Promise<{ job_id: string }> {
  return request("/enrich/tracks", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

/** Get enrichment job status. */
export function getEnrichmentStatus(): Promise<EnrichmentJobStatus> {
  return request<EnrichmentJobStatus>("/enrich/status");
}

/** Enrich all tracks from a library source. */
export function enrichAll(
  sourceId: string,
): Promise<{ job_id: string }> {
  return request(`/enrich/all/${encodeURIComponent(sourceId)}`, {
    method: "POST",
  });
}

/** Cancel running enrichment job. */
export function cancelEnrichment(): Promise<void> {
  return request("/enrich/cancel", { method: "POST" });
}

/** Get enrichment data for a single track. */
export function getTrackEnrichment(
  trackId: number,
): Promise<Record<string, unknown>> {
  return request(`/enrich/track/${trackId}`);
}

/** Get enrichment statistics. */
export function getEnrichmentStats(): Promise<EnrichmentStats> {
  return request<EnrichmentStats>("/enrich/stats");
}

// ---------------------------------------------------------------------------
// Phase 9g-3: Taste Profile
// ---------------------------------------------------------------------------

/** Get the user's taste profile. */
export function getTasteProfile(): Promise<TasteProfile> {
  return request<TasteProfile>("/taste/profile");
}

/** Refresh the taste profile from current data. */
export function refreshTasteProfile(): Promise<TasteProfile> {
  return request<TasteProfile>("/taste/refresh", { method: "POST" });
}

/** Get taste-based prompt suggestions. */
export function getTasteSuggestions(): Promise<{ suggestions: string[] }> {
  return request<{ suggestions: string[] }>("/taste/suggestions");
}

/** Set manual taste overrides. */
export function setTasteOverrides(text: string): Promise<TasteProfile> {
  return request<TasteProfile>("/taste/overrides", {
    method: "PUT",
    body: JSON.stringify({ text }),
  });
}

// ---------------------------------------------------------------------------
// Phase 9g-4: Flywheel Intelligence
// ---------------------------------------------------------------------------

/** Star a generation for re-training. */
export async function starGeneration(
  jobId: string,
): Promise<StarResponse> {
  const res = await fetch(`${_base}/star/${jobId}`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** Unstar a generation. */
export async function unstarGeneration(
  jobId: string,
): Promise<StarResponse> {
  const res = await fetch(`${_base}/star/${jobId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** Get flywheel configuration. */
export function getFlywheelConfig(): Promise<FlywheelConfig> {
  return request<FlywheelConfig>("/flywheel/config");
}

/** Update flywheel configuration. */
export function updateFlywheelConfig(
  config: Partial<FlywheelConfig>,
): Promise<FlywheelConfig> {
  return request<FlywheelConfig>("/flywheel/config", {
    method: "PUT",
    body: JSON.stringify(config),
  });
}

/** Get flywheel status. */
export function getFlywheelStatus(): Promise<FlywheelStatus> {
  return request<FlywheelStatus>("/flywheel/status");
}

/** List versions for a LoRA adapter. */
export function getLoraVersions(
  name: string,
): Promise<VersionSummary[]> {
  return request<VersionSummary[]>(
    `/loras/${encodeURIComponent(name)}/versions`,
  );
}

/** Get changelog for a specific LoRA version. */
export function getLoraChangelog(
  name: string,
  version: number,
): Promise<VersionChangelog> {
  return request<VersionChangelog>(
    `/loras/${encodeURIComponent(name)}/versions/${version}`,
  );
}

/** Set the active version for a LoRA adapter. */
export function setActiveLoraVersion(
  name: string,
  version: number,
): Promise<{ name: string; active_version: number }> {
  return request<{ name: string; active_version: number }>(
    `/loras/${encodeURIComponent(name)}/active/${version}`,
    { method: "PUT" },
  );
}

/** Trigger a flywheel re-train for a LoRA adapter. */
export function triggerRetrain(
  name: string,
): Promise<{ status: string; adapter: string }> {
  return request<{ status: string; adapter: string }>(
    `/flywheel/retrain/${encodeURIComponent(name)}`,
    { method: "POST" },
  );
}

/** Reset kept generations for a LoRA adapter. */
export function resetKeptGenerations(
  name: string,
): Promise<{ status: string; adapter: string }> {
  return request<{ status: string; adapter: string }>(
    `/flywheel/reset/${encodeURIComponent(name)}`,
    { method: "POST" },
  );
}
