import type {
  EnhanceResponse,
  GenerateRequest,
  GenerateResponse,
  JobInfo,
  LLMModelInfo,
  LLMStatus,
  LoRAInfo,
  ModelInfo,
  PresetInfo,
  PromptAnalysis,
  PromptMemoryData,
  ServerSettings,
  StemResult,
  TagDatabase,
  TrainRequest,
  TrainStatus,
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
