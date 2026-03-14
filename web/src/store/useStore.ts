import { create } from "zustand";
import type {
  EnhanceResponse,
  GenerateRequest,
  JobInfo,
  LLMModelInfo,
  LLMStatus,
  LoRAInfo,
  ModelInfo,
  PresetInfo,
  PromptAnalysis,
  PromptMemoryData,
  ServerSettings,
  TagDatabase,
} from "../types/api";
import {
  fetchModels,
  submitGeneration,
  fetchJobStatus,
  getAudioUrl,
  suggestPrompts,
  separateStems,
  fetchPresets as apiFetchPresets,
  savePreset as apiSavePreset,
  loadPreset as apiLoadPreset,
  enhancePrompt as apiEnhancePrompt,
  fetchTags as apiFetchTags,
  fetchLLMModels as apiFetchLLMModels,
  selectLLMModel as apiSelectLLMModel,
  fetchLLMStatus as apiFetchLLMStatus,
  fetchMemory as apiFetchMemory,
  clearMemory as apiClearMemory,
  importMemory as apiImportMemory,
  fetchServerSettings as apiFetchServerSettings,
  updateServerSettings as apiUpdateServerSettings,
  getServerUrl,
  setServerUrl as apiSetServerUrl,
  fetchLoras as apiFetchLoras,
} from "../api/client";
import {
  saveEntry,
  loadAllEntries,
  deleteEntry,
  updateFavorite,
  updateSourceBpm,
  clearAllEntries,
  purgeExpiredEntries,
  loadSettings,
  saveSettings,
  type PersistedEntry,
  type HistorySettings,
} from "./historyDb";

/** A generation entry with a local blob URL for playback. */
export interface HistoryEntry {
  id: string;
  job: JobInfo;
  audioUrl: string; // blob: URL for local playback
  favorite: boolean;
  createdAt: number;
  sourceBpm: number; // 0 = unknown
}

interface AppState {
  // --- Models ---
  models: ModelInfo[];
  modelsLoading: boolean;
  modelsError: string | null;
  loadModels: () => Promise<void>;

  // --- Generation Parameters ---
  params: GenerateRequest;
  setParam: <K extends keyof GenerateRequest>(
    key: K,
    value: GenerateRequest[K],
  ) => void;

  // --- Active Job ---
  activeJob: JobInfo | null;
  isGenerating: boolean;
  generateError: string | null;
  generate: () => Promise<void>;

  // --- History (IndexedDB-backed) ---
  history: HistoryEntry[];
  historyLoaded: boolean;
  loadHistory: () => Promise<void>;
  toggleFavorite: (id: string) => Promise<void>;
  setSourceBpm: (id: string, bpm: number) => Promise<void>;
  deleteHistoryEntry: (id: string) => Promise<void>;
  clearHistory: () => Promise<void>;

  // --- Settings ---
  settings: HistorySettings;
  settingsLoaded: boolean;
  loadSettings: () => Promise<void>;
  updateSettings: (settings: Partial<HistorySettings>) => Promise<void>;

  // --- Active Tab ---
  activeTab: "generate" | "suggest" | "train" | "settings";
  setActiveTab: (tab: "generate" | "suggest" | "train" | "settings") => void;

  // --- Enhance Flow ---
  enhanceResult: EnhanceResponse | null;
  enhanceLoading: boolean;
  enhancePrompt: () => Promise<void>;
  clearEnhanceResult: () => void;

  // --- Server Settings ---
  serverSettings: ServerSettings;
  serverSettingsLoaded: boolean;
  loadServerSettings: () => Promise<void>;
  updateServerSetting: (updates: Partial<ServerSettings>) => Promise<void>;

  // --- Tag Database ---
  tagDatabase: TagDatabase | null;
  tagsLoaded: boolean;
  loadTags: () => Promise<void>;

  // --- Prompt Memory ---
  promptMemory: PromptMemoryData | null;
  loadPromptMemory: () => Promise<void>;
  clearPromptMemory: () => Promise<void>;
  importPromptMemory: (file: File) => Promise<void>;

  // --- LLM Models ---
  llmModels: LLMModelInfo[];
  llmStatus: LLMStatus | null;
  loadLLMModels: () => Promise<void>;
  selectLLM: (modelId: string) => Promise<void>;
  loadLLMStatus: () => Promise<void>;

  // --- Prompt Suggestions ---
  suggestions: PromptAnalysis | null;
  suggestionsLoading: boolean;
  lastAnalyzedPrompt: string;
  fetchSuggestions: () => Promise<void>;

  // --- Presets ---
  presets: PresetInfo[];
  presetsLoading: boolean;
  loadPresets: () => Promise<void>;
  saveCurrentPreset: (name: string) => Promise<void>;
  applyPreset: (name: string) => Promise<void>;

  // --- Stem Separation ---
  stemResults: Record<string, Record<string, string>>; // jobId → {stemName → stemJobId}
  stemsLoading: Record<string, boolean>;
  stemAudioUrls: Record<string, Record<string, string>>; // jobId → {stemName → blobUrl}
  requestStemSeparation: (jobId: string) => Promise<void>;

  // --- Server Connection ---
  serverUrl: string; // "" = local, or "http://host:port"
  setServerUrl: (url: string) => void;

  // --- Batch Queue ---
  queue: GenerateRequest[];
  queueRunning: boolean;
  queueProgress: { current: number; total: number } | null;
  addToQueue: (item?: GenerateRequest) => void;
  removeFromQueue: (index: number) => void;
  clearQueue: () => void;
  runQueue: () => Promise<void>;

  // --- LoRA (Phase 9g) ---
  loras: LoRAInfo[];
  selectedLora: string | null;
  fetchLoras: () => Promise<void>;
  setSelectedLora: (name: string | null) => void;
}

const DEFAULT_PARAMS: GenerateRequest = {
  model: "musicgen",
  prompt: "",
  seconds: 5,
  temperature: 1.0,
  top_k: 250,
  guidance_coef: 3.0,
  steps: 8,
  cfg_scale: 6.0,
  sampler: "euler",
  seed: null,
  melody_path: null,
  style_audio_path: null,
  style_coef: 5.0,
  output_mode: "audio" as const,
};

const POLL_INTERVAL = 500;

/** Convert a PersistedEntry (with Blob) to a HistoryEntry (with blob URL). */
function toHistoryEntry(entry: PersistedEntry): HistoryEntry {
  return {
    id: entry.id,
    job: entry.job,
    audioUrl: URL.createObjectURL(entry.audioBlob),
    favorite: entry.favorite,
    createdAt: entry.createdAt,
    sourceBpm: entry.sourceBpm ?? 0,
  };
}

export const useStore = create<AppState>((set, get) => ({
  // --- Models ---
  models: [],
  modelsLoading: false,
  modelsError: null,
  loadModels: async () => {
    set({ modelsLoading: true, modelsError: null });
    try {
      const models = await fetchModels();
      set({ models, modelsLoading: false });
      if (models.length > 0 && !get().params.model) {
        set((s) => ({
          params: { ...s.params, model: models[0]!.model_type },
        }));
      }
    } catch (e) {
      set({
        modelsError: e instanceof Error ? e.message : String(e),
        modelsLoading: false,
      });
    }
  },

  // --- Generation Parameters ---
  params: DEFAULT_PARAMS,
  setParam: (key, value) =>
    set((s) => ({ params: { ...s.params, [key]: value } })),

  // --- Active Job ---
  activeJob: null,
  isGenerating: false,
  generateError: null,
  generate: async () => {
    const { params, isGenerating, selectedLora } = get();
    if (isGenerating) return;
    if (!params.prompt.trim()) {
      set({ generateError: "Prompt is required" });
      return;
    }

    set({ isGenerating: true, generateError: null, activeJob: null });

    try {
      const reqParams = { ...params, lora: selectedLora };
      const { id } = await submitGeneration(reqParams);

      // Poll until done
      const poll = async (): Promise<JobInfo> => {
        const job = await fetchJobStatus(id);
        set({ activeJob: job });
        if (job.status === "done" || job.status === "error") {
          return job;
        }
        await new Promise((r) => setTimeout(r, POLL_INTERVAL));
        return poll();
      };

      const finalJob = await poll();

      if (finalJob.status === "done") {
        // Eagerly download the audio blob before server cleans it up (5 min)
        const audioRes = await fetch(getAudioUrl(finalJob.id));
        const audioBlob = await audioRes.blob();
        const now = Date.now();

        // Persist to IndexedDB
        const persisted: PersistedEntry = {
          id: finalJob.id,
          job: finalJob,
          audioBlob,
          favorite: false,
          createdAt: now,
          sourceBpm: 0,
        };
        await saveEntry(persisted);

        // Add to in-memory history
        const entry: HistoryEntry = {
          id: finalJob.id,
          job: finalJob,
          audioUrl: URL.createObjectURL(audioBlob),
          favorite: false,
          createdAt: now,
          sourceBpm: 0,
        };
        set((s) => ({
          history: [entry, ...s.history],
          isGenerating: false,
        }));

        // Run auto-purge if retention is configured
        const { settings } = get();
        if (settings.retentionHours > 0) {
          const deleted = await purgeExpiredEntries(settings.retentionHours);
          if (deleted > 0) {
            // Reload history to reflect purged entries
            await get().loadHistory();
          }
        }
      } else {
        set({
          generateError: finalJob.error ?? "Generation failed",
          isGenerating: false,
        });
      }
    } catch (e) {
      set({
        generateError: e instanceof Error ? e.message : String(e),
        isGenerating: false,
      });
    }
  },

  // --- History (IndexedDB-backed) ---
  history: [],
  historyLoaded: false,
  loadHistory: async () => {
    try {
      const entries = await loadAllEntries();
      const historyEntries = entries.map(toHistoryEntry);
      set({ history: historyEntries, historyLoaded: true });
    } catch (e) {
      console.error("Failed to load history:", e);
      set({ historyLoaded: true });
    }
  },

  toggleFavorite: async (id: string) => {
    const entry = get().history.find((h) => h.id === id);
    if (!entry) return;
    const newFav = !entry.favorite;
    await updateFavorite(id, newFav);
    set((s) => ({
      history: s.history.map((h) =>
        h.id === id ? { ...h, favorite: newFav } : h,
      ),
    }));
  },

  setSourceBpm: async (id: string, bpm: number) => {
    await updateSourceBpm(id, bpm);
    set((s) => ({
      history: s.history.map((h) =>
        h.id === id ? { ...h, sourceBpm: bpm } : h,
      ),
    }));
  },

  deleteHistoryEntry: async (id: string) => {
    // Revoke blob URL to free memory
    const entry = get().history.find((h) => h.id === id);
    if (entry) URL.revokeObjectURL(entry.audioUrl);

    await deleteEntry(id);
    set((s) => ({
      history: s.history.filter((h) => h.id !== id),
    }));
  },

  clearHistory: async () => {
    // Revoke all blob URLs
    get().history.forEach((h) => URL.revokeObjectURL(h.audioUrl));
    await clearAllEntries();
    set({ history: [] });
  },

  // --- Settings ---
  settings: { retentionHours: 0, masterBpm: 120, preservePitch: true },
  settingsLoaded: false,
  loadSettings: async () => {
    try {
      const settings = await loadSettings();
      set({ settings, settingsLoaded: true });
    } catch (e) {
      console.error("Failed to load settings:", e);
      set({ settingsLoaded: true });
    }
  },

  updateSettings: async (partial) => {
    const newSettings = { ...get().settings, ...partial };
    await saveSettings(newSettings);
    set({ settings: newSettings });

    // If retention was just enabled, run a purge now
    if (partial.retentionHours && partial.retentionHours > 0) {
      const deleted = await purgeExpiredEntries(partial.retentionHours);
      if (deleted > 0) {
        await get().loadHistory();
      }
    }
  },

  // --- Active Tab ---
  activeTab: "generate",
  setActiveTab: (tab) => set({ activeTab: tab }),

  // --- Enhance Flow ---
  enhanceResult: null,
  enhanceLoading: false,
  enhancePrompt: async () => {
    const { params, enhanceLoading } = get();
    if (enhanceLoading || !params.prompt.trim()) return;
    set({ enhanceLoading: true });
    try {
      const result = await apiEnhancePrompt(params.prompt);
      set({ enhanceResult: result, enhanceLoading: false });
    } catch (e) {
      console.error("Failed to enhance prompt:", e);
      set({ enhanceLoading: false });
    }
  },
  clearEnhanceResult: () => set({ enhanceResult: null }),

  // --- Server Settings ---
  serverSettings: { llm_model: null, ai_enhance: true, history_context_count: 50 },
  serverSettingsLoaded: false,
  loadServerSettings: async () => {
    try {
      const settings = await apiFetchServerSettings();
      set({ serverSettings: settings, serverSettingsLoaded: true });
    } catch (e) {
      console.error("Failed to load server settings:", e);
      set({ serverSettingsLoaded: true });
    }
  },
  updateServerSetting: async (updates) => {
    try {
      const result = await apiUpdateServerSettings(updates);
      set({ serverSettings: result });
    } catch (e) {
      console.error("Failed to update server settings:", e);
    }
  },

  // --- Tag Database ---
  tagDatabase: null,
  tagsLoaded: false,
  loadTags: async () => {
    if (get().tagsLoaded) return;
    try {
      const tags = await apiFetchTags();
      set({ tagDatabase: tags, tagsLoaded: true });
    } catch (e) {
      console.error("Failed to load tags:", e);
    }
  },

  // --- Prompt Memory ---
  promptMemory: null,
  loadPromptMemory: async () => {
    try {
      const memory = await apiFetchMemory();
      set({ promptMemory: memory });
    } catch (e) {
      console.error("Failed to load prompt memory:", e);
    }
  },
  clearPromptMemory: async () => {
    try {
      await apiClearMemory();
      set({ promptMemory: { history: [], style_profile: { top_genres: [], top_moods: [], top_instruments: [], preferred_duration: 0, generation_count: 0 } } });
    } catch (e) {
      console.error("Failed to clear prompt memory:", e);
    }
  },
  importPromptMemory: async (file: File) => {
    try {
      await apiImportMemory(file);
      await get().loadPromptMemory();
    } catch (e) {
      console.error("Failed to import prompt memory:", e);
    }
  },

  // --- LLM Models ---
  llmModels: [],
  llmStatus: null,
  loadLLMModels: async () => {
    try {
      const models = await apiFetchLLMModels();
      set({ llmModels: models });
    } catch (e) {
      console.error("Failed to load LLM models:", e);
    }
  },
  selectLLM: async (modelId: string) => {
    try {
      await apiSelectLLMModel(modelId);
      await get().loadLLMStatus();
    } catch (e) {
      console.error("Failed to select LLM model:", e);
    }
  },
  loadLLMStatus: async () => {
    try {
      const status = await apiFetchLLMStatus();
      set({ llmStatus: status });
    } catch (e) {
      console.error("Failed to load LLM status:", e);
    }
  },

  // --- Prompt Suggestions ---
  suggestions: null,
  suggestionsLoading: false,
  lastAnalyzedPrompt: "",
  fetchSuggestions: async () => {
    const { params, lastAnalyzedPrompt, suggestionsLoading } = get();
    const prompt = params.prompt.trim();
    if (!prompt || suggestionsLoading) return;
    if (prompt === lastAnalyzedPrompt) return; // cached
    set({ suggestionsLoading: true });
    try {
      const analysis = await suggestPrompts(prompt);
      set({ suggestions: analysis, lastAnalyzedPrompt: prompt, suggestionsLoading: false });
    } catch (e) {
      console.error("Failed to fetch suggestions:", e);
      set({ suggestionsLoading: false });
    }
  },

  // --- Presets ---
  presets: [],
  presetsLoading: false,
  loadPresets: async () => {
    set({ presetsLoading: true });
    try {
      const presets = await apiFetchPresets();
      set({ presets, presetsLoading: false });
    } catch (e) {
      console.error("Failed to load presets:", e);
      set({ presetsLoading: false });
    }
  },

  saveCurrentPreset: async (name: string) => {
    const { params } = get();
    try {
      await apiSavePreset(name, params);
      await get().loadPresets(); // refresh list
    } catch (e) {
      console.error("Failed to save preset:", e);
    }
  },

  applyPreset: async (name: string) => {
    try {
      const loaded = await apiLoadPreset(name);
      // Validate model is available
      const { models } = get();
      const validModel = models.some((m) => m.model_type === loaded.model);
      set((s) => ({
        params: {
          ...s.params,
          ...loaded,
          model: validModel ? loaded.model : s.params.model,
        },
        activeTab: "generate",
      }));
    } catch (e) {
      console.error("Failed to apply preset:", e);
    }
  },

  // --- Server Connection ---
  serverUrl: getServerUrl(),
  setServerUrl: (url: string) => {
    apiSetServerUrl(url);
    set({ serverUrl: url, modelsLoading: false, modelsError: null });
    // Re-fetch models from the new server
    get().loadModels();
  },

  // --- Stem Separation ---
  stemResults: {},
  stemsLoading: {},
  stemAudioUrls: {},
  requestStemSeparation: async (jobId: string) => {
    const { stemsLoading } = get();
    if (stemsLoading[jobId]) return;
    set((s) => ({ stemsLoading: { ...s.stemsLoading, [jobId]: true } }));
    try {
      const result = await separateStems(jobId);
      set((s) => ({
        stemResults: { ...s.stemResults, [jobId]: result.stems },
        stemsLoading: { ...s.stemsLoading, [jobId]: false },
      }));
      // Eagerly download stem audio blobs before server cleanup (5 min)
      const urls: Record<string, string> = {};
      for (const [stemName, stemId] of Object.entries(result.stems)) {
        try {
          const res = await fetch(getAudioUrl(stemId));
          const blob = await res.blob();
          urls[stemName] = URL.createObjectURL(blob);
        } catch {
          // Fall back to direct server URL
          urls[stemName] = getAudioUrl(stemId);
        }
      }
      set((s) => ({
        stemAudioUrls: { ...s.stemAudioUrls, [jobId]: urls },
      }));
    } catch (e) {
      console.error("Failed to separate stems:", e);
      set((s) => ({ stemsLoading: { ...s.stemsLoading, [jobId]: false } }));
    }
  },

  // --- Batch Queue ---
  queue: [],
  queueRunning: false,
  queueProgress: null,
  addToQueue: (item?: GenerateRequest) => {
    const params = item ?? { ...get().params };
    if (!params.prompt.trim()) return;
    set((s) => ({ queue: [...s.queue, params] }));
  },
  removeFromQueue: (index: number) => {
    set((s) => ({ queue: s.queue.filter((_, i) => i !== index) }));
  },
  clearQueue: () => set({ queue: [] }),
  runQueue: async () => {
    const { queue, queueRunning } = get();
    if (queueRunning || queue.length === 0) return;

    set({ queueRunning: true, queueProgress: { current: 0, total: queue.length } });
    const items = [...queue];

    for (let i = 0; i < items.length; i++) {
      if (!get().queueRunning) break; // cancelled
      set({ queueProgress: { current: i + 1, total: items.length } });

      try {
        const { id } = await submitGeneration(items[i]!);

        // Poll until done
        let job: JobInfo;
        do {
          await new Promise((r) => setTimeout(r, POLL_INTERVAL));
          job = await fetchJobStatus(id);
          set({ activeJob: job });
        } while (job.status !== "done" && job.status !== "error");

        if (job.status === "done") {
          const audioRes = await fetch(getAudioUrl(job.id));
          const audioBlob = await audioRes.blob();
          const now = Date.now();
          const persisted: PersistedEntry = {
            id: job.id, job, audioBlob, favorite: false, createdAt: now, sourceBpm: 0,
          };
          await saveEntry(persisted);
          const entry: HistoryEntry = {
            id: job.id, job, audioUrl: URL.createObjectURL(audioBlob),
            favorite: false, createdAt: now, sourceBpm: 0,
          };
          set((s) => ({ history: [entry, ...s.history] }));
        }
      } catch (e) {
        console.error(`Queue item ${i + 1} failed:`, e);
      }
    }

    set({ queueRunning: false, queueProgress: null, queue: [], activeJob: null });
  },

  // --- LoRA (Phase 9g) ---
  loras: [],
  selectedLora: null,
  fetchLoras: async () => {
    try {
      const loras = await apiFetchLoras();
      set({ loras });
    } catch {
      set({ loras: [] });
    }
  },
  setSelectedLora: (name) => set({ selectedLora: name }),
}));
