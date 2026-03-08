import { useEffect } from "react";
import { useStore } from "./store/useStore";
import { useServerHeartbeat } from "./hooks/useServerHeartbeat";
import Header from "./components/Header";
import ModelSelector from "./components/ModelSelector";
import PromptInput from "./components/PromptInput";
import ParameterPanel from "./components/ParameterPanel";
import GenerateButton from "./components/GenerateButton";
import HistoryPanel from "./components/HistoryPanel";
import AudioDeviceSelector from "./components/AudioDeviceSelector";
import SettingsPanel from "./components/SettingsPanel";

export default function App() {
  const loadModels = useStore((s) => s.loadModels);
  const loadHistory = useStore((s) => s.loadHistory);
  const loadSettings = useStore((s) => s.loadSettings);
  const modelsLoading = useStore((s) => s.modelsLoading);
  const modelsError = useStore((s) => s.modelsError);
  const connected = useServerHeartbeat();

  useEffect(() => {
    loadModels();
    loadHistory();
    loadSettings();
  }, [loadModels, loadHistory, loadSettings]);

  return (
    <div className="flex h-screen flex-col bg-surface-0">
      {/* Server disconnected banner */}
      {!connected && (
        <div className="bg-error/90 text-surface-0 px-4 py-2 text-center text-sm font-medium">
          Server disconnected — restart with{" "}
          <code className="bg-surface-0/20 px-1 rounded">mlx-audiogen-app</code>
        </div>
      )}

      <Header />

      <main className="flex flex-1 overflow-hidden">
        {/* Left panel: Controls */}
        <div className="flex w-80 shrink-0 flex-col gap-5 overflow-y-auto border-r border-border bg-surface-1 p-5">
          {modelsLoading && (
            <div className="text-xs text-text-muted">Loading models...</div>
          )}
          {modelsError && (
            <div className="rounded border border-error/30 bg-error/10 px-3 py-2 text-xs text-error">
              Failed to connect to server: {modelsError}
            </div>
          )}

          <ModelSelector />
          <PromptInput />
          <ParameterPanel />
          <GenerateButton />

          {/* Bottom section: settings + audio output */}
          <div className="mt-auto space-y-4 pt-4 border-t border-border">
            <SettingsPanel />
            <AudioDeviceSelector />
          </div>
        </div>

        {/* Right panel: History / Output */}
        <div className="flex flex-1 flex-col overflow-y-auto p-5">
          <HistoryPanel />
        </div>
      </main>
    </div>
  );
}
