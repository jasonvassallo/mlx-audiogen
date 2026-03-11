import { useStore } from "../store/useStore";

export default function Header() {
  const models = useStore((s) => s.models);
  const loadedCount = models.filter((m) => m.is_loaded).length;

  return (
    <header className="flex items-center justify-between border-b border-border px-6 py-3 bg-surface-1">
      <div className="flex items-center gap-3">
        <h1 className="text-base font-bold tracking-wide text-text-primary">
          MLX AudioGen
        </h1>
        <span className="text-xs text-text-muted">v0.1.0</span>
      </div>
      <div className="flex items-center gap-4 text-xs text-text-secondary">
        <span>
          {models.length} model{models.length !== 1 ? "s" : ""} available
        </span>
        {loadedCount > 0 && (
          <span className="text-success">
            {loadedCount} loaded
          </span>
        )}
        <a
          href="https://paypal.me/jasonvassallo"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1 text-xs text-text-muted hover:text-accent transition-colors"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
          </svg>
          Support
        </a>
      </div>
    </header>
  );
}
