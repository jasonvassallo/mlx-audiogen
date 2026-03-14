import { useMemo } from "react";
import type { TagDatabase } from "../types/api";

/** Color mapping for each tag category. */
const CATEGORY_COLORS: Record<
  string,
  { bg: string; text: string; label: string }
> = {
  genre: { bg: "bg-warning/20", text: "text-warning", label: "genre" },
  sub_genre: { bg: "bg-orange-500/20", text: "text-orange-400", label: "sub-genre" },
  mood: { bg: "bg-success/20", text: "text-success", label: "mood" },
  instrument: { bg: "bg-info/20", text: "text-info", label: "instrument" },
  vocal: { bg: "bg-teal-500/20", text: "text-teal-400", label: "vocal" },
  key: { bg: "bg-cyan-500/20", text: "text-cyan-400", label: "key" },
  bpm: { bg: "bg-blue-500/20", text: "text-blue-400", label: "bpm" },
  era: { bg: "bg-purple-500/20", text: "text-purple-400", label: "era" },
  production: { bg: "bg-rose-500/20", text: "text-rose-400", label: "production" },
  artist: { bg: "bg-indigo-500/20", text: "text-indigo-400", label: "artist" },
  label: { bg: "bg-fuchsia-500/20", text: "text-fuchsia-400", label: "label" },
  structure: { bg: "bg-lime-500/20", text: "text-lime-400", label: "structure" },
  rating: { bg: "bg-yellow-500/20", text: "text-yellow-400", label: "rating" },
  availability: { bg: "bg-gray-500/20", text: "text-gray-400", label: "available" },
};

const MAX_RESULTS = 8;
const MIN_QUERY_LENGTH = 2;

interface TagAutocompleteProps {
  query: string;
  tagDatabase: TagDatabase | null;
  onSelect: (tag: string) => void;
  onDismiss: () => void;
  visible: boolean;
}

export default function TagAutocomplete({
  query,
  tagDatabase,
  onSelect,
  onDismiss,
  visible,
}: TagAutocompleteProps) {
  const matches = useMemo(() => {
    if (!tagDatabase || query.length < MIN_QUERY_LENGTH) return [];

    // Extract the last word being typed (after comma or space)
    const lastToken = query
      .split(/[,]/)
      .pop()
      ?.trim()
      .toLowerCase();
    if (!lastToken || lastToken.length < MIN_QUERY_LENGTH) return [];

    const results: Array<{ tag: string; category: string }> = [];

    for (const [category, tags] of Object.entries(tagDatabase)) {
      for (const tag of tags) {
        if (tag.toLowerCase().includes(lastToken)) {
          results.push({ tag, category });
          if (results.length >= MAX_RESULTS) break;
        }
      }
      if (results.length >= MAX_RESULTS) break;
    }

    return results;
  }, [query, tagDatabase]);

  if (!visible || matches.length === 0) return null;

  return (
    <div
      className="
        absolute left-0 right-0 top-full z-20 mt-1
        rounded border border-border bg-surface-2 shadow-lg
        max-h-48 overflow-y-auto
      "
    >
      {matches.map(({ tag, category }, idx) => {
        const colors = CATEGORY_COLORS[category] ?? {
          bg: "bg-surface-3",
          text: "text-text-muted",
          label: category,
        };
        return (
          <button
            key={`${category}-${tag}-${idx}`}
            onClick={() => {
              onSelect(tag);
              onDismiss();
            }}
            className="
              flex w-full items-center gap-2 px-3 py-1.5 text-left
              hover:bg-surface-3 transition-colors
            "
          >
            <span className={`shrink-0 rounded px-1.5 py-0.5 text-xs ${colors.bg} ${colors.text}`}>
              {colors.label}
            </span>
            <span className="text-xs text-text-primary truncate">{tag}</span>
          </button>
        );
      })}
    </div>
  );
}
