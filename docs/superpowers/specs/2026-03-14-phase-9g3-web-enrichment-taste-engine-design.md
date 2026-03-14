# Phase 9g-3: Web Enrichment + Taste Learning Engine

## Summary

Enrich library track metadata from MusicBrainz, Last.fm, and Discogs. Build a taste learning engine that combines library listening data with generation history to personalize suggestions, prompt generation, and LoRA training data selection.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scope | Both enrichment + taste engine in one session | 1M context, existing 9g-2 infrastructure reduces new work |
| API keys | macOS Keychain (keyring) primary, env var fallback, graceful degradation | Secure-by-default, no plaintext secrets on disk |
| Enrichment trigger | Unified hybrid: lazy-on-view, batch select, enrich all, per-track | Full user control from passive to active |
| Enrichment cache | SQLite (`~/.mlx-audiogen/enrichment.db`) | 16K+ tracks, indexed lookups, partial updates, proven pattern |
| Taste signals | Library (listening) + generation (creating), generation-weighted | Library = broad base, generation = creative intent |
| Taste output | Visible profile + personalized suggestions + auto-tuning | Transparency + progressive intelligence, user override always wins |
| APIs | MusicBrainz (free, 1/sec), Last.fm (free key, 5/sec), Discogs (token, 1/sec) | No Spotify — audio features API deprecated Nov 2024 |
| Security | Keychain for secrets, FileVault at rest, localhost binding, no DB encryption | Pragmatic — data is mostly public metadata, real secrets in Keychain |

## Backend Modules

### New files

```
mlx_audiogen/
├── credentials.py                    # Keychain-based credential manager
├── library/
│   ├── enrichment/
│   │   ├── __init__.py
│   │   ├── clients.py               # httpx async clients for all 3 APIs
│   │   ├── musicbrainz.py           # Search + tag extraction
│   │   ├── lastfm.py                # Similar artists + crowd tags
│   │   ├── discogs.py               # Labels + style taxonomy
│   │   ├── enrichment_db.py         # SQLite cache (enrichment.db)
│   │   ├── rate_limiter.py          # Per-API token bucket
│   │   └── manager.py               # Orchestrator: cache check → fetch → merge → store
│   └── taste/
│       ├── __init__.py
│       ├── profile.py               # TasteProfile dataclass + computation
│       ├── signals.py               # Signal collectors (library + generation)
│       └── engine.py                # TasteEngine: learns, updates, queries
```

### Modified files

- `mlx_audiogen/library/models.py` — add EnrichedTrackData fields
- `mlx_audiogen/library/description_gen.py` — `generate_description()` gains an optional `enrichment: dict | None` parameter; when provided, enrichment tags (genres, styles, labels, similar artists) are merged into the template/LLM context alongside TrackInfo fields
- `server/app.py` — new endpoints, startup initialization
- `web/src/types/api.ts` — new TypeScript types
- `web/src/api/client.ts` — new fetch wrappers
- `web/src/store/` — enrichment + taste state
- `web/src/components/LibraryPanel.tsx` — enrichment dots, Enrich Selected button
- `web/src/components/SettingsPanel.tsx` — API keys section, taste profile card

### Design notes

- `enrichment/` is a subpackage because each API has distinct response formats, rate limits, and error handling
- `clients.py` uses `httpx.AsyncClient` with connection pooling for FastAPI compatibility
- `rate_limiter.py` is per-API (separate from the existing server DoS rate limiter)
- `credentials.py` at package root (`mlx_audiogen/credentials.py`) serves the whole project
- `taste/` is separate from `enrichment/` — different concerns (aggregation vs fetching)

## Data Model

### SQLite schema (`enrichment.db`)

```sql
-- Canonical track identity for deduplication
CREATE TABLE tracks (
    id            INTEGER PRIMARY KEY,
    artist        TEXT NOT NULL,
    title         TEXT NOT NULL,
    artist_norm   TEXT NOT NULL,
    title_norm    TEXT NOT NULL,
    musicbrainz_id TEXT,
    library_source TEXT,    -- 'apple_music' or 'rekordbox'
    library_track_id TEXT,  -- original string ID from library XML
    UNIQUE(artist_norm, title_norm)
);
CREATE INDEX idx_tracks_norm ON tracks(artist_norm, title_norm);
CREATE INDEX idx_tracks_library ON tracks(library_source, library_track_id);

-- One table per API source (independent refresh cycles)
CREATE TABLE musicbrainz (
    track_id      INTEGER PRIMARY KEY REFERENCES tracks(id),
    tags          TEXT,    -- JSON: [{name, count}]
    genres        TEXT,    -- JSON array
    release_group TEXT,    -- album MBID
    artist_mbid   TEXT,
    similar_artists TEXT,  -- JSON array
    fetched_at    TEXT     -- ISO timestamp
);

CREATE TABLE lastfm (
    track_id      INTEGER PRIMARY KEY REFERENCES tracks(id),
    tags          TEXT,    -- JSON: [{name, count}]
    similar_tracks TEXT,   -- JSON: [{artist, title, match_score}]
    similar_artists TEXT,  -- JSON: [{name, match_score}]
    play_count    INTEGER,
    listeners     INTEGER,
    fetched_at    TEXT
);

CREATE TABLE discogs (
    track_id      INTEGER PRIMARY KEY REFERENCES tracks(id),
    labels        TEXT,    -- JSON: [{name, catno}]
    styles        TEXT,    -- JSON array
    genres        TEXT,    -- JSON array
    year          INTEGER,
    country       TEXT,
    fetched_at    TEXT
);
```

### Taste profile (`~/.mlx-audiogen/taste_profile.json`)

```python
@dataclass
class TasteProfile:
    # Library signals (broad base)
    top_genres: list[WeightedTag]
    top_artists: list[WeightedTag]
    bpm_range: tuple[float, float]      # 10th-90th percentile
    key_preferences: list[WeightedTag]
    era_distribution: dict[str, float]
    mood_profile: list[WeightedTag]
    style_tags: list[WeightedTag]

    # Generation signals (creative intent, higher weight)
    gen_genres: list[WeightedTag]
    gen_moods: list[WeightedTag]
    gen_instruments: list[WeightedTag]  # derived from existing prompt memory instrument extraction
    kept_ratio: float                    # fraction of generations user favorited/kept (0.0-1.0)
    avg_duration: float
    preferred_models: list[str]

    # Metadata
    library_track_count: int
    generation_count: int
    last_updated: str
    version: int

@dataclass
class WeightedTag:
    name: str
    weight: float  # 0.0-1.0, normalized
```

JSON for taste profile because it's a single document read/written as a whole. SQLite for enrichment cache because it's 16K+ rows with indexed lookups.

### Taste signal sources

**Library signals** (via `signals.py`):
- `top_genres`, `mood_profile`, `style_tags`: aggregated from enrichment DB tags (MusicBrainz + Last.fm + Discogs), weighted by play_count and rating from library metadata
- `top_artists`: from library metadata, weighted by play_count
- `bpm_range`, `key_preferences`: from library metadata (Apple Music/rekordbox BPM + key fields)
- `era_distribution`: from library metadata year field, bucketed by decade

**Generation signals** (via `signals.py`):
- `gen_genres`, `gen_moods`, `gen_instruments`: derived from existing prompt memory's `top_genres`/`top_moods`/`top_instruments` in `prompt_memory.json` style profile (already extracted by `analyze_prompt()` on each generation)
- `kept_ratio`: fraction of generation history entries the user favorited (loved flag)
- `avg_duration`: average `seconds` param from generation history
- `preferred_models`: most-used model names from generation history

## API Endpoints

### New enrichment endpoints (6)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/enrich/tracks` | Enrich batch of tracks (see EnrichRequest model below) |
| GET | `/api/enrich/status` | Current enrichment job status (progress, queue, errors) |
| POST | `/api/enrich/all/{source_id}` | Enrich all tracks in a library source (background job, returns `{"job_id": "..."}`) |
| POST | `/api/enrich/cancel` | Cancel running enrichment job |
| GET | `/api/enrich/track/{track_id}` | Get enrichment data for single track (track_id = enrichment DB integer PK; frontend resolves via artist+title lookup) |
| GET | `/api/enrich/stats` | Cache stats (enriched count, per-source, staleness) |

### New credential endpoints (3)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/credentials/status` | Which services have keys configured (no values) |
| POST | `/api/credentials/{service}` | Store key in Keychain |
| DELETE | `/api/credentials/{service}` | Remove key from Keychain |

### New taste profile endpoints (4)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/taste/profile` | Current taste profile |
| POST | `/api/taste/refresh` | Recompute from all signals |
| GET | `/api/taste/suggestions` | Personalized prompt suggestions |
| PUT | `/api/taste/overrides` | Manual preference overrides |

### Modified existing endpoints (3)

| Path | Change |
|------|--------|
| `GET /api/library/tracks/{source_id}` | Add `enrichment_status` field to each track object in the response array (none/partial/complete/stale) |
| `POST /api/library/generate-prompt` | Incorporate taste profile into prompt generation |
| `POST /api/suggest` | Blend taste profile into suggestion ranking |

Total: 13 new + 3 modified = 16 endpoint changes.

### Request/response models

```python
# POST /api/enrich/tracks
class EnrichRequest(BaseModel):
    track_ids: list[str] = []         # library track IDs (requires source_id)
    source_id: str | None = None      # required when using track_ids
    tracks: list[ArtistTitle] = []    # direct artist/title pairs (alternative)

class ArtistTitle(BaseModel):
    artist: str
    title: str

# POST /api/enrich/all/{source_id} response + GET /api/enrich/status
class EnrichmentJobStatus(BaseModel):
    job_id: str
    status: str          # queued | running | done | error | cancelled
    total: int           # total tracks to enrich
    completed: int       # tracks completed so far
    errors: int          # tracks that failed
    current_track: str | None  # "Artist - Title" currently being enriched
```

### Background enrichment job lifecycle

- Only one enrichment job runs at a time (single-thread, like generation queue)
- `POST /api/enrich/all/{source_id}` returns `{"job_id": "..."}` immediately
- `POST /api/enrich/tracks` with small batches (<20 tracks) runs synchronously; larger batches spawn a background job
- `GET /api/enrich/status` returns current job status (polled by UI every 2s)
- `POST /api/enrich/cancel` sets a cancellation flag; worker checks between tracks
- Completed jobs are cleaned up after 5 minutes (same pattern as generation jobs)

### Track ID mapping (library → enrichment DB)

Library tracks use string IDs (Apple Music integer strings, rekordbox TrackID attributes). The enrichment DB `tracks` table stores `library_source` + `library_track_id` columns for the reverse mapping. When the UI sends track_ids from the Library tab, the server:
1. Looks up each track's artist/title from the LibraryCache
2. Finds or creates the enrichment DB entry via `(artist_norm, title_norm)` UNIQUE constraint
3. Enriches and returns results keyed by the original library track_id

## Web UI Changes

### Settings tab additions

**API Keys section** (below existing LLM settings):
- MusicBrainz: green "Ready" badge (no key needed)
- Last.fm: password input + Save/Clear buttons, green dot when configured
- Discogs: same pattern as Last.fm
- "Auto-enrich on browse" toggle
- Description: "Keys are stored securely in macOS Keychain"

**Taste Profile card**:
- Top genres (weighted tags, amber)
- BPM range (large number), preferred keys (rose tags)
- Mood profile (emerald tags)
- Creative Intent section (fuchsia, from generations — visually distinct)
- Generation stats (count, kept ratio, avg duration)
- Manual override text input
- Refresh + Reset buttons

### Library tab additions

**Track table — Enriched column**:
- 3 small dots per row (MusicBrainz, Last.fm, Discogs)
- Green = fetched, gray = not fetched
- Tooltip on hover shows source name

**Track table — Tags column**:
- Color-coded tags from enrichment data (using 14-category schema)
- "Not enriched" italic text when no data

**Action bar**:
- New "Enrich Selected" button (purple) alongside existing Generate Like This / Train on These

### New TypeScript types

- EnrichmentStatus, EnrichedTrackData, CredentialStatus
- TasteProfile, WeightedTag, TasteOverrides
- EnrichmentJobStatus, EnrichmentStats

### New Zustand store slices

- `enrichment`: job status, batch progress
- `credentials`: per-service status (configured/missing)
- `taste`: profile data, loading state

## Credential Management

### Three-tier fallback

1. **macOS Keychain** (via `keyring` library) — primary, encrypted by OS
2. **Environment variables** (`LASTFM_API_KEY`, `DISCOGS_TOKEN`) — fallback for CI/automation/path exports
3. **Not configured** — graceful degradation, enrichment features disabled for that source

### Implementation

- `credentials.py` at package root (`mlx_audiogen/credentials.py`): `get_credential(service)`, `set_credential(service, value)`, `delete_credential(service)`, `get_status()`
- Service names: `lastfm_api_key`, `discogs_token`
- Keyring service name: `mlx-audiogen`
- Settings UI writes to Keychain via API, never to files
- Masked display in UI (last 4 chars visible)

## Error Handling

- **Per-API rate limiters**: token bucket — 1/sec MusicBrainz, 5/sec Last.fm, 1/sec Discogs
- **429 responses**: automatic backoff (double interval, retry after cooldown)
- **No match found**: mark in DB as "attempted, no results" — skip for 30 days
- **Network errors**: retry 3x with exponential backoff, then skip and continue
- **Missing API keys**: skip that source, enrich with available sources only
- **Partial enrichment**: a track with only MusicBrainz data is valid
- **Stale data**: 90-day TTL, re-enrich on next access after expiry. Stale flag only makes a track eligible for re-fetch — existing cached data is never overwritten with blank. If re-fetch fails or returns no results, the existing enrichment data is preserved

## Testing Strategy

- Mock all HTTP calls (httpx mock or monkeypatch) — no real API hits in unit tests
- Enrichment DB: in-memory SQLite (`:memory:`) for test isolation
- Credential tests: monkeypatch `keyring` module
- Taste engine: synthetic library + generation data → verify profile computation
- Integration test (marked `integration`): single real MusicBrainz API call
- Target: ~50-60 new tests

## Dependencies

New in `pyproject.toml`:
- `httpx>=0.27.0` — async HTTP client (explicit, was indirect via FastAPI)
- `keyring>=25.0.0` — macOS Keychain + cross-platform credential storage

No other new dependencies. SQLite is stdlib.

## Gitignore

Add to `.gitignore`:
- `enrichment.db` — user-specific cached enrichment data
- `.superpowers/` — brainstorming mockups
