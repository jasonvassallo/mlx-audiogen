# Phase 9g-3: Web Enrichment + Taste Learning Engine — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich library track metadata from MusicBrainz, Last.fm, and Discogs APIs, and build a taste learning engine that personalizes suggestions and prompt generation.

**Architecture:** New `enrichment/` and `taste/` subpackages under `mlx_audiogen/library/`, SQLite cache for enrichment data, JSON file for taste profiles. New `credentials.py` for Keychain-based API key storage. 13 new + 3 modified API endpoints. Settings UI gains API keys section and taste profile card. Library tab gains enrichment status dots and "Enrich Selected" button.

**Tech Stack:** Python 3.11+, httpx (async HTTP), keyring (macOS Keychain), SQLite3 (stdlib), FastAPI, React 19, TypeScript, Zustand, Tailwind CSS v4.

**Spec:** `docs/superpowers/specs/2026-03-14-phase-9g3-web-enrichment-taste-engine-design.md`

**Full QA command:** `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen && uv run ruff format . && uv run ruff check . && uv run mypy mlx_audiogen/ && uv run bandit -r mlx_audiogen/ -c pyproject.toml && uv run pip-audit && uv run pytest && cd web && npm run build`

---

## File Structure

### New files (14)

| File | Responsibility |
|------|---------------|
| `mlx_audiogen/credentials.py` | Keychain + env var credential management (get/set/delete/status) |
| `mlx_audiogen/library/enrichment/__init__.py` | Package init, re-exports EnrichmentManager |
| `mlx_audiogen/library/enrichment/enrichment_db.py` | SQLite schema, CRUD for tracks + 3 source tables |
| `mlx_audiogen/library/enrichment/rate_limiter.py` | Per-API token bucket rate limiter |
| `mlx_audiogen/library/enrichment/clients.py` | httpx async HTTP client factory with User-Agent |
| `mlx_audiogen/library/enrichment/musicbrainz.py` | MusicBrainz search + tag extraction |
| `mlx_audiogen/library/enrichment/lastfm.py` | Last.fm track.getInfo + artist.getSimilar |
| `mlx_audiogen/library/enrichment/discogs.py` | Discogs search + master release details |
| `mlx_audiogen/library/enrichment/manager.py` | Orchestrator: cache check, fetch missing, merge, store |
| `mlx_audiogen/library/taste/__init__.py` | Package init |
| `mlx_audiogen/library/taste/profile.py` | TasteProfile + WeightedTag dataclasses |
| `mlx_audiogen/library/taste/signals.py` | Signal collectors (library metadata + generation history) |
| `mlx_audiogen/library/taste/engine.py` | TasteEngine: compute, update, query taste profiles |
| `tests/test_enrichment.py` | Tests for enrichment DB, rate limiter, API clients, manager |
| `tests/test_credentials.py` | Tests for credential manager (monkeypatched keyring) |
| `tests/test_taste.py` | Tests for taste profile, signals, engine |

### Modified files (8)

| File | Changes |
|------|---------|
| `pyproject.toml` | Add `httpx>=0.27.0` and `keyring>=25.0.0` to dependencies |
| `mlx_audiogen/library/models.py` | (no changes needed — enrichment data lives in separate SQLite DB, not on TrackInfo. Spec deviation: cleaner separation.) |
| `mlx_audiogen/library/description_gen.py` | Add optional `enrichment` dict parameter to `generate_description()` |
| `mlx_audiogen/server/app.py` | Add 13 new endpoints, modify 3 existing, startup init |
| `web/src/types/api.ts` | Add enrichment, credential, taste TypeScript types |
| `web/src/api/client.ts` | Add 13 new fetch wrappers |
| `web/src/components/SettingsPanel.tsx` | Add API Keys section + Taste Profile card |
| `web/src/components/LibraryPanel.tsx` | Add enrichment dots column, Tags column, Enrich Selected button |
| `.gitignore` | Add `enrichment.db` |

---

## Chunk 1: Foundation (Dependencies + Credentials + Enrichment DB)

### Task 1: Add dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add httpx and keyring to dependencies**

In `pyproject.toml`, add to the `dependencies` list (after `defusedxml>=0.7`):

```toml
"httpx>=0.27.0",
"keyring>=25.0.0",
```

- [ ] **Step 2: Add enrichment.db to .gitignore**

Append to `.gitignore`:

```
# Enrichment cache (user-specific)
enrichment.db
```

- [ ] **Step 3: Sync environment**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen && uv sync`
Expected: Success, httpx and keyring installed.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml .gitignore uv.lock
git commit -m "deps: add httpx and keyring for web enrichment"
```

---

### Task 2: Credential manager

**Files:**
- Create: `mlx_audiogen/credentials.py`
- Create: `tests/test_credentials.py`

- [ ] **Step 1: Write failing tests for credential manager**

Create `tests/test_credentials.py`:

```python
"""Tests for Keychain + env var credential management."""

import pytest

from mlx_audiogen.credentials import CredentialManager


@pytest.fixture
def cred_manager(monkeypatch):
    """CredentialManager with monkeypatched keyring (no real Keychain)."""
    store = {}

    def fake_get(service, username):
        return store.get(f"{service}:{username}")

    def fake_set(service, username, password):
        store[f"{service}:{username}"] = password

    def fake_delete(service, username):
        key = f"{service}:{username}"
        if key in store:
            del store[key]
        else:
            raise Exception("not found")

    monkeypatch.setattr("mlx_audiogen.credentials.keyring.get_password", fake_get)
    monkeypatch.setattr("mlx_audiogen.credentials.keyring.set_password", fake_set)
    monkeypatch.setattr("mlx_audiogen.credentials.keyring.delete_password", fake_delete)

    return CredentialManager()


class TestCredentialManager:
    def test_get_missing_returns_none(self, cred_manager):
        assert cred_manager.get("lastfm_api_key") is None

    def test_set_and_get(self, cred_manager):
        cred_manager.set("lastfm_api_key", "abc123")
        assert cred_manager.get("lastfm_api_key") == "abc123"

    def test_delete(self, cred_manager):
        cred_manager.set("discogs_token", "xyz")
        cred_manager.delete("discogs_token")
        assert cred_manager.get("discogs_token") is None

    def test_delete_missing_no_error(self, cred_manager):
        # Should not raise
        cred_manager.delete("nonexistent")

    def test_env_var_fallback(self, cred_manager, monkeypatch):
        monkeypatch.setenv("LASTFM_API_KEY", "from_env")
        assert cred_manager.get("lastfm_api_key") == "from_env"

    def test_keychain_takes_priority_over_env(self, cred_manager, monkeypatch):
        monkeypatch.setenv("LASTFM_API_KEY", "from_env")
        cred_manager.set("lastfm_api_key", "from_keychain")
        assert cred_manager.get("lastfm_api_key") == "from_keychain"

    def test_status(self, cred_manager):
        status = cred_manager.status()
        assert status["musicbrainz"] is True  # always ready (no key needed)
        assert status["lastfm"] is False
        assert status["discogs"] is False

    def test_status_after_set(self, cred_manager):
        cred_manager.set("lastfm_api_key", "key123")
        status = cred_manager.status()
        assert status["lastfm"] is True

    def test_invalid_service_name(self, cred_manager):
        with pytest.raises(ValueError, match="Unknown service"):
            cred_manager.set("spotify_key", "value")

    def test_masked_value(self, cred_manager):
        cred_manager.set("lastfm_api_key", "abcdef123456")
        masked = cred_manager.get_masked("lastfm_api_key")
        assert masked == "••••••••3456"

    def test_masked_short_value(self, cred_manager):
        cred_manager.set("lastfm_api_key", "abc")
        masked = cred_manager.get_masked("lastfm_api_key")
        assert masked == "•••"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen && uv run pytest tests/test_credentials.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mlx_audiogen.credentials'`

- [ ] **Step 3: Implement credential manager**

Create `mlx_audiogen/credentials.py`:

```python
"""Keychain-based credential manager with environment variable fallback.

Stores API keys securely in macOS Keychain via the `keyring` library.
Falls back to environment variables when Keychain entry is not found.
"""

from __future__ import annotations

import os

import keyring

SERVICE_NAME = "mlx-audiogen"

# Map of internal service names to env var names
_SERVICES: dict[str, str] = {
    "lastfm_api_key": "LASTFM_API_KEY",
    "discogs_token": "DISCOGS_TOKEN",
}


class CredentialManager:
    """Manage API credentials via macOS Keychain + env var fallback."""

    def get(self, service: str) -> str | None:
        """Get a credential. Checks Keychain first, then env var."""
        self._validate_service(service)
        # 1. Try Keychain
        value = keyring.get_password(SERVICE_NAME, service)
        if value is not None:
            return value
        # 2. Fallback to environment variable
        env_var = _SERVICES[service]
        return os.environ.get(env_var)

    def set(self, service: str, value: str) -> None:
        """Store a credential in the Keychain."""
        self._validate_service(service)
        keyring.set_password(SERVICE_NAME, service, value)

    def delete(self, service: str) -> None:
        """Remove a credential from the Keychain. No-op if missing."""
        self._validate_service(service)
        try:
            keyring.delete_password(SERVICE_NAME, service)
        except Exception:
            pass  # Keyring raises if entry doesn't exist

    def get_masked(self, service: str) -> str | None:
        """Get a masked version of the credential (last 4 chars visible)."""
        value = self.get(service)
        if value is None:
            return None
        if len(value) <= 4:
            return "\u2022" * len(value)
        return "\u2022" * (len(value) - 4) + value[-4:]

    def status(self) -> dict[str, bool]:
        """Return which services have credentials configured.

        MusicBrainz is always True (no API key required).
        """
        return {
            "musicbrainz": True,
            "lastfm": self.get("lastfm_api_key") is not None,
            "discogs": self.get("discogs_token") is not None,
        }

    def _validate_service(self, service: str) -> None:
        if service not in _SERVICES:
            raise ValueError(f"Unknown service: {service!r}. Valid: {list(_SERVICES)}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen && uv run pytest tests/test_credentials.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mlx_audiogen/credentials.py tests/test_credentials.py
git commit -m "feat: add Keychain-based credential manager with env var fallback"
```

---

### Task 3: Enrichment SQLite database

**Files:**
- Create: `mlx_audiogen/library/enrichment/__init__.py`
- Create: `mlx_audiogen/library/enrichment/enrichment_db.py`
- Add enrichment DB tests to: `tests/test_enrichment.py`

- [ ] **Step 1: Create package init**

Create `mlx_audiogen/library/enrichment/__init__.py`:

```python
"""Web enrichment subpackage — MusicBrainz, Last.fm, Discogs metadata."""
```

- [ ] **Step 2: Write failing tests for enrichment DB**

Create `tests/test_enrichment.py`:

```python
"""Tests for enrichment database, rate limiter, and API clients."""

import json

import pytest

from mlx_audiogen.library.enrichment.enrichment_db import EnrichmentDB


@pytest.fixture
def db():
    """In-memory enrichment DB for test isolation."""
    return EnrichmentDB(":memory:")


class TestEnrichmentDB:
    def test_get_or_create_track(self, db):
        tid = db.get_or_create_track("Solomun", "Midnight Express")
        assert isinstance(tid, int)
        # Same artist/title returns same ID
        tid2 = db.get_or_create_track("SOLOMUN", "midnight express")
        assert tid2 == tid

    def test_get_or_create_with_library_id(self, db):
        tid = db.get_or_create_track(
            "deadmau5", "Strobe",
            library_source="apple_music", library_track_id="12345",
        )
        assert tid > 0

    def test_store_and_get_musicbrainz(self, db):
        tid = db.get_or_create_track("Solomun", "Midnight Express")
        db.store_musicbrainz(tid, {
            "tags": [{"name": "house", "count": 100}],
            "genres": ["house", "electronic"],
            "artist_mbid": "abc-123",
        })
        data = db.get_musicbrainz(tid)
        assert data is not None
        assert json.loads(data["genres"]) == ["house", "electronic"]

    def test_store_and_get_lastfm(self, db):
        tid = db.get_or_create_track("deadmau5", "Strobe")
        db.store_lastfm(tid, {
            "tags": [{"name": "progressive house", "count": 200}],
            "listeners": 500000,
            "play_count": 3000000,
        })
        data = db.get_lastfm(tid)
        assert data is not None
        assert data["listeners"] == 500000

    def test_store_and_get_discogs(self, db):
        tid = db.get_or_create_track("CamelPhat", "Cola")
        db.store_discogs(tid, {
            "labels": [{"name": "Defected", "catno": "DFTD123"}],
            "styles": ["tech house", "dance"],
            "genres": ["electronic"],
        })
        data = db.get_discogs(tid)
        assert data is not None
        assert json.loads(data["styles"]) == ["tech house", "dance"]

    def test_get_missing_returns_none(self, db):
        tid = db.get_or_create_track("Unknown", "Track")
        assert db.get_musicbrainz(tid) is None

    def test_enrichment_status_none(self, db):
        tid = db.get_or_create_track("A", "B")
        assert db.get_enrichment_status(tid) == "none"

    def test_enrichment_status_partial(self, db):
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"tags": [], "genres": []})
        assert db.get_enrichment_status(tid) == "partial"

    def test_enrichment_status_complete(self, db):
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"tags": [], "genres": []})
        db.store_lastfm(tid, {"tags": [], "listeners": 0})
        db.store_discogs(tid, {"labels": [], "styles": [], "genres": []})
        assert db.get_enrichment_status(tid) == "complete"

    def test_is_stale(self, db):
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"tags": [], "genres": []})
        # Just stored — should not be stale
        assert db.is_stale(tid, "musicbrainz", ttl_days=90) is False

    def test_update_preserves_existing(self, db):
        """Updating one source does not affect others."""
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"tags": [{"name": "rock"}], "genres": ["rock"]})
        db.store_lastfm(tid, {"tags": [{"name": "indie"}], "listeners": 100})
        # Update musicbrainz
        db.store_musicbrainz(tid, {"tags": [{"name": "pop"}], "genres": ["pop"]})
        # lastfm should be unchanged
        lfm = db.get_lastfm(tid)
        assert json.loads(lfm["tags"]) == [{"name": "indie"}]

    def test_get_stats(self, db):
        db.get_or_create_track("A", "B")
        db.get_or_create_track("C", "D")
        tid1 = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid1, {"tags": [], "genres": []})
        stats = db.get_stats()
        assert stats["total_tracks"] == 2
        assert stats["musicbrainz_count"] == 1

    def test_find_track_by_library_id(self, db):
        tid = db.get_or_create_track(
            "X", "Y", library_source="rekordbox", library_track_id="RB42"
        )
        found = db.find_by_library_id("rekordbox", "RB42")
        assert found == tid

    def test_find_track_by_library_id_missing(self, db):
        assert db.find_by_library_id("apple_music", "999") is None

    def test_get_all_enrichment(self, db):
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"tags": [{"name": "jazz"}], "genres": ["jazz"]})
        db.store_lastfm(tid, {"tags": [], "listeners": 50})
        result = db.get_all_enrichment(tid)
        assert "musicbrainz" in result
        assert "lastfm" in result
        assert result["discogs"] is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_enrichment.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement enrichment DB**

Create `mlx_audiogen/library/enrichment/enrichment_db.py`:

```python
"""SQLite cache for web enrichment data.

Stores metadata from MusicBrainz, Last.fm, and Discogs in separate tables
with independent refresh cycles. Keyed by normalized artist+title for
deduplication across library sources.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id              INTEGER PRIMARY KEY,
    artist          TEXT NOT NULL,
    title           TEXT NOT NULL,
    artist_norm     TEXT NOT NULL,
    title_norm      TEXT NOT NULL,
    musicbrainz_id  TEXT,
    library_source  TEXT,
    library_track_id TEXT,
    UNIQUE(artist_norm, title_norm)
);
CREATE INDEX IF NOT EXISTS idx_tracks_norm
    ON tracks(artist_norm, title_norm);
CREATE INDEX IF NOT EXISTS idx_tracks_library
    ON tracks(library_source, library_track_id);

CREATE TABLE IF NOT EXISTS musicbrainz (
    track_id        INTEGER PRIMARY KEY REFERENCES tracks(id),
    tags            TEXT,
    genres          TEXT,
    release_group   TEXT,
    artist_mbid     TEXT,
    similar_artists TEXT,
    fetched_at      TEXT
);

CREATE TABLE IF NOT EXISTS lastfm (
    track_id        INTEGER PRIMARY KEY REFERENCES tracks(id),
    tags            TEXT,
    similar_tracks  TEXT,
    similar_artists TEXT,
    play_count      INTEGER,
    listeners       INTEGER,
    fetched_at      TEXT
);

CREATE TABLE IF NOT EXISTS discogs (
    track_id        INTEGER PRIMARY KEY REFERENCES tracks(id),
    labels          TEXT,
    styles          TEXT,
    genres          TEXT,
    year            INTEGER,
    country         TEXT,
    fetched_at      TEXT
);
"""


def _normalize(s: str) -> str:
    """Lowercase and strip whitespace for deduplication."""
    return s.strip().lower()


class EnrichmentDB:
    """SQLite-backed enrichment cache."""

    def __init__(self, path: str | Path = ""):
        if path == ":memory:":
            self._conn = sqlite3.connect(":memory:")
        else:
            db_path = Path(path) if path else (
                Path.home() / ".mlx-audiogen" / "enrichment.db"
            )
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def get_or_create_track(
        self,
        artist: str,
        title: str,
        library_source: str | None = None,
        library_track_id: str | None = None,
    ) -> int:
        """Find or insert a track, returning its integer ID."""
        a_norm = _normalize(artist)
        t_norm = _normalize(title)
        row = self._conn.execute(
            "SELECT id FROM tracks WHERE artist_norm = ? AND title_norm = ?",
            (a_norm, t_norm),
        ).fetchone()
        if row:
            tid = row["id"]
            # Update library mapping if provided and not already set
            if library_source and library_track_id:
                self._conn.execute(
                    """UPDATE tracks SET library_source = ?, library_track_id = ?
                       WHERE id = ? AND library_source IS NULL""",
                    (library_source, library_track_id, tid),
                )
                self._conn.commit()
            return tid
        cur = self._conn.execute(
            """INSERT INTO tracks (artist, title, artist_norm, title_norm,
                                   library_source, library_track_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (artist, title, a_norm, t_norm, library_source, library_track_id),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def find_by_library_id(
        self, library_source: str, library_track_id: str
    ) -> Optional[int]:
        """Look up enrichment track ID by library source + track ID."""
        row = self._conn.execute(
            "SELECT id FROM tracks WHERE library_source = ? AND library_track_id = ?",
            (library_source, library_track_id),
        ).fetchone()
        return row["id"] if row else None

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # -- MusicBrainz --

    def store_musicbrainz(self, track_id: int, data: dict) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO musicbrainz
               (track_id, tags, genres, release_group, artist_mbid,
                similar_artists, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                track_id,
                json.dumps(data.get("tags", [])),
                json.dumps(data.get("genres", [])),
                data.get("release_group"),
                data.get("artist_mbid"),
                json.dumps(data.get("similar_artists", [])),
                self._now_iso(),
            ),
        )
        self._conn.commit()

    def get_musicbrainz(self, track_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM musicbrainz WHERE track_id = ?", (track_id,)
        ).fetchone()
        return dict(row) if row else None

    # -- Last.fm --

    def store_lastfm(self, track_id: int, data: dict) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO lastfm
               (track_id, tags, similar_tracks, similar_artists,
                play_count, listeners, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                track_id,
                json.dumps(data.get("tags", [])),
                json.dumps(data.get("similar_tracks", [])),
                json.dumps(data.get("similar_artists", [])),
                data.get("play_count"),
                data.get("listeners"),
                self._now_iso(),
            ),
        )
        self._conn.commit()

    def get_lastfm(self, track_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM lastfm WHERE track_id = ?", (track_id,)
        ).fetchone()
        return dict(row) if row else None

    # -- Discogs --

    def store_discogs(self, track_id: int, data: dict) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO discogs
               (track_id, labels, styles, genres, year, country, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                track_id,
                json.dumps(data.get("labels", [])),
                json.dumps(data.get("styles", [])),
                json.dumps(data.get("genres", [])),
                data.get("year"),
                data.get("country"),
                self._now_iso(),
            ),
        )
        self._conn.commit()

    def get_discogs(self, track_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM discogs WHERE track_id = ?", (track_id,)
        ).fetchone()
        return dict(row) if row else None

    # -- Aggregation --

    def get_all_enrichment(self, track_id: int) -> dict:
        """Get all enrichment data for a track (all 3 sources)."""
        return {
            "musicbrainz": self.get_musicbrainz(track_id),
            "lastfm": self.get_lastfm(track_id),
            "discogs": self.get_discogs(track_id),
        }

    def get_enrichment_status(self, track_id: int) -> str:
        """Return 'none', 'partial', or 'complete'."""
        sources = self.get_all_enrichment(track_id)
        filled = sum(1 for v in sources.values() if v is not None)
        if filled == 0:
            return "none"
        if filled == 3:
            return "complete"
        return "partial"

    def is_stale(self, track_id: int, source: str, ttl_days: int = 90) -> bool:
        """Check if a source's data is older than ttl_days."""
        table = {"musicbrainz": "musicbrainz", "lastfm": "lastfm", "discogs": "discogs"}
        if source not in table:
            return True
        row = self._conn.execute(
            f"SELECT fetched_at FROM {table[source]} WHERE track_id = ?",
            (track_id,),
        ).fetchone()
        if row is None:
            return True
        fetched = datetime.fromisoformat(row["fetched_at"])
        return datetime.now(timezone.utc) - fetched > timedelta(days=ttl_days)

    def get_stats(self) -> dict:
        """Cache-wide statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        mb = self._conn.execute("SELECT COUNT(*) FROM musicbrainz").fetchone()[0]
        lfm = self._conn.execute("SELECT COUNT(*) FROM lastfm").fetchone()[0]
        dc = self._conn.execute("SELECT COUNT(*) FROM discogs").fetchone()[0]
        return {
            "total_tracks": total,
            "musicbrainz_count": mb,
            "lastfm_count": lfm,
            "discogs_count": dc,
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_enrichment.py -v`
Expected: All 15 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add mlx_audiogen/library/enrichment/ tests/test_enrichment.py
git commit -m "feat: add enrichment SQLite cache with 3-source schema"
```

---

### Task 4: Per-API rate limiter

**Files:**
- Create: `mlx_audiogen/library/enrichment/rate_limiter.py`
- Add tests to: `tests/test_enrichment.py`

- [ ] **Step 1: Write failing tests for rate limiter**

Append to `tests/test_enrichment.py`:

```python
import asyncio

from mlx_audiogen.library.enrichment.rate_limiter import ApiRateLimiter


class TestApiRateLimiter:
    def test_allows_within_limit(self):
        limiter = ApiRateLimiter(max_per_second=5.0)
        asyncio.run(limiter.acquire())

    def test_tracks_separate_apis(self):
        mb = ApiRateLimiter(max_per_second=1.0)
        lfm = ApiRateLimiter(max_per_second=5.0)
        async def check():
            await mb.acquire()
            await lfm.acquire()
        asyncio.run(check())
```

- [ ] **Step 2: Implement rate limiter**

Create `mlx_audiogen/library/enrichment/rate_limiter.py`:

```python
"""Async token-bucket rate limiter for external API calls.

Each API gets its own limiter instance with configurable requests/second.
"""

from __future__ import annotations

import asyncio
import time


class ApiRateLimiter:
    """Token-bucket rate limiter. Call ``await acquire()`` before each request."""

    def __init__(self, max_per_second: float):
        self._interval = 1.0 / max_per_second
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until a request is allowed, then mark the slot as used."""
        async with self._lock:
            now = time.monotonic()
            wait = self._interval - (now - self._last_call)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = time.monotonic()
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_enrichment.py::TestApiRateLimiter -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add mlx_audiogen/library/enrichment/rate_limiter.py tests/test_enrichment.py
git commit -m "feat: add per-API async token-bucket rate limiter"
```

---

## Chunk 2: API Clients (MusicBrainz + Last.fm + Discogs)

### Task 5: HTTP client factory

**Files:**
- Create: `mlx_audiogen/library/enrichment/clients.py`

- [ ] **Step 1: Implement shared HTTP client**

```python
"""Shared httpx async client factory for enrichment APIs.

Sets a proper User-Agent (required by MusicBrainz) and configurable timeouts.
"""

from __future__ import annotations

import httpx

_USER_AGENT = "mlx-audiogen/0.1.0 (https://github.com/jasonvassallo/mlx-audiogen)"


def create_client(timeout: float = 10.0) -> httpx.AsyncClient:
    """Create an httpx async client with standard headers."""
    return httpx.AsyncClient(
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        timeout=timeout,
        follow_redirects=True,
    )
```

- [ ] **Step 2: Commit**

```bash
git add mlx_audiogen/library/enrichment/clients.py
git commit -m "feat: add httpx async client factory for enrichment APIs"
```

---

### Task 6: MusicBrainz client

**Files:**
- Create: `mlx_audiogen/library/enrichment/musicbrainz.py`
- Add tests to: `tests/test_enrichment.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_enrichment.py`:

```python
import httpx
from mlx_audiogen.library.enrichment.musicbrainz import search_musicbrainz


class TestMusicBrainz:
    def test_parse_response(self):
        """Test parsing a realistic MusicBrainz response."""
        fake_response = {
            "recordings": [{
                "id": "rec-123",
                "title": "Strobe",
                "artist-credit": [{"artist": {"id": "art-456", "name": "deadmau5"}}],
                "releases": [{"id": "rel-789", "release-group": {"id": "rg-101"}}],
                "tags": [{"name": "progressive house", "count": 5}, {"name": "electronic", "count": 3}],
            }]
        }
        result = _parse_musicbrainz_response(fake_response)
        assert result["artist_mbid"] == "art-456"
        assert "progressive house" in [t["name"] for t in result["tags"]]

    def test_parse_empty_response(self):
        result = _parse_musicbrainz_response({"recordings": []})
        assert result is None


from mlx_audiogen.library.enrichment.musicbrainz import _parse_musicbrainz_response
```

- [ ] **Step 2: Implement MusicBrainz client**

Create `mlx_audiogen/library/enrichment/musicbrainz.py`:

```python
"""MusicBrainz API client — search recordings and extract tags.

No API key required. Rate limit: 1 request/second (enforced by caller's rate limiter).
API docs: https://musicbrainz.org/doc/MusicBrainz_API
"""

from __future__ import annotations

from typing import Optional

import httpx

from .clients import create_client
from .rate_limiter import ApiRateLimiter

_BASE = "https://musicbrainz.org/ws/2"


def _parse_musicbrainz_response(data: dict) -> Optional[dict]:
    """Extract structured data from a MusicBrainz recording search response."""
    recordings = data.get("recordings", [])
    if not recordings:
        return None
    rec = recordings[0]
    tags = [{"name": t["name"], "count": t.get("count", 0)} for t in rec.get("tags", [])]
    genres = list({t["name"] for t in tags})
    artist_credit = rec.get("artist-credit", [])
    artist_mbid = artist_credit[0]["artist"]["id"] if artist_credit else None
    releases = rec.get("releases", [])
    rg = releases[0]["release-group"]["id"] if releases and "release-group" in releases[0] else None
    return {
        "tags": tags,
        "genres": genres,
        "release_group": rg,
        "artist_mbid": artist_mbid,
        "similar_artists": [],
    }


async def search_musicbrainz(
    artist: str,
    title: str,
    rate_limiter: ApiRateLimiter,
    client: httpx.AsyncClient | None = None,
) -> Optional[dict]:
    """Search MusicBrainz for a recording and return structured metadata."""
    await rate_limiter.acquire()
    own_client = client is None
    if own_client:
        client = create_client()
    try:
        query = f'recording:"{title}" AND artist:"{artist}"'
        resp = await client.get(
            f"{_BASE}/recording",
            params={"query": query, "fmt": "json", "limit": 1},
        )
        if resp.status_code == 429:
            return None  # Rate limited — caller should retry later
        resp.raise_for_status()
        return _parse_musicbrainz_response(resp.json())
    except httpx.HTTPError:
        return None
    finally:
        if own_client:
            await client.aclose()
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_enrichment.py::TestMusicBrainz -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add mlx_audiogen/library/enrichment/musicbrainz.py tests/test_enrichment.py
git commit -m "feat: add MusicBrainz API client with recording search"
```

---

### Task 7: Last.fm client

**Files:**
- Create: `mlx_audiogen/library/enrichment/lastfm.py`
- Add tests to: `tests/test_enrichment.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_enrichment.py`:

```python
from mlx_audiogen.library.enrichment.lastfm import _parse_lastfm_track_response


class TestLastFm:
    def test_parse_track_response(self):
        fake = {
            "track": {
                "name": "Strobe",
                "listeners": "500000",
                "playcount": "3000000",
                "toptags": {"tag": [
                    {"name": "progressive house", "count": "100"},
                    {"name": "electronic", "count": "80"},
                ]},
                "similar": {"track": [
                    {"name": "Ghosts", "artist": {"name": "deadmau5"}, "match": "0.9"},
                ]},
            }
        }
        result = _parse_lastfm_track_response(fake)
        assert result["listeners"] == 500000
        assert result["play_count"] == 3000000
        assert len(result["tags"]) == 2

    def test_parse_error_response(self):
        result = _parse_lastfm_track_response({"error": 6, "message": "Track not found"})
        assert result is None
```

- [ ] **Step 2: Implement Last.fm client**

Create `mlx_audiogen/library/enrichment/lastfm.py`:

```python
"""Last.fm API client — track info and crowd-sourced tags.

Requires a free API key. Rate limit: 5 requests/second.
API docs: https://www.last.fm/api
"""

from __future__ import annotations

from typing import Optional

import httpx

from .clients import create_client
from .rate_limiter import ApiRateLimiter

_BASE = "https://ws.audioscrobbler.com/2.0/"


def _parse_lastfm_track_response(data: dict) -> Optional[dict]:
    """Extract structured data from Last.fm track.getInfo response."""
    if "error" in data:
        return None
    track = data.get("track", {})
    if not track:
        return None
    raw_tags = track.get("toptags", {}).get("tag", [])
    tags = [{"name": t["name"], "count": int(t.get("count", 0))} for t in raw_tags]
    similar_raw = track.get("similar", {}).get("track", [])
    similar_tracks = [
        {"artist": s["artist"]["name"], "title": s["name"], "match_score": float(s.get("match", 0))}
        for s in similar_raw
    ]
    return {
        "tags": tags,
        "similar_tracks": similar_tracks,
        "similar_artists": [],
        "play_count": int(track.get("playcount", 0)),
        "listeners": int(track.get("listeners", 0)),
    }


async def search_lastfm(
    artist: str,
    title: str,
    api_key: str,
    rate_limiter: ApiRateLimiter,
    client: httpx.AsyncClient | None = None,
) -> Optional[dict]:
    """Get track info from Last.fm."""
    await rate_limiter.acquire()
    own_client = client is None
    if own_client:
        client = create_client()
    try:
        resp = await client.get(
            _BASE,
            params={
                "method": "track.getInfo",
                "artist": artist,
                "track": title,
                "api_key": api_key,
                "format": "json",
            },
        )
        if resp.status_code == 429:
            return None
        resp.raise_for_status()
        return _parse_lastfm_track_response(resp.json())
    except httpx.HTTPError:
        return None
    finally:
        if own_client:
            await client.aclose()
```

- [ ] **Step 3: Run tests and commit**

Run: `uv run pytest tests/test_enrichment.py::TestLastFm -v`
Expected: PASS.

```bash
git add mlx_audiogen/library/enrichment/lastfm.py tests/test_enrichment.py
git commit -m "feat: add Last.fm API client with track info + tags"
```

---

### Task 8: Discogs client

**Files:**
- Create: `mlx_audiogen/library/enrichment/discogs.py`
- Add tests to: `tests/test_enrichment.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_enrichment.py`:

```python
from mlx_audiogen.library.enrichment.discogs import _parse_discogs_search_response


class TestDiscogs:
    def test_parse_search_response(self):
        fake = {
            "results": [{
                "id": 12345,
                "type": "master",
                "title": "CamelPhat & Elderbrook - Cola",
                "genre": ["Electronic"],
                "style": ["Tech House", "Dance"],
                "label": ["Defected Records"],
                "catno": "DFTD123",
                "year": "2017",
                "country": "UK",
            }]
        }
        result = _parse_discogs_search_response(fake)
        assert result is not None
        assert result["genres"] == ["Electronic"]
        assert "Tech House" in result["styles"]
        assert result["year"] == 2017

    def test_parse_empty_results(self):
        result = _parse_discogs_search_response({"results": []})
        assert result is None
```

- [ ] **Step 2: Implement Discogs client**

Create `mlx_audiogen/library/enrichment/discogs.py`:

```python
"""Discogs API client — search releases for labels, styles, and taxonomy.

Requires a personal access token. Rate limit: 60 req/min (~1/sec).
API docs: https://www.discogs.com/developers
"""

from __future__ import annotations

from typing import Optional

import httpx

from .clients import create_client
from .rate_limiter import ApiRateLimiter

_BASE = "https://api.discogs.com"


def _parse_discogs_search_response(data: dict) -> Optional[dict]:
    """Extract structured data from Discogs search response."""
    results = data.get("results", [])
    if not results:
        return None
    r = results[0]
    labels_raw = r.get("label", [])
    catno = r.get("catno", "")
    labels = [{"name": lbl, "catno": catno} for lbl in labels_raw]
    year_raw = r.get("year", "")
    year = int(year_raw) if year_raw and str(year_raw).isdigit() else None
    return {
        "labels": labels,
        "styles": r.get("style", []),
        "genres": r.get("genre", []),
        "year": year,
        "country": r.get("country"),
    }


async def search_discogs(
    artist: str,
    title: str,
    token: str,
    rate_limiter: ApiRateLimiter,
    client: httpx.AsyncClient | None = None,
) -> Optional[dict]:
    """Search Discogs for a release matching artist + title."""
    await rate_limiter.acquire()
    own_client = client is None
    if own_client:
        client = create_client()
    try:
        resp = await client.get(
            f"{_BASE}/database/search",
            params={
                "q": f"{artist} {title}",
                "type": "master",
                "per_page": 1,
            },
            headers={"Authorization": f"Discogs token={token}"},
        )
        if resp.status_code == 429:
            return None
        resp.raise_for_status()
        return _parse_discogs_search_response(resp.json())
    except httpx.HTTPError:
        return None
    finally:
        if own_client:
            await client.aclose()
```

- [ ] **Step 3: Run tests and commit**

Run: `uv run pytest tests/test_enrichment.py::TestDiscogs -v`
Expected: PASS.

```bash
git add mlx_audiogen/library/enrichment/discogs.py tests/test_enrichment.py
git commit -m "feat: add Discogs API client with release search"
```

---

## Chunk 3: Enrichment Manager + Taste Engine

### Task 9: Enrichment manager (orchestrator)

**Files:**
- Create: `mlx_audiogen/library/enrichment/manager.py`
- Add tests to: `tests/test_enrichment.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_enrichment.py`:

```python
from unittest.mock import AsyncMock, patch

from mlx_audiogen.library.enrichment.manager import EnrichmentManager


@pytest.fixture
def manager():
    """EnrichmentManager with in-memory DB and mock credentials."""
    db = EnrichmentDB(":memory:")
    m = EnrichmentManager.__new__(EnrichmentManager)
    m._db = db
    m._cred = None
    m._mb_limiter = ApiRateLimiter(max_per_second=100)
    m._lfm_limiter = ApiRateLimiter(max_per_second=100)
    m._dc_limiter = ApiRateLimiter(max_per_second=100)
    m._cancelled = False
    return m


class TestEnrichmentManager:
    def test_enrich_track_musicbrainz_only(self, manager):
        """When no Last.fm/Discogs keys, only MusicBrainz is called."""
        mb_result = {"tags": [{"name": "house"}], "genres": ["house"], "artist_mbid": "abc"}

        async def run():
            with patch("mlx_audiogen.library.enrichment.manager.search_musicbrainz",
                       new=AsyncMock(return_value=mb_result)):
                result = await manager.enrich_single("Solomun", "Midnight Express")
            return result

        result = asyncio.run(run())
        assert result["musicbrainz"] is not None
        assert result["lastfm"] is None
        assert result["discogs"] is None

    def test_get_enrichment_cached(self, manager):
        """Second call returns cached data without API hit."""
        mb_result = {"tags": [], "genres": [], "artist_mbid": "x"}
        call_count = 0

        async def fake_mb(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mb_result

        async def run():
            with patch("mlx_audiogen.library.enrichment.manager.search_musicbrainz",
                       new=AsyncMock(side_effect=fake_mb)):
                await manager.enrich_single("A", "B")
                await manager.enrich_single("A", "B")

        asyncio.run(run())
        assert call_count == 1  # Only called once, second was cached
```

- [ ] **Step 2: Implement enrichment manager**

Create `mlx_audiogen/library/enrichment/manager.py`:

```python
"""Enrichment orchestrator — check cache, fetch from APIs, merge, and store.

Coordinates MusicBrainz, Last.fm, and Discogs clients with rate limiting
and credential checking. Supports single-track and batch enrichment.
"""

from __future__ import annotations

from typing import Optional

from mlx_audiogen.credentials import CredentialManager

from .discogs import search_discogs
from .enrichment_db import EnrichmentDB
from .lastfm import search_lastfm
from .musicbrainz import search_musicbrainz
from .rate_limiter import ApiRateLimiter


class EnrichmentManager:
    """Orchestrates enrichment across all 3 APIs."""

    def __init__(
        self,
        db: EnrichmentDB | None = None,
        credentials: CredentialManager | None = None,
    ):
        self._db = db or EnrichmentDB()
        self._cred = credentials or CredentialManager()
        self._mb_limiter = ApiRateLimiter(max_per_second=1.0)
        self._lfm_limiter = ApiRateLimiter(max_per_second=5.0)
        self._dc_limiter = ApiRateLimiter(max_per_second=1.0)
        self._cancelled = False

    @property
    def db(self) -> EnrichmentDB:
        return self._db

    def cancel(self) -> None:
        self._cancelled = True

    def reset_cancel(self) -> None:
        self._cancelled = False

    async def enrich_single(
        self,
        artist: str,
        title: str,
        library_source: str | None = None,
        library_track_id: str | None = None,
        force: bool = False,
    ) -> dict:
        """Enrich a single track. Returns all enrichment data.

        Checks cache first. Only fetches from APIs where data is missing or stale.
        Skips APIs where credentials are not configured.
        """
        tid = self._db.get_or_create_track(
            artist, title, library_source, library_track_id
        )

        # MusicBrainz (no key needed)
        if force or self._db.is_stale(tid, "musicbrainz"):
            mb_data = await search_musicbrainz(
                artist, title, self._mb_limiter
            )
            if mb_data:
                self._db.store_musicbrainz(tid, mb_data)

        # Last.fm (needs API key)
        lfm_key = self._cred.get("lastfm_api_key") if self._cred else None
        if lfm_key and (force or self._db.is_stale(tid, "lastfm")):
            lfm_data = await search_lastfm(
                artist, title, lfm_key, self._lfm_limiter
            )
            if lfm_data:
                self._db.store_lastfm(tid, lfm_data)

        # Discogs (needs token)
        dc_token = self._cred.get("discogs_token") if self._cred else None
        if dc_token and (force or self._db.is_stale(tid, "discogs")):
            dc_data = await search_discogs(
                artist, title, dc_token, self._dc_limiter
            )
            if dc_data:
                self._db.store_discogs(tid, dc_data)

        return self._db.get_all_enrichment(tid)

    async def enrich_batch(
        self,
        tracks: list[dict],
        on_progress: Optional[callable] = None,
    ) -> dict:
        """Enrich multiple tracks. Each dict has 'artist', 'title', optional 'library_source'/'library_track_id'.

        Returns {"completed": int, "errors": int, "total": int}.
        """
        self.reset_cancel()
        completed = 0
        errors = 0
        total = len(tracks)
        for t in tracks:
            if self._cancelled:
                break
            try:
                await self.enrich_single(
                    t["artist"], t["title"],
                    t.get("library_source"), t.get("library_track_id"),
                )
                completed += 1
            except Exception:
                errors += 1
            if on_progress:
                on_progress(completed=completed, errors=errors, total=total,
                            current=f"{t['artist']} - {t['title']}")
        return {"completed": completed, "errors": errors, "total": total}
```

- [ ] **Step 3: Update `__init__.py` re-exports**

Update `mlx_audiogen/library/enrichment/__init__.py`:

```python
"""Web enrichment subpackage — MusicBrainz, Last.fm, Discogs metadata."""

from .enrichment_db import EnrichmentDB
from .manager import EnrichmentManager

__all__ = ["EnrichmentDB", "EnrichmentManager"]
```

- [ ] **Step 4: Run tests and commit**

Run: `uv run pytest tests/test_enrichment.py -v`
Expected: All tests PASS.

```bash
git add mlx_audiogen/library/enrichment/ tests/test_enrichment.py
git commit -m "feat: add enrichment manager orchestrating 3 API sources"
```

---

### Task 10: Taste profile dataclasses

**Files:**
- Create: `mlx_audiogen/library/taste/__init__.py`
- Create: `mlx_audiogen/library/taste/profile.py`
- Create: `tests/test_taste.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_taste.py`:

```python
"""Tests for taste profile, signals, and engine."""

import json

import pytest

from mlx_audiogen.library.taste.profile import TasteProfile, WeightedTag


class TestTasteProfile:
    def test_create_empty(self):
        p = TasteProfile.empty()
        assert p.library_track_count == 0
        assert p.generation_count == 0
        assert p.top_genres == []

    def test_to_dict_roundtrip(self):
        p = TasteProfile.empty()
        p.top_genres = [WeightedTag("house", 0.8), WeightedTag("techno", 0.6)]
        d = p.to_dict()
        p2 = TasteProfile.from_dict(d)
        assert p2.top_genres[0].name == "house"
        assert p2.top_genres[0].weight == 0.8

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "taste_profile.json"
        p = TasteProfile.empty()
        p.top_genres = [WeightedTag("jazz", 0.9)]
        p.save(path)
        p2 = TasteProfile.load(path)
        assert p2.top_genres[0].name == "jazz"

    def test_load_missing_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        p = TasteProfile.load(path)
        assert p.library_track_count == 0
```

- [ ] **Step 2: Implement taste profile**

Create `mlx_audiogen/library/taste/__init__.py`:

```python
"""Taste learning engine — personal preference modeling."""
```

Create `mlx_audiogen/library/taste/profile.py`:

```python
"""TasteProfile and WeightedTag dataclasses.

The taste profile captures a user's music preferences from two sources:
- Library signals (what they listen to — broad base)
- Generation signals (what they create — weighted higher for creative intent)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class WeightedTag:
    """A tag name with a normalized weight (0.0-1.0)."""
    name: str
    weight: float

    def to_dict(self) -> dict:
        return {"name": self.name, "weight": round(self.weight, 4)}

    @classmethod
    def from_dict(cls, d: dict) -> WeightedTag:
        return cls(name=d["name"], weight=d["weight"])


@dataclass
class TasteProfile:
    """Combined taste profile from library + generation signals."""

    # Library signals
    top_genres: list[WeightedTag] = field(default_factory=list)
    top_artists: list[WeightedTag] = field(default_factory=list)
    bpm_range: tuple[float, float] = (0.0, 0.0)
    key_preferences: list[WeightedTag] = field(default_factory=list)
    era_distribution: dict[str, float] = field(default_factory=dict)
    mood_profile: list[WeightedTag] = field(default_factory=list)
    style_tags: list[WeightedTag] = field(default_factory=list)

    # Generation signals (creative intent, higher weight)
    gen_genres: list[WeightedTag] = field(default_factory=list)
    gen_moods: list[WeightedTag] = field(default_factory=list)
    gen_instruments: list[WeightedTag] = field(default_factory=list)
    kept_ratio: float = 0.0
    avg_duration: float = 0.0
    preferred_models: list[str] = field(default_factory=list)

    # Metadata
    library_track_count: int = 0
    generation_count: int = 0
    last_updated: str = ""
    version: int = 1

    # User overrides
    overrides: str = ""

    @classmethod
    def empty(cls) -> TasteProfile:
        return cls()

    def to_dict(self) -> dict:
        def tags_list(tags: list[WeightedTag]) -> list[dict]:
            return [t.to_dict() for t in tags]

        return {
            "top_genres": tags_list(self.top_genres),
            "top_artists": tags_list(self.top_artists),
            "bpm_range": list(self.bpm_range),
            "key_preferences": tags_list(self.key_preferences),
            "era_distribution": self.era_distribution,
            "mood_profile": tags_list(self.mood_profile),
            "style_tags": tags_list(self.style_tags),
            "gen_genres": tags_list(self.gen_genres),
            "gen_moods": tags_list(self.gen_moods),
            "gen_instruments": tags_list(self.gen_instruments),
            "kept_ratio": self.kept_ratio,
            "avg_duration": self.avg_duration,
            "preferred_models": self.preferred_models,
            "library_track_count": self.library_track_count,
            "generation_count": self.generation_count,
            "last_updated": self.last_updated,
            "version": self.version,
            "overrides": self.overrides,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TasteProfile:
        def parse_tags(raw: list) -> list[WeightedTag]:
            return [WeightedTag.from_dict(t) for t in raw]

        bpm = d.get("bpm_range", [0.0, 0.0])
        return cls(
            top_genres=parse_tags(d.get("top_genres", [])),
            top_artists=parse_tags(d.get("top_artists", [])),
            bpm_range=(bpm[0], bpm[1]),
            key_preferences=parse_tags(d.get("key_preferences", [])),
            era_distribution=d.get("era_distribution", {}),
            mood_profile=parse_tags(d.get("mood_profile", [])),
            style_tags=parse_tags(d.get("style_tags", [])),
            gen_genres=parse_tags(d.get("gen_genres", [])),
            gen_moods=parse_tags(d.get("gen_moods", [])),
            gen_instruments=parse_tags(d.get("gen_instruments", [])),
            kept_ratio=d.get("kept_ratio", 0.0),
            avg_duration=d.get("avg_duration", 0.0),
            preferred_models=d.get("preferred_models", []),
            library_track_count=d.get("library_track_count", 0),
            generation_count=d.get("generation_count", 0),
            last_updated=d.get("last_updated", ""),
            version=d.get("version", 1),
            overrides=d.get("overrides", ""),
        )

    def save(self, path: Optional[Path] = None) -> None:
        p = path or (Path.home() / ".mlx-audiogen" / "taste_profile.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Optional[Path] = None) -> TasteProfile:
        p = path or (Path.home() / ".mlx-audiogen" / "taste_profile.json")
        if not p.is_file():
            return cls.empty()
        try:
            return cls.from_dict(json.loads(p.read_text()))
        except (json.JSONDecodeError, KeyError):
            return cls.empty()
```

- [ ] **Step 3: Run tests and commit**

Run: `uv run pytest tests/test_taste.py -v`
Expected: All 4 tests PASS.

```bash
git add mlx_audiogen/library/taste/ tests/test_taste.py
git commit -m "feat: add TasteProfile and WeightedTag dataclasses"
```

---

### Task 11: Taste signals collector

**Files:**
- Create: `mlx_audiogen/library/taste/signals.py`
- Add tests to: `tests/test_taste.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_taste.py`:

```python
from mlx_audiogen.library.taste.signals import collect_library_signals, collect_generation_signals
from mlx_audiogen.library.models import TrackInfo


def _make_track(**kwargs) -> TrackInfo:
    defaults = {
        "track_id": "1", "title": "Test", "artist": "Artist", "album": "",
        "genre": "", "bpm": None, "key": None, "year": None, "rating": 0,
        "play_count": 0, "duration_seconds": 0, "comments": "", "file_path": None,
        "file_available": False, "source": "apple_music", "loved": False,
        "description": "", "description_edited": False,
    }
    defaults.update(kwargs)
    return TrackInfo(**defaults)


class TestLibrarySignals:
    def test_genre_weighting(self):
        tracks = [
            _make_track(genre="House", play_count=100),
            _make_track(genre="House", play_count=50),
            _make_track(genre="Techno", play_count=10),
        ]
        signals = collect_library_signals(tracks)
        assert signals["top_genres"][0].name == "house"
        assert signals["top_genres"][0].weight > signals["top_genres"][1].weight

    def test_bpm_range(self):
        tracks = [
            _make_track(bpm=120.0),
            _make_track(bpm=125.0),
            _make_track(bpm=130.0),
        ]
        signals = collect_library_signals(tracks)
        lo, hi = signals["bpm_range"]
        assert lo <= 121 and hi >= 129

    def test_empty_tracks(self):
        signals = collect_library_signals([])
        assert signals["top_genres"] == []


class TestGenerationSignals:
    def test_from_prompt_memory(self):
        style_profile = {
            "top_genres": ["ambient", "techno"],
            "top_moods": ["dark", "hypnotic"],
            "top_instruments": ["synth", "drums"],
            "preferred_duration": 10,
            "generation_count": 25,
        }
        history = [{"prompt": "test", "params": {"seconds": 10}}] * 25
        signals = collect_generation_signals(style_profile, history)
        assert signals["gen_genres"][0].name == "ambient"
        assert signals["generation_count"] == 25
```

- [ ] **Step 2: Implement signals collector**

Create `mlx_audiogen/library/taste/signals.py`:

```python
"""Signal collectors for the taste learning engine.

Two signal streams:
- Library signals: derived from track metadata (genres, BPM, keys, ratings, play counts)
- Generation signals: derived from prompt memory style profile + generation history
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Optional

from mlx_audiogen.library.models import TrackInfo

from .profile import WeightedTag


def _normalize_counter_to_tags(counter: Counter, max_items: int = 10) -> list[WeightedTag]:
    """Convert a Counter to a list of WeightedTags, normalized to 0.0-1.0."""
    if not counter:
        return []
    most_common = counter.most_common(max_items)
    max_count = most_common[0][1] if most_common else 1
    return [
        WeightedTag(name=name.lower(), weight=round(count / max_count, 4))
        for name, count in most_common
        if name
    ]


def collect_library_signals(
    tracks: list[TrackInfo],
    enrichment_tags: Optional[dict[str, list[dict]]] = None,
) -> dict:
    """Aggregate library metadata into taste signals.

    Args:
        tracks: All tracks from user's library.
        enrichment_tags: Optional mapping of track_id -> list of enrichment tag dicts.

    Returns:
        Dict with keys matching TasteProfile library signal fields.
    """
    if not tracks:
        return {
            "top_genres": [], "top_artists": [], "bpm_range": (0.0, 0.0),
            "key_preferences": [], "era_distribution": {}, "mood_profile": [],
            "style_tags": [], "library_track_count": 0,
        }

    # Weight by play_count (minimum 1 to include all tracks)
    genre_counter: Counter = Counter()
    artist_counter: Counter = Counter()
    key_counter: Counter = Counter()
    era_counter: Counter = Counter()
    bpms: list[float] = []

    for t in tracks:
        weight = max(t.play_count, 1)
        if t.genre:
            genre_counter[t.genre.lower()] += weight
        if t.artist:
            artist_counter[t.artist] += weight
        if t.key:
            key_counter[t.key] += weight
        if t.year:
            decade = f"{(t.year // 10) * 10}s"
            era_counter[decade] += weight
        if t.bpm is not None:
            bpms.append(t.bpm)

    # BPM range: 10th-90th percentile
    bpm_range = (0.0, 0.0)
    if bpms:
        sorted_bpms = sorted(bpms)
        n = len(sorted_bpms)
        lo_idx = max(0, int(n * 0.1))
        hi_idx = min(n - 1, int(n * 0.9))
        bpm_range = (sorted_bpms[lo_idx], sorted_bpms[hi_idx])

    # Era distribution as normalized weights
    total_era = sum(era_counter.values()) or 1
    era_distribution = {k: round(v / total_era, 4) for k, v in era_counter.most_common(10)}

    return {
        "top_genres": _normalize_counter_to_tags(genre_counter),
        "top_artists": _normalize_counter_to_tags(artist_counter, max_items=15),
        "bpm_range": bpm_range,
        "key_preferences": _normalize_counter_to_tags(key_counter, max_items=12),
        "era_distribution": era_distribution,
        "mood_profile": [],  # Populated from enrichment data when available
        "style_tags": [],    # Populated from enrichment data when available
        "library_track_count": len(tracks),
    }


def collect_generation_signals(
    style_profile: dict,
    history: list[dict],
) -> dict:
    """Extract generation signals from prompt memory.

    Args:
        style_profile: From PromptMemory.style_profile (top_genres, etc.)
        history: From PromptMemory.history (list of generation entries)

    Returns:
        Dict with keys matching TasteProfile generation signal fields.
    """
    gen_count = style_profile.get("generation_count", len(history))

    # Convert existing style profile lists to WeightedTags
    def to_tags(names: list[str]) -> list[WeightedTag]:
        if not names:
            return []
        return [
            WeightedTag(name=n.lower(), weight=round(1.0 - i * 0.15, 4))
            for i, n in enumerate(names[:7])
        ]

    # Compute avg duration from history
    durations = [
        float(e.get("params", {}).get("seconds", 0))
        for e in history
        if e.get("params", {}).get("seconds")
    ]
    avg_dur = statistics.mean(durations) if durations else 0.0

    # Count models used
    model_counter: Counter = Counter()
    for e in history:
        m = e.get("params", {}).get("model")
        if m:
            model_counter[m] += 1

    return {
        "gen_genres": to_tags(style_profile.get("top_genres", [])),
        "gen_moods": to_tags(style_profile.get("top_moods", [])),
        "gen_instruments": to_tags(style_profile.get("top_instruments", [])),
        "kept_ratio": 0.0,  # TODO: track favorites in future
        "avg_duration": round(avg_dur, 1),
        "preferred_models": [m for m, _ in model_counter.most_common(3)],
        "generation_count": gen_count,
    }
```

- [ ] **Step 3: Run tests and commit**

Run: `uv run pytest tests/test_taste.py -v`
Expected: All tests PASS.

```bash
git add mlx_audiogen/library/taste/signals.py tests/test_taste.py
git commit -m "feat: add library and generation signal collectors for taste engine"
```

---

### Task 12: Taste engine

**Files:**
- Create: `mlx_audiogen/library/taste/engine.py`
- Add tests to: `tests/test_taste.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_taste.py`:

```python
from mlx_audiogen.library.taste.engine import TasteEngine


class TestTasteEngine:
    def test_compute_from_library(self, tmp_path):
        engine = TasteEngine(profile_path=tmp_path / "profile.json")
        tracks = [
            _make_track(genre="House", play_count=100, bpm=122.0, key="4A", year=2020),
            _make_track(genre="Techno", play_count=50, bpm=130.0, key="7A", year=2019),
        ]
        engine.update_library_signals(tracks)
        profile = engine.get_profile()
        assert profile.library_track_count == 2
        assert profile.top_genres[0].name == "house"

    def test_compute_from_generation(self, tmp_path):
        engine = TasteEngine(profile_path=tmp_path / "profile.json")
        style = {"top_genres": ["ambient"], "top_moods": ["calm"],
                 "top_instruments": ["synth"], "preferred_duration": 10,
                 "generation_count": 5}
        history = [{"prompt": "test", "params": {"seconds": 10, "model": "musicgen"}}] * 5
        engine.update_generation_signals(style, history)
        profile = engine.get_profile()
        assert profile.generation_count == 5

    def test_set_overrides(self, tmp_path):
        engine = TasteEngine(profile_path=tmp_path / "profile.json")
        engine.set_overrides("more minimal, less vocal")
        profile = engine.get_profile()
        assert profile.overrides == "more minimal, less vocal"

    def test_persists_to_disk(self, tmp_path):
        path = tmp_path / "profile.json"
        engine = TasteEngine(profile_path=path)
        engine.update_library_signals([_make_track(genre="Jazz", play_count=10)])
        # Load fresh
        engine2 = TasteEngine(profile_path=path)
        assert engine2.get_profile().top_genres[0].name == "jazz"
```

- [ ] **Step 2: Implement taste engine**

Create `mlx_audiogen/library/taste/engine.py`:

```python
"""TasteEngine — computes, updates, and queries user taste profiles.

Combines library signals (what you listen to) and generation signals
(what you create) into a unified TasteProfile. Generation signals are
weighted higher for creative intent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from mlx_audiogen.library.models import TrackInfo

from .profile import TasteProfile
from .signals import collect_generation_signals, collect_library_signals


class TasteEngine:
    """Manages taste profile computation and persistence."""

    def __init__(self, profile_path: Optional[Path] = None):
        self._path = profile_path or (
            Path.home() / ".mlx-audiogen" / "taste_profile.json"
        )
        self._profile = TasteProfile.load(self._path)

    def get_profile(self) -> TasteProfile:
        return self._profile

    def update_library_signals(
        self,
        tracks: list[TrackInfo],
        enrichment_tags: Optional[dict] = None,
    ) -> TasteProfile:
        """Recompute library portion of taste profile from current library data."""
        signals = collect_library_signals(tracks, enrichment_tags)
        self._profile.top_genres = signals["top_genres"]
        self._profile.top_artists = signals["top_artists"]
        self._profile.bpm_range = signals["bpm_range"]
        self._profile.key_preferences = signals["key_preferences"]
        self._profile.era_distribution = signals["era_distribution"]
        self._profile.mood_profile = signals["mood_profile"]
        self._profile.style_tags = signals["style_tags"]
        self._profile.library_track_count = signals["library_track_count"]
        self._profile.last_updated = datetime.now(timezone.utc).isoformat()
        self._profile.save(self._path)
        return self._profile

    def update_generation_signals(
        self,
        style_profile: dict,
        history: list[dict],
    ) -> TasteProfile:
        """Recompute generation portion of taste profile from prompt memory."""
        signals = collect_generation_signals(style_profile, history)
        self._profile.gen_genres = signals["gen_genres"]
        self._profile.gen_moods = signals["gen_moods"]
        self._profile.gen_instruments = signals["gen_instruments"]
        self._profile.kept_ratio = signals["kept_ratio"]
        self._profile.avg_duration = signals["avg_duration"]
        self._profile.preferred_models = signals["preferred_models"]
        self._profile.generation_count = signals["generation_count"]
        self._profile.last_updated = datetime.now(timezone.utc).isoformat()
        self._profile.save(self._path)
        return self._profile

    def set_overrides(self, text: str) -> TasteProfile:
        """Store manual user overrides (e.g., 'more minimal, less vocal')."""
        self._profile.overrides = text
        self._profile.save(self._path)
        return self._profile

    def refresh(
        self,
        tracks: list[TrackInfo],
        style_profile: dict,
        history: list[dict],
        enrichment_tags: Optional[dict] = None,
    ) -> TasteProfile:
        """Full refresh: recompute both library and generation signals."""
        self.update_library_signals(tracks, enrichment_tags)
        self.update_generation_signals(style_profile, history)
        return self._profile
```

- [ ] **Step 3: Run tests and commit**

Run: `uv run pytest tests/test_taste.py -v`
Expected: All tests PASS.

```bash
git add mlx_audiogen/library/taste/engine.py tests/test_taste.py
git commit -m "feat: add TasteEngine for computing and persisting taste profiles"
```

---

## Chunk 4: Server Integration (API Endpoints)

### Task 13: Add credential, enrichment, and taste endpoints to server

**Files:**
- Modify: `mlx_audiogen/server/app.py`

This is a large task. The endpoints follow existing patterns in `app.py`. Add them after the existing collection endpoints.

- [ ] **Step 1: Add imports and startup initialization at top of app.py**

Add to the imports section:

```python
from mlx_audiogen.credentials import CredentialManager
from mlx_audiogen.library.enrichment import EnrichmentDB, EnrichmentManager
from mlx_audiogen.library.taste.engine import TasteEngine
```

Add module-level singletons (near the existing `_rate_limiter`):

```python
_credential_manager = CredentialManager()
_enrichment_db = EnrichmentDB()
_enrichment_manager = EnrichmentManager(db=_enrichment_db, credentials=_credential_manager)
_taste_engine = TasteEngine()
_enrichment_job: dict | None = None  # Active background enrichment job
```

- [ ] **Step 2: Add credential endpoints**

```python
# --- Credentials ---

@app.get("/api/credentials/status")
def credentials_status():
    return _credential_manager.status()

@app.post("/api/credentials/{service}")
def credentials_set(service: str, body: dict):
    api_key = body.get("api_key", "")
    if not api_key:
        raise HTTPException(400, "api_key is required")
    try:
        _credential_manager.set(service, api_key)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "saved", "service": service}

@app.delete("/api/credentials/{service}")
def credentials_delete(service: str):
    try:
        _credential_manager.delete(service)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "deleted", "service": service}
```

- [ ] **Step 3: Add enrichment endpoints**

```python
# --- Enrichment ---

@app.post("/api/enrich/tracks")
async def enrich_tracks(body: dict):
    """Enrich a batch of tracks."""
    track_ids = body.get("track_ids", [])
    source_id = body.get("source_id")
    direct_tracks = body.get("tracks", [])

    tracks_to_enrich = []
    if direct_tracks:
        tracks_to_enrich = [{"artist": t["artist"], "title": t["title"]} for t in direct_tracks]
    elif track_ids and source_id:
        cache = _get_library_cache()
        tracks_dict = cache._data.get(source_id, {}).get("tracks", {})
        for tid in track_ids:
            track = tracks_dict.get(tid)
            if track:
                tracks_to_enrich.append({
                    "artist": track.artist, "title": track.title,
                    "library_source": track.source, "library_track_id": track.track_id,
                })

    if not tracks_to_enrich:
        raise HTTPException(400, "No tracks to enrich")

    # Small batch: synchronous
    if len(tracks_to_enrich) <= 20:
        result = await _enrichment_manager.enrich_batch(tracks_to_enrich)
        return result

    # Large batch: background job
    import uuid
    job_id = str(uuid.uuid4())[:8]
    global _enrichment_job
    _enrichment_job = {
        "job_id": job_id, "status": "running", "total": len(tracks_to_enrich),
        "completed": 0, "errors": 0, "current_track": None,
    }

    async def run_batch():
        global _enrichment_job
        def on_progress(**kwargs):
            if _enrichment_job:
                _enrichment_job.update(kwargs)
        await _enrichment_manager.enrich_batch(tracks_to_enrich, on_progress=on_progress)
        if _enrichment_job:
            _enrichment_job["status"] = "done"

    import asyncio
    asyncio.create_task(run_batch())
    return {"job_id": job_id}

@app.get("/api/enrich/status")
def enrich_status():
    if _enrichment_job is None:
        return {"status": "idle"}
    return _enrichment_job

@app.post("/api/enrich/all/{source_id}")
async def enrich_all(source_id: str):
    cache = _get_library_cache()
    tracks_dict = cache._data.get(source_id, {}).get("tracks", {})
    if not tracks_dict:
        raise HTTPException(404, "Source not found or empty")
    body = {
        "track_ids": list(tracks_dict.keys()),
        "source_id": source_id,
    }
    return await enrich_tracks(body)

@app.post("/api/enrich/cancel")
def enrich_cancel():
    _enrichment_manager.cancel()
    if _enrichment_job:
        _enrichment_job["status"] = "cancelled"
    return {"status": "cancelled"}

@app.get("/api/enrich/track/{track_id}")
def enrich_track_get(track_id: int):
    data = _enrichment_db.get_all_enrichment(track_id)
    status = _enrichment_db.get_enrichment_status(track_id)
    return {"track_id": track_id, "status": status, **data}

@app.get("/api/enrich/stats")
def enrich_stats():
    return _enrichment_db.get_stats()
```

- [ ] **Step 4: Add taste endpoints**

```python
# --- Taste Profile ---

@app.get("/api/taste/profile")
def taste_profile():
    return _taste_engine.get_profile().to_dict()

@app.post("/api/taste/refresh")
def taste_refresh():
    cache = _get_library_cache()
    all_tracks = []
    for source in cache.list_sources():
        tracks_dict = cache._data.get(source.id, {}).get("tracks", {})
        all_tracks.extend(tracks_dict.values())
    memory = _get_memory()
    profile = _taste_engine.refresh(
        all_tracks, memory.style_profile, memory.history,
    )
    return profile.to_dict()

@app.get("/api/taste/suggestions")
def taste_suggestions():
    profile = _taste_engine.get_profile()
    suggestions = []
    # Build personalized suggestions from taste profile
    for genre in profile.top_genres[:3]:
        for mood in (profile.gen_moods or profile.mood_profile)[:2]:
            lo, hi = profile.bpm_range
            bpm = int((lo + hi) / 2) if lo > 0 else ""
            parts = [genre.name, mood.name]
            if bpm:
                parts.append(f"{bpm} BPM")
            if profile.key_preferences:
                parts.append(profile.key_preferences[0].name)
            suggestions.append(", ".join(parts))
    return {"suggestions": suggestions[:6]}

@app.put("/api/taste/overrides")
def taste_overrides(body: dict):
    text = body.get("text", "")
    profile = _taste_engine.set_overrides(text)
    return profile.to_dict()
```

- [ ] **Step 5: Modify existing generate-prompt endpoint to use taste profile**

In the existing `POST /api/library/generate-prompt` handler, after building the prompt from `generate_playlist_prompt()`:

```python
# After: result = generate_playlist_prompt(tracks)
# Add taste profile blending:
taste = _taste_engine.get_profile()
if taste.top_genres:
    taste_genres = [g.name for g in taste.top_genres[:2]]
    for tg in taste_genres:
        if tg not in result["prompt"].lower():
            result["prompt"] += f", {tg} influence"
if taste.overrides:
    result["taste_overrides"] = taste.overrides
```

- [ ] **Step 6: Modify existing suggest endpoint to blend taste profile**

In the existing `POST /api/suggest` handler, after calling `analyze_prompt()`:

```python
# After: analysis = analyze_prompt(req.prompt, count=req.count)
# Blend taste profile into suggestions:
taste = _taste_engine.get_profile()
if taste.top_genres and analysis.get("suggestions"):
    # Prepend a taste-informed suggestion
    taste_hint = ", ".join(g.name for g in taste.top_genres[:2])
    taste_suggestion = f"{req.prompt}, {taste_hint} style"
    analysis["suggestions"].insert(0, taste_suggestion)
```

- [ ] **Step 7: Modify existing library tracks endpoint**

In the existing `GET /api/library/tracks/{source_id}` handler, add `enrichment_status` to each track dict:

```python
# Inside the tracks endpoint, after building track dicts:
for td in track_dicts:
    etid = _enrichment_db.find_by_library_id(source.type, td["track_id"])
    td["enrichment_status"] = _enrichment_db.get_enrichment_status(etid) if etid else "none"
```

- [ ] **Step 8: Run full test suite**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen && uv run pytest -v`
Expected: All existing + new tests PASS (target: ~460 tests).

- [ ] **Step 9: Commit**

```bash
git add mlx_audiogen/server/app.py
git commit -m "feat: add 13 enrichment/credential/taste API endpoints + modify 3 existing"
```

---

### Task 14: Server endpoint tests for new routes

**Files:**
- Add tests to: `tests/test_library_api.py` (or create `tests/test_enrichment_api.py`)

- [ ] **Step 1: Write endpoint tests following existing TestClient pattern**

Add to `tests/test_library_api.py` (or a new file with the same fixture pattern):

```python
class TestCredentialEndpoints:
    def test_credential_status(self, client: TestClient):
        res = client.get("/api/credentials/status")
        assert res.status_code == 200
        data = res.json()
        assert data["musicbrainz"] is True

    def test_set_and_check_credential(self, client: TestClient):
        res = client.post("/api/credentials/lastfm_api_key",
                          json={"api_key": "test_key_123"})
        assert res.status_code == 200
        status = client.get("/api/credentials/status").json()
        assert status["lastfm"] is True

    def test_set_invalid_service(self, client: TestClient):
        res = client.post("/api/credentials/spotify_key",
                          json={"api_key": "value"})
        assert res.status_code == 400

    def test_delete_credential(self, client: TestClient):
        client.post("/api/credentials/lastfm_api_key",
                    json={"api_key": "test_key"})
        res = client.delete("/api/credentials/lastfm_api_key")
        assert res.status_code == 200


class TestEnrichmentEndpoints:
    def test_enrich_status_idle(self, client: TestClient):
        res = client.get("/api/enrich/status")
        assert res.status_code == 200
        assert res.json()["status"] == "idle"

    def test_enrich_stats_empty(self, client: TestClient):
        res = client.get("/api/enrich/stats")
        assert res.status_code == 200
        assert res.json()["total_tracks"] == 0

    def test_enrich_tracks_empty_body(self, client: TestClient):
        res = client.post("/api/enrich/tracks", json={})
        assert res.status_code == 400

    def test_cancel_enrichment(self, client: TestClient):
        res = client.post("/api/enrich/cancel")
        assert res.status_code == 200


class TestTasteEndpoints:
    def test_get_empty_profile(self, client: TestClient):
        res = client.get("/api/taste/profile")
        assert res.status_code == 200
        data = res.json()
        assert data["library_track_count"] == 0

    def test_set_overrides(self, client: TestClient):
        res = client.put("/api/taste/overrides",
                         json={"text": "more minimal"})
        assert res.status_code == 200
        assert res.json()["overrides"] == "more minimal"

    def test_taste_suggestions_empty(self, client: TestClient):
        res = client.get("/api/taste/suggestions")
        assert res.status_code == 200
        assert "suggestions" in res.json()
```

Note: The fixture must monkeypatch `_credential_manager`, `_enrichment_db`, `_enrichment_manager`, and `_taste_engine` with test instances (in-memory DB, monkeypatched keyring). Follow the existing `_clean_library_state` pattern.

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_library_api.py -v -k "Credential or Enrichment or Taste"`
Expected: All endpoint tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_library_api.py
git commit -m "test: add server endpoint tests for credential, enrichment, and taste routes"
```

---

## Chunk 5: Description Gen Enhancement + Full QA

### Task 15: Enhance description_gen with enrichment data

**Files:**
- Modify: `mlx_audiogen/library/description_gen.py`

- [ ] **Step 1: Add optional enrichment parameter**

Modify `generate_description()` to accept an optional `enrichment` dict:

```python
def generate_description(track: TrackInfo, enrichment: dict | None = None) -> str:
    # ... existing logic ...
    # After building parts from track metadata, append enrichment tags:
    if enrichment:
        mb = enrichment.get("musicbrainz")
        if mb:
            import json
            genres = json.loads(mb.get("genres", "[]")) if isinstance(mb.get("genres"), str) else mb.get("genres", [])
            for g in genres[:2]:
                if g.lower() not in [p.lower() for p in parts]:
                    parts.append(g.lower())
        dc = enrichment.get("discogs")
        if dc:
            import json
            styles = json.loads(dc.get("styles", "[]")) if isinstance(dc.get("styles"), str) else dc.get("styles", [])
            for s in styles[:2]:
                if s.lower() not in [p.lower() for p in parts]:
                    parts.append(s.lower())
    # ... rest of function unchanged ...
```

- [ ] **Step 2: Run full QA suite**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen && uv run ruff format . && uv run ruff check . && uv run mypy mlx_audiogen/ && uv run bandit -r mlx_audiogen/ -c pyproject.toml && uv run pip-audit && uv run pytest`
Expected: All checks pass.

- [ ] **Step 3: Commit**

```bash
git add mlx_audiogen/library/description_gen.py
git commit -m "feat: enhance description_gen with enrichment metadata"
```

---

## Chunk 6: Web UI (TypeScript + React Components)

### Task 16: Add TypeScript types

**Files:**
- Modify: `web/src/types/api.ts`

- [ ] **Step 1: Add enrichment, credential, and taste types**

Append to `web/src/types/api.ts`:

```typescript
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
```

- [ ] **Step 2: Commit**

```bash
git add web/src/types/api.ts
git commit -m "feat(web): add enrichment, credential, and taste TypeScript types"
```

---

### Task 17: Add API client functions

**Files:**
- Modify: `web/src/api/client.ts`

- [ ] **Step 1: Add fetch wrappers for all 13 new endpoints**

Append to `web/src/api/client.ts`:

```typescript
// --- Credentials ---

export async function getCredentialStatus(): Promise<CredentialStatus> {
  const res = await fetch(`${getBaseUrl()}/api/credentials/status`);
  return res.json();
}

export async function setCredential(service: string, apiKey: string): Promise<void> {
  await fetch(`${getBaseUrl()}/api/credentials/${service}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey }),
  });
}

export async function deleteCredential(service: string): Promise<void> {
  await fetch(`${getBaseUrl()}/api/credentials/${service}`, { method: "DELETE" });
}

// --- Enrichment ---

export async function enrichTracks(body: {
  track_ids?: string[];
  source_id?: string;
  tracks?: { artist: string; title: string }[];
}): Promise<any> {
  const res = await fetch(`${getBaseUrl()}/api/enrich/tracks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

export async function getEnrichmentStatus(): Promise<EnrichmentJobStatus> {
  const res = await fetch(`${getBaseUrl()}/api/enrich/status`);
  return res.json();
}

export async function enrichAll(sourceId: string): Promise<{ job_id: string }> {
  const res = await fetch(`${getBaseUrl()}/api/enrich/all/${sourceId}`, { method: "POST" });
  return res.json();
}

export async function cancelEnrichment(): Promise<void> {
  await fetch(`${getBaseUrl()}/api/enrich/cancel`, { method: "POST" });
}

export async function getTrackEnrichment(trackId: number): Promise<any> {
  const res = await fetch(`${getBaseUrl()}/api/enrich/track/${trackId}`);
  return res.json();
}

export async function getEnrichmentStats(): Promise<EnrichmentStats> {
  const res = await fetch(`${getBaseUrl()}/api/enrich/stats`);
  return res.json();
}

// --- Taste Profile ---

export async function getTasteProfile(): Promise<TasteProfile> {
  const res = await fetch(`${getBaseUrl()}/api/taste/profile`);
  return res.json();
}

export async function refreshTasteProfile(): Promise<TasteProfile> {
  const res = await fetch(`${getBaseUrl()}/api/taste/refresh`, { method: "POST" });
  return res.json();
}

export async function getTasteSuggestions(): Promise<{ suggestions: string[] }> {
  const res = await fetch(`${getBaseUrl()}/api/taste/suggestions`);
  return res.json();
}

export async function setTasteOverrides(text: string): Promise<TasteProfile> {
  const res = await fetch(`${getBaseUrl()}/api/taste/overrides`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  return res.json();
}
```

- [ ] **Step 2: Add imports for new types at top of client.ts**

Add to the import from `../types/api`:

```typescript
import type { CredentialStatus, EnrichmentJobStatus, EnrichmentStats, TasteProfile } from "../types/api";
```

- [ ] **Step 3: Commit**

```bash
git add web/src/api/client.ts web/src/types/api.ts
git commit -m "feat(web): add API client functions for enrichment, credentials, and taste"
```

---

### Task 18: Update SettingsPanel with API Keys + Taste Profile

**Files:**
- Modify: `web/src/components/SettingsPanel.tsx`

- [ ] **Step 1: Add API Keys section**

Add a new section below the existing LLM settings with:
- MusicBrainz row: green "Ready" badge
- Last.fm row: password input + Save/Clear, green/gray dot
- Discogs row: same pattern
- "Auto-enrich on browse" toggle
- Fetch credential status on mount via `getCredentialStatus()`

- [ ] **Step 2: Add Taste Profile card**

Below the API Keys section, add the Taste Profile card:
- Fetch profile on mount via `getTasteProfile()`
- Display top_genres, bpm_range, key_preferences, mood_profile (using existing tag color scheme)
- Creative Intent section (gen_genres, gen_moods, gen_instruments) in fuchsia
- Generation stats line
- Manual override text input with "Save Override" button
- Refresh + Reset buttons

- [ ] **Step 3: Build and verify**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen/web && npm run build`
Expected: Build succeeds with no errors.

- [ ] **Step 4: Commit**

```bash
git add web/src/components/SettingsPanel.tsx
git commit -m "feat(web): add API keys section and taste profile card to Settings"
```

---

### Task 19: Update LibraryPanel with enrichment indicators

**Files:**
- Modify: `web/src/components/LibraryPanel.tsx`

- [ ] **Step 1: Add Enriched column to track table**

After the existing columns, add an "Enriched" column with 3 dots per row:
- Green dot = data fetched for that source (MusicBrainz / Last.fm / Discogs)
- Gray dot = not fetched
- Tooltip on hover with source name
- Read `enrichment_status` from the track response (now included by server)

- [ ] **Step 2: Add Tags column**

After the Enriched column, add a "Tags" column:
- Display color-coded tags from enrichment data
- Use existing 14-category color scheme from tag system
- Show "Not enriched" in italic gray when no data

- [ ] **Step 3: Add "Enrich Selected" button to action bar**

In the existing selection action bar (alongside "Generate Like This" / "Train on These"):
- Purple "Enrich Selected" button
- Calls `enrichTracks({ track_ids: selectedIds, source_id: activeSource })`
- Shows progress if > 20 tracks (polls `getEnrichmentStatus()` every 2s)

- [ ] **Step 4: Build and verify**

Run: `cd /Users/jasonvassallo/Documents/Code/mlx-audiogen/web && npm run build`
Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
git add web/src/components/LibraryPanel.tsx
git commit -m "feat(web): add enrichment dots, tags column, and Enrich Selected to Library"
```

---

## Chunk 7: Final QA + Deploy

### Task 20: Full quality assurance

- [ ] **Step 1: Run complete QA suite**

```bash
cd /Users/jasonvassallo/Documents/Code/mlx-audiogen
uv run ruff format .
uv run ruff check .
uv run mypy mlx_audiogen/
uv run bandit -r mlx_audiogen/ -c pyproject.toml
uv run pip-audit
uv run pytest
cd web && npm run build
```

Expected: All checks pass. 0 errors across all tools.

- [ ] **Step 2: Fix any issues found**

Address all lint, type, security, or test failures.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: address QA findings from Phase 9g-3"
```

---

### Task 21: Update documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: project memory files

- [ ] **Step 1: Update CLAUDE.md**

Add to the API endpoints table:
- All 13 new endpoints (enrich, credentials, taste)
- Update existing library tracks endpoint description

Add to Architecture section:
- Enrichment subpackage description
- Taste engine description
- credentials.py description

- [ ] **Step 2: Update MEMORY.md**

Mark Phase 9g-3 as complete. Update test count. Add Phase 9g-4 as next TODO.

- [ ] **Step 3: Commit docs**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Phase 9g-3"
```

---

### Task 22: Push to main

- [ ] **Step 1: Push all commits**

```bash
git push origin main
```

- [ ] **Step 2: Deploy to Mac Mini (if applicable)**

```bash
ssh macmini "cd ~/Documents/Code/mlx-audiogen && git pull && ~/.local/bin/uv pip install '.[server]' --python ~/mlx-audiogen-venv/bin/python && cd web && npm run build && cp -r dist/* ~/mlx-audiogen-data/web-dist/"
ssh macmini "launchctl unload ~/Library/LaunchAgents/com.jasonvassallo.mlx-audiogen-server.plist && launchctl load ~/Library/LaunchAgents/com.jasonvassallo.mlx-audiogen-server.plist"
```
