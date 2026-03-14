"""SQLite cache for web enrichment data (MusicBrainz, Last.fm, Discogs).

Stores normalized track identity, library source mapping, and per-API
JSON payloads with fetch timestamps.  Supports staleness checks and
enrichment status queries.

Usage::

    db = EnrichmentDB()                      # default: ~/.mlx-audiogen/enrichment.db
    db = EnrichmentDB(":memory:")            # in-memory for tests
    tid = db.get_or_create_track("Daft Punk", "Around the World")
    db.store_musicbrainz(tid, {"mbid": "abc-123", "tags": ["electronic"]})
    data = db.get_musicbrainz(tid)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _normalize(s: str) -> str:
    """Lowercase + strip whitespace for dedup."""
    return s.strip().lower()


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS tracks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    artist          TEXT NOT NULL,
    title           TEXT NOT NULL,
    artist_norm     TEXT NOT NULL,
    title_norm      TEXT NOT NULL,
    musicbrainz_id  TEXT,
    library_source  TEXT,
    library_track_id TEXT,
    UNIQUE(artist_norm, title_norm)
);

CREATE TABLE IF NOT EXISTS musicbrainz (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id   INTEGER NOT NULL REFERENCES tracks(id),
    data       TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    UNIQUE(track_id)
);

CREATE TABLE IF NOT EXISTS lastfm (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id   INTEGER NOT NULL REFERENCES tracks(id),
    data       TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    UNIQUE(track_id)
);

CREATE TABLE IF NOT EXISTS discogs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id   INTEGER NOT NULL REFERENCES tracks(id),
    data       TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    UNIQUE(track_id)
);
"""


class EnrichmentDB:
    """SQLite-backed enrichment metadata cache."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            config_dir = Path.home() / ".mlx-audiogen"
            config_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(config_dir / "enrichment.db")

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------
    # Track identity
    # ------------------------------------------------------------------

    def get_or_create_track(
        self,
        artist: str,
        title: str,
        library_source: Optional[str] = None,
        library_track_id: Optional[str] = None,
    ) -> int:
        """Return the track row id, creating a new row if needed.

        Dedup is based on normalized (lowercase + stripped) artist+title.
        """
        artist_norm = _normalize(artist)
        title_norm = _normalize(title)

        cur = self._conn.execute(
            "SELECT id FROM tracks WHERE artist_norm = ? AND title_norm = ?",
            (artist_norm, title_norm),
        )
        row = cur.fetchone()
        if row is not None:
            track_id: int = row["id"]
            # Update library mapping if provided
            if library_source is not None and library_track_id is not None:
                self._conn.execute(
                    "UPDATE tracks SET library_source = ?, "
                    "library_track_id = ? WHERE id = ?",
                    (library_source, library_track_id, track_id),
                )
                self._conn.commit()
            return track_id

        cur = self._conn.execute(
            "INSERT INTO tracks "
            "(artist, title, artist_norm, title_norm, "
            "library_source, library_track_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (artist, title, artist_norm, title_norm, library_source, library_track_id),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def find_by_library_id(self, source: str, track_id: str) -> Optional[int]:
        """Find an enrichment track id by library source + track id.

        Returns ``None`` if not found.
        """
        cur = self._conn.execute(
            "SELECT id FROM tracks WHERE library_source = ? AND library_track_id = ?",
            (source, track_id),
        )
        row = cur.fetchone()
        return row["id"] if row is not None else None

    # ------------------------------------------------------------------
    # Store / get per source
    # ------------------------------------------------------------------

    # Allowed table names — validated before any SQL to prevent injection.
    _VALID_TABLES: frozenset[str] = frozenset({"musicbrainz", "lastfm", "discogs"})

    # Pre-built SQL per table -- no runtime string concatenation.
    _UPSERT = (
        "INSERT OR REPLACE INTO {t} "
        "(track_id, data, fetched_at) VALUES (?, ?, ?)"
    )
    _INSERT_SQL: dict[str, str] = {
        "musicbrainz": _UPSERT.format(t="musicbrainz"),
        "lastfm": _UPSERT.format(t="lastfm"),
        "discogs": _UPSERT.format(t="discogs"),
    }
    _SELECT_SQL: dict[str, str] = {
        "musicbrainz": "SELECT data, fetched_at FROM musicbrainz WHERE track_id = ?",
        "lastfm": "SELECT data, fetched_at FROM lastfm WHERE track_id = ?",
        "discogs": "SELECT data, fetched_at FROM discogs WHERE track_id = ?",
    }

    def _validate_table(self, table: str) -> None:
        if table not in self._VALID_TABLES:
            raise ValueError(f"Invalid enrichment source: {table!r}")

    def _store(self, table: str, track_id: int, data: dict[str, Any]) -> None:
        """Insert or replace enrichment data for a source."""
        self._validate_table(table)
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            self._INSERT_SQL[table],
            (track_id, json.dumps(data), now),
        )
        self._conn.commit()

    def _get(self, table: str, track_id: int) -> Optional[dict[str, Any]]:
        """Retrieve enrichment data for a source."""
        self._validate_table(table)
        cur = self._conn.execute(
            self._SELECT_SQL[table],
            (track_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "data": json.loads(row["data"]),
            "fetched_at": row["fetched_at"],
        }

    def store_musicbrainz(self, track_id: int, data: dict[str, Any]) -> None:
        self._store("musicbrainz", track_id, data)

    def store_lastfm(self, track_id: int, data: dict[str, Any]) -> None:
        self._store("lastfm", track_id, data)

    def store_discogs(self, track_id: int, data: dict[str, Any]) -> None:
        self._store("discogs", track_id, data)

    def get_musicbrainz(self, track_id: int) -> Optional[dict[str, Any]]:
        return self._get("musicbrainz", track_id)

    def get_lastfm(self, track_id: int) -> Optional[dict[str, Any]]:
        return self._get("lastfm", track_id)

    def get_discogs(self, track_id: int) -> Optional[dict[str, Any]]:
        return self._get("discogs", track_id)

    # ------------------------------------------------------------------
    # Aggregate queries
    # ------------------------------------------------------------------

    def get_all_enrichment(self, track_id: int) -> dict[str, Optional[dict[str, Any]]]:
        """Return all enrichment data for a track, keyed by source."""
        return {
            "musicbrainz": self.get_musicbrainz(track_id),
            "lastfm": self.get_lastfm(track_id),
            "discogs": self.get_discogs(track_id),
        }

    def get_enrichment_status(self, track_id: int) -> str:
        """Return enrichment completeness.

        One of ``"none"``, ``"partial"``, or ``"complete"``.
        """
        all_data = self.get_all_enrichment(track_id)
        present = sum(1 for v in all_data.values() if v is not None)
        if present == 0:
            return "none"
        if present == 3:
            return "complete"
        return "partial"

    def is_stale(
        self,
        track_id: int,
        source: str,
        ttl_days: int = 90,
    ) -> bool:
        """Check if enrichment data for *source* is stale (older than *ttl_days*).

        Returns ``True`` if no data exists or the data is older than *ttl_days*.
        """
        data = self._get(source, track_id)
        if data is None:
            return True

        fetched_at = datetime.fromisoformat(data["fetched_at"])
        age_days = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 86400
        return age_days > ttl_days

    def get_stats(self) -> dict[str, int]:
        """Return summary counts of tracks and enriched rows per source."""
        total = self._conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        mb = self._conn.execute("SELECT COUNT(*) FROM musicbrainz").fetchone()[0]
        lfm = self._conn.execute("SELECT COUNT(*) FROM lastfm").fetchone()[0]
        dc = self._conn.execute("SELECT COUNT(*) FROM discogs").fetchone()[0]
        return {
            "total_tracks": total,
            "musicbrainz": mb,
            "lastfm": lfm,
            "discogs": dc,
        }
