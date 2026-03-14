"""Enrichment manager — orchestrates metadata fetching across 3 API sources.

Coordinates MusicBrainz, Last.fm, and Discogs enrichment with per-API
rate limiting, credential checks, and caching through :class:`EnrichmentDB`.

Usage::

    manager = EnrichmentManager()
    result = await manager.enrich_single("Daft Punk", "Around the World")
    # result == {"musicbrainz": {...}, "lastfm": {...}, "discogs": {...}}
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from .enrichment_db import EnrichmentDB
from .musicbrainz import search_musicbrainz
from .lastfm import search_lastfm
from .discogs import search_discogs
from .rate_limiter import ApiRateLimiter

logger = logging.getLogger(__name__)


class EnrichmentManager:
    """Orchestrates enrichment across MusicBrainz, Last.fm, and Discogs."""

    def __init__(
        self,
        db: Optional[EnrichmentDB] = None,
        credentials: Optional[Any] = None,
    ) -> None:
        self._db = db if db is not None else EnrichmentDB()
        self._cred = credentials
        self._mb_limiter = ApiRateLimiter(max_per_second=1.0)
        self._lfm_limiter = ApiRateLimiter(max_per_second=5.0)
        self._dc_limiter = ApiRateLimiter(max_per_second=1.0)
        self._cancelled = False

    @property
    def db(self) -> EnrichmentDB:
        """Access the underlying enrichment database."""
        return self._db

    def cancel(self) -> None:
        """Signal that the current batch job should stop."""
        self._cancelled = True

    def reset_cancel(self) -> None:
        """Clear the cancellation flag."""
        self._cancelled = False

    async def enrich_single(
        self,
        artist: str,
        title: str,
        library_source: Optional[str] = None,
        library_track_id: Optional[str] = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Enrich a single track across all available API sources.

        Args:
            artist: Track artist name.
            title: Track title.
            library_source: Optional library source identifier.
            library_track_id: Optional library-specific track ID.
            force: If True, re-fetch even if cached data is fresh.

        Returns:
            Dict with keys ``musicbrainz``, ``lastfm``, ``discogs`` — each
            containing the cached enrichment data or ``None``.
        """
        tid = self._db.get_or_create_track(
            artist, title,
            library_source=library_source,
            library_track_id=library_track_id,
        )

        # --- MusicBrainz (no credentials needed) ---
        if force or self._db.is_stale(tid, "musicbrainz"):
            try:
                mb_data = await search_musicbrainz(
                    artist, title, rate_limiter=self._mb_limiter
                )
                if mb_data is not None:
                    self._db.store_musicbrainz(tid, mb_data)
            except Exception:
                logger.exception("MusicBrainz enrichment failed for %s - %s", artist, title)

        # --- Last.fm (needs lastfm_api_key) ---
        lfm_key = self._cred.get("lastfm_api_key") if self._cred is not None else None
        if lfm_key is not None and (force or self._db.is_stale(tid, "lastfm")):
            try:
                lfm_data = await search_lastfm(
                    artist, title,
                    api_key=lfm_key,
                    rate_limiter=self._lfm_limiter,
                )
                if lfm_data is not None:
                    self._db.store_lastfm(tid, lfm_data)
            except Exception:
                logger.exception("Last.fm enrichment failed for %s - %s", artist, title)

        # --- Discogs (needs discogs_token) ---
        dc_token = self._cred.get("discogs_token") if self._cred is not None else None
        if dc_token is not None and (force or self._db.is_stale(tid, "discogs")):
            try:
                dc_data = await search_discogs(
                    artist, title,
                    token=dc_token,
                    rate_limiter=self._dc_limiter,
                )
                if dc_data is not None:
                    self._db.store_discogs(tid, dc_data)
            except Exception:
                logger.exception("Discogs enrichment failed for %s - %s", artist, title)

        return self._db.get_all_enrichment(tid)

    async def enrich_batch(
        self,
        tracks: list[dict[str, str]],
        on_progress: Optional[Callable[..., Any]] = None,
    ) -> dict[str, int]:
        """Enrich a batch of tracks, reporting progress.

        Args:
            tracks: List of dicts with at least ``artist`` and ``title`` keys.
                    May also contain ``library_source`` and ``library_track_id``.
            on_progress: Optional callback called after each track with kwargs:
                ``completed``, ``errors``, ``total``, ``current``.

        Returns:
            Summary dict with ``completed``, ``errors``, ``total``.
        """
        total = len(tracks)
        completed = 0
        errors = 0

        for track in tracks:
            if self._cancelled:
                break

            artist = track["artist"]
            title = track["title"]
            current = f"{artist} - {title}"

            try:
                await self.enrich_single(
                    artist,
                    title,
                    library_source=track.get("library_source"),
                    library_track_id=track.get("library_track_id"),
                )
                completed += 1
            except Exception:
                logger.exception("Batch enrichment error for %s", current)
                errors += 1
                completed += 1  # count as processed even on error

            if on_progress is not None:
                on_progress(
                    completed=completed,
                    errors=errors,
                    total=total,
                    current=current,
                )

        return {"completed": completed, "errors": errors, "total": total}
