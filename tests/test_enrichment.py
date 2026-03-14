"""Tests for mlx_audiogen.library.enrichment — SQLite cache + rate limiter."""

import asyncio
import time

import pytest

from mlx_audiogen.library.enrichment.enrichment_db import EnrichmentDB
from mlx_audiogen.library.enrichment.rate_limiter import ApiRateLimiter

# ===========================================================================
# EnrichmentDB tests (15 tests, in-memory SQLite)
# ===========================================================================


class TestEnrichmentDB:
    @pytest.fixture()
    def db(self):
        """Fresh in-memory enrichment database for each test."""
        return EnrichmentDB(":memory:")

    # -- get_or_create_track -------------------------------------------------

    def test_get_or_create_track(self, db):
        """Creates a new track and returns its row id."""
        track_id = db.get_or_create_track("Daft Punk", "Around the World")
        assert isinstance(track_id, int)
        assert track_id > 0

    def test_dedup_normalization(self, db):
        """Same artist+title with different casing/whitespace returns same id."""
        id1 = db.get_or_create_track("Daft Punk", "Around the World")
        id2 = db.get_or_create_track("  daft punk ", " around the world  ")
        assert id1 == id2

    def test_library_id_mapping(self, db):
        """get_or_create_track stores library source and track id."""
        track_id = db.get_or_create_track(
            "Daft Punk", "Da Funk",
            library_source="apple_music",
            library_track_id="AM-1234",
        )
        found = db.find_by_library_id("apple_music", "AM-1234")
        assert found == track_id

    # -- store/get all 3 sources --------------------------------------------

    def test_store_get_musicbrainz(self, db):
        """Store and retrieve MusicBrainz data."""
        tid = db.get_or_create_track("Artist", "Song")
        data = {"mbid": "abc-123", "tags": ["electronic"]}
        db.store_musicbrainz(tid, data)
        result = db.get_musicbrainz(tid)
        assert result is not None
        assert result["data"]["mbid"] == "abc-123"

    def test_store_get_lastfm(self, db):
        """Store and retrieve Last.fm data."""
        tid = db.get_or_create_track("Artist", "Song")
        data = {"listeners": 50000, "top_tags": ["dance"]}
        db.store_lastfm(tid, data)
        result = db.get_lastfm(tid)
        assert result is not None
        assert result["data"]["listeners"] == 50000

    def test_store_get_discogs(self, db):
        """Store and retrieve Discogs data."""
        tid = db.get_or_create_track("Artist", "Song")
        data = {"release_id": 999, "year": 1997}
        db.store_discogs(tid, data)
        result = db.get_discogs(tid)
        assert result is not None
        assert result["data"]["year"] == 1997

    # -- missing returns None -----------------------------------------------

    def test_missing_returns_none(self, db):
        """get_* returns None for tracks with no enrichment data."""
        tid = db.get_or_create_track("New Artist", "New Song")
        assert db.get_musicbrainz(tid) is None
        assert db.get_lastfm(tid) is None
        assert db.get_discogs(tid) is None

    # -- enrichment status --------------------------------------------------

    def test_enrichment_status_none(self, db):
        """Status is 'none' when no sources are stored."""
        tid = db.get_or_create_track("A", "B")
        assert db.get_enrichment_status(tid) == "none"

    def test_enrichment_status_partial(self, db):
        """Status is 'partial' when some but not all sources are stored."""
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"mbid": "x"})
        assert db.get_enrichment_status(tid) == "partial"

    def test_enrichment_status_complete(self, db):
        """Status is 'complete' when all 3 sources are stored."""
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"mbid": "x"})
        db.store_lastfm(tid, {"listeners": 1})
        db.store_discogs(tid, {"year": 2020})
        assert db.get_enrichment_status(tid) == "complete"

    # -- is_stale -----------------------------------------------------------

    def test_is_stale(self, db):
        """is_stale returns True when data is older than ttl_days."""
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"mbid": "x"})
        # Not stale with default 90 days
        assert db.is_stale(tid, "musicbrainz", ttl_days=90) is False
        # Stale with 0 days (anything stored is immediately stale)
        assert db.is_stale(tid, "musicbrainz", ttl_days=0) is True
        # Missing source is always stale
        assert db.is_stale(tid, "lastfm", ttl_days=90) is True

    # -- update preserves other sources -------------------------------------

    def test_update_preserves_other_sources(self, db):
        """Updating one source does not affect other sources."""
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"mbid": "x"})
        db.store_lastfm(tid, {"listeners": 100})

        # Update musicbrainz
        db.store_musicbrainz(tid, {"mbid": "y"})

        # lastfm should be untouched
        lfm = db.get_lastfm(tid)
        assert lfm is not None
        assert lfm["data"]["listeners"] == 100

        # musicbrainz should be updated
        mb = db.get_musicbrainz(tid)
        assert mb is not None
        assert mb["data"]["mbid"] == "y"

    # -- get_stats ----------------------------------------------------------

    def test_get_stats(self, db):
        """get_stats returns counts of tracks and enriched sources."""
        tid1 = db.get_or_create_track("A", "Song1")
        tid2 = db.get_or_create_track("B", "Song2")
        db.store_musicbrainz(tid1, {"mbid": "x"})
        db.store_lastfm(tid1, {"listeners": 1})
        db.store_discogs(tid1, {"year": 2020})
        db.store_musicbrainz(tid2, {"mbid": "y"})

        stats = db.get_stats()
        assert stats["total_tracks"] == 2
        assert stats["musicbrainz"] == 2
        assert stats["lastfm"] == 1
        assert stats["discogs"] == 1

    # -- find_by_library_id missing -----------------------------------------

    def test_find_by_library_id_missing(self, db):
        """find_by_library_id returns None for unknown library ids."""
        assert db.find_by_library_id("apple_music", "nonexistent") is None

    # -- get_all_enrichment -------------------------------------------------

    def test_get_all_enrichment(self, db):
        """get_all_enrichment returns all stored sources for a track."""
        tid = db.get_or_create_track("A", "B")
        db.store_musicbrainz(tid, {"mbid": "x"})
        db.store_lastfm(tid, {"listeners": 1})

        result = db.get_all_enrichment(tid)
        assert result["musicbrainz"] is not None
        assert result["lastfm"] is not None
        assert result["discogs"] is None


# ===========================================================================
# Rate limiter tests (2 tests)
# ===========================================================================


class TestApiRateLimiter:
    def test_allows_within_limit(self):
        """A single acquire should complete without delay."""
        limiter = ApiRateLimiter(max_per_second=10.0)

        async def _run():
            t0 = time.monotonic()
            await limiter.acquire()
            elapsed = time.monotonic() - t0
            # Should be nearly instant
            assert elapsed < 0.05

        asyncio.run(_run())

    def test_tracks_separate_apis(self):
        """Separate limiter instances do not interfere with each other."""
        mb_limiter = ApiRateLimiter(max_per_second=1.0)
        lfm_limiter = ApiRateLimiter(max_per_second=5.0)

        async def _run():
            # First acquire on each should be instant
            await mb_limiter.acquire()
            await lfm_limiter.acquire()

            # Second acquire on mb_limiter should be delayed (~1s)
            # Second acquire on lfm_limiter should be much faster (~0.2s)
            t0 = time.monotonic()
            await lfm_limiter.acquire()
            lfm_elapsed = time.monotonic() - t0

            t1 = time.monotonic()
            await mb_limiter.acquire()
            mb_elapsed = time.monotonic() - t1

            # lfm should be faster than mb
            assert lfm_elapsed < mb_elapsed
            # mb should take roughly 1 second (use 0.7 for timing tolerance)
            assert mb_elapsed >= 0.7

        asyncio.run(_run())


# ===========================================================================
# MusicBrainz parser tests (2 tests)
# ===========================================================================


class TestMusicBrainz:
    def test_parse_response(self):
        from mlx_audiogen.library.enrichment.musicbrainz import _parse_musicbrainz_response

        fake = {"recordings": [{"id": "rec-123", "title": "Strobe",
            "artist-credit": [{"artist": {"id": "art-456", "name": "deadmau5"}}],
            "releases": [{"id": "rel-789", "release-group": {"id": "rg-101"}}],
            "tags": [{"name": "progressive house", "count": 5}]}]}
        result = _parse_musicbrainz_response(fake)
        assert result is not None
        assert result["artist_mbid"] == "art-456"
        assert "progressive house" in [t["name"] for t in result["tags"]]

    def test_parse_empty(self):
        from mlx_audiogen.library.enrichment.musicbrainz import _parse_musicbrainz_response

        assert _parse_musicbrainz_response({"recordings": []}) is None


# ===========================================================================
# Last.fm parser tests (2 tests)
# ===========================================================================


class TestLastFm:
    def test_parse_track_response(self):
        from mlx_audiogen.library.enrichment.lastfm import _parse_lastfm_track_response

        fake = {"track": {"name": "Strobe", "listeners": "500000", "playcount": "3000000",
            "toptags": {"tag": [{"name": "progressive house", "count": "100"}]},
            "similar": {"track": [{"name": "Ghosts", "artist": {"name": "deadmau5"}, "match": "0.9"}]}}}
        result = _parse_lastfm_track_response(fake)
        assert result is not None
        assert result["listeners"] == 500000
        assert len(result["tags"]) == 1

    def test_parse_error(self):
        from mlx_audiogen.library.enrichment.lastfm import _parse_lastfm_track_response

        assert _parse_lastfm_track_response({"error": 6}) is None


# ===========================================================================
# Discogs parser tests (2 tests)
# ===========================================================================


class TestDiscogs:
    def test_parse_search_response(self):
        from mlx_audiogen.library.enrichment.discogs import _parse_discogs_search_response

        fake = {"results": [{"id": 12345, "type": "master", "title": "CamelPhat - Cola",
            "genre": ["Electronic"], "style": ["Tech House"], "label": ["Defected"],
            "catno": "DFTD123", "year": "2017", "country": "UK"}]}
        result = _parse_discogs_search_response(fake)
        assert result is not None
        assert result["genres"] == ["Electronic"]
        assert result["year"] == 2017

    def test_parse_empty(self):
        from mlx_audiogen.library.enrichment.discogs import _parse_discogs_search_response

        assert _parse_discogs_search_response({"results": []}) is None


# ===========================================================================
# EnrichmentManager tests (2 tests)
# ===========================================================================

from unittest.mock import AsyncMock, patch

from mlx_audiogen.library.enrichment.manager import EnrichmentManager


@pytest.fixture
def manager():
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
    def test_enrich_musicbrainz_only(self, manager):
        """MusicBrainz is fetched without credentials; Last.fm/Discogs skipped."""
        mb_result = {"tags": [{"name": "house"}], "genres": ["house"], "artist_mbid": "abc"}

        async def run():
            with patch(
                "mlx_audiogen.library.enrichment.manager.search_musicbrainz",
                new=AsyncMock(return_value=mb_result),
            ):
                return await manager.enrich_single("Solomun", "Midnight Express")

        result = asyncio.run(run())
        assert result["musicbrainz"] is not None
        assert result["lastfm"] is None

    def test_cached_no_refetch(self, manager):
        """Second call for same track reuses cached data, no API call."""
        call_count = 0

        async def fake_mb(*a, **kw):
            nonlocal call_count
            call_count += 1
            return {"tags": [], "genres": [], "artist_mbid": "x"}

        async def run():
            with patch(
                "mlx_audiogen.library.enrichment.manager.search_musicbrainz",
                new=AsyncMock(side_effect=fake_mb),
            ):
                await manager.enrich_single("A", "B")
                await manager.enrich_single("A", "B")

        asyncio.run(run())
        assert call_count == 1

    def test_enrich_batch_progress(self, manager):
        """enrich_batch calls on_progress for each track and returns summary."""
        mb_result = {"tags": [], "genres": [], "artist_mbid": "x"}
        progress_calls = []

        async def run():
            with patch(
                "mlx_audiogen.library.enrichment.manager.search_musicbrainz",
                new=AsyncMock(return_value=mb_result),
            ):
                tracks = [
                    {"artist": "A", "title": "Song1"},
                    {"artist": "B", "title": "Song2"},
                ]
                result = await manager.enrich_batch(
                    tracks, on_progress=lambda **kw: progress_calls.append(kw)
                )
                return result

        result = asyncio.run(run())
        assert result["completed"] == 2
        assert result["errors"] == 0
        assert result["total"] == 2
        assert len(progress_calls) == 2

    def test_enrich_batch_cancellation(self, manager):
        """enrich_batch stops when cancel() is called."""
        mb_result = {"tags": [], "genres": [], "artist_mbid": "x"}
        completed = []

        async def run():
            with patch(
                "mlx_audiogen.library.enrichment.manager.search_musicbrainz",
                new=AsyncMock(return_value=mb_result),
            ):
                tracks = [
                    {"artist": "A", "title": "Song1"},
                    {"artist": "B", "title": "Song2"},
                    {"artist": "C", "title": "Song3"},
                ]
                # Cancel after first track via progress callback
                def on_prog(**kw):
                    completed.append(kw)
                    if kw["completed"] >= 1:
                        manager.cancel()

                result = await manager.enrich_batch(tracks, on_progress=on_prog)
                return result

        result = asyncio.run(run())
        # Should have stopped after 1 (cancelled before 2nd starts)
        assert result["completed"] <= 2
        assert len(completed) <= 2
