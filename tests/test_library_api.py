"""API endpoint tests for library scanner + collection routes.

Uses FastAPI TestClient with fixture XML files. Tests all library source,
browsing, collection CRUD, export/import, and AI endpoints.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlx_audiogen.server.app import (
    _rate_limiter,
    app,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _clean_library_state(tmp_path, monkeypatch):
    """Reset library cache and collections between tests."""
    # Get server module (can't use 'import ... as' because 'app' attribute shadows it)
    import sys  # noqa: E402

    import mlx_audiogen.library.collections as col_mod
    from mlx_audiogen.library.cache import LibraryCache

    srv_mod = sys.modules["mlx_audiogen.server.app"]

    # Create a fresh cache with isolated config dir per test
    test_cache = LibraryCache(config_dir=tmp_path / "config")

    # Monkeypatch _get_library_cache to always return the test cache
    monkeypatch.setattr(srv_mod, "_get_library_cache", lambda: test_cache)

    # Patch collections dir to use tmp_path
    collections_dir = tmp_path / "collections"
    collections_dir.mkdir()
    monkeypatch.setattr(col_mod, "DEFAULT_COLLECTIONS_DIR", collections_dir)

    # Clear rate limiter
    _rate_limiter._generate_hits.clear()
    _rate_limiter._general_hits.clear()

    # Patch enrichment / credential / taste singletons for isolation
    from mlx_audiogen.library.enrichment.enrichment_db import EnrichmentDB
    from mlx_audiogen.library.enrichment.manager import EnrichmentManager
    from mlx_audiogen.library.taste.engine import TasteEngine

    test_enrichment_db = EnrichmentDB(":memory:")
    monkeypatch.setattr(srv_mod, "_enrichment_db", test_enrichment_db)
    monkeypatch.setattr(
        srv_mod,
        "_enrichment_manager",
        EnrichmentManager(db=test_enrichment_db, credentials=None),
    )
    monkeypatch.setattr(
        srv_mod, "_taste_engine", TasteEngine(profile_path=str(tmp_path / "taste.json"))
    )
    monkeypatch.setattr(srv_mod, "_enrichment_job", None)

    # Patch credential manager with a fake that doesn't hit the keychain
    class _FakeCredentialManager:
        def status(self):
            return {"musicbrainz": True, "lastfm": False, "discogs": False}

        def set(self, service, value):
            from mlx_audiogen.credentials import _SERVICES

            if service not in _SERVICES:
                raise ValueError(f"Unknown service: {service!r}")

        def delete(self, service):
            from mlx_audiogen.credentials import _SERVICES

            if service not in _SERVICES:
                raise ValueError(f"Unknown service: {service!r}")

        def get(self, service):
            return None

    monkeypatch.setattr(srv_mod, "_credential_manager", _FakeCredentialManager())


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Library Sources
# ---------------------------------------------------------------------------


class TestLibrarySources:
    def test_list_sources_empty(self, client: TestClient):
        res = client.get("/api/library/sources")
        assert res.status_code == 200
        assert res.json() == []

    def test_add_source(self, client: TestClient):
        res = client.post(
            "/api/library/sources",
            json={
                "type": "apple_music",
                "path": str(FIXTURES / "apple_music_sample.xml"),
                "label": "Test AM",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["type"] == "apple_music"
        assert data["label"] == "Test AM"
        assert "id" in data

    def test_add_source_invalid_type(self, client: TestClient):
        res = client.post(
            "/api/library/sources",
            json={
                "type": "spotify",
                "path": str(FIXTURES / "apple_music_sample.xml"),
                "label": "Bad",
            },
        )
        assert res.status_code == 422  # Pydantic validation

    def test_add_source_file_not_found(self, client: TestClient):
        res = client.post(
            "/api/library/sources",
            json={
                "type": "apple_music",
                "path": "/nonexistent/Library.xml",
                "label": "Missing",
            },
        )
        assert res.status_code == 400

    def test_add_source_path_traversal(self, client: TestClient):
        res = client.post(
            "/api/library/sources",
            json={
                "type": "apple_music",
                "path": "/etc/../etc/passwd",
                "label": "Evil",
            },
        )
        assert res.status_code == 400

    def test_list_sources_after_add(self, client: TestClient):
        client.post(
            "/api/library/sources",
            json={
                "type": "apple_music",
                "path": str(FIXTURES / "apple_music_sample.xml"),
                "label": "AM",
            },
        )
        res = client.get("/api/library/sources")
        assert res.status_code == 200
        assert len(res.json()) == 1

    def test_update_source(self, client: TestClient):
        add_res = client.post(
            "/api/library/sources",
            json={
                "type": "rekordbox",
                "path": str(FIXTURES / "rekordbox_sample.xml"),
                "label": "Old Label",
            },
        )
        source_id = add_res.json()["id"]
        res = client.put(
            f"/api/library/sources/{source_id}",
            json={"label": "New Label"},
        )
        assert res.status_code == 200
        assert res.json()["label"] == "New Label"

    def test_update_source_not_found(self, client: TestClient):
        res = client.put(
            "/api/library/sources/nonexistent",
            json={"label": "New"},
        )
        assert res.status_code == 404

    def test_delete_source(self, client: TestClient):
        add_res = client.post(
            "/api/library/sources",
            json={
                "type": "apple_music",
                "path": str(FIXTURES / "apple_music_sample.xml"),
                "label": "Del",
            },
        )
        source_id = add_res.json()["id"]
        res = client.delete(f"/api/library/sources/{source_id}")
        assert res.status_code == 200
        assert res.json()["deleted"] == source_id

        # Verify gone
        list_res = client.get("/api/library/sources")
        assert len(list_res.json()) == 0

    def test_delete_source_not_found(self, client: TestClient):
        res = client.delete("/api/library/sources/nonexistent")
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# Scanning + Browsing
# ---------------------------------------------------------------------------


class TestLibraryBrowsing:
    def _add_and_scan(self, client, source_type, xml_file, label="Test"):
        """Helper: add a source and scan it. Returns source_id."""
        add_res = client.post(
            "/api/library/sources",
            json={
                "type": source_type,
                "path": str(FIXTURES / xml_file),
                "label": label,
            },
        )
        source_id = add_res.json()["id"]
        scan_res = client.post(f"/api/library/scan/{source_id}")
        assert scan_res.status_code == 200
        return source_id

    def test_scan_apple_music(self, client: TestClient):
        source_id = self._add_and_scan(client, "apple_music", "apple_music_sample.xml")
        # Verify track count
        res = client.get(f"/api/library/tracks/{source_id}")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 3

    def test_scan_rekordbox(self, client: TestClient):
        source_id = self._add_and_scan(client, "rekordbox", "rekordbox_sample.xml")
        res = client.get(f"/api/library/tracks/{source_id}")
        assert res.status_code == 200
        assert res.json()["count"] == 3

    def test_scan_not_found(self, client: TestClient):
        res = client.post("/api/library/scan/nonexistent")
        assert res.status_code == 404

    def test_playlists(self, client: TestClient):
        source_id = self._add_and_scan(client, "apple_music", "apple_music_sample.xml")
        res = client.get(f"/api/library/playlists/{source_id}")
        assert res.status_code == 200
        playlists = res.json()
        assert len(playlists) >= 2
        names = [p["name"] for p in playlists]
        assert "DJ Vassallo" in names

    def test_playlist_tracks(self, client: TestClient):
        source_id = self._add_and_scan(client, "apple_music", "apple_music_sample.xml")
        # Get playlist ID for "DJ Vassallo"
        playlists_res = client.get(f"/api/library/playlists/{source_id}")
        dj_pl = next(p for p in playlists_res.json() if p["name"] == "DJ Vassallo")
        res = client.get(f"/api/library/playlist-tracks/{source_id}/{dj_pl['id']}")
        assert res.status_code == 200
        assert res.json()["count"] == 2

    def test_search_tracks_text(self, client: TestClient):
        source_id = self._add_and_scan(client, "apple_music", "apple_music_sample.xml")
        res = client.get(f"/api/library/tracks/{source_id}", params={"q": "Sundown"})
        assert res.status_code == 200
        tracks = res.json()["tracks"]
        assert len(tracks) >= 1
        assert tracks[0]["title"] == "Sundown Groove"

    def test_search_tracks_bpm_range(self, client: TestClient):
        source_id = self._add_and_scan(client, "apple_music", "apple_music_sample.xml")
        res = client.get(
            f"/api/library/tracks/{source_id}",
            params={"bpm_min": 85, "bpm_max": 95},
        )
        assert res.status_code == 200
        tracks = res.json()["tracks"]
        assert len(tracks) == 1
        assert tracks[0]["title"] == "Morning Mist"

    def test_sort_tracks(self, client: TestClient):
        source_id = self._add_and_scan(client, "apple_music", "apple_music_sample.xml")
        res = client.get(
            f"/api/library/tracks/{source_id}",
            params={"sort": "bpm", "order": "desc"},
        )
        assert res.status_code == 200
        tracks = res.json()["tracks"]
        with_bpm = [t for t in tracks if t["bpm"] is not None]
        if len(with_bpm) >= 2:
            assert with_bpm[0]["bpm"] >= with_bpm[1]["bpm"]

    def test_pagination(self, client: TestClient):
        source_id = self._add_and_scan(client, "apple_music", "apple_music_sample.xml")
        res = client.get(
            f"/api/library/tracks/{source_id}",
            params={"offset": 0, "limit": 1},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == 1
        assert data["offset"] == 0
        assert data["limit"] == 1


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


class TestCollections:
    def test_list_collections_empty(self, client: TestClient):
        res = client.get("/api/collections")
        assert res.status_code == 200
        assert res.json() == []

    def test_create_collection(self, client: TestClient):
        res = client.post(
            "/api/collections",
            json={
                "name": "test-col",
                "source": "apple_music",
                "tracks": [{"track_id": "1", "title": "Track 1"}],
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "test-col"
        assert "created_at" in data

    def test_create_collection_invalid_name(self, client: TestClient):
        res = client.post(
            "/api/collections",
            json={"name": "../evil", "tracks": []},
        )
        assert res.status_code == 422  # Pydantic pattern validation

    def test_list_collections_after_create(self, client: TestClient):
        client.post(
            "/api/collections",
            json={"name": "col-a", "tracks": []},
        )
        client.post(
            "/api/collections",
            json={
                "name": "col-b",
                "tracks": [{"track_id": "1"}],
            },
        )
        res = client.get("/api/collections")
        assert res.status_code == 200
        assert len(res.json()) == 2

    def test_get_collection(self, client: TestClient):
        client.post(
            "/api/collections",
            json={
                "name": "my-col",
                "tracks": [{"track_id": "1", "title": "T"}],
            },
        )
        res = client.get("/api/collections/my-col")
        assert res.status_code == 200
        assert res.json()["name"] == "my-col"
        assert len(res.json()["tracks"]) == 1

    def test_get_collection_not_found(self, client: TestClient):
        res = client.get("/api/collections/nonexistent")
        assert res.status_code == 404

    def test_update_collection(self, client: TestClient):
        client.post(
            "/api/collections",
            json={"name": "upd", "tracks": []},
        )
        res = client.put(
            "/api/collections/upd",
            json={"tracks": [{"track_id": "99", "description": "updated"}]},
        )
        assert res.status_code == 200
        assert len(res.json()["tracks"]) == 1
        assert res.json()["tracks"][0]["track_id"] == "99"

    def test_update_collection_not_found(self, client: TestClient):
        res = client.put(
            "/api/collections/nonexistent",
            json={"tracks": []},
        )
        assert res.status_code == 404

    def test_delete_collection(self, client: TestClient):
        client.post(
            "/api/collections",
            json={"name": "del-me", "tracks": []},
        )
        res = client.delete("/api/collections/del-me")
        assert res.status_code == 200
        assert res.json()["deleted"] == "del-me"

        # Verify gone
        res = client.get("/api/collections/del-me")
        assert res.status_code == 404

    def test_delete_collection_not_found(self, client: TestClient):
        res = client.delete("/api/collections/nonexistent")
        assert res.status_code == 404

    def test_export_collection(self, client: TestClient):
        client.post(
            "/api/collections",
            json={
                "name": "export-me",
                "tracks": [{"track_id": "1"}],
            },
        )
        res = client.get("/api/collections/export-me/export")
        assert res.status_code == 200
        assert "application/json" in res.headers["content-type"]
        assert "attachment" in res.headers.get("content-disposition", "")
        data = res.json()
        assert data["name"] == "export-me"

    def test_export_collection_not_found(self, client: TestClient):
        res = client.get("/api/collections/nonexistent/export")
        assert res.status_code == 404

    def test_import_collection(self, client: TestClient):
        collection_data = json.dumps(
            {
                "name": "imported",
                "tracks": [{"track_id": "42", "title": "Imported Track"}],
            }
        )
        res = client.post(
            "/api/collections/import",
            files={"file": ("imported.json", collection_data, "application/json")},
        )
        assert res.status_code == 200
        assert res.json()["name"] == "imported"

        # Verify it's accessible
        get_res = client.get("/api/collections/imported")
        assert get_res.status_code == 200
        assert len(get_res.json()["tracks"]) == 1

    def test_import_collection_invalid_json(self, client: TestClient):
        res = client.post(
            "/api/collections/import",
            files={"file": ("bad.json", "not json {{{", "application/json")},
        )
        assert res.status_code == 400

    def test_import_collection_missing_name(self, client: TestClient):
        res = client.post(
            "/api/collections/import",
            files={
                "file": (
                    "no_name.json",
                    json.dumps({"tracks": []}),
                    "application/json",
                )
            },
        )
        assert res.status_code == 400


# ---------------------------------------------------------------------------
# AI Endpoints
# ---------------------------------------------------------------------------


class TestLibraryAI:
    def _add_and_scan(self, client, source_type, xml_file):
        """Helper: add, scan, return source_id."""
        add_res = client.post(
            "/api/library/sources",
            json={
                "type": source_type,
                "path": str(FIXTURES / xml_file),
                "label": "Test",
            },
        )
        source_id = add_res.json()["id"]
        client.post(f"/api/library/scan/{source_id}")
        return source_id

    def test_describe_template(self, client: TestClient):
        source_id = self._add_and_scan(client, "apple_music", "apple_music_sample.xml")
        res = client.post(
            "/api/library/describe",
            json={
                "source_id": source_id,
                "track_ids": ["101", "102"],
                "mode": "template",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert "101" in data["descriptions"]
        assert "102" in data["descriptions"]
        # Track 101: Electronic, 128 BPM, 4A, DJ Vassallo style
        desc101 = data["descriptions"]["101"]
        assert "electronic" in desc101.lower()
        assert "128" in desc101

    def test_describe_missing_tracks(self, client: TestClient):
        source_id = self._add_and_scan(client, "apple_music", "apple_music_sample.xml")
        res = client.post(
            "/api/library/describe",
            json={
                "source_id": source_id,
                "track_ids": ["999"],
                "mode": "template",
            },
        )
        assert res.status_code == 200
        assert len(res.json()["descriptions"]) == 0

    def test_suggest_name(self, client: TestClient):
        source_id = self._add_and_scan(client, "apple_music", "apple_music_sample.xml")
        res = client.post(
            "/api/library/suggest-name",
            json={
                "source_id": source_id,
                "track_ids": ["101", "102"],
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert "name" in data
        assert isinstance(data["name"], str)
        assert len(data["name"]) > 0
        assert len(data["name"]) <= 64

    def test_suggest_name_no_tracks(self, client: TestClient):
        source_id = self._add_and_scan(client, "apple_music", "apple_music_sample.xml")
        res = client.post(
            "/api/library/suggest-name",
            json={
                "source_id": source_id,
                "track_ids": ["999"],
            },
        )
        assert res.status_code == 200
        assert res.json()["name"] == "my-style"

    def test_generate_prompt(self, client: TestClient):
        source_id = self._add_and_scan(client, "apple_music", "apple_music_sample.xml")
        res = client.post(
            "/api/library/generate-prompt",
            json={
                "source_id": source_id,
                "track_ids": ["101", "102", "103"],
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert "prompt" in data
        assert "track_count" in data
        assert data["track_count"] == 3
        assert "top_genres" in data
        assert len(data["prompt"]) > 0

    def test_generate_prompt_no_tracks(self, client: TestClient):
        source_id = self._add_and_scan(client, "apple_music", "apple_music_sample.xml")
        res = client.post(
            "/api/library/generate-prompt",
            json={
                "source_id": source_id,
                "track_ids": ["999"],
            },
        )
        assert res.status_code == 200
        assert res.json()["track_count"] == 0
        assert res.json()["prompt"] == "instrumental music"


# ---------------------------------------------------------------------------
# Credential Endpoints
# ---------------------------------------------------------------------------


class TestCredentialEndpoints:
    def test_status(self, client: TestClient):
        res = client.get("/api/credentials/status")
        assert res.status_code == 200
        assert res.json()["musicbrainz"] is True

    def test_set_invalid_service(self, client: TestClient):
        res = client.post("/api/credentials/spotify_key", json={"api_key": "v"})
        assert res.status_code == 400

    def test_set_missing_api_key(self, client: TestClient):
        res = client.post("/api/credentials/lastfm_api_key", json={})
        assert res.status_code == 400

    def test_set_valid_service(self, client: TestClient):
        res = client.post("/api/credentials/lastfm_api_key", json={"api_key": "test123"})
        assert res.status_code == 200
        assert res.json()["status"] == "saved"

    def test_delete_invalid_service(self, client: TestClient):
        res = client.delete("/api/credentials/spotify_key")
        assert res.status_code == 400

    def test_delete_valid_service(self, client: TestClient):
        res = client.delete("/api/credentials/lastfm_api_key")
        assert res.status_code == 200
        assert res.json()["status"] == "deleted"


# ---------------------------------------------------------------------------
# Enrichment Endpoints
# ---------------------------------------------------------------------------


class TestEnrichmentEndpoints:
    def test_status_idle(self, client: TestClient):
        res = client.get("/api/enrich/status")
        assert res.status_code == 200
        assert res.json()["status"] == "idle"

    def test_stats_empty(self, client: TestClient):
        res = client.get("/api/enrich/stats")
        assert res.status_code == 200
        assert res.json()["total_tracks"] == 0

    def test_enrich_empty_body(self, client: TestClient):
        res = client.post("/api/enrich/tracks", json={})
        assert res.status_code == 400

    def test_cancel(self, client: TestClient):
        res = client.post("/api/enrich/cancel")
        assert res.status_code == 200
        assert res.json()["status"] == "cancelled"

    def test_track_get_nonexistent(self, client: TestClient):
        res = client.get("/api/enrich/track/99999")
        assert res.status_code == 200
        assert res.json()["status"] == "none"


# ---------------------------------------------------------------------------
# Taste Endpoints
# ---------------------------------------------------------------------------


class TestTasteEndpoints:
    def test_empty_profile(self, client: TestClient):
        res = client.get("/api/taste/profile")
        assert res.status_code == 200
        assert res.json()["library_track_count"] == 0

    def test_set_overrides(self, client: TestClient):
        res = client.put("/api/taste/overrides", json={"text": "more minimal"})
        assert res.status_code == 200
        assert res.json()["overrides"] == "more minimal"

    def test_suggestions_empty(self, client: TestClient):
        res = client.get("/api/taste/suggestions")
        assert res.status_code == 200
        assert "suggestions" in res.json()

    def test_refresh(self, client: TestClient):
        res = client.post("/api/taste/refresh")
        assert res.status_code == 200
        assert "library_track_count" in res.json()
