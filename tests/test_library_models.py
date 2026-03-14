"""Tests for mlx_audiogen.library.models."""

from mlx_audiogen.library.models import (
    LibrarySource,
    PlaylistInfo,
    TrackInfo,
    normalize_rating,
    slugify_playlist_name,
)

# ---------------------------------------------------------------------------
# normalize_rating
# ---------------------------------------------------------------------------


class TestNormalizeRating:
    def test_apple_music_passthrough(self):
        """Apple Music ratings (0-100) are returned unchanged."""
        assert normalize_rating(0, "apple_music") == 0
        assert normalize_rating(20, "apple_music") == 20
        assert normalize_rating(60, "apple_music") == 60
        assert normalize_rating(100, "apple_music") == 100

    def test_rekordbox_zero(self):
        assert normalize_rating(0, "rekordbox") == 0

    def test_rekordbox_max(self):
        """255 → 100 (exactly)."""
        assert normalize_rating(255, "rekordbox") == 100

    def test_rekordbox_midpoint(self):
        """128/255 rounds to 50."""
        result = normalize_rating(128, "rekordbox")
        assert result == round(128 * 100 / 255)

    def test_rekordbox_204(self):
        """204/255 rounds to 80."""
        assert normalize_rating(204, "rekordbox") == round(204 * 100 / 255)

    def test_rekordbox_51(self):
        """51/255 rounds to 20."""
        assert normalize_rating(51, "rekordbox") == round(51 * 100 / 255)


# ---------------------------------------------------------------------------
# slugify_playlist_name
# ---------------------------------------------------------------------------


class TestSlugifyPlaylistName:
    def test_simple_name(self):
        assert slugify_playlist_name("DJ Vassallo") == "dj-vassallo"

    def test_apostrophe_dropped(self):
        assert slugify_playlist_name("90's Music") == "90s-music"

    def test_multiple_spaces(self):
        assert slugify_playlist_name("Top  40") == "top-40"

    def test_leading_trailing_spaces(self):
        assert slugify_playlist_name("  spaces  ") == "spaces"

    def test_special_chars(self):
        assert slugify_playlist_name("Top 40!!!") == "top-40"

    def test_already_clean(self):
        assert slugify_playlist_name("house") == "house"

    def test_mixed_case(self):
        assert slugify_playlist_name("House & Techno") == "house-techno"

    def test_numbers(self):
        assert slugify_playlist_name("2024 Favorites") == "2024-favorites"


# ---------------------------------------------------------------------------
# TrackInfo
# ---------------------------------------------------------------------------


def _make_track(**kwargs) -> TrackInfo:
    defaults = dict(
        track_id="1",
        title="Test Track",
        artist="Test Artist",
        album="Test Album",
        genre="Electronic",
        bpm=128.0,
        key="4A",
        year=2023,
        rating=80,
        play_count=10,
        duration_seconds=210.0,
        comments="4A - 128",
        file_path="/tmp/test.wav",
        file_available=False,
        source="apple_music",
        loved=False,
        description="",
        description_edited=False,
    )
    defaults.update(kwargs)
    return TrackInfo(**defaults)


class TestTrackInfo:
    def test_to_dict_roundtrip(self):
        track = _make_track()
        d = track.to_dict()
        restored = TrackInfo.from_dict(d)
        assert restored.track_id == track.track_id
        assert restored.title == track.title
        assert restored.bpm == track.bpm
        assert restored.key == track.key
        assert restored.loved == track.loved
        assert restored.duration_seconds == track.duration_seconds

    def test_to_dict_contains_all_fields(self):
        track = _make_track()
        d = track.to_dict()
        expected_keys = {
            "track_id",
            "title",
            "artist",
            "album",
            "genre",
            "bpm",
            "key",
            "year",
            "rating",
            "play_count",
            "duration_seconds",
            "comments",
            "file_path",
            "file_available",
            "source",
            "loved",
            "description",
            "description_edited",
        }
        assert set(d.keys()) == expected_keys

    def test_from_dict_ignores_unknown_keys(self):
        track = _make_track()
        d = track.to_dict()
        d["unknown_future_field"] = "ignored"
        restored = TrackInfo.from_dict(d)
        assert restored.track_id == track.track_id

    def test_optional_fields_none(self):
        track = _make_track(bpm=None, key=None, year=None, file_path=None)
        d = track.to_dict()
        assert d["bpm"] is None
        assert d["key"] is None
        assert d["year"] is None
        restored = TrackInfo.from_dict(d)
        assert restored.bpm is None
        assert restored.key is None

    def test_source_field(self):
        track = _make_track(source="rekordbox")
        assert track.to_dict()["source"] == "rekordbox"


# ---------------------------------------------------------------------------
# PlaylistInfo
# ---------------------------------------------------------------------------


class TestPlaylistInfo:
    def test_to_dict(self):
        pl = PlaylistInfo(
            id="dj-vassallo",
            name="DJ Vassallo",
            track_count=2,
            track_ids=["101", "102"],
            source="apple_music",
        )
        d = pl.to_dict()
        assert d["id"] == "dj-vassallo"
        assert d["name"] == "DJ Vassallo"
        assert d["track_count"] == 2
        assert d["track_ids"] == ["101", "102"]
        assert d["source"] == "apple_music"

    def test_empty_playlist(self):
        pl = PlaylistInfo(id="empty", name="Empty", track_count=0)
        d = pl.to_dict()
        assert d["track_ids"] == []
        assert d["track_count"] == 0


# ---------------------------------------------------------------------------
# LibrarySource
# ---------------------------------------------------------------------------


class TestLibrarySource:
    def test_to_dict_roundtrip(self):
        src = LibrarySource(
            id="my-library",
            type="apple_music",
            path="/Users/me/Music/Library.xml",
            label="My Library",
            track_count=500,
            playlist_count=12,
            last_loaded="2026-03-14T10:00:00Z",
        )
        d = src.to_dict()
        restored = LibrarySource.from_dict(d)
        assert restored.id == src.id
        assert restored.type == src.type
        assert restored.path == src.path
        assert restored.track_count == src.track_count
        assert restored.last_loaded == src.last_loaded

    def test_from_dict_ignores_unknown_keys(self):
        src = LibrarySource(
            id="x",
            type="rekordbox",
            path="/tmp/rb.xml",
            label="RB",
            track_count=10,
            playlist_count=2,
            last_loaded=None,
        )
        d = src.to_dict()
        d["future_field"] = "value"
        restored = LibrarySource.from_dict(d)
        assert restored.id == "x"

    def test_last_loaded_none(self):
        src = LibrarySource(
            id="x",
            type="apple_music",
            path="/tmp/lib.xml",
            label="L",
            track_count=0,
            playlist_count=0,
            last_loaded=None,
        )
        assert src.to_dict()["last_loaded"] is None
