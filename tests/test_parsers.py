"""Tests for mlx_audiogen.library.parsers."""

import os
import tempfile
from pathlib import Path

import pytest

from mlx_audiogen.library.parsers import (
    extract_camelot_key,
    parse_apple_music_xml,
    parse_rekordbox_xml,
)

# Path to the fixture files
FIXTURES_DIR = Path(__file__).parent / "fixtures"
APPLE_MUSIC_FIXTURE = str(FIXTURES_DIR / "apple_music_sample.xml")
REKORDBOX_FIXTURE = str(FIXTURES_DIR / "rekordbox_sample.xml")


# ---------------------------------------------------------------------------
# extract_camelot_key
# ---------------------------------------------------------------------------


class TestExtractCamelotKey:
    def test_basic_a_key(self):
        assert extract_camelot_key("4A - 7") == "4A"

    def test_basic_b_key(self):
        assert extract_camelot_key("11B energy") == "11B"

    def test_lowercase_a(self):
        result = extract_camelot_key("key is 8a here")
        assert result is not None
        assert result.lower() == "8a"

    def test_no_key_returns_none(self):
        assert extract_camelot_key("no key here") is None

    def test_empty_string_returns_none(self):
        assert extract_camelot_key("") is None

    def test_12_is_valid(self):
        assert extract_camelot_key("12B - max") is not None

    def test_13_is_not_valid(self):
        """13 is outside Camelot wheel range."""
        assert extract_camelot_key("13A - invalid") is None

    def test_key_in_longer_string(self):
        assert extract_camelot_key("sunset vibes 4A - 128 BPM") == "4A"

    def test_1a_valid(self):
        assert extract_camelot_key("1A start") == "1A"

    def test_key_in_comment_format(self):
        assert extract_camelot_key("7B - 90 BPM gentle") == "7B"


# ---------------------------------------------------------------------------
# Apple Music XML parser
# ---------------------------------------------------------------------------


class TestParseAppleMusicXml:
    def test_track_count(self):
        tracks, _ = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        assert len(tracks) == 3

    def test_track_basic_metadata(self):
        tracks, _ = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        t = tracks["101"]
        assert t.title == "Sundown Groove"
        assert t.artist == "DJ Vassallo"
        assert t.album == "Sunset Sessions"
        assert t.genre == "Electronic"
        assert t.source == "apple_music"

    def test_track_numeric_metadata(self):
        tracks, _ = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        t = tracks["101"]
        assert t.bpm == 128.0
        assert t.year == 2023
        assert t.rating == 80
        assert t.play_count == 42

    def test_track_duration_converts_from_ms(self):
        """Total Time is milliseconds; parser converts to seconds."""
        tracks, _ = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        t = tracks["101"]
        assert t.duration_seconds == pytest.approx(210.0)

    def test_track_duration_360s(self):
        tracks, _ = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        t = tracks["102"]
        assert t.duration_seconds == pytest.approx(360.0)

    def test_camelot_key_extracted_from_comments(self):
        tracks, _ = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        t = tracks["101"]
        assert t.key == "4A"

    def test_loved_flag(self):
        tracks, _ = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        assert tracks["101"].loved is True

    def test_favorited_maps_to_loved(self):
        """Older 'Favorited' key should also set loved=True."""
        tracks, _ = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        assert tracks["102"].loved is True

    def test_file_path_resolved_triple_slash(self):
        tracks, _ = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        t = tracks["101"]
        assert t.file_path == "/Users/testuser/Music/sundown_groove.wav"

    def test_file_path_resolved_localhost(self):
        tracks, _ = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        t = tracks["102"]
        assert t.file_path == "/Users/testuser/Music/morning_mist.mp3"

    def test_file_not_available_for_missing_path(self):
        """Files in fixture don't exist on disk → file_available=False."""
        tracks, _ = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        assert tracks["101"].file_available is False
        assert tracks["102"].file_available is False

    def test_sparse_track_missing_fields(self):
        """Track 103 has minimal metadata; optional fields should be None/defaults."""
        tracks, _ = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        t = tracks["103"]
        assert t.title == "Sparse Track"
        assert t.artist == ""
        assert t.bpm is None
        assert t.year is None
        assert t.key is None
        assert t.rating == 0
        assert t.play_count == 0
        assert t.loved is False
        assert t.duration_seconds == pytest.approx(0.0)

    def test_playlist_count(self):
        _, playlists = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        assert len(playlists) == 3

    def test_playlist_names_and_slugs(self):
        _, playlists = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        names = {p.name: p.id for p in playlists}
        assert "DJ Vassallo" in names
        assert names["DJ Vassallo"] == "dj-vassallo"
        assert "Library" in names
        assert names["Library"] == "library"

    def test_playlist_track_ids(self):
        _, playlists = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        by_name = {p.name: p for p in playlists}
        pl = by_name["DJ Vassallo"]
        assert "101" in pl.track_ids
        assert "102" in pl.track_ids
        assert pl.track_count == 2

    def test_empty_playlist(self):
        _, playlists = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        by_name = {p.name: p for p in playlists}
        pl = by_name["Empty Playlist"]
        assert pl.track_count == 0
        assert pl.track_ids == []

    def test_playlist_source(self):
        _, playlists = parse_apple_music_xml(APPLE_MUSIC_FIXTURE)
        for pl in playlists:
            assert pl.source == "apple_music"

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_apple_music_xml("/nonexistent/library.xml")

    def test_malformed_xml_raises_value_error(self):
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write("<<not valid plist>>")
            tmp = f.name
        try:
            with pytest.raises(ValueError):
                parse_apple_music_xml(tmp)
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# rekordbox XML parser
# ---------------------------------------------------------------------------


class TestParseRekordboxXml:
    def test_track_count(self):
        tracks, _ = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        assert len(tracks) == 3

    def test_track_basic_metadata(self):
        tracks, _ = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        t = tracks["1"]
        assert t.title == "Deep House Anthem"
        assert t.artist == "Studio Artist"
        assert t.album == "Club Sessions"
        assert t.genre == "House"
        assert t.source == "rekordbox"

    def test_track_numeric_metadata(self):
        tracks, _ = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        t = tracks["1"]
        assert t.bpm == pytest.approx(124.0)
        assert t.year == 2021
        assert t.play_count == 30

    def test_track_duration_already_seconds(self):
        """TotalTime in rekordbox is already in seconds."""
        tracks, _ = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        t = tracks["1"]
        assert t.duration_seconds == pytest.approx(240.0)

    def test_track_key_from_tonality(self):
        tracks, _ = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        t = tracks["1"]
        assert t.key == "8A"

    def test_rating_normalization(self):
        """204/255 → ~80."""
        tracks, _ = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        t = tracks["1"]
        from mlx_audiogen.library.models import normalize_rating

        expected = normalize_rating(204, "rekordbox")
        assert t.rating == expected

    def test_rating_midpoint(self):
        """153/255 normalizes correctly."""
        tracks, _ = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        t = tracks["2"]
        from mlx_audiogen.library.models import normalize_rating

        expected = normalize_rating(153, "rekordbox")
        assert t.rating == expected

    def test_soundcloud_track_no_file_path(self):
        """SoundCloud stream tracks → file_path=None, file_available=False."""
        tracks, _ = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        t = tracks["3"]
        assert t.file_path is None
        assert t.file_available is False

    def test_local_wav_file_path_resolved(self):
        tracks, _ = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        t = tracks["1"]
        assert t.file_path == "/Users/testuser/Music/deep_house_anthem.wav"

    def test_local_mp3_file_path_resolved(self):
        tracks, _ = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        t = tracks["2"]
        assert t.file_path == "/Users/testuser/Music/tech_minimal.mp3"

    def test_file_not_available_for_missing_files(self):
        """Fixture paths don't exist on disk."""
        tracks, _ = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        assert tracks["1"].file_available is False
        assert tracks["2"].file_available is False

    def test_playlist_count(self):
        _, playlists = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        # Folder nodes are not playlists — only Type="1" nodes
        assert len(playlists) == 2

    def test_playlist_names(self):
        _, playlists = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        names = {p.name for p in playlists}
        assert "My Playlist" in names
        assert "Empty" in names

    def test_playlist_slugs(self):
        _, playlists = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        by_name = {p.name: p for p in playlists}
        assert by_name["My Playlist"].id == "my-playlist"

    def test_playlist_track_ids(self):
        _, playlists = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        by_name = {p.name: p for p in playlists}
        pl = by_name["My Playlist"]
        assert "1" in pl.track_ids
        assert "2" in pl.track_ids
        assert pl.track_count == 2

    def test_empty_playlist(self):
        _, playlists = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        by_name = {p.name: p for p in playlists}
        pl = by_name["Empty"]
        assert pl.track_count == 0
        assert pl.track_ids == []

    def test_playlist_source(self):
        _, playlists = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        for pl in playlists:
            assert pl.source == "rekordbox"

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            parse_rekordbox_xml("/nonexistent/rekordbox.xml")

    def test_malformed_xml_raises_value_error(self):
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write("<<not valid xml")
            tmp = f.name
        try:
            with pytest.raises(ValueError):
                parse_rekordbox_xml(tmp)
        finally:
            os.unlink(tmp)

    def test_track_empty_tonality_is_none(self):
        """Track 3 has empty Tonality → key=None."""
        tracks, _ = parse_rekordbox_xml(REKORDBOX_FIXTURE)
        t = tracks["3"]
        assert t.key is None
