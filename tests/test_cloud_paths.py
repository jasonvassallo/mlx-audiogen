"""Tests for mlx_audiogen.library.cloud_paths."""

import os
import tempfile

from mlx_audiogen.library.cloud_paths import (
    check_file_available,
    is_icloud_placeholder,
    resolve_file_url,
)

# ---------------------------------------------------------------------------
# resolve_file_url
# ---------------------------------------------------------------------------


class TestResolveFileUrl:
    def test_file_triple_slash(self):
        """file:///path → /path"""
        assert (
            resolve_file_url("file:///Volumes/Music/track.wav")
            == "/Volumes/Music/track.wav"
        )

    def test_file_localhost(self):
        """file://localhost/path → /path"""
        assert (
            resolve_file_url("file://localhost/Users/me/track.mp3")
            == "/Users/me/track.mp3"
        )

    def test_percent_encoding_decoded(self):
        """URL-encoded spaces/special chars are decoded."""
        url = "file:///Users/me/My%20Music/track%231.wav"
        assert resolve_file_url(url) == "/Users/me/My Music/track#1.wav"

    def test_soundcloud_returns_none(self):
        """SoundCloud stream references → None."""
        assert resolve_file_url("soundcloud:tracks:123456") is None

    def test_rekordbox_soundcloud_in_file_url(self):
        """rekordbox embeds SoundCloud as file://localhostsoundcloud:... → None."""
        assert resolve_file_url("file://localhostsoundcloud:tracks:123456789") is None

    def test_catalog_ref_returns_none(self):
        """Apple Music catalog references → None."""
        assert (
            resolve_file_url("https://api.music.apple.com/v4/catalog/us/songs/12345")
            is None
        )

    def test_empty_string_returns_none(self):
        assert resolve_file_url("") is None

    def test_non_file_url_returns_none(self):
        """http:// URLs → None."""
        assert resolve_file_url("http://example.com/track.wav") is None

    def test_apple_music_triple_slash_with_spaces(self):
        result = resolve_file_url("file:///Users/testuser/Music/My%20Library/track.wav")
        assert result == "/Users/testuser/Music/My Library/track.wav"

    def test_nested_path(self):
        url = "file:///Users/testuser/Music/sundown_groove.wav"
        assert resolve_file_url(url) == "/Users/testuser/Music/sundown_groove.wav"

    def test_localhost_path(self):
        url = "file://localhost/Users/testuser/Music/morning_mist.mp3"
        assert resolve_file_url(url) == "/Users/testuser/Music/morning_mist.mp3"


# ---------------------------------------------------------------------------
# check_file_available
# ---------------------------------------------------------------------------


class TestCheckFileAvailable:
    def test_existing_file_returns_true(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_path = f.name
        try:
            assert check_file_available(tmp_path) is True
        finally:
            os.unlink(tmp_path)

    def test_missing_file_returns_false(self):
        assert check_file_available("/nonexistent/path/track.wav") is False

    def test_directory_returns_false(self):
        with tempfile.TemporaryDirectory() as d:
            assert check_file_available(d) is False


# ---------------------------------------------------------------------------
# is_icloud_placeholder
# ---------------------------------------------------------------------------


class TestIsIcloudPlaceholder:
    def test_file_present_returns_false(self):
        """Real file exists → not a placeholder."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_path = f.name
        try:
            assert is_icloud_placeholder(tmp_path) is False
        finally:
            os.unlink(tmp_path)

    def test_missing_file_no_placeholder_returns_false(self):
        """Neither file nor placeholder → False."""
        assert is_icloud_placeholder("/nonexistent/no_placeholder.wav") is False

    def test_icloud_placeholder_detected(self):
        """File absent but .{name}.icloud exists → True."""
        with tempfile.TemporaryDirectory() as d:
            real_file = os.path.join(d, "track.wav")
            placeholder = os.path.join(d, ".track.wav.icloud")
            # Create only the placeholder marker
            open(placeholder, "w").close()
            assert is_icloud_placeholder(real_file) is True

    def test_both_real_and_placeholder_returns_false(self):
        """If real file exists, return False regardless of placeholder."""
        with tempfile.TemporaryDirectory() as d:
            real_file = os.path.join(d, "track.wav")
            placeholder = os.path.join(d, ".track.wav.icloud")
            open(real_file, "w").close()
            open(placeholder, "w").close()
            assert is_icloud_placeholder(real_file) is False
