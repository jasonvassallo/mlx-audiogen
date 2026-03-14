# Phase 9g-2: Music Library Scanner Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Library tab that parses Apple Music and rekordbox XML exports, enabling playlist-driven audio generation and collection-based LoRA training.

**Architecture:** New `mlx_audiogen/library/` module with parsers, collections, and description generation. FastAPI endpoints for library browsing/search/filter. React Library tab with sortable track table, metadata editor, and "Generate Like This" / "Train on These" workflows. Collections bridge library selections to the existing LoRA training pipeline.

**Tech Stack:** Python (plistlib, defusedxml, FastAPI/Pydantic), React 19, TypeScript, Zustand, Tailwind CSS v4

**Spec:** `docs/superpowers/specs/2026-03-14-library-scanner-design.md`

**Deviations from spec file layout (intentional simplifications):**
- Spec lists `playlists.py` as a separate file — plan folds playlist extraction into `parsers.py` (parsers naturally return playlists alongside tracks, a separate file would be a single re-export)
- Spec defines dataclasses inline — plan creates `models.py` to house them (cleaner separation)
- Plan adds `cache.py` (not in spec) — the spec describes caching behavior but didn't define a module for it
- Spec lists single `tests/test_library.py` — plan creates granular test files per module (better isolation)

---

## Chunk 1: Data Models + XML Parsers + Cloud Paths

### Task 1: Add defusedxml dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add defusedxml to dependencies**

In `pyproject.toml`, add `"defusedxml>=0.7"` to the `dependencies` list.

- [ ] **Step 2: Sync and verify**

Run: `uv sync`
Expected: installs defusedxml

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add defusedxml dependency for safe XML parsing"
```

---

### Task 2: Create TrackInfo and PlaylistInfo data models

**Files:**
- Create: `mlx_audiogen/library/__init__.py`
- Create: `mlx_audiogen/library/models.py`
- Test: `tests/test_library_models.py`

- [ ] **Step 1: Write failing tests for data models**

Create `tests/test_library_models.py`:
```python
"""Tests for library data models."""

from mlx_audiogen.library.models import (
    LibrarySource,
    PlaylistInfo,
    TrackInfo,
    normalize_rating,
    slugify_playlist_name,
)


def test_track_info_to_dict():
    track = TrackInfo(
        track_id="12292",
        title="Box Jam",
        artist="Beaumont",
        album="Unknown Album",
        genre="Deep House",
        bpm=122.0,
        key="4A",
        year=2019,
        rating=80,
        play_count=5,
        duration_seconds=283.0,
        comments="4A - 7",
        file_path="/Users/test/Music/box_jam.mp3",
        file_available=True,
        source="apple_music",
        loved=True,
        description="",
        description_edited=False,
    )
    d = track.to_dict()
    assert d["track_id"] == "12292"
    assert d["bpm"] == 122.0
    assert d["loved"] is True


def test_track_info_from_dict():
    d = {
        "track_id": "99",
        "title": "Test",
        "artist": "Artist",
        "album": "",
        "genre": "House",
        "bpm": 128.0,
        "key": "8A",
        "year": 2020,
        "rating": 60,
        "play_count": 0,
        "duration_seconds": 180.0,
        "comments": "",
        "file_path": None,
        "file_available": False,
        "source": "rekordbox",
        "loved": False,
        "description": "house track",
        "description_edited": False,
    }
    track = TrackInfo.from_dict(d)
    assert track.title == "Test"
    assert track.file_available is False


def test_normalize_rating_apple_music():
    assert normalize_rating(80, "apple_music") == 80
    assert normalize_rating(0, "apple_music") == 0
    assert normalize_rating(100, "apple_music") == 100


def test_normalize_rating_rekordbox():
    # rekordbox uses 0-255
    assert normalize_rating(0, "rekordbox") == 0
    assert normalize_rating(255, "rekordbox") == 100
    assert normalize_rating(51, "rekordbox") == 20  # 1 star
    assert normalize_rating(204, "rekordbox") == 80  # 4 stars


def test_normalize_rating_none():
    assert normalize_rating(None, "apple_music") is None


def test_slugify_playlist_name():
    assert slugify_playlist_name("DJ Vassallo") == "dj-vassallo"
    assert slugify_playlist_name("90's Music") == "90s-music"
    assert slugify_playlist_name("Shur-i-kan's April") == "shur-i-kans-april"
    assert slugify_playlist_name("  spaces  ") == "spaces"


def test_playlist_info_to_dict():
    pl = PlaylistInfo(
        id="dj-vassallo",
        name="DJ Vassallo",
        track_count=958,
        track_ids=["1", "2", "3"],
        source="apple_music",
    )
    d = pl.to_dict()
    assert d["id"] == "dj-vassallo"
    assert d["track_count"] == 958


def test_library_source_to_dict():
    src = LibrarySource(
        id="am1",
        type="apple_music",
        path="~/Music/Media/Library.xml",
        label="Apple Music",
        track_count=12104,
        playlist_count=161,
        last_loaded="2026-03-14T09:00:00Z",
    )
    d = src.to_dict()
    assert d["type"] == "apple_music"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_library_models.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Create library module with data models**

Create `mlx_audiogen/library/__init__.py`:
```python
"""Music library scanner: parse Apple Music + rekordbox XML exports."""
```

Create `mlx_audiogen/library/models.py`:
```python
"""Data models for music library tracks, playlists, and sources."""

import re
import unicodedata
from dataclasses import dataclass, field


def normalize_rating(rating: int | None, source: str) -> int | None:
    """Normalize rating to 0-100 scale.

    Apple Music: already 0-100 (0=unrated, 20/40/60/80/100 = 1-5 stars).
    rekordbox: 0-255 → round(rating * 100 / 255).
    """
    if rating is None:
        return None
    if source == "rekordbox":
        return round(rating * 100 / 255)
    return rating


def slugify_playlist_name(name: str) -> str:
    """Convert playlist name to a URL-safe slug.

    "DJ Vassallo" → "dj-vassallo"
    "90's Music" → "90s-music"
    """
    # Normalize unicode, strip accents
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    # Lowercase, replace non-alphanumeric with hyphens
    name = re.sub(r"[^a-z0-9]+", "-", name.lower())
    # Strip leading/trailing hyphens
    return name.strip("-")


@dataclass
class TrackInfo:
    """Unified track metadata from Apple Music or rekordbox."""

    track_id: str
    title: str
    artist: str
    album: str
    genre: str
    bpm: float | None
    key: str | None
    year: int | None
    rating: int | None  # Normalized 0-100
    play_count: int
    duration_seconds: float
    comments: str
    file_path: str | None
    file_available: bool
    source: str  # "apple_music" | "rekordbox"
    loved: bool
    description: str = ""
    description_edited: bool = False

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "genre": self.genre,
            "bpm": self.bpm,
            "key": self.key,
            "year": self.year,
            "rating": self.rating,
            "play_count": self.play_count,
            "duration_seconds": self.duration_seconds,
            "comments": self.comments,
            "file_path": self.file_path,
            "file_available": self.file_available,
            "source": self.source,
            "loved": self.loved,
            "description": self.description,
            "description_edited": self.description_edited,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrackInfo":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PlaylistInfo:
    """Playlist metadata with track references."""

    id: str  # URL-safe slug
    name: str  # Display name
    track_count: int
    track_ids: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "track_count": self.track_count,
            "track_ids": self.track_ids,
            "source": self.source,
        }


@dataclass
class LibrarySource:
    """Configuration for a connected music library."""

    id: str
    type: str  # "apple_music" | "rekordbox"
    path: str
    label: str
    track_count: int | None = None
    playlist_count: int | None = None
    last_loaded: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "path": self.path,
            "label": self.label,
            "track_count": self.track_count,
            "playlist_count": self.playlist_count,
            "last_loaded": self.last_loaded,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LibrarySource":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_library_models.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_audiogen/library/ tests/test_library_models.py
git commit -m "feat(library): data models for tracks, playlists, and sources"
```

---

### Task 3: Cloud path resolution

**Files:**
- Create: `mlx_audiogen/library/cloud_paths.py`
- Test: `tests/test_cloud_paths.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cloud_paths.py`:
```python
"""Tests for cloud path resolution."""

import os
from pathlib import Path
from unittest.mock import patch

from mlx_audiogen.library.cloud_paths import (
    check_file_available,
    is_icloud_placeholder,
    resolve_file_url,
)


def test_resolve_file_url_apple_music():
    url = "file:///Users/test/Music/Media/Music/Artist/Album/01%20Track.m4a"
    assert resolve_file_url(url) == "/Users/test/Music/Media/Music/Artist/Album/01 Track.m4a"


def test_resolve_file_url_rekordbox():
    url = "file://localhost/Users/test/Music/track%20name.wav"
    assert resolve_file_url(url) == "/Users/test/Music/track name.wav"


def test_resolve_file_url_none():
    assert resolve_file_url(None) is None
    assert resolve_file_url("") is None


def test_resolve_file_url_soundcloud():
    url = "file://localhostsoundcloud:tracks:663031988"
    assert resolve_file_url(url) is None  # Not a local file


def test_resolve_file_url_catalog():
    url = "/v4/catalog/tracks/15064522/"
    assert resolve_file_url(url) is None  # Streaming reference


def test_check_file_available_exists(tmp_path: Path):
    f = tmp_path / "test.wav"
    f.write_bytes(b"RIFF")
    assert check_file_available(str(f)) is True


def test_check_file_available_missing():
    assert check_file_available("/nonexistent/path/file.wav") is False


def test_check_file_available_none():
    assert check_file_available(None) is False


def test_check_file_available_icloud_placeholder(tmp_path: Path):
    """When the real file doesn't exist but an .icloud placeholder does."""
    placeholder = tmp_path / ".test.wav.icloud"
    placeholder.write_bytes(b"placeholder")
    real_path = str(tmp_path / "test.wav")
    # File not locally available
    assert check_file_available(real_path) is False
    # But iCloud placeholder exists
    assert is_icloud_placeholder(real_path) is True


def test_is_icloud_placeholder_no_placeholder(tmp_path: Path):
    """File doesn't exist and no placeholder either."""
    assert is_icloud_placeholder(str(tmp_path / "missing.wav")) is False


def test_is_icloud_placeholder_file_exists(tmp_path: Path):
    """File exists locally — not an iCloud placeholder."""
    f = tmp_path / "local.wav"
    f.write_bytes(b"RIFF")
    assert is_icloud_placeholder(str(f)) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cloud_paths.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement cloud_paths.py**

Create `mlx_audiogen/library/cloud_paths.py`:
```python
"""Resolve file:// URLs to local filesystem paths.

Handles Apple Music (file:///...) and rekordbox (file://localhost/...)
URL schemes. Detects iCloud placeholder files (.filename.icloud).
"""

import os
from urllib.parse import unquote


def resolve_file_url(url: str | None) -> str | None:
    """Convert a file:// URL to a local filesystem path.

    Returns None for streaming references (SoundCloud, catalog URLs).
    """
    if not url:
        return None

    # Skip streaming references
    if "soundcloud:" in url or url.startswith("/v4/catalog/"):
        return None

    # Apple Music: file:///path
    if url.startswith("file:///"):
        return unquote(url[7:])  # Strip "file://"

    # rekordbox: file://localhost/path
    if url.startswith("file://localhost/"):
        return unquote(url[16:])  # Strip "file://localhost"

    # Not a recognized local file URL
    if not url.startswith("/"):
        return None

    return unquote(url)


def check_file_available(path: str | None) -> bool:
    """Check if a file exists locally (not just in iCloud).

    macOS stores evicted iCloud files as hidden placeholders:
    .{filename}.icloud in the same directory. This function
    distinguishes "file exists locally" from "file is in iCloud
    but not cached" from "file doesn't exist at all."
    """
    if not path:
        return False
    if os.path.isfile(path):
        return True
    return False


def is_icloud_placeholder(path: str | None) -> bool:
    """Check if a file has an iCloud placeholder (evicted from local disk).

    Returns True if the file doesn't exist locally but has a
    .{filename}.icloud companion file, meaning it's in iCloud
    but not downloaded.
    """
    if not path:
        return False
    if os.path.isfile(path):
        return False  # File exists locally, not a placeholder
    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    placeholder = os.path.join(dirname, f".{basename}.icloud")
    return os.path.isfile(placeholder)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cloud_paths.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_audiogen/library/cloud_paths.py tests/test_cloud_paths.py
git commit -m "feat(library): cloud path resolution for file:// URLs"
```

---

### Task 4: Apple Music XML parser

**Files:**
- Create: `mlx_audiogen/library/parsers.py`
- Create: `tests/fixtures/apple_music_sample.xml`
- Test: `tests/test_parsers.py`

- [ ] **Step 1: Create Apple Music fixture XML**

Create `tests/fixtures/apple_music_sample.xml` — a minimal valid Apple Music plist with 3 tracks and 2 playlists. Include: BPM, Genre, Rating, Comments with Camelot key ("4A - 7"), Location with file:// URL, Loved/Favorited fields, one track without BPM/key.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Major Version</key><integer>1</integer>
    <key>Minor Version</key><integer>1</integer>
    <key>Music Folder</key><string>file:///Users/test/Music/Media/</string>
    <key>Tracks</key>
    <dict>
        <key>100</key>
        <dict>
            <key>Track ID</key><integer>100</integer>
            <key>Name</key><string>Deep Groove</string>
            <key>Artist</key><string>DJ Test</string>
            <key>Album</key><string>Test Album</string>
            <key>Genre</key><string>Deep House</string>
            <key>Total Time</key><integer>300000</integer>
            <key>Year</key><integer>2022</integer>
            <key>BPM</key><integer>122</integer>
            <key>Rating</key><integer>80</integer>
            <key>Play Count</key><integer>15</integer>
            <key>Comments</key><string>4A - 7</string>
            <key>Loved</key><true/>
            <key>Location</key><string>file:///Users/test/Music/Media/Music/DJ%20Test/Test%20Album/01%20Deep%20Groove.m4a</string>
        </dict>
        <key>101</key>
        <dict>
            <key>Track ID</key><integer>101</integer>
            <key>Name</key><string>Acid Techno</string>
            <key>Artist</key><string>Producer X</string>
            <key>Album</key><string></string>
            <key>Genre</key><string>Techno</string>
            <key>Total Time</key><integer>420000</integer>
            <key>Year</key><integer>2023</integer>
            <key>BPM</key><integer>135</integer>
            <key>Rating</key><integer>60</integer>
            <key>Play Count</key><integer>3</integer>
            <key>Comments</key><string>10B</string>
            <key>Favorited</key><true/>
            <key>Location</key><string>file:///Users/test/Music/Media/Music/Producer%20X/02%20Acid%20Techno.mp3</string>
        </dict>
        <key>102</key>
        <dict>
            <key>Track ID</key><integer>102</integer>
            <key>Name</key><string>Ambient Drift</string>
            <key>Artist</key><string>Chill Artist</string>
            <key>Album</key><string>Calm</string>
            <key>Genre</key><string>Ambient</string>
            <key>Total Time</key><integer>600000</integer>
            <key>Location</key><string>file:///Users/test/Music/Media/Music/Chill%20Artist/Calm/01%20Ambient%20Drift.aac</string>
        </dict>
    </dict>
    <key>Playlists</key>
    <array>
        <dict>
            <key>Name</key><string>Library</string>
            <key>Master</key><true/>
            <key>Playlist Items</key>
            <array>
                <dict><key>Track ID</key><integer>100</integer></dict>
                <dict><key>Track ID</key><integer>101</integer></dict>
                <dict><key>Track ID</key><integer>102</integer></dict>
            </array>
        </dict>
        <dict>
            <key>Name</key><string>DJ Vassallo</string>
            <key>Playlist Items</key>
            <array>
                <dict><key>Track ID</key><integer>100</integer></dict>
                <dict><key>Track ID</key><integer>101</integer></dict>
            </array>
        </dict>
        <dict>
            <key>Name</key><string>Empty Playlist</string>
            <key>Playlist Items</key>
            <array/>
        </dict>
    </array>
</dict>
</plist>
```

- [ ] **Step 2: Write failing tests for Apple Music parser**

Create `tests/test_parsers.py`:
```python
"""Tests for Apple Music and rekordbox XML parsers."""

from pathlib import Path

import pytest

from mlx_audiogen.library.parsers import (
    extract_camelot_key,
    parse_apple_music_xml,
    parse_rekordbox_xml,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestCamelotKeyExtraction:
    def test_key_with_dash_number(self):
        assert extract_camelot_key("4A - 7") == "4A"

    def test_key_alone(self):
        assert extract_camelot_key("10B") == "10B"

    def test_key_lowercase(self):
        assert extract_camelot_key("4a - 7") == "4A"

    def test_no_key(self):
        assert extract_camelot_key("some random comment") is None

    def test_empty(self):
        assert extract_camelot_key("") is None

    def test_none(self):
        assert extract_camelot_key(None) is None


class TestAppleMusicParser:
    def test_parse_tracks(self):
        tracks, playlists = parse_apple_music_xml(
            FIXTURES / "apple_music_sample.xml"
        )
        assert len(tracks) == 3

    def test_track_metadata(self):
        tracks, _ = parse_apple_music_xml(FIXTURES / "apple_music_sample.xml")
        t = tracks["100"]
        assert t.title == "Deep Groove"
        assert t.artist == "DJ Test"
        assert t.bpm == 122.0
        assert t.key == "4A"
        assert t.rating == 80
        assert t.genre == "Deep House"
        assert t.loved is True
        assert t.source == "apple_music"

    def test_track_camelot_from_comments(self):
        tracks, _ = parse_apple_music_xml(FIXTURES / "apple_music_sample.xml")
        assert tracks["100"].key == "4A"  # From "4A - 7"
        assert tracks["101"].key == "10B"  # From "10B"

    def test_track_missing_fields(self):
        tracks, _ = parse_apple_music_xml(FIXTURES / "apple_music_sample.xml")
        t = tracks["102"]
        assert t.bpm is None
        assert t.key is None
        assert t.year is None
        assert t.rating is None
        assert t.play_count == 0

    def test_favorited_maps_to_loved(self):
        tracks, _ = parse_apple_music_xml(FIXTURES / "apple_music_sample.xml")
        assert tracks["101"].loved is True  # Uses Favorited key

    def test_file_path_resolved(self):
        tracks, _ = parse_apple_music_xml(FIXTURES / "apple_music_sample.xml")
        assert tracks["100"].file_path == (
            "/Users/test/Music/Media/Music/DJ Test/Test Album/01 Deep Groove.m4a"
        )

    def test_playlists(self):
        _, playlists = parse_apple_music_xml(FIXTURES / "apple_music_sample.xml")
        assert len(playlists) >= 2
        dj_pl = next(p for p in playlists if p.name == "DJ Vassallo")
        assert dj_pl.track_count == 2
        assert "100" in dj_pl.track_ids

    def test_empty_playlist(self):
        _, playlists = parse_apple_music_xml(FIXTURES / "apple_music_sample.xml")
        empty = next(p for p in playlists if p.name == "Empty Playlist")
        assert empty.track_count == 0

    def test_duration_conversion(self):
        tracks, _ = parse_apple_music_xml(FIXTURES / "apple_music_sample.xml")
        assert tracks["100"].duration_seconds == 300.0  # 300000ms

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            parse_apple_music_xml(Path("/nonexistent/Library.xml"))

    def test_malformed_xml(self, tmp_path: Path):
        bad = tmp_path / "bad.xml"
        bad.write_text("this is not valid xml at all {{{")
        with pytest.raises(Exception):  # plistlib raises InvalidFileException
            parse_apple_music_xml(bad)


class TestRekordboxMalformedXml:
    def test_malformed_rekordbox_xml(self, tmp_path: Path):
        bad = tmp_path / "bad.xml"
        bad.write_text("<DJ_PLAYLISTS><broken>")
        with pytest.raises(Exception):  # defusedxml raises ParseError
            parse_rekordbox_xml(bad)
```

- [ ] **Step 3: Implement parsers.py (Apple Music portion)**

Create `mlx_audiogen/library/parsers.py`:
```python
"""Parse Apple Music and rekordbox XML library exports.

Apple Music: plistlib (stdlib, safe against XXE).
rekordbox: defusedxml (prevents XXE, billion laughs, external entities).
"""

import logging
import os
import plistlib
import re
from pathlib import Path

from .cloud_paths import check_file_available, resolve_file_url
from .models import PlaylistInfo, TrackInfo, normalize_rating, slugify_playlist_name

logger = logging.getLogger(__name__)

# Maximum XML file size (500MB)
MAX_XML_SIZE = 500 * 1024 * 1024

# Camelot key pattern: 1A-12B (with optional suffix like "- 7")
_CAMELOT_RE = re.compile(r"\b([1-9]|1[0-2])[AaBb]\b")


def extract_camelot_key(comments: str | None) -> str | None:
    """Extract Camelot notation key from a comments string.

    Handles patterns like "4A - 7", "10B", "4a".
    Returns uppercase (e.g., "4A") or None.
    """
    if not comments:
        return None
    match = _CAMELOT_RE.search(comments)
    if match:
        return match.group(0).upper()
    return None


def _check_file_size(path: Path) -> None:
    """Reject files larger than MAX_XML_SIZE."""
    size = path.stat().st_size
    if size > MAX_XML_SIZE:
        raise ValueError(
            f"XML file too large ({size / 1024 / 1024:.0f}MB, "
            f"max {MAX_XML_SIZE / 1024 / 1024:.0f}MB): {path}"
        )


def parse_apple_music_xml(
    path: Path,
) -> tuple[dict[str, TrackInfo], list[PlaylistInfo]]:
    """Parse an Apple Music Library.xml export.

    Returns:
        Tuple of (tracks_by_id, playlists).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Apple Music XML not found: {path}")
    _check_file_size(path)

    with open(path, "rb") as f:
        lib = plistlib.load(f)

    # Parse tracks
    tracks: dict[str, TrackInfo] = {}
    for track_id_str, entry in lib.get("Tracks", {}).items():
        tid = str(entry.get("Track ID", track_id_str))
        comments = entry.get("Comments", "")
        location = entry.get("Location")
        file_path = resolve_file_url(location)
        total_time_ms = entry.get("Total Time", 0)

        tracks[tid] = TrackInfo(
            track_id=tid,
            title=entry.get("Name", ""),
            artist=entry.get("Artist", ""),
            album=entry.get("Album", ""),
            genre=entry.get("Genre", ""),
            bpm=float(entry["BPM"]) if "BPM" in entry else None,
            key=extract_camelot_key(comments),
            year=entry.get("Year"),
            rating=normalize_rating(entry.get("Rating"), "apple_music"),
            play_count=entry.get("Play Count", 0),
            duration_seconds=total_time_ms / 1000.0 if total_time_ms else 0.0,
            comments=comments,
            file_path=file_path,
            file_available=check_file_available(file_path),
            source="apple_music",
            loved=entry.get("Loved", False) or entry.get("Favorited", False),
        )

    # Parse playlists
    playlists: list[PlaylistInfo] = []
    for pl in lib.get("Playlists", []):
        name = pl.get("Name", "")
        items = pl.get("Playlist Items", [])
        track_ids = [str(item.get("Track ID", "")) for item in items]
        playlists.append(
            PlaylistInfo(
                id=slugify_playlist_name(name),
                name=name,
                track_count=len(track_ids),
                track_ids=track_ids,
                source="apple_music",
            )
        )

    return tracks, playlists
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_parsers.py::TestAppleMusicParser -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_audiogen/library/parsers.py tests/test_parsers.py tests/fixtures/apple_music_sample.xml
git commit -m "feat(library): Apple Music XML parser with Camelot key extraction"
```

---

### Task 5: rekordbox XML parser

**Files:**
- Modify: `mlx_audiogen/library/parsers.py`
- Create: `tests/fixtures/rekordbox_sample.xml`
- Modify: `tests/test_parsers.py`

- [ ] **Step 1: Create rekordbox fixture XML**

Create `tests/fixtures/rekordbox_sample.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="rekordbox" Version="7.2.10" Company="AlphaTheta"/>
  <COLLECTION Entries="3">
    <TRACK TrackID="1001" Name="Warm Groove" Artist="House DJ"
           Album="Vibes" Genre="Deep House" Kind="WAV File"
           Size="52428800" TotalTime="360" DiscNumber="0"
           TrackNumber="1" Year="2021" AverageBpm="122.00"
           DateAdded="2021-06-15" BitRate="1411" SampleRate="44100"
           Comments="Great track" PlayCount="10" Rating="204"
           Location="file://localhost/Users/test/Music/warm%20groove.wav"
           Remixer="" Tonality="4A" Label="Deep Records" Mix="">
      <TEMPO Inizio="0.100" Bpm="122.00" Metro="4/4" Battito="1"/>
    </TRACK>
    <TRACK TrackID="1002" Name="Acid Dreams" Artist="Techno Producer"
           Album="" Genre="Techno" Kind="MP3 File"
           Size="10485760" TotalTime="480" DiscNumber="0"
           TrackNumber="0" Year="2023" AverageBpm="138.50"
           DateAdded="2023-01-20" BitRate="320" SampleRate="44100"
           Comments="" PlayCount="0" Rating="0"
           Location="file://localhost/Users/test/Music/acid%20dreams.mp3"
           Remixer="" Tonality="10B" Label="" Mix=""/>
    <TRACK TrackID="1003" Name="Cloud Track" Artist="Streamer"
           Album="" Genre="House" Kind="Unknown Format"
           Size="0" TotalTime="300" DiscNumber="0"
           TrackNumber="0" Year="0" AverageBpm="125.00"
           DateAdded="2020-01-01" BitRate="128" SampleRate="44100"
           Comments="" PlayCount="5" Rating="102"
           Location="file://localhostsoundcloud:tracks:123456"
           Remixer="" Tonality="8A" Label="" Mix=""/>
  </COLLECTION>
  <PLAYLISTS>
    <NODE Type="0" Name="ROOT" Count="2">
      <NODE Name="My Playlist" Type="1" KeyType="0" Entries="2">
        <TRACK Key="1001"/>
        <TRACK Key="1002"/>
      </NODE>
      <NODE Name="Empty" Type="1" KeyType="0" Entries="0"/>
    </NODE>
  </PLAYLISTS>
</DJ_PLAYLISTS>
```

- [ ] **Step 2: Write failing tests for rekordbox parser**

Add to `tests/test_parsers.py`:
```python
class TestRekordboxParser:
    def test_parse_tracks(self):
        tracks, playlists = parse_rekordbox_xml(FIXTURES / "rekordbox_sample.xml")
        assert len(tracks) == 3

    def test_track_metadata(self):
        tracks, _ = parse_rekordbox_xml(FIXTURES / "rekordbox_sample.xml")
        t = tracks["1001"]
        assert t.title == "Warm Groove"
        assert t.artist == "House DJ"
        assert t.bpm == 122.0
        assert t.key == "4A"
        assert t.rating == 80  # 204 → normalized
        assert t.genre == "Deep House"
        assert t.source == "rekordbox"

    def test_rating_normalization(self):
        tracks, _ = parse_rekordbox_xml(FIXTURES / "rekordbox_sample.xml")
        assert tracks["1001"].rating == 80  # 204/255 * 100
        assert tracks["1002"].rating == 0
        assert tracks["1003"].rating == 40  # 102/255 * 100

    def test_soundcloud_track_no_file(self):
        tracks, _ = parse_rekordbox_xml(FIXTURES / "rekordbox_sample.xml")
        t = tracks["1003"]
        assert t.file_path is None
        assert t.file_available is False

    def test_file_path_resolved(self):
        tracks, _ = parse_rekordbox_xml(FIXTURES / "rekordbox_sample.xml")
        assert tracks["1001"].file_path == "/Users/test/Music/warm groove.wav"

    def test_playlists(self):
        _, playlists = parse_rekordbox_xml(FIXTURES / "rekordbox_sample.xml")
        assert len(playlists) >= 1
        my_pl = next(p for p in playlists if p.name == "My Playlist")
        assert my_pl.track_count == 2
        assert "1001" in my_pl.track_ids

    def test_empty_playlist(self):
        _, playlists = parse_rekordbox_xml(FIXTURES / "rekordbox_sample.xml")
        empty = next(p for p in playlists if p.name == "Empty")
        assert empty.track_count == 0

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            parse_rekordbox_xml(Path("/nonexistent/rekordbox.xml"))
```

- [ ] **Step 3: Implement rekordbox parser**

Add to `mlx_audiogen/library/parsers.py`:
```python
def parse_rekordbox_xml(
    path: Path,
) -> tuple[dict[str, TrackInfo], list[PlaylistInfo]]:
    """Parse a rekordbox XML library export.

    Uses defusedxml to prevent XXE attacks.

    Returns:
        Tuple of (tracks_by_id, playlists).
    """
    import defusedxml.ElementTree as ET

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"rekordbox XML not found: {path}")
    _check_file_size(path)

    tree = ET.parse(str(path))
    root = tree.getroot()

    # Parse tracks from COLLECTION
    tracks: dict[str, TrackInfo] = {}
    collection = root.find("COLLECTION")
    if collection is not None:
        for track_el in collection.findall("TRACK"):
            tid = track_el.get("TrackID", "")
            if not tid:
                continue

            bpm_str = track_el.get("AverageBpm", "")
            bpm = float(bpm_str) if bpm_str else None

            rating_str = track_el.get("Rating", "0")
            raw_rating = int(rating_str) if rating_str else 0

            total_time_str = track_el.get("TotalTime", "0")
            total_time = int(total_time_str) if total_time_str else 0

            year_str = track_el.get("Year", "0")
            year = int(year_str) if year_str and year_str != "0" else None

            play_count_str = track_el.get("PlayCount", "0")
            play_count = int(play_count_str) if play_count_str else 0

            location = track_el.get("Location", "")
            file_path = resolve_file_url(location)

            tracks[tid] = TrackInfo(
                track_id=tid,
                title=track_el.get("Name", ""),
                artist=track_el.get("Artist", ""),
                album=track_el.get("Album", ""),
                genre=track_el.get("Genre", ""),
                bpm=bpm,
                key=track_el.get("Tonality") or None,
                year=year,
                rating=normalize_rating(raw_rating, "rekordbox"),
                play_count=play_count,
                duration_seconds=float(total_time),
                comments=track_el.get("Comments", ""),
                file_path=file_path,
                file_available=check_file_available(file_path),
                source="rekordbox",
                loved=False,  # rekordbox doesn't have a "loved" concept
            )

    # Parse playlists from PLAYLISTS
    playlists: list[PlaylistInfo] = []
    playlists_el = root.find("PLAYLISTS")
    if playlists_el is not None:
        _collect_playlists(playlists_el, playlists)

    return tracks, playlists


def _collect_playlists(
    node: "defusedxml.ElementTree.Element",  # type: ignore[name-defined]
    result: list[PlaylistInfo],
) -> None:
    """Recursively collect playlist nodes (Type=1), skip folders (Type=0)."""
    for child in node.findall("NODE"):
        node_type = child.get("Type", "0")
        if node_type == "1":
            # Playlist node
            name = child.get("Name", "")
            track_els = child.findall("TRACK")
            track_ids = [t.get("Key", "") for t in track_els if t.get("Key")]
            result.append(
                PlaylistInfo(
                    id=slugify_playlist_name(name),
                    name=name,
                    track_count=len(track_ids),
                    track_ids=track_ids,
                    source="rekordbox",
                )
            )
        elif node_type == "0":
            # Folder node — recurse
            _collect_playlists(child, result)
```

- [ ] **Step 4: Run all parser tests**

Run: `uv run pytest tests/test_parsers.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_audiogen/library/parsers.py tests/test_parsers.py tests/fixtures/rekordbox_sample.xml
git commit -m "feat(library): rekordbox XML parser with defusedxml"
```

---

### Task 6: Description generation

**Files:**
- Create: `mlx_audiogen/library/description_gen.py`
- Test: `tests/test_description_gen.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_description_gen.py`:
```python
"""Tests for metadata-to-description generation."""

from mlx_audiogen.library.description_gen import generate_description
from mlx_audiogen.library.models import TrackInfo


def _make_track(**kwargs) -> TrackInfo:
    defaults = dict(
        track_id="1",
        title="Test",
        artist="Artist",
        album="Album",
        genre="House",
        bpm=128.0,
        key="4A",
        year=2022,
        rating=80,
        play_count=0,
        duration_seconds=300.0,
        comments="",
        file_path=None,
        file_available=False,
        source="apple_music",
        loved=False,
        description="",
        description_edited=False,
    )
    defaults.update(kwargs)
    return TrackInfo(**defaults)


def test_full_metadata():
    track = _make_track(genre="Deep House", bpm=122.0, key="4A", artist="Jimpster")
    desc = generate_description(track)
    assert "deep house" in desc.lower()
    assert "122" in desc
    assert "4A" in desc


def test_missing_bpm():
    track = _make_track(genre="Ambient", bpm=None, key="8A")
    desc = generate_description(track)
    assert "ambient" in desc.lower()
    assert "8A" in desc
    assert "BPM" not in desc


def test_missing_key():
    track = _make_track(genre="Techno", bpm=135.0, key=None)
    desc = generate_description(track)
    assert "techno" in desc.lower()
    assert "135" in desc


def test_missing_genre():
    track = _make_track(genre="", bpm=120.0, key="4A")
    desc = generate_description(track)
    assert "120" in desc


def test_all_missing():
    track = _make_track(genre="", bpm=None, key=None, artist="")
    desc = generate_description(track)
    assert isinstance(desc, str)
    assert len(desc) > 0  # Should still produce something
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_description_gen.py -v`
Expected: FAIL

- [ ] **Step 3: Implement description_gen.py**

Create `mlx_audiogen/library/description_gen.py`:
```python
"""Generate text descriptions from track metadata.

Template mode: instant, deterministic.
LLM mode: deferred to server (requires mlx-lm runtime).
"""


def generate_description(track: "TrackInfo") -> str:
    """Generate a template-based description from track metadata.

    Builds a comma-separated phrase from available fields:
    genre, BPM, key, artist style. Skips missing fields.
    """
    from .models import TrackInfo

    parts: list[str] = []

    if track.genre:
        parts.append(track.genre.lower())

    if track.bpm is not None:
        parts.append(f"{track.bpm:.0f} BPM")

    if track.key:
        parts.append(track.key)

    if track.artist:
        parts.append(f"{track.artist} style")

    if not parts:
        # Fallback: use title
        return track.title.lower() if track.title else "instrumental track"

    return ", ".join(parts)


def generate_playlist_prompt(
    tracks: list["TrackInfo"],
) -> dict:
    """Analyze a set of tracks and generate a prompt capturing their vibe.

    Returns analysis dict with prompt and metadata breakdown.
    """
    from collections import Counter
    from statistics import median

    from .models import TrackInfo

    bpms = [t.bpm for t in tracks if t.bpm is not None]
    keys = [t.key for t in tracks if t.key]
    genres = [t.genre for t in tracks if t.genre]
    artists = [t.artist for t in tracks if t.artist]
    years = [t.year for t in tracks if t.year]

    analysis = {
        "bpm_median": round(median(bpms), 1) if bpms else None,
        "bpm_range": [min(bpms), max(bpms)] if bpms else None,
        "top_keys": [k for k, _ in Counter(keys).most_common(3)],
        "top_genres": [g for g, _ in Counter(genres).most_common(3)],
        "top_artists": [a for a, _ in Counter(artists).most_common(5)],
        "year_range": [min(years), max(years)] if years else None,
        "track_count": len(tracks),
        "available_count": sum(1 for t in tracks if t.file_available),
    }

    # Build template prompt
    parts: list[str] = []
    if analysis["top_genres"]:
        parts.append(", ".join(analysis["top_genres"][:2]).lower())
    if analysis["bpm_median"]:
        parts.append(f"{analysis['bpm_median']:.0f} BPM")
    if analysis["top_keys"]:
        parts.append(analysis["top_keys"][0])
    if analysis["top_artists"]:
        parts.append(f"influenced by {' and '.join(analysis['top_artists'][:2])}")

    analysis["prompt"] = ", ".join(parts) if parts else "instrumental music"

    return analysis
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_description_gen.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_audiogen/library/description_gen.py tests/test_description_gen.py
git commit -m "feat(library): template-based description generation from metadata"
```

---

### Task 7: Collections CRUD + training bridge

**Files:**
- Create: `mlx_audiogen/library/collections.py`
- Test: `tests/test_collections.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_collections.py`:
```python
"""Tests for collection CRUD and training bridge."""

import json
from pathlib import Path

import pytest

from mlx_audiogen.library.collections import (
    collection_to_training_data,
    create_collection,
    delete_collection,
    get_collection,
    list_collections,
    update_collection,
)


@pytest.fixture
def collections_dir(tmp_path: Path) -> Path:
    d = tmp_path / "collections"
    d.mkdir()
    return d


def test_create_collection(collections_dir: Path):
    data = {
        "name": "test-collection",
        "source": "rekordbox",
        "playlist": "My Playlist",
        "tracks": [
            {
                "track_id": "1",
                "title": "Track 1",
                "artist": "Artist",
                "description": "house, 128 BPM",
                "file_path": "/tmp/track1.wav",
                "file_available": True,
            }
        ],
    }
    result = create_collection(data, collections_dir)
    assert result["name"] == "test-collection"
    assert (collections_dir / "test-collection.json").exists()


def test_list_collections(collections_dir: Path):
    create_collection(
        {"name": "col-a", "tracks": []}, collections_dir
    )
    create_collection(
        {"name": "col-b", "tracks": [{"track_id": "1"}]}, collections_dir
    )
    result = list_collections(collections_dir)
    assert len(result) == 2
    names = [c["name"] for c in result]
    assert "col-a" in names
    assert "col-b" in names


def test_get_collection(collections_dir: Path):
    create_collection(
        {"name": "my-col", "tracks": [{"track_id": "1", "title": "T"}]},
        collections_dir,
    )
    result = get_collection("my-col", collections_dir)
    assert result["name"] == "my-col"
    assert len(result["tracks"]) == 1


def test_get_collection_not_found(collections_dir: Path):
    with pytest.raises(FileNotFoundError):
        get_collection("nonexistent", collections_dir)


def test_update_collection(collections_dir: Path):
    create_collection({"name": "upd", "tracks": []}, collections_dir)
    update_collection(
        "upd",
        {"tracks": [{"track_id": "99", "description": "updated"}]},
        collections_dir,
    )
    result = get_collection("upd", collections_dir)
    assert len(result["tracks"]) == 1
    assert result["tracks"][0]["track_id"] == "99"


def test_delete_collection(collections_dir: Path):
    create_collection({"name": "del-me", "tracks": []}, collections_dir)
    delete_collection("del-me", collections_dir)
    assert not (collections_dir / "del-me.json").exists()


def test_delete_collection_not_found(collections_dir: Path):
    with pytest.raises(FileNotFoundError):
        delete_collection("nope", collections_dir)


def test_invalid_collection_name(collections_dir: Path):
    with pytest.raises(ValueError):
        create_collection({"name": "../bad", "tracks": []}, collections_dir)


def test_collection_to_training_data(collections_dir: Path, tmp_path: Path):
    # Create a real audio file
    audio_file = tmp_path / "track.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

    create_collection(
        {
            "name": "train-col",
            "tracks": [
                {
                    "track_id": "1",
                    "title": "Track",
                    "description": "house groove",
                    "file_path": str(audio_file),
                    "file_available": True,
                },
                {
                    "track_id": "2",
                    "title": "Missing",
                    "description": "missing file",
                    "file_path": "/nonexistent.wav",
                    "file_available": False,
                },
            ],
        },
        collections_dir,
    )
    entries = collection_to_training_data("train-col", collections_dir)
    # Only the available track should be returned
    assert len(entries) == 1
    assert entries[0]["file"] == str(audio_file)
    assert entries[0]["text"] == "house groove"


def test_collection_to_training_data_no_available(collections_dir: Path):
    create_collection(
        {
            "name": "empty-train",
            "tracks": [
                {
                    "track_id": "1",
                    "file_path": "/nonexistent.wav",
                    "file_available": False,
                }
            ],
        },
        collections_dir,
    )
    with pytest.raises(ValueError, match="No audio files available"):
        collection_to_training_data("empty-train", collections_dir)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_collections.py -v`
Expected: FAIL

- [ ] **Step 3: Implement collections.py**

Create `mlx_audiogen/library/collections.py`:
```python
"""Collection CRUD: save/load/update training data selections.

Collections are JSON files at ~/.mlx-audiogen/collections/{name}.json.
They bridge library selections to the LoRA training pipeline.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_COLLECTIONS_DIR = Path.home() / ".mlx-audiogen" / "collections"
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def _validate_name(name: str) -> None:
    """Validate collection name (alphanumeric, hyphens, underscores)."""
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid collection name: {name!r}. "
            "Use 1-64 chars: letters, numbers, hyphens, underscores."
        )
    if ".." in name:
        raise ValueError("Collection name must not contain '..'")


def create_collection(
    data: dict,
    collections_dir: Path = DEFAULT_COLLECTIONS_DIR,
) -> dict:
    """Create a new collection JSON file."""
    name = data.get("name", "")
    _validate_name(name)

    collections_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    data.setdefault("created_at", now)
    data.setdefault("updated_at", now)
    data.setdefault("tracks", [])

    path = collections_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return data


def list_collections(
    collections_dir: Path = DEFAULT_COLLECTIONS_DIR,
) -> list[dict]:
    """List all saved collections (name, track count, timestamps)."""
    if not collections_dir.is_dir():
        return []

    result = []
    for p in sorted(collections_dir.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            result.append(
                {
                    "name": data.get("name", p.stem),
                    "track_count": len(data.get("tracks", [])),
                    "source": data.get("source", ""),
                    "playlist": data.get("playlist", ""),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                }
            )
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping invalid collection: %s", p)
    return result


def get_collection(
    name: str,
    collections_dir: Path = DEFAULT_COLLECTIONS_DIR,
) -> dict:
    """Load a collection by name."""
    _validate_name(name)
    path = collections_dir / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Collection not found: {name}")
    with open(path) as f:
        return json.load(f)


def update_collection(
    name: str,
    updates: dict,
    collections_dir: Path = DEFAULT_COLLECTIONS_DIR,
) -> dict:
    """Update fields in an existing collection."""
    data = get_collection(name, collections_dir)
    data.update(updates)
    data["updated_at"] = datetime.now(timezone.utc).isoformat()

    path = collections_dir / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def delete_collection(
    name: str,
    collections_dir: Path = DEFAULT_COLLECTIONS_DIR,
) -> None:
    """Delete a collection."""
    _validate_name(name)
    path = collections_dir / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Collection not found: {name}")
    path.unlink()


def collection_to_training_data(
    name: str,
    collections_dir: Path = DEFAULT_COLLECTIONS_DIR,
) -> list[dict[str, str]]:
    """Convert a collection to training data entries.

    Returns list of {"file": "/abs/path", "text": "description"}
    matching the format used by lora/dataset.py scan_dataset().
    Skips tracks where file_available is False or file doesn't exist.

    Raises:
        ValueError: If no tracks have available audio files.
    """
    data = get_collection(name, collections_dir)
    entries = []
    for track in data.get("tracks", []):
        file_path = track.get("file_path")
        if not file_path or not track.get("file_available", False):
            continue
        if not os.path.isfile(file_path):
            logger.warning("File no longer exists: %s", file_path)
            continue
        entries.append(
            {
                "file": file_path,
                "text": track.get("description", track.get("title", "")),
            }
        )

    if not entries:
        raise ValueError(
            "No audio files available for training in this collection. "
            "Download tracks locally to enable training."
        )
    return entries
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_collections.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add mlx_audiogen/library/collections.py tests/test_collections.py
git commit -m "feat(library): collection CRUD + training pipeline bridge"
```

---

### Task 8: Library cache manager

**Files:**
- Create: `mlx_audiogen/library/cache.py`
- Test: `tests/test_library_cache.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_library_cache.py`:
```python
"""Tests for library cache manager."""

import json
from pathlib import Path

import pytest

from mlx_audiogen.library.cache import LibraryCache


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    d = tmp_path / "config"
    d.mkdir()
    return d


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def test_add_source(config_dir: Path, fixtures_dir: Path):
    cache = LibraryCache(config_dir=config_dir)
    src = cache.add_source(
        type="apple_music",
        path=str(fixtures_dir / "apple_music_sample.xml"),
        label="Test AM",
    )
    assert src.id
    assert src.type == "apple_music"
    assert src.label == "Test AM"


def test_scan_apple_music(config_dir: Path, fixtures_dir: Path):
    cache = LibraryCache(config_dir=config_dir)
    src = cache.add_source(
        type="apple_music",
        path=str(fixtures_dir / "apple_music_sample.xml"),
        label="AM",
    )
    cache.scan(src.id)
    assert cache.get_track_count(src.id) == 3
    playlists = cache.get_playlists(src.id)
    assert len(playlists) >= 2


def test_scan_rekordbox(config_dir: Path, fixtures_dir: Path):
    cache = LibraryCache(config_dir=config_dir)
    src = cache.add_source(
        type="rekordbox",
        path=str(fixtures_dir / "rekordbox_sample.xml"),
        label="RB",
    )
    cache.scan(src.id)
    assert cache.get_track_count(src.id) == 3


def test_search_tracks(config_dir: Path, fixtures_dir: Path):
    cache = LibraryCache(config_dir=config_dir)
    src = cache.add_source(
        type="apple_music",
        path=str(fixtures_dir / "apple_music_sample.xml"),
        label="AM",
    )
    cache.scan(src.id)
    results = cache.search_tracks(src.id, q="Deep")
    assert len(results) >= 1
    assert results[0].genre == "Deep House"


def test_search_by_bpm_range(config_dir: Path, fixtures_dir: Path):
    cache = LibraryCache(config_dir=config_dir)
    src = cache.add_source(
        type="apple_music",
        path=str(fixtures_dir / "apple_music_sample.xml"),
        label="AM",
    )
    cache.scan(src.id)
    results = cache.search_tracks(src.id, bpm_min=130, bpm_max=140)
    assert len(results) == 1
    assert results[0].title == "Acid Techno"


def test_sort_tracks(config_dir: Path, fixtures_dir: Path):
    cache = LibraryCache(config_dir=config_dir)
    src = cache.add_source(
        type="apple_music",
        path=str(fixtures_dir / "apple_music_sample.xml"),
        label="AM",
    )
    cache.scan(src.id)
    results = cache.search_tracks(src.id, sort="bpm", order="desc")
    # Filter to tracks with BPM
    with_bpm = [r for r in results if r.bpm is not None]
    if len(with_bpm) >= 2:
        assert with_bpm[0].bpm >= with_bpm[1].bpm


def test_list_sources(config_dir: Path, fixtures_dir: Path):
    cache = LibraryCache(config_dir=config_dir)
    cache.add_source(
        type="apple_music",
        path=str(fixtures_dir / "apple_music_sample.xml"),
        label="AM",
    )
    sources = cache.list_sources()
    assert len(sources) == 1
    assert sources[0].label == "AM"


def test_remove_source(config_dir: Path, fixtures_dir: Path):
    cache = LibraryCache(config_dir=config_dir)
    src = cache.add_source(
        type="rekordbox",
        path=str(fixtures_dir / "rekordbox_sample.xml"),
        label="RB",
    )
    cache.remove_source(src.id)
    assert len(cache.list_sources()) == 0


def test_persist_sources(config_dir: Path, fixtures_dir: Path):
    cache1 = LibraryCache(config_dir=config_dir)
    cache1.add_source(
        type="apple_music",
        path=str(fixtures_dir / "apple_music_sample.xml"),
        label="AM",
    )
    # Create new instance — should load from disk
    cache2 = LibraryCache(config_dir=config_dir)
    assert len(cache2.list_sources()) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_library_cache.py -v`
Expected: FAIL

- [ ] **Step 3: Implement cache.py**

Create `mlx_audiogen/library/cache.py`:
```python
"""In-memory library cache with search, sort, and filter.

Parses XML once on scan(), holds tracks and playlists in memory.
Source config persisted to ~/.mlx-audiogen/library_sources.json.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .models import LibrarySource, PlaylistInfo, TrackInfo
from .parsers import parse_apple_music_xml, parse_rekordbox_xml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = Path.home() / ".mlx-audiogen"


class LibraryCache:
    """Manages parsed library data with search and filter."""

    def __init__(self, config_dir: Path = DEFAULT_CONFIG_DIR) -> None:
        self._config_dir = config_dir
        self._config_path = config_dir / "library_sources.json"
        self._sources: dict[str, LibrarySource] = {}
        self._tracks: dict[str, dict[str, TrackInfo]] = {}  # source_id -> {track_id: TrackInfo}
        self._playlists: dict[str, list[PlaylistInfo]] = {}  # source_id -> [PlaylistInfo]
        self._load_config()

    def _load_config(self) -> None:
        """Load source config from disk."""
        if self._config_path.is_file():
            with open(self._config_path) as f:
                data = json.load(f)
            for entry in data:
                src = LibrarySource.from_dict(entry)
                self._sources[src.id] = src

    def _save_config(self) -> None:
        """Persist source config to disk."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in self._sources.values()]
        with open(self._config_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_source(self, type: str, path: str, label: str) -> LibrarySource:
        """Add a new library source."""
        src = LibrarySource(
            id=str(uuid.uuid4())[:8],
            type=type,
            path=path,
            label=label,
        )
        self._sources[src.id] = src
        self._save_config()
        return src

    def update_source(self, source_id: str, path: str | None = None, label: str | None = None) -> LibrarySource:
        """Update an existing source."""
        src = self._sources.get(source_id)
        if not src:
            raise KeyError(f"Source not found: {source_id}")
        if path is not None:
            src.path = path
        if label is not None:
            src.label = label
        self._save_config()
        return src

    def remove_source(self, source_id: str) -> None:
        """Remove a library source and its cached data."""
        self._sources.pop(source_id, None)
        self._tracks.pop(source_id, None)
        self._playlists.pop(source_id, None)
        self._save_config()

    def list_sources(self) -> list[LibrarySource]:
        """List all configured sources."""
        return list(self._sources.values())

    def scan(self, source_id: str) -> LibrarySource:
        """Parse/refresh a library source."""
        src = self._sources.get(source_id)
        if not src:
            raise KeyError(f"Source not found: {source_id}")

        path = Path(src.path).expanduser()
        if src.type == "apple_music":
            tracks, playlists = parse_apple_music_xml(path)
        elif src.type == "rekordbox":
            tracks, playlists = parse_rekordbox_xml(path)
        else:
            raise ValueError(f"Unknown source type: {src.type}")

        self._tracks[source_id] = tracks
        self._playlists[source_id] = playlists

        src.track_count = len(tracks)
        src.playlist_count = len(playlists)
        src.last_loaded = datetime.now(timezone.utc).isoformat()
        self._save_config()

        return src

    def get_track_count(self, source_id: str) -> int:
        """Get number of cached tracks for a source."""
        return len(self._tracks.get(source_id, {}))

    def get_playlists(self, source_id: str) -> list[PlaylistInfo]:
        """Get playlists for a source."""
        return self._playlists.get(source_id, [])

    def get_playlist_tracks(
        self, source_id: str, playlist_id: str
    ) -> list[TrackInfo]:
        """Get tracks in a specific playlist."""
        playlists = self._playlists.get(source_id, [])
        tracks = self._tracks.get(source_id, {})
        for pl in playlists:
            if pl.id == playlist_id:
                return [tracks[tid] for tid in pl.track_ids if tid in tracks]
        raise KeyError(f"Playlist not found: {playlist_id}")

    def search_tracks(
        self,
        source_id: str,
        *,
        q: str | None = None,
        artist: str | None = None,
        album: str | None = None,
        genre: str | None = None,
        key: str | None = None,
        bpm_min: float | None = None,
        bpm_max: float | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
        rating_min: int | None = None,
        loved: bool | None = None,
        available: bool | None = None,
        sort: str | None = None,
        order: str = "asc",
        offset: int = 0,
        limit: int = 50,
    ) -> list[TrackInfo]:
        """Search, filter, sort, and paginate tracks."""
        all_tracks = list(self._tracks.get(source_id, {}).values())

        # Filter
        results = []
        for t in all_tracks:
            if q:
                q_lower = q.lower()
                if not any(
                    q_lower in (getattr(t, f, "") or "").lower()
                    for f in ("title", "artist", "album", "comments")
                ):
                    continue
            if artist and artist.lower() not in (t.artist or "").lower():
                continue
            if album and album.lower() not in (t.album or "").lower():
                continue
            if genre and genre.lower() not in (t.genre or "").lower():
                continue
            if key and t.key != key:
                continue
            if bpm_min is not None and (t.bpm is None or t.bpm < bpm_min):
                continue
            if bpm_max is not None and (t.bpm is None or t.bpm > bpm_max):
                continue
            if year_min is not None and (t.year is None or t.year < year_min):
                continue
            if year_max is not None and (t.year is None or t.year > year_max):
                continue
            if rating_min is not None and (t.rating is None or t.rating < rating_min):
                continue
            if loved is not None and t.loved != loved:
                continue
            if available is not None and t.file_available != available:
                continue
            results.append(t)

        # Sort
        if sort:
            reverse = order == "desc"
            results.sort(
                key=lambda t: (getattr(t, sort, None) is None, getattr(t, sort, "")),
                reverse=reverse,
            )

        # Paginate
        return results[offset : offset + limit]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_library_cache.py -v`
Expected: all PASS

- [ ] **Step 5: Run full test suite + linting**

Run: `uv run ruff format . && uv run ruff check . && uv run mypy mlx_audiogen/library/ && uv run pytest tests/test_library_models.py tests/test_cloud_paths.py tests/test_parsers.py tests/test_description_gen.py tests/test_collections.py tests/test_library_cache.py -v`
Expected: all PASS, clean

- [ ] **Step 6: Commit**

```bash
git add mlx_audiogen/library/cache.py tests/test_library_cache.py
git commit -m "feat(library): in-memory cache with search, sort, filter, pagination"
```

---

## Chunk 2: Server API Endpoints

### Task 9: Library source + browsing endpoints

**Files:**
- Modify: `mlx_audiogen/server/app.py`

- [ ] **Step 1: Add library source and browsing endpoints to app.py**

Add a new section after the LoRA endpoints section (after line ~935). Use FastAPI's lazy imports pattern (same as existing LoRA endpoints). Add Pydantic models for request bodies.

Key endpoints to add:
- `GET /api/library/sources` — returns `library_cache.list_sources()`
- `POST /api/library/sources` — body: `{type, path, label}`, calls `library_cache.add_source()`
- `PUT /api/library/sources/{id}` — body: `{path?, label?}`, calls `library_cache.update_source()`
- `DELETE /api/library/sources/{id}` — calls `library_cache.remove_source()`
- `POST /api/library/scan/{id}` — calls `library_cache.scan()`
- `GET /api/library/playlists/{id}` — returns playlist list
- `GET /api/library/tracks/{id}` — with all query params from spec, returns paginated track list
- `GET /api/library/playlist-tracks/{source_id}/{playlist_id}` — returns tracks in playlist

The `LibraryCache` instance is created as a module-level singleton (like `_pipeline_cache`).

- [ ] **Step 2: Verify server starts**

Run: `uv run python -c "from mlx_audiogen.server.app import app; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add mlx_audiogen/server/app.py
git commit -m "feat(server): library source + browsing API endpoints"
```

---

### Task 10: Collection endpoints

**Files:**
- Modify: `mlx_audiogen/server/app.py`

- [ ] **Step 1: Add collection CRUD endpoints**

Add after library browsing endpoints:
- `GET /api/collections` — `list_collections()`
- `GET /api/collections/{name}` — `get_collection()`
- `POST /api/collections` — `create_collection()`
- `PUT /api/collections/{name}` — `update_collection()`
- `DELETE /api/collections/{name}` — `delete_collection()`
- `GET /api/collections/{name}/export` — returns JSON file download
- `POST /api/collections/import` — accepts JSON upload

- [ ] **Step 2: Add `collection` field to TrainRequest**

Add `collection: Optional[str] = Field(default=None, ...)` to `TrainRequest`. Update `start_training()` to use `collection_to_training_data()` when `collection` is provided (instead of `scan_dataset(data_dir)`). Make `data_dir` optional when `collection` is provided.

- [ ] **Step 3: Commit**

```bash
git add mlx_audiogen/server/app.py
git commit -m "feat(server): collection CRUD + training bridge endpoints"
```

---

### Task 11: AI endpoints (describe, suggest-name, generate-prompt)

**Files:**
- Modify: `mlx_audiogen/server/app.py`

- [ ] **Step 1: Add AI endpoints**

- `POST /api/library/describe` — body: `{source_id, track_ids, mode}`. In template mode: call `generate_description()` for each track. In LLM mode: use existing `enhance_with_llm()` pattern.
- `POST /api/library/suggest-name` — body: `{source_id, track_ids}`. Analyze top genres/artists → suggest slug like "deep-house-jimpster-style".
- `POST /api/library/generate-prompt` — body: `{source_id, track_ids, mode}`. Calls `generate_playlist_prompt()`.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest -v && uv run ruff check . && uv run mypy mlx_audiogen/`
Expected: all pass

- [ ] **Step 3: Commit**

```bash
git add mlx_audiogen/server/app.py
git commit -m "feat(server): library AI endpoints (describe, suggest-name, generate-prompt)"
```

---

### Task 12: Server API endpoint tests

**Files:**
- Create: `tests/test_library_api.py`

- [ ] **Step 1: Write API endpoint tests using FastAPI TestClient**

Create `tests/test_library_api.py` with tests covering:
- `POST /api/library/sources` — add a source, verify response has id
- `GET /api/library/sources` — list sources after adding
- `POST /api/library/scan/{id}` — scan the fixture XML, verify track/playlist counts
- `GET /api/library/playlists/{id}` — verify playlist list
- `GET /api/library/tracks/{id}` — verify track list with pagination
- `GET /api/library/tracks/{id}?q=Deep` — search filter
- `GET /api/library/tracks/{id}?bpm_min=130&bpm_max=140` — BPM range filter
- `GET /api/library/tracks/{id}?sort=bpm&order=desc` — sort
- `GET /api/library/playlist-tracks/{source_id}/{playlist_id}` — playlist tracks
- `POST /api/collections` — create collection
- `GET /api/collections` — list collections
- `GET /api/collections/{name}` — get collection
- `PUT /api/collections/{name}` — update collection
- `DELETE /api/collections/{name}` — delete collection
- `GET /api/collections/{name}/export` — export as JSON download
- `POST /api/collections/import` — import from JSON upload
- `POST /api/library/describe` — template mode description generation
- `POST /api/library/suggest-name` — adapter name suggestion
- `POST /api/library/generate-prompt` — playlist prompt generation

Use `tmp_path` fixture for collections dir. Point library sources at the test fixture XMLs.

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_library_api.py -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_library_api.py
git commit -m "test: API endpoint tests for library + collection routes"
```

---

### Task 13: Tag schema expansion

**Files:**
- Modify: `mlx_audiogen/shared/prompt_suggestions.py`
- Modify: `web/src/components/TagAutocomplete.tsx`
- Modify: `tests/test_phase7b.py` (or create `tests/test_tag_schema.py`)

- [ ] **Step 1: Write a test for 14-category TAG_DATABASE**

Add to an existing or new test file:
```python
def test_tag_database_has_14_categories():
    from mlx_audiogen.shared.prompt_suggestions import TAG_DATABASE
    assert len(TAG_DATABASE) == 14
    expected = {"genre", "sub_genre", "mood", "instrument", "vocal", "key",
                "bpm", "era", "production", "artist", "label", "structure",
                "rating", "availability"}
    assert set(TAG_DATABASE.keys()) == expected
    # Original categories should still have entries
    assert len(TAG_DATABASE["genre"]) > 0
    assert len(TAG_DATABASE["mood"]) > 0
```

- [ ] **Step 2: Expand TAG_DATABASE from 5 to 14 categories**

In `prompt_suggestions.py`, add 9 new empty categories to `TAG_DATABASE`:
```python
TAG_DATABASE: dict[str, list[str]] = {
    "genre": list(GENRES),
    "sub_genre": [],       # populated by library analysis
    "mood": list(MOODS),
    "instrument": [...],
    "vocal": [],           # populated by library analysis
    "key": [],             # populated by library analysis
    "bpm": [],             # populated by library analysis
    "era": list(ERAS),
    "production": list(PRODUCTION),
    "artist": [],          # populated by library analysis
    "label": [],           # populated by library analysis
    "structure": [],       # populated by library analysis
    "rating": [],          # not used as tags, displayed as stars
    "availability": [],    # not used as tags, displayed as dots
}
```

- [ ] **Step 2: Update CATEGORY_COLORS in TagAutocomplete.tsx**

Expand from 5 to 14 entries with the CSS hex colors from the spec.

- [ ] **Step 3: Run tests + npm build**

Run: `uv run pytest -v && cd web && npm run build`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add mlx_audiogen/shared/prompt_suggestions.py web/src/components/TagAutocomplete.tsx
git commit -m "feat: expand tag schema from 5 to 14 categories with color mapping"
```

---

## Chunk 3: Frontend — Types, API Client, Store

### Task 14: TypeScript types for library + collections

**Files:**
- Modify: `web/src/types/api.ts`

- [ ] **Step 1: Add library and collection types**

Add to `web/src/types/api.ts`:
```typescript
// ---------------------------------------------------------------------------
// Phase 9g-2: Library Scanner
// ---------------------------------------------------------------------------

export interface LibraryTrackInfo {
  track_id: string;
  title: string;
  artist: string;
  album: string;
  genre: string;
  bpm: number | null;
  key: string | null;
  year: number | null;
  rating: number | null;
  play_count: number;
  duration_seconds: number;
  comments: string;
  file_path: string | null;
  file_available: boolean;
  source: string;
  loved: boolean;
  description: string;
  description_edited: boolean;
}

export interface PlaylistInfo {
  id: string;
  name: string;
  track_count: number;
  track_ids: string[];
  source: string;
}

export interface LibrarySource {
  id: string;
  type: "apple_music" | "rekordbox";
  path: string;
  label: string;
  track_count: number | null;
  playlist_count: number | null;
  last_loaded: string | null;
}

export interface CollectionSummary {
  name: string;
  track_count: number;
  source: string;
  playlist: string;
  created_at: string | null;
  updated_at: string | null;
}

export interface CollectionFull {
  name: string;
  created_at: string;
  updated_at: string;
  source: string;
  playlist: string;
  tracks: LibraryTrackInfo[];
}

export interface PlaylistAnalysis {
  bpm_median: number | null;
  bpm_range: [number, number] | null;
  top_keys: string[];
  top_genres: string[];
  top_artists: string[];
  year_range: [number, number] | null;
  track_count: number;
  available_count: number;
  prompt: string;
}
```

- [ ] **Step 2: Commit**

```bash
git add web/src/types/api.ts
git commit -m "feat(web): TypeScript types for library scanner + collections"
```

---

### Task 15: API client functions

**Files:**
- Modify: `web/src/api/client.ts`

- [ ] **Step 1: Add library + collection API wrappers**

Add functions following existing patterns (using `getServerUrl()` + fetch):
- `fetchLibrarySources()`, `addLibrarySource()`, `updateLibrarySource()`, `deleteLibrarySource()`
- `scanLibrarySource()`, `fetchPlaylists()`, `fetchPlaylistTracks()`
- `searchLibraryTracks()` — with all query params
- `describeLibraryTracks()`, `suggestAdapterName()`, `generatePlaylistPrompt()`
- `fetchCollections()`, `getCollection()`, `createCollection()`, `updateCollection()`, `deleteCollection()`
- `exportCollection()`, `importCollection()`

- [ ] **Step 2: Build to verify**

Run: `cd web && npm run build`
Expected: success

- [ ] **Step 3: Commit**

```bash
git add web/src/api/client.ts
git commit -m "feat(web): API client wrappers for library + collections"
```

---

### Task 16: Zustand store — library state

**Files:**
- Modify: `web/src/store/useStore.ts`

- [ ] **Step 1: Add library state to the store**

Add new state slices:
```typescript
// --- Library ---
librarySources: LibrarySource[];
librarySourcesLoading: boolean;
activeSourceId: string | null;
playlists: PlaylistInfo[];
playlistsLoading: boolean;
activePlaylistId: string | null;
libraryTracks: LibraryTrackInfo[];
libraryTracksLoading: boolean;
libraryTracksTotal: number;
selectedTrackIds: Set<string>;
librarySearchParams: { q?: string; sort?: string; order?: string; offset?: number; /* ... */ };
// Actions
loadLibrarySources: () => Promise<void>;
addSource: (type: string, path: string, label: string) => Promise<void>;
removeSource: (id: string) => Promise<void>;
scanSource: (id: string) => Promise<void>;
loadPlaylists: (sourceId: string) => Promise<void>;
loadPlaylistTracks: (sourceId: string, playlistId: string) => Promise<void>;
searchTracks: (params: SearchParams) => Promise<void>;
toggleTrackSelection: (trackId: string) => void;
selectAllTracks: () => void;
deselectAllTracks: () => void;
// Collections
collections: CollectionSummary[];
loadCollections: () => Promise<void>;
```

- [ ] **Step 2: Build to verify**

Run: `cd web && npm run build`
Expected: success

- [ ] **Step 3: Commit**

```bash
git add web/src/store/useStore.ts
git commit -m "feat(web): Zustand store slices for library + collections"
```

---

## Chunk 4: Frontend — Library Tab UI

### Task 17: LibraryPanel component

**Files:**
- Create: `web/src/components/LibraryPanel.tsx`
- Modify: `web/src/components/App.tsx`
- Modify: `web/src/components/TabBar.tsx` (add Library tab)

- [ ] **Step 1: Create LibraryPanel**

Build the component with:
- Source selector dropdown + "Add Library" form (type picker, path input, label)
- Refresh button
- Playlist browser (scrollable list with track counts, "All Tracks" at top)
- Track table with sortable column headers (click to sort), search bar, filter dropdowns
- Checkboxes with select-all in header
- "Generate Like This" and "Train on These" action buttons at bottom
- Green/gray availability dots
- Star ratings (rating / 20 = 1-5 stars)

- [ ] **Step 2: Add Library tab to App.tsx**

Add "Library" to the tab list in `TabBar`. When Library tab is active, show `LibraryPanel` in the sidebar and track table in the main content area (replacing HistoryPanel).

- [ ] **Step 3: Build to verify**

Run: `cd web && npm run build`
Expected: success

- [ ] **Step 4: Commit**

```bash
git add web/src/components/LibraryPanel.tsx web/src/components/App.tsx
git commit -m "feat(web): Library tab with source selector, playlist browser, track table"
```

---

### Task 18: MetadataEditor component

**Files:**
- Create: `web/src/components/MetadataEditor.tsx`

- [ ] **Step 1: Create MetadataEditor**

Modal/inline component that appears when "Train on These" is clicked:
- Table of selected tracks with editable description textarea per row
- AI-suggested adapter name (editable text input) at top
- Profile picker (quick/balanced/deep cards, reuse from TrainPanel)
- "Save Collection & Train" button → calls createCollection + startTraining
- "Save Collection" only button
- Dismiss/close button

- [ ] **Step 2: Build to verify**

Run: `cd web && npm run build`
Expected: success

- [ ] **Step 3: Commit**

```bash
git add web/src/components/MetadataEditor.tsx
git commit -m "feat(web): MetadataEditor for collection curation before training"
```

---

### Task 19: Update TrainRequest to support collection field

**Files:**
- Modify: `web/src/types/api.ts`
- Modify: `web/src/api/client.ts`

- [ ] **Step 1: Add collection field to TrainRequest type**

```typescript
export interface TrainRequest {
  data_dir?: string;      // Now optional (one of data_dir or collection required)
  collection?: string;    // Collection name
  base_model: string;
  name: string;
  // ... rest unchanged
}
```

Update `startTraining()` in `client.ts` to send collection field.

- [ ] **Step 2: Commit**

```bash
git add web/src/types/api.ts web/src/api/client.ts
git commit -m "feat(web): support collection field in training requests"
```

---

### Task 20: TrainPanel collection source + "Generate Like This" flow

**Files:**
- Modify: `web/src/components/TrainPanel.tsx`
- Modify: `web/src/components/LibraryPanel.tsx`

- [ ] **Step 1: Add Collection source to TrainPanel**

Add "Source" dropdown at top of TrainPanel: "Folder" (existing) | "Collection". When "Collection" is selected, show a dropdown of saved collections. Selected collection shows inline track list.

- [ ] **Step 2: Implement "Generate Like This" flow**

When clicked in LibraryPanel:
1. Call `generatePlaylistPrompt()` with selected track IDs
2. Show preview card (reuse EnhancePreview pattern): prompt text, analysis tags (colored), Use & Generate / Edit / Regenerate buttons
3. "Use & Generate": set prompt in store, switch to Generate tab, trigger generation. If available audio files exist in selection AND a style/reference model is loaded, include `style_audio_path` (MusicGen) or `reference_audio_path` (Stable Audio) from the highest-rated available track. User can change reference track via a dropdown in the preview card
4. "Edit": set prompt in store, switch to Generate tab (no auto-generate)
5. Audio conditioning options grayed out with tooltip "Download tracks locally to enable audio conditioning" when no tracks have `file_available: true`

- [ ] **Step 3: Build + test full app**

Run: `cd web && npm run build`
Expected: success

- [ ] **Step 4: Commit**

```bash
git add web/src/components/TrainPanel.tsx web/src/components/LibraryPanel.tsx
git commit -m "feat(web): collection source in Train tab + Generate Like This flow"
```

---

## Chunk 5: Final Integration + Quality

---

### Task 21: Full test suite + linting + docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run full Python quality suite**

Run: `uv run ruff format . && uv run ruff check . && uv run mypy mlx_audiogen/ && uv run bandit -r mlx_audiogen/ -c pyproject.toml && uv run pytest -v`
Expected: all pass. Fix any issues.

- [ ] **Step 2: Run full web build**

Run: `cd web && npm run build`
Expected: success

- [ ] **Step 3: Update CLAUDE.md**

Add library module to Architecture section, add new API endpoints table, document collections, tag schema, Library tab.

- [ ] **Step 4: Commit and push**

```bash
git add -A
git commit -m "docs: update CLAUDE.md with Phase 9g-2 library scanner documentation"
git push origin main
```

---

### Task 22: Integration test with real XML files

**Files:**
- Create: `tests/test_library_integration.py`

- [ ] **Step 1: Write integration tests**

```python
"""Integration tests for library scanner with real XML files.

These require the user's actual library exports on disk.
Run with: pytest -m integration -v
"""

import pytest
from pathlib import Path

from mlx_audiogen.library.cache import LibraryCache

APPLE_MUSIC_XML = Path.home() / "Music" / "Media" / "Library.xml"
REKORDBOX_XML = Path.home() / "Documents" / "rekordbox" / "rekordbox.xml"


@pytest.mark.integration
@pytest.mark.skipif(not APPLE_MUSIC_XML.exists(), reason="Apple Music XML not found")
def test_parse_real_apple_music(tmp_path: Path):
    cache = LibraryCache(config_dir=tmp_path)
    src = cache.add_source("apple_music", str(APPLE_MUSIC_XML), "Apple Music")
    cache.scan(src.id)
    assert cache.get_track_count(src.id) > 100
    playlists = cache.get_playlists(src.id)
    assert len(playlists) > 5


@pytest.mark.integration
@pytest.mark.skipif(not REKORDBOX_XML.exists(), reason="rekordbox XML not found")
def test_parse_real_rekordbox(tmp_path: Path):
    cache = LibraryCache(config_dir=tmp_path)
    src = cache.add_source("rekordbox", str(REKORDBOX_XML), "rekordbox")
    cache.scan(src.id)
    assert cache.get_track_count(src.id) > 100
    playlists = cache.get_playlists(src.id)
    assert len(playlists) > 5
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/test_library_integration.py -m integration -v`
Expected: PASS (if XML files exist)

- [ ] **Step 3: Commit and push**

```bash
git add tests/test_library_integration.py
git commit -m "test: integration tests for library scanner with real XML files"
git push origin main
```
