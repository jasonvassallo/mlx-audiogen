"""Data models for music library tracks, playlists, and library sources."""

import re
from dataclasses import dataclass, field
from typing import Optional


def normalize_rating(rating: int, source: str) -> int:
    """Normalize a track rating to the 0-100 scale.

    Apple Music ratings are already on a 0-100 scale (in increments of 20).
    Rekordbox ratings use 0-255 and are scaled proportionally.
    """
    if source == "rekordbox":
        return round(rating * 100 / 255)
    # apple_music: already 0-100
    return int(rating)


def slugify_playlist_name(name: str) -> str:
    """Convert a playlist name to a URL-safe slug.

    Examples:
        "DJ Vassallo"   -> "dj-vassallo"
        "90's Music"    -> "90s-music"
        "Top 40!!!"     -> "top-40"
        "  spaces  "    -> "spaces"
    """
    # Lowercase
    slug = name.lower()
    # Remove apostrophes and similar characters that should be dropped (not replaced)
    slug = re.sub(r"['\u2018\u2019`]", "", slug)
    # Replace any run of non-alphanumeric characters with a single hyphen
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    # Strip leading/trailing hyphens
    slug = slug.strip("-")
    return slug


@dataclass
class TrackInfo:
    """Represents a single track from a music library."""

    track_id: str
    title: str
    artist: str
    album: str
    genre: str
    bpm: Optional[float]
    key: Optional[str]  # Camelot notation, e.g. "4A"
    year: Optional[int]
    rating: int  # normalized 0-100
    play_count: int
    duration_seconds: float
    comments: str
    file_path: Optional[str]
    file_available: bool
    source: str  # "apple_music" | "rekordbox"
    loved: bool
    description: str
    description_edited: bool

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
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
        """Deserialize from a dict, ignoring unknown keys."""
        known = {
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
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PlaylistInfo:
    """Represents a playlist from a music library."""

    id: str  # URL-safe slug derived from name
    name: str
    track_count: int
    track_ids: list[str] = field(default_factory=list)
    source: str = ""  # "apple_music" | "rekordbox"

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": self.id,
            "name": self.name,
            "track_count": self.track_count,
            "track_ids": self.track_ids,
            "source": self.source,
        }


@dataclass
class LibrarySource:
    """Represents a parsed music library source (a single XML file)."""

    id: str
    type: str  # "apple_music" | "rekordbox"
    path: str
    label: str
    track_count: int
    playlist_count: int
    last_loaded: Optional[str]  # ISO-8601 datetime string or None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
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
        """Deserialize from a dict, ignoring unknown keys."""
        known = {
            "id",
            "type",
            "path",
            "label",
            "track_count",
            "playlist_count",
            "last_loaded",
        }
        return cls(**{k: v for k, v in d.items() if k in known})
