"""Parsers for Apple Music and rekordbox XML library exports.

Both parsers return the same shape:
    (tracks: dict[str, TrackInfo], playlists: list[PlaylistInfo])

where the dict key is the track's ``track_id`` string.
"""

import os
import plistlib
import re
from typing import Any, Optional

import defusedxml.ElementTree as ET

from .cloud_paths import check_file_available, resolve_file_url
from .models import PlaylistInfo, TrackInfo, normalize_rating, slugify_playlist_name

# Files larger than 500 MB are rejected as implausible audio tracks.
_MAX_FILE_BYTES = 500 * 1024 * 1024

# Camelot wheel notation pattern: 1A-12A, 1B-12B (case-insensitive)
_CAMELOT_RE = re.compile(r"\b(1[0-2]|[1-9])[AaBb]\b")


def extract_camelot_key(comments: str) -> Optional[str]:
    """Extract the first Camelot key token from a comments string.

    Returns the matched token in its original case (e.g. ``"4A"``) or
    ``None`` if no Camelot key is found.

    Examples::

        extract_camelot_key("4A - 128 BPM")   # → "4A"
        extract_camelot_key("Energetic 11B")   # → "11B"
        extract_camelot_key("no key here")     # → None
    """
    m = _CAMELOT_RE.search(comments)
    if m:
        return m.group(0)
    return None


def _check_file_size(path: str) -> bool:
    """Return True if the file exists and is ≤ 500 MB; False otherwise."""
    try:
        return os.path.getsize(path) <= _MAX_FILE_BYTES
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Apple Music XML (iTunes Library XML / Music Library.xml)
# ---------------------------------------------------------------------------


def parse_apple_music_xml(
    path: str,
) -> tuple[dict[str, TrackInfo], list[PlaylistInfo]]:
    """Parse an Apple Music / iTunes XML library export.

    Uses :mod:`plistlib` (Python stdlib) which is inherently safe — it does not
    parse XML entity references.

    Args:
        path: Filesystem path to the ``.xml`` plist file.

    Returns:
        A tuple of ``(tracks_dict, playlists_list)``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be parsed as a valid plist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Apple Music XML not found: {path}")

    try:
        with open(path, "rb") as fh:
            lib = plistlib.load(fh)
    except Exception as exc:
        raise ValueError(f"Cannot parse Apple Music XML at {path}: {exc}") from exc

    tracks: dict[str, TrackInfo] = {}
    raw_tracks = lib.get("Tracks", {})

    for tid_str, raw in raw_tracks.items():
        track_id = str(tid_str)
        title = raw.get("Name") or ""
        artist = raw.get("Artist") or ""
        album = raw.get("Album") or ""
        genre = raw.get("Genre") or ""
        comments = raw.get("Comments") or ""
        bpm_raw = raw.get("BPM")
        bpm: Optional[float] = float(bpm_raw) if bpm_raw is not None else None
        key: Optional[str] = extract_camelot_key(comments)
        year_raw = raw.get("Year")
        year: Optional[int] = int(year_raw) if year_raw is not None else None
        rating_raw = raw.get("Rating", 0)
        rating = normalize_rating(int(rating_raw), "apple_music")
        play_count = int(raw.get("Play Count") or 0)
        # Total Time is in milliseconds
        total_time_ms = raw.get("Total Time", 0) or 0
        duration_seconds = total_time_ms / 1000.0
        # loved: either "Loved" flag (newer) or "Favorited" flag (older)
        loved = bool(raw.get("Loved", False)) or bool(raw.get("Favorited", False))

        location = raw.get("Location") or ""
        file_path: Optional[str] = resolve_file_url(location)
        if file_path is not None:
            file_available = check_file_available(file_path) and _check_file_size(
                file_path
            )
        else:
            file_available = False

        tracks[track_id] = TrackInfo(
            track_id=track_id,
            title=title,
            artist=artist,
            album=album,
            genre=genre,
            bpm=bpm,
            key=key,
            year=year,
            rating=rating,
            play_count=play_count,
            duration_seconds=duration_seconds,
            comments=comments,
            file_path=file_path,
            file_available=file_available,
            source="apple_music",
            loved=loved,
            description="",
            description_edited=False,
        )

    playlists: list[PlaylistInfo] = []
    raw_playlists = lib.get("Playlists", [])

    for raw_pl in raw_playlists:
        name = raw_pl.get("Name") or ""
        pl_id = slugify_playlist_name(name)
        raw_items = raw_pl.get("Playlist Items") or []
        track_ids = [str(item["Track ID"]) for item in raw_items if "Track ID" in item]
        playlists.append(
            PlaylistInfo(
                id=pl_id,
                name=name,
                track_count=len(track_ids),
                track_ids=track_ids,
                source="apple_music",
            )
        )

    return tracks, playlists


# ---------------------------------------------------------------------------
# rekordbox XML
# ---------------------------------------------------------------------------


def parse_rekordbox_xml(
    path: str,
) -> tuple[dict[str, TrackInfo], list[PlaylistInfo]]:
    """Parse a rekordbox XML library export.

    Uses :mod:`defusedxml` to prevent XXE attacks.

    Args:
        path: Filesystem path to the rekordbox ``.xml`` file.

    Returns:
        A tuple of ``(tracks_dict, playlists_list)``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file cannot be parsed as valid XML.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"rekordbox XML not found: {path}")

    try:
        tree = ET.parse(path)
    except Exception as exc:
        raise ValueError(f"Cannot parse rekordbox XML at {path}: {exc}") from exc

    root = tree.getroot()

    tracks: dict[str, TrackInfo] = {}
    collection = root.find("COLLECTION")
    if collection is not None:
        for track_el in collection.findall("TRACK"):
            track_id = track_el.get("TrackID", "")
            title = track_el.get("Name") or ""
            artist = track_el.get("Artist") or ""
            album = track_el.get("Album") or ""
            genre = track_el.get("Genre") or ""
            comments = track_el.get("Comments") or ""
            bpm_raw = track_el.get("AverageBpm")
            bpm: Optional[float] = float(bpm_raw) if bpm_raw else None
            key_raw = track_el.get("Tonality") or ""
            key: Optional[str] = key_raw if key_raw else None
            year_raw = track_el.get("Year")
            year: Optional[int] = int(year_raw) if year_raw else None
            # Rating in rekordbox is 0-255
            rating_raw = track_el.get("Rating", "0")
            rating = normalize_rating(int(rating_raw or 0), "rekordbox")
            play_count = int(track_el.get("PlayCount") or 0)
            # TotalTime is in seconds (not ms)
            total_time_raw = track_el.get("TotalTime")
            duration_seconds = float(total_time_raw) if total_time_raw else 0.0

            location = track_el.get("Location") or ""
            file_path: Optional[str] = resolve_file_url(location)
            if file_path is not None:
                file_available = check_file_available(file_path) and _check_file_size(
                    file_path
                )
            else:
                file_available = False

            tracks[track_id] = TrackInfo(
                track_id=track_id,
                title=title,
                artist=artist,
                album=album,
                genre=genre,
                bpm=bpm,
                key=key,
                year=year,
                rating=rating,
                play_count=play_count,
                duration_seconds=duration_seconds,
                comments=comments,
                file_path=file_path,
                file_available=file_available,
                source="rekordbox",
                loved=False,
                description="",
                description_edited=False,
            )

    playlists: list[PlaylistInfo] = []
    playlists_root = root.find("PLAYLISTS")
    if playlists_root is not None:
        _collect_rekordbox_playlists(playlists_root, playlists)

    return tracks, playlists


def _collect_rekordbox_playlists(
    node: Any,
    out: list[PlaylistInfo],
) -> None:
    """Recursively walk rekordbox PLAYLISTS/NODE tree.

    - Type="0" → folder → recurse into children
    - Type="1" → playlist → collect TRACK children
    """
    for child in node.findall("NODE"):
        node_type = child.get("Type", "0")
        name = child.get("Name") or ""
        if node_type == "1":
            # Leaf playlist node
            track_ids = [
                t.get("Key", "") for t in child.findall("TRACK") if t.get("Key")
            ]
            pl_id = slugify_playlist_name(name)
            out.append(
                PlaylistInfo(
                    id=pl_id,
                    name=name,
                    track_count=len(track_ids),
                    track_ids=track_ids,
                    source="rekordbox",
                )
            )
        else:
            # Folder — recurse
            _collect_rekordbox_playlists(child, out)
