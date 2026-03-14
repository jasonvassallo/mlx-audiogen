"""Utilities for resolving cloud / local file paths from music library XML exports.

Apple Music and rekordbox embed file references as ``file://`` URLs that may
point to local files, iCloud Drive placeholders, or streaming-only catalogue
entries.  This module normalises those references to plain filesystem paths and
checks whether the underlying audio file is actually available.
"""

import os
from typing import Optional
from urllib.parse import unquote


def resolve_file_url(url: str) -> Optional[str]:
    """Convert a ``file://`` URL to a local filesystem path.

    Returns ``None`` for:
    - SoundCloud stream references (``soundcloud:`` anywhere in the URL)
    - Apple Music catalogue items (``/v4/catalog/`` in the URL)
    - Empty or non-file URLs

    URL-decodes percent-encoded characters (e.g. ``%20`` → space).

    Examples::

        resolve_file_url("file:///Volumes/Music/track.wav")
        # → "/Volumes/Music/track.wav"

        resolve_file_url("file://localhost/Users/me/track.mp3")
        # → "/Users/me/track.mp3"

        resolve_file_url("soundcloud:tracks:123456")
        # → None

        resolve_file_url("file://localhostsoundcloud:tracks:123")
        # → None  (rekordbox embeds SoundCloud URLs this way)
    """
    if not url:
        return None

    # Reject SoundCloud stream references (rekordbox embeds them as
    # "file://localhostsoundcloud:tracks:123" or bare "soundcloud:...")
    if "soundcloud:" in url:
        return None

    # Reject Apple Music catalogue references
    if "/v4/catalog/" in url:
        return None

    if url.startswith("file://localhost/"):
        # "file://localhost/path" — strip "file://localhost" (16 chars), leaving "/path"
        path = url[16:]
    elif url.startswith("file:///"):
        # "file:///path" — strip "file://" (7 chars), leaving "/path"
        path = url[7:]
    elif url.startswith("file://"):
        # Generic "file://host/path" — not expected in practice, but handle gracefully
        # Strip up to the third slash
        rest = url[7:]
        slash = rest.find("/")
        if slash == -1:
            return None
        path = rest[slash:]
    else:
        return None

    return unquote(path)


def check_file_available(path: str) -> bool:
    """Return True only if *path* points to a regular file that exists on disk.

    iCloud Drive placeholders, missing files, and directories all return False.
    """
    return os.path.isfile(path)


def is_icloud_placeholder(path: str) -> bool:
    """Return True if *path* is stored in iCloud but not yet downloaded locally.

    When iCloud Drive evicts a file, macOS replaces it with a hidden
    ``.<filename>.icloud`` marker file in the same directory.  This function
    checks for that marker when the real file is absent.

    Returns False if the real file exists (it is fully available locally).
    """
    if os.path.isfile(path):
        return False  # file is present locally — not a placeholder

    dirname = os.path.dirname(path)
    basename = os.path.basename(path)
    placeholder = os.path.join(dirname, f".{basename}.icloud")
    return os.path.exists(placeholder)
