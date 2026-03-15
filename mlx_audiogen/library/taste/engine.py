"""TasteEngine — high-level API for building and maintaining a taste profile.

The engine loads an existing profile from disk (or creates a fresh one),
provides methods to update it from library and generation signals, and
persists changes after every update.

Usage::

    engine = TasteEngine()
    engine.update_library_signals(tracks)
    engine.update_generation_signals(style_profile, history)
    profile = engine.get_profile()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..models import TrackInfo
from .profile import TasteProfile, WeightedTag
from .signals import (
    collect_flywheel_signals,
    collect_generation_signals,
    collect_library_signals,
)


class TasteEngine:
    """High-level taste profile manager.

    Loads or creates a :class:`TasteProfile`, updates it from library
    and generation signals, and saves to disk after each change.
    """

    def __init__(self, profile_path: Optional[str] = None) -> None:
        self._path = profile_path
        self._profile = TasteProfile.load(profile_path)

    def get_profile(self) -> TasteProfile:
        """Return the current taste profile."""
        return self._profile

    def update_library_signals(
        self,
        tracks: list[TrackInfo],
        enrichment_tags: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Recompute library signals from tracks and update the profile.

        Args:
            tracks: Full list of library tracks.
            enrichment_tags: Optional mapping of track_id -> extra tag strings.
        """
        signals = collect_library_signals(tracks, enrichment_tags)
        p = self._profile

        p.top_genres = [WeightedTag.from_dict(t) for t in signals["top_genres"]]
        p.top_artists = [WeightedTag.from_dict(t) for t in signals["top_artists"]]
        p.bpm_range = signals["bpm_range"]
        p.key_preferences = [
            WeightedTag.from_dict(t) for t in signals["key_preferences"]
        ]
        p.era_distribution = signals["era_distribution"]
        p.mood_profile = [WeightedTag.from_dict(t) for t in signals["mood_profile"]]
        p.style_tags = [WeightedTag.from_dict(t) for t in signals["style_tags"]]
        p.library_track_count = len(tracks)

        p.save(self._path)

    def update_generation_signals(
        self,
        style_profile: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> None:
        """Recompute generation signals and update the profile.

        Args:
            style_profile: Dict with ``genres``, ``moods``, ``instruments`` lists.
            history: List of generation history entries.
        """
        signals = collect_generation_signals(style_profile, history)
        p = self._profile

        p.gen_genres = [WeightedTag.from_dict(t) for t in signals["gen_genres"]]
        p.gen_moods = [WeightedTag.from_dict(t) for t in signals["gen_moods"]]
        p.gen_instruments = [
            WeightedTag.from_dict(t) for t in signals["gen_instruments"]
        ]
        p.kept_ratio = signals["kept_ratio"]
        p.avg_duration = signals["avg_duration"]
        p.preferred_models = [
            WeightedTag.from_dict(t) for t in signals["preferred_models"]
        ]
        p.generation_count = len(history)

        p.save(self._path)

    def update_flywheel_signals(
        self,
        kept_dir: Path,
    ) -> None:
        """Update the profile with signals from starred generations.

        Flywheel signals are weighted at 1.5x to reflect explicit user
        approval (they starred these generations).

        Args:
            kept_dir: Directory containing kept generation metadata files.
        """
        signals = collect_flywheel_signals(kept_dir)
        p = self._profile

        # Merge flywheel genres into gen_genres with 1.5x weight boost
        flywheel_genres = signals.get("flywheel_genres", [])
        for tag_dict in flywheel_genres:
            tag_dict["weight"] = round(tag_dict.get("weight", 0) * 1.5, 4)
        p.gen_genres = [
            WeightedTag.from_dict(t)
            for t in flywheel_genres + [t.to_dict() for t in p.gen_genres]
        ]

        flywheel_moods = signals.get("flywheel_moods", [])
        for tag_dict in flywheel_moods:
            tag_dict["weight"] = round(tag_dict.get("weight", 0) * 1.5, 4)
        p.gen_moods = [
            WeightedTag.from_dict(t)
            for t in flywheel_moods + [t.to_dict() for t in p.gen_moods]
        ]

        flywheel_instruments = signals.get("flywheel_instruments", [])
        for tag_dict in flywheel_instruments:
            tag_dict["weight"] = round(tag_dict.get("weight", 0) * 1.5, 4)
        p.gen_instruments = [
            WeightedTag.from_dict(t)
            for t in flywheel_instruments + [t.to_dict() for t in p.gen_instruments]
        ]

        p.save(self._path)

    def set_overrides(self, text: str) -> None:
        """Store user override text in the profile.

        Args:
            text: Free-form user preference text.
        """
        self._profile.overrides = text
        self._profile.save(self._path)

    def refresh(
        self,
        tracks: list[TrackInfo],
        style_profile: dict[str, Any],
        history: list[dict[str, Any]],
        enrichment_tags: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Full refresh — recompute both library and generation signals.

        Args:
            tracks: Full list of library tracks.
            style_profile: Dict with ``genres``, ``moods``, ``instruments`` lists.
            history: List of generation history entries.
            enrichment_tags: Optional mapping of track_id -> extra tag strings.
        """
        self.update_library_signals(tracks, enrichment_tags)
        self.update_generation_signals(style_profile, history)
