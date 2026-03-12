"""AI prompt suggestion engine.

Provides prompt refinement suggestions using template-based expansion
and optional LLM integration. The template engine adds genre, mood,
instrument, and production descriptors to enhance generation quality.

Phase 7b adds: TAG_DATABASE (autocomplete), PromptMemory (history +
style profile), discover_mlx_models() (LLM scanner), and
enhance_with_llm() (local LLM prompt enhancement with fallback).
"""

from __future__ import annotations

import json
import random
import statistics
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Genre descriptors
GENRES = [
    "ambient",
    "electronic",
    "hip hop",
    "jazz",
    "classical",
    "rock",
    "synthwave",
    "lo-fi",
    "drum and bass",
    "house",
    "techno",
    "funk",
    "soul",
    "R&B",
    "folk",
    "cinematic",
    "orchestral",
    "industrial",
    "downtempo",
    "trap",
    "chillhop",
    "IDM",
    "breakbeat",
    "garage",
]

# Mood descriptors
MOODS = [
    "upbeat",
    "melancholic",
    "dreamy",
    "aggressive",
    "peaceful",
    "dark",
    "euphoric",
    "nostalgic",
    "mysterious",
    "ethereal",
    "intense",
    "warm",
    "cold",
    "hypnotic",
    "playful",
    "dramatic",
    "haunting",
    "groovy",
    "energetic",
    "contemplative",
]

# Production quality descriptors
PRODUCTION = [
    "warm analog",
    "crisp digital",
    "vinyl crackle",
    "tape saturated",
    "crystal clear",
    "lo-fi gritty",
    "punchy",
    "airy",
    "thick",
    "spacious",
    "tight",
    "compressed",
    "dynamic",
    "layered",
    "minimal",
    "lush",
    "raw",
    "polished",
    "organic",
    "synthetic",
]

# Instrument suggestions by category
INSTRUMENTS = {
    "drums": [
        "808 drums",
        "acoustic drums",
        "breakbeat",
        "trap hi-hats",
        "jazz brushes",
        "electronic percussion",
        "tribal drums",
    ],
    "bass": [
        "sub bass",
        "acid bass",
        "upright bass",
        "synth bass",
        "808 bass",
        "fretless bass",
        "moog bass",
        "reese bass",
    ],
    "keys": [
        "grand piano",
        "rhodes",
        "wurlitzer",
        "organ",
        "clavinet",
        "synth pad",
        "electric piano",
        "harpsichord",
        "marimba",
    ],
    "strings": [
        "orchestral strings",
        "string quartet",
        "solo violin",
        "cello",
        "harp",
        "pizzicato strings",
        "synth strings",
    ],
    "leads": [
        "saw lead",
        "square lead",
        "FM synth",
        "pluck synth",
        "brass section",
        "flute",
        "saxophone",
        "guitar",
    ],
}

# Era/style descriptors
ERAS = [
    "80s",
    "90s",
    "70s",
    "60s",
    "2000s",
    "2010s",
    "vintage",
    "modern",
    "retro",
    "futuristic",
    "classic",
    "Y2K",
    "art deco",
    "baroque",
    "contemporary",
]

# Unified tag database for autocomplete (all categories)
TAG_DATABASE: dict[str, list[str]] = {
    "genre": list(GENRES),
    "mood": list(MOODS),
    "instrument": [
        inst for instruments in INSTRUMENTS.values() for inst in instruments
    ],
    "era": list(ERAS),
    "production": list(PRODUCTION),
}


# ---------------------------------------------------------------------------
# Prompt Memory
# ---------------------------------------------------------------------------


class PromptMemory:
    """Manages prompt history and derived style profile.

    Stores history entries and auto-derives a style profile (top genres,
    moods, instruments) from the history using analyze_prompt().
    """

    def __init__(
        self,
        path: Path | None = None,
        max_entries: int = 2000,
    ):
        self._path = path or (Path.home() / ".mlx-audiogen" / "prompt_memory.json")
        self._max_entries = max_entries
        self.history: list[dict] = []
        self.style_profile: dict = {
            "top_genres": [],
            "top_moods": [],
            "top_instruments": [],
            "preferred_duration": 0,
            "generation_count": 0,
        }
        self._load()

    def _load(self) -> None:
        """Load from disk if file exists."""
        if self._path.is_file():
            try:
                data = json.loads(self._path.read_text())
                self.history = data.get("history", [])
                # Always re-derive profile from history
                self._derive_profile()
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        """Persist to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"history": self.history, "style_profile": self.style_profile}
        self._path.write_text(json.dumps(data, indent=2))

    def add_entry(
        self,
        prompt: str,
        model: str,
        params: dict,
        enhanced_prompt: str | None = None,
    ) -> None:
        """Append a generation entry and re-derive the style profile."""
        entry: dict = {
            "prompt": prompt,
            "model": model,
            "params": params,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if enhanced_prompt:
            entry["enhanced_prompt"] = enhanced_prompt
        self.history.append(entry)

        # Evict oldest if over limit
        if len(self.history) > self._max_entries:
            self.history = self.history[-self._max_entries :]

        self._derive_profile()
        self.save()

    def recent_prompts(self, count: int = 50) -> list[str]:
        """Return the N most recent prompt strings, newest first."""
        if count <= 0:
            # 0 = all history
            prompts = [e["prompt"] for e in reversed(self.history)]
        else:
            prompts = [e["prompt"] for e in reversed(self.history)][:count]
        return prompts

    def clear(self) -> None:
        """Clear all history and reset profile."""
        self.history = []
        self.style_profile = {
            "top_genres": [],
            "top_moods": [],
            "top_instruments": [],
            "preferred_duration": 0,
            "generation_count": 0,
        }
        self.save()

    def to_dict(self) -> dict:
        """Return serializable dict."""
        return {"history": self.history, "style_profile": self.style_profile}

    def _derive_profile(self) -> None:
        """Re-derive style profile from full history."""
        genre_counter: Counter[str] = Counter()
        mood_counter: Counter[str] = Counter()
        instrument_counter: Counter[str] = Counter()
        durations: list[float] = []

        for entry in self.history:
            analysis = analyze_prompt(entry["prompt"], count=0)
            genre_counter.update(analysis["genres"])
            mood_counter.update(analysis["moods"])
            instrument_counter.update(analysis["instruments"])
            secs = entry.get("params", {}).get("seconds")
            if secs is not None:
                durations.append(float(secs))

        self.style_profile = {
            "top_genres": [g for g, _ in genre_counter.most_common(5)],
            "top_moods": [m for m, _ in mood_counter.most_common(5)],
            "top_instruments": [i for i, _ in instrument_counter.most_common(5)],
            "preferred_duration": (
                int(statistics.median(durations)) if durations else 0
            ),
            "generation_count": len(self.history),
        }


# ---------------------------------------------------------------------------
# MLX Model Discovery
# ---------------------------------------------------------------------------


def discover_mlx_models(
    scan_paths: list[Path] | None = None,
) -> list[dict]:
    """Scan filesystem for valid MLX LLM model directories.

    A valid model dir has: config.json, *.safetensors, and a tokenizer file.
    Returns list of dicts with 'id', 'name', 'size_gb', 'source' (no paths).
    """
    if scan_paths is None:
        scan_paths = [
            Path.home() / ".cache" / "huggingface" / "hub",
            Path.home() / "Library" / "Caches" / "huggingface" / "hub",
            Path.home() / ".lmstudio" / "hub" / "models",
        ]

    found: dict[str, dict] = {}  # id -> info (dedup by id)

    for base in scan_paths:
        if not base.is_dir():
            continue
        _scan_dir_for_models(base, base, found)

    return list(found.values())


def _scan_dir_for_models(
    directory: Path,
    scan_root: Path,
    found: dict[str, dict],
) -> None:
    """Recursively scan a directory for MLX model dirs."""
    try:
        children = list(directory.iterdir())
    except PermissionError:
        return

    # Check if this directory IS a model dir
    if _is_valid_llm_dir(directory):
        model_id = _derive_model_id(directory, scan_root)
        if model_id and model_id not in found:
            size_gb = _estimate_size_gb(directory)
            source = "lmstudio" if ".lmstudio" in str(scan_root) else "huggingface"
            found[model_id] = {
                "id": model_id,
                "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                "size_gb": round(size_gb, 1),
                "source": source,
            }
        return  # Don't recurse into model dirs

    # HF snapshot resolution: models--org--name/snapshots/<hash>/
    for child in children:
        if not child.is_dir():
            continue
        if child.name.startswith("models--"):
            snapshots_dir = child / "snapshots"
            if snapshots_dir.is_dir():
                # Use the latest snapshot (last in sorted order)
                snapshot_dirs = sorted(
                    [s for s in snapshots_dir.iterdir() if s.is_dir()]
                )
                if snapshot_dirs:
                    _scan_dir_for_models(snapshot_dirs[-1], scan_root, found)
        else:
            # Recurse max 2 levels deep
            try:
                rel = child.relative_to(scan_root)
            except ValueError:
                continue
            if len(rel.parts) < 3:
                _scan_dir_for_models(child, scan_root, found)


def _is_valid_llm_dir(directory: Path) -> bool:
    """Check if a directory contains a valid MLX LLM model."""
    has_config = (directory / "config.json").is_file()
    has_safetensors = any(directory.glob("*.safetensors"))
    has_tokenizer = (directory / "tokenizer_config.json").is_file() or (
        directory / "tokenizer.json"
    ).is_file()
    return has_config and has_safetensors and has_tokenizer


def _derive_model_id(model_dir: Path, scan_root: Path) -> str | None:
    """Derive a human-readable model identifier from the path."""
    # Try to get org/name from path structure
    try:
        rel = model_dir.relative_to(scan_root)
    except ValueError:
        rel = Path(model_dir.name)

    parts = rel.parts

    # HF snapshot: models--org--name/snapshots/<hash> -> org/name
    for _i, part in enumerate(parts):
        if part.startswith("models--"):
            segments = part.split("--")
            if len(segments) >= 3:
                return f"{segments[1]}/{'/'.join(segments[2:])}"

    # Direct structure: org/name/ or just name/
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    if len(parts) == 1:
        return parts[0]

    return None


def _estimate_size_gb(model_dir: Path) -> float:
    """Estimate model size from safetensors files."""
    total = sum(f.stat().st_size for f in model_dir.glob("*.safetensors"))
    return total / (1024**3)


# ---------------------------------------------------------------------------
# LLM Prompt Enhancement
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """\
You are a music prompt engineer for AI audio generation models (MusicGen, Stable Audio).
Given a user's prompt, enhance it with rich musical descriptors including genre, mood,
instrumentation, tempo, production style, and era details.
Keep the user's core intent and artistic direction. Output ONLY the enhanced prompt
as a single line, nothing else. Do not add explanations or formatting.

{memory_context}"""


def enhance_with_llm(
    prompt: str,
    model_path: str | None = None,
    memory_context: str = "",
    timeout: int = 30,
) -> dict:
    """Enhance a prompt using a local MLX LLM, with template fallback.

    Returns dict with: original, enhanced, analysis_tags, used_llm, warning.
    """
    analysis = analyze_prompt(prompt, count=0)
    analysis_tags = {
        "genres": analysis["genres"],
        "moods": analysis["moods"],
        "instruments": analysis["instruments"],
        "missing": analysis["missing"],
    }

    if model_path is None:
        # No LLM available — use template fallback
        suggestions = suggest_refinements(prompt, count=1)
        return {
            "original": prompt,
            "enhanced": suggestions[0] if suggestions else prompt,
            "analysis_tags": analysis_tags,
            "used_llm": False,
            "warning": None,
        }

    try:
        system = _LLM_SYSTEM_PROMPT.format(memory_context=memory_context)
        enhanced = _run_llm_inference(prompt, system, model_path, timeout)
        # Truncate to 2000 chars for safety
        enhanced = enhanced[:2000].strip()
        if not enhanced:
            raise ValueError("LLM returned empty response")
        return {
            "original": prompt,
            "enhanced": enhanced,
            "analysis_tags": analysis_tags,
            "used_llm": True,
            "warning": None,
        }
    except TimeoutError:
        suggestions = suggest_refinements(prompt, count=1)
        return {
            "original": prompt,
            "enhanced": suggestions[0] if suggestions else prompt,
            "analysis_tags": analysis_tags,
            "used_llm": False,
            "warning": "LLM timed out, used template suggestions",
        }
    except Exception as e:
        suggestions = suggest_refinements(prompt, count=1)
        return {
            "original": prompt,
            "enhanced": suggestions[0] if suggestions else prompt,
            "analysis_tags": analysis_tags,
            "used_llm": False,
            "warning": f"LLM error: {e}",
        }


def _run_llm_inference(
    prompt: str,
    system: str,
    model_path: str,
    timeout: int = 30,
) -> str:
    """Run LLM inference with timeout. Separated for easy mocking."""
    result_holder: list[str] = []
    error_holder: list[Exception] = []

    def _infer() -> None:
        try:
            from mlx_lm import generate, load

            model, tokenizer, *_ = load(model_path)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            response = generate(model, tokenizer, prompt=chat_prompt, max_tokens=512)
            result_holder.append(response)
        except Exception as e:
            error_holder.append(e)

    thread = threading.Thread(target=_infer, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError(f"LLM inference exceeded {timeout}s timeout")
    if error_holder:
        raise error_holder[0]
    if not result_holder:
        raise RuntimeError("LLM produced no output")
    return result_holder[0]


def suggest_refinements(
    prompt: str,
    count: int = 4,
    seed: int | None = None,
) -> list[str]:
    """Generate prompt refinement suggestions.

    Takes a base prompt and returns enhanced versions with added
    descriptors for genre, mood, production quality, and instruments.

    Args:
        prompt: The original user prompt.
        count: Number of suggestions to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of refined prompt strings.
    """
    if seed is not None:
        rng = random.Random(seed)  # nosec B311 — not used for security
    else:
        rng = random.Random()  # nosec B311 — not used for security

    suggestions = []
    prompt_lower = prompt.lower()

    for _ in range(count):
        parts = [prompt.strip()]

        # Add a genre if none is detected
        if not any(g in prompt_lower for g in GENRES):
            parts.append(rng.choice(GENRES))

        # Add a mood
        if not any(m in prompt_lower for m in MOODS):
            parts.append(rng.choice(MOODS))

        # Add production quality
        parts.append(rng.choice(PRODUCTION))

        # Maybe add an instrument
        if rng.random() > 0.5:
            category = rng.choice(list(INSTRUMENTS.keys()))
            parts.append(rng.choice(INSTRUMENTS[category]))

        suggestions.append(", ".join(parts))

    return suggestions


def analyze_prompt(prompt: str, count: int = 3) -> dict:
    """Analyze a prompt and return detected attributes.

    Returns:
        Dict with detected genres, moods, instruments, and suggestions.
    """
    prompt_lower = prompt.lower()

    detected_genres = [g for g in GENRES if g in prompt_lower]
    detected_moods = [m for m in MOODS if m in prompt_lower]
    detected_instruments = []
    for _cat, instruments in INSTRUMENTS.items():
        for inst in instruments:
            if inst in prompt_lower:
                detected_instruments.append(inst)

    missing = []
    if not detected_genres:
        missing.append("genre")
    if not detected_moods:
        missing.append("mood")
    if not detected_instruments:
        missing.append("instruments")

    return {
        "genres": detected_genres,
        "moods": detected_moods,
        "instruments": detected_instruments,
        "missing": missing,
        "suggestions": suggest_refinements(prompt, count=count),
    }
