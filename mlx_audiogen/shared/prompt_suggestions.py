"""AI prompt suggestion engine.

Provides prompt refinement suggestions using template-based expansion
and optional LLM integration. The template engine adds genre, mood,
instrument, and production descriptors to enhance generation quality.
"""

import random

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


def analyze_prompt(prompt: str) -> dict:
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
        "suggestions": suggest_refinements(prompt, count=3),
    }
