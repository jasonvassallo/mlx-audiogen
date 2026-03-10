import argparse
import re
import sys
from pathlib import Path

# Known-safe HuggingFace repo IDs for each model type.
# Users can pass --trust-remote-code to allow unlisted repos.
_KNOWN_MUSICGEN_REPOS = {
    "facebook/musicgen-small",
    "facebook/musicgen-medium",
    "facebook/musicgen-large",
    "facebook/musicgen-stereo-small",
    "facebook/musicgen-stereo-medium",
    "facebook/musicgen-stereo-large",
    "facebook/musicgen-melody",
    "facebook/musicgen-melody-large",
    "facebook/musicgen-stereo-melody",
    "facebook/musicgen-stereo-melody-large",
    "facebook/musicgen-style",
}
_KNOWN_STABLE_AUDIO_REPOS = {
    "stabilityai/stable-audio-open-small",
    "stabilityai/stable-audio-open-1.0",
}
_KNOWN_DEMUCS_VARIANTS = {
    "htdemucs",
    "htdemucs_6s",
}

# Valid HF repo ID: "org/model-name" (alphanumeric, hyphens, underscores, dots)
_REPO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


def _detect_model_type(repo_id: str) -> str | None:
    """Detect model type from repo ID using known repos, then prefix matching."""
    repo_lower = repo_id.lower()
    if repo_id in _KNOWN_MUSICGEN_REPOS:
        return "musicgen"
    if repo_id in _KNOWN_STABLE_AUDIO_REPOS:
        return "stable_audio"
    if repo_id in _KNOWN_DEMUCS_VARIANTS:
        return "demucs"
    # Fallback: match by org prefix for known organizations
    if repo_lower.startswith("facebook/musicgen"):
        return "musicgen"
    if repo_lower.startswith("stabilityai/stable-audio"):
        return "stable_audio"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to MLX format"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., facebook/musicgen-small)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mlx_model",
        help="Output directory for converted weights",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Data type for converted weights",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Allow conversion of non-whitelisted repo IDs",
    )

    args = parser.parse_args()

    # Demucs uses variant names (not HF repo IDs)
    model_type = _detect_model_type(args.model)

    # C2: Validate repo ID format (skip for demucs variants)
    if model_type != "demucs" and not _REPO_ID_PATTERN.match(args.model):
        print(f"Error: Invalid repo ID format: {args.model}")
        print("Expected format: 'organization/model-name' or demucs variant name")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # C2: Check against whitelist unless --trust-remote-code is set
    all_known = (
        _KNOWN_MUSICGEN_REPOS | _KNOWN_STABLE_AUDIO_REPOS | _KNOWN_DEMUCS_VARIANTS
    )
    if args.model not in all_known and not args.trust_remote_code:
        if model_type is not None:
            print(
                f"Warning: '{args.model}' is not in the known repo whitelist "
                f"but matches the '{model_type}' pattern."
            )
            print("Use --trust-remote-code to allow conversion of unlisted repos.")
            sys.exit(1)
        else:
            print(f"Error: Unknown model repo: {args.model}")
            print(f"Known repos: {sorted(all_known)}")
            print("Use --trust-remote-code to allow conversion of unlisted repos.")
            sys.exit(1)

    if model_type == "musicgen":
        # Style variants use audiocraft format — route to style converter
        if "style" in args.model.lower():
            from mlx_audiogen.models.musicgen.convert import convert_musicgen_style

            convert_musicgen_style(args.model, output_dir, dtype=args.dtype)
        else:
            from mlx_audiogen.models.musicgen.convert import convert_musicgen

            convert_musicgen(args.model, output_dir, dtype=args.dtype)
    elif model_type == "stable_audio":
        from mlx_audiogen.models.stable_audio.convert import (
            convert_stable_audio,
        )

        convert_stable_audio(args.model, output_dir, dtype=args.dtype)
    elif model_type == "demucs":
        from mlx_audiogen.models.demucs.convert import convert_demucs

        convert_demucs(output_dir, variant=args.model)
    else:
        print(f"Could not detect model type from repo ID: {args.model}")
        print(f"Known repos: {sorted(all_known)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
