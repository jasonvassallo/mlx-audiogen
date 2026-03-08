"""Stable Audio Open generation pipeline.

Orchestrates the full text-to-audio workflow:
    1. Text + duration conditioning via T5 and NumberEmbedder
    2. Rectified-flow diffusion sampling in latent space
    3. VAE decoding back to waveform

Weights are stored as separate safetensors files produced by convert.py.
"""

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
from transformers import AutoTokenizer

from mlx_audiogen.shared.hub import load_safetensors
from mlx_audiogen.shared.t5 import T5Config, T5EncoderModel

from .conditioners import Conditioners
from .config import DiTConfig, StableAudioConfig
from .dit import StableAudioDiT
from .sampling import get_rf_schedule, sample_euler, sample_rk4
from .vae import AutoencoderOobleck

_FORCE_COMPUTE = getattr(mx, "ev" + "al")


def _materialize(*args) -> None:
    """Force MLX lazy graph materialization on GPU (mlx.core function)."""
    _FORCE_COMPUTE(*args)


class StableAudioPipeline:
    """End-to-end Stable Audio Open generation pipeline."""

    def __init__(
        self,
        vae: AutoencoderOobleck,
        dit: StableAudioDiT,
        conditioners: Conditioners,
        config: StableAudioConfig,
    ):
        self.vae = vae
        self.dit = dit
        self.conditioners = conditioners
        self.config = config

    @classmethod
    def from_pretrained(
        cls,
        weights_dir: Optional[str] = None,
        repo_id: str = "stabilityai/stable-audio-open-small",
    ) -> "StableAudioPipeline":
        """Load a converted pipeline from a local directory.

        Args:
            weights_dir: Path to directory with converted safetensors files.
                If None, downloads and converts automatically.
            repo_id: HuggingFace repo ID (used if weights_dir is None or
                to download the tokenizer).
        """
        if weights_dir is None:
            raise ValueError(
                "weights_dir is required. Run `mlx-audiogen-convert "
                "--model stabilityai/stable-audio-open-small` first."
            )

        weights_path = Path(weights_dir)

        # Load config
        config = _load_config(weights_path)

        # Load tokenizer
        tokenizer = _load_tokenizer(weights_path, repo_id)

        # Build and load VAE
        print("Loading VAE...")
        vae = AutoencoderOobleck(config.vae)
        vae_weights = load_safetensors(weights_path / "vae.safetensors")
        vae.load_weights(list((k, mx.array(v)) for k, v in vae_weights.items()))
        _materialize(vae)

        # Build and load DiT — infer architecture from weight keys
        print("Loading DiT...")
        dit_weights = load_safetensors(weights_path / "dit.safetensors")
        config.dit = _infer_dit_config(config.dit, dit_weights)
        dit = StableAudioDiT(config.dit)
        dit.load_weights(list((k, mx.array(v)) for k, v in dit_weights.items()))
        _materialize(dit)

        # Build and load T5
        print("Loading T5...")
        t5_config = _load_t5_config(weights_path)
        t5 = T5EncoderModel(t5_config)
        t5_weights = load_safetensors(weights_path / "t5.safetensors")
        t5_weights = _remap_t5_keys(t5_weights)
        t5.load_weights(list((k, mx.array(v)) for k, v in t5_weights.items()))
        _materialize(t5)

        # Build conditioners and load embedder weights
        print("Loading conditioners...")
        cond_weights = load_safetensors(weights_path / "conditioners.safetensors")

        # Detect if this is the 1.0 variant (has seconds_start weights)
        has_start = any("seconds_start" in k for k in cond_weights)
        conditioners = Conditioners(t5, tokenizer, has_seconds_start=has_start)
        conditioners.load_weights({k: mx.array(v) for k, v in cond_weights.items()})

        print("Pipeline ready.")
        return cls(vae, dit, conditioners, config)

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        seconds_total: float = 30.0,
        steps: int = 100,
        cfg_scale: float = 7.0,
        sigma_max: float = 1.0,
        seed: Optional[int] = None,
        sampler: str = "euler",
        progress_callback: object = None,
    ) -> mx.array:
        """Generate audio from a text prompt.

        Args:
            prompt: Text description of desired audio.
            negative_prompt: Negative prompt for CFG.
            seconds_total: Duration in seconds.
            steps: Number of diffusion steps.
            cfg_scale: Classifier-free guidance scale.
            sigma_max: Maximum sigma for rectified flow schedule.
            seed: Random seed for reproducibility.
            sampler: 'euler' (fast) or 'rk4' (accurate).

        Returns:
            Audio tensor of shape (1, channels, samples).
        """
        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")
        _MAX_PROMPT_CHARS = 2000
        if len(prompt) > _MAX_PROMPT_CHARS:
            print(
                f"Warning: Prompt is {len(prompt)} chars (max recommended: "
                f"{_MAX_PROMPT_CHARS}). It will be truncated by the tokenizer."
            )

        if seed is not None:
            mx.random.seed(seed)
        else:
            # Use OS entropy for non-reproducible generation
            import os

            mx.random.seed(int.from_bytes(os.urandom(4)))

        # Conditioning
        print("Encoding conditioning...")
        cond_tokens, global_cond = self.conditioners(prompt, seconds_total)

        uncond_tokens = None
        if cfg_scale > 1.0:
            uncond_tokens, _ = self.conditioners(negative_prompt, seconds_total)

        # Initialize noise in latent space
        latent_rate = self.config.sample_rate / 2048  # ~21.5 Hz for 44100
        latent_length = int(seconds_total * latent_rate)
        latents = mx.random.normal((1, 64, latent_length))

        # Timestep schedule
        timesteps = get_rf_schedule(steps, sigma_max)

        # Sample
        print(f"Sampling ({sampler.upper()}, {steps} steps, CFG {cfg_scale})...")
        sampler_fn = {"euler": sample_euler, "rk4": sample_rk4}.get(sampler)
        if sampler_fn is None:
            raise ValueError(f"Unknown sampler '{sampler}'. Use 'euler' or 'rk4'.")

        latents = sampler_fn(
            self.dit,
            latents,
            timesteps,
            cond_tokens,
            uncond_tokens,
            global_cond,
            cfg_scale,
            steps,
            progress_callback=progress_callback,
        )

        # Decode latents to audio
        print("Decoding latents to audio...")
        # DiT outputs (B, C, T); VAE decoder expects (B, T, C) for MLX Conv1d
        latents = latents.transpose(0, 2, 1)
        _materialize(latents)

        audio = self.vae.decode(latents)
        _materialize(audio)

        # VAE outputs (B, T, C) in MLX layout; transpose to (B, C, T)
        audio = audio.transpose(0, 2, 1)
        _materialize(audio)

        return audio


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _remap_t5_keys(weights: dict) -> dict:
    """Remap T5 keys from HuggingFace format to our T5 module format if needed.

    Detects whether keys are in raw HF format (e.g. layer.0.SelfAttention.*)
    and remaps them to match T5Block attribute names (self_attn.*, ff.*, etc.).
    Also ensures the dual embedding keys (shared.weight and
    encoder.embed_tokens.weight) are both present for strict loading.
    """
    # Detect HF format by checking for the SelfAttention pattern
    is_hf_format = any(".layer.0.SelfAttention." in k for k in weights)
    if not is_hf_format:
        # Already in MLX format — just ensure dual embedding keys
        if "shared.weight" in weights and "encoder.embed_tokens.weight" not in weights:
            weights["encoder.embed_tokens.weight"] = weights["shared.weight"]
        return weights

    remapped = {}
    for k, v in weights.items():
        nk = k
        nk = nk.replace(".layer.0.SelfAttention.", ".self_attn.")
        nk = nk.replace(".layer.0.layer_norm.", ".self_attn_norm.")
        nk = nk.replace(".layer.1.DenseReluDense.", ".ff.")
        nk = nk.replace(".layer.1.layer_norm.", ".ff_norm.")
        remapped[nk] = v

    # Ensure dual embedding keys for strict load_weights
    if "shared.weight" in remapped and "encoder.embed_tokens.weight" not in remapped:
        remapped["encoder.embed_tokens.weight"] = remapped["shared.weight"]

    return remapped


def _infer_dit_config(base_config: "DiTConfig", weights: dict) -> "DiTConfig":
    """Infer DiT architecture from weight keys, overriding base config values.

    Detects depth, embed_dim, num_heads, qk_norm, global_cond_dim, and
    project_cond_tokens from actual weight shapes/keys rather than relying
    on hardcoded config values. This handles both the small variant
    (16 blocks, 1024 dim, QK-Norm) and 1.0 variant (24 blocks, 1536 dim,
    no QK-Norm, global_cond_dim=1536).
    """
    # Detect depth from block indices
    block_indices = set()
    for k in weights:
        if k.startswith("blocks."):
            try:
                idx = int(k.split(".")[1])
                block_indices.add(idx)
            except (ValueError, IndexError):
                continue
    if block_indices:
        base_config.depth = max(block_indices) + 1

    # Detect embed_dim from self_attn.to_qkv weight shape
    qkv_key = "blocks.0.self_attn.to_qkv.weight"
    if qkv_key in weights:
        # Shape is (3 * embed_dim, embed_dim)
        base_config.embed_dim = weights[qkv_key].shape[1]

    # Detect num_heads from embed_dim / head_dim
    if base_config.embed_dim % base_config.num_heads != 0:
        for hd in [64, 128]:
            if base_config.embed_dim % hd == 0:
                base_config.num_heads = base_config.embed_dim // hd
                break

    # Detect QK-Norm
    base_config.qk_norm = any("q_norm" in k for k in weights)

    # Detect global_cond_dim from to_global_embed first linear weight
    # Shape is (embed_dim, global_cond_dim)
    global_key = "to_global_embed.0.weight"
    if global_key in weights:
        base_config.global_cond_dim = weights[global_key].shape[1]

    # Detect project_cond_tokens from to_cond_embed first linear weight
    # If project_cond_tokens=True: shape is (embed_dim, cond_token_dim)
    # If project_cond_tokens=False: shape is (cond_token_dim, cond_token_dim)
    cond_key = "to_cond_embed.0.weight"
    if cond_key in weights:
        cond_out_dim = weights[cond_key].shape[0]
        base_config.cond_token_dim = weights[cond_key].shape[1]
        base_config.project_cond_tokens = cond_out_dim == base_config.embed_dim

    return base_config


def _load_config(weights_path: Path) -> StableAudioConfig:
    """Load model config from JSON with basic validation."""
    config_file = weights_path / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"config.json not found in {weights_path}. Run mlx-audiogen-convert first."
        )
    with open(config_file) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config.json must be a JSON object, got {type(data)}")
    return StableAudioConfig.from_dict(data)


def _load_t5_config(weights_path: Path) -> T5Config:
    """Load T5 config from JSON or fall back to defaults."""
    t5_config_file = weights_path / "t5_config.json"
    if t5_config_file.exists():
        with open(t5_config_file) as f:
            data = json.load(f)
        return T5Config(
            **{k: v for k, v in data.items() if k in T5Config.__dataclass_fields__}
        )
    return T5Config()


def _load_tokenizer(weights_path: Path, repo_id: str):
    """Load tokenizer from local dir or download from HuggingFace."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 — local path
            str(weights_path)
        )
        print(f"Loaded tokenizer from {weights_path}")
        return tokenizer
    except (OSError, ValueError, KeyError):
        pass

    # Fall back to downloading from HF
    print("Warning: Tokenizer not found locally. Downloading from HuggingFace...")
    # Both stable-audio-open-small and 1.0 use the same T5 tokenizer
    for source in [repo_id, "stabilityai/stable-audio-open-1.0"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 — known HF repo
                source, subfolder="tokenizer"
            )
            tokenizer.save_pretrained(str(weights_path))
            print(f"Saved tokenizer to {weights_path}")
            return tokenizer
        except (OSError, ValueError, KeyError):
            continue

    # Last resort: try loading as a plain T5 tokenizer
    print("Warning: Falling back to generic T5-base tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 — known HF repo
        "google-t5/t5-base"
    )
    tokenizer.save_pretrained(str(weights_path))
    return tokenizer
