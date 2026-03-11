"""Convert MusicGen weights from HuggingFace safetensors to MLX format.

Reads the HF model.safetensors and splits it into:
    - decoder.safetensors  (transformer + embeddings + LM heads + enc_to_dec_proj)
    - t5.safetensors       (T5 text encoder)
    - config.json          (model configuration)
    - t5_config.json       (T5 configuration)
    + tokenizer files

Weight key remapping:
    Decoder: Strip 'decoder.model.decoder.' prefix, rename encoder_attn -> encoder_attn
    T5: Strip 'text_encoder.' prefix, map HF T5 layout to our T5 module layout
    Audio encoder: SKIPPED (we load EnCodec separately from mlx-community)
"""

import json
from pathlib import Path

import numpy as np

from mlx_audiogen.shared.hub import (
    download_model,
    load_pytorch_bin,
    load_safetensors,
    save_safetensors,
)


def _remap_t5_key(k: str) -> str:
    """Remap a single T5 key from HF format to our T5 module format.

    HF format:
        encoder.block.{i}.layer.0.SelfAttention.{q,k,v,o}.weight
        encoder.block.{i}.layer.0.layer_norm.weight
        encoder.block.{i}.layer.1.DenseReluDense.{wi,wo}.weight
        encoder.block.{i}.layer.1.layer_norm.weight
        encoder.final_layer_norm.weight
        shared.weight

    Our T5 module format:
        encoder.block.{i}.self_attn.{q,k,v,o}.weight
        encoder.block.{i}.self_attn_norm.weight
        encoder.block.{i}.ff.{wi,wo}.weight
        encoder.block.{i}.ff_norm.weight
        encoder.final_layer_norm.weight
        shared.weight

    Also handles relative_attention_bias at block 0.
    """
    # Self-attention
    k = k.replace(".layer.0.SelfAttention.", ".self_attn.")
    # Self-attention norm
    k = k.replace(".layer.0.layer_norm.", ".self_attn_norm.")
    # Feed-forward
    k = k.replace(".layer.1.DenseReluDense.", ".ff.")
    # Feed-forward norm
    k = k.replace(".layer.1.layer_norm.", ".ff_norm.")
    # Relative attention bias embedding
    k = k.replace(
        ".self_attn.relative_attention_bias.",
        ".self_attn.relative_attention_bias.",
    )
    return k


def _remap_decoder_key(k: str) -> str:
    """Remap a decoder key from HF format to our module format.

    HF format:
        decoder.model.decoder.layers.{i}.self_attn.{q,k,v,out}_proj.weight
        decoder.model.decoder.layers.{i}.encoder_attn.{q,k,v,out}_proj.weight
        decoder.model.decoder.layers.{i}.self_attn_layer_norm.weight/bias
        decoder.model.decoder.layers.{i}.encoder_attn_layer_norm.weight/bias
        decoder.model.decoder.layers.{i}.fc1.weight/bias
        decoder.model.decoder.layers.{i}.fc2.weight/bias
        decoder.model.decoder.layers.{i}.final_layer_norm.weight/bias
        decoder.model.decoder.layer_norm.weight/bias
        decoder.model.decoder.embed_tokens.{k}.weight
        decoder.lm_heads.{k}.weight
        enc_to_dec_proj.weight/bias

    Our module format:
        layers.{i}.self_attn.{q,k,v,out}_proj.weight
        layers.{i}.encoder_attn.{q,k,v,out}_proj.weight
        layers.{i}.self_attn_layer_norm.weight/bias
        layers.{i}.encoder_attn_layer_norm.weight/bias
        layers.{i}.fc1.weight/bias
        layers.{i}.fc2.weight/bias
        layers.{i}.final_layer_norm.weight/bias
        layer_norm.weight/bias
        embed_tokens.{k}.weight
        lm_heads.{k}.weight
        enc_to_dec_proj.weight/bias
    """
    # Strip decoder.model.decoder. prefix for transformer layers
    if k.startswith("decoder.model.decoder."):
        k = k[len("decoder.model.decoder.") :]
    # Strip decoder. prefix for lm_heads
    elif k.startswith("decoder.lm_heads."):
        k = k[len("decoder.") :]
    # enc_to_dec_proj stays as-is (already matches our attribute name)

    return k


def _load_weights(model_path: Path) -> dict[str, np.ndarray]:
    """Load model weights from safetensors or pytorch_model.bin.

    Prefers safetensors format. Falls back to pytorch_model.bin
    (requires torch: ``pip install mlx-audiogen[convert]``).

    Handles both single-file and sharded (multi-file) weight storage.
    """
    import glob

    # 1. Single safetensors file
    sf_file = model_path / "model.safetensors"
    if sf_file.exists():
        return load_safetensors(sf_file)

    # 2. Sharded safetensors (model-00001-of-NNNNN.safetensors, ...)
    sf_files = sorted(glob.glob(str(model_path / "model*.safetensors")))
    if sf_files:
        weights: dict[str, np.ndarray] = {}
        for sf in sf_files:
            shard = load_safetensors(sf)
            weights.update(shard)
            print(f"  Loaded {Path(sf).name}: {len(shard)} tensors")
        return weights

    # 3. Single pytorch_model.bin
    pt_file = model_path / "pytorch_model.bin"
    if pt_file.exists():
        print("  No safetensors found, loading pytorch_model.bin (requires torch)...")
        return load_pytorch_bin(pt_file)

    # 4. Sharded pytorch_model-NNNNN-of-NNNNN.bin
    pt_files = sorted(glob.glob(str(model_path / "pytorch_model-*.bin")))
    if pt_files:
        n = len(pt_files)
        print(f"  No safetensors, loading {n} pytorch shards...")
        weights = {}
        for pt in pt_files:
            shard = load_pytorch_bin(pt)
            weights.update(shard)
            print(f"  Loaded {Path(pt).name}: {len(shard)} tensors")
        return weights

    raise FileNotFoundError(
        f"No model weight files found in {model_path}. "
        "Expected model.safetensors or pytorch_model.bin."
    )


def _cast_dtype(dtype: str | None, *dicts: dict[str, np.ndarray]) -> None:
    """Cast floating-point weights to the specified dtype in-place.

    Args:
        dtype: Target dtype string ('float16', 'bfloat16', 'float32')
            or None to skip.
        *dicts: Weight dictionaries to cast.
    """
    if dtype is None:
        return
    np_dtype = {
        "float16": np.float16,
        "bfloat16": np.float16,
        "float32": np.float32,
    }[dtype]
    for d in dicts:
        for k in d:
            if d[k].dtype in (np.float32, np.float64):
                d[k] = d[k].astype(np_dtype)


def convert_musicgen(
    repo_id: str,
    output_dir: str | Path,
    dtype: str | None = None,
) -> None:
    """Download and convert a MusicGen model from HuggingFace.

    Args:
        repo_id: HuggingFace repo ID (e.g. 'facebook/musicgen-small').
        output_dir: Directory to write converted safetensors files.
        dtype: Optional cast dtype ('float16', 'bfloat16', 'float32').
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download from HF (include .bin files as fallback for repos without safetensors)
    print(f"Downloading {repo_id}...")
    model_path = download_model(
        repo_id,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"],
    )

    # Load weights — prefer safetensors, fall back to pytorch_model.bin
    print("Loading weights...")
    weights = _load_weights(model_path)

    # Load HF config
    hf_config = {}
    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            hf_config = json.load(f)

    # Sort weights into buckets
    t5_state: dict[str, np.ndarray] = {}
    decoder_state: dict[str, np.ndarray] = {}
    skipped_audio = 0
    skipped_other = 0

    total = len(weights)
    print(f"Processing {total} tensors...")

    for k, val in weights.items():
        # --- T5 Text Encoder ---
        if k.startswith("text_encoder."):
            clean_k = k[len("text_encoder.") :]
            clean_k = _remap_t5_key(clean_k)
            t5_state[clean_k] = val
            continue

        # --- Audio Encoder (EnCodec) --- SKIP
        # We load EnCodec separately from mlx-community
        if k.startswith("audio_encoder."):
            skipped_audio += 1
            continue

        # --- Decoder (transformer + embed_tokens + lm_heads) ---
        if k.startswith("decoder.model.decoder.") or k.startswith("decoder.lm_heads."):
            clean_k = _remap_decoder_key(k)
            decoder_state[clean_k] = val
            continue

        # --- enc_to_dec_proj (text) ---
        if k.startswith("enc_to_dec_proj."):
            decoder_state[k] = val
            continue

        # --- audio_enc_to_dec_proj (melody variant: chroma projection) ---
        if k.startswith("audio_enc_to_dec_proj."):
            decoder_state[k] = val
            continue

        # Anything else
        skipped_other += 1

    # Our T5EncoderModel has both shared.weight and encoder.embed_tokens.weight
    # pointing to the same embedding. HF only stores shared.weight, so we
    # duplicate it under both keys so load_weights(strict=True) succeeds.
    if "shared.weight" in t5_state and "encoder.embed_tokens.weight" not in t5_state:
        t5_state["encoder.embed_tokens.weight"] = t5_state["shared.weight"]

    # Optional dtype cast
    _cast_dtype(dtype, t5_state, decoder_state)

    # Save split files
    print("\nConverted weights:")
    print(f"  T5 encoder:     {len(t5_state)} tensors")
    print(f"  Decoder:        {len(decoder_state)} tensors")
    print(f"  Audio encoder:  {skipped_audio} tensors (skipped, loaded separately)")
    if skipped_other:
        print(f"  Other skipped:  {skipped_other} tensors")

    save_safetensors(t5_state, output_dir / "t5.safetensors")
    save_safetensors(decoder_state, output_dir / "decoder.safetensors")

    # Save configs
    _save_configs(output_dir, hf_config, repo_id)

    # Save tokenizer
    _save_tokenizer(output_dir, repo_id)

    print(f"\nConversion complete! Weights saved to {output_dir}/")


def _load_audiocraft_state_dict(path: Path) -> dict[str, np.ndarray]:
    """Load an audiocraft state_dict.bin and extract tensor weights.

    Audiocraft checkpoints wrap weights under a 'best_state' key alongside
    metadata (xp.cfg, version, exported). This function unwraps that nesting
    and converts all tensors to numpy arrays.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load audiocraft .bin files. "
            "Install it with: uv sync --extra convert"
        ) from exc

    checkpoint = torch.load(  # nosec B614 — known HF model file
        str(path), map_location="cpu", weights_only=False
    )

    # Unwrap 'best_state' if present (standard audiocraft format)
    if isinstance(checkpoint, dict) and "best_state" in checkpoint:
        state_dict = checkpoint["best_state"]
    else:
        state_dict = checkpoint

    weights: dict[str, np.ndarray] = {}
    for key, val in state_dict.items():
        if hasattr(val, "numpy"):
            weights[key] = val.numpy()
    return weights


def convert_musicgen_style(
    repo_id: str,
    output_dir: str | Path,
    dtype: str | None = None,
) -> None:
    """Download and convert a MusicGen-Style model from audiocraft format.

    MusicGen-Style uses audiocraft's native ``state_dict.bin`` format with
    a ``best_state`` top-level key, rather than HuggingFace transformers
    safetensors. This requires different key remapping and fused attention
    weight splitting.

    Produces:
        - decoder.safetensors (transformer + embeddings + LM heads + projection)
        - t5.safetensors (T5 text encoder)
        - style.safetensors (style conditioner: transformer + BatchNorm + RVQ)
        - mert.safetensors (MERT frozen feature extractor)
        - config.json, t5_config.json, tokenizer files

    Args:
        repo_id: HuggingFace repo ID (e.g. 'facebook/musicgen-style').
        output_dir: Directory to write converted safetensors files.
        dtype: Optional cast dtype ('float16', 'bfloat16', 'float32').
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download audiocraft state dict
    print(f"Downloading {repo_id}...")
    model_path = download_model(
        repo_id,
        allow_patterns=["*.bin", "*.json", "*.txt", "*.model"],
    )

    # Load state dict (audiocraft format wraps weights under 'best_state')
    print("Loading audiocraft state_dict.bin...")
    state_dict_file = model_path / "state_dict.bin"
    if not state_dict_file.exists():
        raise FileNotFoundError(
            f"state_dict.bin not found in {model_path}. "
            "This converter is for audiocraft-format models."
        )
    raw_weights = _load_audiocraft_state_dict(state_dict_file)

    # Sort weights into buckets
    t5_state: dict[str, np.ndarray] = {}
    decoder_state: dict[str, np.ndarray] = {}
    style_state: dict[str, np.ndarray] = {}
    skipped = 0

    total = len(raw_weights)
    print(f"Processing {total} tensors from audiocraft format...")

    # Audiocraft best_state can store decoder keys with or without 'lm.' prefix.
    # Detect whether 'lm.' prefix is present; if not, we add it for routing.
    has_lm_prefix = any(k.startswith("lm.") for k in raw_weights)

    for k, val in raw_weights.items():
        # --- T5 text encoder ---
        # Audiocraft: condition_provider.conditioners.description.t5.{...}
        if "conditioners.description.t5." in k:
            # Strip to get T5 relative key
            t5_key = k.split("conditioners.description.t5.")[-1]
            t5_key = _remap_t5_key(t5_key)
            t5_state[t5_key] = val
            continue

        # --- Style conditioner ---
        # Audiocraft: condition_provider.conditioners.self_wav.{...}
        if "conditioners.self_wav." in k:
            style_key = k.split("conditioners.self_wav.")[-1]
            # Handle fused style transformer attention weights
            if "transformer." in style_key and "in_proj_weight" in style_key:
                _split_fused_style_attn(style_key, val, style_state)
                continue
            remapped_style_key = _remap_style_key(style_key)
            if remapped_style_key is not None:
                style_state[remapped_style_key] = val
            continue

        # --- Decoder transformer ---
        # Normalize to lm.-prefixed key for the remapper
        remap_key = k if has_lm_prefix else f"lm.{k}"
        if remap_key.startswith("lm."):
            dec_key = _remap_audiocraft_decoder_key(remap_key, val)
            if dec_key is not None:
                if isinstance(dec_key, list):
                    # Fused weight was split into multiple keys
                    for dk, dv in dec_key:
                        decoder_state[dk] = dv
                else:
                    decoder_state[dec_key] = val
                continue

        skipped += 1

    # T5 is not in the audiocraft checkpoint — download from google-t5/t5-base
    if not t5_state:
        t5_state = _download_t5_weights(dtype)

    # T5 shared embedding duplication
    if "shared.weight" in t5_state and "encoder.embed_tokens.weight" not in t5_state:
        t5_state["encoder.embed_tokens.weight"] = t5_state["shared.weight"]

    # Optional dtype cast
    _cast_dtype(dtype, t5_state, decoder_state, style_state)

    print("\nConverted weights:")
    print(f"  T5 encoder:        {len(t5_state)} tensors")
    print(f"  Decoder:           {len(decoder_state)} tensors")
    print(f"  Style conditioner: {len(style_state)} tensors")
    if skipped:
        print(f"  Skipped:           {skipped} tensors")

    # Save weight files
    save_safetensors(t5_state, output_dir / "t5.safetensors")
    save_safetensors(decoder_state, output_dir / "decoder.safetensors")
    save_safetensors(style_state, output_dir / "style.safetensors")

    # Download and convert MERT weights (separate model)
    _convert_mert_weights(output_dir, dtype)

    # Build and save configs (style models don't have HF config.json)
    _save_style_configs(output_dir, raw_weights, repo_id)

    # Save tokenizer
    _save_tokenizer(output_dir, repo_id)

    print(f"\nConversion complete! Weights saved to {output_dir}/")


def _split_fused_style_attn(
    k: str, val: np.ndarray, style_state: dict[str, np.ndarray]
) -> None:
    """Split fused in_proj_weight for style transformer self-attention.

    Audiocraft stores Q/K/V as a single (3*dim, dim) matrix.
    We split into three separate (dim, dim) projections.

    Args:
        k: Key like 'transformer.layers.0.self_attn.in_proj_weight'
        val: Fused weight array of shape (3*dim, dim).
        style_state: Dict to store the split weights into.
    """
    # Extract layer index from key
    # k format: transformer.layers.{i}.self_attn.in_proj_weight
    parts = k.split(".")
    layer_idx = parts[2]  # The layer number
    dim = val.shape[0] // 3

    q_weight = val[:dim]
    k_weight = val[dim : 2 * dim]
    v_weight = val[2 * dim :]

    prefix = f"transformer_layers.{layer_idx}"
    style_state[f"{prefix}.self_attn_q_proj.weight"] = q_weight
    style_state[f"{prefix}.self_attn_k_proj.weight"] = k_weight
    style_state[f"{prefix}.self_attn_v_proj.weight"] = v_weight


def _remap_style_key(k: str) -> str | None:
    """Remap an audiocraft style conditioner key to our module format.

    Audiocraft style keys (after stripping condition_provider.conditioners.self_wav.):
        embed.{weight/bias}        -> embed.{weight/bias}  (input proj: 768→512)
        output_proj.{weight/bias}  -> output_proj.{weight/bias}  (output proj: 512→1536)
        transformer.{layer_key}    -> transformer_layers.{remapped}
        batch_norm.{running_mean/var} -> batch_norm_running_{mean/var}
        rvq.vq.layers.{i}._codebook.embed -> rvq.layers_{i}.codebook.weight

    Returns None for keys we don't need.
    """
    # Input embedding projection (MERT 768 → style_dim 512)
    if k.startswith("embed."):
        return k  # Already named correctly

    # Output projection (style_dim 512 → decoder hidden 1536)
    if k.startswith("output_proj."):
        return k  # Keep as output_proj

    # Style transformer layers
    if k.startswith("transformer."):
        rest = k[len("transformer.") :]
        return _remap_style_transformer_key(rest)

    # Batch normalization running statistics
    if k == "batch_norm.running_mean":
        return "batch_norm_running_mean"
    if k == "batch_norm.running_var":
        return "batch_norm_running_var"
    if k == "batch_norm.num_batches_tracked":
        return None  # Not needed for inference

    # RVQ codebook embeddings
    # rvq.vq.layers.{i}._codebook.embed -> rvq.layers_{i}.codebook.weight
    if k.startswith("rvq.vq.layers."):
        rest = k[len("rvq.vq.layers.") :]
        # rest: "0._codebook.embed" or similar
        parts = rest.split(".", 1)
        layer_idx = parts[0]
        if len(parts) > 1 and parts[1] == "_codebook.embed":
            return f"rvq.layers_{layer_idx}.codebook.weight"
        # Skip embed_avg, cluster_size, inited (training-only)
        return None

    return None


def _remap_style_transformer_key(k: str) -> str | None:
    """Remap a style transformer layer key.

    Audiocraft format (inside transformer.):
        layers.{i}.self_attn.in_proj_weight -> split into Q/K/V
        layers.{i}.self_attn.out_proj.weight
        layers.{i}.norm1.weight/bias
        layers.{i}.norm2.weight/bias
        layers.{i}.linear1.weight
        layers.{i}.linear2.weight

    Our format:
        transformer_layers.{i}.self_attn_q_proj.weight
        transformer_layers.{i}.self_attn_k_proj.weight
        transformer_layers.{i}.self_attn_v_proj.weight
        transformer_layers.{i}.self_attn_out_proj.weight
        transformer_layers.{i}.norm1.weight/bias
        transformer_layers.{i}.norm2.weight/bias
        transformer_layers.{i}.linear1.weight
        transformer_layers.{i}.linear2.weight
    """
    # Skip fused in_proj_weight — handled separately during conversion
    if "in_proj_weight" in k:
        return None  # Handled in _split_fused_style_attn

    # All other keys: just replace 'layers.' with 'transformer_layers.'
    # and 'self_attn.out_proj.' with 'self_attn_out_proj.'
    k = k.replace("self_attn.out_proj.", "self_attn_out_proj.")
    return f"transformer_{k}"


def _remap_audiocraft_decoder_key(
    k: str, val: np.ndarray
) -> str | list[tuple[str, np.ndarray]] | None:
    """Remap an audiocraft decoder key to our module format.

    Audiocraft decoder keys (under lm.):
        lm.emb.{k}.weight          -> embed_tokens.{k}.weight
        lm.transformer.layers.{i}.{component}
        lm.linears.{k}.weight      -> lm_heads.{k}.weight
        lm.out_norm.weight/bias     -> layer_norm.weight/bias
        lm.condition_provider.conditioners.description.output_proj.weight
            -> enc_to_dec_proj.weight

    Handles fused in_proj_weight splitting for self-attention and cross-attention.
    """
    # Strip 'lm.' prefix
    if not k.startswith("lm."):
        return None
    key = k[3:]

    # Embeddings: emb.{k}.weight -> embed_tokens.{k}.weight
    if key.startswith("emb."):
        return key.replace("emb.", "embed_tokens.")

    # LM heads: linears.{k}.weight -> lm_heads.{k}.weight
    if key.startswith("linears."):
        return key.replace("linears.", "lm_heads.")

    # Output layer norm
    if key.startswith("out_norm."):
        return key.replace("out_norm.", "layer_norm.")

    # enc_to_dec_proj (text projection)
    if "output_proj." in key and "description" in key:
        suffix = key.split("output_proj.")[-1]
        return f"enc_to_dec_proj.{suffix}"

    # Transformer layers
    if key.startswith("transformer."):
        rest = key[len("transformer.") :]
        return _remap_audiocraft_transformer_layer(rest, val)

    return None


def _remap_audiocraft_transformer_layer(
    k: str, val: np.ndarray
) -> str | list[tuple[str, np.ndarray]] | None:
    """Remap audiocraft transformer layer keys, splitting fused attention weights.

    Audiocraft uses:
        layers.{i}.self_attn.in_proj_weight  (3*dim, dim) — fused QKV
        layers.{i}.self_attn.out_proj.weight
        layers.{i}.cross_attention.in_proj_weight  (3*dim, dim) — fused QKV
        layers.{i}.cross_attention.out_proj.weight
        layers.{i}.norm1.weight/bias         -> self_attn_layer_norm
        layers.{i}.norm_cross.weight/bias    -> encoder_attn_layer_norm
        layers.{i}.norm2.weight/bias         -> final_layer_norm
        layers.{i}.linear1.weight            -> fc1.weight
        layers.{i}.linear2.weight            -> fc2.weight
    """
    # Split fused self-attention QKV
    if "self_attn.in_proj_weight" in k:
        prefix = k.replace("self_attn.in_proj_weight", "")
        dim = val.shape[0] // 3
        q_weight = val[:dim]
        k_weight = val[dim : 2 * dim]
        v_weight = val[2 * dim :]
        return [
            (f"{prefix}self_attn.q_proj.weight", q_weight),
            (f"{prefix}self_attn.k_proj.weight", k_weight),
            (f"{prefix}self_attn.v_proj.weight", v_weight),
        ]

    # Split fused cross-attention QKV
    if "cross_attention.in_proj_weight" in k:
        prefix = k.replace("cross_attention.in_proj_weight", "")
        dim = val.shape[0] // 3
        q_weight = val[:dim]
        k_weight = val[dim : 2 * dim]
        v_weight = val[2 * dim :]
        return [
            (f"{prefix}encoder_attn.q_proj.weight", q_weight),
            (f"{prefix}encoder_attn.k_proj.weight", k_weight),
            (f"{prefix}encoder_attn.v_proj.weight", v_weight),
        ]

    # Self-attention out projection
    k = k.replace("self_attn.out_proj.", "self_attn.out_proj.")

    # Cross-attention out projection
    k = k.replace("cross_attention.out_proj.", "encoder_attn.out_proj.")

    # Layer norms: audiocraft -> our naming
    k = k.replace(".norm1.", ".self_attn_layer_norm.")
    k = k.replace(".norm_cross.", ".encoder_attn_layer_norm.")
    k = k.replace(".norm2.", ".final_layer_norm.")

    # FFN
    k = k.replace(".linear1.", ".fc1.")
    k = k.replace(".linear2.", ".fc2.")

    return k


def _download_t5_weights(dtype: str | None = None) -> dict[str, np.ndarray]:
    """Download T5-base encoder weights and remap to our T5 module format.

    Audiocraft loads T5 at runtime from HuggingFace, so it's not in the
    checkpoint. We download the encoder weights and apply key remapping.
    """
    print("Downloading T5-base encoder weights...")
    t5_path = download_model(
        "google-t5/t5-base",
        allow_patterns=["model.safetensors", "*.json"],
    )

    all_weights = load_safetensors(t5_path / "model.safetensors")
    t5_state: dict[str, np.ndarray] = {}
    for k, v in all_weights.items():
        # Only keep encoder weights (skip decoder)
        if k.startswith("encoder.") or k.startswith("shared."):
            t5_state[_remap_t5_key(k)] = v

    if dtype:
        dtype_map = {
            "float16": np.float16,
            "bfloat16": np.float16,
            "float32": np.float32,
        }
        np_dtype = dtype_map[dtype]
        for k in t5_state:
            if t5_state[k].dtype in (np.float32, np.float64):
                t5_state[k] = t5_state[k].astype(np_dtype)

    print(f"  T5: {len(t5_state)} tensors")
    return t5_state


def _convert_mert_weights(output_dir: Path, dtype: str | None = None) -> None:
    """Download and convert MERT-v1-95M weights to MLX safetensors.

    MERT weights are stored as pytorch_model.bin in the m-a-p/MERT-v1-95M repo.
    We extract and remap the keys to match our MERTModel structure.
    """
    mert_output = output_dir / "mert.safetensors"
    if mert_output.exists():
        print("MERT weights already converted, skipping...")
        return

    print("Downloading MERT-v1-95M weights...")
    mert_path = download_model(
        "m-a-p/MERT-v1-95M",
        allow_patterns=["*.bin", "*.json"],
    )

    mert_bin = mert_path / "pytorch_model.bin"
    if not mert_bin.exists():
        raise FileNotFoundError("pytorch_model.bin not found in MERT-v1-95M download.")

    print("Loading MERT weights...")
    raw = load_pytorch_bin(mert_bin)

    mert_state: dict[str, np.ndarray] = {}
    for k, v in raw.items():
        new_k = _remap_mert_key(k)
        if new_k is not None:
            # Transpose Conv1d weights: (Out, In, K) -> (Out, K, In)
            if "conv.weight" in new_k and v.ndim == 3:
                v = np.transpose(v, (0, 2, 1))
            mert_state[new_k] = v

    _cast_dtype(dtype, mert_state)

    print(f"  MERT: {len(mert_state)} tensors")
    save_safetensors(mert_state, mert_output)


def _remap_mert_key(k: str) -> str | None:
    """Remap a MERT HuggingFace key to our MERTModel structure.

    HF MERT keys:
        hubert.feature_extractor.conv_layers.{i}.conv.weight
        hubert.feature_extractor.conv_layers.0.layer_norm.{weight/bias}
        hubert.feature_projection.layer_norm.{weight/bias}
        hubert.feature_projection.projection.{weight/bias}
        hubert.encoder.pos_conv_embed.conv.{weight/bias}
        hubert.encoder.layer_norm.{weight/bias}
        hubert.encoder.layers.{i}.attention.{q,k,v,out}_proj.{weight/bias}
        hubert.encoder.layers.{i}.layer_norm.{weight/bias}
        hubert.encoder.layers.{i}.feed_forward.intermediate_dense.{weight/bias}
        hubert.encoder.layers.{i}.feed_forward.output_dense.{weight/bias}
        hubert.encoder.layers.{i}.final_layer_norm.{weight/bias}

    Our MERTModel keys:
        feature_extractor.conv_layers.{i}.conv.weight
        feature_extractor.conv_layers.0.layer_norm.{weight/bias}
        feature_projection.layer_norm.{weight/bias}
        feature_projection.projection.{weight/bias}
        encoder.pos_conv_embed.conv.{weight/bias}
        encoder.layer_norm.{weight/bias}
        encoder.layers.{i}.attention.{q,k,v,out}_proj.{weight/bias}
        encoder.layers.{i}.layer_norm.{weight/bias}
        encoder.layers.{i}.feed_forward.intermediate_dense.{weight/bias}
        encoder.layers.{i}.feed_forward.output_dense.{weight/bias}
        encoder.layers.{i}.final_layer_norm.{weight/bias}
    """
    # Strip 'hubert.' prefix if present — our model doesn't have it
    if k.startswith("hubert."):
        return k[len("hubert.") :]

    # Accept keys already in our format (encoder.*, feature_*)
    if k.startswith(("encoder.", "feature_extractor.", "feature_projection.")):
        return k

    # Skip non-model keys (like masked_spec_embed, projector, etc.)
    return None


def _count_rvq_codebooks(raw_weights: dict, has_lm: bool) -> int:
    """Count RVQ codebook layers from weight keys.

    Audiocraft stores codebooks as rvq.vq.layers.{i}._codebook.embed.
    We look under the self_wav conditioner prefix for these keys.
    """
    max_idx = -1
    for k in raw_weights:
        if "self_wav.rvq.vq.layers." in k and "_codebook.embed" in k:
            # Extract layer index
            rest = k.split("self_wav.rvq.vq.layers.")[-1]
            idx = int(rest.split(".")[0])
            # Only count actual codebook embed (not embed_avg)
            if rest.endswith("_codebook.embed"):
                max_idx = max(max_idx, idx)
    return max_idx + 1 if max_idx >= 0 else 3  # Default to 3 if not found


def _save_style_configs(output_dir: Path, raw_weights: dict, repo_id: str) -> None:
    """Build and save config.json for a style variant.

    Since audiocraft-format models don't have a HF config.json, we infer
    the configuration from the weight shapes.
    """
    # Infer decoder config from weight shapes.
    # Audiocraft keys lack 'lm.' prefix; HF keys have it. Handle both.
    has_lm = any(k.startswith("lm.") for k in raw_weights)
    lm_prefix = "lm." if has_lm else ""

    hidden_size = 1024  # Default for musicgen-style (small)
    num_codebooks = 4
    ffn_dim = 4096

    # Try to infer from weights
    linear1_key = f"{lm_prefix}transformer.layers.0.linear1.weight"
    for k, v in raw_weights.items():
        if k == linear1_key:
            ffn_dim = v.shape[0]
            hidden_size = v.shape[1]
            break

    # Count decoder layers
    layer_prefix = f"{lm_prefix}transformer.layers."
    num_layers = 0
    for k in raw_weights:
        if layer_prefix in k:
            parts = k.split(layer_prefix)[-1]
            layer_idx = int(parts.split(".")[0])
            num_layers = max(num_layers, layer_idx + 1)

    # Count codebooks from embedding count
    emb_prefix = f"{lm_prefix}emb."
    for k in raw_weights:
        if emb_prefix in k:
            idx = int(k.split(emb_prefix)[-1].split(".")[0])
            num_codebooks = max(num_codebooks, idx + 1)

    config = {
        "model_type": "musicgen_style",
        "is_style": True,
        "decoder": {
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": hidden_size // 64,  # 64 head_dim is standard
            "ffn_dim": ffn_dim,
            "num_codebooks": num_codebooks,
            "vocab_size": 2048,
            "bos_token_id": 2048,  # nosec B105 — token ID, not a password
            "pad_token_id": 2048,  # nosec B105 — token ID, not a password
        },
        "audio_encoder": {
            "codebook_size": 2048,
            "sampling_rate": 32000,
            "upsampling_ratios": [8, 5, 4, 4],
        },
        "text_encoder": {
            "d_model": 768,
            "num_heads": 12,
            "d_kv": 64,
            "d_ff": 3072,
            "num_layers": 12,
            "vocab_size": 32128,
        },
        # Style-specific config (inferred from weights where possible)
        "style_dim": 512,
        "style_num_heads": 8,
        "style_num_layers": 8,
        "style_ffn_dim": 2048,
        "style_ds_factor": 15,
        "style_n_q": _count_rvq_codebooks(raw_weights, has_lm),
        "style_bins": 1024,
        "style_excerpt_length": 3.0,
        "style_output_dim": hidden_size,  # Output projection to decoder hidden
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # T5 config
    t5_config = config["text_encoder"]
    with open(output_dir / "t5_config.json", "w") as f:
        json.dump(t5_config, f, indent=2)


def _save_configs(output_dir: Path, hf_config: dict, repo_id: str = "") -> None:
    """Write config.json and t5_config.json from HF's config."""
    # Detect melody variant and annotate config
    model_type = hf_config.get("model_type", "")
    if "melody" in model_type or "melody" in repo_id.lower():
        hf_config.setdefault("is_melody", True)
        hf_config.setdefault("num_chroma", 12)
        hf_config.setdefault("chroma_length", 235)

    # Save the full HF config as our config (from_dict handles the nesting)
    with open(output_dir / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    # Extract T5 config separately for easy loading
    t5_section = hf_config.get("text_encoder", {})
    t5_config = {
        "d_model": t5_section.get("d_model", 768),
        "num_heads": t5_section.get("num_heads", 12),
        "d_kv": t5_section.get("d_kv", 64),
        "d_ff": t5_section.get("d_ff", 3072),
        "num_layers": t5_section.get("num_layers", 12),
        "vocab_size": t5_section.get("vocab_size", 32128),
        "relative_attention_num_buckets": t5_section.get(
            "relative_attention_num_buckets", 32
        ),
        "relative_attention_max_distance": t5_section.get(
            "relative_attention_max_distance", 128
        ),
    }
    with open(output_dir / "t5_config.json", "w") as f:
        json.dump(t5_config, f, indent=2)


def _save_tokenizer(output_dir: Path, repo_id: str) -> None:
    """Download and save the T5 tokenizer alongside the weights."""
    from transformers import AutoTokenizer

    # Check if already saved
    if (output_dir / "tokenizer_config.json").exists():
        return

    print("Downloading T5 tokenizer...")
    # MusicGen uses the t5-base tokenizer
    for source in [repo_id, "google-t5/t5-base"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(source)  # nosec B615 — known HF repo
            tokenizer.save_pretrained(str(output_dir))
            print(f"Saved tokenizer to {output_dir}")
            return
        except (OSError, ValueError, KeyError) as e:
            print(f"  Could not load tokenizer from {source}: {e}")
            continue

    print("Warning: Could not save tokenizer. It will be downloaded at runtime.")
