"""Convert HTDemucs weights from PyTorch (.th) to MLX safetensors.

Requires ``torch`` (install via ``uv sync --extra convert``).

Downloads the pretrained checkpoint from Meta's servers, remaps weight
keys, transposes convolution weights, splits fused QKV projections,
and saves as safetensors + config.json.
"""

import json
import logging
import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from ...shared.hub import save_safetensors

logger = logging.getLogger(__name__)

# Canonical download URLs
_WEIGHT_URLS: dict[str, str] = {
    "htdemucs": (
        "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"
    ),
    "htdemucs_6s": (
        "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th"
    ),
}


def _download(url: str, dest: Path) -> Path:
    """Download a file with progress logging."""
    if dest.exists():
        logger.info("Using cached %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s", url, dest)
    urlretrieve(url, dest)  # noqa: S310  # nosec B310 — trusted Meta URL
    return dest


def _is_conv_weight(key: str) -> bool:
    """Check whether a key corresponds to a Conv weight that needs transposition."""
    if not key.endswith(".weight"):
        return False
    # Conv / ConvTranspose weights in encoder, decoder, tencoder, tdecoder, dconv
    for prefix in ("encoder.", "decoder.", "tencoder.", "tdecoder."):
        if key.startswith(prefix):
            # conv.weight, conv_tr.weight, rewrite.weight, dconv convs
            part = key[len(prefix) :]
            if ".conv.weight" in part or ".conv_tr.weight" in part:
                return True
            if ".rewrite.weight" in part:
                return True
            if ".dconv." in part and ".weight" in part:
                # DConv layers: 0.weight (Conv1d), 3.weight (Conv1d)
                # But NOT norm weights (1.weight, 4.weight) or scale (6.scale)
                segments = part.split(".")
                for i, seg in enumerate(segments):
                    if seg in ("0", "3") and i > 0 and segments[i - 1].isdigit():
                        return True
    return False


def _is_conv_transpose(key: str) -> bool:
    return ".conv_tr.weight" in key


def _is_conv2d(key: str, state: dict) -> bool:
    """Check if a weight is Conv2d (4-D) vs Conv1d (3-D)."""
    return state[key].ndim == 4


def _transpose_conv(w: np.ndarray, is_transpose: bool) -> np.ndarray:
    """Transpose conv weight from PyTorch to MLX layout."""
    if w.ndim == 3:
        if is_transpose:
            # ConvTranspose1d: (C_in, C_out, K) → (C_out, K, C_in)
            return np.transpose(w, (1, 2, 0))
        else:
            # Conv1d: (C_out, C_in, K) → (C_out, K, C_in)
            return np.transpose(w, (0, 2, 1))
    elif w.ndim == 4:
        if is_transpose:
            # ConvTranspose2d: (C_in, C_out, kH, kW) → (C_out, kH, kW, C_in)
            return np.transpose(w, (1, 2, 3, 0))
        else:
            # Conv2d: (C_out, C_in, kH, kW) → (C_out, kH, kW, C_in)
            return np.transpose(w, (0, 2, 3, 1))
    return w


def _split_fused_qkv(
    state: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Split fused ``in_proj_weight`` / ``in_proj_bias`` into separate Q, K, V."""
    new_state: dict[str, np.ndarray] = {}
    for key, val in state.items():
        if "in_proj_weight" in key:
            prefix = key.replace("in_proj_weight", "")
            dim = val.shape[0] // 3
            new_state[prefix + "q_proj.weight"] = val[:dim]
            new_state[prefix + "k_proj.weight"] = val[dim : 2 * dim]
            new_state[prefix + "v_proj.weight"] = val[2 * dim :]
        elif "in_proj_bias" in key:
            prefix = key.replace("in_proj_bias", "")
            dim = val.shape[0] // 3
            new_state[prefix + "q_proj.bias"] = val[:dim]
            new_state[prefix + "k_proj.bias"] = val[dim : 2 * dim]
            new_state[prefix + "v_proj.bias"] = val[2 * dim :]
        else:
            new_state[key] = val
    return new_state


def _remap_transformer_keys(key: str) -> str:
    """Remap PyTorch transformer key names to our module names."""
    # Self-attention layers use nn.TransformerEncoderLayer internals
    key = key.replace(".self_attn.", ".")

    # Cross-attention layers
    key = key.replace(".cross_attn.", ".")

    # DConv block renaming
    key = key.replace(".dconv.", ".dconv_block.")

    # Rewrite conv renaming
    key = key.replace(".rewrite.", ".rewrite_conv.")

    # Norm naming: encoder/decoder use norm_enabled → norm1, norm2
    # but we keep the same names (norm1, norm2) so no change needed

    # LayerScale: gamma_1.scale, gamma_2.scale stay the same

    # norm_out in transformer
    key = key.replace(".norm_out.", ".norm_out_layer.gn.")

    return key


def _remap_dconv_keys(key: str) -> str:
    """Remap DConv sequential indices to named attributes.

    PyTorch: dconv_block.layers.{d}.{0,1,2,3,4,5,6}
    MLX:     dconv_block.layers.{d}.{conv1,norm1,conv2,norm2,layer_scale}
    """
    import re

    # Match dconv_block.layers.{d}.{idx}.{param}
    m = re.match(
        r"(.*\.dconv_block\.layers\.\d+)\."
        r"(\d+)\.(.*)",
        key,
    )
    if not m:
        return key
    prefix, idx_str, param = m.groups()
    idx = int(idx_str)
    name_map = {
        0: "conv1",
        1: "norm1",
        # 2 is GELU (no params)
        3: "conv2",
        4: "norm2",
        # 5 is GLU (no params)
        6: "layer_scale",
    }
    if idx in name_map:
        return f"{prefix}.{name_map[idx]}.{param}"
    return key


def convert_demucs(
    output_dir: str,
    variant: str = "htdemucs",
) -> None:
    """Convert HTDemucs weights to MLX format.

    Args:
        output_dir: Directory to save converted weights.
        variant: Model variant (``htdemucs`` or ``htdemucs_6s``).
    """
    import torch

    if variant not in _WEIGHT_URLS:
        msg = f"Unknown variant {variant!r}. Available: {list(_WEIGHT_URLS)}"
        raise ValueError(msg)

    url = _WEIGHT_URLS[variant]
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache_dir = cache_dir / "mlx-audiogen" / "demucs"
    filename = url.split("/")[-1]
    th_path = _download(url, cache_dir / filename)

    logger.info("Loading PyTorch checkpoint: %s", th_path)
    pkg = torch.load(th_path, map_location="cpu", weights_only=False)  # nosec B614 — trusted Meta checkpoint

    # Extract state dict and kwargs
    state_dict = pkg["state"]
    kwargs = pkg.get("kwargs", {})

    # Build config
    sources = pkg.get("args", [["drums", "bass", "other", "vocals"]])[0]
    config_dict = {
        "sources": sources,
        "audio_channels": kwargs.get("audio_channels", 2),
        "channels": kwargs.get("channels", 48),
        "growth": kwargs.get("growth", 2),
        "depth": kwargs.get("depth", 4),
        "nfft": kwargs.get("nfft", 4096),
        "kernel_size": kwargs.get("kernel_size", 8),
        "stride": kwargs.get("stride", 4),
        "time_stride": kwargs.get("time_stride", 2),
        "cac": kwargs.get("cac", True),
        "freq_emb": kwargs.get("freq_emb", 0.2),
        "emb_scale": kwargs.get("emb_scale", 10),
        "emb_smooth": kwargs.get("emb_smooth", True),
        "norm_starts": kwargs.get("norm_starts", 4),
        "norm_groups": kwargs.get("norm_groups", 4),
        "dconv_mode": kwargs.get("dconv_mode", 1),
        "dconv_depth": kwargs.get("dconv_depth", 2),
        "dconv_comp": kwargs.get("dconv_comp", 8),
        "dconv_init": kwargs.get("dconv_init", 1e-3),
        "rewrite": kwargs.get("rewrite", True),
        "context": kwargs.get("context", 1),
        "context_enc": kwargs.get("context_enc", 0),
        "bottom_channels": kwargs.get("bottom_channels", 0),
        "t_layers": kwargs.get("t_layers", 5),
        "t_heads": kwargs.get("t_heads", 8),
        "t_hidden_scale": kwargs.get("t_hidden_scale", 4.0),
        "t_emb": kwargs.get("t_emb", "sin"),
        "t_norm_first": kwargs.get("t_norm_first", True),
        "t_norm_in": kwargs.get("t_norm_in", True),
        "t_norm_out": kwargs.get("t_norm_out", True),
        "t_layer_scale": kwargs.get("t_layer_scale", True),
        "t_gelu": kwargs.get("t_gelu", True),
        "t_weight_pos_embed": kwargs.get("t_weight_pos_embed", 1.0),
        "t_max_period": kwargs.get("t_max_period", 10000.0),
        "t_cross_first": kwargs.get("t_cross_first", False),
        "samplerate": kwargs.get("samplerate", 44100),
        "segment": kwargs.get("segment", 10),
        "use_train_segment": kwargs.get("use_train_segment", True),
    }

    # Convert tensors to numpy
    np_state: dict[str, np.ndarray] = {}
    for k, v in state_dict.items():
        np_state[k] = v.float().numpy()

    # Transpose conv weights
    for key in list(np_state):
        if _is_conv_weight(key):
            is_tr = _is_conv_transpose(key)
            np_state[key] = _transpose_conv(np_state[key], is_tr)

    # Split fused QKV weights
    np_state = _split_fused_qkv(np_state)

    # Remap key names to match MLX module structure
    remapped: dict[str, np.ndarray] = {}
    for key, val in np_state.items():
        new_key = _remap_transformer_keys(key)
        new_key = _remap_dconv_keys(new_key)
        remapped[new_key] = val

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_safetensors(remapped, str(out_path / "model.safetensors"))

    with open(out_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    logger.info("Converted %d weights → %s", len(remapped), out_path)
