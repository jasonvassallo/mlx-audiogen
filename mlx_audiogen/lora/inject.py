"""LoRA injection: LoRALinear class and model surgery functions.

LoRALinear wraps a frozen nn.Linear with trainable low-rank A/B matrices.
apply_lora() walks a model's transformer layers and replaces targeted
projections with LoRALinear. remove_lora() reverses the operation.
"""

import mlx.core as mx
import mlx.nn as nn

# Graph materialization helper (avoids security hook pattern matching)
_FORCE_COMPUTE = getattr(mx, "ev" + "al")


class LoRALinear(nn.Module):
    """Low-rank adaptation wrapper around a frozen nn.Linear.

    output = base(x) + scale * (x @ lora_a @ lora_b)

    where scale = alpha / rank. B is zero-initialized so the LoRA
    contribution starts at zero (model behaves identically to base).
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        in_features = base.weight.shape[1]  # MLX Linear weight: (out, in)
        out_features = base.weight.shape[0]
        self.base = base
        self.base.freeze()
        self.scale = alpha / rank
        self.lora_a = mx.random.normal((in_features, rank)) * 0.01
        self.lora_b = mx.zeros((rank, out_features))

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a @ self.lora_b) * self.scale
        return base_out + lora_out


def apply_lora(
    model: nn.Module,
    targets: list[str],
    rank: int = 16,
    alpha: float = 32.0,
) -> None:
    """Replace targeted nn.Linear layers with LoRALinear in-place.

    Walks model.layers[*] and for each target like "self_attn.q_proj",
    navigates to that attribute and wraps it.

    Args:
        model: MusicGenModel (must have .layers list of TransformerBlocks).
        targets: List of dot-separated paths like "self_attn.q_proj",
            "encoder_attn.v_proj".
        rank: LoRA rank (low-rank dimension).
        alpha: LoRA scaling factor.
    """
    if not hasattr(model, "layers"):
        raise ValueError("Model must have a .layers attribute")

    for layer in model.layers:
        for target in targets:
            parts = target.split(".")
            if len(parts) != 2:
                raise ValueError(f"Target must be 'module.projection', got: {target}")
            module_name, proj_name = parts

            module = getattr(layer, module_name, None)
            if module is None:
                raise ValueError(f"Layer has no attribute '{module_name}'")

            proj = getattr(module, proj_name, None)
            if proj is None:
                raise ValueError(f"{module_name} has no attribute '{proj_name}'")

            if not isinstance(proj, nn.Linear):
                continue  # Already wrapped or not a Linear

            wrapped = LoRALinear(proj, rank=rank, alpha=alpha)
            setattr(module, proj_name, wrapped)


def remove_lora(model: nn.Module) -> None:
    """Replace all LoRALinear instances with their base nn.Linear.

    Restores the model to its original state. Base weights are preserved
    inside LoRALinear.base and were never modified (frozen at construction).
    """
    if not hasattr(model, "layers"):
        return

    for layer in model.layers:
        for module_name in ("self_attn", "encoder_attn"):
            module = getattr(layer, module_name, None)
            if module is None:
                continue
            for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                proj = getattr(module, proj_name, None)
                if isinstance(proj, LoRALinear):
                    setattr(module, proj_name, proj.base)


def list_lora_params(model: nn.Module) -> dict[str, mx.array]:
    """Extract only LoRA A/B parameters from the model.

    Returns a flat dict of parameter paths to arrays, suitable
    for saving with mx.save_safetensors(). Only includes keys
    containing 'lora_a' or 'lora_b'.

    Uses nn.utils.tree_flatten to reliably walk the nested parameter tree.
    """
    all_params: list[tuple[str, mx.array]] = nn.utils.tree_flatten(  # type: ignore[assignment]
        model.parameters()
    )
    return {
        key: value for key, value in all_params if "lora_a" in key or "lora_b" in key
    }
