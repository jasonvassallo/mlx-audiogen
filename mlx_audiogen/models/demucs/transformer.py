"""Cross-domain transformer for HTDemucs.

Implements the bidirectional cross-attention bridge between the
spectral and temporal branches at the U-Net bottleneck.
"""

import math

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------


def create_sin_embedding(
    length: int,
    dim: int,
    max_period: float = 10000.0,
) -> mx.array:
    """1-D sinusoidal positional embedding, shape ``(1, length, dim)``."""
    if dim % 2 != 0:
        msg = f"Embedding dim must be even, got {dim}"
        raise ValueError(msg)
    pos = mx.arange(length)[:, None]  # (L, 1)
    half = dim // 2
    adim = mx.arange(half)[None, :]  # (1, D/2)
    phase = pos / (max_period ** (adim / max(half - 1, 1)))
    emb = mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)  # (L, D)
    return emb[None]  # (1, L, D)


def create_2d_sin_embedding(
    d_model: int,
    height: int,
    width: int,
    max_period: float = 10000.0,
) -> mx.array:
    """2-D sinusoidal positional embedding, shape ``(1, d_model, height, width)``.

    Height encodes frequency, width encodes time.
    """
    if d_model % 4 != 0:
        msg = f"Cannot use sin/cos 2D embedding with dim={d_model} (need multiple of 4)"
        raise ValueError(msg)

    quarter = d_model // 4
    div_term = mx.exp(mx.arange(0, quarter) * -(math.log(max_period) / quarter))

    pos_w = mx.arange(width)[:, None]  # (W, 1)
    pos_h = mx.arange(height)[:, None]  # (H, 1)

    # Width (time) embeddings → first half of channels
    sin_w = mx.sin(pos_w * div_term)  # (W, quarter)
    cos_w = mx.cos(pos_w * div_term)  # (W, quarter)
    # Height (freq) embeddings → second half of channels
    sin_h = mx.sin(pos_h * div_term)  # (H, quarter)
    cos_h = mx.cos(pos_h * div_term)  # (H, quarter)

    # Build (d_model, H, W) by concatenating width and height embeddings
    # Width (time): sin_w (W, quarter) → broadcast to (quarter, H, W)
    w_sin = mx.broadcast_to(sin_w.T[:, None, :], (quarter, height, width))
    w_cos = mx.broadcast_to(cos_w.T[:, None, :], (quarter, height, width))
    # Height (freq): sin_h (H, quarter) → broadcast to (quarter, H, W)
    h_sin = mx.broadcast_to(sin_h.T[:, :, None], (quarter, height, width))
    h_cos = mx.broadcast_to(cos_h.T[:, :, None], (quarter, height, width))
    pe = mx.concatenate([w_sin, w_cos, h_sin, h_cos], axis=0)  # (d_model, H, W)
    return pe[None]  # (1, d_model, H, W)


# ---------------------------------------------------------------------------
# Layer Scale (channel-last variant for transformer)
# ---------------------------------------------------------------------------


class LayerScale(nn.Module):
    """Per-channel residual scale (channel-last tensors)."""

    def __init__(self, channels: int, init: float = 1e-4):
        super().__init__()
        self.scale = mx.full((channels,), init)

    def __call__(self, x: mx.array) -> mx.array:
        return self.scale * x  # broadcast over (B, T, C)


# ---------------------------------------------------------------------------
# Self-Attention layer
# ---------------------------------------------------------------------------


class SelfAttentionLayer(nn.Module):
    """Pre-norm self-attention + FFN with LayerScale."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        gelu: bool = True,
        layer_scale: bool = True,
        norm_out: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Separate Q, K, V projections (converted from fused in_proj_weight)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU() if gelu else nn.ReLU()

        self.gamma_1 = LayerScale(dim) if layer_scale else nn.Identity()
        self.gamma_2 = LayerScale(dim) if layer_scale else nn.Identity()
        self.norm_out_layer = nn.GroupNorm(1, dim) if norm_out else None

    def _attn(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, T, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.head_dim)
        )
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, T, C))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.gamma_1(self._attn(self.norm1(x)))
        x = x + self.gamma_2(self.linear2(self.act(self.linear1(self.norm2(x)))))
        if self.norm_out_layer is not None:
            # GroupNorm(1, C) on (B, T, C) — channels already last
            x = self.norm_out_layer(x)
        return x


# ---------------------------------------------------------------------------
# Cross-Attention layer
# ---------------------------------------------------------------------------


class CrossAttentionLayer(nn.Module):
    """Pre-norm cross-attention + FFN with LayerScale."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int,
        gelu: bool = True,
        layer_scale: bool = True,
        norm_out: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim)  # for query
        self.norm2 = nn.LayerNorm(dim)  # for key/value
        self.norm3 = nn.LayerNorm(dim)  # for FFN

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU() if gelu else nn.ReLU()

        self.gamma_1 = LayerScale(dim) if layer_scale else nn.Identity()
        self.gamma_2 = LayerScale(dim) if layer_scale else nn.Identity()
        self.norm_out_layer = nn.GroupNorm(1, dim) if norm_out else None

    def _cross_attn(self, q: mx.array, kv: mx.array) -> mx.array:
        B, Tq, C = q.shape
        Tkv = kv.shape[1]
        q_ = (
            self.q_proj(q)
            .reshape(B, Tq, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k_ = (
            self.k_proj(kv)
            .reshape(B, Tkv, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v_ = (
            self.v_proj(kv)
            .reshape(B, Tkv, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        out = mx.fast.scaled_dot_product_attention(
            q_, k_, v_, scale=1.0 / math.sqrt(self.head_dim)
        )
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(B, Tq, C))

    def __call__(self, q: mx.array, kv: mx.array) -> mx.array:
        x = q + self.gamma_1(self._cross_attn(self.norm1(q), self.norm2(kv)))
        x = x + self.gamma_2(self.linear2(self.act(self.linear1(self.norm3(x)))))
        if self.norm_out_layer is not None:
            x = self.norm_out_layer(x)
        return x


# ---------------------------------------------------------------------------
# CrossTransformerEncoder
# ---------------------------------------------------------------------------


class CrossTransformerEncoder(nn.Module):
    """Bidirectional cross-attention bridge between spectral and temporal branches."""

    def __init__(
        self,
        dim: int,
        num_layers: int = 5,
        num_heads: int = 8,
        hidden_scale: float = 4.0,
        cross_first: bool = False,
        norm_in: bool = True,
        norm_out: bool = True,
        layer_scale: bool = True,
        gelu: bool = True,
        weight_pos_embed: float = 1.0,
        max_period: float = 10000.0,
    ):
        super().__init__()
        hidden_dim = int(dim * hidden_scale)
        self.num_layers = num_layers
        self.classic_parity = 1 if cross_first else 0
        self.weight_pos_embed = weight_pos_embed
        self.max_period = max_period

        self.norm_in = nn.LayerNorm(dim) if norm_in else nn.Identity()
        self.norm_in_t = nn.LayerNorm(dim) if norm_in else nn.Identity()

        kw = dict(
            dim=dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            gelu=gelu,
            layer_scale=layer_scale,
            norm_out=norm_out,
        )

        self.layers: list[nn.Module] = []
        self.layers_t: list[nn.Module] = []
        for idx in range(num_layers):
            if idx % 2 == self.classic_parity:
                self.layers.append(SelfAttentionLayer(**kw))  # type: ignore[arg-type]
                self.layers_t.append(SelfAttentionLayer(**kw))  # type: ignore[arg-type]
            else:
                self.layers.append(CrossAttentionLayer(**kw))  # type: ignore[arg-type]
                self.layers_t.append(CrossAttentionLayer(**kw))  # type: ignore[arg-type]

    def __call__(self, x: mx.array, xt: mx.array) -> tuple[mx.array, mx.array]:
        """
        Args:
            x: Spectral features ``(B, C, Fr, T1)`` in NCHW.
            xt: Temporal features ``(B, C, T2)`` in NCL.

        Returns:
            Processed ``(x, xt)`` with same shapes.
        """
        B, C, Fr, T1 = x.shape

        # 2-D positional embedding for spectral branch
        pos_2d = create_2d_sin_embedding(C, Fr, T1, self.max_period)
        # Flatten (B, C, Fr, T1) → (B, T1*Fr, C)
        pos_2d = pos_2d.transpose(0, 3, 2, 1).reshape(1, T1 * Fr, C)
        x = x.transpose(0, 3, 2, 1).reshape(B, T1 * Fr, C)
        x = self.norm_in(x) + self.weight_pos_embed * pos_2d

        # 1-D positional embedding for temporal branch
        T2 = xt.shape[-1]
        pos_1d = create_sin_embedding(T2, C, self.max_period)  # (1, T2, C)
        xt = xt.transpose(0, 2, 1)  # (B, T2, C)
        xt = self.norm_in_t(xt) + self.weight_pos_embed * pos_1d

        for idx in range(self.num_layers):
            if idx % 2 == self.classic_parity:
                x = self.layers[idx](x)  # type: ignore[call-arg]
                xt = self.layers_t[idx](xt)  # type: ignore[call-arg]
            else:
                old_x = x
                x = self.layers[idx](x, xt)  # type: ignore[call-arg]
                xt = self.layers_t[idx](xt, old_x)  # type: ignore[call-arg]

        # Reshape back
        x = x.reshape(B, T1, Fr, C).transpose(0, 3, 2, 1)  # (B, C, Fr, T1)
        xt = xt.transpose(0, 2, 1)  # (B, C, T2)
        return x, xt
