"""MusicGen autoregressive transformer decoder with KV cache.

Architecture:
    - Pre-norm transformer blocks with self-attention + cross-attention + FFN
    - Sinusoidal positional embeddings (not learned, not RoPE)
    - Separate Q/K/V projections (no bias on attention, bias on FFN)
    - GELU activation in feed-forward network
    - KV cache for efficient autoregressive generation

Weight key alignment with HuggingFace safetensors:
    Our attribute names are chosen to match HF key structure after prefix stripping,
    minimizing remapping needed during weight conversion.

Ported from Apple's mlx-examples/musicgen (Apache 2.0 License).
"""

from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class KVCache:
    """Pre-allocated key-value cache with step-based growth.

    Instead of concatenating on every token (O(n²) copies), we pre-allocate
    in chunks of ``step`` tokens and write into the buffer. This is the
    standard MLX pattern for efficient autoregressive generation.
    """

    def __init__(self, head_dim: int, n_kv_heads: int, step: int = 256):
        self.n_kv_heads = n_kv_heads
        if isinstance(head_dim, int):
            self.k_head_dim = self.v_head_dim = head_dim
        elif isinstance(head_dim, tuple) and len(head_dim) == 2:
            self.k_head_dim, self.v_head_dim = head_dim
        else:
            raise ValueError("head_dim must be an int or a tuple of two ints")
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset = 0
        self.step = step

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Append new keys/values to cache, growing buffer as needed.

        Args:
            keys: Shape (B, n_heads, new_seq_len, head_dim)
            values: Shape (B, n_heads, new_seq_len, head_dim)

        Returns:
            Tuple of (all_keys, all_values) up to current offset.
        """
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B = keys.shape[0]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, self.n_kv_heads, n_steps * self.step, self.k_head_dim)
            v_shape = (B, self.n_kv_heads, n_steps * self.step, self.v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]  # type: ignore[index]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)  # type: ignore[list-item]
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys  # type: ignore[index]
        self.values[..., prev : self.offset, :] = values  # type: ignore[index]
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]  # type: ignore[index]

    @property
    def state(self):
        return self.keys, self.values


class CrossAttentionKVCache:
    """Cache for cross-attention K/V computed from static conditioning.

    Unlike self-attention KV cache which grows per step, cross-attention
    K/V are computed once from the T5 conditioning output (which is static
    during generation) and reused for all subsequent steps. This avoids
    redundant K/V projection computations: for a 24-layer model with 250
    generation steps, this saves 24 × 249 = 5,976 matrix multiplications.
    """

    def __init__(self):
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None

    @property
    def is_populated(self) -> bool:
        return self.keys is not None


class MultiHeadAttention(nn.Module):
    """Multi-head attention with separate Q/K/V projections.

    Uses ``mx.fast.scaled_dot_product_attention`` for Metal-optimized computation.
    Attention projections have no bias (matching MusicGen's architecture).

    Supports an optional ``cross_kv_cache`` for cross-attention: when provided,
    K/V are computed and cached on the first call, then reused on all subsequent
    calls (since the conditioning tensor is static during generation).
    """

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        head_dim = dim // n_heads
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        cross_kv_cache: Optional[CrossAttentionKVCache] = None,
    ) -> mx.array:
        B, L_q, _ = queries.shape

        queries = self.q_proj(queries)
        queries = queries.reshape(B, L_q, self.n_heads, -1).transpose(0, 2, 1, 3)

        # Cross-attention K/V caching: reuse pre-computed K/V if available
        if cross_kv_cache is not None and cross_kv_cache.is_populated:
            assert cross_kv_cache.keys is not None  # guaranteed by is_populated
            assert cross_kv_cache.values is not None
            keys = cross_kv_cache.keys
            values = cross_kv_cache.values
        else:
            L_k = keys.shape[1]
            keys = self.k_proj(keys)
            values = self.v_proj(values)
            keys = keys.reshape(B, L_k, self.n_heads, -1).transpose(0, 2, 1, 3)
            values = values.reshape(B, L_k, self.n_heads, -1).transpose(0, 2, 1, 3)
            if cross_kv_cache is not None:
                cross_kv_cache.keys = keys
                cross_kv_cache.values = values

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L_q, -1)
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: self-attn → cross-attn → FFN.

    Attribute names match HF weight keys after prefix stripping:
        - self_attn / self_attn_layer_norm
        - encoder_attn / encoder_attn_layer_norm  (cross-attention)
        - fc1 / fc2 / final_layer_norm
    """

    def __init__(self, hidden_size: int, num_attention_heads: int, ffn_dim: int):
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(hidden_size, num_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

        # Cross-attention to T5 conditioning
        self.encoder_attn = MultiHeadAttention(hidden_size, num_attention_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

        # Feed-forward with GELU (no bias — matches HF MusicGen weights)
        self.fc1 = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, hidden_size, bias=False)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

    def __call__(
        self,
        x: mx.array,
        conditioning: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
        cross_kv_cache: Optional[CrossAttentionKVCache] = None,
    ) -> mx.array:
        # Self-attention with pre-norm
        xn = self.self_attn_layer_norm(x)
        x = x + self.self_attn(xn, xn, xn, mask, cache)

        # Cross-attention with pre-norm (no mask — attend to all conditioning tokens)
        xn = self.encoder_attn_layer_norm(x)
        x = x + self.encoder_attn(
            xn, conditioning, conditioning, cross_kv_cache=cross_kv_cache
        )

        # FFN with pre-norm + GELU
        xn = self.final_layer_norm(x)
        x = x + self.fc2(nn.gelu(self.fc1(xn)))

        return x


def create_sin_embedding(positions: mx.array, dim: int, max_period: float = 10000):
    """Create sinusoidal positional embeddings.

    MusicGen uses non-learned sinusoidal embeddings (not RoPE, not learned).
    Each position gets a dim-dimensional vector of interleaved cos/sin features.

    Args:
        positions: Position indices, shape (1, seq_len, 1) or broadcastable.
        dim: Embedding dimension (must be even).
        max_period: Maximum period for the sinusoidal functions.

    Returns:
        Embeddings of shape matching positions + (dim,).
    """
    if dim % 2 != 0:
        raise ValueError(f"Embedding dimension must be even, got {dim}")
    half_dim = dim // 2
    adim = mx.arange(half_dim).reshape(1, 1, -1)
    phase = positions / (max_period ** (adim / (half_dim - 1)))
    return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def top_k_sampling(
    logits: mx.array, top_k: float, temperature: float, axis: int = -1
) -> mx.array:
    """Top-k sampling with temperature scaling.

    Compiled with ``mx.compile`` for optimal Metal performance. State-aware
    for ``mx.random`` to ensure reproducible sampling across calls.

    Args:
        logits: Raw model output logits.
        top_k: Number of top candidates to sample from.
        temperature: Softmax temperature (higher = more random).
        axis: Axis to sample along.

    Returns:
        Sampled token indices.
    """
    probs = mx.softmax(logits * (1 / temperature), axis=axis)

    # Sort in ascending order, threshold at top-k
    sorted_indices = mx.argsort(probs, axis=axis)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=axis)
    prob_threshold = mx.take(sorted_probs, mx.array(-top_k), axis=axis)

    # Zero out everything below the threshold
    top_probs = mx.where(sorted_probs > prob_threshold, sorted_probs, 0)

    # Sample from filtered distribution
    sorted_token = mx.random.categorical(mx.log(top_probs), axis=axis)
    token = mx.take_along_axis(
        sorted_indices, mx.expand_dims(sorted_token, axis), axis=axis
    )
    return token
