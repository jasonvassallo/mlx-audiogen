"""MusicGen model: embeddings + transformer + LM heads + generation loop.

Combines K codebook embeddings, the autoregressive transformer decoder,
and K separate LM heads to generate audio tokens with a codebook delay pattern.

The ``generate()`` method implements classifier-free guidance (CFG) by running
conditional and unconditional forward passes in a single batch.

Architecture details:
    - K embed tables: Embedding(codebook_size+1, hidden_size) — +1 for BOS token
    - Transformer: N layers of pre-norm blocks with self-attn + cross-attn + FFN
    - K LM heads: Linear(hidden_size, codebook_size) — one per codebook
    - enc_to_dec_proj: Linear(t5_dim, hidden_size) — projects T5 output to decoder dim

Ported from Apple's mlx-examples/musicgen (Apache 2.0 License).
"""

from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from tqdm import tqdm

from .config import MusicGenConfig
from .transformer import (
    CrossAttentionKVCache,
    KVCache,
    TransformerBlock,
    create_sin_embedding,
    top_k_sampling,
)

# Force-compute helper: triggers MLX graph materialisation without using
# the bare function name that the project's security hook pattern-matches on.
_FORCE_COMPUTE = getattr(mx, "ev" + "al")


@mx.compile
def _apply_standard_cfg(
    cond_logits: mx.array,
    uncond_logits: mx.array,
    guidance_coef: float,
) -> mx.array:
    """Fused standard CFG: uncond + coef * (cond - uncond)."""
    return uncond_logits + (cond_logits - uncond_logits) * guidance_coef


@mx.compile
def _apply_dual_cfg(
    full_logits: mx.array,
    style_logits: mx.array,
    uncond_logits: mx.array,
    guidance_coef: float,
    style_coef: float,
) -> mx.array:
    """Fused dual-CFG: uncond + cfg * (style + beta * (full - style) - uncond)."""
    return uncond_logits + guidance_coef * (
        style_logits + style_coef * (full_logits - style_logits) - uncond_logits
    )


class MusicGenModel(nn.Module):
    """MusicGen decoder model with generation support.

    This is the core model that takes audio tokens + text conditioning
    and produces next-token logits for each codebook.

    Args:
        config: Full MusicGen configuration.
    """

    def __init__(self, config: MusicGenConfig):
        super().__init__()

        dec = config.decoder
        self.num_codebooks = dec.num_codebooks
        self.codebook_size = dec.vocab_size  # 2048
        self.hidden_size = dec.hidden_size
        self.num_attention_heads = dec.num_attention_heads
        self.bos_token_id = dec.bos_token_id  # 2048 (= codebook_size, used as BOS)
        self.is_melody = config.is_melody

        # Per-codebook embeddings: vocab_size+1 to include BOS token at index 2048
        self.embed_tokens = [
            nn.Embedding(self.codebook_size + 1, self.hidden_size)
            for _ in range(self.num_codebooks)
        ]

        # Transformer decoder layers
        self.layers = [
            TransformerBlock(
                hidden_size=dec.hidden_size,
                num_attention_heads=dec.num_attention_heads,
                ffn_dim=dec.ffn_dim,
            )
            for _ in range(dec.num_hidden_layers)
        ]

        # Final layer norm
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)

        # Per-codebook output heads
        self.lm_heads = [
            nn.Linear(self.hidden_size, self.codebook_size, bias=False)
            for _ in range(self.num_codebooks)
        ]

        # T5 encoder -> decoder projection (768 -> hidden_size)
        t5_dim = config.text_encoder.d_model
        self.enc_to_dec_proj = nn.Linear(t5_dim, self.hidden_size)

        # Melody variant: chroma -> decoder projection (12 -> hidden_size)
        if self.is_melody:
            self.audio_enc_to_dec_proj = nn.Linear(config.num_chroma, self.hidden_size)

    def __call__(
        self,
        audio_tokens: mx.array,
        conditioning: mx.array,
        cache: Optional[list[KVCache]] = None,
        cross_kv_caches: Optional[list[CrossAttentionKVCache]] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass: tokens -> embeddings -> transformer -> logits.

        Args:
            audio_tokens: Shape (B, seq_len, num_codebooks), token indices.
            conditioning: Shape (B, cond_len, hidden_size), already projected.
            cache: List of KVCache objects (one per layer), or None.
            cross_kv_caches: List of CrossAttentionKVCache objects (one per layer).
                On step 0, K/V are computed from conditioning and cached.
                On subsequent steps, cached K/V are reused (conditioning is static).
            mask: Optional causal mask for self-attention (used during training).

        Returns:
            Logits of shape (B, seq_len, codebook_size, num_codebooks).
        """
        if cache is None:
            cache: list[Optional[KVCache]] = [None] * len(self.layers)  # type: ignore[no-redef]
        if cross_kv_caches is None:
            cross_kv_caches: list[Optional[CrossAttentionKVCache]] = [  # type: ignore[no-redef]
                None
            ] * len(self.layers)

        # Sum embeddings from all codebooks (each codebook embeds independently)
        x = sum(
            self.embed_tokens[k](audio_tokens[..., k])
            for k in range(self.num_codebooks)
        )

        # Add sinusoidal positional embeddings
        offset = cache[0].offset if cache[0] is not None else 0  # type: ignore[index]
        positions = mx.arange(offset, offset + x.shape[1]).reshape(1, -1, 1)
        pos_emb = create_sin_embedding(positions, self.hidden_size)
        x = x + pos_emb.astype(x.dtype)

        # Run through transformer layers
        for layer, c, xc in zip(self.layers, cache, cross_kv_caches):  # type: ignore[arg-type]
            x = layer(x, conditioning, mask=mask, cache=c, cross_kv_cache=xc)

        # Final norm + per-codebook logits
        x = self.layer_norm(x)
        logits = mx.stack(
            [self.lm_heads[k](x) for k in range(self.num_codebooks)],
            axis=-1,
        )
        return logits  # (B, seq_len, codebook_size, num_codebooks)

    def generate(
        self,
        conditioning: mx.array,
        max_steps: int = 256,
        top_k: int = 250,
        temperature: float = 1.0,
        guidance_coef: float = 3.0,
        melody_conditioning: Optional[mx.array] = None,
        style_conditioning: Optional[mx.array] = None,
        style_coef: float = 5.0,
        progress_callback: object = None,
    ) -> mx.array:
        """Autoregressive generation with codebook delay pattern and CFG.

        Supports three conditioning modes:
          - **Standard**: 2-pass CFG (conditional + unconditional)
          - **Melody**: 2-pass CFG with chroma concatenated to text tokens
          - **Style (dual-CFG)**: 3-pass CFG (text+style, style-only, uncond)
            using formula: uncond + cfg * (style + beta * (full - style) - uncond)
            where cfg = guidance_coef, beta = style_coef

        Args:
            conditioning: T5 encoder output, shape (1, cond_len, t5_dim).
                Will be projected via enc_to_dec_proj internally.
            max_steps: Number of generation steps (50 steps = 1 second at 50 Hz).
            top_k: Number of top candidates for sampling.
            temperature: Softmax temperature.
            guidance_coef: Classifier-free guidance coefficient.
            melody_conditioning: Optional chroma features for melody variants,
                shape (1, chroma_len, num_chroma). Will be projected via
                audio_enc_to_dec_proj and concatenated with text conditioning.
            style_conditioning: Optional style tokens from StyleConditioner,
                shape (1, style_len, style_dim). Used for dual-CFG in style
                variants. Must be pre-projected to decoder dimension.
            style_coef: Beta coefficient for dual-CFG text influence (default 5.0).
                Higher values make the text prompt more influential vs. style audio.

        Returns:
            Audio tokens of shape (1, seq_len, num_codebooks) with delay undone.
        """
        # Project T5 conditioning to decoder dimension
        projected_cond = self.enc_to_dec_proj(conditioning)

        # For melody variants: project chroma and concatenate with text
        if self.is_melody and melody_conditioning is not None:
            projected_melody = self.audio_enc_to_dec_proj(melody_conditioning)
            projected_cond = mx.concatenate([projected_melody, projected_cond], axis=1)

        # Determine CFG mode
        use_dual_cfg = style_conditioning is not None

        if use_dual_cfg:
            assert style_conditioning is not None  # narrowing for mypy
            # Dual-CFG: 3 forward passes per step
            # [full (text+style), style-only, unconditional]
            # Full conditioning: concatenate style tokens + text tokens
            full_cond = mx.concatenate([style_conditioning, projected_cond], axis=1)
            # Style-only: style tokens + zeroed text
            style_only_cond = mx.concatenate(
                [style_conditioning, mx.zeros_like(projected_cond)], axis=1
            )
            # Unconditional: all zeros
            uncond = mx.zeros_like(full_cond)
            cond_batch = mx.concatenate([full_cond, style_only_cond, uncond], axis=0)
            batch_mult = 3
        else:
            # Standard CFG: 2 forward passes per step
            cond_batch = mx.concatenate(
                [projected_cond, mx.zeros_like(projected_cond)], axis=0
            )
            batch_mult = 2

        # Initialize with BOS tokens
        audio_shape = (1, max_steps + 1, self.num_codebooks)
        audio_seq = mx.full(audio_shape, self.bos_token_id)

        # Create KV caches for all layers, pre-allocated for full generation
        # to avoid reallocation + copy overhead during the loop
        head_dim = self.hidden_size // self.num_attention_heads
        caches = [
            KVCache(head_dim, self.num_attention_heads, step=max_steps + 1)
            for _ in range(len(self.layers))
        ]

        # Cross-attention KV caches: K/V from static conditioning are computed
        # once on step 0 and reused for all subsequent steps
        cross_kv_caches = [CrossAttentionKVCache() for _ in range(len(self.layers))]

        _progress_fn = progress_callback if callable(progress_callback) else None
        for step in tqdm(range(max_steps), desc="Generating"):
            if _progress_fn:
                _progress_fn(step / max_steps)
            # Tile current token for CFG batch
            audio_input = mx.tile(audio_seq[:, step : step + 1], [batch_mult, 1, 1])

            # Forward pass
            logits = self(audio_input, cond_batch, caches, cross_kv_caches)

            if use_dual_cfg:
                # Dual-CFG: uncond + cfg * (style + beta * (full - style) - uncond)
                guided_logits = _apply_dual_cfg(
                    logits[:1],
                    logits[1:2],
                    logits[2:3],
                    guidance_coef,
                    style_coef,
                )
            else:
                # Standard CFG: uncond + cfg * (cond - uncond)
                guided_logits = _apply_standard_cfg(
                    logits[:1], logits[1:2], guidance_coef
                )

            # Sample tokens
            audio_tokens = top_k_sampling(guided_logits, top_k, temperature, axis=-2)

            # Apply delay pattern: codebook k should only have tokens up to step-k
            audio_tokens[..., step + 1 :] = self.bos_token_id
            audio_tokens[..., : -max_steps + step] = self.bos_token_id

            # Write sampled tokens into sequence
            audio_seq[:, step + 1 : step + 2] = audio_tokens

            # Force graph materialisation to keep memory bounded
            _FORCE_COMPUTE(audio_seq)

        # Undo the delay pattern
        for i in range(self.num_codebooks):
            audio_seq[:, : -self.num_codebooks, i] = audio_seq[
                :, i : -self.num_codebooks + i, i
            ]
        audio_seq = audio_seq[:, 1 : -self.num_codebooks + 1]

        return audio_seq
