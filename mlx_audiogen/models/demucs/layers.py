"""Core building blocks for HTDemucs in MLX.

All convolution and norm wrappers operate in PyTorch layout (NCL / NCHW)
and internally transpose for MLX's channels-last convention. This keeps
the forward-pass reshape/permute logic identical to the reference code.
"""

import mlx.core as mx
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Conv wrappers: accept NCL / NCHW, transpose internally for MLX
# ---------------------------------------------------------------------------


class Conv1d(nn.Conv1d):
    """``nn.Conv1d`` that accepts **(B, C, L)** input."""

    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.transpose(0, 2, 1)).transpose(0, 2, 1)


class Conv2d(nn.Conv2d):
    """``nn.Conv2d`` that accepts **(B, C, H, W)** input."""

    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


class ConvTranspose1d(nn.ConvTranspose1d):
    """``nn.ConvTranspose1d`` that accepts **(B, C, L)** input."""

    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.transpose(0, 2, 1)).transpose(0, 2, 1)


class ConvTranspose2d(nn.ConvTranspose2d):
    """``nn.ConvTranspose2d`` that accepts **(B, C, H, W)** input."""

    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# GroupNorm wrapper for NCL / NCHW tensors
# ---------------------------------------------------------------------------


class GroupNorm(nn.GroupNorm):
    """``nn.GroupNorm`` that handles NCL and NCHW inputs.

    MLX's GroupNorm expects channels as the *last* dimension.
    """

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3:
            return super().__call__(x.transpose(0, 2, 1)).transpose(0, 2, 1)
        if x.ndim == 4:
            return super().__call__(x.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        return super().__call__(x)


# ---------------------------------------------------------------------------
# Small helper modules
# ---------------------------------------------------------------------------


class LayerScale(nn.Module):
    """Learnable per-channel scale initialised near zero."""

    def __init__(self, channels: int, init: float = 1e-4):
        super().__init__()
        self.scale = mx.full((channels,), init)

    def __call__(self, x: mx.array) -> mx.array:
        # x is (B, C, T) — broadcast scale over (1, C, 1)
        return self.scale[:, None] * x


class ScaledEmbedding(nn.Module):
    """Embedding with boosted learning rate via ``scale``.

    Optionally smooths initialisation via cumulative sum.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        scale: float = 10.0,
        smooth: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    @property
    def weight(self) -> mx.array:
        return self.embedding.weight * self.scale

    def __call__(self, x: mx.array) -> mx.array:
        return self.embedding(x) * self.scale


# ---------------------------------------------------------------------------
# DConv — dilated residual branch
# ---------------------------------------------------------------------------


class _DConvBlock(nn.Module):
    """A single dilated-conv residual sub-layer inside DConv."""

    def __init__(
        self,
        channels: int,
        compress: int,
        dilation: int,
        kernel: int,
        init: float,
        gelu: bool,
    ):
        super().__init__()
        hidden = max(1, int(channels / compress))
        padding = dilation * (kernel // 2)

        # Sequential: Conv1d → GN → Act → Conv1d → GN → GLU → LayerScale
        self.conv1 = Conv1d(
            channels, hidden, kernel, padding=padding, dilation=dilation
        )
        self.norm1 = GroupNorm(1, hidden)
        self.conv2 = Conv1d(hidden, 2 * channels, 1)
        self.norm2 = GroupNorm(1, 2 * channels)
        self.layer_scale = LayerScale(channels, init)
        self.gelu = gelu

    def __call__(self, x: mx.array) -> mx.array:
        y = self.conv1(x)
        y = self.norm1(y)
        y = nn.gelu(y) if self.gelu else nn.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        # GLU along channel dim (dim=1 for NCL)
        a, b = mx.split(y, 2, axis=1)
        y = a * mx.sigmoid(b)
        return self.layer_scale(y)


class DConv(nn.Module):
    """Dilated convolution residual branch."""

    def __init__(
        self,
        channels: int,
        compress: int = 4,
        depth: int = 2,
        init: float = 1e-4,
        gelu: bool = True,
        kernel: int = 3,
    ):
        super().__init__()
        self.layers = [
            _DConvBlock(channels, compress, 2**d, kernel, init, gelu)
            for d in range(depth)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = x + layer(x)
        return x


# ---------------------------------------------------------------------------
# HEncLayer — encoder layer (works for both freq and time branches)
# ---------------------------------------------------------------------------


class HEncLayer(nn.Module):
    def __init__(
        self,
        chin: int,
        chout: int,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 1,
        empty: bool = False,
        freq: bool = True,
        dconv: bool = True,
        norm: bool = True,
        context: int = 0,
        dconv_kw: dict | None = None,
        pad: bool = True,
        rewrite: bool = True,
    ):
        super().__init__()
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm_enabled = norm
        self.pad_size = kernel_size // 4 if pad else 0

        if freq:
            self.conv = Conv2d(
                chin,
                chout,
                (kernel_size, 1),
                stride=(stride, 1),
                padding=(self.pad_size, 0),
            )
        else:
            self.conv = Conv1d(  # type: ignore[assignment]
                chin, chout, kernel_size, stride=stride, padding=self.pad_size
            )

        if empty:
            return

        self.norm1 = GroupNorm(norm_groups, chout) if norm else nn.Identity()

        self.rewrite_conv = None
        if rewrite:
            k = 1 + 2 * context
            if freq:
                self.rewrite_conv = Conv2d(
                    chout, 2 * chout, (k, k), padding=(context, context)
                )
            else:
                self.rewrite_conv = Conv1d(chout, 2 * chout, k, padding=context)  # type: ignore[assignment]
            self.norm2 = GroupNorm(norm_groups, 2 * chout) if norm else nn.Identity()

        self.dconv_block = None
        if dconv:
            kw = dconv_kw or {}
            self.dconv_block = DConv(chout, **kw)

    def __call__(self, x: mx.array, inject: mx.array | None = None) -> mx.array:
        if not self.freq and x.ndim == 4:
            B, C, Fr, T = x.shape
            x = x.reshape(B, -1, T)

        if not self.freq:
            le = x.shape[-1]
            rem = le % self.stride
            if rem != 0:
                x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, self.stride - rem)])

        y = self.conv(x)

        if self.empty:
            return y

        if inject is not None:
            if inject.shape[-1] != y.shape[-1]:  # nosec B101
                msg = f"inject time {inject.shape[-1]} != y time {y.shape[-1]}"
                raise ValueError(msg)
            if inject.ndim == 3 and y.ndim == 4:
                inject = inject[:, :, None]
            y = y + inject

        y = nn.gelu(self.norm1(y))

        if self.dconv_block is not None:
            if self.freq:
                B, C, Fr, T = y.shape
                y = y.transpose(0, 2, 1, 3).reshape(-1, C, T)
            y = self.dconv_block(y)
            if self.freq:
                y = y.reshape(B, Fr, C, T).transpose(0, 2, 1, 3)

        if self.rewrite_conv is not None:
            z = self.norm2(self.rewrite_conv(y))
            # GLU along channel dim
            a, b = mx.split(z, 2, axis=1)
            z = a * mx.sigmoid(b)
        else:
            z = y
        return z


# ---------------------------------------------------------------------------
# HDecLayer — decoder layer
# ---------------------------------------------------------------------------


class HDecLayer(nn.Module):
    def __init__(
        self,
        chin: int,
        chout: int,
        last: bool = False,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 1,
        empty: bool = False,
        freq: bool = True,
        dconv: bool = True,
        norm: bool = True,
        context: int = 1,
        dconv_kw: dict | None = None,
        pad: bool = True,
        context_freq: bool = True,
        rewrite: bool = True,
    ):
        super().__init__()
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.last = last
        self.pad_size = kernel_size // 4 if pad else 0
        self.context_freq = context_freq

        if freq:
            self.conv_tr = ConvTranspose2d(
                chin, chout, (kernel_size, 1), stride=(stride, 1)
            )
        else:
            self.conv_tr = ConvTranspose1d(chin, chout, kernel_size, stride=stride)  # type: ignore[assignment]

        self.norm2 = GroupNorm(norm_groups, chout) if norm else nn.Identity()

        if empty:
            return

        self.rewrite_conv = None
        if rewrite:
            k = 1 + 2 * context
            if freq:
                if context_freq:
                    self.rewrite_conv = Conv2d(
                        chin, 2 * chin, (k, k), padding=(context, context)
                    )
                else:
                    self.rewrite_conv = Conv2d(
                        chin, 2 * chin, (1, k), padding=(0, context)
                    )
            else:
                self.rewrite_conv = Conv1d(chin, 2 * chin, k, padding=context)  # type: ignore[assignment]
            self.norm1 = GroupNorm(norm_groups, 2 * chin) if norm else nn.Identity()

        self.dconv_block = None
        if dconv:
            kw = dconv_kw or {}
            self.dconv_block = DConv(chin, **kw)

    def __call__(
        self, x: mx.array, skip: mx.array | None, length: int
    ) -> tuple[mx.array, mx.array]:
        if self.freq and x.ndim == 3:
            B, C, T = x.shape
            x = x.reshape(B, self.chin, -1, T)

        if not self.empty:
            x = x + skip  # type: ignore[operator]

            if self.rewrite_conv is not None:
                y = self.norm1(self.rewrite_conv(x))
                a, b = mx.split(y, 2, axis=1)
                y = a * mx.sigmoid(b)
            else:
                y = x

            if self.dconv_block is not None:
                if self.freq:
                    B, C, Fr, T = y.shape
                    y = y.transpose(0, 2, 1, 3).reshape(-1, C, T)
                y = self.dconv_block(y)
                if self.freq:
                    y = y.reshape(B, Fr, C, T).transpose(0, 2, 1, 3)
        else:
            y = x

        z = self.norm2(self.conv_tr(y))

        if self.freq:
            if self.pad_size:
                z = z[:, :, self.pad_size : -self.pad_size, :]
        else:
            z = z[:, :, self.pad_size : self.pad_size + length]

        if not self.last:
            z = nn.gelu(z)

        return z, y
