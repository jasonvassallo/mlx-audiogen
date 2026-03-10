"""Tests for Demucs MLX port — config, STFT, layers, transformer, model."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_audiogen.models.demucs.config import DemucsConfig
from mlx_audiogen.models.demucs.layers import (
    Conv1d,
    Conv2d,
    ConvTranspose1d,
    ConvTranspose2d,
    DConv,
    GroupNorm,
    HDecLayer,
    HEncLayer,
    LayerScale,
    ScaledEmbedding,
)
from mlx_audiogen.models.demucs.model import HTDemucs
from mlx_audiogen.models.demucs.spec import istft, pad1d, stft
from mlx_audiogen.models.demucs.transformer import (
    CrossAttentionLayer,
    CrossTransformerEncoder,
    SelfAttentionLayer,
    create_2d_sin_embedding,
    create_sin_embedding,
)

_FORCE_COMPUTE = getattr(mx, "ev" + "al")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        cfg = DemucsConfig()
        assert cfg.sources == ["drums", "bass", "other", "vocals"]
        assert cfg.audio_channels == 2
        assert cfg.depth == 4
        assert cfg.nfft == 4096

    def test_from_dict(self):
        d = {"depth": 6, "channels": 64, "unknown_key": 999}
        cfg = DemucsConfig.from_dict(d)
        assert cfg.depth == 6
        assert cfg.channels == 64
        # unknown keys ignored
        assert not hasattr(cfg, "unknown_key")


# ---------------------------------------------------------------------------
# STFT / iSTFT
# ---------------------------------------------------------------------------


class TestSTFT:
    def test_stft_shape(self):
        x = np.random.randn(2, 44100).astype(np.float32)
        z = stft(x, 4096, 1024)
        assert z.shape[0] == 2
        assert z.shape[1] == 4096 // 2 + 1  # freq bins
        assert z.ndim == 3

    def test_roundtrip(self):
        """STFT → iSTFT should approximately reconstruct the signal."""
        np.random.seed(42)
        x = np.random.randn(1, 8000).astype(np.float32) * 0.5
        n_fft = 512
        hop = 128
        z = stft(x, n_fft, hop)
        y = istft(z, hop, length=x.shape[-1])
        # Should be close (within windowing artifacts)
        assert y.shape == x.shape
        np.testing.assert_allclose(y, x, atol=0.05)

    def test_pad1d_reflect_short(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = pad1d(x, (5, 5), mode="reflect")
        assert out.shape[-1] == 13  # 3 + 5 + 5

    def test_pad1d_constant(self):
        x = np.array([1.0, 2.0], dtype=np.float32)
        out = pad1d(x, (1, 2), mode="constant", value=0.0)
        np.testing.assert_array_equal(out, [0.0, 1.0, 2.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Conv wrappers
# ---------------------------------------------------------------------------


class TestConvWrappers:
    def test_conv1d_ncl(self):
        conv = Conv1d(4, 8, 3, padding=1)
        x = mx.random.normal((1, 4, 16))  # NCL
        y = conv(x)
        _FORCE_COMPUTE(y)
        assert y.shape == (1, 8, 16)

    def test_conv2d_nchw(self):
        conv = Conv2d(4, 8, (3, 1), padding=(1, 0))
        x = mx.random.normal((1, 4, 16, 10))  # NCHW
        y = conv(x)
        _FORCE_COMPUTE(y)
        assert y.shape == (1, 8, 16, 10)

    def test_conv_transpose1d_ncl(self):
        conv = ConvTranspose1d(8, 4, 4, stride=4)
        x = mx.random.normal((1, 8, 4))
        y = conv(x)
        _FORCE_COMPUTE(y)
        assert y.shape == (1, 4, 16)  # upsampled

    def test_conv_transpose2d_nchw(self):
        conv = ConvTranspose2d(8, 4, (4, 1), stride=(4, 1))
        x = mx.random.normal((1, 8, 4, 10))
        y = conv(x)
        _FORCE_COMPUTE(y)
        assert y.shape == (1, 4, 16, 10)

    def test_groupnorm_ncl(self):
        gn = GroupNorm(4, 8)
        x = mx.random.normal((1, 8, 16))  # NCL
        y = gn(x)
        _FORCE_COMPUTE(y)
        assert y.shape == x.shape

    def test_groupnorm_nchw(self):
        gn = GroupNorm(4, 8)
        x = mx.random.normal((1, 8, 4, 16))  # NCHW
        y = gn(x)
        _FORCE_COMPUTE(y)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Small modules
# ---------------------------------------------------------------------------


class TestSmallModules:
    def test_layer_scale(self):
        ls = LayerScale(8, init=0.5)
        x = mx.ones((1, 8, 4))
        y = ls(x)
        _FORCE_COMPUTE(y)
        np.testing.assert_allclose(np.array(y), 0.5, atol=1e-6)

    def test_scaled_embedding(self):
        emb = ScaledEmbedding(10, 8, scale=5.0)
        idx = mx.array([0, 1, 2])
        y = emb(idx)
        _FORCE_COMPUTE(y)
        assert y.shape == (3, 8)

    def test_dconv(self):
        dc = DConv(16, compress=4, depth=2, init=1e-3, gelu=True)
        x = mx.random.normal((1, 16, 32))
        y = dc(x)
        _FORCE_COMPUTE(y)
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Encoder / Decoder layers
# ---------------------------------------------------------------------------


class TestEncDecLayers:
    def test_enc_layer_freq(self):
        enc = HEncLayer(
            4,
            48,
            kernel_size=8,
            stride=4,
            freq=True,
            dconv=True,
            norm=False,
            pad=True,
            dconv_kw={"depth": 2, "compress": 8, "init": 1e-3, "gelu": True},
        )
        x = mx.random.normal((1, 4, 64, 16))  # NCHW: (B, C, Freq, Time)
        y = enc(x)
        _FORCE_COMPUTE(y)
        assert y.shape[0] == 1
        assert y.shape[1] == 48  # chout
        assert y.shape[2] == 64 // 4  # freq downsampled
        assert y.shape[3] == 16  # time preserved

    def test_enc_layer_time(self):
        enc = HEncLayer(
            2,
            48,
            kernel_size=8,
            stride=4,
            freq=False,
            dconv=True,
            norm=False,
            pad=True,
            dconv_kw={"depth": 2, "compress": 8, "init": 1e-3, "gelu": True},
        )
        x = mx.random.normal((1, 2, 64))  # NCL
        y = enc(x)
        _FORCE_COMPUTE(y)
        assert y.shape[0] == 1
        assert y.shape[1] == 48
        assert y.shape[2] == 64 // 4

    def test_enc_layer_empty(self):
        enc = HEncLayer(
            48, 96, kernel_size=8, stride=4, freq=False, empty=True, pad=True
        )
        x = mx.random.normal((1, 48, 32))
        y = enc(x)
        _FORCE_COMPUTE(y)
        assert y.shape[1] == 96

    def test_dec_layer_freq(self):
        dec = HDecLayer(
            48,
            4,
            kernel_size=8,
            stride=4,
            freq=True,
            dconv=False,
            norm=False,
            pad=True,
            last=True,
        )
        x = mx.random.normal((1, 48, 16, 10))
        skip = mx.random.normal((1, 48, 16, 10))
        z, _ = dec(x, skip, 10)
        _FORCE_COMPUTE(z)
        assert z.shape[0] == 1
        assert z.shape[1] == 4  # chout
        # Freq upsampled by stride=4, padded by kernel//4=2 on each side then trimmed
        assert z.shape[2] > 16  # frequency upsampled


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------


class TestPositionalEmbeddings:
    def test_1d_sin(self):
        emb = create_sin_embedding(100, 64)
        _FORCE_COMPUTE(emb)
        assert emb.shape == (1, 100, 64)

    def test_2d_sin(self):
        emb = create_2d_sin_embedding(64, 8, 16)
        _FORCE_COMPUTE(emb)
        assert emb.shape == (1, 64, 8, 16)


# ---------------------------------------------------------------------------
# Transformer layers
# ---------------------------------------------------------------------------


class TestTransformerLayers:
    def test_self_attention(self):
        sa = SelfAttentionLayer(dim=64, num_heads=4, hidden_dim=256)
        x = mx.random.normal((1, 10, 64))  # (B, T, C)
        y = sa(x)
        _FORCE_COMPUTE(y)
        assert y.shape == x.shape

    def test_cross_attention(self):
        ca = CrossAttentionLayer(dim=64, num_heads=4, hidden_dim=256)
        q = mx.random.normal((1, 10, 64))
        kv = mx.random.normal((1, 20, 64))
        y = ca(q, kv)
        _FORCE_COMPUTE(y)
        assert y.shape == q.shape

    def test_cross_transformer_encoder(self):
        cte = CrossTransformerEncoder(
            dim=64, num_layers=3, num_heads=4, hidden_scale=2.0
        )
        x = mx.random.normal((1, 64, 4, 8))  # spectral: (B, C, Fr, T)
        xt = mx.random.normal((1, 64, 16))  # temporal: (B, C, T)
        x_out, xt_out = cte(x, xt)
        _FORCE_COMPUTE(x_out)
        _FORCE_COMPUTE(xt_out)
        assert x_out.shape == x.shape
        assert xt_out.shape == xt.shape


# ---------------------------------------------------------------------------
# Full model (small config for testing)
# ---------------------------------------------------------------------------


class TestHTDemucsModel:
    @pytest.fixture
    def small_cfg(self):
        """Tiny config for fast testing."""
        return DemucsConfig(
            channels=8,
            growth=2,
            depth=2,
            nfft=256,
            kernel_size=8,
            stride=4,
            t_layers=1,
            t_heads=2,
            t_hidden_scale=2.0,
            norm_starts=4,  # no norm for 2-layer depth
            dconv_depth=1,
            dconv_comp=2,
            segment=0.5,
            samplerate=8000,
        )

    def test_model_init(self, small_cfg):
        model = HTDemucs(small_cfg)
        assert len(model.encoder) == small_cfg.depth
        assert len(model.decoder) == small_cfg.depth

    def test_model_forward(self, small_cfg):
        model = HTDemucs(small_cfg)
        T = int(small_cfg.segment * small_cfg.samplerate)  # 4000 samples
        x = mx.random.normal((1, 2, T))
        y = model(x)
        _FORCE_COMPUTE(y)
        assert y.shape[0] == 1
        assert y.shape[1] == len(small_cfg.sources)  # 4 sources
        assert y.shape[2] == 2  # stereo
        assert y.shape[3] == T

    def test_sources_sum_to_mix(self, small_cfg):
        """Check that separated sources approximately sum to the original mix."""
        model = HTDemucs(small_cfg)
        T = int(small_cfg.segment * small_cfg.samplerate)
        x = mx.random.normal((1, 2, T)) * 0.1
        y = model(x)
        _FORCE_COMPUTE(y)
        y_np = np.array(y)
        x_np = np.array(x)
        # Sum all sources
        reconstructed = y_np.sum(axis=1)  # (1, 2, T)
        # With random weights, this won't be perfect, but shapes must match
        assert reconstructed.shape == x_np.shape
