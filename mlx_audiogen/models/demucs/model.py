"""HTDemucs model — hybrid temporal-spectral U-Net with cross-domain transformer.

Ported from Meta's ``facebookresearch/demucs`` to Apple MLX.
"""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import DemucsConfig
from .layers import HDecLayer, HEncLayer, ScaledEmbedding
from .spec import istft, pad1d, stft
from .transformer import CrossTransformerEncoder


class HTDemucs(nn.Module):
    """Hybrid Transformer Demucs (v4) for music source separation."""

    def __init__(self, cfg: DemucsConfig):
        super().__init__()
        self.cfg = cfg
        self.hop_length = cfg.nfft // 4

        # Build encoder / decoder
        self.encoder: list[HEncLayer] = []
        self.decoder: list[HDecLayer] = []
        self.tencoder: list[HEncLayer] = []
        self.tdecoder: list[HDecLayer] = []

        chin = cfg.audio_channels
        chin_z = chin * 2 if cfg.cac else chin  # complex-as-channels
        chout = cfg.channels
        chout_z = cfg.channels
        freqs = cfg.nfft // 2

        for index in range(cfg.depth):
            norm = index >= cfg.norm_starts
            freq = freqs > 1
            stri = cfg.stride
            ker = cfg.kernel_size
            if not freq:
                ker = cfg.time_stride * 2
                stri = cfg.time_stride

            pad = True
            last_freq = False
            if freq and freqs <= cfg.kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            dconv_kw = {
                "depth": cfg.dconv_depth,
                "compress": cfg.dconv_comp,
                "init": cfg.dconv_init,
                "gelu": True,
            }
            kw: dict = dict(
                kernel_size=ker,
                stride=stri,
                freq=freq,
                pad=pad,
                norm=norm,
                rewrite=cfg.rewrite,
                norm_groups=cfg.norm_groups,
                dconv_kw=dconv_kw,
            )
            kwt = dict(
                kw, freq=False, kernel_size=cfg.kernel_size, stride=cfg.stride, pad=True
            )

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayer(
                chin_z,
                chout_z,
                dconv=bool(cfg.dconv_mode & 1),
                context=cfg.context_enc,
                **kw,
            )
            self.encoder.append(enc)

            if freq:
                tenc = HEncLayer(
                    chin,
                    chout,
                    dconv=bool(cfg.dconv_mode & 1),
                    context=cfg.context_enc,
                    empty=last_freq,
                    **kwt,
                )
                self.tencoder.append(tenc)

            # --- decoder (built in reverse) ---
            if index == 0:
                dec_chin = cfg.audio_channels * len(cfg.sources)
                dec_chin_z = dec_chin * 2 if cfg.cac else dec_chin
            else:
                dec_chin = chin
                dec_chin_z = chin_z

            dec = HDecLayer(
                chout_z,
                dec_chin_z,
                dconv=bool(cfg.dconv_mode & 2),
                last=(index == 0),
                context=cfg.context,
                **kw,
            )
            self.decoder.insert(0, dec)

            if freq:
                tdec = HDecLayer(
                    chout,
                    dec_chin,
                    dconv=bool(cfg.dconv_mode & 2),
                    empty=last_freq,
                    last=(index == 0),
                    context=cfg.context,
                    **kwt,
                )
                self.tdecoder.insert(0, tdec)

            chin = chout
            chin_z = chout_z
            chout = int(cfg.growth * chout)
            chout_z = int(cfg.growth * chout_z)
            if freq:
                freqs = 1 if freqs <= cfg.kernel_size else freqs // cfg.stride

            if index == 0 and cfg.freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=cfg.emb_smooth, scale=cfg.emb_scale
                )
                self.freq_emb_scale = cfg.freq_emb

        # Transformer at bottleneck
        transformer_channels = cfg.channels * cfg.growth ** (cfg.depth - 1)
        if cfg.t_layers > 0:
            self.crosstransformer = CrossTransformerEncoder(
                dim=transformer_channels,
                num_layers=cfg.t_layers,
                num_heads=cfg.t_heads,
                hidden_scale=cfg.t_hidden_scale,
                cross_first=cfg.t_cross_first,
                norm_in=cfg.t_norm_in,
                norm_out=cfg.t_norm_out,
                layer_scale=cfg.t_layer_scale,
                gelu=cfg.t_gelu,
                weight_pos_embed=cfg.t_weight_pos_embed,
                max_period=cfg.t_max_period,
            )
        else:
            self.crosstransformer = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # STFT helpers (numpy ↔ mx boundary)
    # ------------------------------------------------------------------

    def _spec(self, mix_np: np.ndarray) -> np.ndarray:
        """Compute STFT with Demucs padding convention."""
        hl = self.hop_length
        nfft = self.cfg.nfft
        le = int(math.ceil(mix_np.shape[-1] / hl))
        pad_amount = hl // 2 * 3
        x = pad1d(
            mix_np,
            (pad_amount, pad_amount + le * hl - mix_np.shape[-1]),
            mode="reflect",
        )
        z = stft(x, nfft, hl)
        # Remove last freq bin (2049 → 2048) and trim edge frames
        z = z[..., :-1, :]
        z = z[..., 2 : 2 + le]
        return z

    def _ispec(self, z: np.ndarray, length: int) -> np.ndarray:
        """Inverse STFT with Demucs padding convention."""
        hl = self.hop_length
        # Restore last freq bin
        pad_shape = list(z.shape)
        pad_shape[-2] = 1
        z = np.concatenate([z, np.zeros(pad_shape, dtype=z.dtype)], axis=-2)
        # Pad 2 frames on each side of time
        pad_shape[-2] = z.shape[-2]
        pad_shape[-1] = 2
        z = np.concatenate(
            [np.zeros(pad_shape, dtype=z.dtype), z, np.zeros(pad_shape, dtype=z.dtype)],
            axis=-1,
        )
        pad_amount = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad_amount
        x = istft(z, hl, length=le)
        x = x[..., pad_amount : pad_amount + length]
        return x

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def __call__(self, mix: mx.array) -> mx.array:
        """Separate a stereo mixture.

        Args:
            mix: Input waveform ``(B, 2, T)`` at 44.1 kHz.

        Returns:
            Separated sources ``(B, S, 2, T)`` where S = number of sources.
        """
        length = mix.shape[-1]
        training_length = int(self.cfg.segment * self.cfg.samplerate)

        # Pad to training segment length
        length_pre_pad = None
        if self.cfg.use_train_segment and length < training_length:
            length_pre_pad = length
            mix = mx.pad(mix, [(0, 0), (0, 0), (0, training_length - length)])

        # --- STFT (in numpy) ---
        mix_np = np.array(mix)
        z_np = self._spec(mix_np)  # complex spectrogram

        # Complex-as-channels: split real/imag into channel dim
        B, C_audio, Fr, T = z_np.shape
        z_real = z_np.real  # (B, C, Fr, T)
        z_imag = z_np.imag
        mag_np = np.concatenate([z_real, z_imag], axis=1).reshape(B, C_audio * 2, Fr, T)

        x = mx.array(mag_np.astype(np.float32))
        Fq = Fr
        T_spec = T

        # Normalise spectral branch
        mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
        std = mx.sqrt(mx.mean((x - mean) ** 2, axis=(1, 2, 3), keepdims=True))
        x = (x - mean) / (1e-5 + std)

        # Normalise temporal branch
        xt = mix
        meant = mx.mean(xt, axis=(1, 2), keepdims=True)
        stdt = mx.sqrt(mx.mean((xt - meant) ** 2, axis=(1, 2), keepdims=True))
        xt = (xt - meant) / (1e-5 + stdt)

        # --- Encoder ---
        saved = []
        saved_t = []
        lengths = []
        lengths_t = []

        for idx, enc in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = enc(x, inject)

            if idx == 0 and hasattr(self, "freq_emb"):
                frs = mx.arange(x.shape[-2])
                emb = self.freq_emb(frs)  # (Fr, C)
                emb = emb.T[None, :, :, None]  # (1, C, Fr, 1)
                x = x + self.freq_emb_scale * mx.broadcast_to(emb, x.shape)

            saved.append(x)

        # --- Cross-transformer ---
        if self.crosstransformer is not None:
            x, xt = self.crosstransformer(x, xt)

        # --- Decoder ---
        for idx, dec in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = dec(x, skip, lengths.pop(-1))

            offset = self.cfg.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre_flat = pre[:, :, 0]  # freq dim collapsed to 1
                    xt, _ = tdec(pre_flat, None, length_t)
                else:
                    skip_t = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip_t, length_t)

        # --- Reconstruct output ---
        S = len(self.cfg.sources)
        work_length = training_length if self.cfg.use_train_segment else length

        # Spectral branch → CaC mask → iSTFT
        x = x.reshape(B, S, -1, Fq, T_spec)
        x = x * std[:, None] + mean[:, None]

        # Convert CaC back to complex spectrogram (numpy)
        x_np = np.array(x)
        # x_np: (B, S, C*2, Fq, T) → split into real/imag
        half_c = x_np.shape[2] // 2
        out_real = x_np[:, :, :half_c]
        out_imag = x_np[:, :, half_c:]
        zout = out_real + 1j * out_imag  # (B, S, C, Fq, T)

        # iSTFT per source
        spec_out = np.zeros((B, S, C_audio, work_length), dtype=np.float32)
        for s in range(S):
            spec_out[:, s] = self._ispec(zout[:, s], work_length)

        # Temporal branch
        xt = xt.reshape(B, S, -1, work_length)
        xt = xt * stdt[:, None] + meant[:, None]
        xt_np = np.array(xt)

        result = mx.array((xt_np + spec_out).astype(np.float32))

        if length_pre_pad is not None:
            result = result[..., :length_pre_pad]

        return result
