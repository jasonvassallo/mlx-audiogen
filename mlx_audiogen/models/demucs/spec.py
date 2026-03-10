"""STFT / iSTFT utilities using numpy.

Matches the behaviour of ``demucs.spec.spectro`` / ``ispectro`` from
the reference PyTorch implementation.
"""

import numpy as np


def _hann_window(n: int) -> np.ndarray:
    """Periodic Hann window (matches ``torch.hann_window``)."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def stft(
    x: np.ndarray,
    n_fft: int,
    hop_length: int | None = None,
) -> np.ndarray:
    """Compute normalised, centered STFT.

    Args:
        x: Audio array of shape ``(..., length)``.
        n_fft: FFT size.
        hop_length: Hop size (default ``n_fft // 4``).

    Returns:
        Complex spectrogram of shape ``(..., n_fft // 2 + 1, frames)``.
    """
    if hop_length is None:
        hop_length = n_fft // 4

    other = x.shape[:-1]
    length = x.shape[-1]
    x = x.reshape(-1, length)
    batch = x.shape[0]

    # Centre-pad with reflection
    pad = n_fft // 2
    x = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")

    window = _hann_window(n_fft)
    n_frames = 1 + (x.shape[-1] - n_fft) // hop_length

    # Vectorised framing via stride tricks
    shape = (batch, n_frames, n_fft)
    strides = (x.strides[0], x.strides[1] * hop_length, x.strides[1])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()
    frames = frames * window  # apply window

    # FFT → complex spectrogram
    spec = np.fft.rfft(frames, n=n_fft, axis=-1)  # (batch, frames, freq)
    # Normalise to match torch.stft(normalized=True)
    spec = spec / np.sqrt(n_fft)
    spec = spec.transpose(0, 2, 1)  # (batch, freq, frames)
    return spec.reshape(*other, spec.shape[-2], spec.shape[-1]).astype(np.complex64)


def istft(
    z: np.ndarray,
    hop_length: int,
    length: int | None = None,
) -> np.ndarray:
    """Inverse STFT (overlap-add with Hann window).

    Args:
        z: Complex spectrogram ``(..., freq_bins, frames)``.
        hop_length: Hop size.
        length: Desired output length.

    Returns:
        Reconstructed waveform ``(..., samples)``.
    """
    other = z.shape[:-2]
    freq_bins, n_frames = z.shape[-2], z.shape[-1]
    n_fft = 2 * (freq_bins - 1)
    z = z.reshape(-1, freq_bins, n_frames)
    batch = z.shape[0]

    window = _hann_window(n_fft)
    # Un-normalise
    z = z * np.sqrt(n_fft)

    z_t = z.transpose(0, 2, 1)  # (batch, frames, freq)
    frames = np.fft.irfft(z_t, n=n_fft, axis=-1).astype(np.float32)

    # Overlap-add
    out_len = n_fft + hop_length * (n_frames - 1)
    output = np.zeros((batch, out_len), dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)

    for i in range(n_frames):
        s = i * hop_length
        output[:, s : s + n_fft] += frames[:, i, :] * window
        win_sum[s : s + n_fft] += window**2

    win_sum = np.maximum(win_sum, 1e-8)
    output /= win_sum

    # Remove centre-padding
    pad = n_fft // 2
    output = output[:, pad:]

    if length is not None:
        output = output[:, :length]

    return output.reshape(*other, output.shape[-1])


def pad1d(
    x: np.ndarray,
    paddings: tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> np.ndarray:
    """Pad last dimension, handling reflect mode on short inputs."""
    pad_left, pad_right = paddings
    length = x.shape[-1]

    if mode == "reflect":
        max_pad = max(pad_left, pad_right)
        if length <= max_pad:
            extra = max_pad - length + 1
            extra_r = min(pad_right, extra)
            extra_l = extra - extra_r
            nd = x.ndim
            padding = [(0, 0)] * (nd - 1) + [(extra_l, extra_r)]
            x = np.pad(x, padding, mode="constant")
            pad_left -= extra_l
            pad_right -= extra_r

    nd = x.ndim
    padding = [(0, 0)] * (nd - 1) + [(pad_left, pad_right)]
    if mode == "constant":
        return np.pad(x, padding, mode=mode, constant_values=value).astype(x.dtype)  # type: ignore[call-overload]
    return np.pad(x, padding, mode=mode).astype(x.dtype)  # type: ignore[call-overload]
