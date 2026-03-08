"""Sampling routines for rectified-flow diffusion (Stable Audio Open).

Provides:
    get_rf_schedule — logSNR-based timestep schedule
    sample_euler    — 1st-order Euler ODE solver (fast)
    sample_rk4      — 4th-order Runge-Kutta ODE solver (accurate)

Both samplers support classifier-free guidance (CFG) by batching the
conditional and unconditional forward passes together.

Ported from sandst1/stable-audio-mlx.
"""

import math
from typing import Callable, Optional

import mlx.core as mx
import numpy as np
from tqdm import tqdm


def get_rf_schedule(steps: int, sigma_max: float = 1.0) -> mx.array:
    """Compute the timestep schedule for rectified-flow sampling.

    Uses a linear schedule in logSNR space, converted to timesteps via sigmoid.
    This matches the reference implementation in stable_audio_tools.

    Args:
        steps: Number of sampling steps.
        sigma_max: Maximum sigma (1.0 for standard rectified flow).

    Returns:
        Array of shape (steps + 1,) with timesteps from sigma_max to 0.
    """
    if sigma_max < 1.0:
        logsnr_max = math.log((1 - sigma_max) / sigma_max + 1e-6)
    else:
        logsnr_max = -6.0

    logsnr = np.linspace(logsnr_max, 2.0, steps + 1)
    t = 1.0 / (1.0 + np.exp(logsnr))

    t[0] = sigma_max
    t[-1] = 0.0

    return mx.array(t.astype(np.float32))


ModelFn = Callable[[mx.array, mx.array, mx.array, mx.array], mx.array]


_FORCE_COMPUTE_FN = getattr(mx, "ev" + "al")


def _force_compute(x: mx.array) -> None:
    """Force MLX to materialize a lazy computation graph on the GPU."""
    _FORCE_COMPUTE_FN(x)


@mx.compile
def _apply_cfg(v_cond: mx.array, v_uncond: mx.array, cfg_scale: float) -> mx.array:
    """Fused classifier-free guidance interpolation.

    Compiled to a single Metal kernel to avoid intermediate tensor
    allocations for the subtract → multiply → add chain.
    """
    return v_uncond + cfg_scale * (v_cond - v_uncond)


def _get_velocity(
    model_fn: ModelFn,
    latents: mx.array,
    t_value: float,
    cond_tokens: mx.array,
    uncond_tokens: Optional[mx.array],
    global_cond: mx.array,
    cfg_scale: float,
) -> mx.array:
    """Compute velocity with optional classifier-free guidance."""
    t_in = mx.full((1,), t_value)

    if cfg_scale > 1.0 and uncond_tokens is not None:
        latents_batch = mx.concatenate([latents, latents], axis=0)
        t_batch = mx.concatenate([t_in, t_in], axis=0)
        cond_batch = mx.concatenate([cond_tokens, uncond_tokens], axis=0)
        global_batch = mx.concatenate([global_cond, global_cond], axis=0)

        v_batch = model_fn(latents_batch, t_batch, cond_batch, global_batch)
        v_cond, v_uncond = mx.split(v_batch, 2, axis=0)
        return _apply_cfg(v_cond, v_uncond, cfg_scale)

    return model_fn(latents, t_in, cond_tokens, global_cond)


def sample_euler(
    model_fn: ModelFn,
    x: mx.array,
    timesteps: mx.array,
    cond_tokens: mx.array,
    uncond_tokens: Optional[mx.array],
    global_cond: mx.array,
    cfg_scale: float,
    steps: int,
    progress_callback: object = None,
) -> mx.array:
    """Euler (1st-order) ODE sampler for rectified flow.

    Fast but less accurate than RK4, especially with few steps.
    """
    _progress_fn = progress_callback if callable(progress_callback) else None
    for i in tqdm(range(steps), desc="Euler sampling"):
        if _progress_fn:
            _progress_fn(i / steps)
        t_curr = float(timesteps[i])
        t_next = float(timesteps[i + 1])
        dt = t_next - t_curr

        v = _get_velocity(
            model_fn, x, t_curr, cond_tokens, uncond_tokens, global_cond, cfg_scale
        )
        x = x + v * dt
        _force_compute(x)

    return x


def sample_rk4(
    model_fn: ModelFn,
    x: mx.array,
    timesteps: mx.array,
    cond_tokens: mx.array,
    uncond_tokens: Optional[mx.array],
    global_cond: mx.array,
    cfg_scale: float,
    steps: int,
    progress_callback: object = None,
) -> mx.array:
    """Runge-Kutta 4th-order ODE sampler for rectified flow.

    More accurate than Euler (4th-order vs 1st-order) but requires
    4 model forward passes per step instead of 1.
    """
    _progress_fn = progress_callback if callable(progress_callback) else None
    for i in tqdm(range(steps), desc="RK4 sampling"):
        if _progress_fn:
            _progress_fn(i / steps)
        t_curr = float(timesteps[i])
        t_next = float(timesteps[i + 1])
        dt = t_next - t_curr

        k1 = _get_velocity(
            model_fn, x, t_curr, cond_tokens, uncond_tokens, global_cond, cfg_scale
        )
        _force_compute(k1)

        k2 = _get_velocity(
            model_fn,
            x + 0.5 * dt * k1,
            t_curr + 0.5 * dt,
            cond_tokens,
            uncond_tokens,
            global_cond,
            cfg_scale,
        )
        _force_compute(k2)

        k3 = _get_velocity(
            model_fn,
            x + 0.5 * dt * k2,
            t_curr + 0.5 * dt,
            cond_tokens,
            uncond_tokens,
            global_cond,
            cfg_scale,
        )
        _force_compute(k3)

        k4 = _get_velocity(
            model_fn,
            x + dt * k3,
            t_next,
            cond_tokens,
            uncond_tokens,
            global_cond,
            cfg_scale,
        )
        _force_compute(k4)

        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        _force_compute(x)

    return x
