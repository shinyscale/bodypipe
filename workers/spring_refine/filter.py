"""Semi-implicit second-order IIR spring-damper filter.

Per-joint, per-component. Stable for large Kp*dt unlike explicit Euler.
Optional forward-backward pass for zero-phase filtering (kills the group
delay so the output doesn't feel laggy on sharp motion).
"""

from __future__ import annotations

import numpy as np


def _forward(
    target: np.ndarray,
    kp: np.ndarray,
    inv_one_plus_kv_dt: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Run the forward semi-implicit Euler pass.

    Recurrence (per joint, per channel):
        v_{t+1} = (v_t + Kp * (target_t - x_t) * dt) * inv_one_plus_kv_dt
        x_{t+1} = x_t + v_{t+1} * dt
    """
    T = target.shape[0]
    x = target[0].copy()
    v = np.zeros_like(x)
    out = np.empty_like(target)
    out[0] = x
    for t in range(1, T):
        v = (v + kp * (target[t] - x) * dt) * inv_one_plus_kv_dt
        x = x + v * dt
        out[t] = x
    return out


def critically_damped_filter(
    target: np.ndarray,
    kp: np.ndarray,
    kv: np.ndarray,
    dt: float,
    zero_phase: bool = True,
    pad_frames: int = 0,
) -> np.ndarray:
    """Filter ``target`` (T, J, 3) with a per-joint spring-damper.

    Args:
        target: (T, J, 3) signal in quaternion log space.
        kp: (J,) stiffness per joint.
        kv: (J,) damping per joint.
        dt: timestep in seconds (1 / fps).
        zero_phase: if True, run filter forward then backward (filtfilt
            style) to cancel the group delay.
        pad_frames: number of frames to replicate at each end before
            filtering (mitigates the warm-up transient). Trimmed after.
    """
    target = np.asarray(target, dtype=np.float32)
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError(f"target must be (T, J, 3); got {target.shape}")
    T, J, _ = target.shape
    if T == 0:
        return target.copy()

    kp = np.asarray(kp, dtype=np.float32).reshape(J, 1)
    kv = np.asarray(kv, dtype=np.float32).reshape(J, 1)
    inv_one_plus_kv_dt = 1.0 / (1.0 + kv * dt)

    if pad_frames > 0 and T > 1:
        head = np.repeat(target[:1], pad_frames, axis=0)
        tail = np.repeat(target[-1:], pad_frames, axis=0)
        padded = np.concatenate([head, target, tail], axis=0)
    else:
        padded = target

    fwd = _forward(padded, kp, inv_one_plus_kv_dt, dt)
    if zero_phase:
        out = _forward(fwd[::-1], kp, inv_one_plus_kv_dt, dt)[::-1]
    else:
        out = fwd

    if pad_frames > 0 and T > 1:
        out = out[pad_frames : pad_frames + T]

    return out.astype(np.float32)
