"""Axis-angle / quaternion / log-quaternion helpers for the spring filter.

The spring filter runs in quaternion log space — equivalently, half the
axis-angle representation. Filtering each of the three log-vector
components independently is exact when the rotations stay near identity
(the 99% case for body joints) and a very close approximation elsewhere
for body rotations that never approach the antipode.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def aa_sequence_to_log(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle (T, J, 3) to continuous quaternion-log (T, J, 3).

    - Converts per-frame AA to unit quats via scipy.
    - Runs a hemisphere-continuity pass along T so consecutive quats
      have positive dot product (eliminates sign-flip discontinuities).
    - Takes the half-axis-angle log: log(q) = (theta/2) * axis.
    """
    aa = np.asarray(aa, dtype=np.float32)
    if aa.ndim != 3 or aa.shape[-1] != 3:
        raise ValueError(f"aa must have shape (T, J, 3); got {aa.shape}")
    T, J, _ = aa.shape
    if T == 0:
        return np.zeros((0, J, 3), dtype=np.float32)

    # AA -> unit quat (xyzw)
    q = Rotation.from_rotvec(aa.reshape(T * J, 3)).as_quat().reshape(T, J, 4)

    # Hemisphere continuity along T, per joint.
    if T > 1:
        dots = np.sum(q[1:] * q[:-1], axis=-1, keepdims=True)  # (T-1, J, 1)
        signs = np.ones((T, J, 1), dtype=q.dtype)
        signs[1:] = np.cumprod(np.where(dots < 0, -1.0, 1.0), axis=0)
        q = q * signs

    # Log map (half-angle form): half_angle = atan2(|xyz|, w) in [0, pi],
    # then factor = half_angle / |xyz| so result = xyz * factor.
    xyz = q[..., :3]
    w = q[..., 3:4]
    xyz_norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    half_angle = np.arctan2(xyz_norm, np.clip(w, -1.0, 1.0))
    factor = np.where(xyz_norm > 1e-8, half_angle / np.maximum(xyz_norm, 1e-8), 1.0)
    return (xyz * factor).astype(np.float32)


def log_sequence_to_aa(log_vec: np.ndarray) -> np.ndarray:
    """Inverse of ``aa_sequence_to_log``: (T, J, 3) log-quat -> (T, J, 3) AA."""
    log_vec = np.asarray(log_vec, dtype=np.float32)
    if log_vec.ndim != 3 or log_vec.shape[-1] != 3:
        raise ValueError(
            f"log_vec must have shape (T, J, 3); got {log_vec.shape}"
        )
    T, J, _ = log_vec.shape
    if T == 0:
        return np.zeros((0, J, 3), dtype=np.float32)

    half_angle = np.linalg.norm(log_vec, axis=-1, keepdims=True)
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)
    factor = np.where(
        half_angle > 1e-8, sin_half / np.maximum(half_angle, 1e-8), 1.0
    )
    xyz = log_vec * factor
    q = np.concatenate([xyz, cos_half], axis=-1)
    # Normalize (scipy will complain otherwise on near-zero vectors).
    q_norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(q_norm, 1e-12)
    return Rotation.from_quat(q.reshape(T * J, 4)).as_rotvec().reshape(T, J, 3).astype(
        np.float32
    )
