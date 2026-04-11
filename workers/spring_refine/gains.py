"""Per-joint spring gain tables and preset scaling.

Kp is spring stiffness (units 1/s^2), Kv is damping (units 1/s). For a
second-order spring-damper system x'' = Kp*(target - x) - Kv*x', the
damping ratio is zeta = Kv / (2 * sqrt(Kp)). Values below target zeta
approx 1.0 (critically damped) at proximal joints, with lighter damping
at extremities to give a touch of follow-through without oscillation.

The preset multiplies Kp by kp_scale; Kv is scaled by sqrt(kp_scale) to
preserve the damping ratio exactly (since zeta is proportional to
Kv / sqrt(Kp)).
"""

from __future__ import annotations

import numpy as np

# Body joint order matches SMPL body_pose indices 0..20 (equivalent to
# full-SMPL joints 1..21 — the pelvis is carried separately as
# global_orient). (Kp, Kv) per joint.
_BASE_BODY_GAINS = np.array(
    [
        (300.0, 40.0),  #  0: L_Hip
        (300.0, 40.0),  #  1: R_Hip
        (300.0, 40.0),  #  2: Spine1 (Torso)
        (180.0, 28.0),  #  3: L_Knee
        (180.0, 28.0),  #  4: R_Knee
        (260.0, 36.0),  #  5: Spine2
        (80.0, 15.0),   #  6: L_Ankle
        (80.0, 15.0),   #  7: R_Ankle
        (220.0, 32.0),  #  8: Spine3 (Chest)
        (50.0, 10.0),   #  9: L_Foot
        (50.0, 10.0),   # 10: R_Foot
        (150.0, 25.0),  # 11: Neck
        (220.0, 32.0),  # 12: L_Collar
        (220.0, 32.0),  # 13: R_Collar
        (150.0, 25.0),  # 14: Head
        (200.0, 30.0),  # 15: L_Shoulder
        (200.0, 30.0),  # 16: R_Shoulder
        (150.0, 25.0),  # 17: L_Elbow
        (150.0, 25.0),  # 18: R_Elbow
        (80.0, 15.0),   # 19: L_Wrist
        (80.0, 15.0),   # 20: R_Wrist
    ],
    dtype=np.float32,
)

_BASE_ROOT_GAINS = np.array([250.0, 36.0], dtype=np.float32)

# kp_scale: higher = stiffer = less filtering (snappier).
_PRESET_SCALES: dict[str, float] = {
    "light": 2.0,
    "moderate": 1.0,
    "heavy": 0.5,
}


def get_preset_gains(
    preset: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return (body_kp, body_kv, root_kp, root_kv) scaled by *preset*.

    body_kp and body_kv are shape (21,). root_kp and root_kv are scalars.
    Kv is scaled by sqrt(kp_scale) so the damping ratio zeta is preserved
    across presets.
    """
    if preset not in _PRESET_SCALES:
        raise ValueError(
            f"Unknown spring preset: {preset!r} "
            f"(expected one of {sorted(_PRESET_SCALES)})"
        )
    kp_scale = _PRESET_SCALES[preset]
    kv_scale = float(np.sqrt(kp_scale))
    body_kp = (_BASE_BODY_GAINS[:, 0] * kp_scale).astype(np.float32)
    body_kv = (_BASE_BODY_GAINS[:, 1] * kv_scale).astype(np.float32)
    root_kp = float(_BASE_ROOT_GAINS[0] * kp_scale)
    root_kv = float(_BASE_ROOT_GAINS[1] * kv_scale)
    return body_kp, body_kv, root_kp, root_kv
