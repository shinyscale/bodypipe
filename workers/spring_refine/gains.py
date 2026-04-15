"""Per-joint spring gain tables and preset scaling.

Kp is spring stiffness (units 1/s^2), Kv is damping (units 1/s). For a
second-order spring-damper system x'' = Kp*(target - x) - Kv*x', the
damping ratio is zeta = Kv / (2 * sqrt(Kp)). Values below are
underdamped (zeta ~0.3–0.5) to inject follow-through and momentum feel
into floaty mocap input. Proximal joints sit around zeta≈0.45–0.49 for
solid follow-through; extremities drop to zeta≈0.33–0.35 for visible
whip on wrists and feet.

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
        (300.0, 17.0),  #  0: L_Hip         zeta=0.491
        (300.0, 17.0),  #  1: R_Hip         zeta=0.491
        (300.0, 17.0),  #  2: Spine1 (Torso) zeta=0.491
        (180.0, 12.0),  #  3: L_Knee        zeta=0.447
        (180.0, 12.0),  #  4: R_Knee        zeta=0.447
        (260.0, 15.0),  #  5: Spine2        zeta=0.465
        (80.0,   6.0),  #  6: L_Ankle       zeta=0.335
        (80.0,   6.0),  #  7: R_Ankle       zeta=0.335
        (220.0, 14.0),  #  8: Spine3 (Chest) zeta=0.472
        (50.0,   5.0),  #  9: L_Foot        zeta=0.354
        (50.0,   5.0),  # 10: R_Foot        zeta=0.354
        (150.0, 11.0),  # 11: Neck          zeta=0.449
        (220.0, 14.0),  # 12: L_Collar      zeta=0.472
        (220.0, 14.0),  # 13: R_Collar      zeta=0.472
        (150.0, 11.0),  # 14: Head          zeta=0.449
        (200.0, 13.0),  # 15: L_Shoulder    zeta=0.460
        (200.0, 13.0),  # 16: R_Shoulder    zeta=0.460
        (150.0, 11.0),  # 17: L_Elbow       zeta=0.449
        (150.0, 11.0),  # 18: R_Elbow       zeta=0.449
        (80.0,   6.0),  # 19: L_Wrist       zeta=0.335
        (80.0,   6.0),  # 20: R_Wrist       zeta=0.335
    ],
    dtype=np.float32,
)

_BASE_ROOT_GAINS = np.array([250.0, 15.0], dtype=np.float32)

# kp_scale: higher = stiffer = tracks input faster = less weight effect.
_PRESET_SCALES: dict[str, float] = {
    "light": 4.0,      # subtle weight — barely perceptible follow-through
    "moderate": 1.5,    # clear weight — visible momentum on fast moves
    "heavy": 0.5,       # exaggerated weight — slow, heavy, deliberate motion
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
