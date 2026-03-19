"""Skeleton definitions — joint names, hierarchy, offsets, and rendering data.

Why: Skeleton constants were hardcoded in mesh_viewport.py and
pose_corrector_panel.py, coupling view code to SMPL-X specifics. This module
provides a SkeletonDef dataclass with named presets (SMPLX_SKELETON,
SOMA_SKELETON) so the codebase can support multiple body model formats
without scattering magic numbers across view files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SkeletonDef:
    """Immutable skeleton definition for a body model format.

    All skeleton-specific constants live here: joint names, parent hierarchy,
    rest-pose offsets, bone connections, body region groupings, L/R swap pairs,
    and the joint color palette for viewport rendering.
    """

    name: str
    joint_names: tuple[str, ...]
    joint_parents: tuple[int, ...]
    default_offsets: dict[str, list[float]]
    bone_connections: list[tuple[int, int]]
    n_body_joints: int
    lr_swap_pairs: list[tuple[int, int]]
    lr_pairs: dict[int, int]
    joint_regions: dict[str, list[int]]
    joint_palette: np.ndarray  # (n_body_joints, 3) float32

    @property
    def n_joints(self) -> int:
        return len(self.joint_names)

    def __hash__(self) -> int:
        return hash(self.name)


# ---------------------------------------------------------------------------
# SMPL-X 52-joint skeleton
# ---------------------------------------------------------------------------

_SMPLX_JOINT_NAMES = (
    # Body (0-21)
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    # Left hand (22-36)
    "L_Index1", "L_Index2", "L_Index3",
    "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3",
    "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    # Right hand (37-51)
    "R_Index1", "R_Index2", "R_Index3",
    "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3",
    "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
)

_SMPLX_JOINT_PARENTS = (
    -1,  # 0  Pelvis (root)
    0, 0, 0,       # 1 L_Hip, 2 R_Hip, 3 Spine1
    1, 2, 3,       # 4 L_Knee, 5 R_Knee, 6 Spine2
    4, 5, 6,       # 7 L_Ankle, 8 R_Ankle, 9 Spine3
    7, 8,          # 10 L_Foot, 11 R_Foot
    9, 9, 9,       # 12 Neck, 13 L_Collar, 14 R_Collar
    12,            # 15 Head
    13, 14,        # 16 L_Shoulder, 17 R_Shoulder
    16, 17,        # 18 L_Elbow, 19 R_Elbow
    18, 19,        # 20 L_Wrist, 21 R_Wrist
    # Left hand: finger_base→wrist, then chain
    20, 22, 23,    # L_Index 1,2,3
    20, 25, 26,    # L_Middle 1,2,3
    20, 28, 29,    # L_Pinky 1,2,3
    20, 31, 32,    # L_Ring 1,2,3
    20, 34, 35,    # L_Thumb 1,2,3
    # Right hand
    21, 37, 38,    # R_Index 1,2,3
    21, 40, 41,    # R_Middle 1,2,3
    21, 43, 44,    # R_Pinky 1,2,3
    21, 46, 47,    # R_Ring 1,2,3
    21, 49, 50,    # R_Thumb 1,2,3
)

_SMPLX_DEFAULT_OFFSETS = {
    "Pelvis": [0.003, -0.351, 0.012],
    "L_Hip": [0.058, -0.093, -0.026], "R_Hip": [-0.063, -0.104, -0.021],
    "Spine1": [-0.003, 0.110, -0.028],
    "L_Knee": [0.055, -0.379, -0.009], "R_Knee": [-0.044, -0.362, -0.017],
    "Spine2": [0.009, 0.132, -0.006],
    "L_Ankle": [-0.043, -0.403, -0.032], "R_Ankle": [0.015, -0.411, -0.020],
    "Spine3": [-0.011, 0.052, 0.028],
    "L_Foot": [0.047, -0.058, 0.118], "R_Foot": [-0.039, -0.058, 0.119],
    "Neck": [-0.012, 0.165, -0.032],
    "L_Collar": [0.046, 0.085, -0.007], "R_Collar": [-0.048, 0.084, -0.013],
    "Head": [0.025, 0.160, 0.021],
    "L_Shoulder": [0.119, 0.058, -0.015], "R_Shoulder": [-0.103, 0.054, -0.013],
    "L_Elbow": [0.254, -0.072, -0.042], "R_Elbow": [-0.271, -0.036, -0.026],
    "L_Wrist": [0.252, 0.023, -0.002], "R_Wrist": [-0.249, -0.005, -0.015],
    # Left hand
    "L_Index1": [0.102, -0.009, 0.019], "L_Index2": [0.032, 0.002, 0.003],
    "L_Index3": [0.023, -0.002, 0.000],
    "L_Middle1": [0.109, -0.006, -0.004], "L_Middle2": [0.031, 0.001, -0.004],
    "L_Middle3": [0.024, -0.002, -0.004],
    "L_Pinky1": [0.084, -0.015, -0.044], "L_Pinky2": [0.015, -0.001, -0.012],
    "L_Pinky3": [0.016, -0.002, -0.011],
    "L_Ring1": [0.097, -0.009, -0.027], "L_Ring2": [0.028, 0.001, -0.005],
    "L_Ring3": [0.023, -0.001, -0.007],
    "L_Thumb1": [0.041, -0.018, 0.026], "L_Thumb2": [0.017, 0.001, 0.025],
    "L_Thumb3": [0.021, -0.005, 0.016],
    # Right hand
    "R_Index1": [-0.100, -0.012, 0.020], "R_Index2": [-0.032, 0.002, 0.003],
    "R_Index3": [-0.023, -0.002, 0.000],
    "R_Middle1": [-0.107, -0.009, -0.004], "R_Middle2": [-0.031, 0.001, -0.004],
    "R_Middle3": [-0.024, -0.002, -0.004],
    "R_Pinky1": [-0.082, -0.018, -0.044], "R_Pinky2": [-0.015, -0.001, -0.012],
    "R_Pinky3": [-0.016, -0.002, -0.011],
    "R_Ring1": [-0.095, -0.012, -0.027], "R_Ring2": [-0.028, 0.001, -0.005],
    "R_Ring3": [-0.023, -0.001, -0.007],
    "R_Thumb1": [-0.039, -0.021, 0.026], "R_Thumb2": [-0.017, 0.001, 0.025],
    "R_Thumb3": [-0.021, -0.005, 0.016],
}

# Body bone connections (indices 0-21 only)
_SMPLX_BONE_CONNECTIONS: list[tuple[int, int]] = [
    # Spine chain
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left leg
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Right leg
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Left arm
    (9, 13), (13, 16), (16, 18), (18, 20),
    # Right arm
    (9, 14), (14, 17), (17, 19), (19, 21),
]

# Add hand bones programmatically
for _wrist, _start_idx in [(20, 22), (21, 37)]:
    for _finger_base in range(_start_idx, _start_idx + 15, 3):
        _SMPLX_BONE_CONNECTIONS.append((_wrist, _finger_base))
        _SMPLX_BONE_CONNECTIONS.append((_finger_base, _finger_base + 1))
        _SMPLX_BONE_CONNECTIONS.append((_finger_base + 1, _finger_base + 2))

_SMPLX_LR_SWAP_PAIRS = [
    (1, 2),    # L_Hip <-> R_Hip
    (4, 5),    # L_Knee <-> R_Knee
    (7, 8),    # L_Ankle <-> R_Ankle
    (10, 11),  # L_Foot <-> R_Foot
    (13, 14),  # L_Collar <-> R_Collar
    (16, 17),  # L_Shoulder <-> R_Shoulder
    (18, 19),  # L_Elbow <-> R_Elbow
    (20, 21),  # L_Wrist <-> R_Wrist
]

_SMPLX_LR_PAIRS = {
    1: 2, 2: 1,       # L_Hip <-> R_Hip
    4: 5, 5: 4,       # L_Knee <-> R_Knee
    7: 8, 8: 7,       # L_Ankle <-> R_Ankle
    10: 11, 11: 10,   # L_Foot <-> R_Foot
    13: 14, 14: 13,   # L_Collar <-> R_Collar
    16: 17, 17: 16,   # L_Shoulder <-> R_Shoulder
    18: 19, 19: 18,   # L_Elbow <-> R_Elbow
    20: 21, 21: 20,   # L_Wrist <-> R_Wrist
}

_SMPLX_JOINT_REGIONS = {
    "spine": [0, 3, 6, 9, 12, 15],
    "left_leg": [1, 4, 7, 10],
    "right_leg": [2, 5, 8, 11],
    "left_arm": [13, 16, 18, 20],
    "right_arm": [14, 17, 19, 21],
    "left_hand": list(range(22, 37)),
    "right_hand": list(range(37, 52)),
}

_SMPLX_JOINT_PALETTE = np.array([
    [0.90, 0.10, 0.10],  # 0  Pelvis — red
    [0.10, 0.72, 0.30],  # 1  L_Hip — green
    [0.30, 0.10, 0.72],  # 2  R_Hip — purple
    [0.90, 0.50, 0.10],  # 3  Spine1 — orange
    [0.10, 0.90, 0.50],  # 4  L_Knee — teal
    [0.50, 0.10, 0.90],  # 5  R_Knee — violet
    [0.90, 0.90, 0.10],  # 6  Spine2 — yellow
    [0.10, 0.70, 0.90],  # 7  L_Ankle — sky blue
    [0.72, 0.10, 0.60],  # 8  R_Ankle — magenta
    [0.60, 0.80, 0.20],  # 9  Spine3 — lime
    [0.20, 0.50, 0.80],  # 10 L_Foot — blue
    [0.80, 0.30, 0.50],  # 11 R_Foot — rose
    [0.40, 0.90, 0.90],  # 12 Neck — cyan
    [0.90, 0.60, 0.40],  # 13 L_Collar — peach
    [0.60, 0.40, 0.90],  # 14 R_Collar — lavender
    [0.90, 0.20, 0.50],  # 15 Head — crimson
    [0.20, 0.90, 0.20],  # 16 L_Shoulder — bright green
    [0.20, 0.20, 0.90],  # 17 R_Shoulder — bright blue
    [0.80, 0.80, 0.20],  # 18 L_Elbow — gold
    [0.20, 0.80, 0.80],  # 19 R_Elbow — aqua
    [0.90, 0.50, 0.70],  # 20 L_Wrist — pink
    [0.50, 0.90, 0.70],  # 21 R_Wrist — mint
], dtype=np.float32)

SMPLX_SKELETON = SkeletonDef(
    name="smplx_52",
    joint_names=_SMPLX_JOINT_NAMES,
    joint_parents=_SMPLX_JOINT_PARENTS,
    default_offsets=_SMPLX_DEFAULT_OFFSETS,
    bone_connections=_SMPLX_BONE_CONNECTIONS,
    n_body_joints=22,
    lr_swap_pairs=_SMPLX_LR_SWAP_PAIRS,
    lr_pairs=_SMPLX_LR_PAIRS,
    joint_regions=_SMPLX_JOINT_REGIONS,
    joint_palette=_SMPLX_JOINT_PALETTE,
)


# ---------------------------------------------------------------------------
# SOMA 77-joint skeleton (placeholder — real data from py-soma-x at runtime)
# ---------------------------------------------------------------------------

# Placeholder: 77 joints. The first 22 body joints share names with SMPL-X.
# Joints 22-76 are placeholder names that will be replaced when py-soma-x
# is available and we can read soma.rig_data["joint_names"].
_SOMA_JOINT_NAMES = (
    # Body (0-21) — same naming as SMPL-X
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    # Left hand (22-36) — 15 joints
    "L_Index1", "L_Index2", "L_Index3",
    "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3",
    "L_Ring1", "L_Ring2", "L_Ring3",
    "L_Thumb1", "L_Thumb2", "L_Thumb3",
    # Right hand (37-51) — 15 joints
    "R_Index1", "R_Index2", "R_Index3",
    "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3",
    "R_Ring1", "R_Ring2", "R_Ring3",
    "R_Thumb1", "R_Thumb2", "R_Thumb3",
    # Face (52-76) — 25 placeholder joints
    "Jaw",
    "L_Eye", "R_Eye",
    "L_Brow_Inner", "L_Brow_Mid", "L_Brow_Outer",
    "R_Brow_Inner", "R_Brow_Mid", "R_Brow_Outer",
    "L_Cheek", "R_Cheek",
    "Nose_Tip",
    "Upper_Lip", "Lower_Lip",
    "L_Lip_Corner", "R_Lip_Corner",
    "L_Eyelid_Upper", "L_Eyelid_Lower",
    "R_Eyelid_Upper", "R_Eyelid_Lower",
    "L_Nostril", "R_Nostril",
    "Chin",
    "L_Ear", "R_Ear",
)

# Placeholder parents — body matches SMPL-X, face joints parent to Head (15)
_SOMA_JOINT_PARENTS = (
    -1,  # 0  Pelvis (root)
    0, 0, 0,       # 1-3
    1, 2, 3,       # 4-6
    4, 5, 6,       # 7-9
    7, 8,          # 10-11
    9, 9, 9,       # 12-14
    12,            # 15 Head
    13, 14,        # 16-17
    16, 17,        # 18-19
    18, 19,        # 20-21
    # Left hand
    20, 22, 23,
    20, 25, 26,
    20, 28, 29,
    20, 31, 32,
    20, 34, 35,
    # Right hand
    21, 37, 38,
    21, 40, 41,
    21, 43, 44,
    21, 46, 47,
    21, 49, 50,
    # Face — all parented to Head (15)
    15,            # 52 Jaw
    15, 15,        # 53-54 Eyes
    15, 15, 15,    # 55-57 L_Brow
    15, 15, 15,    # 58-60 R_Brow
    15, 15,        # 61-62 Cheeks
    15,            # 63 Nose_Tip
    52, 52,        # 64-65 Lips (parent to Jaw)
    52, 52,        # 66-67 Lip corners (parent to Jaw)
    53, 53,        # 68-69 L eyelids (parent to L_Eye)
    54, 54,        # 70-71 R eyelids (parent to R_Eye)
    15, 15,        # 72-73 Nostrils
    52,            # 74 Chin (parent to Jaw)
    15, 15,        # 75-76 Ears
)

# Placeholder offsets — zeros for face joints, body matches SMPL-X
_SOMA_DEFAULT_OFFSETS = dict(_SMPLX_DEFAULT_OFFSETS)
for _name in _SOMA_JOINT_NAMES[52:]:
    _SOMA_DEFAULT_OFFSETS[_name] = [0.0, 0.0, 0.0]

# Bone connections — body + hand from SMPL-X, plus face bones
_SOMA_BONE_CONNECTIONS: list[tuple[int, int]] = list(_SMPLX_BONE_CONNECTIONS)
# Face bones: Head→Jaw, Head→Eyes, Jaw→Lips, etc.
_SOMA_BONE_CONNECTIONS.extend([
    (15, 52),  # Head → Jaw
    (15, 53), (15, 54),  # Head → Eyes
    (52, 64), (52, 65),  # Jaw → Lips
    (52, 66), (52, 67),  # Jaw → Lip corners
    (53, 68), (53, 69),  # L_Eye → eyelids
    (54, 70), (54, 71),  # R_Eye → eyelids
    (52, 74),  # Jaw → Chin
])

# SOMA uses same L/R body swap pairs as SMPL-X, plus face pairs
_SOMA_LR_SWAP_PAIRS = list(_SMPLX_LR_SWAP_PAIRS) + [
    (53, 54),   # L_Eye <-> R_Eye
    (55, 58),   # L_Brow_Inner <-> R_Brow_Inner
    (56, 59),   # L_Brow_Mid <-> R_Brow_Mid
    (57, 60),   # L_Brow_Outer <-> R_Brow_Outer
    (61, 62),   # L_Cheek <-> R_Cheek
    (66, 67),   # L_Lip_Corner <-> R_Lip_Corner
    (68, 70),   # L_Eyelid_Upper <-> R_Eyelid_Upper
    (69, 71),   # L_Eyelid_Lower <-> R_Eyelid_Lower
    (72, 73),   # L_Nostril <-> R_Nostril
    (75, 76),   # L_Ear <-> R_Ear
]

_SOMA_LR_PAIRS = dict(_SMPLX_LR_PAIRS)
for _l, _r in _SOMA_LR_SWAP_PAIRS:
    _SOMA_LR_PAIRS[_l] = _r
    _SOMA_LR_PAIRS[_r] = _l

_SOMA_JOINT_REGIONS = dict(_SMPLX_JOINT_REGIONS)
_SOMA_JOINT_REGIONS["face"] = list(range(52, 77))

SOMA_SKELETON = SkeletonDef(
    name="soma_77",
    joint_names=_SOMA_JOINT_NAMES,
    joint_parents=_SOMA_JOINT_PARENTS,
    default_offsets=_SOMA_DEFAULT_OFFSETS,
    bone_connections=_SOMA_BONE_CONNECTIONS,
    n_body_joints=22,
    lr_swap_pairs=_SOMA_LR_SWAP_PAIRS,
    lr_pairs=_SOMA_LR_PAIRS,
    joint_regions=_SOMA_JOINT_REGIONS,
    joint_palette=_SMPLX_JOINT_PALETTE,  # same 22-color body palette
)
