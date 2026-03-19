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
# SOMA 77-joint skeleton (from kimodo.skeleton.SOMASkeleton77)
# ---------------------------------------------------------------------------

_SOMA_JOINT_NAMES = (
    # Spine / head (0-10)
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd",
    "Jaw", "LeftEye", "RightEye",
    # Left arm (11-14)
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    # Left fingers (15-38) — thumb(4) + index(5) + middle(5) + ring(5) + pinky(5) = 24
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumb4",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndex5",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddle5",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRing5",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinky5",
    # Right arm (39-42)
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    # Right fingers (43-66) — thumb(4) + index(5) + middle(5) + ring(5) + pinky(5) = 24
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumb4",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndex5",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddle5",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRing5",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinky5",
    # Left leg (67-71)
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    # Right leg (72-76)
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
)

_SOMA_JOINT_PARENTS = (
    -1,  # 0  Hips (root)
    0, 1, 2,          # 1 Spine1, 2 Spine2, 3 Chest
    3, 4, 5,          # 4 Neck1, 5 Neck2, 6 Head
    6, 6, 6, 6,       # 7 HeadEnd, 8 Jaw, 9 LeftEye, 10 RightEye
    3,                # 11 LeftShoulder
    11, 12, 13,       # 12 LeftArm, 13 LeftForeArm, 14 LeftHand
    # Left thumb (15-18): 14→15→16→17→18
    14, 15, 16, 17,
    # Left index (19-23): 14→19→20→21→22→23
    14, 19, 20, 21, 22,
    # Left middle (24-28): 14→24→25→26→27→28
    14, 24, 25, 26, 27,
    # Left ring (29-33): 14→29→30→31→32→33
    14, 29, 30, 31, 32,
    # Left pinky (34-38): 14→34→35→36→37→38
    14, 34, 35, 36, 37,
    3,                # 39 RightShoulder
    39, 40, 41,       # 40 RightArm, 41 RightForeArm, 42 RightHand
    # Right thumb (43-46): 42→43→44→45→46
    42, 43, 44, 45,
    # Right index (47-51): 42→47→48→49→50→51
    42, 47, 48, 49, 50,
    # Right middle (52-56): 42→52→53→54→55→56
    42, 52, 53, 54, 55,
    # Right ring (57-61): 42→57→58→59→60→61
    42, 57, 58, 59, 60,
    # Right pinky (62-66): 42→62→63→64→65→66
    42, 62, 63, 64, 65,
    0,                # 67 LeftLeg
    67, 68, 69, 70,   # 68 LeftShin, 69 LeftFoot, 70 LeftToeBase, 71 LeftToeEnd
    0,                # 72 RightLeg
    72, 73, 74, 75,   # 73 RightShin, 74 RightFoot, 75 RightToeBase, 76 RightToeEnd
)

# Parent-relative offsets from SOMASkeleton77.neutral_joints (meters)
_SOMA_DEFAULT_OFFSETS = {
    "Hips": [0.000, 0.923, 0.000],
    "Spine1": [0.000, 0.103, -0.012],
    "Spine2": [0.000, 0.104, 0.003],
    "Chest": [0.000, 0.137, 0.015],
    "Neck1": [0.000, 0.146, -0.019],
    "Neck2": [0.000, 0.057, 0.002],
    "Head": [0.000, 0.076, 0.017],
    "HeadEnd": [0.000, 0.159, 0.029],
    "Jaw": [0.000, 0.012, 0.064],
    "LeftEye": [0.032, 0.074, 0.066],
    "RightEye": [-0.032, 0.074, 0.066],
    "LeftShoulder": [0.040, 0.114, -0.005],
    "LeftArm": [0.127, 0.031, -0.011],
    "LeftForeArm": [0.262, -0.017, -0.014],
    "LeftHand": [0.249, 0.001, 0.003],
    "LeftHandThumb1": [0.035, -0.010, 0.024],
    "LeftHandThumb2": [0.025, -0.003, 0.019],
    "LeftHandThumb3": [0.020, -0.002, 0.014],
    "LeftHandThumb4": [0.016, -0.002, 0.011],
    "LeftHandIndex1": [0.093, -0.006, 0.021],
    "LeftHandIndex2": [0.033, 0.001, 0.003],
    "LeftHandIndex3": [0.023, -0.001, 0.001],
    "LeftHandIndex4": [0.018, -0.001, 0.000],
    "LeftHandIndex5": [0.014, -0.001, 0.000],
    "LeftHandMiddle1": [0.098, -0.003, 0.001],
    "LeftHandMiddle2": [0.034, 0.001, -0.002],
    "LeftHandMiddle3": [0.024, -0.001, -0.002],
    "LeftHandMiddle4": [0.019, -0.001, -0.001],
    "LeftHandMiddle5": [0.015, -0.001, -0.001],
    "LeftHandRing1": [0.088, -0.007, -0.020],
    "LeftHandRing2": [0.030, 0.001, -0.004],
    "LeftHandRing3": [0.022, -0.001, -0.003],
    "LeftHandRing4": [0.017, -0.001, -0.003],
    "LeftHandRing5": [0.013, -0.001, -0.002],
    "LeftHandPinky1": [0.076, -0.012, -0.038],
    "LeftHandPinky2": [0.020, -0.001, -0.006],
    "LeftHandPinky3": [0.015, -0.001, -0.005],
    "LeftHandPinky4": [0.012, -0.001, -0.004],
    "LeftHandPinky5": [0.010, -0.001, -0.003],
    "RightShoulder": [-0.040, 0.114, -0.005],
    "RightArm": [-0.127, 0.031, -0.011],
    "RightForeArm": [-0.262, -0.017, -0.014],
    "RightHand": [-0.249, 0.001, 0.003],
    "RightHandThumb1": [-0.035, -0.010, 0.024],
    "RightHandThumb2": [-0.025, -0.003, 0.019],
    "RightHandThumb3": [-0.020, -0.002, 0.014],
    "RightHandThumb4": [-0.016, -0.002, 0.011],
    "RightHandIndex1": [-0.093, -0.006, 0.021],
    "RightHandIndex2": [-0.033, 0.001, 0.003],
    "RightHandIndex3": [-0.023, -0.001, 0.001],
    "RightHandIndex4": [-0.018, -0.001, 0.000],
    "RightHandIndex5": [-0.014, -0.001, 0.000],
    "RightHandMiddle1": [-0.098, -0.003, 0.001],
    "RightHandMiddle2": [-0.034, 0.001, -0.002],
    "RightHandMiddle3": [-0.024, -0.001, -0.002],
    "RightHandMiddle4": [-0.019, -0.001, -0.001],
    "RightHandMiddle5": [-0.015, -0.001, -0.001],
    "RightHandRing1": [-0.088, -0.007, -0.020],
    "RightHandRing2": [-0.030, 0.001, -0.004],
    "RightHandRing3": [-0.022, -0.001, -0.003],
    "RightHandRing4": [-0.017, -0.001, -0.003],
    "RightHandRing5": [-0.013, -0.001, -0.002],
    "RightHandPinky1": [-0.076, -0.012, -0.038],
    "RightHandPinky2": [-0.020, -0.001, -0.006],
    "RightHandPinky3": [-0.015, -0.001, -0.005],
    "RightHandPinky4": [-0.012, -0.001, -0.004],
    "RightHandPinky5": [-0.010, -0.001, -0.003],
    "LeftLeg": [0.075, -0.045, 0.001],
    "LeftShin": [0.013, -0.397, -0.013],
    "LeftFoot": [-0.007, -0.398, -0.027],
    "LeftToeBase": [0.020, -0.058, 0.108],
    "LeftToeEnd": [0.000, -0.010, 0.060],
    "RightLeg": [-0.075, -0.045, 0.001],
    "RightShin": [-0.013, -0.397, -0.013],
    "RightFoot": [0.007, -0.398, -0.027],
    "RightToeBase": [-0.020, -0.058, 0.108],
    "RightToeEnd": [0.000, -0.010, 0.060],
}

# Bone connections derived from parent hierarchy (every parent→child pair)
_SOMA_BONE_CONNECTIONS: list[tuple[int, int]] = [
    (_SOMA_JOINT_PARENTS[i], i) for i in range(1, 77)
]

_SOMA_LR_SWAP_PAIRS = [
    (9, 10),    # LeftEye <-> RightEye
    (11, 39),   # LeftShoulder <-> RightShoulder
    (12, 40),   # LeftArm <-> RightArm
    (13, 41),   # LeftForeArm <-> RightForeArm
    (14, 42),   # LeftHand <-> RightHand
] + [
    (15 + i, 43 + i) for i in range(24)  # Left fingers 15-38 <-> Right fingers 43-66
] + [
    (67, 72),   # LeftLeg <-> RightLeg
    (68, 73),   # LeftShin <-> RightShin
    (69, 74),   # LeftFoot <-> RightFoot
    (70, 75),   # LeftToeBase <-> RightToeBase
    (71, 76),   # LeftToeEnd <-> RightToeEnd
]

_SOMA_LR_PAIRS: dict[int, int] = {}
for _l, _r in _SOMA_LR_SWAP_PAIRS:
    _SOMA_LR_PAIRS[_l] = _r
    _SOMA_LR_PAIRS[_r] = _l

_SOMA_JOINT_REGIONS = {
    "spine": [0, 1, 2, 3],
    "neck_head": [4, 5, 6, 7, 8, 9, 10],
    "left_arm": [11, 12, 13, 14],
    "right_arm": [39, 40, 41, 42],
    "left_hand": list(range(15, 39)),
    "right_hand": list(range(43, 67)),
    "left_leg": [67, 68, 69, 70, 71],
    "right_leg": [72, 73, 74, 75, 76],
}

_SOMA_JOINT_PALETTE = np.array([
    [0.90, 0.10, 0.10],  # 0  Hips — red
    [0.90, 0.50, 0.10],  # 1  Spine1 — orange
    [0.90, 0.90, 0.10],  # 2  Spine2 — yellow
    [0.60, 0.80, 0.20],  # 3  Chest — lime
    [0.40, 0.90, 0.90],  # 4  Neck1 — cyan
    [0.20, 0.80, 0.80],  # 5  Neck2 — aqua
    [0.90, 0.20, 0.50],  # 6  Head — crimson
    [0.80, 0.30, 0.50],  # 7  HeadEnd — rose
    [0.72, 0.10, 0.60],  # 8  Jaw — magenta
    [0.10, 0.72, 0.30],  # 9  LeftEye — green
    [0.30, 0.10, 0.72],  # 10 RightEye — purple
    [0.90, 0.60, 0.40],  # 11 LeftShoulder — peach
    [0.20, 0.90, 0.20],  # 12 LeftArm — bright green
    [0.80, 0.80, 0.20],  # 13 LeftForeArm — gold
    [0.90, 0.50, 0.70],  # 14 LeftHand — pink
], dtype=np.float32)

SOMA_SKELETON = SkeletonDef(
    name="soma_77",
    joint_names=_SOMA_JOINT_NAMES,
    joint_parents=_SOMA_JOINT_PARENTS,
    default_offsets=_SOMA_DEFAULT_OFFSETS,
    bone_connections=_SOMA_BONE_CONNECTIONS,
    n_body_joints=15,
    lr_swap_pairs=_SOMA_LR_SWAP_PAIRS,
    lr_pairs=_SOMA_LR_PAIRS,
    joint_regions=_SOMA_JOINT_REGIONS,
    joint_palette=_SOMA_JOINT_PALETTE,
)
