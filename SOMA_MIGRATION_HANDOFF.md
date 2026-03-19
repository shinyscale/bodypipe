# SOMA Migration — Phase 0 & 1 Handoff

**Target machine**: Powerhouse (`ssh powerhouse`, user `shinyscale`, repo at `~/bodypipe`)
**Branch**: Create `soma/skeleton-registry` from `main`
**Baseline**: 1507 tests pass, 38 skipped, 1 fail (missing torch — ignore)

---

## Overview

Phase 0 extracts all hardcoded SMPL-X skeleton constants from `views/mesh_viewport.py` and `views/pose_corrector_panel.py` into a new `models/skeleton.py` registry, and creates a `models/body_model.py` adapter protocol. Phase 1 adds `soma_params` and `body_model_type` to `PersonTrack`, adds new fields to `PipelineConfig`, and creates a stub `models/param_converter.py`.

**Zero behavior changes.** All existing tests must pass unchanged. New tests are added for new fields only.

---

## Step 0: Create branch

```bash
cd ~/bodypipe
git checkout main
git pull origin main
git checkout -b soma/skeleton-registry
```

---

## Step 1: Create `models/skeleton.py`

Create the file `models/skeleton.py` with the following exact content:

```python
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
```

**Verify**: `python -c "from models.skeleton import SMPLX_SKELETON, SOMA_SKELETON; print(f'SMPLX: {SMPLX_SKELETON.n_joints} joints'); print(f'SOMA: {SOMA_SKELETON.n_joints} joints')"`

Expected output:
```
SMPLX: 52 joints
SOMA: 77 joints
```

---

## Step 2: Create `models/body_model.py`

Create the file `models/body_model.py` with the following exact content:

```python
"""Body model adapters — protocol + implementations for SMPL-X and SOMA.

Why: The viewport, pose corrector, and export pipeline all need to compute
forward kinematics and access skeleton metadata. This protocol decouples
those consumers from the specific body model format, enabling SOMA support
alongside the existing SMPL-X path without touching consumer code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from models.skeleton import SkeletonDef, SMPLX_SKELETON, SOMA_SKELETON


@runtime_checkable
class BodyModelAdapter(Protocol):
    """Interface for body model forward passes."""

    @property
    def skeleton(self) -> SkeletonDef:
        """The skeleton definition for this body model."""
        ...

    def forward_kinematics(self, params: dict, frame_idx: int) -> np.ndarray:
        """Compute joint positions for one frame.

        Returns (n_joints, 3) float64 positions in camera/world space.
        """
        ...


class SmplxAdapter:
    """SMPL-X body model adapter wrapping the existing FK implementation."""

    @property
    def skeleton(self) -> SkeletonDef:
        return SMPLX_SKELETON

    def forward_kinematics(self, params: dict, frame_idx: int) -> np.ndarray:
        # Import here to avoid circular dependency (mesh_viewport imports skeleton)
        from views.mesh_viewport import forward_kinematics
        return forward_kinematics(params, frame_idx)


class SomaAdapter:
    """SOMA body model adapter — placeholder until py-soma-x is available."""

    @property
    def skeleton(self) -> SkeletonDef:
        return SOMA_SKELETON

    def forward_kinematics(self, params: dict, frame_idx: int) -> np.ndarray:
        raise NotImplementedError(
            "SOMA forward kinematics requires py-soma-x (Phase 3)"
        )
```

**Verify**: `python -c "from models.body_model import SmplxAdapter, SomaAdapter, BodyModelAdapter; a = SmplxAdapter(); print(f'SmplxAdapter implements protocol: {isinstance(a, BodyModelAdapter)}'); print(f'Skeleton: {a.skeleton.name}')"`

Expected:
```
SmplxAdapter implements protocol: True
Skeleton: smplx_52
```

---

## Step 3: Modify `views/mesh_viewport.py`

Replace the inline skeleton constants with imports from `models/skeleton.py`. The goal is to delete lines 124-302 (the constants section) and replace with imports + module-level aliases.

### 3a. Find this block (lines ~124-127, the section header + start of JOINT_NAMES):

```python
# ---------------------------------------------------------------------------
# Skeleton data — 52-joint SMPL-X hierarchy (from smplx_to_bvh.py)
# ---------------------------------------------------------------------------

JOINT_NAMES = [
```

Replace **everything** from that section header through the end of `_JOINT_PALETTE` (line 302, ending with `], dtype=np.float32)`) with:

```python
# ---------------------------------------------------------------------------
# Skeleton data — sourced from models/skeleton.py (single source of truth)
# ---------------------------------------------------------------------------

from models.skeleton import SMPLX_SKELETON as _SKEL

# Module-level aliases for backward compatibility — existing code and tests
# import these names directly from this module.
JOINT_NAMES = list(_SKEL.joint_names)
JOINT_PARENTS = list(_SKEL.joint_parents)
DEFAULT_OFFSETS = _SKEL.default_offsets
BONE_CONNECTIONS = list(_SKEL.bone_connections)
_N_BODY_JOINTS = _SKEL.n_body_joints
_JOINT_PALETTE = _SKEL.joint_palette
```

**What to delete**: Everything from line 128 (`JOINT_NAMES = [`) through line 302 (`], dtype=np.float32)`) inclusive. That's the JOINT_NAMES list, JOINT_PARENTS list, DEFAULT_OFFSETS dict, BONE_CONNECTIONS list, the hand bone generation loop (lines 227-231), and _N_BODY_JOINTS assignment, plus _JOINT_PALETTE. The section header comment (lines 124-126) gets replaced too.

**What to keep**: Everything from `_JOINT_PICK_THRESHOLD` (currently line 237) onward stays exactly as-is. The color constants (_ACCENT_COLOR, _BONE_COLOR, etc.), label constants, point sizes — none of those move.

### 3b. Replace the `_LR_PAIRS` and `_JOINT_REGIONS` block

Find this block (currently lines ~488-510):

```python
# Left↔Right joint index mapping for body joints (0-21).
# Hand joints (22-36 ↔ 37-51) are offset by 15.
_LR_PAIRS = {
    1: 2, 2: 1,       # L_Hip ↔ R_Hip
    4: 5, 5: 4,       # L_Knee ↔ R_Knee
    7: 8, 8: 7,       # L_Ankle ↔ R_Ankle
    10: 11, 11: 10,   # L_Foot ↔ R_Foot
    13: 14, 14: 13,   # L_Collar ↔ R_Collar
    16: 17, 17: 16,   # L_Shoulder ↔ R_Shoulder
    18: 19, 19: 18,   # L_Elbow ↔ R_Elbow
    20: 21, 21: 20,   # L_Wrist ↔ R_Wrist
}

# Body regions — groups of joint indices sharing a kinematic purpose.
_JOINT_REGIONS = {
    "spine": [0, 3, 6, 9, 12, 15],
    "left_leg": [1, 4, 7, 10],
    "right_leg": [2, 5, 8, 11],
    "left_arm": [13, 16, 18, 20],
    "right_arm": [14, 17, 19, 21],
    "left_hand": list(range(22, 37)),
    "right_hand": list(range(37, 52)),
}
```

Replace with:

```python
# Left↔Right and region data — sourced from skeleton registry.
_LR_PAIRS = _SKEL.lr_pairs
_JOINT_REGIONS = _SKEL.joint_regions
```

### 3c. Important: `_JOINT_TO_REGION` derivation stays

The `_JOINT_TO_REGION` dict (currently lines ~512-516) that iterates over `_JOINT_REGIONS` to build the reverse lookup should remain exactly as-is — it already reads from `_JOINT_REGIONS` which now delegates to the skeleton:

```python
# Reverse lookup: joint index → region name
_JOINT_TO_REGION: dict[int, str] = {}
for _region_name, _region_joints in _JOINT_REGIONS.items():
    for _j in _region_joints:
        _JOINT_TO_REGION[_j] = _region_name
```

This block stays unchanged.

### 3d. Verify no line number drift

After edits, the file should be ~180 lines shorter than before (the deleted constants). Run:

```bash
python -c "from views.mesh_viewport import JOINT_NAMES, JOINT_PARENTS, DEFAULT_OFFSETS, BONE_CONNECTIONS, _N_BODY_JOINTS, _JOINT_PALETTE, _LR_PAIRS, _JOINT_REGIONS, _JOINT_TO_REGION, forward_kinematics; print(f'JOINT_NAMES: {len(JOINT_NAMES)} entries'); print(f'JOINT_PARENTS: {len(JOINT_PARENTS)} entries'); print(f'DEFAULT_OFFSETS: {len(DEFAULT_OFFSETS)} entries'); print(f'BONE_CONNECTIONS: {len(BONE_CONNECTIONS)} entries'); print(f'_N_BODY_JOINTS: {_N_BODY_JOINTS}'); print(f'_JOINT_PALETTE shape: {_JOINT_PALETTE.shape}'); print(f'_LR_PAIRS: {len(_LR_PAIRS)} entries'); print(f'_JOINT_REGIONS: {len(_JOINT_REGIONS)} entries'); print(f'_JOINT_TO_REGION: {len(_JOINT_TO_REGION)} entries')"
```

Expected output:
```
JOINT_NAMES: 52 entries
JOINT_PARENTS: 52 entries
DEFAULT_OFFSETS: 52 entries
BONE_CONNECTIONS: 51 entries
_N_BODY_JOINTS: 22
_JOINT_PALETTE shape: (22, 3)
_LR_PAIRS: 16 entries
_JOINT_REGIONS: 7 entries
_JOINT_TO_REGION: 52 entries
```

(51 bone connections = 17 body + 10 fingers × 3 + 4 wrist-to-finger — verify this matches pre-edit count)

---

## Step 4: Modify `views/pose_corrector_panel.py`

### 4a. Replace the constants block

Find these lines (currently lines 55-72):

```python
from views.mesh_viewport import MeshViewport, JOINT_NAMES, JOINT_PARENTS

log = logging.getLogger(__name__)

# Number of body joints (0-21); hand joints are 22-51
_N_BODY_JOINTS = 22

# Left/Right body joint swap pairs for mirroring (joint indices, not body_pose indices)
_LR_SWAP_PAIRS = [
    (1, 2),    # L_Hip <-> R_Hip
    (4, 5),    # L_Knee <-> R_Knee
    (7, 8),    # L_Ankle <-> R_Ankle
    (10, 11),  # L_Foot <-> R_Foot
    (13, 14),  # L_Collar <-> R_Collar
    (16, 17),  # L_Shoulder <-> R_Shoulder
    (18, 19),  # L_Elbow <-> R_Elbow
    (20, 21),  # L_Wrist <-> R_Wrist
]
```

Replace with:

```python
from views.mesh_viewport import MeshViewport, JOINT_NAMES, JOINT_PARENTS
from models.skeleton import SMPLX_SKELETON as _SKEL

log = logging.getLogger(__name__)

# Skeleton constants — sourced from models/skeleton.py
_N_BODY_JOINTS = _SKEL.n_body_joints
_LR_SWAP_PAIRS = _SKEL.lr_swap_pairs
```

---

## Step 5: Run full test suite

```bash
cd ~/bodypipe
python -m pytest tests/ -x -q
```

**Expected**: All 1507+ tests pass. Zero failures from the refactor — the module-level aliases ensure all imports resolve to the same values as before.

If any test fails, it means an alias was missed or a value differs. Debug by comparing the alias value to the original hardcoded value.

---

## Step 6: Commit Phase 0

```bash
git add models/skeleton.py models/body_model.py views/mesh_viewport.py views/pose_corrector_panel.py
git commit -m "refactor: extract skeleton constants into models/skeleton.py registry (Phase 0)

Move JOINT_NAMES, JOINT_PARENTS, DEFAULT_OFFSETS, BONE_CONNECTIONS,
_N_BODY_JOINTS, _LR_SWAP_PAIRS, _LR_PAIRS, _JOINT_REGIONS, and
_JOINT_PALETTE from mesh_viewport.py and pose_corrector_panel.py into
a new SkeletonDef dataclass in models/skeleton.py.

Add SMPLX_SKELETON (52-joint) and SOMA_SKELETON (77-joint placeholder)
presets. Create BodyModelAdapter protocol in models/body_model.py with
SmplxAdapter and SomaAdapter stub.

Module-level aliases in mesh_viewport.py preserve all existing imports.
Zero test changes required."
```

---

## Step 7: Modify `models/session.py` — Phase 1

### 7a. Add new fields to PersonTrack

Find the PersonTrack dataclass (line 110):

```python
@dataclass
class PersonTrack:
    """Per-person state."""

    person_id: int = -1
    person_dir: Path | None = None
    # These hold references to backend objects (IdentityTrack, etc.)
    # Typed as Any to avoid importing GVHMR backend at module level
    identity_track: Any = None
    smplx_params: dict | None = None
    confidences: list | None = None
```

Add two new fields immediately after `smplx_params`:

```python
    smplx_params: dict | None = None
    soma_params: dict | None = None
    body_model_type: str = "smplx"  # "smplx" | "soma"
    confidences: list | None = None
```

So the full field list becomes:
```python
    person_id: int = -1
    person_dir: Path | None = None
    identity_track: Any = None
    smplx_params: dict | None = None
    soma_params: dict | None = None
    body_model_type: str = "smplx"  # "smplx" | "soma"
    confidences: list | None = None
    bboxes: np.ndarray | None = None
    original_bboxes: np.ndarray | None = None
    bbox_corrections: np.ndarray | None = None
    keyframes: list[dict] = field(default_factory=list)
    confidence_breakdown: dict[str, list[float]] | None = None
```

### 7b. Update PersonTrack.to_dict()

Find the `to_dict` method (line 129):

```python
    def to_dict(self) -> dict:
        d = {
            "person_id": self.person_id,
            "person_dir": str(self.person_dir) if self.person_dir else None,
            "keyframes": self.keyframes,
        }
```

Add `body_model_type` to the dict:

```python
    def to_dict(self) -> dict:
        d = {
            "person_id": self.person_id,
            "person_dir": str(self.person_dir) if self.person_dir else None,
            "body_model_type": self.body_model_type,
            "keyframes": self.keyframes,
        }
```

### 7c. Update PersonTrack.from_dict()

Find the `from_dict` method (line 139):

```python
    @classmethod
    def from_dict(cls, data: dict) -> PersonTrack:
        return cls(
            person_id=data.get("person_id", -1),
            person_dir=Path(data["person_dir"]) if data.get("person_dir") else None,
            keyframes=data.get("keyframes", []),
        )
```

Add `body_model_type`:

```python
    @classmethod
    def from_dict(cls, data: dict) -> PersonTrack:
        return cls(
            person_id=data.get("person_id", -1),
            person_dir=Path(data["person_dir"]) if data.get("person_dir") else None,
            body_model_type=data.get("body_model_type", "smplx"),
            keyframes=data.get("keyframes", []),
        )
```

---

## Step 8: Create `models/param_converter.py`

Create the file `models/param_converter.py`:

```python
"""Parameter format conversion between SMPL-X and SOMA representations.

Why: The session model supports both smplx_params and soma_params. When
loading legacy sessions (SMPL-X) into a SOMA-native pipeline, or exporting
SOMA results to SMPL-X-based tools, these converters bridge the gap.

Actual conversion requires py-soma-x for analytical inversion (<1° error).
These stubs establish the interface for Phase 3 implementation.
"""

from __future__ import annotations


def smplx_to_soma(smplx_params: dict) -> dict:
    """Convert SMPL-X parameters to SOMA format.

    Parameters
    ----------
    smplx_params : dict
        SMPL-X format with keys: global_orient (N, 3), body_pose (N, 21, 3),
        left_hand_pose (N, 15, 3), right_hand_pose (N, 15, 3), transl (N, 3),
        betas (1, 10).

    Returns
    -------
    dict
        SOMA format with keys: poses (N, 77, 3), transl (N, 3),
        global_orient (N, 3), identity_coeffs (1, 45), scale_params (1, 68),
        identity_model_type str.

    Raises
    ------
    NotImplementedError
        Until py-soma-x is available (Phase 3).
    """
    raise NotImplementedError(
        "SMPL-X → SOMA conversion requires py-soma-x. Install with: pip install py-soma-x"
    )


def soma_to_smplx(soma_params: dict) -> dict:
    """Convert SOMA parameters to SMPL-X format.

    Parameters
    ----------
    soma_params : dict
        SOMA format with keys: poses (N, 77, 3), transl (N, 3),
        identity_coeffs (1, 45), scale_params (1, 68).

    Returns
    -------
    dict
        SMPL-X format with keys: global_orient (N, 3), body_pose (N, 21, 3),
        left_hand_pose (N, 15, 3), right_hand_pose (N, 15, 3), transl (N, 3).

    Raises
    ------
    NotImplementedError
        Until py-soma-x is available (Phase 3).
    """
    raise NotImplementedError(
        "SOMA → SMPL-X conversion requires py-soma-x. Install with: pip install py-soma-x"
    )
```

---

## Step 9: Modify `models/pipeline_config.py`

### 9a. Add new fields to PipelineConfig

Find the end of the existing fields (after line 32, `use_vitpose_face_crops`):

```python
    use_vitpose_face_crops: bool = True
```

Add two new fields:

```python
    use_vitpose_face_crops: bool = True
    # Body model and estimation backend (SOMA migration)
    body_model: str = "smplx"            # "smplx" | "soma"
    estimation_backend: str = "gvhmr"    # "gvhmr" | "gemx"
```

### 9b. Update to_dict()

Find the `to_dict` method and add the two new keys. Add these lines before the closing `}`:

```python
            "use_vitpose_face_crops": self.use_vitpose_face_crops,
```

Change to:

```python
            "use_vitpose_face_crops": self.use_vitpose_face_crops,
            "body_model": self.body_model,
            "estimation_backend": self.estimation_backend,
```

### 9c. from_dict already handles new fields

The existing `from_dict` uses `**{k: v for k, v in data.items() if k in cls.__dataclass_fields__}` — this automatically picks up new dataclass fields from saved dicts, and ignores them if absent (using defaults). No change needed.

---

## Step 10: Add tests to `tests/test_models.py`

Append the following test classes at the end of the file (after line 409, the last line):

```python


class TestPersonTrackSomaFields:
    """Tests for SOMA migration fields on PersonTrack."""

    def test_soma_params_default_none(self):
        track = PersonTrack()
        assert track.soma_params is None

    def test_body_model_type_default(self):
        track = PersonTrack()
        assert track.body_model_type == "smplx"

    def test_body_model_type_roundtrip(self):
        track = PersonTrack(person_id=1, body_model_type="soma")
        d = track.to_dict()
        assert d["body_model_type"] == "soma"
        loaded = PersonTrack.from_dict(d)
        assert loaded.body_model_type == "soma"

    def test_body_model_type_default_on_legacy_load(self):
        """Loading a dict without body_model_type should default to 'smplx'."""
        d = {"person_id": 5, "keyframes": []}
        loaded = PersonTrack.from_dict(d)
        assert loaded.body_model_type == "smplx"

    def test_session_roundtrip_with_body_model_type(self, tmp_path):
        s = Session(video_path=Path("/tmp/test.mp4"), num_frames=10)
        s.person_tracks[0] = PersonTrack(person_id=0, body_model_type="soma")
        s.person_tracks[1] = PersonTrack(person_id=1, body_model_type="smplx")
        path = tmp_path / "session.json"
        s.save(path)
        loaded = Session.load(path)
        assert loaded.person_tracks[0].body_model_type == "soma"
        assert loaded.person_tracks[1].body_model_type == "smplx"


class TestPipelineConfigSomaFields:
    """Tests for SOMA migration fields on PipelineConfig."""

    def test_body_model_default(self):
        c = PipelineConfig()
        assert c.body_model == "smplx"

    def test_estimation_backend_default(self):
        c = PipelineConfig()
        assert c.estimation_backend == "gvhmr"

    def test_new_fields_roundtrip(self, tmp_path):
        c = PipelineConfig(body_model="soma", estimation_backend="gemx")
        path = tmp_path / "config.json"
        c.save(path)
        loaded = PipelineConfig.load(path)
        assert loaded.body_model == "soma"
        assert loaded.estimation_backend == "gemx"

    def test_new_fields_in_to_dict(self):
        c = PipelineConfig(body_model="soma", estimation_backend="gemx")
        d = c.to_dict()
        assert d["body_model"] == "soma"
        assert d["estimation_backend"] == "gemx"

    def test_legacy_load_without_new_fields(self):
        """Loading a dict without body_model/estimation_backend uses defaults."""
        c = PipelineConfig.from_dict({"mode": "single"})
        assert c.body_model == "smplx"
        assert c.estimation_backend == "gvhmr"


class TestSkeletonRegistry:
    """Smoke tests for the skeleton registry module."""

    def test_smplx_skeleton_basic(self):
        from models.skeleton import SMPLX_SKELETON
        assert SMPLX_SKELETON.n_joints == 52
        assert SMPLX_SKELETON.n_body_joints == 22
        assert SMPLX_SKELETON.name == "smplx_52"
        assert len(SMPLX_SKELETON.joint_parents) == 52
        assert len(SMPLX_SKELETON.default_offsets) == 52

    def test_soma_skeleton_basic(self):
        from models.skeleton import SOMA_SKELETON
        assert SOMA_SKELETON.n_joints == 77
        assert SOMA_SKELETON.n_body_joints == 22
        assert SOMA_SKELETON.name == "soma_77"
        assert len(SOMA_SKELETON.joint_parents) == 77

    def test_smplx_skeleton_frozen(self):
        from models.skeleton import SMPLX_SKELETON
        import pytest
        with pytest.raises(AttributeError):
            SMPLX_SKELETON.name = "modified"

    def test_body_model_protocol(self):
        from models.body_model import SmplxAdapter, SomaAdapter, BodyModelAdapter
        smplx = SmplxAdapter()
        soma = SomaAdapter()
        assert isinstance(smplx, BodyModelAdapter)
        assert isinstance(soma, BodyModelAdapter)
        assert smplx.skeleton.name == "smplx_52"
        assert soma.skeleton.name == "soma_77"

    def test_param_converter_stubs(self):
        from models.param_converter import smplx_to_soma, soma_to_smplx
        import pytest
        with pytest.raises(NotImplementedError):
            smplx_to_soma({})
        with pytest.raises(NotImplementedError):
            soma_to_smplx({})
```

---

## Step 11: Run full test suite

```bash
cd ~/bodypipe
python -m pytest tests/ -x -q
```

**Expected**: All previous 1507 tests pass + ~15 new tests = ~1522 pass.

---

## Step 12: Commit Phase 1

```bash
git add models/session.py models/param_converter.py models/pipeline_config.py tests/test_models.py
git commit -m "feat: add SOMA params support to session model and pipeline config (Phase 1)

PersonTrack gains soma_params (dict|None) and body_model_type (str,
default 'smplx'). PipelineConfig gains body_model and estimation_backend
fields. Both serialize/deserialize with backward-compatible defaults.

Add models/param_converter.py with stub smplx_to_soma() and
soma_to_smplx() functions (NotImplementedError until py-soma-x).

Add 15 new tests covering new fields, round-trips, legacy loading,
skeleton registry smoke tests, and body model protocol verification."
```

---

## Step 13: Push

```bash
git push -u origin soma/skeleton-registry
```

---

## Post-implementation checklist

- [ ] `models/skeleton.py` exists with `SMPLX_SKELETON` (52 joints) and `SOMA_SKELETON` (77 joints)
- [ ] `models/body_model.py` exists with `BodyModelAdapter`, `SmplxAdapter`, `SomaAdapter`
- [ ] `models/param_converter.py` exists with stub functions
- [ ] `views/mesh_viewport.py` no longer defines JOINT_NAMES etc. inline — imports from skeleton.py
- [ ] `views/pose_corrector_panel.py` no longer defines _N_BODY_JOINTS / _LR_SWAP_PAIRS inline
- [ ] `models/session.py` PersonTrack has `soma_params` and `body_model_type` fields
- [ ] `models/pipeline_config.py` has `body_model` and `estimation_backend` fields
- [ ] All ~1522 tests pass
- [ ] `from views.mesh_viewport import JOINT_NAMES, JOINT_PARENTS, BONE_CONNECTIONS` still works
- [ ] `from views.mesh_viewport import _LR_PAIRS, _JOINT_REGIONS, _JOINT_TO_REGION` still works
- [ ] `from views.mesh_viewport import _JOINT_PALETTE, _N_BODY_JOINTS` still works
