"""Convert PHC simulation output back to SMPL parameters.

PHC outputs AMASS-format NPZ in Z-up coordinates. This module converts
back to Y-up (GVHMR world space) and reshapes to the params dict format.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Z-up (PHC/MuJoCo) → Y-up (GVHMR): rotate +90° around X
# (x, y, z) → (x, -z, y)
_ZUP_TO_YUP = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0],
], dtype=np.float64)


def _rotate_root_orient(global_orient_aa: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Pre-multiply root axis-angle orientation by rotation matrix R."""
    from scipy.spatial.transform import Rotation as sRot
    root_rot = sRot.from_rotvec(global_orient_aa)
    applied = sRot.from_matrix(R) * root_rot
    return applied.as_rotvec()


def phc_output_to_params(phc_output_path: Path, original_params: dict) -> dict:
    """Convert PHC simulation output (Z-up) to SMPL params dict (Y-up).

    Parameters
    ----------
    phc_output_path : Path
        Path to PHC output NPZ (Z-up coordinates).
    original_params : dict
        Original GVHMR params for metadata preservation.

    Returns
    -------
    dict
        SMPL params dict in Y-up GVHMR world space.
    """
    data = np.load(str(phc_output_path), allow_pickle=True)

    n_original = original_params["num_frames"]

    # Extract poses and translation (Z-up)
    if "poses" in data:
        poses = data["poses"]
        n = poses.shape[0]
        global_orient_zup = poses[:, :3]
        body_pose = poses[:, 3:66].reshape(n, 21, 3)
    elif "body_pose" in data and "global_orient" in data:
        global_orient_zup = data["global_orient"]
        body_pose = data["body_pose"].reshape(-1, 21, 3)
        n = body_pose.shape[0]
    else:
        logger.warning(
            "PHC output format not recognized, keys: %s — returning original params",
            list(data.keys()),
        )
        return original_params

    transl_zup = np.asarray(
        data.get("trans", data.get("transl", np.zeros((n, 3)))),
        dtype=np.float64,
    )

    # --- Z-up → Y-up coordinate transform ---
    transl = transl_zup @ _ZUP_TO_YUP.T
    global_orient = _rotate_root_orient(
        global_orient_zup.astype(np.float64), _ZUP_TO_YUP
    )
    # Body pose: local rotations are parent-relative, no transform needed

    # Resample if frame count differs
    if n != n_original:
        global_orient = _resample_to_length(global_orient, n_original)
        body_pose = _resample_to_length(
            body_pose.reshape(n, -1), n_original
        ).reshape(n_original, 21, 3)
        transl = _resample_to_length(transl, n_original)
        n = n_original

    result = {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": transl,
        "left_hand_pose": original_params.get(
            "left_hand_pose", np.zeros((n, 15, 3))
        ),
        "right_hand_pose": original_params.get(
            "right_hand_pose", np.zeros((n, 15, 3))
        ),
        "betas": original_params.get("betas", np.zeros((n, 10))),
        "num_frames": n,
        "coordinate_space": original_params.get("coordinate_space", "world"),
        "camera_model": original_params.get("camera_model", "world_space"),
        "translation_origin": original_params.get("translation_origin", "pelvis"),
        "source": "phc_refined",
    }

    # Preserve camera-space params if they exist
    for key in [
        "global_orient_cam", "body_pose_cam", "transl_cam",
        "global_orient_world", "body_pose_world", "transl_world",
        "K_fullimg",
    ]:
        if key in original_params:
            result[key] = original_params[key]

    # Update world-space params with refined values
    result["global_orient_world"] = global_orient
    result["body_pose_world"] = body_pose
    result["transl_world"] = transl

    return result


def _resample_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample array along axis 0 to target_len."""
    if arr.shape[0] == target_len:
        return arr
    source_idx = np.linspace(0, arr.shape[0] - 1, target_len)
    idx_floor = np.floor(source_idx).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, arr.shape[0] - 1)
    frac = (source_idx - idx_floor).reshape(-1, *([1] * (arr.ndim - 1)))
    return arr[idx_floor] * (1 - frac) + arr[idx_ceil] * frac
