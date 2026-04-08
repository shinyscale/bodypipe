"""GVHMR params dict <-> AMASS format conversion for PHC MotionLib."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Y-up (GVHMR) → Z-up (PHC/MuJoCo): rotate -90° around X
# (x, y, z) → (x, z, -y)
_YUP_TO_ZUP = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0],
], dtype=np.float64)

# Z-up (PHC/MuJoCo) → Y-up (GVHMR): rotate +90° around X
# (x, y, z) → (x, -z, y)
_ZUP_TO_YUP = _YUP_TO_ZUP.T


def _rotate_root_orient(global_orient_aa: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Pre-multiply root axis-angle orientation by rotation matrix R.

    Parameters
    ----------
    global_orient_aa : (N, 3) axis-angle root orientations.
    R : (3, 3) rotation matrix to apply.

    Returns
    -------
    (N, 3) rotated axis-angle.
    """
    from scipy.spatial.transform import Rotation as sRot
    root_rot = sRot.from_rotvec(global_orient_aa)
    applied = sRot.from_matrix(R) * root_rot
    return applied.as_rotvec()


def _axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle (N, 3) to quaternion (N, 4) in wxyz order."""
    angle = np.linalg.norm(aa, axis=-1, keepdims=True)  # (N, 1)
    half = angle / 2
    # Avoid division by zero
    safe_angle = np.where(angle > 1e-8, angle, 1.0)
    axis = aa / safe_angle
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.concatenate([w, xyz], axis=-1)  # (N, 4) wxyz


def _local_to_global_quats(
    pose_aa: np.ndarray, parents: list[int]
) -> np.ndarray:
    """Convert local axis-angle rotations to global quaternions.

    Parameters
    ----------
    pose_aa : (N, J*3) axis-angle rotations.
    parents : list of parent joint indices (-1 for root).

    Returns
    -------
    np.ndarray : (N, J, 4) global quaternions in wxyz order.
    """
    from scipy.spatial.transform import Rotation as sRot

    n_frames = pose_aa.shape[0]
    n_joints = pose_aa.shape[1] // 3
    local_aa = pose_aa.reshape(n_frames, n_joints, 3)

    global_quats = np.zeros((n_frames, n_joints, 4))
    global_quats[:, :, 0] = 1.0  # wxyz identity

    for j in range(n_joints):
        local_rot = sRot.from_rotvec(local_aa[:, j])
        local_q = local_rot.as_quat()  # xyzw
        # Convert to wxyz
        local_q_wxyz = np.concatenate([local_q[:, 3:4], local_q[:, :3]], axis=-1)

        parent = parents[j]
        if parent < 0:
            global_quats[:, j] = local_q_wxyz
        else:
            # global = parent_global * local
            pq = global_quats[:, parent]
            global_quats[:, j] = _quat_mul(pq, local_q_wxyz)

    return global_quats


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply quaternions in wxyz format. Shapes: (N, 4)."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


# SMPL parent joint indices (24 joints, -1 = root)
_SMPL_PARENTS = [
    -1,  # 0 Pelvis
    0,   # 1 L_Hip
    0,   # 2 R_Hip
    0,   # 3 Spine1
    1,   # 4 L_Knee
    2,   # 5 R_Knee
    3,   # 6 Spine2
    4,   # 7 L_Ankle
    5,   # 8 R_Ankle
    6,   # 9 Spine3
    7,   # 10 L_Foot
    8,   # 11 R_Foot
    9,   # 12 Neck
    9,   # 13 L_Collar
    9,   # 14 R_Collar
    12,  # 15 Head
    13,  # 16 L_Shoulder
    14,  # 17 R_Shoulder
    16,  # 18 L_Elbow
    17,  # 19 R_Elbow
    18,  # 20 L_Wrist
    19,  # 21 R_Wrist
    20,  # 22 L_Hand
    21,  # 23 R_Hand
]


def params_to_amass_pkl(params: dict, output_path: Path, fps: int = 30) -> Path:
    """Convert GVHMR params dict to PHC MotionLib PKL format.

    PHC's MotionLibSMPL expects a joblib PKL with structure::

        {"motion_name": {"pose_aa": (N,72), "pose_quat_global": (N,24,4),
         "pose_quat": (N,24,4), "trans_orig": (N,3),
         "root_trans_offset": torch.Tensor(N,3), "beta": (16,),
         "gender": "neutral", "fps": 30.0}}
    """
    import torch
    import joblib
    from scipy.spatial.transform import Rotation as sRot

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    global_orient = np.asarray(params["global_orient"], dtype=np.float64)  # (N, 3)
    body_pose = np.asarray(params["body_pose"], dtype=np.float64).reshape(-1, 63)
    transl = np.asarray(params["transl"], dtype=np.float64)               # (N, 3)
    n = global_orient.shape[0]

    # --- Y-up (GVHMR) → Z-up (PHC/MuJoCo) coordinate transform ---
    # Translation: (x, y, z)_yup → (x, z, -y)_zup
    transl_zup = transl @ _YUP_TO_ZUP.T
    # Root orientation: pre-multiply by Y-up→Z-up rotation
    global_orient_zup = _rotate_root_orient(global_orient, _YUP_TO_ZUP)
    # Body pose: local rotations are parent-relative, no transform needed

    # pose_aa: (N, 72) = root(3) + 23 joints * 3
    poses_66 = np.concatenate([global_orient_zup, body_pose], axis=1)  # (N, 66)
    toe_pad = np.zeros((n, 6), dtype=poses_66.dtype)
    pose_aa = np.concatenate([poses_66, toe_pad], axis=1)          # (N, 72)

    # Local quaternions: per-joint axis-angle -> quaternion (N, 24, 4) wxyz
    pose_quat = np.zeros((n, 24, 4))
    pose_quat[:, :, 0] = 1.0  # identity default
    for j in range(24):
        aa = pose_aa[:, j*3:(j+1)*3]
        pose_quat[:, j] = _axis_angle_to_quat(aa)

    # Global quaternions: accumulate through kinematic chain
    pose_quat_global = _local_to_global_quats(pose_aa, _SMPL_PARENTS)

    # Betas: pad to (16,)
    betas = np.asarray(params.get("betas", np.zeros((n, 10))))
    if betas.ndim == 2:
        betas = betas[0]
    beta = np.zeros(16)
    beta[:min(len(betas), 16)] = betas[:16]

    motion_data = {
        "pose_aa": pose_aa.astype(np.float64),
        "pose_quat_global": pose_quat_global.astype(np.float64),
        "pose_quat": pose_quat.astype(np.float64),
        "trans_orig": transl_zup.astype(np.float64),
        "root_trans_offset": torch.tensor(transl_zup, dtype=torch.float64),
        "beta": beta.astype(np.float64),
        "gender": "neutral",
        "fps": float(fps),
    }

    # Save as joblib PKL keyed by motion name
    motion_name = output_path.stem
    joblib.dump({motion_name: motion_data}, str(output_path))

    logger.info("Saved PHC motion PKL: %s (%d frames, fps=%d)", output_path, n, fps)
    return output_path


def params_to_amass_npz(params: dict, output_path: Path, fps: int = 30) -> Path:
    """Convert a GVHMR params dict to AMASS-format NPZ.

    Also saves a companion PKL for PHC MotionLib consumption.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    global_orient = np.asarray(params["global_orient"], dtype=np.float64)
    body_pose = np.asarray(params["body_pose"], dtype=np.float64).reshape(-1, 63)
    transl = np.asarray(params["transl"], dtype=np.float64)
    n = global_orient.shape[0]

    # Y-up → Z-up transform for NPZ as well
    transl_zup = transl @ _YUP_TO_ZUP.T
    global_orient_zup = _rotate_root_orient(global_orient, _YUP_TO_ZUP)

    poses_66 = np.concatenate([global_orient_zup, body_pose], axis=1)
    toe_pad = np.zeros((n, 6), dtype=poses_66.dtype)
    poses = np.concatenate([poses_66, toe_pad], axis=1)

    betas = np.asarray(params.get("betas", np.zeros((n, 10))))
    if betas.ndim == 2:
        betas = betas[0]
    betas = betas[:10]

    np.savez(
        str(output_path),
        poses=poses, betas=betas, trans=transl_zup,
        gender="neutral", mocap_framerate=fps,
    )

    # Also save PKL for PHC MotionLib
    pkl_path = output_path.with_suffix(".pkl")
    params_to_amass_pkl(params, pkl_path, fps=fps)

    logger.info("Saved AMASS NPZ+PKL: %s (%d frames, fps=%d)", output_path, n, fps)
    return output_path


def amass_to_params(npz_path: Path) -> dict:
    """Load an AMASS-format NPZ back to a GVHMR params dict.

    Parameters
    ----------
    npz_path : Path
        Path to an AMASS NPZ file with keys ``poses``, ``trans``, ``betas``.

    Returns
    -------
    dict
        Params dict with same keys as ``extract_gvhmr_params`` output.
    """
    data = np.load(str(npz_path), allow_pickle=True)

    poses = data["poses"]          # (N, 72)
    transl = data["trans"]         # (N, 3)
    betas = data["betas"]          # (10,)
    n = poses.shape[0]

    global_orient = poses[:, :3]                          # (N, 3)
    body_pose = poses[:, 3:66].reshape(n, 21, 3)          # (N, 21, 3)
    # Ignore toe joints (66:72)

    # Broadcast betas to (N, 10)
    betas_full = np.tile(betas[:10], (n, 1))

    fps = float(data["mocap_framerate"]) if "mocap_framerate" in data else 30.0

    return {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": transl,
        "betas": betas_full,
        "left_hand_pose": np.zeros((n, 15, 3), dtype=np.float32),
        "right_hand_pose": np.zeros((n, 15, 3), dtype=np.float32),
        "num_frames": n,
        "coordinate_space": "world",
        "camera_model": "world_space",
        "translation_origin": "pelvis",
        "source": "amass_reimport",
        "fps": fps,
    }
