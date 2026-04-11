"""Quality metrics for comparing raw vs physics-refined mocap.

Uses simple FK to get joint positions from axis-angle rotations + skeleton
offsets, then computes smoothness, foot skating, spectral preservation, and
weight metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from models.skeleton import SMPLX_SKELETON

logger = logging.getLogger(__name__)

# Default verdict thresholds. Documented in tools/verify_physics.py.
DEFAULT_THRESHOLDS = {
    "spectral_correlation_min": 0.9,
    "root_drift_horizontal_max": 0.5,  # meters
    "ldlj_improvement_min": 0.0,
    "require_frame_count_match": True,
}

# Body joint indices (SMPL-X ordering, first 22 joints)
_PELVIS = 0
_L_ANKLE, _R_ANKLE = 7, 8
_L_FOOT, _R_FOOT = 10, 11
_L_WRIST, _R_WRIST = 20, 21

_N_BODY_JOINTS = 22


def evaluate_refinement(
    raw_params: dict,
    refined_params: dict,
    fps: float = 30.0,
) -> dict:
    """Compute quality metrics comparing raw and physics-refined mocap.

    Parameters
    ----------
    raw_params : dict
        Original GVHMR params (pre-physics).
    refined_params : dict
        PHC-refined params.
    fps : float
        Frame rate.

    Returns
    -------
    dict
        Metric names to values.
    """
    n_raw = int(raw_params["num_frames"])
    n_ref = int(refined_params["num_frames"])
    n = min(n_raw, n_ref)

    metrics: dict = {
        "frame_count_raw": n_raw,
        "frame_count_refined": n_ref,
        "frame_count_preserved": bool(n_raw == n_ref),
        "fps": float(fps),
    }

    # NaN check on the refined arrays the viewport actually consumes.
    try:
        nan_count = 0
        for key in ("global_orient", "body_pose", "transl"):
            arr = np.asarray(refined_params.get(key))
            if arr.dtype.kind == "f":
                nan_count += int(np.isnan(arr).sum())
        metrics["nan_count_refined"] = nan_count
        metrics["has_nan"] = nan_count > 0
    except Exception as exc:
        logger.warning("NaN check failed: %s", exc)
        metrics["has_nan"] = False

    # Root translation drift (raw vs refined). Y is vertical in GVHMR Y-up
    # world space; horizontal = X+Z. Catches the "policy walked away" case.
    try:
        raw_t = np.asarray(raw_params["transl"], dtype=np.float64)[:n]
        ref_t = np.asarray(refined_params["transl"], dtype=np.float64)[:n]
        diff = ref_t - raw_t
        horiz = np.linalg.norm(diff[:, [0, 2]], axis=-1)
        vert = np.abs(diff[:, 1])
        metrics["root_drift_horizontal_mean"] = float(horiz.mean())
        metrics["root_drift_horizontal_max"] = float(horiz.max())
        metrics["root_drift_vertical_mean"] = float(vert.mean())
        metrics["root_drift_vertical_max"] = float(vert.max())
    except Exception as exc:
        logger.warning("Root drift computation failed: %s", exc)

    # Compute FK positions for a sample of frames (every 3rd frame)
    sample_indices = list(range(0, n, 3))
    if not sample_indices:
        sample_indices = [0]

    raw_positions = np.stack(
        [_forward_kinematics_body(raw_params, i) for i in sample_indices]
    )  # (S, 22, 3)
    ref_positions = np.stack(
        [_forward_kinematics_body(refined_params, i) for i in sample_indices]
    )  # (S, 22, 3)

    sample_fps = fps / 3.0  # Effective FPS for sampled frames

    # LDLJ on wrist positions
    try:
        raw_wrist_pos = raw_positions[:, [_L_WRIST, _R_WRIST], :]  # (S, 2, 3)
        ref_wrist_pos = ref_positions[:, [_L_WRIST, _R_WRIST], :]
        raw_ldlj = compute_ldlj(
            raw_wrist_pos.reshape(-1, 6), sample_fps
        )
        ref_ldlj = compute_ldlj(
            ref_wrist_pos.reshape(-1, 6), sample_fps
        )
        metrics["ldlj_wrist_raw"] = raw_ldlj
        metrics["ldlj_wrist_refined"] = ref_ldlj
        metrics["ldlj_wrist_improvement"] = raw_ldlj - ref_ldlj
    except Exception as exc:
        logger.warning("LDLJ computation failed: %s", exc)

    # Foot contact detection on ankle positions
    try:
        raw_ankle = raw_positions[:, [_L_ANKLE, _R_ANKLE], :]  # (S, 2, 3)
        ref_ankle = ref_positions[:, [_L_ANKLE, _R_ANKLE], :]

        raw_contact = detect_foot_contacts(raw_ankle, sample_fps)
        ref_contact = detect_foot_contacts(ref_ankle, sample_fps)

        # Foot skating on toe positions
        raw_toe = raw_positions[:, [_L_FOOT, _R_FOOT], :]
        ref_toe = ref_positions[:, [_L_FOOT, _R_FOOT], :]

        raw_skating = compute_foot_skating(raw_toe, raw_contact, sample_fps)
        ref_skating = compute_foot_skating(ref_toe, ref_contact, sample_fps)
        metrics["foot_skating_raw"] = raw_skating
        metrics["foot_skating_refined"] = ref_skating
        metrics["foot_skating_improvement"] = raw_skating - ref_skating
    except Exception as exc:
        logger.warning("Foot skating computation failed: %s", exc)

    # Spectral preservation
    try:
        spectral = compute_spectral_preservation(
            raw_positions.reshape(-1, _N_BODY_JOINTS * 3),
            ref_positions.reshape(-1, _N_BODY_JOINTS * 3),
            sample_fps,
        )
        metrics["spectral"] = spectral
    except Exception as exc:
        logger.warning("Spectral preservation failed: %s", exc)

    # Weight metric on pelvis
    try:
        raw_weight = compute_weight_metric(raw_positions[:, _PELVIS, :], sample_fps)
        ref_weight = compute_weight_metric(ref_positions[:, _PELVIS, :], sample_fps)
        metrics["weight_raw"] = raw_weight
        metrics["weight_refined"] = ref_weight
    except Exception as exc:
        logger.warning("Weight metric computation failed: %s", exc)

    return metrics


def compute_ldlj(positions: np.ndarray, fps: float) -> float:
    """Compute Log Dimensionless Jerk (lower = smoother).

    Parameters
    ----------
    positions : (N, D) array of positions.
    fps : Frame rate.

    Returns
    -------
    float
        LDLJ value.
    """
    if positions.shape[0] < 4:
        return 0.0

    dt = 1.0 / fps
    vel = np.diff(positions, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt

    duration = (positions.shape[0] - 1) * dt
    # Displacement
    displacement = np.linalg.norm(positions[-1] - positions[0])
    if displacement < 1e-6:
        displacement = 1e-6

    jerk_magnitude_sq = np.sum(jerk ** 2) * dt
    # Log Dimensionless Jerk
    ldlj = -np.log(
        (duration ** 5 / displacement ** 2) * jerk_magnitude_sq + 1e-10
    )
    return float(ldlj)


def compute_foot_skating(
    toe_positions: np.ndarray,
    contact_mask: np.ndarray,
    fps: float,
) -> float:
    """Compute mean horizontal displacement during foot contact.

    Parameters
    ----------
    toe_positions : (N, 2, 3) — left and right toe positions.
    contact_mask : (N, 2) bool — True when foot is in contact.
    fps : float

    Returns
    -------
    float
        Mean horizontal skating distance (meters) per contact frame.
    """
    if toe_positions.shape[0] < 2:
        return 0.0

    dt = 1.0 / fps
    # Horizontal displacement (XZ plane)
    vel = np.diff(toe_positions, axis=0)  # (N-1, 2, 3)
    horiz_vel = vel[:, :, [0, 2]]  # (N-1, 2, 2) — X and Z only
    horiz_speed = np.linalg.norm(horiz_vel, axis=-1)  # (N-1, 2)

    # Contact mask for velocity frames (use mask[:-1] since vel is one shorter)
    contact = contact_mask[:-1]  # (N-1, 2)

    skating_distances = horiz_speed * contact * dt
    total_contact = contact.sum()

    if total_contact < 1:
        return 0.0

    return float(skating_distances.sum() / total_contact)


def compute_spectral_preservation(
    raw: np.ndarray,
    refined: np.ndarray,
    fps: float,
) -> dict:
    """Compare frequency content between raw and refined motion.

    Parameters
    ----------
    raw : (N, D) array.
    refined : (N, D) array.
    fps : float

    Returns
    -------
    dict
        Spectral metrics.
    """
    n = min(raw.shape[0], refined.shape[0])
    if n < 4:
        return {"correlation": 1.0, "low_freq_ratio": 1.0}

    raw = raw[:n]
    refined = refined[:n]

    # FFT along time axis
    raw_fft = np.abs(np.fft.rfft(raw, axis=0))
    ref_fft = np.abs(np.fft.rfft(refined, axis=0))

    # Overall spectral correlation
    raw_flat = raw_fft.flatten()
    ref_flat = ref_fft.flatten()

    if np.std(raw_flat) < 1e-10 or np.std(ref_flat) < 1e-10:
        correlation = 1.0
    else:
        correlation = float(np.corrcoef(raw_flat, ref_flat)[0, 1])

    # Low-frequency energy ratio (below 5 Hz)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    low_mask = freqs < 5.0

    raw_low = np.sum(raw_fft[low_mask] ** 2)
    ref_low = np.sum(ref_fft[low_mask] ** 2)

    if raw_low < 1e-10:
        low_freq_ratio = 1.0
    else:
        low_freq_ratio = float(ref_low / raw_low)

    return {
        "correlation": correlation,
        "low_freq_ratio": low_freq_ratio,
    }


def compute_weight_metric(positions: np.ndarray, fps: float) -> float:
    """Compute downward/upward acceleration ratio (weight perception).

    Higher values suggest more physically plausible gravity influence.

    Parameters
    ----------
    positions : (N, 3) — pelvis positions.
    fps : float

    Returns
    -------
    float
        Ratio of downward to upward vertical acceleration magnitude.
    """
    if positions.shape[0] < 3:
        return 1.0

    dt = 1.0 / fps
    vel = np.diff(positions, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt

    # Vertical acceleration (Y axis)
    vert_acc = acc[:, 1]

    down_acc = np.abs(vert_acc[vert_acc < 0]).sum()
    up_acc = np.abs(vert_acc[vert_acc > 0]).sum()

    if up_acc < 1e-10:
        return 1.0

    return float(down_acc / up_acc)


def detect_foot_contacts(
    ankle_positions: np.ndarray,
    fps: float,
    height_threshold: float = 0.05,
    velocity_threshold: float = 0.3,
) -> np.ndarray:
    """Detect foot contacts using velocity + height heuristic.

    Parameters
    ----------
    ankle_positions : (N, 2, 3) — left and right ankle positions.
    fps : float
    height_threshold : float
        Maximum ankle height (Y) for contact (meters).
    velocity_threshold : float
        Maximum ankle speed for contact (m/s).

    Returns
    -------
    np.ndarray
        (N, 2) bool mask — True when foot is in contact.
    """
    n = ankle_positions.shape[0]
    contact = np.zeros((n, 2), dtype=bool)

    if n < 2:
        return contact

    dt = 1.0 / fps

    for foot_idx in range(2):
        pos = ankle_positions[:, foot_idx, :]  # (N, 3)

        # Height criterion: Y coordinate near minimum (ground)
        min_height = pos[:, 1].min()
        height_ok = (pos[:, 1] - min_height) < height_threshold

        # Velocity criterion
        vel = np.zeros_like(pos)
        vel[1:] = (pos[1:] - pos[:-1]) / dt
        vel[0] = vel[1]
        speed = np.linalg.norm(vel, axis=-1)
        vel_ok = speed < velocity_threshold

        contact[:, foot_idx] = height_ok & vel_ok

    return contact


def _forward_kinematics_body(params: dict, frame_idx: int) -> np.ndarray:
    """Simple FK using skeleton offsets for one frame.

    Parameters
    ----------
    params : dict
        GVHMR params with global_orient (N, 3) and body_pose (N, 21, 3).
    frame_idx : int
        Frame to compute FK for.

    Returns
    -------
    np.ndarray
        (22, 3) joint positions.
    """
    global_orient = np.asarray(params["global_orient"])[frame_idx]  # (3,)
    body_pose = np.asarray(params["body_pose"])[frame_idx]           # (21, 3)
    transl = np.asarray(params["transl"])[frame_idx]                 # (3,)

    # Get skeleton data
    joint_names = SMPLX_SKELETON.joint_names
    joint_parents = SMPLX_SKELETON.joint_parents
    offsets = SMPLX_SKELETON.default_offsets

    # Build rotation matrices for body joints
    # Joint 0 = global_orient, joints 1-21 = body_pose
    rotations = np.zeros((_N_BODY_JOINTS, 3, 3))
    rotations[0] = _axis_angle_to_matrix(global_orient)
    for j in range(1, _N_BODY_JOINTS):
        rotations[j] = _axis_angle_to_matrix(body_pose[j - 1])

    # Forward kinematics
    positions = np.zeros((_N_BODY_JOINTS, 3))
    world_rotations = np.zeros((_N_BODY_JOINTS, 3, 3))

    # Get offset array
    offset_arr = np.zeros((_N_BODY_JOINTS, 3))
    for j in range(_N_BODY_JOINTS):
        name = joint_names[j]
        offset_arr[j] = offsets[name]

    # Root
    world_rotations[0] = rotations[0]
    positions[0] = transl

    for j in range(1, _N_BODY_JOINTS):
        parent = joint_parents[j]
        world_rotations[j] = world_rotations[parent] @ rotations[j]
        positions[j] = positions[parent] + world_rotations[parent] @ offset_arr[j]

    return positions


def compute_verdict(
    metrics: dict,
    thresholds: dict | None = None,
) -> tuple[str, str]:
    """Classify a metrics dict as ok / warn / fail.

    Returns ``(status, reason)`` where ``status`` is one of ``"ok"``,
    ``"warn"``, ``"fail"`` and ``reason`` is a short human-readable string
    explaining any non-ok status (empty when ``"ok"``).
    """
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    fails: list[str] = []
    warns: list[str] = []

    if metrics.get("has_nan"):
        n = metrics.get("nan_count_refined", "?")
        fails.append(f"refined output has {n} NaN values")

    if th["require_frame_count_match"] and not metrics.get(
        "frame_count_preserved", True
    ):
        warns.append(
            "frame count changed "
            f"({metrics.get('frame_count_raw')}→{metrics.get('frame_count_refined')})"
        )

    drift = metrics.get("root_drift_horizontal_mean")
    if drift is not None and drift > th["root_drift_horizontal_max"]:
        warns.append(f"root drifted {drift:.2f}m horizontally (mean)")

    ldlj_imp = metrics.get("ldlj_wrist_improvement")
    if ldlj_imp is not None and ldlj_imp < th["ldlj_improvement_min"]:
        warns.append(f"LDLJ improvement {ldlj_imp:.2f} (physics added jerk)")

    spectral = metrics.get("spectral") or {}
    corr = spectral.get("correlation")
    if corr is not None and corr < th["spectral_correlation_min"]:
        warns.append(f"spectral correlation {corr:.2f} below threshold")

    if fails:
        return "fail", "; ".join(fails + warns)
    if warns:
        return "warn", "; ".join(warns)
    return "ok", ""


def _to_jsonable(obj):
    """Recursively convert numpy scalars/arrays into JSON-friendly Python types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_metrics_json(
    metrics: dict,
    output_path: Path,
    thresholds: dict | None = None,
) -> Path:
    """Persist a metrics dict (with verdict) to ``output_path`` as JSON.

    Adds ``status`` and ``reason`` fields based on ``compute_verdict``.
    Returns the written path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    status, reason = compute_verdict(metrics, thresholds)
    payload = _to_jsonable(metrics)
    payload["status"] = status
    payload["reason"] = reason
    payload["thresholds"] = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    with open(output_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return output_path


def _axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Convert axis-angle rotation vector to 3x3 rotation matrix (Rodrigues)."""
    angle = np.linalg.norm(rotvec)
    if angle < 1e-8:
        return np.eye(3)

    axis = rotvec / angle
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])

    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
