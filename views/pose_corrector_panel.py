"""Pose Corrector panel — joint selection, euler sliders, quick-fix buttons,
corrections table, space overrides, BVH/FBX export, and real-time preview.

Why: The pose corrector enables interactive correction of bad GVHMR poses
(flipped orientations, impossible limb positions during lifts). Users select
a joint via viewport click or dropdown, adjust rotation with euler sliders
(real-time mesh preview), then commit corrections to a CorrectionTrack.
Quick-fix buttons handle common operations: flip body 180° on an axis,
invert upside-down poses, mirror L/R joints, or copy pose from another frame.
The corrections table shows all keyframes for the current person with
Go/Delete actions. Space overrides control per-frame coordinate space
(world/camera/carried) for complex multi-person scenarios like lifts.
BVH/FBX re-export applies all corrections and space overrides.

Phase 3.4: Person selector, joint selector, euler sliders, apply, reset.
Phase 3.5: Quick-fix buttons (flip, invert, mirror, copy) + corrections table.
Phase 3.6: Space overrides table + BVH/FBX export.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QCheckBox,
    QComboBox,
    QPushButton,
    QDoubleSpinBox,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QInputDialog,
    QMessageBox,
    QFileDialog,
)
from PySide6.QtCore import Signal, Qt, QTimer

from models.session import Session, UndoEntry
from theme import COLORS
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


def _safe_import_pose_correction():
    """Lazily import pose_correction from GVHMR backend.

    Returns (axis_angle_to_euler_deg, euler_deg_to_axis_angle, CorrectionTrack)
    or (None, None, None) when backend is unavailable.
    """
    try:
        from pose_correction import (
            axis_angle_to_euler_deg,
            euler_deg_to_axis_angle,
            CorrectionTrack,
        )
        return axis_angle_to_euler_deg, euler_deg_to_axis_angle, CorrectionTrack
    except ImportError:
        return None, None, None


def axis_angle_to_euler_deg_fallback(aa: np.ndarray) -> np.ndarray:
    """Fallback euler conversion when GVHMR backend is unavailable."""
    try:
        from scipy.spatial.transform import Rotation
        return Rotation.from_rotvec(aa).as_euler("XYZ", degrees=True).astype(np.float32)
    except ImportError:
        return np.zeros(3, dtype=np.float32)


def euler_deg_to_axis_angle_fallback(euler_deg: np.ndarray) -> np.ndarray:
    """Fallback axis-angle conversion when GVHMR backend is unavailable."""
    try:
        from scipy.spatial.transform import Rotation
        return Rotation.from_euler("XYZ", euler_deg, degrees=True).as_rotvec().astype(np.float32)
    except ImportError:
        return np.zeros(3, dtype=np.float32)


def _safe_import_quick_fix():
    """Lazily import quick-fix functions from GVHMR backend.

    Returns (flip_global_orient, mirror_lr_pose, copy_pose_from_frame)
    or (None, None, None) when backend is unavailable.
    """
    try:
        from pose_correction import (
            flip_global_orient,
            mirror_lr_pose,
            copy_pose_from_frame,
        )
        return flip_global_orient, mirror_lr_pose, copy_pose_from_frame
    except ImportError:
        return None, None, None


def _safe_import_space_override():
    """Lazily import FrameSpaceOverride from GVHMR backend.

    Returns FrameSpaceOverride class or None.
    """
    try:
        from pose_correction import FrameSpaceOverride
        return FrameSpaceOverride
    except ImportError:
        return None


def _safe_import_bvh_export():
    """Lazily import BVH export function from GVHMR backend.

    Returns convert_params_to_bvh or None.
    """
    try:
        from smplx_to_bvh import convert_params_to_bvh
        return convert_params_to_bvh
    except ImportError:
        return None


def _safe_import_fbx_export():
    """Lazily import FBX export function from GVHMR backend.

    Returns convert_bvh_to_fbx or None.
    """
    try:
        from bvh_to_fbx import convert_bvh_to_fbx
        return convert_bvh_to_fbx
    except ImportError:
        return None


def flip_global_orient_fallback(
    params: dict, frame: int, axis: str = "yaw",
) -> np.ndarray:
    """Fallback 180° flip when GVHMR backend is unavailable."""
    from scipy.spatial.transform import Rotation

    go = np.asarray(params["global_orient"], dtype=np.float32)
    if go.ndim >= 2 and frame < go.shape[0]:
        go_vec = go[frame].copy()
    else:
        go_vec = go.ravel()[:3].copy()
    R_orig = Rotation.from_rotvec(go_vec.astype(np.float64))
    axis_map = {"pitch": [1, 0, 0], "yaw": [0, 1, 0], "roll": [0, 0, 1]}
    flip_axis = axis_map.get(axis, [0, 1, 0])
    R_flip = Rotation.from_rotvec(np.array(flip_axis, dtype=np.float64) * np.pi)
    R_new = R_flip * R_orig
    return R_new.as_rotvec().astype(np.float32)


def mirror_lr_pose_fallback(params: dict, frame: int) -> dict[int, np.ndarray]:
    """Fallback L/R mirror when GVHMR backend is unavailable."""
    bp = np.asarray(params["body_pose"], dtype=np.float32)
    if bp.ndim == 2 and bp.shape[-1] != 3:
        bp = bp.reshape(bp.shape[0], -1, 3)
    if bp.ndim >= 3 and frame < bp.shape[0]:
        bp_frame = bp[frame].copy()
    else:
        return {}
    mirrored: dict[int, np.ndarray] = {}
    for l_idx, r_idx in _LR_SWAP_PAIRS:
        l_bp = l_idx - 1
        r_bp = r_idx - 1
        if 0 <= l_bp < bp_frame.shape[0] and 0 <= r_bp < bp_frame.shape[0]:
            mirrored[l_bp] = bp_frame[r_bp].copy()
            mirrored[r_bp] = bp_frame[l_bp].copy()
    return mirrored


def get_joint_euler(session: Session, person_id: int, frame_idx: int, joint_idx: int) -> np.ndarray:
    """Get current XYZ euler angles (degrees) for a joint.

    Checks CorrectionTrack first, then falls back to raw params.
    Returns (3,) float32 array.
    """
    aa_to_euler, _, _ = _safe_import_pose_correction()
    if aa_to_euler is None:
        aa_to_euler = axis_angle_to_euler_deg_fallback

    aa = _get_joint_axis_angle(session, person_id, frame_idx, joint_idx)
    return aa_to_euler(aa)


def _get_joint_axis_angle(session: Session, person_id: int, frame_idx: int, joint_idx: int) -> np.ndarray:
    """Get current axis-angle (3,) for a joint, including committed corrections."""
    if session is None or person_id < 0:
        return np.zeros(3, dtype=np.float32)

    track = session.person_tracks.get(person_id)
    if track is None or track.smplx_params is None:
        return np.zeros(3, dtype=np.float32)

    params = track.smplx_params

    # Check for existing committed correction first
    ct = session.correction_tracks.get(person_id)
    if ct is not None:
        corr = ct.get_correction(frame_idx)
        if corr is not None:
            if joint_idx == 0 and corr.global_orient is not None:
                return np.asarray(corr.global_orient, dtype=np.float32).ravel()[:3]
            elif joint_idx > 0 and corr.body_pose is not None:
                bp_idx = joint_idx - 1
                if bp_idx in corr.body_pose:
                    return np.asarray(corr.body_pose[bp_idx], dtype=np.float32).ravel()[:3]

    # Fall back to raw params
    try:
        if joint_idx == 0:
            go = np.asarray(params["global_orient"], dtype=np.float32)
            if go.ndim >= 2 and frame_idx < go.shape[0]:
                return go[frame_idx].ravel()[:3]
            elif go.ndim == 1:
                return go.ravel()[:3]
        elif 1 <= joint_idx <= 21:
            bp = np.asarray(params["body_pose"], dtype=np.float32)
            if bp.ndim == 2 and bp.shape[-1] != 3:
                bp = bp.reshape(bp.shape[0], -1, 3)
            bp_idx = joint_idx - 1
            if bp.ndim >= 3 and frame_idx < bp.shape[0] and bp_idx < bp.shape[1]:
                return bp[frame_idx, bp_idx].ravel()[:3]
            elif bp.ndim == 2 and bp_idx < bp.shape[0]:
                return bp[bp_idx].ravel()[:3]
        elif 22 <= joint_idx <= 36:
            lh = params.get("left_hand_pose")
            if lh is not None:
                lh = np.asarray(lh, dtype=np.float32)
                if lh.ndim == 2 and lh.shape[-1] != 3:
                    lh = lh.reshape(lh.shape[0], -1, 3)
                hi = joint_idx - 22
                if lh.ndim >= 3 and frame_idx < lh.shape[0] and hi < lh.shape[1]:
                    return lh[frame_idx, hi].ravel()[:3]
        elif 37 <= joint_idx <= 51:
            rh = params.get("right_hand_pose")
            if rh is not None:
                rh = np.asarray(rh, dtype=np.float32)
                if rh.ndim == 2 and rh.shape[-1] != 3:
                    rh = rh.reshape(rh.shape[0], -1, 3)
                hi = joint_idx - 37
                if rh.ndim >= 3 and frame_idx < rh.shape[0] and hi < rh.shape[1]:
                    return rh[frame_idx, hi].ravel()[:3]
    except Exception:
        pass

    return np.zeros(3, dtype=np.float32)


# ------------------------------------------------------------------
# Auto-Detect: Pose Issue Detection
# ------------------------------------------------------------------


@dataclass
class PoseIssue:
    """A flagged pose quality issue detected by automatic scanning.

    Why: Manual frame-by-frame pose review is tedious. Auto-detection
    highlights problematic spans — large angular jumps (bad HMR estimates),
    jitter (high-frequency oscillation), and low confidence — so users can
    focus correction effort on the frames that matter most.
    """

    frame: int                   # representative frame (midpoint of span)
    person_id: int
    issue_type: str              # "angular_jump", "low_confidence", "jitter"
    description: str
    severity: float              # 0-1, higher = more urgent
    span: tuple[int, int]        # (start, end) inclusive frame range


def _rotation_angles_deg(poses_a: np.ndarray, poses_b: np.ndarray) -> np.ndarray:
    """Compute per-joint angular difference between axis-angle arrays (degrees).

    Uses L2 norm of axis-angle difference as an efficient approximation.
    For detecting large jumps this is sufficient; exact geodesic distance
    would require scipy Rotation composition.

    Parameters
    ----------
    poses_a, poses_b : (..., 3) axis-angle arrays of matching shape.

    Returns
    -------
    (...) angular differences in degrees.
    """
    diff = np.linalg.norm(poses_b - poses_a, axis=-1)
    return np.degrees(diff)


def compute_pose_issues(
    session: Session,
    jump_threshold_deg: float = 45.0,
    jitter_window: int = 7,
    jitter_threshold_deg: float = 20.0,
    conf_threshold: float = 0.4,
    min_span: int = 3,
) -> list[PoseIssue]:
    """Scan active person tracks for pose quality issues.

    Detects three issue types:
    - angular_jump: single-frame change > jump_threshold_deg in any body joint
    - jitter: high std of angular velocity in sliding window > jitter_threshold_deg
    - low_confidence: spans of >= min_span frames where confidence < conf_threshold

    Returns issues sorted by severity (desc) then frame (asc).
    """
    issues: list[PoseIssue] = []

    for pid, track in session.person_tracks.items():
        if pid in session.inactive_tracks:
            continue

        params = track.smplx_params
        if params is not None:
            bp = params.get("body_pose")
            if bp is not None:
                bp = np.asarray(bp, dtype=np.float32)
                if bp.ndim == 2 and bp.shape[-1] != 3:
                    bp = bp.reshape(bp.shape[0], -1, 3)
                if bp.ndim == 3 and bp.shape[0] > 1:
                    issues.extend(
                        _detect_angular_jumps(bp, pid, jump_threshold_deg)
                    )
                    issues.extend(
                        _detect_jitter(bp, pid, jitter_window, jitter_threshold_deg)
                    )

        confs = track.confidences
        if confs and len(confs) > 0:
            issues.extend(
                _detect_low_confidence(confs, pid, conf_threshold, min_span)
            )

    issues.sort(key=lambda i: (-i.severity, i.frame))
    return issues


def _detect_angular_jumps(
    body_pose: np.ndarray,
    person_id: int,
    threshold_deg: float = 45.0,
) -> list[PoseIssue]:
    """Detect frames with large sudden rotations in body pose.

    Computes per-joint angular change between consecutive frames using
    vectorized operations. Flags any frame where the worst joint exceeds
    the threshold.
    """
    # body_pose: (N, J, 3)
    angles_deg = _rotation_angles_deg(body_pose[:-1], body_pose[1:])  # (N-1, J)
    max_per_frame = angles_deg.max(axis=1)       # (N-1,)
    worst_joint = angles_deg.argmax(axis=1)      # (N-1,)

    issues: list[PoseIssue] = []
    for i in range(len(max_per_frame)):
        if max_per_frame[i] >= threshold_deg:
            f = i + 1  # change detected AT frame f
            j = int(worst_joint[i])
            j_name = (
                JOINT_NAMES[j + 1]
                if j + 1 < len(JOINT_NAMES)
                else f"joint_{j}"
            )
            severity = min(1.0, float(max_per_frame[i]) / 180.0)
            issues.append(PoseIssue(
                frame=f,
                person_id=person_id,
                issue_type="angular_jump",
                description=(
                    f"Large pose jump: {j_name} changed "
                    f"{max_per_frame[i]:.1f}\u00b0 at frame {f}"
                ),
                severity=severity,
                span=(max(0, f - 1), f),
            ))

    return issues


def _detect_jitter(
    body_pose: np.ndarray,
    person_id: int,
    window: int = 7,
    threshold_deg: float = 20.0,
) -> list[PoseIssue]:
    """Detect spans with high-frequency oscillation in body pose.

    Computes the max angular velocity per frame, then uses a sliding
    window standard deviation to identify noisy spans.
    """
    n_frames = body_pose.shape[0]
    if n_frames < window + 1:
        return []

    angles_deg = _rotation_angles_deg(body_pose[:-1], body_pose[1:])  # (N-1, J)
    ang_vel = angles_deg.max(axis=1)  # (N-1,)

    half_w = window // 2
    issues: list[PoseIssue] = []
    in_span = False
    span_start = 0

    for f in range(half_w, len(ang_vel) - half_w):
        w_slice = ang_vel[f - half_w : f + half_w + 1]
        std_val = float(np.std(w_slice))
        if std_val >= threshold_deg:
            if not in_span:
                span_start = f
                in_span = True
        else:
            if in_span:
                span_len = f - span_start
                if span_len >= 3:
                    mid = (span_start + f - 1) // 2
                    issues.append(PoseIssue(
                        frame=mid,
                        person_id=person_id,
                        issue_type="jitter",
                        description=(
                            f"Jitter frames {span_start}\u2013{f - 1} "
                            f"({span_len} frames)"
                        ),
                        severity=0.6,
                        span=(span_start, f - 1),
                    ))
                in_span = False

    if in_span:
        f_end = len(ang_vel) - half_w
        span_len = f_end - span_start
        if span_len >= 3:
            mid = (span_start + f_end - 1) // 2
            issues.append(PoseIssue(
                frame=mid,
                person_id=person_id,
                issue_type="jitter",
                description=(
                    f"Jitter frames {span_start}\u2013{f_end - 1} "
                    f"({span_len} frames)"
                ),
                severity=0.6,
                span=(span_start, f_end - 1),
            ))

    return issues


def _detect_low_confidence(
    confidences: list,
    person_id: int,
    threshold: float = 0.4,
    min_span: int = 3,
) -> list[PoseIssue]:
    """Detect spans of low confidence scores.

    Handles both raw float lists and TrackConfidence objects (with .overall).
    """
    n = len(confidences)
    conf_values = []
    for c in confidences:
        if isinstance(c, (int, float)):
            conf_values.append(float(c))
        elif hasattr(c, "overall"):
            conf_values.append(float(c.overall))
        else:
            conf_values.append(1.0)

    issues: list[PoseIssue] = []
    span_start: int | None = None

    for f in range(n):
        if conf_values[f] < threshold:
            if span_start is None:
                span_start = f
        else:
            if span_start is not None and f - span_start >= min_span:
                mid = (span_start + f - 1) // 2
                issues.append(PoseIssue(
                    frame=mid,
                    person_id=person_id,
                    issue_type="low_confidence",
                    description=(
                        f"Low confidence frames {span_start}\u2013{f - 1} "
                        f"({f - span_start} frames)"
                    ),
                    severity=0.7,
                    span=(span_start, f - 1),
                ))
            span_start = None

    if span_start is not None and n - span_start >= min_span:
        mid = (span_start + n - 1) // 2
        issues.append(PoseIssue(
            frame=mid,
            person_id=person_id,
            issue_type="low_confidence",
            description=(
                f"Low confidence frames {span_start}\u2013{n - 1} "
                f"({n - span_start} frames)"
            ),
            severity=0.7,
            span=(span_start, n - 1),
        ))

    return issues


def smooth_joint_rotations(
    body_pose: np.ndarray,
    start: int,
    end: int,
    joint_indices: list[int] | None = None,
    window: int = 7,
    method: str = "gaussian",
) -> np.ndarray:
    """Apply temporal smoothing to joint rotations over a frame range.

    Why: GVHMR body pose estimates often exhibit high-frequency jitter,
    especially in occluded joints. Temporal smoothing reduces this noise
    while preserving deliberate motion. Operates on axis-angle representation
    directly — sufficient for small perturbations where geodesic smoothing
    would be overkill.

    Parameters
    ----------
    body_pose : (N, J, 3) axis-angle array
    start, end : inclusive frame range to smooth
    joint_indices : which joints to smooth (body_pose indices, 0-based).
        None means all joints.
    window : smoothing kernel size (must be odd, >= 3)
    method : "gaussian" or "moving_average"

    Returns
    -------
    (N, J, 3) smoothed copy — only the [start:end+1] range is modified.
    """
    if body_pose.ndim != 3:
        return body_pose.copy()

    n_frames, n_joints, _ = body_pose.shape
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1

    result = body_pose.copy()
    start = max(0, start)
    end = min(n_frames - 1, end)

    if joint_indices is None:
        joint_indices = list(range(n_joints))

    half = window // 2

    if method == "gaussian":
        x = np.arange(-half, half + 1, dtype=np.float64)
        sigma = half / 2.0
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
    else:
        kernel = np.ones(window, dtype=np.float64) / window

    for ji in joint_indices:
        if ji < 0 or ji >= n_joints:
            continue
        for f in range(start, end + 1):
            w_start = max(0, f - half)
            w_end = min(n_frames - 1, f + half)
            k_start = w_start - (f - half)
            k_end = window - ((f + half) - w_end)
            k_slice = kernel[k_start:k_end]
            k_slice = k_slice / k_slice.sum()  # renormalize at edges
            result[f, ji] = np.einsum(
                "f,fd->d", k_slice, body_pose[w_start : w_end + 1, ji],
            )

    return result


def find_similar_frames(
    body_pose: np.ndarray,
    frame_idx: int,
    joint_idx: int,
    threshold_deg: float = 15.0,
) -> list[int]:
    """Find frames where a joint's rotation is similar to the reference frame.

    Why: When GVHMR produces the same bad pose estimate repeatedly (e.g.,
    a flipped arm in every other frame), manually correcting each frame is
    tedious. Finding similar poses lets the user apply one correction to all
    matching frames at once.

    Parameters
    ----------
    body_pose : (N, J, 3) axis-angle array
    frame_idx : reference frame
    joint_idx : body_pose joint index (0-based, not SMPL-X index)
    threshold_deg : maximum angular distance for a match

    Returns
    -------
    Sorted list of matching frame indices (excluding the reference frame).
    """
    if body_pose.ndim != 3:
        return []

    n_frames, n_joints, _ = body_pose.shape
    if joint_idx < 0 or joint_idx >= n_joints:
        return []
    if frame_idx < 0 or frame_idx >= n_frames:
        return []

    ref = body_pose[frame_idx, joint_idx]  # (3,)
    all_joints = body_pose[:, joint_idx]  # (N, 3)
    diffs = np.linalg.norm(all_joints - ref, axis=-1)  # (N,)
    diffs_deg = np.degrees(diffs)
    matches = np.where(diffs_deg <= threshold_deg)[0].tolist()
    # Remove the reference frame itself
    if frame_idx in matches:
        matches.remove(frame_idx)
    return sorted(matches)


def propagate_corrections(
    start_frame: int,
    end_frame: int,
    start_aa: np.ndarray,
    end_aa: np.ndarray,
) -> dict[int, np.ndarray]:
    """Interpolate corrections between two keyframes using SLERP.

    Why: When the user corrects poses at two keyframes (e.g., frame 10 and
    frame 50), the correction should transition smoothly between them.
    SLERP on the rotation produces natural interpolation through rotation
    space, avoiding the gimbal-lock artifacts of linear euler interpolation.

    Parameters
    ----------
    start_frame, end_frame : inclusive frame range
    start_aa, end_aa : (3,) axis-angle rotations at keyframes

    Returns
    -------
    Dict mapping frame_index → (3,) axis-angle for each intermediate frame.
    Start and end frames are included.
    """
    if end_frame <= start_frame:
        return {start_frame: start_aa.copy()}

    try:
        from scipy.spatial.transform import Rotation, Slerp

        rots = Rotation.from_rotvec(np.stack([start_aa, end_aa]).astype(np.float64))
        times = [0.0, 1.0]
        slerp = Slerp(times, rots)

        n = end_frame - start_frame + 1
        t_values = np.linspace(0.0, 1.0, n)
        interp = slerp(t_values)
        result = {}
        for i, f in enumerate(range(start_frame, end_frame + 1)):
            result[f] = interp[i].as_rotvec().astype(np.float32)
        return result
    except ImportError:
        # Fallback: linear interpolation in axis-angle space
        n = end_frame - start_frame + 1
        result = {}
        for i, f in enumerate(range(start_frame, end_frame + 1)):
            t = i / max(1, n - 1)
            result[f] = ((1.0 - t) * start_aa + t * end_aa).astype(np.float32)
        return result


class PoseCorrectorPanel(QWidget):
    """Pose correction panel with embedded 3D viewport and joint controls.

    Signals
    -------
    joint_selected(int)
        Emitted when a joint is selected (by viewport click or dropdown).
    correction_applied(int, int)
        Emitted when a correction is committed (person_id, frame_index).
    """

    joint_selected = Signal(int)
    correction_applied = Signal(int, int)
    frame_requested = Signal(int)  # corrections table "Go" → seek to frame
    export_requested = Signal(str)  # "bvh" or "fbx"

    def __init__(self, session: Session, gvhmr_root: Path | None = None, parent=None):
        super().__init__(parent)
        self._session = session
        self._gvhmr_root = gvhmr_root
        self._current_person: int = -1
        self._current_frame: int = 0
        self._current_joint: int = -1
        self._updating_sliders: bool = False  # guard against signal loops

        # Auto-detect state
        self._pose_issues: list[PoseIssue] = []
        self._pose_issue_idx: int = 0

        self._setup_ui()
        self._connect_signals()

    @property
    def mesh_viewport(self) -> MeshViewport:
        """Expose embedded MeshViewport for external signal wiring."""
        return self._viewport

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)

        # ---- Left: Viewport + camera/color mode ----
        viewport_widget = QWidget()
        vp_layout = QVBoxLayout(viewport_widget)
        vp_layout.setContentsMargins(4, 4, 4, 4)

        self._viewport = MeshViewport(gvhmr_root=self._gvhmr_root)
        self._viewport.set_session(self._session)
        vp_layout.addWidget(self._viewport, stretch=1)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Camera:"))
        self._camera_combo = QComboBox()
        self._camera_combo.addItems(["In-camera", "Free orbit"])
        mode_row.addWidget(self._camera_combo)
        mode_row.addWidget(QLabel("Color:"))
        self._color_combo = QComboBox()
        self._color_combo.addItems(["Solid", "Joint influence", "Confidence"])
        mode_row.addWidget(self._color_combo)
        self._labels_checkbox = QCheckBox("Labels")
        self._labels_checkbox.setToolTip("Show joint name labels on skeleton")
        mode_row.addWidget(self._labels_checkbox)
        mode_row.addStretch()
        vp_layout.addLayout(mode_row)

        # Corrections table
        corr_group = QGroupBox("Corrections")
        corr_layout = QVBoxLayout(corr_group)
        self._corrections_table = QTableWidget(0, 5)
        self._corrections_table.setHorizontalHeaderLabels(
            ["Frame", "Type", "Joint", "Go", "Del"]
        )
        self._corrections_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self._corrections_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._corrections_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._corrections_table.setMaximumHeight(160)
        corr_layout.addWidget(self._corrections_table)
        vp_layout.addWidget(corr_group)

        # Export group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        export_row = QHBoxLayout()
        self._reexport_bvh_btn = QPushButton("Re-export BVH")
        self._reexport_bvh_btn.setToolTip("Apply corrections + space overrides and export to BVH")
        export_row.addWidget(self._reexport_bvh_btn)
        self._reexport_fbx_btn = QPushButton("Re-export FBX")
        self._reexport_fbx_btn.setToolTip("Export BVH then convert to FBX via Blender")
        export_row.addWidget(self._reexport_fbx_btn)
        export_layout.addLayout(export_row)
        self._export_status = QLabel("")
        self._export_status.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        self._export_status.setWordWrap(True)
        export_layout.addWidget(self._export_status)
        vp_layout.addWidget(export_group)

        splitter.addWidget(viewport_widget)

        # ---- Right: Joint controls ----
        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setContentsMargins(8, 8, 8, 8)

        # Person selector
        person_group = QHBoxLayout()
        person_group.addWidget(QLabel("Person:"))
        self._person_combo = QComboBox()
        person_group.addWidget(self._person_combo, stretch=1)
        ctrl_layout.addLayout(person_group)

        # Joint selector
        joint_group = QHBoxLayout()
        joint_group.addWidget(QLabel("Joint:"))
        self._joint_combo = QComboBox()
        self._populate_joint_dropdown()
        joint_group.addWidget(self._joint_combo, stretch=1)
        ctrl_layout.addLayout(joint_group)

        # Joint info label
        self._joint_info = QLabel("")
        self._joint_info.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        self._joint_info.setWordWrap(True)
        ctrl_layout.addWidget(self._joint_info)

        # Euler rotation group
        rot_group = QGroupBox("Rotation (degrees)")
        rot_layout = QVBoxLayout(rot_group)

        self._euler_x, self._slider_x = self._make_euler_row("X:", rot_layout)
        self._euler_y, self._slider_y = self._make_euler_row("Y:", rot_layout)
        self._euler_z, self._slider_z = self._make_euler_row("Z:", rot_layout)

        ctrl_layout.addWidget(rot_group)

        # Preview range — auto-play ±N frames after correction to see impact
        preview_group = QGroupBox("Preview Range")
        preview_layout = QVBoxLayout(preview_group)

        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("±"))
        self._preview_half_range = QSpinBox()
        self._preview_half_range.setRange(1, 60)
        self._preview_half_range.setValue(15)
        self._preview_half_range.setSuffix(" frames")
        self._preview_half_range.setToolTip(
            "Number of frames before and after the current frame to preview"
        )
        preview_row.addWidget(self._preview_half_range)
        self._preview_btn = QPushButton("Preview")
        self._preview_btn.setToolTip("Play ±N frames around current frame")
        preview_row.addWidget(self._preview_btn)
        preview_layout.addLayout(preview_row)

        self._auto_preview_check = QCheckBox("Auto-preview after Apply")
        self._auto_preview_check.setToolTip(
            "Automatically play the preview range after applying a correction"
        )
        preview_layout.addWidget(self._auto_preview_check)

        ctrl_layout.addWidget(preview_group)

        # Preview playback timer
        self._preview_timer = QTimer(self)
        self._preview_timer.setInterval(33)  # ~30fps
        self._preview_frame = 0
        self._preview_end = 0

        # Action buttons
        btn_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setToolTip("Commit correction as keyframe")
        btn_row.addWidget(self._apply_btn)

        self._reset_joint_btn = QPushButton("Reset Joint")
        self._reset_joint_btn.setToolTip("Reset selected joint to original pose")
        btn_row.addWidget(self._reset_joint_btn)

        self._reset_all_btn = QPushButton("Reset All")
        self._reset_all_btn.setToolTip("Remove all corrections at this frame")
        btn_row.addWidget(self._reset_all_btn)

        ctrl_layout.addLayout(btn_row)

        # Frame Range group — preview and apply corrections across frames
        range_group = QGroupBox("Frame Range")
        range_layout = QVBoxLayout(range_group)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Start:"))
        self._range_start = QSpinBox()
        self._range_start.setRange(0, 999999)
        range_row.addWidget(self._range_start)
        range_row.addWidget(QLabel("End:"))
        self._range_end = QSpinBox()
        self._range_end.setRange(0, 999999)
        range_row.addWidget(self._range_end)
        range_layout.addLayout(range_row)

        self._apply_range_check = QCheckBox("Apply to range")
        self._apply_range_check.setToolTip(
            "When checked, Apply commits the correction to all frames in range"
        )
        range_layout.addWidget(self._apply_range_check)

        ctrl_layout.addWidget(range_group)

        # Quick Fix group
        qf_group = QGroupBox("Quick Fix")
        qf_layout = QVBoxLayout(qf_group)

        qf_row1 = QHBoxLayout()
        self._flip_btn = QPushButton("Flip Body")
        self._flip_btn.setToolTip("180° rotation on selected axis (yaw/pitch/roll)")
        qf_row1.addWidget(self._flip_btn)
        self._invert_btn = QPushButton("Invert")
        self._invert_btn.setToolTip("Flip upside-down pose (180° pitch)")
        qf_row1.addWidget(self._invert_btn)
        qf_layout.addLayout(qf_row1)

        qf_row2 = QHBoxLayout()
        self._mirror_btn = QPushButton("Mirror L/R")
        self._mirror_btn.setToolTip("Swap left/right joint pairs")
        qf_row2.addWidget(self._mirror_btn)
        self._copy_from_btn = QPushButton("Copy From")
        self._copy_from_btn.setToolTip("Copy pose from another frame")
        qf_row2.addWidget(self._copy_from_btn)
        qf_layout.addLayout(qf_row2)

        ctrl_layout.addWidget(qf_group)

        # Smoothing group — temporal smoothing to reduce jitter
        smooth_group = QGroupBox("Smoothing")
        smooth_layout = QVBoxLayout(smooth_group)

        smooth_row1 = QHBoxLayout()
        smooth_row1.addWidget(QLabel("Window:"))
        self._smooth_window = QSpinBox()
        self._smooth_window.setRange(3, 31)
        self._smooth_window.setSingleStep(2)
        self._smooth_window.setValue(7)
        self._smooth_window.setToolTip("Kernel size (odd, 3-31 frames)")
        smooth_row1.addWidget(self._smooth_window)
        smooth_row1.addWidget(QLabel("Method:"))
        self._smooth_method = QComboBox()
        self._smooth_method.addItems(["Gaussian", "Moving Average"])
        smooth_row1.addWidget(self._smooth_method)
        smooth_layout.addLayout(smooth_row1)

        smooth_row2 = QHBoxLayout()
        smooth_row2.addWidget(QLabel("Scope:"))
        self._smooth_scope = QComboBox()
        self._smooth_scope.addItems(["Current Joint", "All Body Joints"])
        smooth_row2.addWidget(self._smooth_scope)
        self._smooth_btn = QPushButton("Smooth")
        self._smooth_btn.setToolTip(
            "Apply temporal smoothing to the selected joint(s) over the frame range"
        )
        smooth_row2.addWidget(self._smooth_btn)
        smooth_layout.addLayout(smooth_row2)

        ctrl_layout.addWidget(smooth_group)

        # Apply-to-Similar group — find and correct similar poses
        sim_group = QGroupBox("Apply to Similar")
        sim_layout = QVBoxLayout(sim_group)

        sim_row = QHBoxLayout()
        sim_row.addWidget(QLabel("Threshold:"))
        self._sim_threshold = QDoubleSpinBox()
        self._sim_threshold.setRange(1.0, 90.0)
        self._sim_threshold.setValue(15.0)
        self._sim_threshold.setSingleStep(1.0)
        self._sim_threshold.setSuffix("°")
        self._sim_threshold.setToolTip("Maximum angular distance for a match")
        sim_row.addWidget(self._sim_threshold)
        self._sim_apply_btn = QPushButton("Apply to Similar")
        self._sim_apply_btn.setToolTip(
            "Find frames with similar joint rotation and apply the same correction"
        )
        sim_row.addWidget(self._sim_apply_btn)
        sim_layout.addLayout(sim_row)

        self._sim_status = QLabel("")
        self._sim_status.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        self._sim_status.setWordWrap(True)
        sim_layout.addWidget(self._sim_status)

        ctrl_layout.addWidget(sim_group)

        # Propagation group — interpolate corrections between keyframes
        prop_group = QGroupBox("Correction Propagation")
        prop_layout = QVBoxLayout(prop_group)

        self._propagate_btn = QPushButton("Propagate (SLERP)")
        self._propagate_btn.setToolTip(
            "Interpolate the current joint correction across the frame range "
            "using spherical linear interpolation (SLERP). Set start/end "
            "frames in the Frame Range group above."
        )
        prop_layout.addWidget(self._propagate_btn)

        self._prop_status = QLabel("")
        self._prop_status.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        self._prop_status.setWordWrap(True)
        prop_layout.addWidget(self._prop_status)

        ctrl_layout.addWidget(prop_group)

        # Auto-Detect group
        ad_group = QGroupBox("Auto-Detect")
        ad_layout = QVBoxLayout(ad_group)

        ad_btn_row = QHBoxLayout()
        self._detect_btn = QPushButton("Detect Bad Spans")
        self._detect_btn.setToolTip(
            "Scan for angular jumps, jitter, and low confidence spans"
        )
        ad_btn_row.addWidget(self._detect_btn)
        self._prev_issue_btn = QPushButton("\u25c4 Prev")
        self._prev_issue_btn.setToolTip("Navigate to previous issue")
        self._prev_issue_btn.setEnabled(False)
        ad_btn_row.addWidget(self._prev_issue_btn)
        self._next_issue_btn = QPushButton("\u25ba Next")
        self._next_issue_btn.setToolTip("Navigate to next issue")
        self._next_issue_btn.setEnabled(False)
        ad_btn_row.addWidget(self._next_issue_btn)
        ad_layout.addLayout(ad_btn_row)

        self._issues_label = QLabel("No scan performed")
        self._issues_label.setStyleSheet("font-style: italic;")
        ad_layout.addWidget(self._issues_label)

        self._issues_table = QTableWidget(0, 4)
        self._issues_table.setHorizontalHeaderLabels(
            ["Frame", "Person", "Type", "Description"]
        )
        self._issues_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._issues_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._issues_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._issues_table.verticalHeader().hide()
        self._issues_table.setMaximumHeight(120)
        ad_layout.addWidget(self._issues_table)

        ctrl_layout.addWidget(ad_group)

        # Space Overrides group
        so_group = QGroupBox("Space Overrides")
        so_layout = QVBoxLayout(so_group)

        so_row1 = QHBoxLayout()
        so_row1.addWidget(QLabel("Space:"))
        self._space_combo = QComboBox()
        self._space_combo.addItems(["World", "Camera", "Carried"])
        so_row1.addWidget(self._space_combo)
        so_layout.addLayout(so_row1)

        so_row2 = QHBoxLayout()
        so_row2.addWidget(QLabel("Start:"))
        self._space_start = QSpinBox()
        self._space_start.setRange(0, 999999)
        so_row2.addWidget(self._space_start)
        so_row2.addWidget(QLabel("End:"))
        self._space_end = QSpinBox()
        self._space_end.setRange(0, 999999)
        so_row2.addWidget(self._space_end)
        so_layout.addLayout(so_row2)

        so_row3 = QHBoxLayout()
        so_row3.addWidget(QLabel("Ref Person:"))
        self._space_ref_combo = QComboBox()
        self._space_ref_combo.addItem("—", userData=None)
        so_row3.addWidget(self._space_ref_combo)
        so_row3.addWidget(QLabel("Y offset:"))
        self._space_y_offset = QDoubleSpinBox()
        self._space_y_offset.setRange(0.0, 5.0)
        self._space_y_offset.setValue(0.4)
        self._space_y_offset.setDecimals(2)
        self._space_y_offset.setSuffix(" m")
        so_row3.addWidget(self._space_y_offset)
        so_layout.addLayout(so_row3)

        so_btn_row = QHBoxLayout()
        self._add_override_btn = QPushButton("Add Override")
        so_btn_row.addWidget(self._add_override_btn)
        self._del_override_btn = QPushButton("Delete Selected")
        so_btn_row.addWidget(self._del_override_btn)
        so_layout.addLayout(so_btn_row)

        self._space_table = QTableWidget(0, 5)
        self._space_table.setHorizontalHeaderLabels(
            ["Start", "End", "Space", "Ref Person", "Y Offset"]
        )
        self._space_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self._space_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._space_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._space_table.setMaximumHeight(120)
        so_layout.addWidget(self._space_table)

        ctrl_layout.addWidget(so_group)

        ctrl_layout.addStretch()

        splitter.addWidget(controls)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

    def _make_euler_row(self, label: str, parent_layout: QVBoxLayout):
        """Create a slider + spinbox row for one euler angle.

        Returns (spinbox, slider) tuple.
        """
        row = QHBoxLayout()
        row.addWidget(QLabel(label))

        slider = QSlider(Qt.Horizontal)
        slider.setRange(-180, 180)
        slider.setValue(0)
        slider.setSingleStep(1)
        row.addWidget(slider, stretch=1)

        spinbox = QDoubleSpinBox()
        spinbox.setRange(-180.0, 180.0)
        spinbox.setValue(0.0)
        spinbox.setSingleStep(0.1)
        spinbox.setDecimals(1)
        spinbox.setSuffix("°")
        spinbox.setFixedWidth(80)
        row.addWidget(spinbox)

        parent_layout.addLayout(row)
        return spinbox, slider

    def _populate_joint_dropdown(self):
        """Populate joint dropdown with all SMPL-X joints."""
        self._joint_combo.clear()
        # Body joints (0-21)
        for i in range(min(_N_BODY_JOINTS, len(JOINT_NAMES))):
            self._joint_combo.addItem(f"{i}: {JOINT_NAMES[i]}", userData=i)
        # Separator then hand joints
        if len(JOINT_NAMES) > _N_BODY_JOINTS:
            self._joint_combo.insertSeparator(self._joint_combo.count())
            for i in range(_N_BODY_JOINTS, len(JOINT_NAMES)):
                self._joint_combo.addItem(f"{i}: {JOINT_NAMES[i]}", userData=i)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        # Viewport joint click → update dropdown + sliders
        self._viewport.joint_clicked.connect(self._on_joint_clicked)

        # Dropdowns
        self._joint_combo.currentIndexChanged.connect(self._on_joint_dropdown_changed)
        self._person_combo.currentIndexChanged.connect(self._on_person_dropdown_changed)

        # Camera/color mode and labels toggle
        self._camera_combo.currentIndexChanged.connect(self._on_camera_mode_changed)
        self._color_combo.currentIndexChanged.connect(self._on_color_mode_changed)
        self._labels_checkbox.toggled.connect(self._viewport.set_show_joint_labels)

        # Euler spinboxes (primary — sliders sync from these)
        self._euler_x.valueChanged.connect(self._on_euler_changed)
        self._euler_y.valueChanged.connect(self._on_euler_changed)
        self._euler_z.valueChanged.connect(self._on_euler_changed)

        # Slider → spinbox sync
        self._slider_x.valueChanged.connect(lambda v: self._on_slider_moved(self._euler_x, v))
        self._slider_y.valueChanged.connect(lambda v: self._on_slider_moved(self._euler_y, v))
        self._slider_z.valueChanged.connect(lambda v: self._on_slider_moved(self._euler_z, v))

        # Preview
        self._preview_btn.clicked.connect(self._on_preview_play)
        self._preview_timer.timeout.connect(self._on_preview_tick)

        # Action buttons
        self._apply_btn.clicked.connect(self._on_apply)
        self._reset_joint_btn.clicked.connect(self._on_reset_joint)
        self._reset_all_btn.clicked.connect(self._on_reset_all)

        # Quick fix buttons
        self._flip_btn.clicked.connect(self._on_flip_body)
        self._invert_btn.clicked.connect(self._on_invert_upright)
        self._mirror_btn.clicked.connect(self._on_mirror_lr)
        self._copy_from_btn.clicked.connect(self._on_copy_from_frame)

        # Space override buttons
        self._add_override_btn.clicked.connect(self._on_add_space_override)
        self._del_override_btn.clicked.connect(self._on_delete_space_override)

        # Export buttons
        self._reexport_bvh_btn.clicked.connect(self._on_reexport_bvh)
        self._reexport_fbx_btn.clicked.connect(self._on_reexport_fbx)

        # Smoothing
        self._smooth_btn.clicked.connect(self._on_smooth)

        # Apply to Similar
        self._sim_apply_btn.clicked.connect(self._on_apply_to_similar)

        # Propagation
        self._propagate_btn.clicked.connect(self._on_propagate)

        # Auto-detect buttons
        self._detect_btn.clicked.connect(self._on_detect_bad_spans)
        self._prev_issue_btn.clicked.connect(self._on_prev_pose_issue)
        self._next_issue_btn.clicked.connect(self._on_next_pose_issue)
        self._issues_table.cellDoubleClicked.connect(
            self._on_pose_issue_double_clicked
        )

    # ------------------------------------------------------------------
    # Public API (used by MultiPersonTab)
    # ------------------------------------------------------------------

    def set_session(self, session: Session):
        """Bind session data source."""
        self._session = session
        self._viewport.set_session(session)
        self._refresh_corrections_table()

    def set_person(self, person_id: int):
        """Select person externally (e.g., from identity inspector)."""
        if person_id == self._current_person:
            return
        self._current_person = person_id
        self._viewport.set_person(person_id)
        # Sync dropdown without re-triggering the callback
        self._person_combo.blockSignals(True)
        for i in range(self._person_combo.count()):
            if self._person_combo.itemData(i) == person_id:
                self._person_combo.setCurrentIndex(i)
                break
        self._person_combo.blockSignals(False)
        self._update_sliders()
        self._refresh_corrections_table()
        self._refresh_space_table()

    def on_frame_changed(self, frame_idx: int):
        """Update frame externally."""
        self._current_frame = frame_idx
        self._viewport.on_frame_changed(frame_idx)
        # Clear any in-progress preview when navigating frames
        self._viewport.set_pose_override(None)
        self._update_sliders()

    def refresh(self):
        """Refresh person list from session."""
        self._update_person_dropdown()
        self._update_space_ref_dropdown()
        self._refresh_corrections_table()
        self._refresh_space_table()

    # ------------------------------------------------------------------
    # Dropdown handlers
    # ------------------------------------------------------------------

    def set_joint(self, joint_idx: int):
        """Public API: select a joint by index (e.g. from external MeshViewport click).

        Syncs the dropdown, which triggers slider update and viewport highlight.
        """
        self._on_joint_clicked(joint_idx)

    def _on_joint_clicked(self, joint_idx: int):
        """Handle joint click from viewport — sync dropdown."""
        for i in range(self._joint_combo.count()):
            if self._joint_combo.itemData(i) == joint_idx:
                self._joint_combo.setCurrentIndex(i)
                return

    def _on_joint_dropdown_changed(self, idx: int):
        """Handle joint selection from dropdown."""
        if idx < 0:
            return
        joint_idx = self._joint_combo.itemData(idx)
        if joint_idx is None:
            return
        self._current_joint = joint_idx
        self._viewport.highlight_joint(joint_idx)
        # Clear any in-progress preview for the previous joint
        self._viewport.set_pose_override(None)
        self._update_sliders()
        self._update_joint_info()
        self.joint_selected.emit(joint_idx)

    def _on_person_dropdown_changed(self, idx: int):
        """Handle person selection from dropdown."""
        if idx < 0:
            return
        person_id = self._person_combo.itemData(idx)
        if person_id is None:
            return
        self._current_person = person_id
        self._viewport.set_person(person_id)
        self._viewport.set_pose_override(None)
        self._update_sliders()
        self._refresh_corrections_table()
        self._refresh_space_table()

    def _on_camera_mode_changed(self, idx: int):
        """Handle camera mode dropdown change."""
        mode = "incam" if idx == 0 else "orbit"
        self._viewport.set_camera_mode(mode)

    def _on_color_mode_changed(self, idx: int):
        """Handle color mode dropdown change."""
        modes = ["solid", "joint", "confidence"]
        if 0 <= idx < len(modes):
            self._viewport.set_color_mode(modes[idx])

    # ------------------------------------------------------------------
    # Euler slider handlers
    # ------------------------------------------------------------------

    def _on_slider_moved(self, spinbox: QDoubleSpinBox, value: int):
        """Sync slider (integer) → spinbox (float)."""
        if not self._updating_sliders:
            self._updating_sliders = True
            spinbox.setValue(float(value))
            self._updating_sliders = False

    def _on_euler_changed(self):
        """Handle euler spinbox value change — sync slider + preview."""
        if self._updating_sliders:
            return
        self._updating_sliders = True
        # Sync spinbox → slider (truncate to int)
        self._slider_x.setValue(int(self._euler_x.value()))
        self._slider_y.setValue(int(self._euler_y.value()))
        self._slider_z.setValue(int(self._euler_z.value()))
        self._updating_sliders = False

        # Real-time mesh preview
        self._preview_correction()

    # ------------------------------------------------------------------
    # Preview + Apply + Reset
    # ------------------------------------------------------------------

    def _preview_correction(self):
        """Apply current euler values as a temporary pose override for preview."""
        if self._current_joint < 0 or self._current_person < 0:
            return

        _, euler_to_aa, _ = _safe_import_pose_correction()
        if euler_to_aa is None:
            euler_to_aa = euler_deg_to_axis_angle_fallback

        euler = np.array([
            self._euler_x.value(),
            self._euler_y.value(),
            self._euler_z.value(),
        ], dtype=np.float32)

        aa = euler_to_aa(euler)

        override = {"frame_idx": self._current_frame}
        if self._current_joint == 0:
            override["global_orient"] = aa
        elif 1 <= self._current_joint <= 21:
            override["body_pose"] = {self._current_joint - 1: aa}
        else:
            # Hand joints: no preview override support yet (body joints only)
            return

        self._viewport.set_pose_override(override)

    def _snapshot_raw_param(self, joint_idx: int, frame_idx: int) -> np.ndarray | None:
        """Capture the raw param axis-angle for a joint at a frame (for undo)."""
        track = self._session.person_tracks.get(self._current_person)
        if not track or not track.smplx_params:
            return None
        params = track.smplx_params
        if joint_idx == 0:
            go = np.array(params.get("global_orient", []), dtype=np.float32)
            if go.ndim >= 2 and frame_idx < go.shape[0]:
                return go[frame_idx].copy()
        elif 1 <= joint_idx <= 21:
            bp = np.array(params.get("body_pose", []), dtype=np.float32)
            if bp.ndim == 2 and bp.shape[-1] != 3:
                bp = bp.reshape(bp.shape[0], -1, 3)
            bp_idx = joint_idx - 1
            if bp.ndim >= 3 and frame_idx < bp.shape[0] and bp_idx < bp.shape[1]:
                return bp[frame_idx, bp_idx].copy()
        return None

    def _snapshot_correction(self, pid: int, frame_idx: int):
        """Capture the current PoseCorrection entry for a person/frame (for undo).

        Returns a dict with go, bp, ctype or None if no correction exists.
        """
        ct = self._session.correction_tracks.get(pid)
        if ct is None:
            return None
        corr = ct.get_correction(frame_idx)
        if corr is None:
            return None
        return {
            "global_orient": corr.global_orient.copy() if corr.global_orient is not None else None,
            "body_pose": {k: v.copy() for k, v in corr.body_pose.items()} if corr.body_pose else None,
            "correction_type": corr.correction_type,
        }

    def _restore_correction(self, pid: int, frame_idx: int, snap):
        """Restore a correction from a snapshot (for undo/redo)."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            return
        ct = self._session.correction_tracks.get(pid)
        if snap is None:
            if ct is not None:
                ct.remove_correction(frame_idx)
        else:
            if ct is None:
                ct = CorrectionTrack(person_id=pid)
                self._session.correction_tracks[pid] = ct
            ct.add_correction(
                frame_index=frame_idx,
                correction_type=snap["correction_type"],
                global_orient=snap["global_orient"],
                body_pose=snap["body_pose"],
            )

    def _apply_correction_at_frame(
        self,
        pid: int,
        frame: int,
        joint: int,
        aa: np.ndarray,
        CorrectionTrack,
    ):
        """Apply a single correction at one frame (helper for _on_apply).

        Returns (old_corr_snap, old_raw, new_corr_snap, new_raw) for undo.
        """
        old_corr_snap = self._snapshot_correction(pid, frame)
        old_raw = self._snapshot_raw_param(joint, frame)

        if pid not in self._session.correction_tracks:
            self._session.correction_tracks[pid] = CorrectionTrack(person_id=pid)
        ct = self._session.correction_tracks[pid]

        go = None
        bp = None
        ctype = "joint"

        if joint == 0:
            go = aa
            ctype = "global_orient"
        elif 1 <= joint <= 21:
            bp = {joint - 1: aa}
        else:
            return None

        existing = ct.get_correction(frame)
        if existing is not None:
            if go is None:
                go = existing.global_orient
            if bp is None:
                bp = existing.body_pose
            elif existing.body_pose is not None:
                merged_bp = dict(existing.body_pose)
                merged_bp.update(bp)
                bp = merged_bp

        ct.add_correction(
            frame_index=frame,
            correction_type=ctype,
            global_orient=go,
            body_pose=bp,
        )

        self._apply_to_raw_params(joint, aa, frame)

        new_corr_snap = self._snapshot_correction(pid, frame)
        new_raw = self._snapshot_raw_param(joint, frame)

        return (old_corr_snap, old_raw, new_corr_snap, new_raw)

    def _on_apply(self):
        """Commit current correction to CorrectionTrack.

        When 'Apply to range' is checked, applies the same correction to
        every frame in the Frame Range. Otherwise applies to current frame only.
        """
        if self._current_joint < 0 or self._current_person < 0:
            return

        _, euler_to_aa, CorrectionTrack = _safe_import_pose_correction()
        if euler_to_aa is None or CorrectionTrack is None:
            log.warning("Cannot apply correction: pose_correction backend unavailable")
            return

        euler = np.array([
            self._euler_x.value(),
            self._euler_y.value(),
            self._euler_z.value(),
        ], dtype=np.float32)

        aa = euler_to_aa(euler)

        pid = self._current_person
        joint = self._current_joint

        if joint > 21:
            log.info("Hand joint corrections not supported in CorrectionTrack")
            return

        # Determine which frames to apply to
        if self._apply_range_check.isChecked():
            f_start = self._range_start.value()
            f_end = self._range_end.value()
            if f_end < f_start:
                f_start, f_end = f_end, f_start
            frames = list(range(f_start, f_end + 1))
        else:
            frames = [self._current_frame]

        # Collect undo data for all frames
        undo_data = []
        for f in frames:
            result = self._apply_correction_at_frame(pid, f, joint, aa, CorrectionTrack)
            if result is not None:
                undo_data.append((f, result))

        if not undo_data:
            return

        def undo(p=pid, j=joint, ud=undo_data):
            self._current_person = p
            for f, (oc, oraw, _nc, _nraw) in ud:
                self._restore_correction(p, f, oc)
                if oraw is not None:
                    self._apply_to_raw_params(j, oraw, f)
                self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()
            self._refresh_corrections_table()

        def redo(p=pid, j=joint, ud=undo_data):
            self._current_person = p
            for f, (_oc, _oraw, nc, nraw) in ud:
                self._restore_correction(p, f, nc)
                if nraw is not None:
                    self._apply_to_raw_params(j, nraw, f)
                self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()
            self._refresh_corrections_table()

        n = len(frames)
        desc = f"Apply correction ({n} frame{'s' if n > 1 else ''})"
        self._session.undo_stack.push(UndoEntry(desc, undo, redo))

        self._viewport.set_pose_override(None)
        self._save_correction_track(pid)
        self._refresh_corrections_table()

        log.info("Correction applied: person=%d, %d frames, joint=%d",
                 pid, n, self._current_joint)
        self.correction_applied.emit(pid, frames[0])
        self._trigger_auto_preview()

    def _on_reset_joint(self):
        """Reset current joint to original pose (remove from correction)."""
        if self._current_joint < 0 or self._current_person < 0:
            return

        pid = self._current_person
        frame = self._current_frame
        joint = self._current_joint

        # Snapshot for undo
        old_corr_snap = self._snapshot_correction(pid, frame)
        old_raw = self._snapshot_raw_param(joint, frame)

        ct = self._session.correction_tracks.get(pid)
        if ct is not None:
            corr = ct.get_correction(frame)
            if corr is not None:
                if self._current_joint == 0:
                    corr.global_orient = None
                elif corr.body_pose is not None:
                    bp_idx = self._current_joint - 1
                    corr.body_pose.pop(bp_idx, None)
                    if not corr.body_pose:
                        corr.body_pose = None

                # If correction is now empty, remove it entirely
                if (corr.global_orient is None
                        and corr.body_pose is None
                        and corr.transl is None):
                    ct.remove_correction(frame)

                self._save_correction_track(pid)

        new_corr_snap = self._snapshot_correction(pid, frame)
        new_raw = self._snapshot_raw_param(joint, frame)

        def undo(p=pid, f=frame, j=joint, oc=old_corr_snap, oraw=old_raw):
            self._restore_correction(p, f, oc)
            if oraw is not None:
                self._current_person = p
                self._apply_to_raw_params(j, oraw, f)
            self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()
            self._refresh_corrections_table()

        def redo(p=pid, f=frame, j=joint, nc=new_corr_snap, nraw=new_raw):
            self._restore_correction(p, f, nc)
            if nraw is not None:
                self._current_person = p
                self._apply_to_raw_params(j, nraw, f)
            self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()
            self._refresh_corrections_table()

        self._session.undo_stack.push(UndoEntry("Reset joint", undo, redo))

        # Clear preview and invalidate cache
        self._viewport.set_pose_override(None)
        self._viewport.invalidate_cache(pid, frame)
        self._viewport._refresh_mesh()
        self._update_sliders()
        self._refresh_corrections_table()

    def _on_reset_all(self):
        """Remove all corrections at current frame."""
        if self._current_person < 0:
            return

        pid = self._current_person
        frame = self._current_frame

        old_corr_snap = self._snapshot_correction(pid, frame)

        ct = self._session.correction_tracks.get(pid)
        if ct is not None:
            ct.remove_correction(frame)
            self._save_correction_track(pid)

        def undo(p=pid, f=frame, oc=old_corr_snap):
            self._restore_correction(p, f, oc)
            self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()
            self._refresh_corrections_table()

        def redo(p=pid, f=frame):
            self._restore_correction(p, f, None)
            self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()
            self._refresh_corrections_table()

        self._session.undo_stack.push(UndoEntry("Reset all corrections", undo, redo))

        # Clear preview and invalidate cache
        self._viewport.set_pose_override(None)
        self._viewport.invalidate_cache(pid, frame)
        self._viewport._refresh_mesh()
        self._update_sliders()
        self._refresh_corrections_table()

    # ------------------------------------------------------------------
    # Quick-fix handlers
    # ------------------------------------------------------------------

    def _on_flip_body(self):
        """Flip body 180° on user-selected axis."""
        if self._current_person < 0:
            return

        track = self._session.person_tracks.get(self._current_person)
        if track is None or track.smplx_params is None:
            return

        items = ["Yaw (Y-axis)", "Pitch (X-axis)", "Roll (Z-axis)"]
        item, ok = QInputDialog.getItem(
            self, "Flip Body", "Rotation axis:", items, 0, False,
        )
        if not ok:
            return

        axis_map = {
            "Yaw (Y-axis)": "yaw",
            "Pitch (X-axis)": "pitch",
            "Roll (Z-axis)": "roll",
        }
        axis = axis_map[item]

        flip_fn, _, _ = _safe_import_quick_fix()
        if flip_fn is not None:
            new_go = flip_fn(track.smplx_params, self._current_frame, axis)
        else:
            new_go = flip_global_orient_fallback(
                track.smplx_params, self._current_frame, axis,
            )

        self._apply_quick_fix(global_orient=new_go, correction_type="flip")

    def _on_invert_upright(self):
        """Flip global_orient 180° about pitch (fix upside-down)."""
        if self._current_person < 0:
            return

        track = self._session.person_tracks.get(self._current_person)
        if track is None or track.smplx_params is None:
            return

        flip_fn, _, _ = _safe_import_quick_fix()
        if flip_fn is not None:
            new_go = flip_fn(track.smplx_params, self._current_frame, "pitch")
        else:
            new_go = flip_global_orient_fallback(
                track.smplx_params, self._current_frame, "pitch",
            )

        self._apply_quick_fix(global_orient=new_go, correction_type="invert")

    def _on_mirror_lr(self):
        """Swap left/right body joint pairs."""
        if self._current_person < 0:
            return

        track = self._session.person_tracks.get(self._current_person)
        if track is None or track.smplx_params is None:
            return

        _, mirror_fn, _ = _safe_import_quick_fix()
        if mirror_fn is not None:
            mirrored = mirror_fn(track.smplx_params, self._current_frame)
        else:
            mirrored = mirror_lr_pose_fallback(
                track.smplx_params, self._current_frame,
            )

        if not mirrored:
            return

        self._apply_quick_fix(body_pose=mirrored, correction_type="mirror")

    def _on_copy_from_frame(self):
        """Copy full pose from a source frame."""
        if self._current_person < 0:
            return

        track = self._session.person_tracks.get(self._current_person)
        if track is None or track.smplx_params is None:
            return

        max_frame = max(0, self._session.num_frames - 1)
        src_frame, ok = QInputDialog.getInt(
            self, "Copy From Frame", "Source frame:", 0, 0, max_frame, 1,
        )
        if not ok:
            return

        params = track.smplx_params
        go = np.asarray(params["global_orient"], dtype=np.float32)
        bp = np.asarray(params["body_pose"], dtype=np.float32)
        if bp.ndim == 2 and bp.shape[-1] != 3:
            bp = bp.reshape(bp.shape[0], -1, 3)

        sf = max(0, min(src_frame, go.shape[0] - 1))
        new_go = go[sf].copy()
        sparse_bp = None
        if bp.ndim >= 3 and sf < bp.shape[0]:
            bp_frame = bp[sf].copy()
            sparse_bp = {j: bp_frame[j].copy() for j in range(bp_frame.shape[0])}

        self._apply_quick_fix(
            global_orient=new_go,
            body_pose=sparse_bp,
            correction_type="copy_from_frame",
            source_frame=src_frame,
        )

    def _apply_quick_fix(
        self,
        global_orient: np.ndarray | None = None,
        body_pose: dict[int, np.ndarray] | None = None,
        correction_type: str = "quick_fix",
        source_frame: int | None = None,
    ):
        """Apply a quick-fix correction (global_orient and/or body_pose)."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            log.warning("Cannot apply correction: backend unavailable")
            return

        pid = self._current_person
        frame = self._current_frame

        # Snapshot for undo: correction entry and all affected raw params
        old_corr_snap = self._snapshot_correction(pid, frame)
        old_raw_go = self._snapshot_raw_param(0, frame) if global_orient is not None else None
        old_raw_bp = {}
        if body_pose:
            for bp_idx in body_pose:
                old_raw_bp[bp_idx] = self._snapshot_raw_param(bp_idx + 1, frame)

        if pid not in self._session.correction_tracks:
            self._session.correction_tracks[pid] = CorrectionTrack(person_id=pid)

        ct = self._session.correction_tracks[pid]

        # Merge with existing correction at this frame
        existing = ct.get_correction(frame)
        if existing is not None:
            if global_orient is None:
                global_orient = existing.global_orient
            if body_pose is None:
                body_pose = existing.body_pose
            elif existing.body_pose is not None:
                merged = dict(existing.body_pose)
                merged.update(body_pose)
                body_pose = merged

        ct.add_correction(
            frame_index=frame,
            correction_type=correction_type,
            global_orient=global_orient,
            body_pose=body_pose,
            source_frame=source_frame,
        )

        # Apply to raw params
        if global_orient is not None:
            self._apply_to_raw_params(0, global_orient, frame)
        if body_pose:
            for bp_idx, aa in body_pose.items():
                self._apply_to_raw_params(bp_idx + 1, aa, frame)

        # Snapshot post-state for redo
        new_corr_snap = self._snapshot_correction(pid, frame)
        new_raw_go = self._snapshot_raw_param(0, frame) if old_raw_go is not None else None
        new_raw_bp = {}
        if old_raw_bp:
            for bp_idx in old_raw_bp:
                new_raw_bp[bp_idx] = self._snapshot_raw_param(bp_idx + 1, frame)

        def undo(p=pid, f=frame, oc=old_corr_snap, org=old_raw_go, orbp=old_raw_bp):
            self._restore_correction(p, f, oc)
            self._current_person = p
            if org is not None:
                self._apply_to_raw_params(0, org, f)
            for bi, oraw in orbp.items():
                if oraw is not None:
                    self._apply_to_raw_params(bi + 1, oraw, f)
            self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()
            self._refresh_corrections_table()

        def redo(p=pid, f=frame, nc=new_corr_snap, nrg=new_raw_go, nrbp=new_raw_bp):
            self._restore_correction(p, f, nc)
            self._current_person = p
            if nrg is not None:
                self._apply_to_raw_params(0, nrg, f)
            for bi, nraw in nrbp.items():
                if nraw is not None:
                    self._apply_to_raw_params(bi + 1, nraw, f)
            self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()
            self._refresh_corrections_table()

        self._session.undo_stack.push(
            UndoEntry(f"Quick fix ({correction_type})", undo, redo)
        )

        # Refresh viewport and UI
        self._viewport.set_pose_override(None)
        self._viewport.invalidate_cache(pid, frame)
        self._viewport._refresh_mesh()
        self._save_correction_track(pid)
        self._update_sliders()
        self._refresh_corrections_table()

        log.info("Quick fix %s: person=%d, frame=%d", correction_type, pid, frame)
        self.correction_applied.emit(pid, frame)
        self._trigger_auto_preview()

    # ------------------------------------------------------------------
    # Corrections table
    # ------------------------------------------------------------------

    def _refresh_corrections_table(self):
        """Populate corrections table from current person's CorrectionTrack."""
        self._corrections_table.setRowCount(0)

        if self._current_person < 0:
            return

        ct = self._session.correction_tracks.get(self._current_person)
        if ct is None or not ct.corrections:
            return

        corrections = sorted(ct.corrections, key=lambda c: c.frame_index)
        self._corrections_table.setRowCount(len(corrections))

        for row, corr in enumerate(corrections):
            # Frame
            self._corrections_table.setItem(
                row, 0, QTableWidgetItem(str(corr.frame_index)),
            )

            # Type
            self._corrections_table.setItem(
                row, 1, QTableWidgetItem(corr.correction_type),
            )

            # Joint
            joint_str = self._describe_correction_joints(corr)
            self._corrections_table.setItem(
                row, 2, QTableWidgetItem(joint_str),
            )

            # Go button
            go_btn = QPushButton("Go")
            go_btn.setFixedWidth(40)
            go_btn.clicked.connect(
                lambda checked, f=corr.frame_index: self._on_correction_go(f),
            )
            self._corrections_table.setCellWidget(row, 3, go_btn)

            # Delete button
            del_btn = QPushButton("Del")
            del_btn.setFixedWidth(40)
            del_btn.clicked.connect(
                lambda checked, f=corr.frame_index: self._on_correction_delete(f),
            )
            self._corrections_table.setCellWidget(row, 4, del_btn)

    def _describe_correction_joints(self, corr) -> str:
        """Build a readable string describing which joints are corrected."""
        parts: list[str] = []
        if corr.global_orient is not None:
            parts.append("Global")
        if corr.body_pose:
            for bp_idx in sorted(corr.body_pose.keys()):
                joint_idx = bp_idx + 1
                if joint_idx < len(JOINT_NAMES):
                    parts.append(JOINT_NAMES[joint_idx])
                else:
                    parts.append(f"J{joint_idx}")
        if not parts:
            parts.append("—")
        if len(parts) > 3:
            return ", ".join(parts[:3]) + "…"
        return ", ".join(parts)

    def _on_correction_go(self, frame_idx: int):
        """Navigate to correction frame."""
        self.frame_requested.emit(frame_idx)

    def _on_correction_delete(self, frame_idx: int):
        """Delete correction at frame."""
        if self._current_person < 0:
            return

        pid = self._current_person
        ct = self._session.correction_tracks.get(pid)
        if ct is None:
            return

        old_corr_snap = self._snapshot_correction(pid, frame_idx)

        ct.remove_correction(frame_idx)
        self._save_correction_track(pid)

        def undo(p=pid, f=frame_idx, oc=old_corr_snap):
            self._restore_correction(p, f, oc)
            self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()
            self._refresh_corrections_table()

        def redo(p=pid, f=frame_idx):
            self._restore_correction(p, f, None)
            self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()
            self._refresh_corrections_table()

        self._session.undo_stack.push(UndoEntry("Delete correction", undo, redo))

        self._viewport.invalidate_cache(pid, frame_idx)
        self._viewport._refresh_mesh()
        self._update_sliders()
        self._refresh_corrections_table()

    # ------------------------------------------------------------------
    # Space overrides
    # ------------------------------------------------------------------

    def _refresh_space_table(self):
        """Populate space overrides table from current person's CorrectionTrack."""
        self._space_table.setRowCount(0)

        if self._current_person < 0:
            return

        ct = self._session.correction_tracks.get(self._current_person)
        if ct is None or not ct.space_overrides:
            return

        overrides = sorted(ct.space_overrides, key=lambda o: o.frame_start)
        self._space_table.setRowCount(len(overrides))

        for row, ovr in enumerate(overrides):
            self._space_table.setItem(row, 0, QTableWidgetItem(str(ovr.frame_start)))
            self._space_table.setItem(row, 1, QTableWidgetItem(str(ovr.frame_end)))
            self._space_table.setItem(row, 2, QTableWidgetItem(ovr.space))
            ref_str = f"Person {ovr.reference_person}" if ovr.reference_person is not None else "—"
            self._space_table.setItem(row, 3, QTableWidgetItem(ref_str))
            self._space_table.setItem(row, 4, QTableWidgetItem(f"{ovr.y_offset:.2f}"))

    def _on_add_space_override(self):
        """Add a frame-space override for the current person."""
        if self._current_person < 0:
            return

        FrameSpaceOverride = _safe_import_space_override()
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if FrameSpaceOverride is None or CorrectionTrack is None:
            log.warning("Cannot add space override: backend unavailable")
            return

        pid = self._current_person
        space_map = {"World": "world", "Camera": "camera", "Carried": "carried"}
        space = space_map.get(self._space_combo.currentText(), "world")

        f_start = self._space_start.value()
        f_end = self._space_end.value()
        if f_end < f_start:
            f_start, f_end = f_end, f_start

        ref_person = self._space_ref_combo.currentData()
        y_offset = self._space_y_offset.value()

        override = FrameSpaceOverride(
            frame_start=f_start,
            frame_end=f_end,
            space=space,
            reference_person=ref_person,
            y_offset=y_offset,
        )

        if pid not in self._session.correction_tracks:
            self._session.correction_tracks[pid] = CorrectionTrack(person_id=pid)

        ct = self._session.correction_tracks[pid]
        ct.add_space_override(override)
        self._save_correction_track(pid)

        # Capture data for undo (use primitives, not backend objects)
        ovr_data = {
            "frame_start": f_start, "frame_end": f_end,
            "space": space, "reference_person": ref_person, "y_offset": y_offset,
        }

        def undo(p=pid, d=ovr_data):
            c = self._session.correction_tracks.get(p)
            if c is not None:
                c.remove_space_override(d["frame_start"], d["frame_end"])
            self._refresh_space_table()

        def redo(p=pid, d=ovr_data):
            FSO = _safe_import_space_override()
            _, _, CT = _safe_import_pose_correction()
            if FSO is None or CT is None:
                return
            if p not in self._session.correction_tracks:
                self._session.correction_tracks[p] = CT(person_id=p)
            c = self._session.correction_tracks[p]
            c.add_space_override(FSO(
                frame_start=d["frame_start"], frame_end=d["frame_end"],
                space=d["space"], reference_person=d["reference_person"],
                y_offset=d["y_offset"],
            ))
            self._refresh_space_table()

        self._session.undo_stack.push(UndoEntry("Add space override", undo, redo))
        self._refresh_space_table()

        log.info("Space override added: person=%d, frames=%d-%d, space=%s",
                 pid, f_start, f_end, space)

    def _on_delete_space_override(self):
        """Delete the selected space override."""
        if self._current_person < 0:
            return

        row = self._space_table.currentRow()
        if row < 0:
            return

        ct = self._session.correction_tracks.get(self._current_person)
        if ct is None:
            return

        overrides = sorted(ct.space_overrides, key=lambda o: o.frame_start)
        if row >= len(overrides):
            return

        ovr = overrides[row]
        pid = self._current_person
        ovr_data = {
            "frame_start": ovr.frame_start, "frame_end": ovr.frame_end,
            "space": ovr.space,
            "reference_person": getattr(ovr, "reference_person", None),
            "y_offset": getattr(ovr, "y_offset", 0.4),
        }

        ct.remove_space_override(ovr.frame_start, ovr.frame_end)
        self._save_correction_track(self._current_person)

        def undo(p=pid, d=ovr_data):
            FSO = _safe_import_space_override()
            _, _, CT = _safe_import_pose_correction()
            if FSO is None or CT is None:
                return
            if p not in self._session.correction_tracks:
                self._session.correction_tracks[p] = CT(person_id=p)
            c = self._session.correction_tracks[p]
            c.add_space_override(FSO(
                frame_start=d["frame_start"], frame_end=d["frame_end"],
                space=d["space"], reference_person=d["reference_person"],
                y_offset=d["y_offset"],
            ))
            self._refresh_space_table()

        def redo(p=pid, d=ovr_data):
            c = self._session.correction_tracks.get(p)
            if c is not None:
                c.remove_space_override(d["frame_start"], d["frame_end"])
            self._refresh_space_table()

        self._session.undo_stack.push(UndoEntry("Delete space override", undo, redo))
        self._refresh_space_table()

        log.info("Space override deleted: person=%d, frames=%d-%d",
                 self._current_person, ovr.frame_start, ovr.frame_end)

    def _update_space_ref_dropdown(self):
        """Populate reference person dropdown for space overrides."""
        self._space_ref_combo.blockSignals(True)
        self._space_ref_combo.clear()
        self._space_ref_combo.addItem("—", userData=None)
        if self._session:
            for pid in sorted(self._session.person_tracks.keys()):
                if pid not in self._session.inactive_tracks:
                    self._space_ref_combo.addItem(f"Person {pid}", userData=pid)
        self._space_ref_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Smoothing / Apply-to-Similar / Propagation (Phase 5 UX)
    # ------------------------------------------------------------------

    def _on_smooth(self):
        """Apply temporal smoothing to selected joint(s) over the frame range.

        Why: GVHMR body pose estimates often exhibit high-frequency jitter,
        especially in occluded joints. Temporal smoothing reduces this noise
        while preserving deliberate motion. Uses the Frame Range start/end
        and Smoothing group controls (window, method, scope). Operates
        directly on raw body_pose params with undo support.
        """
        if self._current_person < 0:
            return

        track = self._session.person_tracks.get(self._current_person)
        if track is None or track.smplx_params is None:
            return

        pid = self._current_person
        params = track.smplx_params
        bp = np.asarray(params.get("body_pose", []), dtype=np.float32)
        if bp.ndim == 2 and bp.shape[-1] != 3:
            bp = bp.reshape(bp.shape[0], -1, 3)
        if bp.ndim != 3:
            return

        f_start = self._range_start.value()
        f_end = self._range_end.value()
        if f_end < f_start:
            f_start, f_end = f_end, f_start
        f_start = max(0, f_start)
        f_end = min(f_end, bp.shape[0] - 1)
        if f_start > f_end:
            return

        window = self._smooth_window.value()
        method = "gaussian" if self._smooth_method.currentIndex() == 0 else "moving_average"

        scope = self._smooth_scope.currentIndex()
        if scope == 0:  # Current Joint
            if self._current_joint < 1 or self._current_joint > 21:
                log.info("Smoothing only supports body joints (1-21)")
                return
            joint_indices = [self._current_joint - 1]
        else:  # All Body Joints
            joint_indices = list(range(min(21, bp.shape[1])))

        # Snapshot for undo
        old_bp_slice = bp[f_start : f_end + 1].copy()

        # Compute smoothed pose
        smoothed = smooth_joint_rotations(
            bp, f_start, f_end,
            joint_indices=joint_indices,
            window=window,
            method=method,
        )

        # Apply smoothed values to raw params
        bp[f_start : f_end + 1] = smoothed[f_start : f_end + 1]
        params["body_pose"] = bp

        # Invalidate viewport cache
        for f in range(f_start, f_end + 1):
            self._viewport.invalidate_cache(pid, f)

        new_bp_slice = bp[f_start : f_end + 1].copy()

        def undo(p=pid, fs=f_start, fe=f_end, old=old_bp_slice):
            t = self._session.person_tracks.get(p)
            if t and t.smplx_params:
                bpp = np.asarray(t.smplx_params["body_pose"], dtype=np.float32)
                if bpp.ndim == 2 and bpp.shape[-1] != 3:
                    bpp = bpp.reshape(bpp.shape[0], -1, 3)
                bpp[fs : fe + 1] = old
                t.smplx_params["body_pose"] = bpp
                for f in range(fs, fe + 1):
                    self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()

        def redo(p=pid, fs=f_start, fe=f_end, new=new_bp_slice):
            t = self._session.person_tracks.get(p)
            if t and t.smplx_params:
                bpp = np.asarray(t.smplx_params["body_pose"], dtype=np.float32)
                if bpp.ndim == 2 and bpp.shape[-1] != 3:
                    bpp = bpp.reshape(bpp.shape[0], -1, 3)
                bpp[fs : fe + 1] = new
                t.smplx_params["body_pose"] = bpp
                for f in range(fs, fe + 1):
                    self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()

        n_frames = f_end - f_start + 1
        n_joints = len(joint_indices)
        self._session.undo_stack.push(
            UndoEntry(f"Smooth {n_joints} joint(s), {n_frames} frames", undo, redo),
        )

        self._viewport.set_pose_override(None)
        self._viewport._refresh_mesh()
        self._update_sliders()

        log.info(
            "Smoothing applied: person=%d, frames=%d-%d, %d joints, %s, window=%d",
            pid, f_start, f_end, n_joints, method, window,
        )
        self.correction_applied.emit(pid, self._current_frame)

    def _on_apply_to_similar(self):
        """Find frames with similar joint rotation and apply the same correction.

        Why: GVHMR produces the same bad pose estimate repeatedly (e.g.,
        a flipped arm in every other frame). Finding similar poses lets
        the user apply one correction to all matching frames at once,
        saving tedious frame-by-frame editing. Uses the current euler
        slider values as the target correction and the threshold spinbox
        to control match sensitivity.
        """
        if self._current_person < 0 or self._current_joint < 1 or self._current_joint > 21:
            self._sim_status.setText("Select a body joint first (1-21).")
            return

        track = self._session.person_tracks.get(self._current_person)
        if track is None or track.smplx_params is None:
            return

        _, euler_to_aa, _ = _safe_import_pose_correction()
        if euler_to_aa is None:
            euler_to_aa = euler_deg_to_axis_angle_fallback

        pid = self._current_person
        params = track.smplx_params
        bp = np.asarray(params.get("body_pose", []), dtype=np.float32)
        if bp.ndim == 2 and bp.shape[-1] != 3:
            bp = bp.reshape(bp.shape[0], -1, 3)
        if bp.ndim != 3:
            return

        joint_idx = self._current_joint
        bp_idx = joint_idx - 1
        frame_idx = self._current_frame
        threshold = self._sim_threshold.value()

        # Find similar frames
        matches = find_similar_frames(bp, frame_idx, bp_idx, threshold)
        if not matches:
            self._sim_status.setText("No similar frames found.")
            return

        # Get the correction to apply (current euler slider values)
        euler = np.array([
            self._euler_x.value(),
            self._euler_y.value(),
            self._euler_z.value(),
        ], dtype=np.float32)
        new_aa = euler_to_aa(euler)

        # Snapshot for undo
        old_values = {}
        for f in matches:
            if f < bp.shape[0] and bp_idx < bp.shape[1]:
                old_values[f] = bp[f, bp_idx].copy()

        # Apply correction to all matching frames
        for f in matches:
            if f < bp.shape[0] and bp_idx < bp.shape[1]:
                bp[f, bp_idx] = new_aa
                self._viewport.invalidate_cache(pid, f)
        params["body_pose"] = bp

        saved_aa = new_aa.copy()

        def undo(p=pid, ji=bp_idx, old=old_values):
            t = self._session.person_tracks.get(p)
            if t and t.smplx_params:
                bpp = np.asarray(t.smplx_params["body_pose"], dtype=np.float32)
                if bpp.ndim == 2 and bpp.shape[-1] != 3:
                    bpp = bpp.reshape(bpp.shape[0], -1, 3)
                for f, old_val in old.items():
                    if f < bpp.shape[0] and ji < bpp.shape[1]:
                        bpp[f, ji] = old_val
                        self._viewport.invalidate_cache(p, f)
                t.smplx_params["body_pose"] = bpp
            self._viewport._refresh_mesh()
            self._update_sliders()

        def redo(p=pid, ji=bp_idx, frames=matches, aa=saved_aa):
            t = self._session.person_tracks.get(p)
            if t and t.smplx_params:
                bpp = np.asarray(t.smplx_params["body_pose"], dtype=np.float32)
                if bpp.ndim == 2 and bpp.shape[-1] != 3:
                    bpp = bpp.reshape(bpp.shape[0], -1, 3)
                for f in frames:
                    if f < bpp.shape[0] and ji < bpp.shape[1]:
                        bpp[f, ji] = aa
                        self._viewport.invalidate_cache(p, f)
                t.smplx_params["body_pose"] = bpp
            self._viewport._refresh_mesh()
            self._update_sliders()

        n = len(matches)
        self._session.undo_stack.push(
            UndoEntry(f"Apply to {n} similar frames", undo, redo),
        )

        self._viewport.set_pose_override(None)
        self._viewport._refresh_mesh()
        self._update_sliders()

        self._sim_status.setText(
            f"Applied to {n} similar frame{'s' if n != 1 else ''} "
            f"(threshold {threshold:.0f}\u00b0)."
        )
        log.info(
            "Apply to similar: person=%d, joint=%d, %d frames (threshold=%.1f\u00b0)",
            pid, joint_idx, n, threshold,
        )
        self.correction_applied.emit(pid, self._current_frame)

    def _on_propagate(self):
        """Interpolate the current joint correction across the frame range using SLERP.

        Why: When the user corrects poses at the start and end of a range,
        the correction should transition smoothly between them. SLERP on the
        rotation produces natural interpolation through rotation space,
        avoiding gimbal-lock artifacts of linear euler interpolation.
        Uses the Frame Range start/end as the two keyframes.
        """
        if self._current_person < 0 or self._current_joint < 0:
            self._prop_status.setText("Select a person and joint first.")
            return

        if self._current_joint > 21:
            self._prop_status.setText("Propagation only supports body joints (0-21).")
            return

        track = self._session.person_tracks.get(self._current_person)
        if track is None or track.smplx_params is None:
            return

        pid = self._current_person
        joint_idx = self._current_joint

        f_start = self._range_start.value()
        f_end = self._range_end.value()
        if f_end < f_start:
            f_start, f_end = f_end, f_start
        if f_end == f_start:
            self._prop_status.setText("Start and end frames must differ.")
            return

        # Get axis-angle at start and end frames
        start_aa = _get_joint_axis_angle(self._session, pid, f_start, joint_idx)
        end_aa = _get_joint_axis_angle(self._session, pid, f_end, joint_idx)

        # Compute SLERP interpolation
        interp = propagate_corrections(f_start, f_end, start_aa, end_aa)
        if not interp:
            return

        params = track.smplx_params

        # Snapshot for undo
        old_values = {}
        if joint_idx == 0:
            go = np.asarray(params.get("global_orient", []), dtype=np.float32)
            for f in interp:
                if go.ndim >= 2 and f < go.shape[0]:
                    old_values[f] = go[f].copy()
        else:
            bp = np.asarray(params.get("body_pose", []), dtype=np.float32)
            if bp.ndim == 2 and bp.shape[-1] != 3:
                bp = bp.reshape(bp.shape[0], -1, 3)
            bp_idx = joint_idx - 1
            for f in interp:
                if bp.ndim >= 3 and f < bp.shape[0] and bp_idx < bp.shape[1]:
                    old_values[f] = bp[f, bp_idx].copy()

        # Apply interpolated values
        for f, aa in interp.items():
            self._apply_to_raw_params(joint_idx, aa, f)
            self._viewport.invalidate_cache(pid, f)

        new_values = {f: aa.copy() for f, aa in interp.items()}

        def undo(p=pid, ji=joint_idx, old=old_values):
            self._current_person = p
            for f, old_val in old.items():
                self._apply_to_raw_params(ji, old_val, f)
                self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()

        def redo(p=pid, ji=joint_idx, new=new_values):
            self._current_person = p
            for f, new_val in new.items():
                self._apply_to_raw_params(ji, new_val, f)
                self._viewport.invalidate_cache(p, f)
            self._viewport._refresh_mesh()
            self._update_sliders()

        n = len(interp)
        self._session.undo_stack.push(
            UndoEntry(f"Propagate correction ({n} frames)", undo, redo),
        )

        self._viewport.set_pose_override(None)
        self._viewport._refresh_mesh()
        self._update_sliders()

        self._prop_status.setText(
            f"Propagated across {n} frames ({f_start}\u2013{f_end})."
        )
        log.info(
            "Propagation applied: person=%d, joint=%d, frames=%d-%d",
            pid, joint_idx, f_start, f_end,
        )
        self.correction_applied.emit(pid, self._current_frame)

    # ------------------------------------------------------------------
    # Preview range playback
    # ------------------------------------------------------------------

    def _on_preview_play(self):
        """Play ±N frames around the current frame to preview correction impact.

        Why: After applying a correction, artists need to see how it looks
        in temporal context — does the motion flow smoothly into and out of
        the corrected frame? This auto-plays the surrounding frames at ~30fps,
        emitting frame_requested signals to drive the video player and all
        other synced panels.
        """
        half = self._preview_half_range.value()
        center = self._current_frame
        self._preview_frame = max(0, center - half)
        self._preview_end = min(
            max(0, self._session.num_frames - 1), center + half,
        )
        if self._preview_frame >= self._preview_end:
            return
        self._preview_btn.setText("Stop")
        self._preview_timer.start()

    def _on_preview_tick(self):
        """Advance one frame in the preview playback."""
        self.frame_requested.emit(self._preview_frame)
        self._preview_frame += 1
        if self._preview_frame > self._preview_end:
            self._preview_timer.stop()
            self._preview_btn.setText("Preview")

    def _trigger_auto_preview(self):
        """Start preview playback if auto-preview is enabled."""
        if self._auto_preview_check.isChecked():
            self._on_preview_play()

    # ------------------------------------------------------------------
    # BVH/FBX export
    # ------------------------------------------------------------------

    def _on_reexport_bvh(self):
        """Re-export corrected BVH for the current person."""
        if self._current_person < 0:
            return

        track = self._session.person_tracks.get(self._current_person)
        if track is None or track.smplx_params is None:
            self._export_status.setText("No SMPL-X params for this person.")
            return

        convert_fn = _safe_import_bvh_export()
        if convert_fn is None:
            self._export_status.setText("BVH export unavailable (backend missing).")
            return

        pid = self._current_person
        ct = self._session.correction_tracks.get(pid)

        # Save corrections first
        self._save_correction_track(pid)

        # Determine output path
        if track.person_dir is not None:
            out_path = str(Path(track.person_dir) / "corrected_body.bvh")
        else:
            out_path, _ = QFileDialog.getSaveFileName(
                self, "Save BVH", "corrected_body.bvh", "BVH files (*.bvh)",
            )
            if not out_path:
                return

        # Build reference_params for "carried" space overrides
        ref_params = None
        if ct is not None and ct.space_overrides:
            ref_pids = {
                o.reference_person for o in ct.space_overrides
                if o.space == "carried" and o.reference_person is not None
            }
            if ref_pids:
                ref_params = {}
                for rpid in ref_pids:
                    rp = self._session.person_tracks.get(rpid)
                    if rp is not None and rp.smplx_params is not None:
                        ref_params[rpid] = rp.smplx_params

        try:
            self._export_status.setText("Exporting BVH...")
            result = convert_fn(
                track.smplx_params,
                out_path,
                fps=self._session.fps,
                skip_world_grounding=True,
                corrections=ct,
                reference_params=ref_params,
            )
            self._export_status.setText(f"BVH exported: {result}")
            log.info("BVH exported: %s", result)
            self.export_requested.emit("bvh")
        except Exception as e:
            self._export_status.setText(f"BVH export failed: {e}")
            log.error("BVH export failed: %s", e)

    def _on_reexport_fbx(self):
        """Re-export corrected BVH + FBX via Blender subprocess."""
        if self._current_person < 0:
            return

        track = self._session.person_tracks.get(self._current_person)
        if track is None or track.smplx_params is None:
            self._export_status.setText("No SMPL-X params for this person.")
            return

        convert_bvh = _safe_import_bvh_export()
        convert_fbx = _safe_import_fbx_export()
        if convert_bvh is None:
            self._export_status.setText("BVH export unavailable (backend missing).")
            return

        pid = self._current_person
        ct = self._session.correction_tracks.get(pid)

        # Save corrections first
        self._save_correction_track(pid)

        # Determine output paths
        if track.person_dir is not None:
            bvh_path = str(Path(track.person_dir) / "corrected_body.bvh")
        else:
            bvh_path, _ = QFileDialog.getSaveFileName(
                self, "Save BVH", "corrected_body.bvh", "BVH files (*.bvh)",
            )
            if not bvh_path:
                return

        # Build reference_params for "carried" space overrides
        ref_params = None
        if ct is not None and ct.space_overrides:
            ref_pids = {
                o.reference_person for o in ct.space_overrides
                if o.space == "carried" and o.reference_person is not None
            }
            if ref_pids:
                ref_params = {}
                for rpid in ref_pids:
                    rp = self._session.person_tracks.get(rpid)
                    if rp is not None and rp.smplx_params is not None:
                        ref_params[rpid] = rp.smplx_params

        try:
            self._export_status.setText("Exporting BVH...")
            bvh_result = convert_bvh(
                track.smplx_params,
                bvh_path,
                fps=self._session.fps,
                skip_world_grounding=True,
                corrections=ct,
                reference_params=ref_params,
            )
            self._export_status.setText(f"BVH exported: {bvh_result}")
            log.info("BVH exported: %s", bvh_result)
        except Exception as e:
            self._export_status.setText(f"BVH export failed: {e}")
            log.error("BVH export failed: %s", e)
            return

        if convert_fbx is None:
            self._export_status.setText(
                f"BVH exported: {bvh_result}\nFBX conversion unavailable (Blender not found)."
            )
            self.export_requested.emit("bvh")
            return

        try:
            fbx_path = str(Path(bvh_path).with_suffix(".fbx"))
            self._export_status.setText("Converting to FBX via Blender...")
            fbx_log = convert_fbx(bvh_path, fbx_path)
            self._export_status.setText(f"FBX exported: {fbx_path}")
            log.info("FBX exported: %s (%s)", fbx_path, fbx_log)
            self.export_requested.emit("fbx")
        except Exception as e:
            self._export_status.setText(
                f"BVH exported: {bvh_result}\nFBX conversion failed: {e}"
            )
            log.error("FBX export failed: %s", e)
            self.export_requested.emit("bvh")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_person_dropdown(self):
        """Populate person dropdown from session."""
        self._person_combo.blockSignals(True)
        self._person_combo.clear()
        if self._session:
            for pid in sorted(self._session.person_tracks.keys()):
                if pid not in self._session.inactive_tracks:
                    self._person_combo.addItem(f"Person {pid}", userData=pid)
        self._person_combo.blockSignals(False)
        self._update_space_ref_dropdown()

    def _update_sliders(self):
        """Update euler sliders to reflect current joint's rotation."""
        if self._current_joint < 0 or self._current_person < 0:
            return

        euler = get_joint_euler(
            self._session, self._current_person,
            self._current_frame, self._current_joint,
        )

        self._updating_sliders = True
        self._euler_x.setValue(float(euler[0]))
        self._euler_y.setValue(float(euler[1]))
        self._euler_z.setValue(float(euler[2]))
        self._slider_x.setValue(int(round(euler[0])))
        self._slider_y.setValue(int(round(euler[1])))
        self._slider_z.setValue(int(round(euler[2])))
        self._updating_sliders = False

    def _update_joint_info(self):
        """Update joint info label with name and parent."""
        if self._current_joint < 0:
            self._joint_info.setText("")
            return
        if self._current_joint >= len(JOINT_NAMES):
            self._joint_info.setText("Unknown joint")
            return
        name = JOINT_NAMES[self._current_joint]
        parent_idx = JOINT_PARENTS[self._current_joint]
        parent_name = JOINT_NAMES[parent_idx] if 0 <= parent_idx < len(JOINT_NAMES) else "root"
        self._joint_info.setText(f"{name} (parent: {parent_name})")

    def _apply_to_raw_params(self, joint_idx: int, aa: np.ndarray, frame_idx: int):
        """Apply axis-angle correction directly to session params.

        This updates the raw smplx_params so the mesh renders correctly
        without the temporary pose override.
        """
        track = self._session.person_tracks.get(self._current_person)
        if not track or not track.smplx_params:
            return

        params = track.smplx_params

        if joint_idx == 0:
            go = np.array(params.get("global_orient", []), dtype=np.float32)
            if go.ndim >= 2 and frame_idx < go.shape[0]:
                go[frame_idx] = aa.astype(np.float32)
                params["global_orient"] = go
        elif 1 <= joint_idx <= 21:
            bp = np.array(params.get("body_pose", []), dtype=np.float32)
            if bp.ndim == 2 and bp.shape[-1] != 3:
                bp = bp.reshape(bp.shape[0], -1, 3)
            bp_idx = joint_idx - 1
            if bp.ndim >= 3 and frame_idx < bp.shape[0] and bp_idx < bp.shape[1]:
                bp[frame_idx, bp_idx] = aa.astype(np.float32)
                params["body_pose"] = bp

        # Invalidate vertex cache for this frame
        self._viewport.invalidate_cache(self._current_person, frame_idx)

    def _save_correction_track(self, person_id: int):
        """Persist correction track to JSON file in person directory."""
        track = self._session.person_tracks.get(person_id)
        ct = self._session.correction_tracks.get(person_id)
        if track is None or ct is None:
            return

        if track.person_dir is not None:
            path = Path(track.person_dir) / "correction_track.json"
            try:
                ct.save_json(path)
                log.info("Saved correction track: %s", path)
            except Exception as e:
                log.warning("Failed to save correction track: %s", e)

    # ------------------------------------------------------------------
    # Auto-Detect: Bad span detection
    # ------------------------------------------------------------------

    def _on_detect_bad_spans(self):
        """Run automatic pose issue detection across active tracks.

        Why: Users need to quickly identify frames that need correction
        without scrubbing through the entire video. This scans for angular
        jumps (bad HMR estimates), jitter (high-frequency noise), and low
        confidence spans, presenting results in a navigable table.
        """
        self._pose_issues = compute_pose_issues(self._session)
        self._pose_issue_idx = 0

        n = len(self._pose_issues)
        self._update_pose_issues_table()

        if n == 0:
            self._issues_label.setText("No issues found.")
            self._issues_label.setStyleSheet(f"font-style: italic; color: {COLORS['success']};")
        else:
            self._issues_label.setText(
                f"Found {n} issue{'s' if n != 1 else ''}. "
                f"Use Next/Prev to navigate."
            )
            self._issues_label.setStyleSheet(f"font-style: italic; color: {COLORS['warning']};")
            self._navigate_to_pose_issue(0)

        has_issues = n > 0
        self._prev_issue_btn.setEnabled(has_issues)
        self._next_issue_btn.setEnabled(has_issues)

        log.info("Pose scan complete: %d issues found", n)

    def _on_next_pose_issue(self):
        """Navigate to the next pose issue (wraps around)."""
        if not self._pose_issues:
            return
        self._pose_issue_idx = (self._pose_issue_idx + 1) % len(self._pose_issues)
        self._navigate_to_pose_issue(self._pose_issue_idx)

    def _on_prev_pose_issue(self):
        """Navigate to the previous pose issue (wraps around)."""
        if not self._pose_issues:
            return
        self._pose_issue_idx = (self._pose_issue_idx - 1) % len(self._pose_issues)
        self._navigate_to_pose_issue(self._pose_issue_idx)

    def _navigate_to_pose_issue(self, idx: int):
        """Seek to an issue's frame and update the status label."""
        issue = self._pose_issues[idx]
        n = len(self._pose_issues)
        self._issues_label.setText(
            f"[{idx + 1}/{n}] P{issue.person_id}: {issue.description}"
        )
        # Select the issue's person if different
        if issue.person_id != self._current_person:
            self.set_person(issue.person_id)
        # Seek to the issue's frame
        self.frame_requested.emit(issue.frame)
        # Highlight row in table
        self._issues_table.selectRow(idx)

    def _on_pose_issue_double_clicked(self, row: int, _col: int):
        """Navigate to an issue when double-clicking its table row."""
        if 0 <= row < len(self._pose_issues):
            self._pose_issue_idx = row
            self._navigate_to_pose_issue(row)

    def _update_pose_issues_table(self):
        """Rebuild the issues table from current scan results."""
        self._issues_table.setRowCount(0)
        self._issues_table.setRowCount(len(self._pose_issues))

        for row, issue in enumerate(self._pose_issues):
            frame_item = QTableWidgetItem(str(issue.frame))
            frame_item.setTextAlignment(Qt.AlignCenter)
            self._issues_table.setItem(row, 0, frame_item)

            person_item = QTableWidgetItem(f"Person {issue.person_id}")
            person_item.setTextAlignment(Qt.AlignCenter)
            self._issues_table.setItem(row, 1, person_item)

            type_item = QTableWidgetItem(issue.issue_type.replace("_", " "))
            type_item.setTextAlignment(Qt.AlignCenter)
            self._issues_table.setItem(row, 2, type_item)

            desc_item = QTableWidgetItem(issue.description)
            self._issues_table.setItem(row, 3, desc_item)
