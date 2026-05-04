"""Main application window with dock-based layout, menu bar, status bar, and log panel.

Why dockable layout: Replaces the fixed 3-tab QTabWidget with QDockWidgets that
users can rearrange, tabify, float, and close/reopen via the View menu.  A mode
selector in PipelineSettingsDock switches the settings panel and shows/hides
multi-person-only docks (Identity Inspector, Pose Corrector, Track Overview).
QMainWindow.saveState()/restoreState() persists the layout across sessions.
"""

import enum
import logging
from pathlib import Path

import cv2
import numpy as np


# Ankle/foot body_pose indices (0-based: SMPL-X joint index - 1).
_ANKLE_FOOT_JOINTS = {6, 7, 9, 10}  # L_Ankle, R_Ankle, L_Foot, R_Foot

# Default gain for ankle/foot rotation amplification.  GVHMR's decoder
# under-predicts ankle articulation (~30-50% of true ROM for dance footage).
_DEFAULT_ANKLE_GAIN = 1.5


def _amplify_ankle_rotations(
    params: dict,
    gain: float = _DEFAULT_ANKLE_GAIN,
    joint_indices: set[int] | None = None,
) -> None:
    """Scale ankle/foot axis-angle magnitudes in-place.

    Preserves rotation axis — only the magnitude (angle) is scaled.
    """
    if gain == 1.0:
        return

    if joint_indices is None:
        joint_indices = _ANKLE_FOOT_JOINTS

    bp = params.get("body_pose")
    if bp is None:
        return

    # Ensure (N, 21, 3) shape
    needs_reshape = bp.ndim == 2 and bp.shape[-1] != 3
    if needs_reshape:
        bp = bp.reshape(bp.shape[0], -1, 3)

    for j in joint_indices:
        joint_aa = bp[:, j, :]  # (N, 3)
        norms = np.linalg.norm(joint_aa, axis=-1, keepdims=True)
        safe = np.where(norms > 1e-8, norms, np.ones_like(norms))
        bp[:, j, :] = (joint_aa / safe) * (norms * gain)

    if needs_reshape:
        bp = bp.reshape(bp.shape[0], -1)
    params["body_pose"] = bp


def _compute_camera_c2w(go_incam, tr_incam, go_world, tr_world):
    """Derive per-frame camera-to-world (4x4) from body pose pairs."""
    from scipy.spatial.transform import Rotation
    N = go_world.shape[0]
    c2w = np.zeros((N, 4, 4), dtype=np.float32)
    R_cams = Rotation.from_rotvec(go_incam).as_matrix()    # (N,3,3)
    R_worlds = Rotation.from_rotvec(go_world).as_matrix()   # (N,3,3)
    R_c2w = R_worlds @ np.swapaxes(R_cams, -1, -2)         # (N,3,3)
    t_cam = tr_world - np.einsum('nij,nj->ni', R_c2w, tr_incam)
    c2w[:, :3, :3] = R_c2w
    c2w[:, :3, 3] = t_cam
    c2w[:, 3, 3] = 1.0
    return c2w


def _crop_transl_to_fullframe(tr_crop, K_crop, K_full, crop_x1, crop_y1):
    """Convert SMPL translations from crop camera space to full-frame camera space.

    Same math as mesh_viewport._incam_transform_points but operates on (N,3)
    numpy arrays (pure numpy, no OpenGL context needed).
    """
    X, Y, Z = tr_crop[:, 0], tr_crop[:, 1], tr_crop[:, 2]
    X_f = (K_crop[0, 0] * X + (K_crop[0, 2] + crop_x1 - K_full[0, 2]) * Z) / K_full[0, 0]
    Y_f = (K_crop[1, 1] * Y + (K_crop[1, 2] + crop_y1 - K_full[1, 2]) * Z) / K_full[1, 1]
    return np.stack([X_f, Y_f, Z], axis=-1).astype(np.float32)


def _normalize_slam_w2c(slam) -> np.ndarray:
    """Return SLAM poses as W2C matrices regardless of on-disk format.

    SimpleVO stores homogeneous ``T_w2c`` matrices directly. DPVO stores
    per-frame rows as ``[tx, ty, tz, qx, qy, qz, qw]`` in C2W form after an
    internal inverse, so those rows must be converted back to W2C before the
    rest of the viewport pipeline consumes them.
    """
    arr = np.array(slam, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == 7:
        from scipy.spatial.transform import Rotation as _R

        quats_xyzw = arr[:, 3:7]
        R_c2w = _R.from_quat(quats_xyzw).as_matrix()
        t_c2w = arr[:, :3].astype(np.float32)
        R_w2c = R_c2w.transpose(0, 2, 1)
        t_w2c = -np.einsum("nij,nj->ni", R_w2c, t_c2w)
        T = np.tile(np.eye(4, dtype=np.float32), (len(arr), 1, 1))
        T[:, :3, :3] = R_w2c
        T[:, :3, 3] = t_w2c
        arr = T
    return arr


class _LowPassFilter:
    def __init__(self):
        self.s = None

    def __call__(self, value, alpha):
        if self.s is None:
            self.s = value
        else:
            self.s = alpha * value + (1.0 - alpha) * self.s
        return self.s


class _OneEuroFilter:
    """Adaptive low-pass filter: smooths heavily on slow motion, backs off on fast motion."""

    def __init__(self, freq: float, min_cutoff: float = 0.5, beta: float = 0.007, d_cutoff: float = 1.0):
        import math as _math
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._math = _math
        self.x_filt = _LowPassFilter()
        self.dx_filt = _LowPassFilter()

    def _alpha(self, cutoff, freq):
        tau = 1.0 / (2.0 * self._math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        prev = self.x_filt.s
        dx = 0.0 if prev is None else (x - prev) * self.freq
        edx = self.dx_filt(dx, self._alpha(self.d_cutoff, self.freq))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filt(x, self._alpha(cutoff, self.freq))


_CAM_SMOOTH_PRESETS = {
    "light":    {"min_cutoff": 0.8,  "beta": 0.05,  "med_kernel": 5},
    "moderate": {"min_cutoff": 0.15, "beta": 0.01,  "med_kernel": 7},
    "heavy":    {"min_cutoff": 0.04, "beta": 0.005, "med_kernel": 9},
}


def _smooth_c2w(c2w, fps=30.0, preset="moderate"):
    """Temporal smoothing of camera-to-world matrices.

    Median pre-pass removes outlier spikes, then One Euro filter per channel
    gives adaptive smoothing (heavy on slow motion, responsive on fast).
    """
    from scipy.ndimage import median_filter
    from scipy.spatial.transform import Rotation

    N = c2w.shape[0]
    if N < 3:
        return c2w

    p = _CAM_SMOOTH_PRESETS.get(preset, _CAM_SMOOTH_PRESETS["moderate"])
    med_k = min(p["med_kernel"], N | 1)  # must be odd, <= N
    out = c2w.copy()

    # --- Smooth translation: median + One Euro ---
    for axis in range(3):
        t = out[:, axis, 3].copy()
        t = median_filter(t, size=med_k)
        filt = _OneEuroFilter(fps, min_cutoff=p["min_cutoff"], beta=p["beta"])
        for i in range(N):
            t[i] = filt(float(t[i]))
        out[:, axis, 3] = t

    # --- Smooth rotation via quaternions: median + One Euro ---
    R_mats = out[:, :3, :3].copy()
    quats = Rotation.from_matrix(R_mats).as_quat()  # (N, 4) xyzw

    # Sign consistency: flip quaternion if dot with previous is negative
    for i in range(1, N):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    for c in range(4):
        q = quats[:, c].copy()
        q = median_filter(q, size=med_k)
        filt = _OneEuroFilter(fps, min_cutoff=p["min_cutoff"], beta=p["beta"])
        for i in range(N):
            q[i] = filt(float(q[i]))
        quats[:, c] = q

    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = quats / np.where(norms > 1e-8, norms, np.ones_like(norms))
    out[:, :3, :3] = Rotation.from_quat(quats).as_matrix()
    return out


def _umeyama_align(src, dst):
    """Umeyama similarity alignment: find s, R, t such that dst ≈ s*R@src + t.

    Args:
        src: (N, 3) source points (SLAM camera positions)
        dst: (N, 3) destination points (body-derived camera positions)
    Returns:
        s: float scale factor
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = np.sum(src_c ** 2) / n
    if var_s < 1e-12:
        return 1.0, np.eye(3, dtype=np.float32), (mu_d - mu_s).astype(np.float32)
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    return s, R.astype(np.float32), t.astype(np.float32)


def _align_slam_to_world(slam_w2c, body_c2w, n_refs=10):
    """Align SLAM W2C trajectory to GVHMR world frame via similarity transform.

    Monocular VO has unknown scale and its world frame is its first camera
    frame (OpenCV convention, not gravity-aligned). We want the similarity
    transform (R, s, t) from SLAM world to body/GV world such that SLAM's
    clean per-frame camera poses land in the Y-up world frame at the correct
    metric scale.

    Critical: position-only Umeyama alignment (which is what an earlier
    version used) fits trajectory *point clouds* — the resulting R has no
    intrinsic reason to match the rotation needed to align camera reference
    frames, and with noisy body-derived positions it drifts badly. Instead we
    derive:

    * **R_align** from rotations: per-frame ``R_body[i] @ R_slam[i]^T`` is,
      in theory, the frame-independent SLAM→body rotation. Average in
      quaternion space (sign-aligned) for noise robustness.
    * **scale** from trajectory path-length ratio (always positive, metric).
      Body and SLAM should measure the same physical path length at
      different scales; body-derived per-frame positions are noisy but the
      total path length is a robust, always-positive scale estimator.
    * **t_align** by anchoring at frame 0 — because SimpleVO starts at
      identity (camera at origin, R=I), frame 0 of the aligned trajectory
      should match body-derived frame 0 exactly. Downstream rendering then
      inherits the body-derived anchor and gets clean SLAM motion on top.
    """
    from scipy.spatial.transform import Rotation

    N = slam_w2c.shape[0]
    slam_c2w_raw = np.linalg.inv(slam_w2c)  # (N, 4, 4) in SLAM's world frame
    R_slam = slam_c2w_raw[:, :3, :3]        # (N, 3, 3)
    t_slam = slam_c2w_raw[:, :3, 3]         # (N, 3)
    R_body = body_c2w[:, :3, :3]
    t_body = body_c2w[:, :3, 3]

    # --- R_align from rotation alignment (quaternion mean) ---
    R_rel = R_body @ R_slam.transpose(0, 2, 1)        # per-frame SLAM→body rotation
    q = Rotation.from_matrix(R_rel).as_quat()         # (N, 4) xyzw
    # Sign-align to the first quaternion so the mean is meaningful
    flip = np.einsum('ij,j->i', q, q[0]) < 0
    q[flip] = -q[flip]
    q_mean = q.mean(axis=0)
    q_mean = q_mean / max(np.linalg.norm(q_mean), 1e-12)
    R_align = Rotation.from_quat(q_mean).as_matrix().astype(np.float32)

    # --- Scale from trajectory path-length ratio ---
    body_path = float(np.linalg.norm(np.diff(t_body, axis=0), axis=1).sum())
    slam_path = float(np.linalg.norm(np.diff(t_slam, axis=0), axis=1).sum())
    s = body_path / max(slam_path, 1e-8)

    # --- Anchor translation at frame 0 ---
    t_align = t_body[0] - s * (R_align @ t_slam[0])

    logging.getLogger(__name__).info(
        "SLAM alignment: scale=%.4f, body_path=%.2fm, slam_path=%.2f",
        s, body_path, slam_path,
    )

    aligned = np.zeros((N, 4, 4), dtype=np.float32)
    for i in range(N):
        aligned[i, :3, :3] = R_align @ R_slam[i]
        aligned[i, :3, 3]  = s * (R_align @ t_slam[i]) + t_align
        aligned[i, 3, 3]   = 1.0

    return aligned


def _convert_wrist_orient_to_local(
    params: dict,
    camera_global_orient: np.ndarray | None = None,
) -> None:
    """Convert global-frame wrist_orient to local (relative-to-elbow).

    HaMeR's wrist_orient is MANO global_orient (absolute camera-space).
    SMPL-X body_pose expects local rotations relative to parent joint.
    Walk FK chain root->elbow per frame to get the parent's accumulated
    rotation, then factor it out.

    Parameters
    ----------
    params : dict
        Must contain ``body_pose`` and either ``global_orient`` or the
        caller must supply *camera_global_orient*.  ``left_wrist_orient``
        and/or ``right_wrist_orient`` are converted in-place.
    camera_global_orient : array, optional
        Camera-space global_orient to use for the FK chain root.  Required
        when *params* holds world-space ``global_orient`` (which would be
        the wrong frame for camera-space wrist_orient).
    """
    from scipy.spatial.transform import Rotation

    go = camera_global_orient if camera_global_orient is not None else params.get("global_orient")
    bp = params.get("body_pose")
    if go is None or bp is None:
        return

    go = np.asarray(go, dtype=np.float64)
    bp = np.asarray(bp, dtype=np.float64)
    if bp.ndim == 2 and bp.shape[-1] != 3:
        bp = bp.reshape(bp.shape[0], -1, 3)

    N = go.shape[0]

    # Body-pose indices (0-based) for the chain from pelvis to each elbow.
    # SMPL-X joint_id -> body_pose_index = joint_id - 1
    _L_CHAIN = [2, 5, 8, 12, 15, 17]  # Spine1,Spine2,Spine3,L_Collar,L_Shoulder,L_Elbow
    _R_CHAIN = [2, 5, 8, 13, 16, 18]  # Spine1,Spine2,Spine3,R_Collar,R_Shoulder,R_Elbow

    for chain, key in [
        (_L_CHAIN, "left_wrist_orient"),
        (_R_CHAIN, "right_wrist_orient"),
    ]:
        wo = params.get(key)
        if wo is None:
            continue
        wo = np.asarray(wo, dtype=np.float64)
        if wo.ndim == 1:
            wo = wo[np.newaxis]

        n_use = min(N, wo.shape[0])
        for t in range(n_use):
            # Accumulate rotations: root global_orient, then each chain joint
            R_acc = Rotation.from_rotvec(go[t].ravel()[:3])
            for bp_idx in chain:
                R_acc = R_acc * Rotation.from_rotvec(bp[t, bp_idx])
            # R_acc is now the elbow's accumulated rotation
            R_global_wrist = Rotation.from_rotvec(wo[t])
            R_local = R_acc.inv() * R_global_wrist
            wo[t] = R_local.as_rotvec()

        params[key] = wo.astype(np.float32)


def _smooth_hand_poses(params: dict, fps: float = 30.0) -> None:
    """Apply One Euro filter to hand + wrist poses — same params as BVH export."""
    from smplx_to_bvh import _smooth_rotations_one_euro

    hand_keys = [k for k in ("left_hand_pose", "right_hand_pose") if k in params]
    wrist_keys = [k for k in ("left_wrist_orient", "right_wrist_orient") if k in params]

    if not hand_keys and not wrist_keys:
        return

    # --- Finger joints: min_cutoff=0.3, beta=0.007 ---
    if hand_keys:
        # Reshape flat (N, 45) → (N, 15, 3) so the filter sees proper rotvecs
        orig_shapes = {}
        for k in hand_keys:
            v = params[k]
            orig_shapes[k] = v.shape
            if v.ndim == 2 and v.shape[-1] != 3:
                params[k] = v.reshape(v.shape[0], -1, 3)

        smoothed = _smooth_rotations_one_euro(
            params, keys=hand_keys, fps=fps, min_cutoff=0.3, beta=0.007,
        )
        for k in hand_keys:
            params[k] = smoothed[k].reshape(orig_shapes[k])

    # --- Wrist orient: heavier smoothing (min_cutoff=0.15, beta=0.01) ---
    if wrist_keys:
        smoothed_w = _smooth_rotations_one_euro(
            params, keys=wrist_keys, fps=fps, min_cutoff=0.15, beta=0.01,
        )
        for k in wrist_keys:
            params[k] = smoothed_w[k]

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStatusBar,
    QDockWidget,
    QPlainTextEdit,
    QWidget,
    QLabel,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QMenu,
)
from PySide6.QtCore import Signal, QSettings, Qt, QByteArray
from PySide6.QtGui import QAction, QPalette, QColor


class InteractionMode(enum.Enum):
    """Context-aware interaction modes that change keyboard shortcuts.

    Why modes: A professional motion capture tool needs different keyboard
    bindings depending on the current task.  Navigate mode uses WASD for
    orbit camera control; Select mode makes click pick a joint; Correct mode
    opens euler sliders; Track mode navigates between persons and unreviewed
    keyframes.  The active mode is shown in the status bar and HUD overlay.
    """
    NAVIGATE = "Navigate"
    SELECT = "Select"
    CORRECT = "Correct"
    TRACK = "Track"

from models.pipeline_config import PipelineConfig
from models.session import Session
from views.video_player import VideoPlayer
from views.mesh_viewport import MeshViewport, RenderMode
from views.identity_inspector import IdentityInspector
from views.track_overview import TrackOverview
from views.pipeline_settings import (
    SinglePipelineSettings,
    PerfPipelineSettings,
    MultiPipelineSettings,
)
from views.session_library import SessionLibrary
from views.dock_widgets import (
    VideoDock,
    MeshViewportDock,
    PersonPanelDock,
    TrackOverviewDock,
    PipelineSettingsDock,
    SessionLibraryDock,
)
from views.person_selector_bar import PersonSelectorBar
from views.bbox_overlay import render_bbox_overlay, render_edit_preview
from views.keyboard_shortcuts_dialog import KeyboardShortcutsDialog
from workers.reprocess_worker import ReprocessWorker
from theme import COLORS

log = logging.getLogger(__name__)


def _apply_dark_theme(app):
    """Apply dark palette + Fusion style."""
    from PySide6.QtWidgets import QApplication

    QApplication.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS["bg_primary"]))
    palette.setColor(QPalette.WindowText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.Base, QColor(COLORS["bg_input"]))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS["bg_active_tab"]))
    palette.setColor(QPalette.ToolTipBase, QColor(COLORS["tooltip_bg"]))
    palette.setColor(QPalette.ToolTipText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.Text, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.Button, QColor(COLORS["bg_input"]))
    palette.setColor(QPalette.ButtonText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.BrightText, QColor(COLORS["accent"]))
    palette.setColor(QPalette.Link, QColor(COLORS["accent"]))
    palette.setColor(QPalette.Highlight, QColor(COLORS["accent"]))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(COLORS["text_disabled"]))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(COLORS["text_disabled"]))
    app.setPalette(palette)

    # Fine-grained stylesheet for accent elements
    app.setStyleSheet(f"""
        QTabBar::tab {{
            background: {COLORS["bg_input"]};
            color: {COLORS["text_secondary"]};
            padding: 8px 16px;
            border: 1px solid {COLORS["border"]};
            border-bottom: none;
            border-top-left-radius: 2px;
            border-top-right-radius: 2px;
        }}
        QTabBar::tab:selected {{
            background: {COLORS["bg_active_tab"]};
            color: {COLORS["text_primary"]};
            border-bottom: 2px solid {COLORS["accent"]};
        }}
        QTabBar::tab:hover {{
            background: {COLORS["hover"]};
        }}
        QPushButton {{
            background: {COLORS["bg_input"]};
            color: {COLORS["text_primary"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 2px;
            padding: 6px 12px;
        }}
        QPushButton:hover {{
            background: {COLORS["accent"]};
        }}
        QPushButton:pressed {{
            background: {COLORS["accent_pressed"]};
        }}
        QPushButton:disabled {{
            background: {COLORS["bg_primary"]};
            color: {COLORS["text_disabled"]};
            border-color: {COLORS["border_disabled"]};
        }}
        QGroupBox {{
            border: 1px solid {COLORS["border"]};
            border-radius: 2px;
            margin-top: 8px;
            padding-top: 16px;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }}
        QSlider::groove:horizontal {{
            background: {COLORS["slider_groove"]};
            height: 6px;
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {COLORS["accent"]};
            width: 14px;
            height: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {COLORS["accent_active"]};
        }}
        QProgressBar {{
            background: {COLORS["bg_input"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 2px;
            text-align: center;
            color: {COLORS["text_primary"]};
        }}
        QProgressBar::chunk {{
            background: {COLORS["accent"]};
            border-radius: 2px;
        }}
        QScrollBar:vertical {{
            background: {COLORS["bg_scroll"]};
            width: 12px;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: {COLORS["slider_grip"]};
            min-height: 20px;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {COLORS["accent"]};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar:horizontal {{
            background: {COLORS["bg_scroll"]};
            height: 12px;
            border-radius: 4px;
        }}
        QScrollBar::handle:horizontal {{
            background: {COLORS["slider_grip"]};
            min-width: 20px;
            border-radius: 4px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {COLORS["accent"]};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0;
        }}
        QToolTip {{
            background: {COLORS["tooltip_bg"]};
            color: {COLORS["text_primary"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 2px;
            padding: 4px;
        }}
        QHeaderView::section {{
            background: {COLORS["bg_input"]};
            color: {COLORS["text_primary"]};
            border: 1px solid {COLORS["border"]};
            padding: 4px;
        }}
        QHeaderView::section:checked {{
            background: {COLORS["header_checked"]};
        }}
        QDockWidget {{
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }}
        QDockWidget::title {{
            background: {COLORS["bg_input"]};
            padding: 4px;
            border: 1px solid {COLORS["border"]};
        }}
    """)


class LogPanel(QPlainTextEdit):
    """Read-only log panel with color-coded output and line limit."""

    MAX_LINES = 10000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_LINES)
        self.setFont(self.font())  # monospace inherited from app

    def append_line(self, line: str, level: str = "info"):
        color = {
            "info": COLORS["text_primary"],
            "warning": COLORS["warning"],
            "error": COLORS["error"],
        }.get(level, COLORS["text_primary"])
        self.appendHtml(f'<span style="color: {color};">{line}</span>')

    def append_stdout(self, line: str):
        """Auto-detect level from line content."""
        lower = line.lower()
        if "error" in lower or "exception" in lower or "traceback" in lower:
            self.append_line(line, "error")
        elif "warning" in lower or "warn" in lower:
            self.append_line(line, "warning")
        else:
            self.append_line(line, "info")


class AppWindow(QMainWindow):
    """Main application window with dock-based layout.

    All panels live in QDockWidgets that the user can rearrange.  A mode
    selector (in PipelineSettingsDock) switches between single/perf/multi
    settings and shows/hides multi-only docks.  AppWindow acts as the
    central signal hub connecting all panels.
    """

    session_loaded = Signal(object)
    session_saved = Signal(Path)
    tab_changed = Signal(int)   # backward compat — emitted on mode change
    mode_changed = Signal(str)  # "single", "perf", "multi"
    interaction_mode_changed = Signal(str)  # emitted with InteractionMode.value

    MAX_RECENT = 5
    _DOCK_VERSION = 4  # increment when dock layout structure changes

    _MULTI_ONLY_DOCKS = (
        "_person_panel_dock", "_track_overview_dock",
    )

    # All content docks (excludes _log_dock which is created separately)
    _ALL_CONTENT_DOCKS = (
        "_pipeline_dock", "_video_dock", "_mesh_dock",
        "_person_panel_dock", "_track_overview_dock",
        "_session_library_dock",
    )

    # Built-in workspace presets: name → (description, set of visible dock attrs)
    _WORKSPACE_PRESETS = {
        "Review": (
            "Video + Inspector + Timeline",
            {"_pipeline_dock", "_video_dock", "_person_panel_dock",
             "_track_overview_dock"},
        ),
        "Correction": (
            "Video + 3D + Pose Corrector + Inspector",
            {"_pipeline_dock", "_video_dock", "_mesh_dock",
             "_person_panel_dock", "_track_overview_dock"},
        ),
        "Tracking": (
            "Video + Inspector + Track Overview",
            {"_pipeline_dock", "_video_dock", "_person_panel_dock",
             "_track_overview_dock"},
        ),
        "Pipeline": (
            "Video + Settings + Library + Log",
            {"_pipeline_dock", "_video_dock", "_session_library_dock"},
        ),
    }

    def __init__(self, gvhmr_root: Path | None = None, parent=None):
        super().__init__(parent)
        self._session = Session()
        self._session_path: Path | None = None
        self._settings = QSettings("GVHMR", "bodypipe")
        self._gvhmr_root = gvhmr_root or Path(__file__).resolve().parent.parent / "GVHMR"
        self._reprocess_worker: ReprocessWorker | None = None
        self._show_all_tracks = False
        self._edit_preview: dict | None = None
        self._interaction_mode: InteractionMode = InteractionMode.NAVIGATE

        self.setWindowTitle("bodypipe \u2014 Motion Capture Studio")
        self.setMinimumSize(1200, 700)

        _apply_dark_theme(QApplication.instance())

        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._setup_status_bar_toggle()
        self._setup_log_panel()
        self._setup_undo_redo()
        self._setup_signal_hub()
        self._setup_interaction_modes()
        self._add_dock_view_toggles()
        # Capture default dock layout before restoring user's saved state
        self._default_state = self.saveState(self._DOCK_VERSION)
        self._restore_geometry()
        self._restore_pipeline_configs()
        self._restore_last_video()
        self._refresh_session_library()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Create dock-based layout with mode selector.

        Why empty central widget: All real content lives in docks so users
        can freely rearrange, tabify, and float every panel.  The central
        widget is hidden (zero size) and dock nesting is enabled for
        maximum workspace flexibility.
        """
        # Empty central widget — all content in docks
        central = QWidget()
        central.setMaximumSize(0, 0)
        self.setCentralWidget(central)
        self.setDockNestingEnabled(True)

        # ---- Create inner widgets ----
        self._video_player = VideoPlayer()
        self._mesh_viewport = MeshViewport(gvhmr_root=self._gvhmr_root)
        self._mesh_viewport.set_session(self._session)
        self._identity_inspector = IdentityInspector(self._session)
        self._track_overview = TrackOverview()
        self._session_library = SessionLibrary(gvhmr_root=self._gvhmr_root)

        # ---- Create settings widgets ----
        self._single_settings = SinglePipelineSettings(self._session, self._gvhmr_root)
        self._perf_settings = PerfPipelineSettings(self._session, self._gvhmr_root)
        self._multi_settings = MultiPipelineSettings(self._session, self._gvhmr_root)

        # ---- Create dock widgets ----
        self._video_dock = VideoDock(self._video_player, self)
        self._mesh_dock = MeshViewportDock(self._mesh_viewport, self)
        # Viewport toolbar → viewport signals
        dock = self._mesh_dock
        dock._camera_combo.currentIndexChanged.connect(
            lambda idx: self._mesh_viewport.set_camera_mode("incam" if idx == 0 else "orbit")
        )
        dock._motion_source_combo.currentIndexChanged.connect(
            lambda idx: self._mesh_viewport.set_motion_source(
                dock._motion_source_combo.itemData(idx) or "auto"
            )
        )
        self._mesh_viewport.source_status_changed.connect(
            lambda text: (
                dock._motion_status.setText(text),
                dock._motion_status.setToolTip(
                    self._mesh_viewport.current_source_status_detail_text()
                ),
            )
        )
        self._mesh_viewport.mesh_status_changed.connect(
            lambda text: (
                dock._mesh_status.setText(text),
                dock._mesh_status.setToolTip(
                    self._mesh_viewport.current_mesh_status_detail_text()
                ),
            )
        )
        dock._grid_cb.toggled.connect(self._mesh_viewport.set_show_grid)
        dock._labels_cb.toggled.connect(self._mesh_viewport.set_show_joint_labels)
        dock._frustum_cb.toggled.connect(self._mesh_viewport.set_show_camera_frustum)
        dock._frustum_cb.toggled.connect(dock._fov_label.setVisible)
        dock._frustum_cb.toggled.connect(dock._fov_spin.setVisible)
        dock._fov_spin.valueChanged.connect(self._mesh_viewport.set_frustum_fov)
        # Playback quality combo → mesh viewport
        dock._playback_quality.currentIndexChanged.connect(self._on_playback_quality_changed)
        dock._motion_status.setText(self._mesh_viewport.current_source_status_text())
        dock._motion_status.setToolTip(
            self._mesh_viewport.current_source_status_detail_text()
        )
        dock._mesh_status.setText(self._mesh_viewport.current_mesh_status_text())
        dock._mesh_status.setToolTip(
            self._mesh_viewport.current_mesh_status_detail_text()
        )
        # Person panel: selector bar + Identity/Pose Corrector tabs
        self._person_bar = PersonSelectorBar()
        self._person_bar.set_session(self._session)
        from views.pose_corrector_panel import PoseCorrectorPanel
        self._pose_corrector = PoseCorrectorPanel(
            session=self._session, gvhmr_root=self._gvhmr_root,
            viewport=self._mesh_viewport, parent=self,
        )
        self._pose_corrector.set_track_overview(self._track_overview)
        self._person_panel_dock = PersonPanelDock(
            self._person_bar, self._identity_inspector, self._pose_corrector, self,
        )
        self._track_overview_dock = TrackOverviewDock(self._track_overview, self)
        self._pipeline_dock = PipelineSettingsDock(
            self._single_settings, self._perf_settings, self._multi_settings, self,
        )
        self._session_library_dock = SessionLibraryDock(self._session_library, self)

        # ---- Arrange docks ----
        # Put dock tabs at the top for side panels; bottom area keeps default (South)
        from PySide6.QtWidgets import QTabWidget
        self.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.North)
        self.setTabPosition(Qt.LeftDockWidgetArea, QTabWidget.North)

        # Left: Pipeline Settings + Session Library (tabified, Settings on top)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._pipeline_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._session_library_dock)
        self.tabifyDockWidget(self._pipeline_dock, self._session_library_dock)
        self._pipeline_dock.raise_()

        # Right: Video, then Person panel split to its right (full-height column)
        self.addDockWidget(Qt.RightDockWidgetArea, self._video_dock)
        self.splitDockWidget(self._video_dock, self._person_panel_dock, Qt.Horizontal)

        # Video/3D stacked: split Video vertically so Mesh goes below Video
        # (Identity stays full-height in the right column)
        self.splitDockWidget(self._video_dock, self._mesh_dock, Qt.Vertical)
        self.resizeDocks(
            [self._video_dock, self._mesh_dock],
            [520, 760],
            Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [self._video_dock, self._person_panel_dock],
            [920, 560],
            Qt.Orientation.Horizontal,
        )

        # Bottom: Track overview (log dock added later in _setup_log_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._track_overview_dock)

    def _setup_menu(self):
        """Create menu bar with File, Edit, View, Help menus."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_video_action = QAction("Open &Video...", self)
        open_video_action.setShortcut("Ctrl+O")
        open_video_action.triggered.connect(self._on_open_video)
        file_menu.addAction(open_video_action)

        open_session_action = QAction("Open &Session...", self)
        open_session_action.setShortcut("Ctrl+Shift+O")
        open_session_action.triggered.connect(self._on_open_session)
        file_menu.addAction(open_session_action)

        save_session_action = QAction("&Save Session", self)
        save_session_action.setShortcut("Ctrl+S")
        save_session_action.triggered.connect(self._on_save_session)
        file_menu.addAction(save_session_action)

        save_as_action = QAction("Save Session &As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._on_save_session_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        self._recent_menu = QMenu("Recent Sessions", self)
        file_menu.addMenu(self._recent_menu)
        self._update_recent_menu()

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        self._undo_action = QAction("&Undo", self)
        self._undo_action.setShortcut("Ctrl+Z")
        self._undo_action.setEnabled(False)
        self._undo_action.triggered.connect(self._on_undo)
        edit_menu.addAction(self._undo_action)

        self._redo_action = QAction("&Redo", self)
        self._redo_action.setShortcut("Ctrl+Shift+Z")
        self._redo_action.setEnabled(False)
        self._redo_action.triggered.connect(self._on_redo)
        edit_menu.addAction(self._redo_action)

        # View menu (stored for _add_dock_view_toggles)
        self._view_menu = menubar.addMenu("&View")

        self._toggle_statusbar_action = QAction("Toggle &Status Bar", self)
        self._toggle_statusbar_action.setCheckable(True)
        self._toggle_statusbar_action.setChecked(True)
        self._view_menu.addAction(self._toggle_statusbar_action)

        self._toggle_log_action = QAction("Toggle &Log Panel", self)
        self._toggle_log_action.setShortcut("Ctrl+L")
        self._toggle_log_action.setCheckable(True)
        self._toggle_log_action.setChecked(True)
        self._view_menu.addAction(self._toggle_log_action)

        self._toggle_hud_action = QAction("Toggle &HUD Overlay", self)
        self._toggle_hud_action.setShortcut("Ctrl+H")
        self._toggle_hud_action.setCheckable(True)
        self._toggle_hud_action.setChecked(False)
        self._toggle_hud_action.toggled.connect(self._on_toggle_hud)
        self._view_menu.addAction(self._toggle_hud_action)

        self._toggle_transport_action = QAction("Hide &Transport Controls", self)
        self._toggle_transport_action.setShortcut("T")
        self._toggle_transport_action.setCheckable(True)
        self._toggle_transport_action.setChecked(False)
        self._toggle_transport_action.toggled.connect(
            lambda hidden: self._video_player.set_transport_hidden(hidden)
        )
        self._view_menu.addAction(self._toggle_transport_action)

        self._view_menu.addSeparator()
        self._workspace_menu = QMenu("&Workspace", self)
        self._view_menu.addMenu(self._workspace_menu)
        self._build_workspace_menu()

        # Help menu
        help_menu = menubar.addMenu("&Help")

        shortcuts_action = QAction("&Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._on_keyboard_shortcuts)
        help_menu.addAction(shortcuts_action)

        help_menu.addSeparator()

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _add_dock_view_toggles(self):
        """Add per-dock toggle actions to the View menu."""
        self._view_menu.addSeparator()
        for dock in (
            self._pipeline_dock, self._video_dock, self._mesh_dock,
            self._person_panel_dock,
            self._track_overview_dock, self._session_library_dock,
        ):
            self._view_menu.addAction(dock.toggleViewAction())

    def _setup_status_bar(self):
        """Create status bar with operation status, mode indicator, frame counter, FPS."""
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("Ready")
        self._mode_label = QLabel("")
        self._mode_label.setStyleSheet(
            f"QLabel {{ background: {COLORS['accent']}; color: #ffffff; "
            f"padding: 1px 8px; border-radius: 3px; font-weight: bold; "
            f"font-size: 11px; }}"
        )
        self._frame_label = QLabel("")
        self._fps_label = QLabel("")

        self._status_bar.addWidget(self._status_label, 1)
        self._status_bar.addPermanentWidget(self._mode_label)
        self._status_bar.addPermanentWidget(self._frame_label)
        self._status_bar.addPermanentWidget(self._fps_label)

    def _setup_status_bar_toggle(self):
        """Wire status bar toggle after both status bar and menu are created."""
        self._toggle_statusbar_action.toggled.connect(self._status_bar.setVisible)

    def _setup_log_panel(self):
        """Create log dock widget, split horizontally beside track overview."""
        self._log_panel = LogPanel()
        self._log_dock = QDockWidget("Log", self)
        self._log_dock.setObjectName("LogDock")
        self._log_dock.setWidget(self._log_panel)
        self._log_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._log_dock)
        # Split side-by-side instead of tabifying — both visible at once
        self.splitDockWidget(self._track_overview_dock, self._log_dock, Qt.Horizontal)

        self._toggle_log_action.toggled.connect(self._log_dock.setVisible)
        self._log_dock.visibilityChanged.connect(self._toggle_log_action.setChecked)

    def _restore_geometry(self):
        """Restore window geometry and dock state from settings."""
        geometry = self._settings.value("geometry")
        if geometry and isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        else:
            self.resize(1600, 900)

        state = self._settings.value("windowState")
        if state and isinstance(state, QByteArray):
            self.restoreState(state, self._DOCK_VERSION)

        # Enforce mode-based dock visibility — restoreState may have made
        # multi-only docks visible from a previous session.
        self._update_dock_visibility()

    # ------------------------------------------------------------------
    # Workspace presets
    # ------------------------------------------------------------------

    def _build_workspace_menu(self):
        """Populate the View > Workspace submenu with presets and actions."""
        self._workspace_menu.clear()
        for name, (desc, _visible) in self._WORKSPACE_PRESETS.items():
            action = QAction(f"{name}  —  {desc}", self)
            action.triggered.connect(lambda checked, n=name: self._apply_preset(n))
            self._workspace_menu.addAction(action)

        self._workspace_menu.addSeparator()

        # Custom saved layouts
        custom_names = self._get_custom_workspace_names()
        if custom_names:
            for cname in custom_names:
                action = QAction(cname, self)
                action.triggered.connect(
                    lambda checked, n=cname: self._apply_custom_workspace(n),
                )
                self._workspace_menu.addAction(action)
            self._workspace_menu.addSeparator()

        save_action = QAction("Save Current Layout...", self)
        save_action.triggered.connect(self._on_save_workspace)
        self._workspace_menu.addAction(save_action)

        reset_action = QAction("Reset to Default", self)
        reset_action.triggered.connect(self._on_reset_workspace)
        self._workspace_menu.addAction(reset_action)

    def _apply_preset(self, name: str):
        """Apply a built-in workspace preset by showing/hiding docks."""
        _desc, visible_attrs = self._WORKSPACE_PRESETS[name]

        # Restore default dock arrangement first
        self.restoreState(self._default_state, self._DOCK_VERSION)

        # Show/hide content docks per preset
        for attr in self._ALL_CONTENT_DOCKS:
            dock = getattr(self, attr, None)
            if dock:
                dock.setVisible(attr in visible_attrs)

        # Log dock: visible in all presets that have track overview
        if hasattr(self, "_log_dock"):
            show_log = "_track_overview_dock" in visible_attrs or name == "Pipeline"
            self._log_dock.setVisible(show_log)
            self._toggle_log_action.setChecked(show_log)

        # Raise video dock in tabified groups
        self._video_dock.raise_()

        self.set_status(f"Workspace: {name}")

    def _get_custom_workspace_names(self) -> list[str]:
        """Return names of user-saved custom workspace layouts."""
        raw = self._settings.value("workspace/custom_names")
        if raw and isinstance(raw, list):
            return raw
        return []

    def _on_save_workspace(self):
        """Prompt for a name and save the current dock layout."""
        name, ok = QInputDialog.getText(
            self, "Save Workspace", "Layout name:",
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        # Save the current state bytes
        state = self.saveState(self._DOCK_VERSION)
        self._settings.setValue(f"workspace/state/{name}", state)

        # Update the custom names list
        names = self._get_custom_workspace_names()
        if name not in names:
            names.append(name)
        self._settings.setValue("workspace/custom_names", names)

        # Rebuild menu to include the new entry
        self._build_workspace_menu()
        self.set_status(f"Workspace saved: {name}")

    def _apply_custom_workspace(self, name: str):
        """Restore a user-saved custom workspace layout."""
        state = self._settings.value(f"workspace/state/{name}")
        if state and isinstance(state, QByteArray):
            self.restoreState(state, self._DOCK_VERSION)
            self._update_dock_visibility()
            self.set_status(f"Workspace: {name}")

    def _on_reset_workspace(self):
        """Restore the hardcoded initial dock layout."""
        self.restoreState(self._default_state, self._DOCK_VERSION)
        self._update_dock_visibility()
        self.set_status("Workspace reset to default")

    # ------------------------------------------------------------------
    # Pipeline config persistence
    # ------------------------------------------------------------------

    _TAB_CONFIG_MAP = {
        "single": "_single_settings",
        "perf": "_perf_settings",
        "multi": "_multi_settings",
    }

    def _save_pipeline_configs(self):
        """Save each settings panel's pipeline config to QSettings."""
        import json

        for mode, attr in self._TAB_CONFIG_MAP.items():
            widget = getattr(self, attr, None)
            if widget and hasattr(widget, "get_config"):
                config = widget.get_config()
                self._settings.setValue(
                    f"pipeline_config/{mode}",
                    json.dumps(config.to_dict()),
                )

    def _restore_pipeline_configs(self):
        """Restore each settings panel's pipeline config from QSettings."""
        import json

        for mode, attr in self._TAB_CONFIG_MAP.items():
            widget = getattr(self, attr, None)
            if not widget or not hasattr(widget, "set_config"):
                continue
            raw = self._settings.value(f"pipeline_config/{mode}")
            if not raw or not isinstance(raw, str):
                continue
            try:
                data = json.loads(raw)
                config = PipelineConfig.from_dict(data)
                widget.set_config(config)
            except Exception:
                pass  # Ignore corrupt/stale settings

    def closeEvent(self, event):
        """Save window geometry, dock state, and pipeline configs on close."""
        self._save_pipeline_configs()
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState(self._DOCK_VERSION))
        # Wait for any running worker threads to prevent QThread destruction crash
        for settings in (self._single_settings, self._perf_settings, self._multi_settings):
            worker = getattr(settings, "_worker", None)
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait(5000)
        # Clean up reprocess worker
        if self._reprocess_worker is not None and self._reprocess_worker.isRunning():
            self._reprocess_worker.cancel()
            self._reprocess_worker.wait(5000)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Signal Hub
    # ------------------------------------------------------------------

    def _setup_signal_hub(self):
        """Wire all cross-panel signals.

        Why centralised: In the tab layout, MultiPersonTab was the signal
        hub connecting sub-panels.  With docks, AppWindow owns all panels
        directly and takes over the hub role so panels stay decoupled.
        """
        # Settings widgets → status bar + log
        for settings in (self._single_settings, self._perf_settings, self._multi_settings):
            settings.status_message.connect(self.set_status)
            settings.log_message.connect(
                lambda text, level: self._log_panel.append_line(text, level)
            )

        # Video loaded → shared VideoPlayer
        self._single_settings.video_loaded.connect(self._on_video_loaded)
        self._perf_settings.video_loaded.connect(self._on_video_loaded)
        self._multi_settings.video_loaded.connect(self._on_video_loaded)

        # Pipeline finished handlers
        self._single_settings.pipeline_finished.connect(self._on_pipeline_output)
        self._perf_settings.pipeline_finished.connect(self._on_pipeline_output)
        self._multi_settings.pipeline_finished.connect(self._on_multi_pipeline_finished)

        # Shared VideoPlayer → status bar + multi-mode broadcast
        self._video_player.frame_changed.connect(self._on_video_frame_changed)

        # Auto-switch 3D viewport to wireframe during scrubbing/playback
        self._video_player.scrub_started.connect(
            lambda: self._mesh_viewport.set_scrubbing(True)
        )
        self._video_player.scrub_ended.connect(
            lambda: self._mesh_viewport.set_scrubbing(False)
        )
        self._video_player.playback_toggled.connect(
            lambda playing: self._mesh_viewport.set_scrubbing(playing)
        )

        # Multi-mode signal hub (signals only fire when panels are visible)
        self._video_player.frame_clicked.connect(self._identity_inspector.on_frame_click)
        self._video_player.bbox_dragged.connect(self._identity_inspector.on_bbox_drag)
        self._track_overview.person_clicked.connect(self._on_track_clicked)
        self._identity_inspector.frame_requested.connect(self._video_player.seek)
        self._person_bar.person_changed.connect(self._on_person_changed)
        self._person_bar.show_all_toggled.connect(self._identity_inspector.set_show_all_tracks)
        self._identity_inspector.person_changed.connect(self._on_identity_person_changed)
        self._identity_inspector.bbox_overlay_changed.connect(self._on_bbox_overlay_changed)
        self._identity_inspector.keyframe_changed.connect(self._on_keyframe_changed)
        self._identity_inspector.track_modified.connect(self._on_tracks_modified)
        self._identity_inspector.reprocess_requested.connect(self._on_reprocess_requested)
        self._pose_corrector.frame_requested.connect(self._video_player.seek)

        # Auto-raise PoseCorrector dock when a joint is clicked in 3D viewport
        self._mesh_viewport.joint_clicked.connect(self._on_joint_clicked_auto_raise)

        # Speed sync between video player and track timeline
        self._video_player.speed_changed.connect(self._track_overview.set_speed)
        self._track_overview.speed_changed.connect(self._video_player.set_playback_speed)

        # Track overview transport → video player
        self._track_overview.play_toggled.connect(self._video_player._toggle_play)
        self._track_overview.go_to_start.connect(lambda: self._video_player.seek(0))
        self._track_overview.loop_toggled.connect(self._video_player.set_looping)

        # Video player → track overview (visual sync)
        self._video_player.playback_toggled.connect(self._track_overview.set_playing)
        self._video_player.loop_toggled.connect(self._track_overview.set_loop)

        # Session library → load session on double-click
        self._session_library.session_load_requested.connect(
            lambda path: self._load_session(Path(path))
        )
        # Refresh library when a session is saved
        self.session_saved.connect(lambda _: self._refresh_session_library())

        # Mode selector
        self._pipeline_dock.mode_changed.connect(self._on_mode_changed)

    # ------------------------------------------------------------------
    # Interaction modes (Phase 10)
    # ------------------------------------------------------------------

    # Mode → number key mapping
    _MODE_KEYS = {
        Qt.Key_1: InteractionMode.NAVIGATE,
        Qt.Key_2: InteractionMode.SELECT,
        Qt.Key_3: InteractionMode.CORRECT,
        Qt.Key_4: InteractionMode.TRACK,
    }

    def _setup_interaction_modes(self):
        """Initialise interaction mode state and update the status bar indicator."""
        self._update_mode_indicator()

    def set_interaction_mode(self, mode: InteractionMode):
        """Switch the active interaction mode and update UI."""
        if self._interaction_mode == mode:
            return
        self._interaction_mode = mode
        self._update_mode_indicator()
        self.interaction_mode_changed.emit(mode.value)
        self._mesh_viewport.set_hud_mode(mode.value)
        log.info("Interaction mode → %s", mode.value)

    def _update_mode_indicator(self):
        """Update the status bar mode pill label."""
        mode = self._interaction_mode
        # Color-code the mode pill for quick visual identification
        mode_colors = {
            InteractionMode.NAVIGATE: COLORS["accent"],
            InteractionMode.SELECT:   "#3d85c6",  # blue
            InteractionMode.CORRECT:  "#cc4125",  # red
            InteractionMode.TRACK:    "#6aa84f",  # green
        }
        bg = mode_colors.get(mode, COLORS["accent"])
        self._mode_label.setStyleSheet(
            f"QLabel {{ background: {bg}; color: #ffffff; "
            f"padding: 1px 8px; border-radius: 3px; font-weight: bold; "
            f"font-size: 11px; }}"
        )
        self._mode_label.setText(f"{mode.value} (#{list(InteractionMode).index(mode) + 1})")

    def _sync_camera_mode(self, mode: str):
        """Set viewport camera mode AND sync the toolbar dropdown."""
        self._mesh_viewport.set_camera_mode(mode)
        combo = self._mesh_dock._camera_combo
        combo.blockSignals(True)
        combo.setCurrentIndex(0 if mode == "incam" else 1)
        combo.blockSignals(False)

    def keyPressEvent(self, event):
        """Route key events based on active interaction mode.

        Why centralised: Each mode remaps the same physical keys to different
        actions — e.g. G means "go-to-frame" in Navigate but "next unreviewed"
        in Track mode.  Processing here before child widgets see the event
        ensures mode-awareness without each widget needing to know about modes.
        """
        key = event.key()
        mod = event.modifiers()

        # Mode switching: 1-4 keys (without modifiers)
        if not mod and key in self._MODE_KEYS:
            self.set_interaction_mode(self._MODE_KEYS[key])
            return

        # Global shortcuts (mode-independent)
        if not mod and key == Qt.Key_V:
            # Toggle camera mode between incam and orbit
            vp = self._mesh_viewport
            new_mode = "orbit" if vp._camera_mode == "incam" else "incam"
            self._sync_camera_mode(new_mode)
            self.set_status(f"Camera: {new_mode}")
            return
        if not mod and key == Qt.Key_M:
            # Toggle multi-person skeleton view
            vp = self._mesh_viewport
            vp._show_all_persons = not vp._show_all_persons
            vp._refresh_mesh()
            self.set_status(f"Multi-person: {'on' if vp._show_all_persons else 'off'}")
            return

        # Dispatch to mode-specific handler
        handled = False
        if self._interaction_mode == InteractionMode.NAVIGATE:
            handled = self._key_navigate(key, mod)
        elif self._interaction_mode == InteractionMode.SELECT:
            handled = self._key_select(key, mod)
        elif self._interaction_mode == InteractionMode.CORRECT:
            handled = self._key_correct(key, mod)
        elif self._interaction_mode == InteractionMode.TRACK:
            handled = self._key_track(key, mod)

        if not handled:
            super().keyPressEvent(event)

    def _key_navigate(self, key, mod) -> bool:
        """Navigate mode: WASD orbit camera, G go-to-frame."""
        if key == Qt.Key_W:
            self._orbit_nudge(pitch=-5)
            return True
        if key == Qt.Key_S:
            self._orbit_nudge(pitch=5)
            return True
        if key == Qt.Key_A:
            self._orbit_nudge(yaw=-5)
            return True
        if key == Qt.Key_D:
            self._orbit_nudge(yaw=5)
            return True
        if key == Qt.Key_G:
            self._go_to_frame_dialog()
            return True
        return False

    def _key_select(self, key, mod) -> bool:
        """Select mode: G go-to-frame, Escape deselect."""
        if key == Qt.Key_G:
            self._go_to_frame_dialog()
            return True
        if key == Qt.Key_Escape:
            self._mesh_viewport._selected_joint = -1
            self._mesh_viewport.joint_clicked.emit(-1)
            if hasattr(self._mesh_viewport, 'update'):
                self._mesh_viewport.update()
            return True
        return False

    def _key_correct(self, key, mod) -> bool:
        """Correct mode: G open euler, R reset joint, Escape deselect."""
        if key == Qt.Key_G:
            # Focus the person panel dock and raise Pose Corrector tab
            self._person_panel_dock.setVisible(True)
            self._person_panel_dock.raise_()
            self._person_panel_dock.tabs.setCurrentIndex(1)
            return True
        if key == Qt.Key_R:
            # Reset the currently selected joint rotation
            self._pose_corrector.reset_current_joint()
            return True
        if key == Qt.Key_Escape:
            self._mesh_viewport._selected_joint = -1
            self._mesh_viewport.joint_clicked.emit(-1)
            if hasattr(self._mesh_viewport, 'update'):
                self._mesh_viewport.update()
            return True
        return False

    def _key_track(self, key, mod) -> bool:
        """Track mode: G next unreviewed, Tab next person, Shift+Tab prev, E edit bbox."""
        if key == Qt.Key_G:
            self._identity_inspector.go_to_next_unreviewed()
            return True
        if key == Qt.Key_Tab:
            if mod & Qt.ShiftModifier:
                self._cycle_person(-1)
            else:
                self._cycle_person(1)
            return True
        if key == Qt.Key_E:
            self._person_panel_dock.setVisible(True)
            self._person_panel_dock.raise_()
            self._person_panel_dock.tabs.setCurrentIndex(0)
            self._identity_inspector._on_edit_bbox()
            return True
        return False

    def _orbit_nudge(self, yaw: float = 0, pitch: float = 0):
        """Nudge the orbit camera by the given yaw/pitch degrees."""
        vp = self._mesh_viewport
        if vp._camera_mode != "orbit":
            self._sync_camera_mode("orbit")
        vp._orbit_yaw += yaw
        vp._orbit_pitch = float(np.clip(vp._orbit_pitch + pitch, -89, 89))
        vp._update_camera()
        vp.camera_changed.emit(vp._camera_state())

    def _go_to_frame_dialog(self):
        """Show a go-to-frame input dialog."""
        max_frame = max(0, self._video_player.num_frames - 1)
        frame, ok = QInputDialog.getInt(
            self, "Go to Frame", f"Frame (0–{max_frame}):",
            self._video_player._current_frame, 0, max_frame,
        )
        if ok:
            self._video_player.seek(frame)

    def _cycle_person(self, direction: int):
        """Cycle through person tracks by direction (+1 = next, -1 = prev)."""
        pids = sorted(self._session.person_tracks.keys())
        if not pids:
            return
        current = self._session.selected_person
        if current in pids:
            idx = pids.index(current)
            idx = (idx + direction) % len(pids)
        else:
            idx = 0
        new_pid = pids[idx]
        # Route through person bar — triggers _on_person_changed
        self._person_bar.set_person(new_pid)
        self._on_person_changed(new_pid)

    def _on_toggle_hud(self, visible: bool):
        """Toggle the HUD overlay on the mesh viewport."""
        self._mesh_viewport.set_hud_visible(visible)

    def _on_playback_quality_changed(self, index: int):
        """Handle playback quality combo change."""
        mode_map = {0: RenderMode.WIREFRAME, 1: RenderMode.FAST, 2: RenderMode.FULL}
        self._mesh_viewport.set_playback_render_mode(mode_map.get(index, RenderMode.FAST))

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _on_mode_changed(self, mode: str):
        """Show/hide multi-only docks and emit signals."""
        self._update_dock_visibility()
        mode_to_index = {"single": 0, "perf": 1, "multi": 2}
        self.tab_changed.emit(mode_to_index.get(mode, 0))
        self.mode_changed.emit(mode)

    def _update_dock_visibility(self):
        """Show/hide docks based on current pipeline mode."""
        is_multi = self._pipeline_dock.current_mode == "multi"
        for attr in self._MULTI_ONLY_DOCKS:
            dock = getattr(self, attr, None)
            if dock:
                dock.setVisible(is_multi)

    # ------------------------------------------------------------------
    # Video frame handling
    # ------------------------------------------------------------------

    def _on_video_frame_changed(self, frame_idx: int):
        """Update status bar, HUD, video display, and 3D viewport on every frame."""
        self.set_frame_info(frame_idx, self._video_player.num_frames)
        self.set_fps_info(self._video_player.fps)
        self._session.current_frame = frame_idx
        # Update HUD overlay with current state
        self._mesh_viewport.update_hud(
            frame=frame_idx,
            total_frames=self._video_player.num_frames,
            speed=self._video_player._playback_speed,
            person=self._session.selected_person,
        )

        # Always update video display and 3D viewport
        raw = self._video_player.get_raw_frame(frame_idx)
        self._mesh_viewport.set_video_frame(raw)
        self._mesh_viewport.on_frame_changed(frame_idx)
        self._show_frame(frame_idx)

        # Multi-only widgets
        if self._pipeline_dock.current_mode == "multi":
            self._track_overview.set_current_frame(frame_idx)
            self._identity_inspector.set_frame(frame_idx)
            self._pose_corrector.on_frame_changed(frame_idx)

    def _on_video_loaded(self, video_path):
        """Load video into the shared VideoPlayer and restore cached results."""
        self._video_player.set_video(
            video_path, self._session.num_frames, self._session.fps,
        )
        # Persist last video path for startup restore
        self._settings.setValue("last_video_path", str(video_path))
        # Check for existing pipeline results on disk
        self._try_restore_results(video_path)

    def _show_frame(self, frame_idx: int):
        """Display the current frame with bbox overlays and edit preview."""
        frame = self._video_player.get_raw_frame(frame_idx)
        if frame is not None:
            composited = render_bbox_overlay(
                frame,
                self._session,
                frame_idx,
                selected_person=self._session.selected_person,
                show_all_tracks=self._show_all_tracks,
            )
            if self._edit_preview:
                composited = render_edit_preview(composited, self._edit_preview)
            self._video_player.set_frame(composited)

        # Update bbox corner handles for the selected person
        pid = self._session.selected_person
        track = self._session.person_tracks.get(pid) if pid >= 0 else None
        if track is not None and track.bboxes is not None and frame_idx < len(track.bboxes):
            bbox = track.bboxes[frame_idx]
            # Use corrected bbox if available
            if (track.bbox_corrections is not None
                    and frame_idx < len(track.bbox_corrections)
                    and not np.all(track.bbox_corrections[frame_idx] == 0)):
                bbox = track.bbox_corrections[frame_idx]
            w, h = self._session.img_width, self._session.img_height
            if w > 0 and h > 0:
                self._video_player.set_current_bbox((
                    float(bbox[0]) / w, float(bbox[1]) / h,
                    float(bbox[2]) / w, float(bbox[3]) / h,
                ))
            else:
                self._video_player.set_current_bbox(None)
        else:
            self._video_player.set_current_bbox(None)

    # ------------------------------------------------------------------
    # Startup restore & result loading
    # ------------------------------------------------------------------

    def _restore_last_video(self):
        """On startup, restore the last-loaded video into the current settings panel.

        Why: Without this, the app starts with all docks empty even if the user
        was working on a video in the previous session. Pipeline configs are
        restored by _restore_pipeline_configs(), but the video itself was not.
        """
        last_path = self._settings.value("last_video_path")
        if not last_path or not isinstance(last_path, str):
            return
        path = Path(last_path)
        if not path.is_file():
            return
        settings = self._pipeline_dock.current_settings
        if hasattr(settings, "_load_video"):
            settings._load_video(str(path))

    def _try_restore_results(self, video_path):
        """Load cached pipeline results if a previous run exists for this video.

        Why: When a previously-processed video is loaded (on startup or via
        File > Open), the user expects to see their results immediately — not
        a blank 3D viewport and empty inspector.  This checks for existing
        person tracks (from a loaded session) or scans the expected output
        directory for the current pipeline mode.
        """
        video_path = Path(video_path)

        # If session already has person tracks (loaded from session JSON),
        # hydrate heavy data from disk and refresh all panels.
        if self._session.person_tracks:
            self._hydrate_person_tracks(force_motion_reload=True)
            self._refresh_all_panels()
            return

        # Otherwise, check for cached results on disk
        mode = self._pipeline_dock.current_mode

        # Use existing output_dir from session if valid, else derive from mode
        if self._session.output_dir and self._session.output_dir.is_dir():
            output_dir = self._session.output_dir
        elif mode == "multi":
            output_dir = self._gvhmr_root / "outputs" / "multi_person" / video_path.stem
        else:
            output_dir = self._gvhmr_root / "outputs" / "demo" / video_path.stem

        if not output_dir.is_dir():
            return

        self._session.output_dir = output_dir

        if mode == "multi":
            self._load_results_from_output_dir(output_dir)
        else:
            # Single/perf: load output preview video if available
            for name in ("side_by_side.mp4", "incam.mp4"):
                preview = output_dir / name
                if preview.is_file():
                    self._load_output_preview(preview)
                    break

    def _load_results_from_output_dir(self, output_dir: Path):
        """Load person tracks from an existing multi-person output directory.

        Why: After the tab→dock migration, pipeline results were only loaded
        when the pipeline finished (via _on_multi_pipeline_finished). This
        method enables loading cached results from disk on startup or when
        a previously-processed video is opened, without re-running the pipeline.
        """
        from models.session import PersonTrack

        # Discover person directories on disk
        person_dirs = sorted(
            [d for d in output_dir.iterdir()
             if d.is_dir() and d.name.startswith("person_")],
            key=lambda d: d.name,
        )
        if not person_dirs:
            return

        for pdir in person_dirs:
            try:
                pid = int(pdir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            confidences, confidence_breakdown = self._load_confidences_csv(pdir)
            smplx_params, soma_params, body_model_type, motion_sources = self._load_motion_params(pdir)

            pt = PersonTrack(
                person_id=pid,
                person_dir=pdir,
                confidences=confidences,
                confidence_breakdown=confidence_breakdown,
                smplx_params=smplx_params,
                soma_params=soma_params,
                body_model_type=body_model_type,
                motion_sources=motion_sources,
            )
            self._session.person_tracks[pid] = pt

        # Load crossing spans
        import json as _json
        for pid, pt in self._session.person_tracks.items():
            if pt.person_dir:
                spans_path = pt.person_dir / "crossing_spans.json"
                if spans_path.is_file():
                    try:
                        spans = _json.loads(spans_path.read_text())
                        self._session.crossing_spans[pid] = [
                            tuple(s) for s in spans
                        ]
                    except Exception:
                        pass

        self._refresh_all_panels()

    def _motion_file_stamp(self, person_dir: Path | None) -> tuple[str | None, int | None]:
        if person_dir is None or not person_dir.is_dir():
            return None, None
        hybrid_pt = self._select_hybrid_motion_file(person_dir)
        if hybrid_pt is not None:
            return str(hybrid_pt), int(hybrid_pt.stat().st_mtime_ns)
        hmr4d_pt = person_dir / "demo" / "isolated_video" / "hmr4d_results.pt"
        if hmr4d_pt.is_file():
            return str(hmr4d_pt), int(hmr4d_pt.stat().st_mtime_ns)
        return None, None

    def _refresh_track_motion_from_disk(self, track, *, force: bool = False) -> bool:
        person_dir = track.person_dir
        stamp = self._motion_file_stamp(person_dir)
        current_stamp = getattr(track, "_motion_file_stamp", (None, None))
        needs_reload = (
            force
            or (track.smplx_params is None and track.soma_params is None)
            or getattr(track, "motion_sources", None) is None
            or current_stamp != stamp
        )
        if not needs_reload:
            return False
        smplx, soma, bmt, motion_sources = self._load_motion_params(person_dir)
        track.smplx_params = smplx
        track.soma_params = soma
        track.body_model_type = bmt
        track.motion_sources = motion_sources
        track._motion_file_stamp = stamp
        return True

    def _hydrate_person_tracks(self, *, force_motion_reload: bool = False):
        """Load heavy data (smplx_params, confidences) from disk for existing tracks.

        Why: Session JSON stores lightweight fields (person_id, person_dir,
        keyframes) but not heavy data like SMPL-X parameters or per-frame
        confidence arrays.  This method fills in the gaps from disk so that
        the 3D viewport, identity inspector, and track overview can render.
        """
        self._session.derived_c2w = None
        self._session.raw_c2w = None
        for _pid, track in self._session.person_tracks.items():
            if not track.person_dir or not track.person_dir.is_dir():
                continue
            self._refresh_track_motion_from_disk(
                track,
                force=force_motion_reload,
            )
            if track.confidences is None:
                track.confidences, track.confidence_breakdown = (
                    self._load_confidences_csv(track.person_dir)
                )
            # Fall back to GEM-X embedded confidences
            if track.confidences is None:
                params = track.soma_params or track.smplx_params
                if params is not None:
                    gemx_conf = params.get("confidences")
                    if gemx_conf is not None:
                        track.confidences = gemx_conf.tolist()
            # Load crop metadata for world-grounding
            self._load_crop_metadata(track)

    def _load_crop_metadata(self, track):
        """Load crop_bbox and K_crop for a person track (world-grounding)."""
        if track.person_dir is None or not track.person_dir.is_dir():
            return
        # Load crop_bbox from person_meta.json
        if track.crop_bbox is None:
            meta_path = track.person_dir / "person_meta.json"
            if meta_path.is_file():
                try:
                    import json as _json
                    meta = _json.loads(meta_path.read_text())
                    bbox = meta.get("crop_bbox")
                    if bbox is not None and len(bbox) == 4:
                        track.crop_bbox = [int(v) for v in bbox]
                except Exception as e:
                    log.debug("Failed to load crop_bbox from %s: %s", meta_path, e)
        # Extract K_crop from loaded params' K_fullimg[0]
        if track.K_crop is None:
            params = track.soma_params or track.smplx_params
            if params is not None:
                K_full = params.get("K_fullimg")
                if K_full is not None:
                    K_arr = np.array(K_full, dtype=np.float32)
                    if K_arr.ndim == 3:
                        track.K_crop = K_arr[0]  # (3, 3)
                    elif K_arr.ndim == 2 and K_arr.shape == (3, 3):
                        track.K_crop = K_arr

    def _load_slam_if_needed(self):
        """Load SLAM camera trajectory — shared first, then per-person."""
        if self._session.slam_c2w is not None:
            return
        if self._session.output_dir is None:
            return
        # Search candidates: shared_slam.pt, then per-person preprocess paths
        candidates = [self._session.output_dir / "shared_slam.pt"]
        for track in self._session.person_tracks.values():
            if track.person_dir:
                candidates.append(
                    Path(track.person_dir) / "demo" / "preprocess" / "slam.pt"
                )
        for path in candidates:
            if not path.is_file():
                continue
            try:
                import torch
                slam = torch.load(str(path), map_location="cpu", weights_only=False)
                arr = _normalize_slam_w2c(slam)
                self._session.slam_c2w = arr
                log.info("Loaded SLAM w2c from %s: %s", path.name, arr.shape)
                return
            except Exception as e:
                log.warning("Failed to load SLAM from %s: %s", path, e)

    def _current_cam_smooth_preset(self) -> str:
        """Return the active fallback camera smoothing preset."""
        try:
            settings = self._pipeline_dock.current_settings
            if hasattr(settings, "get_config"):
                config = settings.get_config()
                preset = getattr(config, "cam_smooth_preset", None)
                if preset in _CAM_SMOOTH_PRESETS:
                    return preset
        except Exception:
            pass
        return "moderate"

    def _refresh_all_panels(self):
        """Refresh all panels from current session state after results are loaded.

        Why: Multiple code paths need the same "populate everything" logic —
        startup restore, session load, pipeline finish, reprocess complete.
        Centralising it here prevents duplication and ensures nothing is missed.
        """
        if not self._session.person_tracks:
            return

        # Load SLAM camera trajectory for world-space rendering
        self._load_slam_if_needed()

        # Compute K_orig (original video intrinsics, GEM-X convention)
        s = self._session
        if s.K_orig is None and s.img_width > 0 and s.img_height > 0:
            focal = float(max(s.img_width, s.img_height))
            K = np.eye(3, dtype=np.float32)
            K[0, 0] = focal
            K[1, 1] = focal
            K[0, 2] = s.img_width / 2.0
            K[1, 2] = s.img_height / 2.0
            s.K_orig = K
            log.info("Computed K_orig (GEM-X convention): focal=%.0f", focal)

        # Populate track overview and markers
        self._populate_tracks()

        # Refresh person bar and identity inspector
        self._person_bar.refresh()
        self._identity_inspector.refresh()

        # Auto-select first person if none selected
        if self._session.selected_person < 0:
            first_pid = min(self._session.person_tracks.keys())
            self._session.selected_person = first_pid

        pid = self._session.selected_person
        self._person_bar.set_person(pid)
        self._identity_inspector.set_person(pid)
        self._mesh_viewport.set_person(pid)
        self._mesh_viewport.refresh()
        self._pose_corrector.set_person(pid)

        # Auto-switch to orbit camera when world-grounding data is available
        track = self._session.person_tracks.get(pid)
        if track is not None:
            params = track.soma_params or track.smplx_params
            if params is not None and "transl_world" in params:
                if self._mesh_viewport._camera_mode != "orbit":
                    self._sync_camera_mode("orbit")

        # Broadcast current frame to all panels
        frame = self._session.current_frame
        self._track_overview.set_current_frame(frame)
        self._identity_inspector.set_frame(frame)
        self._pose_corrector.on_frame_changed(frame)
        raw = self._video_player.get_raw_frame(frame)
        if raw is not None:
            self._mesh_viewport.set_video_frame(raw)
            self._mesh_viewport.on_frame_changed(frame)

        self._show_frame(frame)

        log.info(
            "Panels refreshed: %d person tracks, selected person %d",
            len(self._session.person_tracks), pid,
        )

    # ------------------------------------------------------------------
    # Track / person interaction (signal hub for multi mode)
    # ------------------------------------------------------------------

    def _on_track_clicked(self, person_id: int, frame_idx: int):
        """Select person and seek to frame from track overview."""
        self._person_bar.set_person(person_id)
        self._on_person_changed(person_id)
        self._video_player.seek(frame_idx)
        self.set_status(f"Selected Person {person_id} at frame {frame_idx}")

    def _apply_person_selection(self, person_id: int):
        """Update all UI panes for the selected person exactly once."""
        self._session.selected_person = person_id
        self._identity_inspector.set_person(person_id)
        self._pose_corrector.set_person(person_id)
        self._show_frame(self._session.current_frame)
        self.set_status(f"Selected Person {person_id}")

    def _on_person_changed(self, person_id: int):
        """Handle person change from PersonSelectorBar."""
        self._apply_person_selection(person_id)

    def _on_identity_person_changed(self, person_id: int):
        """Handle person change from identity inspector (issue navigation etc.)."""
        self._person_bar.set_person(person_id)
        self._apply_person_selection(person_id)

    def _on_bbox_overlay_changed(self, data: object):
        """Handle overlay changes (show-all-tracks toggle, edit preview)."""
        if isinstance(data, dict):
            if "show_all" in data:
                self._show_all_tracks = data["show_all"]
            if "edit_preview" in data:
                self._edit_preview = data["edit_preview"]
        self._show_frame(self._session.current_frame)

    def _on_keyframe_changed(self, person_id: int, frame_idx: int):
        """Redraw overlay when keyframes change."""
        self._show_frame(self._session.current_frame)

    def _on_tracks_modified(self):
        """Handle track modifications (swap, split, merge)."""
        self._populate_tracks()
        self._person_bar.refresh()
        self._identity_inspector.refresh()
        self._show_frame(self._session.current_frame)

    def _on_joint_clicked_auto_raise(self, joint_idx: int):
        """Auto-raise PoseCorrector tab when a joint is clicked in 3D viewport."""
        if joint_idx >= 0 and self._pipeline_dock.current_mode == "multi":
            self._person_panel_dock.setVisible(True)
            self._person_panel_dock.raise_()
            self._person_panel_dock.tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Pipeline output handling
    # ------------------------------------------------------------------

    def _on_pipeline_output(self, result: dict):
        """Handle single/perf pipeline completion — load output preview and switch workspace."""
        output_dir = result.get("output_dir")
        if output_dir:
            self._session.output_dir = Path(output_dir)
        loaded = False
        for key in ("side_by_side", "incam"):
            path = result.get(key)
            if path and Path(path).is_file():
                self._load_output_preview(Path(path))
                loaded = True
                break
        # Auto-switch to Review workspace if we loaded results
        if loaded or result.get("soma_params") or result.get("merged_pt"):
            self._apply_preset("Review")

    def _load_output_preview(self, video_path: Path):
        """Load an output video into the shared VideoPlayer."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        self._video_player.set_video(video_path, num_frames, fps)

    def _on_multi_pipeline_finished(self, result: dict):
        """Handle multi pipeline completion — load tracks, auto-detect issues, switch workspace."""
        output_dir = result.get("output_dir")
        if output_dir:
            self._session.output_dir = Path(output_dir)

        multi_result = result.get("result")
        if multi_result is not None:
            self._load_person_tracks_from_result(multi_result)

        self._session.current_frame = 0
        self._refresh_all_panels()
        self._video_player.seek(0)

        # Auto-detect pose issues and switch to Correction workspace
        if self._session.person_tracks:
            try:
                self._pose_corrector.run_auto_detection()
            except Exception as exc:
                log.warning("Auto-detection failed: %s", exc)
            self._apply_preset("Correction")

    # ------------------------------------------------------------------
    # Reprocess
    # ------------------------------------------------------------------

    def _on_reprocess_requested(self, person_ids: list):
        """Launch ReprocessWorker for dirty persons."""
        if self._reprocess_worker is not None:
            self.set_status("Reprocess already running")
            return

        self._reprocess_worker = ReprocessWorker(
            session=self._session,
            person_ids=person_ids,
        )
        self._reprocess_worker.progress.connect(self._on_reprocess_progress)
        self._reprocess_worker.person_done.connect(self._on_reprocess_person_done)
        self._reprocess_worker.finished.connect(self._on_reprocess_finished)
        self._reprocess_worker.error.connect(self._on_reprocess_error)
        self._reprocess_worker.log_line.connect(
            lambda msg: self._log_panel.append_line(msg, "info")
        )
        self._reprocess_worker.start()

        self._multi_settings._progress_bar.show()
        self._multi_settings._progress_label.show()
        self.set_status(f"Reprocessing {len(person_ids)} person(s)...")
        self._log_panel.append_line(
            f"Reprocess started for persons: {person_ids}", "info",
        )

    def _on_reprocess_progress(self, fraction: float, stage: str):
        self._multi_settings._progress_bar.setValue(int(fraction * 1000))
        self._multi_settings._progress_label.setText(stage)
        self.set_status(f"{stage} ({fraction:.0%})")

    def _on_reprocess_person_done(self, person_id: int):
        self._session.dirty_persons.discard(person_id)
        self._identity_inspector.update_reprocess_button()
        self._log_panel.append_line(f"Person {person_id} reprocessed", "info")

    def _on_reprocess_finished(self, result: dict):
        if self._reprocess_worker is not None:
            self._reprocess_worker.wait()
            self._reprocess_worker = None
        self._multi_settings._progress_bar.hide()
        self._multi_settings._progress_label.hide()
        self._multi_settings._progress_bar.setValue(0)

        reprocessed = result.get("reprocessed", [])
        self._session.dirty_persons -= set(reprocessed)

        # Reload motion params, confidences, and bboxes from disk for
        # reprocessed persons — the old data in memory is stale.
        for pid in reprocessed:
            track = self._session.person_tracks.get(pid)
            if track is None or track.person_dir is None:
                continue
            # Clear stale params so _hydrate reloads from new output
            track.smplx_params = None
            track.soma_params = None
            track.motion_sources = None
            track.confidences = None
            track.confidence_breakdown = None
        self._hydrate_person_tracks()

        # Update all panels with new data
        self._populate_tracks()
        self._person_bar.refresh()
        self._identity_inspector.refresh()
        pid = self._session.selected_person
        self._person_bar.set_person(pid)
        self._mesh_viewport.set_person(-1)  # force re-select
        self._mesh_viewport.set_person(pid)
        self._show_frame(self._session.current_frame)

        self.set_status(f"Reprocess complete: {len(reprocessed)} person(s) updated")
        self._log_panel.append_line(f"Reprocess finished: {reprocessed}", "info")

    def _on_reprocess_error(self, message: str):
        if self._reprocess_worker is not None:
            self._reprocess_worker.wait()
            self._reprocess_worker = None
        self._multi_settings._progress_bar.hide()
        self._multi_settings._progress_label.hide()
        self._multi_settings._progress_bar.setValue(0)

        self.set_status(f"Reprocess error: {message}")
        self._log_panel.append_line(f"Reprocess error: {message}", "error")

    # ------------------------------------------------------------------
    # Person track loading (from multi pipeline results)
    # ------------------------------------------------------------------

    def _load_person_tracks_from_result(self, multi_result):
        """Convert MultiPersonResult into session.person_tracks."""
        from models.session import PersonTrack

        self._session.person_tracks.clear()
        self._session.inactive_tracks.clear()
        self._session.derived_c2w = None
        self._session.raw_c2w = None

        all_tracks = getattr(multi_result, "all_tracks", [])
        person_dirs = getattr(multi_result, "person_dirs", [])
        identity_tracks = getattr(multi_result, "identity_tracks", [])

        for i, track in enumerate(all_tracks):
            tid = track.get("track_id", i)
            bboxes_raw = track["bbx_xyxy"]
            if hasattr(bboxes_raw, "numpy"):
                bboxes = bboxes_raw.cpu().numpy()
            else:
                bboxes = np.asarray(bboxes_raw)

            person_dir = Path(person_dirs[i]) if i < len(person_dirs) else None
            id_track = identity_tracks[i] if i < len(identity_tracks) else None

            keyframes = []
            if id_track and hasattr(id_track, "keyframes"):
                for kf in id_track.keyframes:
                    keyframes.append({
                        "frame": kf.frame_index,
                        "verified": kf.verified,
                        "confidence": getattr(kf, "confidence", None),
                    })

            confidences = None
            confidence_breakdown = None
            if person_dir:
                confidences, confidence_breakdown = self._load_confidences_csv(
                    person_dir
                )

            smplx_params = None
            soma_params = None
            body_model_type = "smplx"
            if person_dir:
                smplx_params, soma_params, body_model_type, motion_sources = self._load_motion_params(person_dir)

            # Use GEM-X embedded confidences if no CSV exists
            if confidences is None:
                _p = soma_params or smplx_params
                if _p is not None:
                    gemx_conf = _p.get("confidences")
                    if gemx_conf is not None:
                        confidences = gemx_conf.tolist()

            pt = PersonTrack(
                person_id=tid,
                person_dir=person_dir,
                identity_track=id_track,
                confidences=confidences,
                bboxes=bboxes,
                keyframes=keyframes,
                confidence_breakdown=confidence_breakdown,
                smplx_params=smplx_params,
                soma_params=soma_params,
                body_model_type=body_model_type,
                motion_sources=motion_sources,
            )
            self._load_crop_metadata(pt)
            self._session.person_tracks[tid] = pt

        # Mark inactive tracks
        for track in getattr(multi_result, "inactive_tracks", []):
            tid = track.get("track_id", -1) if isinstance(track, dict) else -1
            if tid >= 0:
                self._session.inactive_tracks.add(tid)

        # Load crossing spans from person dirs
        for pid, pt in self._session.person_tracks.items():
            if pt.person_dir:
                spans_path = pt.person_dir / "crossing_spans.json"
                if spans_path.is_file():
                    import json
                    try:
                        spans = json.loads(spans_path.read_text())
                        self._session.crossing_spans[pid] = [
                            tuple(s) for s in spans
                        ]
                    except Exception:
                        pass

    def _load_motion_params(
        self, person_dir: Path
    ) -> tuple[dict | None, dict | None, str, dict[str, dict]]:
        """Load motion params, preferring the richest viewport/export bundle."""
        preferred_backend = "gemx"
        try:
            settings = self._pipeline_dock.current_settings
            backend, _ = settings._selected_backend()
            preferred_backend = backend
        except Exception:
            pass

        candidates: list[tuple[int, int, str, tuple[dict | None, dict | None, str, dict[str, dict]]]] = []
        for backend_name, loader in (
            ("gemx", self._try_load_gemx),
            ("gvhmr", self._try_load_gvhmr),
        ):
            result = loader(person_dir)
            if result is None:
                continue
            score = self._motion_bundle_score(*result)
            preferred = 1 if backend_name == preferred_backend else 0
            candidates.append((score, preferred, backend_name, result))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            chosen = candidates[0]
            if len(candidates) > 1:
                runner_up = candidates[1]
                log.info(
                    "Selected %s motion bundle for %s (score=%d, runner_up=%s:%d)",
                    chosen[2],
                    person_dir,
                    chosen[0],
                    runner_up[2],
                    runner_up[0],
                )
            result = chosen[3]
            # Convert HaMeR wrist_orient (camera-space global) → local for
            # every motion source *and* the default smplx_params, using the
            # camera_baseline FK chain.
            smplx_params, _, _, motion_sources = result
            cam = motion_sources.get("camera_baseline")
            cam_go = (
                np.asarray(cam["global_orient"], dtype=np.float32)
                if cam is not None and "global_orient" in cam
                else None
            )
            for src_params in motion_sources.values():
                _convert_wrist_orient_to_local(src_params, camera_global_orient=cam_go)
            if smplx_params is not None:
                _convert_wrist_orient_to_local(smplx_params, camera_global_orient=cam_go)
            return result
        return None, None, "smplx", {}

    def _motion_bundle_score(
        self,
        smplx_params: dict | None,
        soma_params: dict | None,
        body_model_type: str,
        motion_sources: dict[str, dict],
    ) -> int:
        score = 0
        if body_model_type == "smplx":
            score += 10
        elif body_model_type == "soma":
            score += 5

        if smplx_params is not None:
            if smplx_params.get("betas") is not None:
                score += 40
            if (
                smplx_params.get("global_orient_world") is not None
                and smplx_params.get("transl_world") is not None
            ):
                score += 20
            for key in (
                "left_hand_pose",
                "right_hand_pose",
                "left_wrist_orient",
                "right_wrist_orient",
            ):
                if smplx_params.get(key) is not None:
                    score += 5

        if soma_params is not None and soma_params.get("transl_world") is not None:
            score += 20

        if motion_sources.get("camera_baseline") is not None:
            score += 10
        if motion_sources.get("world_baseline") is not None:
            score += 30
        if motion_sources.get("world_physics") is not None:
            score += 50

        return score

    def _build_gemx_motion_sources(self, gemx_result: dict) -> dict[str, dict]:
        if not gemx_result:
            return {}

        def _clone_params(params: dict) -> dict:
            cloned: dict = {}
            for key, value in params.items():
                if isinstance(value, np.ndarray):
                    cloned[key] = np.array(value, copy=True)
                else:
                    cloned[key] = value
            return cloned

        camera_params = _clone_params(gemx_result)
        motion_sources: dict[str, dict] = {"camera_baseline": camera_params}
        if (
            gemx_result.get("global_orient_world") is not None
            and gemx_result.get("transl_world") is not None
        ):
            world_params = _clone_params(gemx_result)
            world_params["global_orient"] = np.array(
                gemx_result["global_orient_world"], dtype=np.float32
            )
            world_params["transl"] = np.array(
                gemx_result["transl_world"], dtype=np.float32
            )
            if gemx_result.get("body_pose_world") is not None:
                world_params["body_pose"] = np.array(
                    gemx_result["body_pose_world"], dtype=np.float32
                )
            motion_sources["world_baseline"] = world_params
        return motion_sources

    def _try_load_gemx(
        self, person_dir: Path
    ) -> tuple[dict | None, dict | None, str, dict[str, dict]] | None:
        """Try loading GEM-X results from person_dir. Returns None on failure."""
        try:
            from workers.gemx_worker import load_gemx_soma_output
            for search_dir in [person_dir, person_dir / "gemx_demo"]:
                if not search_dir.is_dir():
                    continue
                gemx_result = load_gemx_soma_output(search_dir)
                if gemx_result is not None:
                    # Log hand joint data quality
                    if "poses" in gemx_result and gemx_result["poses"].shape[1] > 21:
                        poses = gemx_result["poses"]
                        hand_joints = np.concatenate(
                            [poses[:, 15:39], poses[:, 43:67]], axis=1,
                        )
                        nz = np.count_nonzero(hand_joints)
                        tot = hand_joints.size
                        self._log_panel.append_line(
                            f"GEM-X hand joints: nonzero={nz}/{tot} ({100*nz/max(tot,1):.0f}%)",
                            "info",
                        )
                    bmt = gemx_result.pop("body_model_type", "soma")
                    if bmt == "smplx":
                        motion_sources = self._build_gemx_motion_sources(gemx_result)
                        return gemx_result, None, "smplx", motion_sources
                    else:
                        return None, gemx_result, "soma", {}
        except Exception:
            log.warning("GEM-X load failed for %s:\n%s", person_dir,
                        __import__("traceback").format_exc())
        return None

    def _try_load_gvhmr(
        self, person_dir: Path
    ) -> tuple[dict | None, dict | None, str, dict[str, dict]] | None:
        """Try loading GVHMR results from person_dir. Returns None on failure."""
        smplx_params, motion_sources = self._load_smplx_sources_legacy(person_dir)
        if smplx_params is not None:
            return smplx_params, None, "smplx", motion_sources
        return None

    def _select_hybrid_motion_file(self, person_dir: Path) -> Path | None:
        hybrid_pts = list(person_dir.glob("*_hybrid_smplx.pt"))
        if not hybrid_pts:
            return None

        # Read current pipeline settings to avoid loading stale refined files
        # when those refinement features are now disabled.
        want_spring = False
        want_pin = False
        try:
            cfg = self._pipeline_dock.current_settings.get_config()
            want_spring = cfg.use_spring_refine
            want_pin = cfg.use_foot_pin
        except Exception:
            pass

        try:
            import torch

            scored: list[tuple[int, int, Path]] = []
            for path in hybrid_pts:
                score = 0
                try:
                    data = torch.load(str(path), map_location="cpu", weights_only=False)
                    source_tag = data.get("source", "")

                    # Only boost refined files when those features are enabled
                    if source_tag == "spring_refined_pinned" and want_spring and want_pin:
                        score += 100
                    elif source_tag == "spring_refined" and want_spring:
                        score += 100
                    elif source_tag == "phc_refined":
                        score += 100

                    if all(
                        key in data
                        for key in (
                            "global_orient_world_physics",
                            "body_pose_world_physics",
                            "transl_world_physics",
                        )
                    ):
                        score += 50
                    if all(
                        key in data
                        for key in (
                            "global_orient_world_spring",
                            "body_pose_world_spring",
                            "transl_world_spring",
                        )
                    ):
                        score += 50
                    if all(
                        key in data
                        for key in (
                            "global_orient_world_baseline",
                            "body_pose_world_baseline",
                            "transl_world_baseline",
                        )
                    ):
                        score += 25
                except Exception:
                    pass
                scored.append((score, int(path.stat().st_mtime_ns), path))
            scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return scored[0][2]
        except Exception:
            return max(hybrid_pts, key=lambda p: p.stat().st_mtime_ns)

    def _load_smplx_sources_legacy(self, person_dir: Path) -> tuple[dict | None, dict[str, dict]]:
        """Load GVHMR/Hybrid SMPL-X params and preserve multiple viewport sources."""
        import torch

        hmr4d_pt = person_dir / "demo" / "isolated_video" / "hmr4d_results.pt"
        hybrid_pt = self._select_hybrid_motion_file(person_dir)
        if not hmr4d_pt.is_file() and hybrid_pt is None:
            return None, {}
        try:
            results = (
                torch.load(hmr4d_pt, map_location="cpu", weights_only=False)
                if hmr4d_pt.is_file()
                else {}
            )
            hybrid_data = (
                torch.load(str(hybrid_pt), map_location="cpu", weights_only=False)
                if hybrid_pt is not None
                else {}
            )

            def _to_numpy(value):
                if value is None:
                    return None
                if hasattr(value, "detach"):
                    value = value.detach()
                if hasattr(value, "cpu"):
                    value = value.cpu()
                if hasattr(value, "numpy"):
                    value = value.numpy()
                return np.array(value)

            def _copy_motion_dict(src: dict, orient_key: str, body_key: str, transl_key: str) -> dict | None:
                if not src or orient_key not in src or body_key not in src:
                    return None
                params_dict = {
                    "global_orient": _to_numpy(src.get(orient_key)).astype(np.float32),
                    "body_pose": _to_numpy(src.get(body_key)).astype(np.float32),
                }
                transl = _to_numpy(src.get(transl_key))
                if transl is not None:
                    params_dict["transl"] = transl.astype(np.float32)
                return params_dict

            def _extract_optional(*sources: dict, key: str) -> np.ndarray | None:
                for src in sources:
                    if not src or key not in src:
                        continue
                    arr = _to_numpy(src[key])
                    if arr is not None:
                        return arr.astype(np.float32)
                return None

            def _clone_params(src: dict | None) -> dict | None:
                if not src:
                    return None
                out = {}
                for key, value in src.items():
                    arr = _to_numpy(value)
                    out[key] = arr.astype(np.float32) if arr is not None else value
                return out

            def _normalize_body_pose(params: dict):
                bp = params.get("body_pose")
                if bp is None:
                    return
                params["body_pose"] = np.array(bp, dtype=np.float32)
                if params["body_pose"].ndim == 2 and params["body_pose"].shape[-1] != 3:
                    params["body_pose"] = params["body_pose"].reshape(
                        params["body_pose"].shape[0], -1, 3
                    )

            results_source = str(results.get("source") or "")
            camera_results = _clone_params(results.get("smpl_params_incam"))
            world_results = _clone_params(results.get("smpl_params_global"))
            camera_results_flat = _copy_motion_dict(
                results, "global_orient_cam", "body_pose_cam", "transl_cam"
            )
            results_world_baseline = _copy_motion_dict(
                results,
                "global_orient_world_baseline",
                "body_pose_world_baseline",
                "transl_world_baseline",
            )
            results_world_physics = _copy_motion_dict(
                results,
                "global_orient_world_physics",
                "body_pose_world_physics",
                "transl_world_physics",
            )
            results_world_generic = _copy_motion_dict(
                results, "global_orient_world", "body_pose_world", "transl_world"
            )
            results_refined_generic = _copy_motion_dict(
                results, "global_orient", "body_pose", "transl"
            )
            camera_hybrid = _copy_motion_dict(
                hybrid_data, "global_orient_cam", "body_pose_cam", "transl_cam"
            )
            world_baseline = (
                _copy_motion_dict(
                    hybrid_data,
                    "global_orient_world_baseline",
                    "body_pose_world_baseline",
                    "transl_world_baseline",
                )
                or _copy_motion_dict(
                    hybrid_data, "global_orient_world", "body_pose_world", "transl_world"
                )
                or results_world_baseline
                or (
                    results_world_generic
                    if results_source != "phc_refined"
                    else None
                )
                or world_results
            )
            world_physics = (
                _copy_motion_dict(
                    hybrid_data,
                    "global_orient_world_physics",
                    "body_pose_world_physics",
                    "transl_world_physics",
                )
            )
            if world_physics is None and hybrid_data.get("source") in (
                "phc_refined",
                "spring_refined",
                "spring_refined_pinned",
            ):
                world_physics = _copy_motion_dict(
                    hybrid_data, "global_orient_world", "body_pose_world", "transl_world"
                )
            if world_physics is None:
                world_physics = results_world_physics
            if world_physics is None and results_source in (
                "phc_refined",
                "spring_refined",
                "spring_refined_pinned",
            ):
                world_physics = results_world_generic or results_refined_generic

            camera_baseline = camera_hybrid or camera_results_flat or camera_results
            if camera_baseline is None and hybrid_data:
                camera_baseline = _copy_motion_dict(
                    hybrid_data, "global_orient", "body_pose", "transl"
                )
            if camera_baseline is None or "body_pose" not in camera_baseline:
                return None, {}

            K = hybrid_data.get("K_fullimg", results.get("K_fullimg"))
            if K is not None and self._session.camera_K is None:
                from views.mesh_viewport import estimate_K as _estimate_K

                self._session.camera_K = _estimate_K(
                    self._session.img_width or 1920,
                    self._session.img_height or 1080,
                )
                fov = self._mesh_viewport.get_frustum_fov()
                self._mesh_dock._fov_spin.blockSignals(True)
                self._mesh_dock._fov_spin.setValue(fov)
                self._mesh_dock._fov_spin.blockSignals(False)

            body_pose_raw = _to_numpy(results.get("raw_body_pose"))
            if body_pose_raw is not None:
                body_pose_raw = body_pose_raw.astype(np.float32)
                if body_pose_raw.ndim == 2 and body_pose_raw.shape[-1] != 3:
                    body_pose_raw = body_pose_raw.reshape(body_pose_raw.shape[0], -1, 3)

            wrist_and_hand = {}
            for key in [
                "left_hand_pose",
                "right_hand_pose",
                "left_wrist_orient",
                "right_wrist_orient",
            ]:
                arr = _extract_optional(hybrid_data, results, key=key)
                if arr is not None:
                    wrist_and_hand[key] = arr
            wrist_source = (
                hybrid_pt.name
                if any(k in hybrid_data for k in ("left_wrist_orient", "right_wrist_orient"))
                else hmr4d_pt.name if any(k in results for k in ("left_wrist_orient", "right_wrist_orient"))
                else "none"
            )

            if "left_hand_pose" in wrist_and_hand:
                lh = wrist_and_hand["left_hand_pose"]
                msg = (
                    f"Hand data ({wrist_source}): shape={lh.shape}, "
                    f"nonzero={np.count_nonzero(lh)}/{lh.size}"
                )
                self._log_panel.append_line(msg, "info")
            else:
                self._log_panel.append_line(
                    f"No hand data in {person_dir.name} — hands will be rest pose",
                    "warning",
                )

            betas = _extract_optional(camera_hybrid or {}, camera_results or {}, hybrid_data, key="betas")
            if betas is None and camera_results is not None:
                betas = _to_numpy(camera_results.get("betas"))
                if betas is not None:
                    betas = betas.astype(np.float32)

            if camera_results is None and "betas" in results:
                camera_results = camera_results or {}
                camera_results["betas"] = _to_numpy(results["betas"]).astype(np.float32)
            if hybrid_data and "betas" in hybrid_data and camera_hybrid is not None:
                camera_hybrid["betas"] = _to_numpy(hybrid_data["betas"]).astype(np.float32)
            if world_results is not None and "betas" not in world_results and betas is not None:
                world_results["betas"] = np.array(betas, dtype=np.float32)
            if world_baseline is not None and "betas" not in world_baseline and betas is not None:
                world_baseline["betas"] = np.array(betas, dtype=np.float32)
            if world_physics is not None and "betas" not in world_physics and betas is not None:
                world_physics["betas"] = np.array(betas, dtype=np.float32)

            baseline_artifact = (
                hybrid_pt.name
                if world_baseline is not None and hybrid_pt is not None
                else hmr4d_pt.name if world_baseline is not None and hmr4d_pt.is_file()
                else None
            )
            physics_artifact = (
                hybrid_pt.name
                if world_physics is not None and hybrid_pt is not None
                else hmr4d_pt.name if world_physics is not None and hmr4d_pt.is_file()
                else None
            )

            def _augment_source(
                base_params: dict | None,
                *,
                attach_world: dict | None = None,
                include_raw_pose: bool = False,
                add_hand_mean: bool = False,
                motion_contract: str | None = None,
                motion_artifact: str | None = None,
            ) -> dict | None:
                if base_params is None:
                    return None
                params = _clone_params(base_params) or {}
                if K is not None:
                    params["K_fullimg"] = _to_numpy(K).astype(np.float32)
                if betas is not None and "betas" not in params:
                    params["betas"] = np.array(betas, dtype=np.float32)
                if include_raw_pose and body_pose_raw is not None:
                    params["body_pose_raw"] = body_pose_raw.copy()
                for key, value in wrist_and_hand.items():
                    params[key] = np.array(value, dtype=np.float32)
                params["wrist_debug_source"] = wrist_source
                _normalize_body_pose(params)
                if add_hand_mean and "left_hand_pose" in params:
                    try:
                        from smplx_to_bvh import _add_hand_mean_pose

                        params = _add_hand_mean_pose(params)
                    except Exception:
                        log.debug("Hand mean pose not applied (smplx model unavailable)")
                _smooth_hand_poses(params)
                _amplify_ankle_rotations(params, gain=_DEFAULT_ANKLE_GAIN)
                if attach_world is not None:
                    params["global_orient_world"] = np.array(
                        attach_world["global_orient"], dtype=np.float32
                    )
                    params["body_pose_world"] = np.array(
                        attach_world["body_pose"], dtype=np.float32
                    )
                    params["transl_world"] = np.array(
                        attach_world["transl"], dtype=np.float32
                    )
                if motion_contract is not None:
                    params["motion_contract"] = motion_contract
                if motion_artifact is not None:
                    params["motion_artifact"] = motion_artifact
                if results_source:
                    params["source"] = results_source
                return params

            def _normalize_world_source(source_params: dict | None) -> dict | None:
                if source_params is None:
                    return None
                params = _clone_params(source_params) or {}
                _normalize_body_pose(params)
                tr_world = np.array(params["transl"], dtype=np.float32)
                if tr_world.shape[0] > 0:
                    floor_y = float(tr_world[0, 1]) - 0.933
                    tr_world[:, 1] -= floor_y
                offset_cam = self._get_person_world_offset(person_dir)
                if offset_cam is not None and camera_baseline is not None:
                    from scipy.spatial.transform import Rotation

                    go_incam_0 = np.array(camera_baseline["global_orient"][0])
                    go_world_0 = np.array(params["global_orient"][0])
                    R_incam = Rotation.from_rotvec(go_incam_0).as_matrix()
                    R_world = Rotation.from_rotvec(go_world_0).as_matrix()
                    R_c2w = R_world @ R_incam.T
                    offset_3d = np.array(
                        [offset_cam[0], 0.0, offset_cam[2] if len(offset_cam) > 2 else 0.0],
                        dtype=np.float32,
                    )
                    off_w = R_c2w @ offset_3d
                    tr_world[:, 0] += off_w[0]
                    tr_world[:, 2] += off_w[2]
                params["transl"] = tr_world.astype(np.float32)
                return params

            world_baseline = _normalize_world_source(world_baseline)
            world_physics = _normalize_world_source(world_physics)

            world_anchor = world_baseline or world_physics

            camera_params = _augment_source(
                camera_baseline,
                attach_world=world_anchor,
                include_raw_pose=True,
                add_hand_mean=True,
                motion_contract="camera_baseline",
                motion_artifact=(
                    hybrid_pt.name if camera_hybrid is not None and hybrid_pt is not None
                    else hmr4d_pt.name if hmr4d_pt.is_file()
                    else None
                ),
            )
            world_baseline_params = _augment_source(
                world_baseline,
                add_hand_mean=True,
                motion_contract="world_baseline",
                motion_artifact=baseline_artifact,
            )
            world_physics_params = _augment_source(
                world_physics,
                add_hand_mean=True,
                motion_contract=(
                    "refined_hmr4d"
                    if physics_artifact == hmr4d_pt.name and results_source == "phc_refined"
                    else "world_physics"
                ),
                motion_artifact=physics_artifact,
            )

            if (
                self._session.derived_c2w is None
                and camera_params is not None
                and (world_baseline_params is not None or world_physics_params is not None)
            ):
                go_incam = np.array(camera_params["global_orient"]).astype(np.float32)
                tr_incam = np.array(camera_params["transl"]).astype(np.float32)
                align_world = world_baseline_params or world_physics_params
                go_world = np.array(align_world["global_orient"]).astype(np.float32)
                tr_world = np.array(align_world["transl"]).astype(np.float32)
                c2w_body = _compute_camera_c2w(go_incam, tr_incam, go_world, tr_world)
                self._session.raw_c2w = c2w_body.copy()
                self._load_slam_if_needed()
                if self._session.slam_c2w is not None:
                    self._session.derived_c2w = _align_slam_to_world(
                        self._session.slam_c2w, c2w_body
                    )
                    log.info("Using SLAM-aligned camera trajectory")
                else:
                    self._session.derived_c2w = _smooth_c2w(
                        c2w_body,
                        fps=self._session.fps,
                        preset=self._current_cam_smooth_preset(),
                    )
                    log.info("Using smoothed body-derived camera (no SLAM)")

            motion_sources: dict[str, dict] = {}
            if camera_params is not None:
                motion_sources["camera_baseline"] = camera_params
            if world_baseline_params is not None:
                motion_sources["world_baseline"] = world_baseline_params
            if world_physics_params is not None:
                motion_sources["world_physics"] = world_physics_params

            default_params = dict(camera_params) if camera_params is not None else None
            if default_params is None and world_baseline_params is not None:
                default_params = dict(world_baseline_params)
            if default_params is None and world_physics_params is not None:
                default_params = dict(world_physics_params)
            if default_params is None:
                return None, motion_sources
            if world_physics_params is not None:
                default_params["global_orient_world_physics"] = np.array(
                    world_physics_params["global_orient"], dtype=np.float32
                )
                default_params["body_pose_world_physics"] = np.array(
                    world_physics_params["body_pose"], dtype=np.float32
                )
                default_params["transl_world_physics"] = np.array(
                    world_physics_params["transl"], dtype=np.float32
                )
            return default_params, motion_sources
        except Exception:
            log.warning(
                "Legacy SMPL-X load failed for %s:\n%s",
                person_dir,
                __import__("traceback").format_exc(),
            )
        return None, {}

    def _load_smplx_params_legacy(self, person_dir: Path) -> dict | None:
        """Backward-compatible wrapper returning the default SMPL-X params."""
        params, _sources = self._load_smplx_sources_legacy(person_dir)
        return params

    def _get_person_world_offset(self, person_dir: Path):
        """Return [x, y, z] world offset for this person from assembly data."""
        import json as _json
        try:
            meta_path = person_dir / "person_meta.json"
            if not meta_path.is_file():
                return None
            track_id = str(_json.loads(meta_path.read_text()).get("track_id"))
            offsets_path = person_dir.parent / "assembly" / "person_offsets.json"
            if not offsets_path.is_file():
                return None
            data = _json.loads(offsets_path.read_text())
            # Support both {"offsets": {...}} and flat {track_id: [...]} formats
            offsets = data.get("offsets", data) if isinstance(data, dict) else data
            off = offsets.get(track_id)
            if off is not None and len(off) >= 3:
                return [float(off[0]), float(off[1]), float(off[2])]
        except Exception:
            pass
        return None

    def _load_confidences_csv(self, person_dir: Path):
        """Load confidence.csv -> (overall_list, breakdown_dict)."""
        import csv

        csv_path = person_dir / "confidence.csv"
        if not csv_path.is_file():
            return None, None

        overall = []
        breakdown = {
            m: [] for m in [
                "detection", "visibility", "overlap",
                "shape", "motion", "overall",
            ]
        }
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                overall.append(float(row["overall"]))
                breakdown["detection"].append(float(row["detection"]))
                breakdown["visibility"].append(float(row["visible_kp"]))
                breakdown["overlap"].append(float(row["bbox_overlap"]))
                breakdown["shape"].append(float(row["shape_dist"]))
                breakdown["motion"].append(float(row["motion_dist"]))
                breakdown["overall"].append(float(row["overall"]))
        return overall if overall else None, breakdown if overall else None

    def _populate_tracks(self):
        """Populate track overview from session person_tracks."""
        if not self._session.person_tracks:
            return

        tracks: dict[int, np.ndarray] = {}
        for pid, track in self._session.person_tracks.items():
            if track.confidences is not None:
                tracks[pid] = np.array(track.confidences)
            else:
                tracks[pid] = np.ones(max(1, self._session.num_frames)) * 0.8
        self._track_overview.set_tracks(tracks)
        self._populate_track_markers()
        self._person_bar.refresh()
        self._pose_corrector.refresh()

    def _populate_track_markers(self):
        """Push keyframe, issue, correction, and crossing markers to track overview."""
        for pid, track in self._session.person_tracks.items():
            kf_frames = [kf["frame"] for kf in (track.keyframes or [])]
            verified = {
                kf["frame"] for kf in (track.keyframes or [])
                if kf.get("verified", False)
            }
            corr_frames: list[int] = []
            interp_spans: list[tuple[int, int]] = []
            ct = self._session.correction_tracks.get(pid)
            if ct is not None and hasattr(ct, "corrections") and ct.corrections:
                corr_frames = [c.frame_index for c in ct.corrections]
                # Compute interpolation spans between adjacent corrections
                sorted_corrs = sorted(ct.corrections, key=lambda c: c.frame_index)
                for i in range(len(sorted_corrs) - 1):
                    f_s = sorted_corrs[i].frame_index
                    f_e = sorted_corrs[i + 1].frame_index
                    if f_e - f_s > 1:
                        interp_spans.append((f_s, f_e))
            spans = self._session.crossing_spans.get(pid, [])
            self._track_overview.set_track_markers(
                pid,
                keyframes=kf_frames,
                verified_frames=verified,
                correction_frames=corr_frames,
                crossing_spans=spans,
                interpolation_spans=interp_spans,
            )
        # Issue flags from review scanner
        try:
            from views.identity_inspector import compute_review_issues

            issues = compute_review_issues(self._session)
            by_person: dict[int, list[int]] = {}
            for issue in issues:
                by_person.setdefault(issue.person_id, []).append(issue.frame)
            for pid, frames in by_person.items():
                self._track_overview.set_track_markers(pid, issue_frames=frames)
        except Exception:
            pass

        # Drift severity spans from VLM drift corrections
        self._load_drift_spans_if_available()

    def _load_drift_spans_if_available(self):
        """Auto-load drift spans from drift_corrections.json.

        Supports two JSON layouts:
        - **drift_spans** dict (keyed by person index): explicit frame ranges.
        - **persons** list with ``anchors``: derive spans between consecutive
          anchors whose XZ correction magnitude exceeds *threshold*.
        """
        if self._session.output_dir is None:
            return
        corr_path = self._session.output_dir / "drift_corrections.json"
        if not corr_path.is_file():
            return
        try:
            import json as _json

            with open(corr_path) as fh:
                corr_data = _json.load(fh)

            # Remap person_index → track_id via session_manifest
            manifest_path = self._session.output_dir / "session_manifest.json"
            bindings: list[dict] = []
            if manifest_path.is_file():
                with open(manifest_path) as mf:
                    bindings = _json.load(mf).get("person_bindings", [])

            def _pid_to_tid(pidx: int) -> int:
                if bindings and pidx < len(bindings):
                    return int(bindings[pidx]["track_id"])
                return pidx

            n_loaded = 0

            # --- Format 1: explicit drift_spans dict ---
            explicit = corr_data.get("drift_spans", {})
            if explicit:
                for pidx_str, spans in explicit.items():
                    tid = _pid_to_tid(int(pidx_str))
                    frame_spans = [(int(s[0]), int(s[1])) for s in spans]
                    self._track_overview.set_track_markers(
                        tid, drift_correction_spans=frame_spans,
                    )
                    n_loaded += len(frame_spans)

            # --- Format 2: derive from VLM anchors ---
            threshold_m = 0.15
            for person in corr_data.get("persons", []):
                anchors = person.get("anchors", [])
                if len(anchors) < 2:
                    continue
                pidx = int(person.get("person_id", 0))
                tid = _pid_to_tid(pidx)
                spans: list[tuple[int, int]] = []
                for i in range(len(anchors) - 1):
                    a_cur = anchors[i]
                    a_nxt = anchors[i + 1]
                    t = a_nxt["target"]
                    mag_xz = (t[0] ** 2 + t[2] ** 2) ** 0.5
                    if mag_xz >= threshold_m:
                        spans.append((int(a_cur["frame"]), int(a_nxt["frame"])))
                if spans:
                    self._track_overview.set_track_markers(
                        tid, drift_correction_spans=spans,
                    )
                    n_loaded += len(spans)

            if n_loaded:
                log.info(
                    "Loaded %d drift spans from %s", n_loaded, corr_path.name,
                )
        except Exception:
            log.debug("Failed to load drift spans", exc_info=True)

    # ------------------------------------------------------------------
    # Open video
    # ------------------------------------------------------------------

    def _on_open_video(self):
        """Open Video menu action — load video into the active settings panel."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv);;All Files (*)",
        )
        if path:
            settings = self._pipeline_dock.current_settings
            if hasattr(settings, "_load_video"):
                settings._load_video(path)

    @property
    def session(self) -> Session:
        return self._session

    @property
    def log_panel(self) -> LogPanel:
        return self._log_panel

    def set_status(self, text: str):
        self._status_label.setText(text)

    def set_frame_info(self, current: int, total: int):
        self._frame_label.setText(f"Frame {current} / {total}")

    def set_fps_info(self, fps: float):
        self._fps_label.setText(f"{fps:.1f} FPS")

    # --- Undo / Redo ---

    def _setup_undo_redo(self):
        """Wire undo stack on_changed callback to keep Edit menu in sync."""
        self._session.undo_stack.on_changed = self._update_undo_redo_state

    def _update_undo_redo_state(self):
        """Enable/disable and label the Edit > Undo/Redo actions."""
        stack = self._session.undo_stack
        self._undo_action.setEnabled(stack.can_undo())
        self._redo_action.setEnabled(stack.can_redo())

        undo_desc = stack.peek_undo()
        self._undo_action.setText(
            f"&Undo {undo_desc}" if undo_desc else "&Undo"
        )
        redo_desc = stack.peek_redo()
        self._redo_action.setText(
            f"&Redo {redo_desc}" if redo_desc else "&Redo"
        )

    def _on_undo(self):
        desc = self._session.undo_stack.undo()
        if desc:
            self.set_status(f"Undo: {desc}")

    def _on_redo(self):
        desc = self._session.undo_stack.redo()
        if desc:
            self.set_status(f"Redo: {desc}")

    # --- Session I/O ---

    def _on_save_session(self):
        """Save session to JSON — auto-path if output_dir exists, else prompt."""
        if self._session_path:
            save_path = self._session_path
        elif self._session.output_dir:
            save_path = self._session.output_dir / "bodypipe_session.json"
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Session", "", "Session Files (*.json);;All Files (*)"
            )
            if not path:
                return
            save_path = Path(path)

        try:
            self._session.save(save_path)
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save session:\n{e}")
            return

        self._session_path = save_path
        self._add_recent(save_path)
        self.set_status(f"Session saved to {save_path.name}")
        self.session_saved.emit(save_path)

    def _on_save_session_as(self):
        """Save session to a user-chosen location."""
        default_dir = str(self._session.output_dir) if self._session.output_dir else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session As", default_dir,
            "Session Files (*.json);;All Files (*)",
        )
        if not path:
            return
        save_path = Path(path)
        try:
            self._session.save(save_path)
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save session:\n{e}")
            return
        self._session_path = save_path
        self._add_recent(save_path)
        self.set_status(f"Session saved to {save_path}")
        self.session_saved.emit(save_path)

    def _on_open_session(self):
        """Open session from JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Session", "", "Session Files (*.json);;All Files (*)"
        )
        if not path:
            return
        self._load_session(Path(path))

    def _load_session(self, session_path: Path):
        """Load session from path, update shared session, refresh UI."""
        try:
            loaded = Session.load(session_path)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load session:\n{e}")
            return

        # Copy all fields to shared session object (panels hold a reference)
        # Preserve the existing undo stack (with its on_changed callback)
        saved_undo_stack = self._session.undo_stack
        for attr in vars(loaded):
            setattr(self._session, attr, getattr(loaded, attr))
        self._session.undo_stack = saved_undo_stack
        self._session.undo_stack.clear()

        self._session_path = session_path
        self._add_recent(session_path)

        # If video exists, load into current settings panel
        if self._session.video_path and self._session.video_path.is_file():
            settings = self._pipeline_dock.current_settings
            if hasattr(settings, "_load_video"):
                settings._load_video(str(self._session.video_path))

        self.set_status(f"Session loaded from {session_path.name}")
        self.session_loaded.emit(self._session)

    # --- Session Library ---

    def _refresh_session_library(self):
        """Scan for sessions and refresh the library panel.

        Passes recent session paths as extra scan targets so sessions saved
        outside the standard GVHMR output directories still appear.
        """
        recent = self._get_recent()
        extra_paths = [Path(p) for p in recent]
        self._session_library.scan(extra_paths=extra_paths)

    # --- Recent Sessions ---

    def _get_recent(self) -> list[str]:
        """Get recent session paths from QSettings."""
        val = self._settings.value("recent_sessions", [])
        if isinstance(val, str):
            return [val] if val else []
        return list(val) if val else []

    def _add_recent(self, path: Path):
        """Add path to recent sessions list."""
        recent = self._get_recent()
        path_str = str(path)
        if path_str in recent:
            recent.remove(path_str)
        recent.insert(0, path_str)
        recent = recent[: self.MAX_RECENT]
        self._settings.setValue("recent_sessions", recent)
        self._update_recent_menu()

    def _update_recent_menu(self):
        """Rebuild the Recent Sessions submenu from QSettings."""
        self._recent_menu.clear()
        recent = self._get_recent()
        if not recent:
            action = self._recent_menu.addAction("(No recent sessions)")
            action.setEnabled(False)
            return
        for path_str in recent:
            action = self._recent_menu.addAction(Path(path_str).name)
            action.setData(path_str)
            action.triggered.connect(
                lambda checked, p=path_str: self._open_recent(p)
            )

    def _open_recent(self, path_str: str):
        """Open a session from the recent sessions list."""
        path = Path(path_str)
        if not path.is_file():
            QMessageBox.warning(
                self, "File Not Found", f"Session file not found:\n{path}"
            )
            recent = self._get_recent()
            if path_str in recent:
                recent.remove(path_str)
                self._settings.setValue("recent_sessions", recent)
                self._update_recent_menu()
            return
        self._load_session(path)

    # --- Keyboard Shortcuts ---

    def _on_keyboard_shortcuts(self):
        """Show the Keyboard Shortcuts dialog."""
        dlg = KeyboardShortcutsDialog(self)
        dlg.exec()

    # --- About ---

    def _on_about(self):
        """Show About dialog."""
        QMessageBox.about(
            self,
            "About bodypipe",
            "<h3>bodypipe — Motion Capture Studio</h3>"
            "<p>PySide6 interface for GVHMR body, hand, and face capture.</p>"
            "<p>Multi-person tracking with identity verification "
            "and pose correction.</p>",
        )
