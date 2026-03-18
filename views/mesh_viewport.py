"""SMPL-X mesh viewport — QOpenGLWidget with Phong shading + skeleton overlay.

Renders the SMPL-X body mesh for the selected person at the current frame.
The body model is loaded once; vertices are recomputed per frame from the
session's SMPL-X parameters (torch forward pass → numpy), uploaded to
dynamic VBOs, and rendered with Phong shading matching shaders/mesh.vert
and shaders/mesh.frag.

Phase 3.3 adds skeleton overlay (bones as GL_LINES, joints as GL_POINTS)
and click-to-select joint picking via screen-space distance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QPainter, QFont, QColor, QFontMetrics
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from models.session import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenGL availability — graceful degradation when deps are missing
# ---------------------------------------------------------------------------
_HAS_GL = False
_QOpenGLWidget = None
try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget as _QOpenGLWidget
    from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader
    from OpenGL import GL as gl

    _HAS_GL = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Shader sources — embedded copies of shaders/mesh.vert and mesh.frag.
# The canonical files are loaded if found; these are fallbacks.
# ---------------------------------------------------------------------------
_VERT_SRC = """\
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 frag_normal;
out vec3 frag_color;
out vec3 frag_pos;

void main() {
    frag_pos = vec3(model * vec4(position, 1.0));
    frag_normal = mat3(transpose(inverse(model))) * normal;
    frag_color = color;
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

_FRAG_SRC = """\
#version 330 core
in vec3 frag_normal;
in vec3 frag_color;
in vec3 frag_pos;

uniform vec3 light_dir;
uniform vec3 light_color;
uniform vec3 ambient;

out vec4 out_color;

void main() {
    vec3 norm = normalize(frag_normal);
    float diff = max(dot(norm, -light_dir), 0.0);
    vec3 diffuse = diff * light_color;
    vec3 result = (ambient + diffuse) * frag_color;
    out_color = vec4(result, 1.0);
}
"""

_SHADER_DIR = Path(__file__).resolve().parent.parent / "shaders"


def _load_shader_source(name: str, fallback: str) -> str:
    """Load shader from file, fall back to embedded source."""
    path = _SHADER_DIR / name
    if path.is_file():
        return path.read_text()
    return fallback


# ---------------------------------------------------------------------------
# Skeleton data — 52-joint SMPL-X hierarchy (from smplx_to_bvh.py)
# ---------------------------------------------------------------------------

JOINT_NAMES = [
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
]

JOINT_PARENTS = [
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
]

DEFAULT_OFFSETS = {
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

# Body bone connections (indices 0-21 only — hand bones omitted for clarity)
BONE_CONNECTIONS = [
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

# Add hand bones
for _wrist, _start_idx in [(20, 22), (21, 37)]:
    for _finger_base in range(_start_idx, _start_idx + 15, 3):
        BONE_CONNECTIONS.append((_wrist, _finger_base))
        BONE_CONNECTIONS.append((_finger_base, _finger_base + 1))
        BONE_CONNECTIONS.append((_finger_base + 1, _finger_base + 2))

# Number of body joints (for joint picking — only pick body, not hand joints)
_N_BODY_JOINTS = 22

# Joint picking threshold in pixels
_JOINT_PICK_THRESHOLD = 20.0

# Accent color for selected joint highlight (matches app theme)
_ACCENT_COLOR = np.array([0.914, 0.271, 0.376], dtype=np.float32)  # #e94560

# Bone color (light gray)
_BONE_COLOR = np.array([0.7, 0.7, 0.7], dtype=np.float32)

# Joint colors: body joints = white, hand joints = slightly dimmer
_BODY_JOINT_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float32)
_HAND_JOINT_COLOR = np.array([0.6, 0.6, 0.6], dtype=np.float32)

# GL point/line sizes for skeleton rendering
_JOINT_POINT_SIZE = 6.0
_SELECTED_JOINT_POINT_SIZE = 12.0
_BONE_LINE_WIDTH = 2.0

# Joint label rendering constants
_LABEL_FONT_SIZE = 9
_LABEL_BG_COLOR = QColor(0, 0, 0, 180)       # semi-transparent black
_LABEL_TEXT_COLOR = QColor(220, 220, 220)      # light gray
_LABEL_SELECTED_BG = QColor(233, 69, 96, 200) # accent with alpha
_LABEL_SELECTED_TEXT = QColor(255, 255, 255)   # white
_LABEL_PADDING_X = 3   # horizontal padding inside label bg
_LABEL_PADDING_Y = 1   # vertical padding inside label bg
_LABEL_OFFSET_X = 8    # pixels right of joint point
_LABEL_OFFSET_Y = -4   # pixels above joint point center
_LABEL_MARGIN = 4       # viewport edge margin (labels clamped inside)

# Joint color palette for per-joint vertex coloring — 22 distinct colors for body joints.
# Hand joints (22–51) inherit the color of their parent wrist (L=20, R=21).
_JOINT_PALETTE = np.array([
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


# ---------------------------------------------------------------------------
# Pure helper functions (testable without OpenGL)
# ---------------------------------------------------------------------------


def compute_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals by averaging incident face normals.

    Parameters
    ----------
    vertices : (V, 3) float32
    faces : (F, 3) int32

    Returns
    -------
    normals : (V, 3) float32 — unit-length per-vertex normals
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)

    vertex_normals = np.zeros_like(vertices)
    np.add.at(vertex_normals, faces[:, 0], face_normals)
    np.add.at(vertex_normals, faces[:, 1], face_normals)
    np.add.at(vertex_normals, faces[:, 2], face_normals)

    lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)
    return (vertex_normals / lengths).astype(np.float32)


def forward_kinematics(params: dict, frame_idx: int) -> np.ndarray:
    """Compute 3D joint positions in camera space for one frame.

    Uses DEFAULT_OFFSETS (model-derived rest-pose, not shape-dependent).
    Mirrors ``visualize_skeleton.py:forward_kinematics``.

    Parameters
    ----------
    params : dict with keys global_orient (N,3), body_pose (N,21*3 or N,21,3),
             transl (N,3). Optional: left_hand_pose (N,15,3), right_hand_pose (N,15,3).
    frame_idx : which frame to compute

    Returns
    -------
    positions : (52, 3) float64 — joint positions in camera space
    """
    from scipy.spatial.transform import Rotation

    n_joints = len(JOINT_NAMES)
    positions = np.zeros((n_joints, 3))
    accumulated_R = np.zeros((n_joints, 3, 3))

    offsets = np.zeros((n_joints, 3))
    for i, name in enumerate(JOINT_NAMES):
        offsets[i] = DEFAULT_OFFSETS.get(name, [0, 0, 0])

    def _get_array(key):
        v = params.get(key)
        if v is None:
            return None
        return np.asarray(v) if not isinstance(v, np.ndarray) else v

    go = _get_array("global_orient")
    bp = _get_array("body_pose")
    tr = _get_array("transl")
    lh = _get_array("left_hand_pose")
    rh = _get_array("right_hand_pose")

    if go is None or bp is None:
        return positions

    # Root
    go_frame = go[frame_idx] if go.ndim >= 2 else go
    accumulated_R[0] = Rotation.from_rotvec(go_frame.ravel()[:3]).as_matrix()
    positions[0] = tr[frame_idx] if tr is not None and tr.ndim >= 2 else (tr if tr is not None else np.zeros(3))

    # Reshape body_pose to (N, 21, 3) if flat
    if bp.ndim == 2 and bp.shape[-1] != 3:
        bp = bp.reshape(bp.shape[0], -1, 3)

    for j in range(1, n_joints):
        parent = JOINT_PARENTS[j]

        if 1 <= j <= 21:
            if bp.ndim >= 3 and frame_idx < bp.shape[0]:
                rot_aa = bp[frame_idx, j - 1]
            elif bp.ndim == 2:
                rot_aa = bp[j - 1]
            else:
                rot_aa = np.zeros(3)
        elif 22 <= j <= 36:
            if lh is not None:
                if lh.ndim == 3 and frame_idx < lh.shape[0]:
                    rot_aa = lh[frame_idx, j - 22]
                elif lh.ndim == 2:
                    lh_r = lh.reshape(-1, 15, 3) if lh.shape[-1] != 3 else lh
                    rot_aa = lh_r[frame_idx, j - 22] if lh_r.ndim == 3 else np.zeros(3)
                else:
                    rot_aa = np.zeros(3)
            else:
                rot_aa = np.zeros(3)
        elif 37 <= j <= 51:
            if rh is not None:
                if rh.ndim == 3 and frame_idx < rh.shape[0]:
                    rot_aa = rh[frame_idx, j - 37]
                elif rh.ndim == 2:
                    rh_r = rh.reshape(-1, 15, 3) if rh.shape[-1] != 3 else rh
                    rot_aa = rh_r[frame_idx, j - 37] if rh_r.ndim == 3 else np.zeros(3)
                else:
                    rot_aa = np.zeros(3)
            else:
                rot_aa = np.zeros(3)
        else:
            rot_aa = np.zeros(3)

        R_local = Rotation.from_rotvec(np.asarray(rot_aa).ravel()[:3]).as_matrix()
        accumulated_R[j] = accumulated_R[parent] @ R_local
        positions[j] = positions[parent] + accumulated_R[parent] @ offsets[j]

    return positions


def project_joints_to_screen(
    joints_3d: np.ndarray,
    mvp: np.ndarray,
    viewport_w: int,
    viewport_h: int,
) -> np.ndarray:
    """Project 3D joint positions to 2D screen coordinates.

    Parameters
    ----------
    joints_3d : (N, 3) joint positions in model/camera space
    mvp : (4, 4) model-view-projection matrix
    viewport_w, viewport_h : widget pixel dimensions

    Returns
    -------
    screen : (N, 2) float64 — pixel coordinates (x, y) with origin at top-left
    """
    N = joints_3d.shape[0]
    # Homogeneous coordinates
    pts = np.hstack([joints_3d, np.ones((N, 1))])  # (N, 4)
    clip = (mvp @ pts.T).T  # (N, 4)

    # Perspective divide (avoid division by zero)
    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    ndc = clip[:, :3] / w  # (N, 3) in [-1, 1]

    # NDC → screen (OpenGL: x right, y up; screen: y down)
    screen = np.zeros((N, 2))
    screen[:, 0] = (ndc[:, 0] + 1.0) * 0.5 * viewport_w
    screen[:, 1] = (1.0 - ndc[:, 1]) * 0.5 * viewport_h
    return screen


def find_nearest_joint(
    click_x: float,
    click_y: float,
    joints_2d: np.ndarray,
    threshold: float = _JOINT_PICK_THRESHOLD,
    max_joint: int = _N_BODY_JOINTS,
) -> int | None:
    """Hit-test: return body joint index nearest to click, within threshold px.

    Only considers joints 0..max_joint-1 (body joints by default).
    """
    body = joints_2d[:max_joint]
    dists = np.sqrt((body[:, 0] - click_x) ** 2 + (body[:, 1] - click_y) ** 2)
    min_idx = int(np.argmin(dists))
    if dists[min_idx] <= threshold:
        return min_idx
    return None


def k_to_projection(
    K: np.ndarray,
    width: int,
    height: int,
    near: float = 0.01,
    far: float = 100.0,
) -> np.ndarray:
    """Convert 3x3 camera intrinsics *K* to a 4x4 OpenGL projection matrix.

    Camera-space vertices from SMPL-X use +Y down, +Z into screen (CV
    convention).  The view matrix (_CV_TO_GL) flips Y/Z so that OpenGL's
    +Y-up / -Z-into-screen convention is satisfied; this projection matrix
    works with that view matrix.
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 2.0 * fx / width
    proj[1, 1] = 2.0 * fy / height
    proj[0, 2] = 1.0 - 2.0 * cx / width
    proj[1, 2] = 2.0 * cy / height - 1.0
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    return proj


def estimate_K(width: int, height: int) -> np.ndarray:
    """Estimate camera intrinsics from image dimensions.

    Mirrors ``hmr4d.utils.geo.hmr_cam.estimate_K``.
    """
    focal = (width**2 + height**2) ** 0.5
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = width / 2.0
    K[1, 2] = height / 2.0
    return K


def compute_orbit_view(
    yaw_deg: float,
    pitch_deg: float,
    distance: float,
    center: np.ndarray,
) -> np.ndarray:
    """Compute a look-at view matrix for an orbit camera.

    The camera sits on a sphere of radius *distance* around *center*,
    positioned by spherical coordinates *yaw_deg* (horizontal angle from
    +Z axis) and *pitch_deg* (vertical, positive = above horizontal).

    Parameters
    ----------
    yaw_deg   : horizontal angle in degrees
    pitch_deg : vertical angle in degrees (clamped to ±89°)
    distance  : orbit radius (meters)
    center    : (3,) pivot point in GL world space

    Returns
    -------
    view : (4, 4) float32 view matrix (GL convention, +Y up, -Z fwd)
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(np.clip(pitch_deg, -89.0, 89.0))
    cos_p = np.cos(pitch)
    eye = np.asarray(center, dtype=np.float32) + distance * np.array(
        [cos_p * np.sin(yaw), np.sin(pitch), cos_p * np.cos(yaw)],
        dtype=np.float32,
    )

    fwd = np.asarray(center, dtype=np.float32) - eye
    fwd_len = np.linalg.norm(fwd)
    if fwd_len < 1e-8:
        fwd = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        fwd = fwd / fwd_len

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(fwd, world_up)
    r_len = np.linalg.norm(right)
    if r_len < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / r_len

    up = np.cross(right, fwd)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = -fwd
    view[0, 3] = -np.dot(right, eye)
    view[1, 3] = -np.dot(up, eye)
    view[2, 3] = np.dot(fwd, eye)
    return view


def perspective_fov(
    fov_deg: float,
    aspect: float,
    near: float = 0.01,
    far: float = 100.0,
) -> np.ndarray:
    """Standard symmetric perspective projection from vertical FOV.

    Parameters
    ----------
    fov_deg : vertical field-of-view in degrees
    aspect  : width / height
    near, far : clipping planes

    Returns
    -------
    proj : (4, 4) float32 projection matrix
    """
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / max(aspect, 1e-6)
    proj[1, 1] = f
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    return proj


# View matrix: flip Y and Z to convert from CV camera space to GL eye space.
_CV_TO_GL = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)

def compute_joint_colors(
    lbs_weights: np.ndarray,
    palette: np.ndarray | None = None,
) -> np.ndarray:
    """Assign per-vertex RGB colors based on dominant LBS joint weight.

    Parameters
    ----------
    lbs_weights : (V, J) float array
        Linear blend skinning weights from the SMPL-X body model.
    palette : (22, 3) float array, optional
        Joint color palette.  Defaults to ``_JOINT_PALETTE``.

    Returns
    -------
    (V, 3) float32 array of per-vertex RGB colors.

    Hand joints (22+) inherit the color of their parent wrist
    (L_Wrist=20 for left hand joints 22-36, R_Wrist=21 for right hand 37-51).
    Joints beyond 51 (jaw/eyes) map to the Head color (15).
    """
    if palette is None:
        palette = _JOINT_PALETTE

    V, J = lbs_weights.shape
    # Map each SMPL-X joint index → body palette index (0-21)
    joint_map = np.zeros(J, dtype=np.int32)
    for j in range(J):
        if j < 22:
            joint_map[j] = j
        elif j < 37:
            joint_map[j] = 20  # L_Wrist
        elif j < 52:
            joint_map[j] = 21  # R_Wrist
        else:
            joint_map[j] = 15  # Head (jaw/eyes)

    # Find dominant joint per vertex
    dominant = np.argmax(lbs_weights, axis=1)  # (V,)
    palette_idx = joint_map[dominant]  # (V,)
    return palette[palette_idx].astype(np.float32)


def confidence_to_color(confidence: float) -> np.ndarray:
    """Map a confidence value [0, 1] to an RGB color (red → yellow → green).

    Parameters
    ----------
    confidence : float
        Value in [0, 1].  Values outside this range are clamped.

    Returns
    -------
    (3,) float32 RGB array.
    """
    c = float(np.clip(confidence, 0.0, 1.0))
    # 0.0 → red, 0.5 → yellow, 1.0 → green
    if c < 0.5:
        t = c * 2.0
        return np.array([1.0, t, 0.0], dtype=np.float32)
    else:
        t = (c - 0.5) * 2.0
        return np.array([1.0 - t, 1.0, 0.0], dtype=np.float32)


# Default skin-tone for mesh rendering (warm beige).
_SKIN_COLOR = np.array([0.82, 0.72, 0.63], dtype=np.float32)

# Lighting parameters.
_LIGHT_DIR = np.array([0.0, -0.5, -1.0], dtype=np.float32)
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)
_LIGHT_COLOR = np.array([0.8, 0.8, 0.8], dtype=np.float32)
_AMBIENT = np.array([0.35, 0.35, 0.35], dtype=np.float32)

# Maximum vertex cache size (per person+frame).
_CACHE_MAX = 200

# Orbit camera defaults and sensitivity.
_ORBIT_SENSITIVITY = 0.3       # degrees per pixel drag
_PAN_SENSITIVITY = 0.003       # fraction of distance per pixel
_ZOOM_FACTOR = 0.1             # fraction of distance per wheel step
_ORBIT_DEFAULT_FOV = 45.0      # vertical FOV in degrees
_ORBIT_DEFAULT_DISTANCE = 3.0  # meters from orbit center
_ORBIT_DEFAULT_YAW = 0.0       # look from +Z
_ORBIT_DEFAULT_PITCH = 10.0    # slight tilt from above
_PITCH_LIMIT = 89.0            # clamp to avoid gimbal lock
_ORBIT_DEFAULT_CENTER = np.array([0.0, 0.0, -2.5], dtype=np.float32)

# Grid floor constants (orbit mode reference plane).
_GRID_SIZE = 10.0            # half-extent in meters (grid spans ±size)
_GRID_DIVISIONS = 20         # number of cells per half (total 2*N lines per axis)
_GRID_COLOR = np.array([0.25, 0.25, 0.28], dtype=np.float32)      # dim gray
_GRID_AXIS_COLOR = np.array([0.40, 0.40, 0.45], dtype=np.float32) # brighter center lines


def compute_joint_label_layout(
    joints_2d: np.ndarray,
    viewport_w: int,
    viewport_h: int,
    selected_joint: int = -1,
    body_only: bool = True,
    margin: int = _LABEL_MARGIN,
) -> list[tuple[int, str, float, float, bool]]:
    """Compute label positions for visible joints, clamped to viewport.

    Pure function — no Qt dependency, testable without a widget.

    Parameters
    ----------
    joints_2d : (N, 2) screen-space joint positions (from project_joints_to_screen)
    viewport_w, viewport_h : widget pixel dimensions
    selected_joint : index of the currently selected joint (-1 for none)
    body_only : if True, only label body joints (0-21) plus selected joint
    margin : minimum pixel distance from viewport edge

    Returns
    -------
    labels : list of (joint_idx, name, screen_x, screen_y, is_selected)
        Sorted with selected joint last (drawn on top).
    """
    n_joints = min(len(joints_2d), len(JOINT_NAMES))
    max_idx = _N_BODY_JOINTS if body_only else n_joints

    labels = []
    for i in range(n_joints):
        # Include body joints or all joints, always include selected
        if i >= max_idx and i != selected_joint:
            continue

        sx, sy = float(joints_2d[i, 0]), float(joints_2d[i, 1])

        # Skip joints projected outside the viewport (with margin)
        if sx < -50 or sx > viewport_w + 50 or sy < -50 or sy > viewport_h + 50:
            continue

        # Clamp label anchor to within viewport bounds
        sx = max(margin, min(sx, viewport_w - margin))
        sy = max(margin, min(sy, viewport_h - margin))

        is_sel = (i == selected_joint)
        labels.append((i, JOINT_NAMES[i], sx, sy, is_sel))

    # Sort so selected joint label is drawn last (on top)
    labels.sort(key=lambda t: t[4])
    return labels


def compute_grid_lines(
    size: float = _GRID_SIZE,
    divisions: int = _GRID_DIVISIONS,
    y: float = 0.0,
    center_x: float = 0.0,
    center_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate grid line vertices on the XZ plane at the given Y level.

    The grid is centered at (center_x, y, center_z) and spans ±size in X and Z.
    Lines parallel to each axis are spaced ``size / divisions`` meters apart.
    The center lines (through center_x and center_z) use a brighter color.

    Parameters
    ----------
    size : float
        Half-extent of the grid in meters.
    divisions : int
        Number of cells per half-axis (total lines = 2*divisions + 1 per axis).
    y : float
        Y coordinate of the grid plane (GL space, Y-up).
    center_x, center_z : float
        Center of the grid in the XZ plane.

    Returns
    -------
    positions : (N, 3) float32 — line segment endpoints (pairs of vertices)
    colors : (N, 3) float32 — per-vertex colors
    """
    step = size / max(divisions, 1)
    lines_per_axis = 2 * divisions + 1
    total_verts = lines_per_axis * 2 * 2  # 2 axes × lines × 2 endpoints

    positions = np.zeros((total_verts, 3), dtype=np.float32)
    colors = np.zeros((total_verts, 3), dtype=np.float32)

    idx = 0
    for i in range(-divisions, divisions + 1):
        x = center_x + i * step
        is_center = (i == 0)
        color = _GRID_AXIS_COLOR if is_center else _GRID_COLOR

        # Line parallel to Z axis
        positions[idx] = [x, y, center_z - size]
        positions[idx + 1] = [x, y, center_z + size]
        colors[idx] = color
        colors[idx + 1] = color
        idx += 2

    for i in range(-divisions, divisions + 1):
        z = center_z + i * step
        is_center = (i == 0)
        color = _GRID_AXIS_COLOR if is_center else _GRID_COLOR

        # Line parallel to X axis
        positions[idx] = [center_x - size, y, z]
        positions[idx + 1] = [center_x + size, y, z]
        colors[idx] = color
        colors[idx + 1] = color
        idx += 2

    return positions[:idx], colors[:idx]


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------
_BaseWidget = _QOpenGLWidget if _HAS_GL else QWidget


class MeshViewport(_BaseWidget):
    """QOpenGLWidget rendering SMPL-X mesh with Phong shading.

    Falls back to a placeholder label when OpenGL or PyOpenGL is unavailable.

    Public API (used by MultiPersonTab and future PoseCorrectorPanel):
        set_session(session)        — bind data source
        set_person(person_id)       — select mesh to render
        on_frame_changed(frame_idx) — update displayed frame

    Signals:
        joint_clicked(int)   — emitted when a joint is clicked (Phase 3.3)
        camera_changed(dict) — emitted when camera state changes
    """

    joint_clicked = Signal(int)
    camera_changed = Signal(object)

    def __init__(self, gvhmr_root: Path | None = None, parent=None):
        super().__init__(parent)
        self._gvhmr_root = gvhmr_root
        self._session: Session | None = None
        self._person_id: int = -1
        self._current_frame: int = 0

        # Mesh data
        self._vertices: np.ndarray | None = None  # (V, 3) float32
        self._normals: np.ndarray | None = None  # (V, 3) float32
        self._faces: np.ndarray | None = None  # (F, 3) int32
        self._n_vertices: int = 0
        self._n_faces: int = 0

        # SMPL-X body model (loaded lazily)
        self._body_model: object | None = None
        self._model_loaded: bool = False
        self._lbs_weights: np.ndarray | None = None  # (V, J) from model

        # Vertex color mode ("solid", "joint", "confidence")
        self._color_mode: str = "solid"

        # Camera matrices
        self._projection = np.eye(4, dtype=np.float32)
        self._view = _CV_TO_GL.copy()
        self._model_mat = np.eye(4, dtype=np.float32)

        # Vertex cache  {(person_id, frame_idx): (vertices, normals)}
        self._vertex_cache: dict[
            tuple[int, int], tuple[np.ndarray, np.ndarray]
        ] = {}

        # GL state
        self._gl_ready: bool = False
        self._shader: object | None = None  # QOpenGLShaderProgram
        self._vao_id: int = 0
        self._vbo_pos: int = 0
        self._vbo_norm: int = 0
        self._vbo_color: int = 0
        self._ebo: int = 0
        self._n_indices: int = 0
        self._faces_uploaded: bool = False

        # Camera mode ("incam" or "orbit")
        self._camera_mode: str = "incam"

        # Orbit camera state (used in "orbit" mode)
        self._orbit_yaw: float = _ORBIT_DEFAULT_YAW
        self._orbit_pitch: float = _ORBIT_DEFAULT_PITCH
        self._orbit_distance: float = _ORBIT_DEFAULT_DISTANCE
        self._orbit_center: np.ndarray = _ORBIT_DEFAULT_CENTER.copy()
        self._orbit_auto_centered: bool = False

        # Mouse tracking for orbit interaction
        self._mouse_last_pos: tuple[int, int] | None = None

        # Skeleton state (Phase 3.3)
        self._joint_positions: np.ndarray | None = None  # (52, 3) camera-space
        self._selected_joint: int = -1  # -1 = no selection
        self._show_skeleton: bool = True  # whether to draw skeleton overlay

        # Pose override for real-time preview (Phase 3.4)
        # dict with keys: frame_idx (int), global_orient (3,) optional,
        #                  body_pose {int: (3,)} optional
        self._pose_override: dict | None = None

        # Joint label overlay state (QPainter text over GL)
        self._show_joint_labels: bool = False  # off by default, toggled by user

        # Grid floor state (orbit mode reference plane)
        self._show_grid: bool = True  # visible by default in orbit mode
        self._grid_y: float = 0.0  # Y level of the grid in GL space

        # Status message for fallback rendering
        self._status_msg: str = ""

        self.setMinimumSize(QSize(200, 150))

        if _HAS_GL:
            from PySide6.QtGui import QSurfaceFormat

            fmt = QSurfaceFormat()
            fmt.setVersion(3, 3)
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
            fmt.setDepthBufferSize(24)
            self.setFormat(fmt)
        else:
            self._status_msg = "OpenGL not available"
            self._setup_fallback()

    def _setup_fallback(self):
        """Show a label when GL is not available."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        self._fallback_label = QLabel(self._status_msg or "3D Viewport")
        self._fallback_label.setAlignment(Qt.AlignCenter)
        self._fallback_label.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(self._fallback_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session(self, session: Session):
        """Bind session data source."""
        self._session = session
        self._vertex_cache.clear()

    def set_person(self, person_id: int):
        """Select which person's mesh to display."""
        if person_id == self._person_id:
            return
        self._person_id = person_id
        self._refresh_mesh()

    def on_frame_changed(self, frame_idx: int):
        """Update displayed frame."""
        if frame_idx == self._current_frame:
            return
        self._current_frame = frame_idx
        self._refresh_mesh()

    def set_camera_mode(self, mode: str):
        """Set camera mode ('incam' or 'orbit').

        In 'incam' mode the projection matches the video camera using K
        intrinsics.  In 'orbit' mode the user can rotate, pan, and zoom
        with the mouse.
        """
        if mode not in ("incam", "orbit"):
            return
        if mode == self._camera_mode:
            return
        self._camera_mode = mode
        if mode == "orbit":
            self._auto_center_orbit()
        self._update_camera()
        self.camera_changed.emit(self._camera_state())

    def set_color_mode(self, mode: str):
        """Set vertex color mode ('solid', 'joint', 'confidence').

        Recomputes vertex colors and triggers a repaint.
        """
        if mode not in ("solid", "joint", "confidence"):
            return
        if mode == self._color_mode:
            return
        self._color_mode = mode
        # Re-upload colors if we have vertices loaded
        if self._vertices is not None and self._gl_ready:
            self._upload_buffers()
        if _HAS_GL:
            self.update()

    def highlight_joint(self, joint_idx: int):
        """Highlight a joint in accent color and trigger repaint."""
        if joint_idx == self._selected_joint:
            return
        self._selected_joint = joint_idx
        if _HAS_GL:
            self.update()

    def set_show_skeleton(self, show: bool):
        """Toggle skeleton overlay visibility."""
        if show == self._show_skeleton:
            return
        self._show_skeleton = show
        if _HAS_GL:
            self.update()

    def set_show_grid(self, show: bool):
        """Toggle grid floor visibility in orbit mode."""
        if show == self._show_grid:
            return
        self._show_grid = show
        if _HAS_GL:
            self.update()

    def set_show_joint_labels(self, show: bool):
        """Toggle joint name text labels drawn over the skeleton."""
        if show == self._show_joint_labels:
            return
        self._show_joint_labels = show
        if _HAS_GL:
            self.update()

    def set_pose_override(self, override: dict | None):
        """Set temporary per-frame pose override for real-time preview.

        Parameters
        ----------
        override : dict or None
            If dict, keys are:
                frame_idx (int) — which frame to override
                global_orient (3,) ndarray — optional global orientation
                body_pose (dict[int, (3,)]) — optional sparse body joint overrides
            Pass None to clear all overrides.
        """
        self._pose_override = override
        # Invalidate cache for affected frame to force recomputation
        if override is not None and self._person_id >= 0:
            frame_idx = override.get("frame_idx", self._current_frame)
            self._vertex_cache.pop((self._person_id, frame_idx), None)
        self._refresh_mesh()

    def invalidate_cache(self, person_id: int | None = None, frame_idx: int | None = None):
        """Invalidate vertex cache entries.

        If both person_id and frame_idx are given, removes that single entry.
        Otherwise clears the entire cache.
        """
        if person_id is not None and frame_idx is not None:
            self._vertex_cache.pop((person_id, frame_idx), None)
        else:
            self._vertex_cache.clear()

    def _apply_override_to_params(self, params: dict, frame_idx: int) -> dict:
        """Create a modified copy of params with pose override applied.

        Returns a shallow copy of params with overridden arrays replaced
        by numpy copies. Non-overridden arrays are left as-is.
        """
        if self._pose_override is None:
            return params

        ov_frame = self._pose_override.get("frame_idx", self._current_frame)
        if ov_frame != frame_idx:
            return params

        params = dict(params)

        go_override = self._pose_override.get("global_orient")
        bp_overrides = self._pose_override.get("body_pose")

        if go_override is not None:
            go = np.array(params["global_orient"], dtype=np.float32)
            if go.ndim >= 2 and frame_idx < go.shape[0]:
                go[frame_idx] = go_override
            params["global_orient"] = go

        if bp_overrides:
            bp = np.array(params["body_pose"], dtype=np.float32)
            if bp.ndim == 2 and bp.shape[-1] != 3:
                bp = bp.reshape(bp.shape[0], -1, 3)
            for j_idx, aa in bp_overrides.items():
                if bp.ndim >= 3 and frame_idx < bp.shape[0] and j_idx < bp.shape[1]:
                    bp[frame_idx, j_idx] = aa
            params["body_pose"] = bp

        return params

    # ------------------------------------------------------------------
    # Camera modes & mouse interaction
    # ------------------------------------------------------------------

    def _camera_state(self) -> dict:
        """Return current camera state for the camera_changed signal."""
        if self._camera_mode == "orbit":
            return {
                "mode": "orbit",
                "yaw": self._orbit_yaw,
                "pitch": self._orbit_pitch,
                "distance": self._orbit_distance,
                "center": self._orbit_center.tolist(),
            }
        return {"mode": "incam"}

    def _update_camera(self):
        """Recompute model/view/projection from current camera state."""
        if self._camera_mode == "orbit":
            self._model_mat = _CV_TO_GL.copy()
            self._view = compute_orbit_view(
                self._orbit_yaw,
                self._orbit_pitch,
                self._orbit_distance,
                self._orbit_center,
            )
        else:  # incam
            self._model_mat = np.eye(4, dtype=np.float32)
            self._view = _CV_TO_GL.copy()
        w = self.width() if self.width() > 0 else 200
        h = self.height() if self.height() > 0 else 150
        self._update_projection(w, h)
        if _HAS_GL:
            self.update()

    def _reset_orbit(self):
        """Reset orbit camera to defaults, auto-centering on mesh."""
        self._orbit_yaw = _ORBIT_DEFAULT_YAW
        self._orbit_pitch = _ORBIT_DEFAULT_PITCH
        self._orbit_distance = _ORBIT_DEFAULT_DISTANCE
        self._orbit_center = _ORBIT_DEFAULT_CENTER.copy()
        self._orbit_auto_centered = False
        self._auto_center_orbit()

    def _auto_center_orbit(self):
        """Set orbit center to the current mesh centroid in GL space."""
        if self._vertices is None:
            return
        # Convert centroid from CV camera space to GL space (flip Y, Z)
        centroid = self._vertices.mean(axis=0).copy()
        centroid[1] *= -1
        centroid[2] *= -1
        self._orbit_center = centroid.astype(np.float32)
        # Set distance from mesh extent
        gl_verts = self._vertices.copy()
        gl_verts[:, 1] *= -1
        gl_verts[:, 2] *= -1
        extent = np.max(
            np.linalg.norm(gl_verts - self._orbit_center, axis=1)
        )
        self._orbit_distance = max(float(extent) * 2.5, 1.0)
        # Place grid floor at lowest mesh point (feet)
        self._grid_y = float(np.min(gl_verts[:, 1]))
        self._orbit_auto_centered = True

    def mousePressEvent(self, event):
        """Joint picking on left-click, orbit drag on left-drag in orbit mode."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Try joint picking first
            hit = self._pick_joint(event.position().x(), event.position().y())
            if hit is not None:
                self._selected_joint = hit
                self.joint_clicked.emit(hit)
                if _HAS_GL:
                    self.update()

        if self._camera_mode == "orbit":
            self._mouse_last_pos = (event.position().x(), event.position().y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Update orbit/pan during drag."""
        if self._camera_mode != "orbit" or self._mouse_last_pos is None:
            super().mouseMoveEvent(event)
            return

        x, y = event.position().x(), event.position().y()
        dx = x - self._mouse_last_pos[0]
        dy = y - self._mouse_last_pos[1]
        self._mouse_last_pos = (x, y)

        buttons = event.buttons()
        if buttons & Qt.MouseButton.LeftButton:
            # Orbit: rotate camera around center
            self._orbit_yaw += dx * _ORBIT_SENSITIVITY
            self._orbit_pitch += dy * _ORBIT_SENSITIVITY
            self._orbit_pitch = float(
                np.clip(self._orbit_pitch, -_PITCH_LIMIT, _PITCH_LIMIT)
            )
            self._update_camera()
            self.camera_changed.emit(self._camera_state())
        elif buttons & Qt.MouseButton.MiddleButton:
            # Pan: translate orbit center in the camera's screen plane
            yaw = np.radians(self._orbit_yaw)
            pitch = np.radians(self._orbit_pitch)
            cos_p = np.cos(pitch)
            eye_dir = np.array(
                [cos_p * np.sin(yaw), np.sin(pitch), cos_p * np.cos(yaw)],
                dtype=np.float32,
            )
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(-eye_dir, world_up)
            r_len = np.linalg.norm(right)
            if r_len > 1e-6:
                right /= r_len
            up = np.cross(right, -eye_dir)
            pan_speed = self._orbit_distance * _PAN_SENSITIVITY
            self._orbit_center += right * float(dx * pan_speed)
            self._orbit_center -= up * float(dy * pan_speed)
            self._update_camera()
            self.camera_changed.emit(self._camera_state())

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """End orbit/pan drag."""
        self._mouse_last_pos = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        """Zoom in/out in orbit mode."""
        if self._camera_mode == "orbit":
            delta = event.angleDelta().y()
            factor = 1.0 - (delta / 120.0) * _ZOOM_FACTOR
            self._orbit_distance = max(self._orbit_distance * factor, 0.1)
            self._update_camera()
            self.camera_changed.emit(self._camera_state())
            event.accept()
            return
        super().wheelEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Reset orbit camera on double-click."""
        if self._camera_mode == "orbit":
            self._reset_orbit()
            self._update_camera()
            self.camera_changed.emit(self._camera_state())
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # ------------------------------------------------------------------
    # Skeleton: joint computation + joint picking
    # ------------------------------------------------------------------

    def _compute_joints(self) -> np.ndarray | None:
        """Compute 3D joint positions for current person/frame.

        Returns (52, 3) array in camera space, or None if params unavailable.
        """
        if self._session is None or self._person_id < 0:
            return None
        track = self._session.person_tracks.get(self._person_id)
        if track is None or track.smplx_params is None:
            return None

        params = track.smplx_params

        # Apply pose override for real-time preview
        if self._pose_override is not None:
            ov_frame = self._pose_override.get("frame_idx", self._current_frame)
            if ov_frame == self._current_frame:
                params = self._apply_override_to_params(params, self._current_frame)

        try:
            return forward_kinematics(params, self._current_frame)
        except Exception as e:
            logger.warning("FK failed (pid=%d, f=%d): %s",
                           self._person_id, self._current_frame, e)
            return None

    def _pick_joint(self, screen_x: float, screen_y: float) -> int | None:
        """Screen-space joint picking: project joints, find nearest within threshold."""
        if self._joint_positions is None:
            return None
        w = self.width() if self.width() > 0 else 200
        h = self.height() if self.height() > 0 else 150
        mvp = self._projection @ self._view @ self._model_mat
        screen = project_joints_to_screen(self._joint_positions, mvp, w, h)
        return find_nearest_joint(screen_x, screen_y, screen)

    # ------------------------------------------------------------------
    # SMPL-X model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        """Lazily load SMPL-X body model. Returns True on success."""
        if self._model_loaded:
            return self._body_model is not None
        self._model_loaded = True

        try:
            import torch  # noqa: F401
            from hmr4d.utils.body_model.smplx_lite import SmplxLite

            if self._gvhmr_root:
                model_dir = (
                    self._gvhmr_root
                    / "inputs"
                    / "checkpoints"
                    / "body_models"
                    / "smplx"
                )
            else:
                model_dir = None

            self._body_model = SmplxLite(
                model_dir=str(model_dir) if model_dir else None
            )
            self._body_model.cpu().eval()

            # Extract face topology (static — does not change across frames)
            self._faces = np.asarray(self._body_model.faces, dtype=np.int32)
            self._n_faces = len(self._faces)

            # Extract LBS weights for joint-influence coloring
            if hasattr(self._body_model, "lbs_weights"):
                self._lbs_weights = self._body_model.lbs_weights.detach().cpu().numpy()
                logger.info("LBS weights: %s", self._lbs_weights.shape)

            logger.info("SMPL-X model loaded: %d faces", self._n_faces)
            return True
        except Exception as e:
            logger.warning("Could not load SMPL-X model: %s", e)
            self._status_msg = f"Model unavailable: {e}"
            return False

    # ------------------------------------------------------------------
    # Vertex computation
    # ------------------------------------------------------------------

    def _compute_vertices(
        self, person_id: int, frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute SMPL-X vertices and normals for one frame.

        Returns ``(vertices, normals)`` each (V, 3) float32, or ``None``
        when the model / params are unavailable.
        """
        cache_key = (person_id, frame_idx)

        # Skip cache when pose override is active for this frame
        override_active = (
            self._pose_override is not None
            and self._pose_override.get("frame_idx", self._current_frame) == frame_idx
        )

        if not override_active and cache_key in self._vertex_cache:
            return self._vertex_cache[cache_key]

        if not self._load_model():
            return None
        if self._session is None:
            return None

        track = self._session.person_tracks.get(person_id)
        if track is None or track.smplx_params is None:
            return None

        params = track.smplx_params

        # Apply pose override for real-time preview
        if override_active:
            params = self._apply_override_to_params(params, frame_idx)

        try:
            import torch

            with torch.no_grad():
                go = params.get("global_orient")
                bp = params.get("body_pose")
                be = params.get("betas")
                tr = params.get("transl")

                if go is None or bp is None or be is None:
                    return None

                def _to_tensor(x):
                    if isinstance(x, torch.Tensor):
                        return x.float().cpu()
                    return torch.tensor(np.asarray(x), dtype=torch.float32)

                go_t = _to_tensor(go)
                bp_t = _to_tensor(bp)
                be_t = _to_tensor(be)
                tr_t = _to_tensor(tr) if tr is not None else None

                # Slice single frame — params may be (N, D) or (D,)
                def _frame_slice(t, idx):
                    if t.ndim >= 2 and idx < t.shape[0]:
                        return t[idx : idx + 1]
                    if t.ndim == 1:
                        return t.unsqueeze(0)
                    return None

                go_frame = _frame_slice(go_t, frame_idx)
                bp_frame = _frame_slice(bp_t, frame_idx)
                if go_frame is None or bp_frame is None:
                    return None

                # betas: take first row (shape is shared across frames)
                be_frame = be_t[:1] if be_t.ndim >= 2 else be_t.unsqueeze(0)

                tr_frame = None
                if tr_t is not None:
                    tr_frame = _frame_slice(tr_t, frame_idx)

                verts = self._body_model(
                    body_pose=bp_frame,
                    betas=be_frame,
                    global_orient=go_frame,
                    transl=tr_frame,
                )  # (1, V, 3)

                vertices = verts[0].cpu().numpy().astype(np.float32)
                normals = compute_normals(vertices, self._faces)

                result = (vertices, normals)

                # Only cache non-overridden results
                if not override_active:
                    if len(self._vertex_cache) >= _CACHE_MAX:
                        oldest = next(iter(self._vertex_cache))
                        del self._vertex_cache[oldest]
                    self._vertex_cache[cache_key] = result

                return result
        except Exception as e:
            logger.warning("Vertex computation failed (pid=%d, f=%d): %s",
                           person_id, frame_idx, e)
            return None

    # ------------------------------------------------------------------
    # OpenGL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self):
        if not _HAS_GL:
            return
        try:
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glEnable(gl.GL_CULL_FACE)
            gl.glCullFace(gl.GL_BACK)
            gl.glClearColor(0.1, 0.1, 0.12, 1.0)  # dark bg matching theme

            # Compile shader program
            self._shader = QOpenGLShaderProgram(self)
            vert_src = _load_shader_source("mesh.vert", _VERT_SRC)
            frag_src = _load_shader_source("mesh.frag", _FRAG_SRC)

            if not self._shader.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Vertex, vert_src
            ):
                logger.error("Vertex shader failed: %s", self._shader.log())
                return
            if not self._shader.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Fragment, frag_src
            ):
                logger.error("Fragment shader failed: %s", self._shader.log())
                return
            if not self._shader.link():
                logger.error("Shader link failed: %s", self._shader.log())
                return

            # Create VAO + VBOs
            self._vao_id = gl.glGenVertexArrays(1)
            gl.glBindVertexArray(self._vao_id)

            self._vbo_pos = gl.glGenBuffers(1)
            self._vbo_norm = gl.glGenBuffers(1)
            self._vbo_color = gl.glGenBuffers(1)
            self._ebo = gl.glGenBuffers(1)

            gl.glBindVertexArray(0)

            self._gl_ready = True
            logger.info("MeshViewport: OpenGL initialized (GL %s)",
                        gl.glGetString(gl.GL_VERSION))
        except Exception as e:
            logger.error("OpenGL init failed: %s", e)
            self._gl_ready = False

    def resizeGL(self, w: int, h: int):
        if not _HAS_GL or not self._gl_ready:
            return
        gl.glViewport(0, 0, w, h)
        self._update_projection(w, h)

    def paintGL(self):
        if not _HAS_GL:
            return

        if not self._gl_ready:
            return

        # Use QPainter to enable 2D text overlay after GL rendering.
        # beginNativePainting() brackets the raw GL calls; after
        # endNativePainting() we draw joint labels with QPainter.
        painter = QPainter(self)
        painter.beginNativePainting()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Grid floor (orbit mode only, drawn first so mesh occludes it)
        if self._show_grid and self._camera_mode == "orbit":
            self._draw_grid()

        if self._vertices is not None and self._n_indices > 0:
            self._shader.bind()

            # Uniforms
            self._set_mat4("model", self._model_mat)
            self._set_mat4("view", self._view)
            self._set_mat4("projection", self._projection)
            self._set_vec3("light_dir", _LIGHT_DIR)
            self._set_vec3("light_color", _LIGHT_COLOR)
            self._set_vec3("ambient", _AMBIENT)

            gl.glBindVertexArray(self._vao_id)
            gl.glDrawElements(
                gl.GL_TRIANGLES, self._n_indices, gl.GL_UNSIGNED_INT, None
            )
            gl.glBindVertexArray(0)

            self._shader.release()

            # Skeleton overlay (drawn on top of mesh)
            if self._show_skeleton and self._joint_positions is not None:
                self._draw_skeleton()

        painter.endNativePainting()

        # Joint name text overlay (QPainter 2D, after GL rendering)
        if (
            self._show_joint_labels
            and self._show_skeleton
            and self._joint_positions is not None
        ):
            self._draw_joint_labels(painter)

        painter.end()

    # ------------------------------------------------------------------
    # Skeleton GL rendering
    # ------------------------------------------------------------------

    def _draw_skeleton(self):
        """Draw skeleton overlay: bones as GL_LINES, joints as GL_POINTS.

        Uses legacy-ish immediate-mode via temporary VBOs for simplicity,
        reusing the mesh shader with ambient=1 so the skeleton is unlit.
        """
        if not _HAS_GL or not self._gl_ready:
            return

        joints = self._joint_positions  # (52, 3) camera space
        if joints is None:
            return

        self._shader.bind()

        # Set uniforms — use full-bright ambient so skeleton is unlit
        self._set_mat4("model", self._model_mat)
        self._set_mat4("view", self._view)
        self._set_mat4("projection", self._projection)
        self._set_vec3("light_dir", _LIGHT_DIR)
        self._set_vec3("light_color", np.zeros(3, dtype=np.float32))
        self._set_vec3("ambient", np.ones(3, dtype=np.float32))

        # Disable depth test so skeleton renders on top
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)

        # --- Draw bones as GL_LINES ---
        bone_verts = []
        bone_colors = []
        for a, b in BONE_CONNECTIONS:
            if a < len(joints) and b < len(joints):
                bone_verts.append(joints[a])
                bone_verts.append(joints[b])
                bone_colors.append(_BONE_COLOR)
                bone_colors.append(_BONE_COLOR)

        if bone_verts:
            bv = np.array(bone_verts, dtype=np.float32)
            bc = np.array(bone_colors, dtype=np.float32)
            bn = np.zeros_like(bv)  # normals not used (unlit)

            self._draw_primitive(gl.GL_LINES, bv, bn, bc, _BONE_LINE_WIDTH)

        # --- Draw joints as GL_POINTS ---
        jv = joints.astype(np.float32)
        jn = np.zeros_like(jv)
        jc = np.zeros((len(joints), 3), dtype=np.float32)

        for i in range(len(joints)):
            if i == self._selected_joint:
                jc[i] = _ACCENT_COLOR
            elif i < _N_BODY_JOINTS:
                jc[i] = _BODY_JOINT_COLOR
            else:
                jc[i] = _HAND_JOINT_COLOR

        # Draw non-selected joints at normal size
        mask = np.arange(len(joints)) != self._selected_joint
        if np.any(mask):
            self._draw_primitive(
                gl.GL_POINTS, jv[mask], jn[mask], jc[mask],
                point_size=_JOINT_POINT_SIZE,
            )

        # Draw selected joint larger
        if 0 <= self._selected_joint < len(joints):
            si = self._selected_joint
            self._draw_primitive(
                gl.GL_POINTS,
                jv[si:si + 1], jn[si:si + 1], jc[si:si + 1],
                point_size=_SELECTED_JOINT_POINT_SIZE,
            )

        # Restore state
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

        self._shader.release()

    def _draw_grid(self):
        """Draw grid floor in orbit mode as GL_LINES.

        The grid is drawn on the XZ plane at the mesh's foot level (self._grid_y),
        centered around the orbit center's XZ position.  It uses depth testing so
        the mesh properly occludes grid lines behind it.
        """
        if not _HAS_GL or not self._gl_ready:
            return

        positions, colors = compute_grid_lines(
            y=self._grid_y,
            center_x=self._orbit_center[0],
            center_z=self._orbit_center[2],
        )
        if len(positions) == 0:
            return

        self._shader.bind()

        # Uniforms — unlit (full ambient) so grid color is exact
        self._set_mat4("model", np.eye(4, dtype=np.float32))
        self._set_mat4("view", self._view)
        self._set_mat4("projection", self._projection)
        self._set_vec3("light_dir", _LIGHT_DIR)
        self._set_vec3("light_color", np.zeros(3, dtype=np.float32))
        self._set_vec3("ambient", np.ones(3, dtype=np.float32))

        # Grid uses depth test (mesh occludes it) but no face culling
        gl.glDisable(gl.GL_CULL_FACE)

        normals = np.zeros_like(positions)
        self._draw_primitive(gl.GL_LINES, positions, normals, colors, line_width=1.0)

        gl.glEnable(gl.GL_CULL_FACE)

        self._shader.release()

    def _draw_joint_labels(self, painter: QPainter):
        """Draw joint name text labels at projected joint positions.

        Called from paintGL() after endNativePainting(), using QPainter for
        crisp 2D text over the GL-rendered scene.  Label layout is computed
        by the pure helper compute_joint_label_layout().
        """
        joints = self._joint_positions
        if joints is None:
            return

        w = self.width() if self.width() > 0 else 200
        h = self.height() if self.height() > 0 else 150
        mvp = self._projection @ self._view @ self._model_mat
        screen = project_joints_to_screen(joints, mvp, w, h)

        labels = compute_joint_label_layout(
            screen, w, h,
            selected_joint=self._selected_joint,
            body_only=True,
        )
        if not labels:
            return

        font = QFont("sans-serif", _LABEL_FONT_SIZE)
        painter.setFont(font)
        fm = QFontMetrics(font)

        for _idx, name, sx, sy, is_sel in labels:
            text_w = fm.horizontalAdvance(name)
            text_h = fm.height()

            # Position label to the right and slightly above the joint point
            lx = int(sx + _LABEL_OFFSET_X)
            ly = int(sy + _LABEL_OFFSET_Y)

            # Clamp label box within viewport
            lx = max(_LABEL_MARGIN, min(lx, w - text_w - 2 * _LABEL_PADDING_X - _LABEL_MARGIN))
            ly = max(_LABEL_MARGIN + text_h, min(ly, h - _LABEL_MARGIN))

            # Background rect
            bg_color = _LABEL_SELECTED_BG if is_sel else _LABEL_BG_COLOR
            txt_color = _LABEL_SELECTED_TEXT if is_sel else _LABEL_TEXT_COLOR

            bg_x = lx - _LABEL_PADDING_X
            bg_y = ly - text_h - _LABEL_PADDING_Y + fm.descent()
            bg_w = text_w + 2 * _LABEL_PADDING_X
            bg_h = text_h + 2 * _LABEL_PADDING_Y

            painter.setPen(Qt.NoPen)
            painter.setBrush(bg_color)
            painter.drawRoundedRect(bg_x, bg_y, bg_w, bg_h, 2, 2)

            painter.setPen(txt_color)
            painter.drawText(lx, ly, name)

    def _draw_primitive(
        self,
        mode,
        positions: np.ndarray,
        normals: np.ndarray,
        colors: np.ndarray,
        line_width: float = 1.0,
        point_size: float = 1.0,
    ):
        """Draw a GL primitive using temporary buffer uploads.

        Reuses the mesh shader (position/normal/color layout).
        """
        if not _HAS_GL or len(positions) == 0:
            return

        vao = gl.glGenVertexArrays(1)
        vbo_p, vbo_n, vbo_c = gl.glGenBuffers(3)

        gl.glBindVertexArray(vao)

        # Position (location 0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_p)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, positions.nbytes, positions, gl.GL_STREAM_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

        # Normal (location 1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_n)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, normals.nbytes, normals, gl.GL_STREAM_DRAW)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)

        # Color (location 2)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_c)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, colors.nbytes, colors, gl.GL_STREAM_DRAW)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(2)

        if mode == gl.GL_LINES:
            gl.glLineWidth(line_width)
        elif mode == gl.GL_POINTS:
            gl.glPointSize(point_size)

        gl.glDrawArrays(mode, 0, len(positions))

        gl.glBindVertexArray(0)
        gl.glDeleteVertexArrays(1, [vao])
        gl.glDeleteBuffers(3, [vbo_p, vbo_n, vbo_c])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_colors(self) -> np.ndarray:
        """Compute per-vertex RGB colors based on current color mode.

        Returns (V, 3) float32 array.
        """
        V = self._n_vertices

        if self._color_mode == "joint" and self._lbs_weights is not None:
            return compute_joint_colors(self._lbs_weights)

        if self._color_mode == "confidence":
            conf = 0.5  # default mid confidence
            if self._session is not None and self._person_id >= 0:
                track = self._session.person_tracks.get(self._person_id)
                if track is not None:
                    # Try confidence_breakdown["overall"] first, then raw confidences
                    if (
                        track.confidence_breakdown is not None
                        and "overall" in track.confidence_breakdown
                    ):
                        vals = track.confidence_breakdown["overall"]
                        if 0 <= self._current_frame < len(vals):
                            conf = float(vals[self._current_frame])
                    elif track.confidences is not None:
                        if 0 <= self._current_frame < len(track.confidences):
                            conf = float(track.confidences[self._current_frame])
            color = confidence_to_color(conf)
            return np.tile(color, (V, 1)).astype(np.float32)

        # Default: solid skin tone
        return np.tile(_SKIN_COLOR, (V, 1)).astype(np.float32)

    def _refresh_mesh(self):
        """Recompute vertices + joints for current person/frame and trigger repaint."""
        result = self._compute_vertices(self._person_id, self._current_frame)
        if result is not None:
            self._vertices, self._normals = result
            self._n_vertices = len(self._vertices)
            if self._gl_ready:
                self._upload_buffers()
            # Auto-center orbit camera on first mesh load
            if self._camera_mode == "orbit" and not self._orbit_auto_centered:
                self._auto_center_orbit()
                self._update_camera()
        else:
            self._vertices = None
            self._normals = None
            self._n_indices = 0

        # Recompute skeleton joint positions
        self._joint_positions = self._compute_joints()

        if _HAS_GL:
            self.update()

    def _upload_buffers(self):
        """Upload vertices, normals, colors, and faces to GPU."""
        if (
            not self._gl_ready
            or self._vertices is None
            or self._faces is None
        ):
            return

        self.makeCurrent()
        gl.glBindVertexArray(self._vao_id)

        # Position VBO (location 0)
        pos_data = self._vertices.astype(np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_pos)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            pos_data.nbytes,
            pos_data,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

        # Normal VBO (location 1)
        norm_data = self._normals.astype(np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_norm)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            norm_data.nbytes,
            norm_data,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(1)

        # Color VBO (location 2) — computed from current color mode
        colors = self._compute_colors()
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_color)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            colors.nbytes,
            colors,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(2)

        # Element buffer (face indices) — upload once or if faces changed
        if not self._faces_uploaded:
            idx_data = self._faces.astype(np.uint32).flatten()
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
            gl.glBufferData(
                gl.GL_ELEMENT_ARRAY_BUFFER,
                idx_data.nbytes,
                idx_data,
                gl.GL_STATIC_DRAW,
            )
            self._n_indices = len(idx_data)
            self._faces_uploaded = True

        gl.glBindVertexArray(0)
        self.doneCurrent()

    def _update_projection(self, width: int, height: int):
        """Recompute projection from camera mode and session intrinsics."""
        w = max(width, 1)
        h = max(height, 1)
        if self._camera_mode == "orbit":
            self._projection = perspective_fov(_ORBIT_DEFAULT_FOV, w / h)
        else:
            if self._session and self._session.camera_K is not None:
                K = self._session.camera_K
            elif self._session and self._session.img_width > 0:
                K = estimate_K(self._session.img_width, self._session.img_height)
            else:
                K = estimate_K(w, h)
            self._projection = k_to_projection(K, w, h)

    def _set_mat4(self, name: str, mat: np.ndarray):
        loc = self._shader.uniformLocation(name)
        if loc >= 0:
            gl.glUniformMatrix4fv(
                loc, 1, gl.GL_TRUE, mat.astype(np.float32)
            )

    def _set_vec3(self, name: str, vec: np.ndarray):
        loc = self._shader.uniformLocation(name)
        if loc >= 0:
            gl.glUniform3f(loc, float(vec[0]), float(vec[1]), float(vec[2]))
