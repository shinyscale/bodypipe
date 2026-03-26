"""SMPL-X mesh viewport — QOpenGLWidget with Phong shading + skeleton overlay.

Renders the SMPL-X body mesh for the selected person at the current frame.
The body model is loaded once; vertices are recomputed per frame from the
session's SMPL-X parameters (torch forward pass → numpy), uploaded to
dynamic VBOs, and rendered with Phong shading matching shaders/mesh.vert
and shaders/mesh.frag.

Phase 3.3 adds skeleton overlay (bones as GL_LINES, joints as GL_POINTS)
and click-to-select joint picking via screen-space distance.

FBO color-coded joint picking renders each joint as a uniquely colored
GL_POINT into an offscreen FBO, reads the pixel at the click position,
and decodes the joint index.  Depth-tested so the front-most joint wins
when multiple joints overlap on screen — more accurate than screen-space
distance for overlapping joints.  Toggled via set_picking_mode().
"""

from __future__ import annotations

import enum
import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QPainter, QFont, QColor, QFontMetrics, QImage
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QMenu

from models.session import Session
from theme import COLORS


class RenderMode(enum.Enum):
    """Viewport rendering quality level.

    WIREFRAME skips the SMPL-X mesh entirely — only skeleton joints and bones
    are drawn.  This is the fastest mode because it avoids the expensive
    torch forward pass for vertex computation.

    FAST renders the mesh with ambient-only shading (no Phong diffuse term),
    suitable for real-time scrubbing when mesh visibility is still desired.

    FULL is the default — full SMPL-X mesh with Phong shading and all overlays.
    """
    WIREFRAME = "wireframe"
    FAST = "fast"
    FULL = "full"

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

import platform as _platform
_IS_WSL = False
try:
    _IS_WSL = "microsoft" in _platform.uname().release.lower()
except Exception:
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
# Skeleton data — sourced from models/skeleton.py (single source of truth)
# ---------------------------------------------------------------------------

from models.skeleton import SMPLX_SKELETON as _SKEL, SOMA_SKELETON as _SOMA_SKEL

# Module-level aliases for backward compatibility — existing code and tests
# import these names directly from this module.
JOINT_NAMES = list(_SKEL.joint_names)
JOINT_PARENTS = list(_SKEL.joint_parents)
DEFAULT_OFFSETS = _SKEL.default_offsets
BONE_CONNECTIONS = list(_SKEL.bone_connections)
_N_BODY_JOINTS = _SKEL.n_body_joints
_JOINT_PALETTE = _SKEL.joint_palette

# Native SMPL-X exposes 55 joints including jaw/eyes. The viewport's 52-joint
# skeleton skips those 3 joints and expects the hand chains to follow directly.
_SMPLX_NATIVE_TO_VIEWPORT = list(range(22)) + list(range(25, 40)) + list(range(40, 55))

# Joint picking threshold in pixels
_JOINT_PICK_THRESHOLD = 20.0

# Accent color for selected joint highlight (matches app theme — amber)
_ACCENT_COLOR = np.array([0.792, 0.584, 0.180], dtype=np.float32)  # #ca952e

# Brighter accent for the selected joint itself (within the chain)
_SELECTED_ACCENT_COLOR = np.array([1.0, 0.78, 0.25], dtype=np.float32)  # brighter amber

# Chain bone color (amber, same as accent — distinguishes chain from non-chain)
_CHAIN_BONE_COLOR = np.array([0.792, 0.584, 0.180], dtype=np.float32)

# Chain joint color (slightly dimmer than selected, brighter than default)
_CHAIN_JOINT_COLOR = np.array([0.85, 0.65, 0.22], dtype=np.float32)

# Bone color (light gray)
_BONE_COLOR = np.array([0.7, 0.7, 0.7], dtype=np.float32)

# Joint colors: body joints = white, hand joints = slightly dimmer
_BODY_JOINT_COLOR = np.array([1.0, 1.0, 1.0], dtype=np.float32)
_HAND_JOINT_COLOR = np.array([0.6, 0.6, 0.6], dtype=np.float32)

# GL point/line sizes for skeleton rendering
_JOINT_POINT_SIZE = 6.0
_SELECTED_JOINT_POINT_SIZE = 12.0
_CHAIN_JOINT_POINT_SIZE = 8.0
_BONE_LINE_WIDTH = 2.0
_CHAIN_BONE_LINE_WIDTH = 3.0

# Joint label rendering constants
_LABEL_FONT_SIZE = 9
_LABEL_BG_COLOR = QColor(0, 0, 0, 180)       # semi-transparent black
_LABEL_TEXT_COLOR = QColor(220, 220, 220)      # light gray
_LABEL_SELECTED_BG = QColor(202, 149, 46, 200) # accent with alpha
_LABEL_SELECTED_TEXT = QColor(255, 255, 255)   # white
_LABEL_PADDING_X = 3   # horizontal padding inside label bg
_LABEL_PADDING_Y = 1   # vertical padding inside label bg
_LABEL_OFFSET_X = 8    # pixels right of joint point
_LABEL_OFFSET_Y = -4   # pixels above joint point center
_LABEL_MARGIN = 4       # viewport edge margin (labels clamped inside)


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


def _forward_kinematics_soma(
    params: dict, frame_idx: int, offsets_override: dict | None = None,
) -> np.ndarray:
    """Compute 3D joint positions for SOMA's unified poses tensor.

    Parameters
    ----------
    params : dict with keys poses (N, 77, 3), transl (N, 3)
    frame_idx : which frame to compute
    offsets_override : optional dict mapping joint name → (3,) rest-pose offset

    Returns
    -------
    positions : (77, 3) float64 — joint positions in camera space
    """
    from scipy.spatial.transform import Rotation
    from models.skeleton import SOMA_SKELETON

    n_joints = SOMA_SKELETON.n_joints
    positions = np.zeros((n_joints, 3))
    accumulated_R = np.zeros((n_joints, 3, 3))

    src = offsets_override or SOMA_SKELETON.default_offsets
    offsets = np.zeros((n_joints, 3))
    for i, name in enumerate(SOMA_SKELETON.joint_names):
        offsets[i] = src.get(name, [0, 0, 0])

    poses = np.asarray(params["poses"])
    tr = params.get("transl")
    if tr is not None:
        tr = np.asarray(tr)

    # Root (joint 0 = global orient)
    root_aa = poses[frame_idx, 0] if poses.ndim == 3 else poses[0]
    accumulated_R[0] = Rotation.from_rotvec(np.asarray(root_aa).ravel()[:3]).as_matrix()
    if tr is not None and tr.ndim >= 2 and frame_idx < tr.shape[0]:
        positions[0] = tr[frame_idx]
    elif tr is not None and tr.ndim == 1:
        positions[0] = tr

    parents = SOMA_SKELETON.joint_parents
    for j in range(1, n_joints):
        parent = parents[j]
        if poses.ndim == 3 and frame_idx < poses.shape[0] and j < poses.shape[1]:
            rot_aa = poses[frame_idx, j]
        else:
            rot_aa = np.zeros(3)

        R_local = Rotation.from_rotvec(np.asarray(rot_aa).ravel()[:3]).as_matrix()
        accumulated_R[j] = accumulated_R[parent] @ R_local
        positions[j] = positions[parent] + accumulated_R[parent] @ offsets[j]

    return positions


def forward_kinematics(
    params: dict, frame_idx: int, offsets_override: dict | None = None,
) -> np.ndarray:
    """Compute 3D joint positions in camera space for one frame.

    Parameters
    ----------
    params : dict with keys global_orient (N,3), body_pose (N,21*3 or N,21,3),
             transl (N,3). Optional: left_hand_pose (N,15,3), right_hand_pose (N,15,3).
    frame_idx : which frame to compute
    offsets_override : optional dict mapping joint name → (3,) rest-pose offset.
        When provided, uses shape-dependent bone lengths instead of DEFAULT_OFFSETS.

    Returns
    -------
    positions : (52, 3) float64 — joint positions in camera space
    """
    # SOMA path: unified poses tensor (N, J, 3)
    if "poses" in params and "body_pose" not in params:
        return _forward_kinematics_soma(params, frame_idx, offsets_override)

    from scipy.spatial.transform import Rotation

    n_joints = len(JOINT_NAMES)
    positions = np.zeros((n_joints, 3))
    accumulated_R = np.zeros((n_joints, 3, 3))

    src = offsets_override or DEFAULT_OFFSETS
    offsets = np.zeros((n_joints, 3))
    for i, name in enumerate(JOINT_NAMES):
        offsets[i] = src.get(name, [0, 0, 0])

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


def _coerce_smplx_joints_to_viewport_order(joints: np.ndarray) -> np.ndarray:
    """Map native 55-joint SMPL-X output to the viewport's 52-joint layout."""
    joints = np.asarray(joints)
    if joints.ndim != 2:
        return joints
    if joints.shape[0] >= max(_SMPLX_NATIVE_TO_VIEWPORT) + 1:
        return joints[_SMPLX_NATIVE_TO_VIEWPORT]
    return joints


def _frame_slice_array(arr, idx: int):
    """Return a single-frame slice from an array-like param."""
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim >= 2 and idx < arr.shape[0]:
        return arr[idx : idx + 1]
    if arr.ndim == 1:
        return arr[None]
    return None


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


# ---------------------------------------------------------------------------
# Joint chain / region helpers (pure functions — no Qt/GL dependency)
# ---------------------------------------------------------------------------

# Left↔Right and region data — sourced from skeleton registry.
_LR_PAIRS = _SKEL.lr_pairs
_JOINT_REGIONS = _SKEL.joint_regions

# Reverse lookup: joint index → region name
_JOINT_TO_REGION: dict[int, str] = {}
for _region_name, _region_joints in _JOINT_REGIONS.items():
    for _j in _region_joints:
        _JOINT_TO_REGION[_j] = _region_name


def get_joint_chain(joint_idx: int) -> list[int]:
    """Walk JOINT_PARENTS from *joint_idx* to the root, returning the chain.

    The returned list starts with *joint_idx* and ends with the root (Pelvis=0).
    Returns an empty list if *joint_idx* is out of range.
    """
    if joint_idx < 0 or joint_idx >= len(JOINT_PARENTS):
        return []
    chain = [joint_idx]
    current = joint_idx
    while JOINT_PARENTS[current] >= 0:
        current = JOINT_PARENTS[current]
        chain.append(current)
    return chain


def get_joint_chain_bones(joint_idx: int) -> set[tuple[int, int]]:
    """Return the set of (parent, child) bone pairs along the chain to root.

    Each pair is ordered (parent, child) matching BONE_CONNECTIONS convention.
    """
    chain = get_joint_chain(joint_idx)
    bones = set()
    for i in range(len(chain) - 1):
        child = chain[i]
        parent = chain[i + 1]
        # Normalize to (min, max) for easy lookup against BONE_CONNECTIONS
        bones.add((min(parent, child), max(parent, child)))
    return bones


def get_joint_siblings(joint_idx: int) -> list[int]:
    """Return joints sharing the same parent as *joint_idx* (excluding self).

    Returns an empty list for the root or out-of-range indices.
    """
    if joint_idx < 0 or joint_idx >= len(JOINT_PARENTS):
        return []
    parent = JOINT_PARENTS[joint_idx]
    if parent < 0:
        return []
    siblings = []
    for j, p in enumerate(JOINT_PARENTS):
        if p == parent and j != joint_idx:
            siblings.append(j)
    return siblings


def get_opposite_joint(joint_idx: int) -> int | None:
    """Return the left↔right mirror of *joint_idx*, or None for center joints.

    Body joints use _LR_PAIRS; hand joints map 22+i ↔ 37+i.
    """
    if joint_idx in _LR_PAIRS:
        return _LR_PAIRS[joint_idx]
    # Hand joints: left 22-36 ↔ right 37-51
    if 22 <= joint_idx <= 36:
        return joint_idx + 15
    if 37 <= joint_idx <= 51:
        return joint_idx - 15
    return None


def get_joint_region(joint_idx: int) -> list[int]:
    """Return all joints in the same body region as *joint_idx*.

    Regions: spine, left_leg, right_leg, left_arm, right_arm, left_hand, right_hand.
    Returns an empty list if the joint is not in any defined region.
    """
    region = _JOINT_TO_REGION.get(joint_idx)
    if region is None:
        return []
    return list(_JOINT_REGIONS[region])


def encode_joint_id(joint_idx: int) -> tuple[int, int, int]:
    """Encode a joint index as a unique RGB color for FBO picking.

    Joint 0 → (1, 0, 0), Joint 1 → (2, 0, 0), etc.
    Background (no joint) is (0, 0, 0).  Supports up to 16M joints.
    """
    encoded = joint_idx + 1  # 0 reserved for background
    r = encoded & 0xFF
    g = (encoded >> 8) & 0xFF
    b = (encoded >> 16) & 0xFF
    return (r, g, b)


def decode_joint_id(r: int, g: int, b: int) -> int | None:
    """Decode an RGB pixel from FBO picking back to a joint index.

    Returns None if the pixel is background (0, 0, 0).
    """
    encoded = r | (g << 8) | (b << 16)
    if encoded == 0:
        return None
    return encoded - 1


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
    joint_names: list[str] | None = None,
    n_body_joints: int | None = None,
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
    names = joint_names if joint_names is not None else JOINT_NAMES
    n_body = n_body_joints if n_body_joints is not None else _N_BODY_JOINTS
    n_joints = min(len(joints_2d), len(names))
    max_idx = n_body if body_only else n_joints

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
        labels.append((i, names[i], sx, sy, is_sel))

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
# HUD overlay — semi-transparent info display on viewport (Phase 10)
# ---------------------------------------------------------------------------

_HUD_HIDE_DELAY_MS = 2000


class _ViewportHUD(QWidget):
    """Semi-transparent HUD overlay showing frame, speed, mode, person, FPS.

    Why overlay on viewport: Mocha/Nuke pattern — at-a-glance status without
    taking dock space.  Auto-hides after 2s of mouse inactivity; permanently
    visible while the mouse hovers over it.  Toggle via View > Toggle HUD.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)

        from PySide6.QtCore import QTimer

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(_HUD_HIDE_DELAY_MS)
        self._hide_timer.timeout.connect(self.hide)

        self._frame_text = ""
        self._speed_text = "1x"
        self._mode_text = "Navigate"
        self._person_text = ""
        self._fps_text = ""

        self.setFixedSize(180, 90)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # Semi-transparent dark background
        p.setBrush(QColor(0, 0, 0, 160))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 6, 6)

        # Draw text lines
        p.setPen(QColor(220, 220, 220))
        font = QFont("monospace", 9)
        p.setFont(font)
        fm = QFontMetrics(font)
        line_height = fm.height() + 2
        x = 8
        y = 4 + fm.ascent()

        lines = []
        if self._frame_text:
            lines.append(self._frame_text)
        if self._speed_text:
            lines.append(f"Speed: {self._speed_text}")
        if self._mode_text:
            lines.append(f"Mode: {self._mode_text}")
        if self._person_text:
            lines.append(self._person_text)
        if self._fps_text:
            lines.append(self._fps_text)

        for line in lines:
            p.drawText(x, y, line)
            y += line_height

        p.end()

    def update_info(
        self,
        frame: int | None = None,
        total_frames: int | None = None,
        speed: float | None = None,
        mode: str | None = None,
        person: int | None = None,
        fps: float | None = None,
    ):
        """Update HUD fields and repaint. Only non-None args are changed."""
        if frame is not None and total_frames is not None:
            self._frame_text = f"Frame: {frame} / {total_frames}"
        elif frame is not None:
            self._frame_text = f"Frame: {frame}"
        if speed is not None:
            label = f"{int(speed)}x" if speed == int(speed) else f"{speed}x"
            self._speed_text = label
        if mode is not None:
            self._mode_text = mode
        if person is not None:
            self._person_text = f"Person: {person}" if person >= 0 else ""
        if fps is not None:
            self._fps_text = f"FPS: {fps:.1f}"
        self.update()

    def show_with_timer(self):
        """Show the HUD and (re)start the auto-hide timer."""
        self.show()
        self.raise_()
        self._hide_timer.start()

    def enterEvent(self, event):
        self._hide_timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hide_timer.start()
        super().leaveEvent(event)


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
    gl_rendered = Signal()  # emitted after paintGL completes

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

        # SOMA body model (loaded lazily, separate from SMPL-X)
        self._soma_model: object | None = None
        self._soma_model_loaded: bool = False
        self._soma_faces: np.ndarray | None = None  # (F, 3) int32 — SOMA topology
        self._uploaded_face_source: str = ""  # "smplx" or "soma" — which EBO is active

        # Shape-dependent FK offsets cache: {person_id: dict[joint_name, (3,)]}
        self._shape_offsets_cache: dict[int, dict[str, list[float]]] = {}

        # Vertex color mode ("solid", "joint", "confidence")
        self._color_mode: str = "solid"

        # Camera matrices
        self._projection = np.eye(4, dtype=np.float32)
        self._view = _CV_TO_GL.copy()
        self._model_mat = np.eye(4, dtype=np.float32)

        # Vertex cache keyed by person/frame/render context.
        self._vertex_cache: dict[
            tuple[object, ...], tuple[np.ndarray, np.ndarray]
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
        self._orbit_center_override: np.ndarray | None = None

        # Mouse tracking for orbit interaction
        self._mouse_last_pos: tuple[int, int] | None = None

        # Skeleton state (Phase 3.3)
        self._joint_positions: np.ndarray | None = None  # (J, 3) camera-space
        self._selected_joint: int = -1  # -1 = no selection
        self._show_skeleton: bool = True  # whether to draw skeleton overlay
        self._active_skel = _SKEL  # skeleton def for current person (SMPLX or SOMA)
        self._data_is_global: bool = False  # True = world-space (Y-up), skip CV→GL flip
        self._skeleton_heatmap: bool = False  # confidence heatmap on skeleton
        self._show_all_persons: bool = True  # render all persons' skeletons
        self._all_joint_positions: dict[int, np.ndarray] = {}  # pid → (J, 3)
        self._all_active_skels: dict[int, object] = {}  # pid → skeleton def

        # Pose override for real-time preview (Phase 3.4)
        # dict with keys: frame_idx (int), global_orient (3,) optional,
        #                  body_pose {int: (3,)} optional
        self._pose_override: dict | None = None

        # Joint label overlay state (QPainter text over GL)
        self._show_joint_labels: bool = False  # off by default, toggled by user

        # Grid floor state (orbit mode reference plane)
        self._show_grid: bool = True  # visible by default in orbit mode
        self._grid_y: float = 0.0  # Y level of the grid in GL space

        # FBO picking state (color-coded render pass for accurate joint selection)
        self._picking_mode: str = "fbo"  # "screen" or "fbo"
        self._pick_fbo_id: int = 0
        self._pick_rbo_color: int = 0
        self._pick_rbo_depth: int = 0
        self._pick_fbo_size: tuple[int, int] = (0, 0)

        # Video frame background for in-camera composite
        self._video_frame: np.ndarray | None = None  # RGB (H, W, 3) uint8

        # Render quality mode (wireframe/fast/full)
        self._render_mode: RenderMode = RenderMode.FULL
        # Saved mode before auto-switch during scrubbing (None = not scrubbing)
        self._pre_scrub_mode: RenderMode | None = None

        # Status message for fallback rendering
        self._status_msg: str = ""

        self.setMinimumSize(QSize(200, 150))

        if _HAS_GL:
            from PySide6.QtGui import QSurfaceFormat

            # Surface format is set globally in main.py (no alpha buffer)
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        else:
            self._status_msg = "OpenGL not available"
            self._setup_fallback()

        # HUD overlay (Phase 10) — positioned bottom-right
        self._hud = _ViewportHUD(self)
        self._hud.hide()
        self._hud_enabled = True  # toggleable via View menu
        self.setMouseTracking(True)  # enable mouseMoveEvent without button held

    def _setup_fallback(self):
        """Show a label when GL is not available."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        self._fallback_label = QLabel(self._status_msg or "3D Viewport")
        self._fallback_label.setAlignment(Qt.AlignCenter)
        self._fallback_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 14px;")
        layout.addWidget(self._fallback_label)

    def resizeEvent(self, event):
        """Reposition HUD overlay on resize."""
        super().resizeEvent(event)
        self._position_hud()

    def _position_hud(self):
        """Place HUD in bottom-right corner with margin."""
        margin = 10
        hud_w = self._hud.width()
        hud_h = self._hud.height()
        self._hud.move(self.width() - hud_w - margin, self.height() - hud_h - margin)

    # ------------------------------------------------------------------
    # HUD overlay API (Phase 10)
    # ------------------------------------------------------------------

    def set_hud_visible(self, visible: bool):
        """Enable or disable the HUD overlay (View menu toggle)."""
        self._hud_enabled = visible
        if visible:
            self._hud.show_with_timer()
        else:
            self._hud.hide()

    def set_hud_mode(self, mode_name: str):
        """Update the mode display on the HUD."""
        self._hud.update_info(mode=mode_name)
        if self._hud_enabled:
            self._hud.show_with_timer()

    def update_hud(
        self,
        frame: int | None = None,
        total_frames: int | None = None,
        speed: float | None = None,
        person: int | None = None,
        fps: float | None = None,
    ):
        """Update HUD fields and briefly show if enabled."""
        self._hud.update_info(
            frame=frame, total_frames=total_frames,
            speed=speed, person=person, fps=fps,
        )
        if self._hud_enabled:
            self._hud.show_with_timer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_session(self, session: Session):
        """Bind session data source."""
        self._session = session
        self._vertex_cache.clear()
        self._shape_offsets_cache.clear()

    def set_person(self, person_id: int):
        """Select which person's mesh to display."""
        if person_id == self._person_id:
            return
        self._person_id = person_id
        if self._camera_mode == "orbit" and self._orbit_center_override is None:
            self._orbit_auto_centered = False
        # Update active skeleton def and coordinate space based on track data
        if self._session is not None:
            track = self._session.person_tracks.get(person_id)
            if track is not None and track.body_model_type == "soma":
                self._active_skel = _SOMA_SKEL
            else:
                self._active_skel = _SKEL
            # Reset — _data_is_global is set dynamically by _compute_joints
            # based on whether world-grounding transform succeeds
            self._data_is_global = False
        self._refresh_mesh()

    def center_orbit_on_selected(self):
        """Auto-center orbit around the currently selected person's geometry."""
        self._orbit_center_override = None
        self._orbit_auto_centered = False
        if self._camera_mode != "orbit":
            return
        self._auto_center_orbit()
        self._update_camera()
        self.camera_changed.emit(self._camera_state())

    def set_orbit_center(self, center):
        """Set a manual orbit pivot in GL-space coordinates."""
        center_arr = np.asarray(center, dtype=np.float32).reshape(3)
        self._orbit_center_override = center_arr.copy()
        self._orbit_center = center_arr.copy()
        self._orbit_auto_centered = True
        if self._camera_mode != "orbit":
            return
        self._update_camera()
        self.camera_changed.emit(self._camera_state())

    def clear_orbit_center_override(self, recenter: bool = True):
        """Clear any manual orbit pivot override."""
        self._orbit_center_override = None
        self._orbit_auto_centered = False
        if self._camera_mode != "orbit":
            return
        if recenter:
            self._auto_center_orbit()
        self._update_camera()
        self.camera_changed.emit(self._camera_state())

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
            self._orbit_auto_centered = False
            # Re-compute joints with orbit-mode world params BEFORE
            # auto-centering so the orbit center uses correct positions.
            self._refresh_mesh()
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

    def set_skeleton_heatmap(self, enabled: bool):
        """Toggle confidence heatmap coloring on the skeleton overlay.

        When enabled, joints and bones are colored by per-frame confidence
        using the red→yellow→green gradient from confidence_to_color().
        Chain highlighting and selected-joint accent still take priority.
        """
        if enabled == self._skeleton_heatmap:
            return
        self._skeleton_heatmap = enabled
        if _HAS_GL:
            self.update()

    def set_picking_mode(self, mode: str):
        """Set joint picking mode ('screen' or 'fbo').

        'screen' uses screen-space distance (fast, less accurate for overlapping).
        'fbo' uses an offscreen color-coded render pass (more accurate).
        """
        if mode not in ("screen", "fbo"):
            return
        self._picking_mode = mode

    def set_show_grid(self, show: bool):
        """Toggle grid floor visibility in orbit mode."""
        if show == self._show_grid:
            return
        self._show_grid = show
        if _HAS_GL:
            self.update()

    def set_video_frame(self, frame: np.ndarray | None):
        """Set the video frame to render as background in in-camera mode.

        Parameters
        ----------
        frame : (H, W, 3) uint8 RGB array, or None to clear.
        """
        if frame is not None:
            self._video_frame = frame.copy()
        else:
            self._video_frame = None
        if _HAS_GL:
            self.update()

    def set_show_joint_labels(self, show: bool):
        """Toggle joint name text labels drawn over the skeleton."""
        if show == self._show_joint_labels:
            return
        self._show_joint_labels = show
        if _HAS_GL:
            self.update()

    def set_render_mode(self, mode: str | RenderMode):
        """Set viewport render quality ('wireframe', 'fast', or 'full').

        In wireframe mode only skeleton joints/bones are drawn — the SMPL-X
        mesh forward pass is skipped entirely for maximum scrubbing speed.
        In fast mode the mesh is rendered with ambient-only shading.
        In full mode (default) full Phong shading is applied.
        """
        if isinstance(mode, str):
            try:
                mode = RenderMode(mode)
            except ValueError:
                return
        if mode == self._render_mode:
            return
        self._render_mode = mode
        # Refresh mesh data — in wireframe mode we can skip vertex computation
        self._refresh_mesh()

    def set_scrubbing(self, active: bool):
        """Auto-switch to wireframe during active scrubbing/playback.

        When *active* is True the current render mode is saved and the
        viewport switches to wireframe (skeleton only — SOMA/SMPL-X forward
        pass is too slow for real-time on CPU).  When *active* becomes False
        the previous mode is restored, re-computing the mesh for the current
        frame.
        """
        if active:
            if self._pre_scrub_mode is None:
                self._pre_scrub_mode = self._render_mode
                if self._render_mode != RenderMode.WIREFRAME:
                    self._render_mode = RenderMode.WIREFRAME
                    self._refresh_mesh()
        else:
            if self._pre_scrub_mode is not None:
                restore = self._pre_scrub_mode
                self._pre_scrub_mode = None
                if restore != self._render_mode:
                    self._render_mode = restore
                    self._refresh_mesh()

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
            self.invalidate_cache(self._person_id, frame_idx)
        self._refresh_mesh()

    def invalidate_cache(
        self,
        person_id: int | None = None,
        frame_idx: int | None = None,
        *,
        include_shape: bool = False,
    ):
        """Invalidate vertex cache entries.

        If both person_id and frame_idx are given, removes that single entry.
        Otherwise clears the entire cache.
        """
        if person_id is not None and frame_idx is not None:
            for key in list(self._vertex_cache.keys()):
                if len(key) >= 2 and key[0] == person_id and key[1] == frame_idx:
                    del self._vertex_cache[key]
        elif person_id is not None:
            for key in list(self._vertex_cache.keys()):
                if len(key) >= 1 and key[0] == person_id:
                    del self._vertex_cache[key]
            if include_shape:
                self._shape_offsets_cache.pop(person_id, None)
        elif frame_idx is not None:
            for key in list(self._vertex_cache.keys()):
                if len(key) >= 2 and key[1] == frame_idx:
                    del self._vertex_cache[key]
        else:
            self._vertex_cache.clear()
            if include_shape:
                self._shape_offsets_cache.clear()

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

    def _params_have_world_motion(self, params: dict | None) -> bool:
        """Return True when params include world-space root pose + translation."""
        if params is None:
            return False
        return (
            params.get("global_orient_world") is not None
            and params.get("transl_world") is not None
        )

    def _resolve_track_render_context(
        self,
        track,
        *,
        apply_pose_override: bool = False,
    ) -> dict | None:
        """Resolve the active params and coordinate-space flags for a track.

        This is the single source of truth for viewport rendering decisions:
        selected skeleton, active params, and whether orbit mode should treat
        the data as already world-grounded (Y-up, no CV->GL model flip).
        """
        if track is None:
            return None

        source = "smplx"
        params = track.smplx_params
        if track.soma_params is not None and track.smplx_params is None:
            source = "soma"
            params = track.soma_params

        if params is None:
            return None

        active_params = params
        is_global = self._camera_mode == "orbit" and self._params_have_world_motion(params)
        if is_global:
            active_params = dict(params)
            active_params["global_orient"] = np.asarray(
                params["global_orient_world"], dtype=np.float32
            )
            active_params["transl"] = np.asarray(
                params["transl_world"], dtype=np.float32
            )
            if source == "soma" and active_params.get("poses") is not None:
                poses = np.asarray(active_params["poses"], dtype=np.float32)
                if poses.ndim == 3 and poses.shape[0] == len(active_params["transl"]):
                    poses = poses.copy()
                    poses[:, 0] = active_params["global_orient"]
                    active_params["poses"] = poses

        if apply_pose_override:
            active_params = self._apply_override_to_params(
                active_params, self._current_frame
            )

        return {
            "source": source,
            "params": active_params,
            "is_global": is_global,
            "needs_cv_to_gl": not is_global,
            "skeleton": _SOMA_SKEL if source == "soma" else _SKEL,
        }

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
            # Global-space data (GEM-X) is already Y-up; camera-space data
            # (GVHMR) needs _CV_TO_GL to flip Y/Z from CV to GL convention.
            if self._data_is_global:
                self._model_mat = np.eye(4, dtype=np.float32)
            else:
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
        self._orbit_center_override = None
        self._orbit_auto_centered = False
        self._auto_center_orbit()

    def _auto_center_orbit(self):
        """Set orbit center to the current mesh/skeleton centroid in GL space."""
        # Use vertices if available, otherwise fall back to joint positions
        pts = self._vertices
        if pts is None:
            pts = self._joint_positions
        if pts is None:
            return

        # Global-space data is already Y-up;
        # camera-space data needs Y/Z flip to GL convention.
        need_flip = not self._data_is_global
        gl_pts = pts.copy()
        if need_flip:
            gl_pts[:, 1] *= -1
            gl_pts[:, 2] *= -1

        foot_y = float(np.min(gl_pts[:, 1]))
        head_y = float(np.max(gl_pts[:, 1]))

        if self._data_is_global:
            # World-grounded: center between ground (Y=0) and head so
            # characters appear standing on the grid, not floating.
            self._grid_y = 0.0
            center_y = (0.0 + head_y) / 2.0
        else:
            # Camera space: grid at feet, center at midpoint
            self._grid_y = foot_y
            center_y = (foot_y + head_y) / 2.0

        if self._orbit_center_override is not None:
            self._orbit_center = self._orbit_center_override.astype(np.float32).copy()
            self._orbit_auto_centered = True
            return

        centroid = gl_pts.mean(axis=0).copy()
        centroid[1] = center_y
        self._orbit_center = centroid.astype(np.float32)

        # Distance: ensure both characters + ground are visible
        extent = np.max(np.linalg.norm(gl_pts - self._orbit_center, axis=1))
        if self._data_is_global:
            # Include ground plane in extent calculation
            ground_dist = np.linalg.norm(
                np.array([centroid[0], 0.0, centroid[2]]) - self._orbit_center
            )
            extent = max(extent, ground_dist)
        self._orbit_distance = max(float(extent) * 2.5, 1.0)
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
        event.accept()

    def mouseMoveEvent(self, event):
        """Update orbit/pan during drag; show HUD on movement."""
        event.accept()
        # Show HUD on any mouse activity over the viewport
        if self._hud_enabled:
            self._hud.show_with_timer()
        if self._camera_mode != "orbit" or self._mouse_last_pos is None:
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

    def mouseReleaseEvent(self, event):
        """End orbit/pan drag."""
        self._mouse_last_pos = None
        event.accept()

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

    def contextMenuEvent(self, event):
        """Right-click context menu for joint selection shortcuts.

        Offers Select Chain, Select Siblings, Select Opposite, Select Region
        based on the joint nearest to the click position.
        """
        # Find the joint under or nearest to the right-click
        target = self._pick_joint(event.pos().x(), event.pos().y())
        if target is None:
            # No joint near click — use currently selected joint
            target = self._selected_joint
        if target < 0 or target >= len(JOINT_NAMES):
            super().contextMenuEvent(event)
            return

        joint_name = JOINT_NAMES[target]
        menu = QMenu(self)

        # Select Chain — highlight the chain from this joint to root
        chain = get_joint_chain(target)
        chain_names = " → ".join(JOINT_NAMES[j] for j in chain[:4])
        if len(chain) > 4:
            chain_names += " → …"
        act_chain = menu.addAction(f"Select Chain ({chain_names})")

        # Select Siblings — joints sharing same parent
        siblings = get_joint_siblings(target)
        act_siblings = None
        if siblings:
            sib_names = ", ".join(JOINT_NAMES[s] for s in siblings)
            act_siblings = menu.addAction(f"Select Siblings ({sib_names})")

        # Select Opposite — L↔R mirror
        opposite = get_opposite_joint(target)
        act_opposite = None
        if opposite is not None:
            act_opposite = menu.addAction(f"Select Opposite ({JOINT_NAMES[opposite]})")

        # Select Region — all joints in same body region
        region = get_joint_region(target)
        act_region = None
        if region:
            region_name = _JOINT_TO_REGION.get(target, "")
            act_region = menu.addAction(f"Select Region ({region_name})")

        act_center_selected = None
        act_center_joint = None
        act_reset_center = None
        if self._camera_mode == "orbit":
            menu.addSeparator()
            act_center_selected = menu.addAction("Center Orbit On Selected Person")
            if (
                self._joint_positions is not None
                and 0 <= target < len(self._joint_positions)
            ):
                act_center_joint = menu.addAction(f"Center Orbit On {JOINT_NAMES[target]}")
            act_reset_center = menu.addAction("Reset Orbit Center")

        action = menu.exec_(event.globalPos())
        if action is None:
            return

        if action == act_center_selected:
            self.center_orbit_on_selected()
        elif action == act_center_joint and self._joint_positions is not None:
            center = np.asarray(self._joint_positions[target], dtype=np.float32).copy()
            if not self._data_is_global:
                center[1] *= -1.0
                center[2] *= -1.0
            self.set_orbit_center(center)
        elif action == act_reset_center:
            self.clear_orbit_center_override(recenter=True)
        elif action == act_chain:
            # Select the root of the chain — visually highlights the full path
            # (The chain is highlighted via _draw_skeleton whenever the
            # selected joint is set; selecting the tip highlights the whole chain.)
            self._selected_joint = target
            self.joint_clicked.emit(target)
        elif action == act_siblings and siblings:
            # Select the first sibling (user can right-click again to pick others)
            self._selected_joint = siblings[0]
            self.joint_clicked.emit(siblings[0])
        elif action == act_opposite and opposite is not None:
            self._selected_joint = opposite
            self.joint_clicked.emit(opposite)
        elif action == act_region and region:
            # Select first joint in region (chain will highlight from there)
            self._selected_joint = target
            self.joint_clicked.emit(target)

        if _HAS_GL:
            self.update()

    # ------------------------------------------------------------------
    # Skeleton: joint computation + joint picking
    # ------------------------------------------------------------------

    def _get_shape_offsets(self, person_id: int) -> dict[str, list[float]] | None:
        """Compute shape-dependent FK offsets for a person from the body model.

        Uses the SOMA or SMPL-X body model at zero pose with the person's
        identity parameters to get rest-pose joint positions, then computes
        parent-relative offsets.  Results are cached per person_id.

        Returns dict mapping joint name → [x, y, z] offset, or None if
        the body model is unavailable.
        """
        if person_id in self._shape_offsets_cache:
            return self._shape_offsets_cache[person_id]

        if self._session is None:
            return None
        track = self._session.person_tracks.get(person_id)
        if track is None:
            return None

        try:
            import torch

            offsets_dict: dict[str, list[float]] = {}

            if track.soma_params is not None and self._load_soma_model():
                # SOMA path: use SomaLayer.get_skeleton() for rest-pose joints
                from models.skeleton import SOMA_SKELETON
                sp = track.soma_params
                ic = sp.get("identity_coeffs")
                sc = sp.get("scale_params")

                n_coeffs = 128  # SOMA PCA identity dimension
                if ic is not None:
                    ic_t = torch.tensor(np.asarray(ic), dtype=torch.float32)
                    if ic_t.ndim == 1:
                        ic_t = ic_t.unsqueeze(0)
                    if ic_t.shape[-1] != n_coeffs:
                        ic_t = torch.zeros(1, n_coeffs)
                else:
                    ic_t = torch.zeros(1, n_coeffs)

                sc_t = None
                if sc is not None:
                    sc_t = torch.tensor(np.asarray(sc), dtype=torch.float32)
                    if sc_t.ndim == 1:
                        sc_t = sc_t.unsqueeze(0)
                if sc_t is None:
                    sc_t = torch.ones(1, 1)

                with torch.no_grad():
                    rest_joints = self._soma_model.get_skeleton(
                        ic_t, sc_t
                    )  # (1, 77, 3)
                joints_np = rest_joints[0].cpu().numpy()

                parents = SOMA_SKELETON.joint_parents
                for i, name in enumerate(SOMA_SKELETON.joint_names):
                    if i == 0:
                        offsets_dict[name] = [0.0, 0.0, 0.0]
                    else:
                        off = joints_np[i] - joints_np[parents[i]]
                        offsets_dict[name] = off.tolist()

            elif track.smplx_params is not None and self._load_model():
                # SMPL-X path: run body model at zero pose with betas
                params = track.smplx_params
                betas = params.get("betas")
                if betas is None:
                    return None

                be_t = torch.tensor(np.asarray(betas), dtype=torch.float32)
                if be_t.ndim >= 2:
                    be_t = be_t[:1]
                else:
                    be_t = be_t.unsqueeze(0)

                n_body = 21
                with torch.no_grad():
                    out = self._body_model(
                        body_pose=torch.zeros(1, n_body * 3),
                        betas=be_t,
                        global_orient=torch.zeros(1, 3),
                        transl=torch.zeros(1, 3),
                    )
                # out is (1, V, 3) vertices; get joints from the regressor
                if hasattr(out, "joints"):
                    joints_np = out.joints[0].cpu().numpy()
                elif isinstance(out, dict) and "joints" in out:
                    joints_np = out["joints"][0].cpu().numpy()
                else:
                    # SmplxLite exposes beta-dependent rest joints via
                    # get_skeleton(); use those so FK matches the mesh scale.
                    if hasattr(self._body_model, "get_skeleton"):
                        joints_np = (
                            self._body_model.get_skeleton(be_t)[0].cpu().numpy()
                        )
                    else:
                        return None

                joints_np = _coerce_smplx_joints_to_viewport_order(joints_np)

                for i, name in enumerate(JOINT_NAMES):
                    if i == 0 or i >= len(joints_np):
                        offsets_dict[name] = DEFAULT_OFFSETS.get(name, [0, 0, 0])
                    else:
                        parent = JOINT_PARENTS[i]
                        if parent < len(joints_np):
                            off = joints_np[i] - joints_np[parent]
                            offsets_dict[name] = off.tolist()
                        else:
                            offsets_dict[name] = DEFAULT_OFFSETS.get(name, [0, 0, 0])
            else:
                return None

            self._shape_offsets_cache[person_id] = offsets_dict
            logger.info("Computed shape-dependent offsets for pid=%d", person_id)
            return offsets_dict

        except Exception as e:
            logger.debug("Shape offset computation failed for pid=%d: %s", person_id, e)
            return None

    def _compute_joints(self) -> np.ndarray | None:
        """Compute 3D joint positions for current person/frame.

        Returns (J, 3) array in camera space (incam) or world space (orbit
        with world-grounding data), or None if params unavailable.
        """
        if self._session is None or self._person_id < 0:
            return None
        track = self._session.person_tracks.get(self._person_id)
        if track is None:
            logger.debug("_compute_joints: no track for pid=%d", self._person_id)
            return None
        ctx = self._resolve_track_render_context(
            track,
            apply_pose_override=self._pose_override is not None,
        )
        if ctx is None:
            logger.debug(
                "_compute_joints: no params for pid=%d (model=%s, soma=%s, smplx=%s)",
                self._person_id, track.body_model_type,
                track.soma_params is not None, track.smplx_params is not None,
            )
            return None
        params = ctx["params"]

        if ctx["source"] == "smplx":
            model_joints = self._compute_smplx_model_joints(params, self._current_frame)
            if model_joints is not None:
                return model_joints

        # Use shape-dependent offsets when available
        shape_off = self._get_shape_offsets(self._person_id) if ctx["source"] == "smplx" else None

        try:
            joints = forward_kinematics(params, self._current_frame, shape_off)
        except Exception as e:
            logger.warning("FK failed (pid=%d, f=%d): %s",
                           self._person_id, self._current_frame, e)
            return None
        if joints is None:
            return None

        return joints

    def _compute_smplx_model_joints(
        self,
        params: dict,
        frame_idx: int,
    ) -> np.ndarray | None:
        """Compute posed SMPL-X joints from the same model path as the mesh."""
        if not self._load_model() or self._body_model is None:
            return None

        try:
            import torch
            from pytorch3d.transforms import axis_angle_to_matrix
            from hmr4d.utils.body_model.smplx_lite import batch_rigid_transform_v2

            go_frame = _frame_slice_array(params.get("global_orient"), frame_idx)
            bp_frame = _frame_slice_array(params.get("body_pose"), frame_idx)
            be_frame = _frame_slice_array(params.get("betas"), frame_idx)
            tr_frame = _frame_slice_array(params.get("transl"), frame_idx)

            if go_frame is None or bp_frame is None or be_frame is None:
                return None

            go_t = torch.tensor(go_frame, dtype=torch.float32)
            bp_t = torch.tensor(bp_frame, dtype=torch.float32)
            be_t = torch.tensor(be_frame, dtype=torch.float32)
            tr_t = (
                torch.tensor(tr_frame, dtype=torch.float32)
                if tr_frame is not None
                else None
            )

            if bp_t.ndim == 3:
                bp_t = bp_t.reshape(bp_t.shape[0], -1)

            with torch.no_grad():
                other_default_pose = self._body_model.other_default_pose.expand(
                    *bp_t.shape[:-1], -1
                )
                full_pose = torch.cat([go_t, bp_t, other_default_pose], dim=-1)
                rot_mats = axis_angle_to_matrix(full_pose.reshape(*full_pose.shape[:-1], 55, 3))
                joints_native = self._body_model.get_skeleton(be_t)
                posed_joints = batch_rigid_transform_v2(
                    rot_mats, joints_native, self._body_model.parents
                )[0]
                if tr_t is not None:
                    posed_joints = posed_joints + tr_t[:, None, :]

            joints_np = posed_joints[0].cpu().numpy()
            return _coerce_smplx_joints_to_viewport_order(joints_np)
        except Exception as e:
            logger.debug("SMPL-X model joint computation failed (f=%d): %s", frame_idx, e)
            return None

    def _pick_joint(self, screen_x: float, screen_y: float) -> int | None:
        """Pick the joint at screen position, using the active picking mode.

        When ``_picking_mode`` is ``'fbo'``, renders joints as uniquely
        colored points into an offscreen FBO and reads back the pixel.
        Falls back to screen-space distance if FBO picking fails or is
        disabled.
        """
        if self._joint_positions is None:
            return None

        if self._picking_mode == "fbo" and _HAS_GL and self._gl_ready:
            result = self._fbo_pick_joint(screen_x, screen_y)
            if result is not None:
                return result
            # Fall through to screen-space if FBO returned no hit

        # Screen-space distance fallback
        w = self.width() if self.width() > 0 else 200
        h = self.height() if self.height() > 0 else 150
        mvp = self._projection @ self._view @ self._model_mat
        screen = project_joints_to_screen(self._joint_positions, mvp, w, h)
        return find_nearest_joint(screen_x, screen_y, screen)

    def _ensure_pick_fbo(self, width: int, height: int) -> bool:
        """Create or resize the offscreen FBO for color-coded joint picking.

        Returns True if the FBO is ready to use.
        """
        if not _HAS_GL:
            return False

        if self._pick_fbo_id and self._pick_fbo_size == (width, height):
            return True  # Already the right size

        # Clean up old FBO
        if self._pick_fbo_id:
            gl.glDeleteFramebuffers(1, [self._pick_fbo_id])
            gl.glDeleteRenderbuffers(
                2, [self._pick_rbo_color, self._pick_rbo_depth]
            )

        self._pick_fbo_id = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._pick_fbo_id)

        # Color renderbuffer (RGBA8 for ID encoding)
        self._pick_rbo_color = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._pick_rbo_color)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_RGBA8, width, height
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
            gl.GL_RENDERBUFFER, self._pick_rbo_color,
        )

        # Depth renderbuffer (for correct occlusion among overlapping joints)
        self._pick_rbo_depth = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, self._pick_rbo_depth)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, width, height
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
            gl.GL_RENDERBUFFER, self._pick_rbo_depth,
        )

        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            logger.warning("Pick FBO incomplete: 0x%x", status)
            gl.glDeleteFramebuffers(1, [self._pick_fbo_id])
            gl.glDeleteRenderbuffers(
                2, [self._pick_rbo_color, self._pick_rbo_depth]
            )
            self._pick_fbo_id = 0
            return False

        self._pick_fbo_size = (width, height)
        return True

    def _fbo_pick_joint(self, screen_x: float, screen_y: float) -> int | None:
        """Color-coded FBO joint picking for accurate overlapping-joint selection.

        Renders each joint as a uniquely colored GL_POINT into an offscreen
        FBO with depth testing, reads back the pixel at the click position,
        and decodes the color to a joint index.  Depth testing ensures the
        front-most joint wins when multiple joints overlap on screen.
        """
        if not _HAS_GL or not self._gl_ready:
            return None

        joints = self._joint_positions
        if joints is None:
            return None

        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return None

        self.makeCurrent()

        if not self._ensure_pick_fbo(w, h):
            self.doneCurrent()
            return None

        # Bind FBO and clear
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._pick_fbo_id)
        gl.glViewport(0, 0, w, h)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Depth test for correct occlusion; no backface culling for points
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)

        # Render joints with unique ID colors (unlit)
        self._shader.bind()
        self._set_mat4("model", self._model_mat)
        self._set_mat4("view", self._view)
        self._set_mat4("projection", self._projection)
        self._set_vec3("light_dir", _LIGHT_DIR)
        self._set_vec3("light_color", np.zeros(3, dtype=np.float32))
        self._set_vec3("ambient", np.ones(3, dtype=np.float32))

        # Build joint positions + ID colors (body joints only by default)
        n = min(len(joints), _N_BODY_JOINTS)
        jv = joints[:n].astype(np.float32)
        jn = np.zeros_like(jv)
        jc = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            r, g, b = encode_joint_id(i)
            jc[i] = [r / 255.0, g / 255.0, b / 255.0]

        # Large point size so clicking near a joint registers
        pick_size = max(_JOINT_PICK_THRESHOLD, _SELECTED_JOINT_POINT_SIZE)
        self._draw_primitive(
            gl.GL_POINTS, jv, jn, jc, point_size=pick_size,
        )

        self._shader.release()

        # Read pixel at click position (flip Y for GL coordinate system)
        px = max(0, min(int(screen_x), w - 1))
        py = max(0, min(h - 1 - int(screen_y), h - 1))

        pixel = gl.glReadPixels(px, py, 1, 1, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)

        # Restore default framebuffer + state
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, w, h)
        gl.glEnable(gl.GL_CULL_FACE)

        self.doneCurrent()

        # Decode pixel → joint index
        if pixel is not None and len(pixel) >= 3:
            r_val = int(pixel[0]) if not isinstance(pixel[0], int) else pixel[0]
            g_val = int(pixel[1]) if not isinstance(pixel[1], int) else pixel[1]
            b_val = int(pixel[2]) if not isinstance(pixel[2], int) else pixel[2]
            result = decode_joint_id(r_val, g_val, b_val)
            if result is not None and result < _N_BODY_JOINTS:
                return result
        return None

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
                model_path=str(model_dir) if model_dir else None
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

    def _load_soma_model(self) -> bool:
        """Lazily load SOMA body model. Returns True on success."""
        if self._soma_model_loaded:
            return self._soma_model is not None
        self._soma_model_loaded = True

        try:
            from gem.utils.soma_utils.soma_layer import SomaLayer
            from soma.assets import get_assets_dir

            data_root = str(get_assets_dir())
            self._soma_model = SomaLayer(
                data_root=data_root,
                low_lod=True,
                device="cpu",
                identity_model_type="soma",
                mode="dense",
            )
            self._soma_model.eval()

            faces_tensor = self._soma_model.faces
            if hasattr(faces_tensor, "numpy"):
                self._soma_faces = faces_tensor.long().cpu().numpy().astype(np.int32)
            else:
                self._soma_faces = np.asarray(faces_tensor, dtype=np.int32)

            logger.info(
                "SOMA model loaded: %d faces, %d vertices per frame",
                len(self._soma_faces),
                self._soma_model.soma.rest_shape.shape[-2]
                if hasattr(self._soma_model.soma, "rest_shape")
                else "?",
            )
            return True
        except Exception as e:
            logger.warning("Could not load SOMA model: %s", e)
            return False

    def _compute_soma_vertices(
        self,
        soma_params: dict,
        frame_idx: int,
        override_active: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute SOMA mesh vertices and normals for one frame.

        Returns ``(vertices, normals)`` each (V, 3) float32, or ``None``.
        """
        if not self._load_soma_model():
            return None

        try:
            import torch

            with torch.no_grad():
                # soma_params already reflects the active render context chosen
                # by _resolve_track_render_context(), so orbit mode can pass
                # world-grounded root motion while incam keeps camera-space data.
                poses = soma_params.get("poses")  # (N, 77, 3)
                transl = soma_params.get("transl")  # (N, 3)
                identity_coeffs = soma_params.get("identity_coeffs")  # (1, C)
                scale_params = soma_params.get("scale_params")  # (1, S)

                if poses is None:
                    return None

                def _to_t(x):
                    if x is None:
                        return None
                    if isinstance(x, torch.Tensor):
                        return x.float().cpu()
                    return torch.tensor(np.asarray(x), dtype=torch.float32)

                poses_t = _to_t(poses)
                transl_t = _to_t(transl)

                # Single frame slice
                if poses_t.ndim == 3 and frame_idx < poses_t.shape[0]:
                    poses_frame = poses_t[frame_idx : frame_idx + 1]  # (1, 77, 3)
                else:
                    return None

                tr_frame = None
                if transl_t is not None and transl_t.ndim >= 2:
                    tr_frame = transl_t[frame_idx : frame_idx + 1]  # (1, 3)

                # Identity: SOMA PCA model expects (B, 128). GEM-X MHR gives (1, 45).
                # Use neutral body (zeros) when coefficients don't match.
                n_soma_coeffs = 128  # SOMA PCA identity dimension
                if identity_coeffs is not None:
                    ic = _to_t(identity_coeffs)
                    if ic.ndim == 1:
                        ic = ic.unsqueeze(0)
                    if ic.shape[-1] != n_soma_coeffs:
                        ic = torch.zeros(1, n_soma_coeffs)
                else:
                    ic = torch.zeros(1, n_soma_coeffs)

                # Scale params: SOMA PCA doesn't use scale_params, pass None
                # (SomaLayer.static_forward splits scale_params[:, :1] as global_scale)
                # For 'soma' identity model, there are no per-joint scales.
                # We create a dummy [global_scale=1.0] + zeros to satisfy the API.
                sp = None
                if scale_params is not None:
                    sp = _to_t(scale_params)
                    if sp.ndim == 1:
                        sp = sp.unsqueeze(0)
                if sp is None:
                    # (1, 1) — global_scale=1.0 (neutral)
                    sp = torch.ones(1, 1)

                out = self._soma_model.static_forward(
                    poses=poses_frame,
                    identity_coeffs=ic,
                    scale_params=sp,
                    transl=tr_frame,
                    pose2rot=True,
                )

                verts = out["vertices"]  # (1, V, 3)
                vertices = verts[0].cpu().numpy().astype(np.float32)
                normals = compute_normals(vertices, self._soma_faces)

                # Set faces to SOMA topology so _upload_buffers uses the right EBO
                self._faces = self._soma_faces
                self._n_faces = len(self._soma_faces)

                return (vertices, normals)

        except Exception as e:
            logger.warning("SOMA vertex computation failed (f=%d): %s", frame_idx, e)
            return None

    # ------------------------------------------------------------------
    # Vertex computation
    # ------------------------------------------------------------------

    def _compute_vertices(
        self, person_id: int, frame_idx: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute SMPL-X or SOMA vertices and normals for one frame.

        Returns ``(vertices, normals)`` each (V, 3) float32, or ``None``
        when the model / params are unavailable.
        """
        # Skip cache when pose override is active for this frame
        override_active = (
            self._pose_override is not None
            and self._pose_override.get("frame_idx", self._current_frame) == frame_idx
        )

        if self._session is None:
            return None

        track = self._session.person_tracks.get(person_id)
        if track is None:
            return None

        ctx = self._resolve_track_render_context(
            track,
            apply_pose_override=override_active,
        )
        if ctx is None:
            return None

        cache_key = (
            person_id,
            frame_idx,
            self._camera_mode,
            ctx["source"],
            int(ctx["is_global"]),
            id(ctx["params"]),
        )

        if not override_active and cache_key in self._vertex_cache:
            return self._vertex_cache[cache_key]

        # SOMA track with no SMPL-X fallback — use SOMA body model
        if ctx["source"] == "soma":
            return self._compute_soma_vertices(
                ctx["params"], frame_idx, override_active
            )

        # SMPL-X path (includes SOMA tracks with SMPL-X fallback from GVHMR)
        if not self._load_model():
            return None

        params = ctx["params"]
        if params is None:
            return None

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
            if not bool(gl.glGenVertexArrays):
                raise RuntimeError(
                    "glGenVertexArrays unavailable (GL too old?) — "
                    "try setting LIBGL_ALWAYS_SOFTWARE=1"
                )
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
            self._status_msg = f"OpenGL init failed: {e}"
            self._setup_fallback()

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

        # Draw video frame background in in-camera mode (before GL rendering)
        _has_bg = (
            self._camera_mode == "incam"
            and self._video_frame is not None
        )
        if _has_bg:
            fh, fw = self._video_frame.shape[:2]
            bpl = fw * 3  # bytes per line for RGB888
            qimg = QImage(
                self._video_frame.data, fw, fh, bpl,
                QImage.Format.Format_RGB888,
            )
            painter.drawImage(self.rect(), qimg)

        painter.beginNativePainting()

        # Clear with alpha=1.0 to ensure the surface is fully opaque,
        # then disable alpha writes so GL draw calls can't make it transparent.
        # Without this, the Wayland compositor on WSL2 treats GL-rendered
        # pixels as transparent → click-through to windows behind.
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
        if _has_bg:
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        else:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_FALSE)

        # Grid floor (orbit mode only, drawn first so mesh occludes it)
        if self._show_grid and self._camera_mode == "orbit":
            self._draw_grid()

        # Mesh triangles — skip entirely in wireframe mode
        if (
            self._render_mode != RenderMode.WIREFRAME
            and self._vertices is not None
            and self._n_indices > 0
        ):
            self._shader.bind()

            # Uniforms
            self._set_mat4("model", self._model_mat)
            self._set_mat4("view", self._view)
            self._set_mat4("projection", self._projection)
            self._set_vec3("light_dir", _LIGHT_DIR)
            if self._render_mode == RenderMode.FAST:
                # Ambient-only: no diffuse shading for speed
                self._set_vec3("light_color", np.zeros(3, dtype=np.float32))
                self._set_vec3("ambient", np.ones(3, dtype=np.float32))
            else:
                self._set_vec3("light_color", _LIGHT_COLOR)
                self._set_vec3("ambient", _AMBIENT)

            gl.glBindVertexArray(self._vao_id)
            gl.glDrawElements(
                gl.GL_TRIANGLES, self._n_indices, gl.GL_UNSIGNED_INT, None
            )
            gl.glBindVertexArray(0)

            self._shader.release()

        # Skeleton overlay (always drawn when visible — even in wireframe)
        if self._show_skeleton:
            if self._show_all_persons and self._all_joint_positions:
                # Draw non-selected persons first (dimmer)
                for pid, joints in self._all_joint_positions.items():
                    if pid == self._person_id:
                        continue
                    skel = self._all_active_skels.get(pid, self._active_skel)
                    self._draw_skeleton_for(joints, skel, selected=False)
            # Draw selected person on top (full brightness)
            if self._joint_positions is not None:
                self._draw_skeleton()

        painter.endNativePainting()

        # Joint name text overlay (QPainter 2D, after GL rendering)
        if (
            self._show_joint_labels
            and self._show_skeleton
            and self._joint_positions is not None
        ):
            self._draw_joint_labels(painter)

        # Debug: show joint count + camera info when skeleton is loaded
        if self._joint_positions is not None:
            painter.setPen(QColor(120, 200, 120))
            painter.setFont(QFont("sans-serif", 10))
            jp = self._joint_positions
            centroid = jp.mean(axis=0)
            spread = float(np.max(np.linalg.norm(jp - centroid, axis=1)))
            painter.drawText(10, 20, (
                f"{len(jp)} joints  |  centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]"
                f"  |  spread: {spread:.2f}m  |  skel: {self._active_skel.name}"
            ))

        # Status text when no mesh/skeleton is rendering
        if self._vertices is None and self._joint_positions is None:
            painter.setPen(QColor(180, 180, 180))
            painter.setFont(QFont("sans-serif", 11))
            lines = []
            if self._session is None:
                lines.append("No session loaded")
            elif self._person_id < 0:
                lines.append("No person selected")
            else:
                track = self._session.person_tracks.get(self._person_id)
                if track is None:
                    lines.append(f"No track for person {self._person_id}")
                else:
                    lines.append(f"Person {self._person_id}  |  model: {track.body_model_type}")
                    lines.append(f"smplx_params: {'yes' if track.smplx_params else 'None'}")
                    lines.append(f"soma_params: {'yes' if track.soma_params else 'None'}")
            lines.append(f"Camera: {self._camera_mode}  |  GL: {'ready' if self._gl_ready else 'NOT READY'}")
            lines.append("Press V to toggle orbit/incam camera")
            y = self.height() // 2 - len(lines) * 10
            for line in lines:
                painter.drawText(10, y, line)
                y += 20

        painter.end()

        # WSL2/Wayland fix: QPainter text rendering writes alpha < 1.0 for
        # antialiasing.  The Wayland compositor treats those pixels as
        # transparent, causing click-through to windows behind.
        # Fix: after QPainter is done, do one final GL pass that overwrites
        # the entire framebuffer alpha channel to 1.0 (fully opaque).
        if _IS_WSL and _HAS_GL:
            self.makeCurrent()
            gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_TRUE)
            gl.glClearColor(0, 0, 0, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
            # Restore clear color for next frame
            gl.glClearColor(0.1, 0.1, 0.12, 1.0)

    # ------------------------------------------------------------------
    # Skeleton GL rendering
    # ------------------------------------------------------------------

    def _get_frame_confidence(self) -> float:
        """Return the confidence value for the current frame and person.

        Checks confidence_breakdown["overall"] first, then raw confidences.
        Returns 0.5 (mid-confidence) when no data is available.
        """
        if self._session is None or self._person_id < 0:
            return 0.5
        track = self._session.person_tracks.get(self._person_id)
        if track is None:
            return 0.5
        if (
            track.confidence_breakdown is not None
            and "overall" in track.confidence_breakdown
        ):
            vals = track.confidence_breakdown["overall"]
            if 0 <= self._current_frame < len(vals):
                return float(vals[self._current_frame])
        elif track.confidences is not None:
            if 0 <= self._current_frame < len(track.confidences):
                return float(track.confidences[self._current_frame])
        return 0.5

    # Per-person colors for multi-person view (up to 8 distinct colors)
    _PERSON_COLORS = [
        np.array([0.30, 0.70, 1.00], dtype=np.float32),  # blue
        np.array([1.00, 0.50, 0.20], dtype=np.float32),  # orange
        np.array([0.30, 0.90, 0.40], dtype=np.float32),  # green
        np.array([0.90, 0.30, 0.60], dtype=np.float32),  # pink
        np.array([0.80, 0.80, 0.20], dtype=np.float32),  # yellow
        np.array([0.50, 0.30, 0.90], dtype=np.float32),  # purple
        np.array([0.20, 0.85, 0.85], dtype=np.float32),  # cyan
        np.array([0.90, 0.40, 0.40], dtype=np.float32),  # red
    ]

    def _draw_skeleton_for(self, joints: np.ndarray, skel, selected: bool = True):
        """Draw a skeleton with the given joints/skeleton def.

        When selected=False, draws with a dimmer per-person color and thinner lines.
        Used for non-selected persons in multi-person view.
        """
        if not _HAS_GL or not self._gl_ready or joints is None:
            return

        self._shader.bind()
        self._set_mat4("model", self._model_mat)
        self._set_mat4("view", self._view)
        self._set_mat4("projection", self._projection)
        self._set_vec3("light_dir", _LIGHT_DIR)
        self._set_vec3("light_color", np.zeros(3, dtype=np.float32))
        self._set_vec3("ambient", np.ones(3, dtype=np.float32))

        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)

        # Pick a color based on person ID
        pid = None
        for p, j in self._all_joint_positions.items():
            if j is joints:
                pid = p
                break
        color_idx = (pid if pid is not None else 0) % len(self._PERSON_COLORS)
        base_color = self._PERSON_COLORS[color_idx]
        dim = 1.0 if selected else 0.55
        bone_color = base_color * dim
        joint_color = base_color * min(dim * 1.2, 1.0)

        # Draw bones
        bone_verts = []
        bone_colors = []
        for a, b in skel.bone_connections:
            if a < len(joints) and b < len(joints):
                bone_verts.append(joints[a])
                bone_verts.append(joints[b])
                bone_colors.append(bone_color)
                bone_colors.append(bone_color)
        if bone_verts:
            bv = np.array(bone_verts, dtype=np.float32)
            bc = np.array(bone_colors, dtype=np.float32)
            self._draw_primitive(gl.GL_LINES, bv, np.zeros_like(bv), bc,
                                 line_width=2.0 if selected else 1.5)

        # Draw joints
        jv = joints.astype(np.float32)
        jc = np.tile(joint_color, (len(joints), 1))
        self._draw_primitive(gl.GL_POINTS, jv, np.zeros_like(jv), jc,
                             point_size=6.0 if selected else 4.0)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        self._shader.release()

    def _draw_skeleton(self):
        """Draw skeleton overlay: bones as GL_LINES, joints as GL_POINTS.

        Uses legacy-ish immediate-mode via temporary VBOs for simplicity,
        reusing the mesh shader with ambient=1 so the skeleton is unlit.

        When a joint is selected, the chain from that joint to root is
        highlighted in accent amber with thicker lines; the selected joint
        itself gets a brighter accent color and larger point size.
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

        # Use active skeleton's bone connections and parent hierarchy
        skel = self._active_skel
        active_bones = skel.bone_connections
        active_parents = list(skel.joint_parents)
        active_n_body = skel.n_body_joints

        # Compute chain highlight set when a joint is selected
        chain_set: set[int] = set()
        chain_bones: set[tuple[int, int]] = set()
        if 0 <= self._selected_joint < len(joints):
            # Walk parent chain using active skeleton
            chain = [self._selected_joint]
            cur = self._selected_joint
            while cur < len(active_parents) and active_parents[cur] >= 0:
                cur = active_parents[cur]
                chain.append(cur)
            chain_set = set(chain)
            for i in range(len(chain) - 1):
                child, parent = chain[i], chain[i + 1]
                chain_bones.add((min(parent, child), max(parent, child)))

        # Heatmap base color: confidence-mapped color for non-highlighted elements
        if self._skeleton_heatmap:
            heatmap_color = confidence_to_color(self._get_frame_confidence())
        else:
            heatmap_color = None

        # --- Draw non-chain bones as GL_LINES (normal width) ---
        bone_verts = []
        bone_colors = []
        chain_bone_verts = []
        chain_bone_colors = []
        base_bone_color = heatmap_color if heatmap_color is not None else _BONE_COLOR
        for a, b in active_bones:
            if a < len(joints) and b < len(joints):
                key = (min(a, b), max(a, b))
                if key in chain_bones:
                    chain_bone_verts.append(joints[a])
                    chain_bone_verts.append(joints[b])
                    chain_bone_colors.append(_CHAIN_BONE_COLOR)
                    chain_bone_colors.append(_CHAIN_BONE_COLOR)
                else:
                    bone_verts.append(joints[a])
                    bone_verts.append(joints[b])
                    bone_colors.append(base_bone_color)
                    bone_colors.append(base_bone_color)

        if bone_verts:
            bv = np.array(bone_verts, dtype=np.float32)
            bc = np.array(bone_colors, dtype=np.float32)
            bn = np.zeros_like(bv)
            self._draw_primitive(gl.GL_LINES, bv, bn, bc, _BONE_LINE_WIDTH)

        # --- Draw chain bones thicker in accent amber ---
        if chain_bone_verts:
            cbv = np.array(chain_bone_verts, dtype=np.float32)
            cbc = np.array(chain_bone_colors, dtype=np.float32)
            cbn = np.zeros_like(cbv)
            self._draw_primitive(gl.GL_LINES, cbv, cbn, cbc, _CHAIN_BONE_LINE_WIDTH)

        # --- Draw joints as GL_POINTS ---
        jv = joints.astype(np.float32)
        jn = np.zeros_like(jv)
        jc = np.zeros((len(joints), 3), dtype=np.float32)
        jp = np.full(len(joints), _JOINT_POINT_SIZE, dtype=np.float32)

        for i in range(len(joints)):
            if i == self._selected_joint:
                jc[i] = _SELECTED_ACCENT_COLOR
                jp[i] = _SELECTED_JOINT_POINT_SIZE
            elif i in chain_set:
                jc[i] = _CHAIN_JOINT_COLOR
                jp[i] = _CHAIN_JOINT_POINT_SIZE
            elif heatmap_color is not None:
                jc[i] = heatmap_color
            elif i < active_n_body:
                jc[i] = _BODY_JOINT_COLOR
            else:
                jc[i] = _HAND_JOINT_COLOR

        # Draw non-selected, non-chain joints at normal size
        normal_mask = np.array([
            i != self._selected_joint and i not in chain_set
            for i in range(len(joints))
        ])
        if np.any(normal_mask):
            self._draw_primitive(
                gl.GL_POINTS, jv[normal_mask], jn[normal_mask], jc[normal_mask],
                point_size=_JOINT_POINT_SIZE,
            )

        # Draw chain joints (excluding selected) at chain size
        chain_mask = np.array([
            i in chain_set and i != self._selected_joint
            for i in range(len(joints))
        ])
        if np.any(chain_mask):
            self._draw_primitive(
                gl.GL_POINTS, jv[chain_mask], jn[chain_mask], jc[chain_mask],
                point_size=_CHAIN_JOINT_POINT_SIZE,
            )

        # Draw selected joint largest
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
            joint_names=list(self._active_skel.joint_names),
            n_body_joints=self._active_skel.n_body_joints,
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
            try:
                gl.glLineWidth(line_width)
            except Exception:
                gl.glLineWidth(1.0)
        elif mode == gl.GL_POINTS:
            try:
                gl.glPointSize(point_size)
            except Exception:
                pass

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
        selected_track = None
        selected_ctx = None
        if self._session is not None:
            selected_track = self._session.person_tracks.get(self._person_id)
            selected_ctx = self._resolve_track_render_context(selected_track)

        if selected_ctx is not None:
            self._active_skel = selected_ctx["skeleton"]

        old_is_global = self._data_is_global
        self._data_is_global = bool(selected_ctx and selected_ctx["is_global"])
        if self._data_is_global != old_is_global:
            self._update_camera()
        # Skip expensive body model forward pass in wireframe mode and
        # during scrubbing (SOMA on CPU takes ~1s/frame).
        skip_mesh = (
            self._render_mode == RenderMode.WIREFRAME
            or self._pre_scrub_mode is not None  # actively scrubbing/playing
        )
        if not skip_mesh:
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

        # Recompute skeleton joint positions (lightweight FK — always computed)
        prev_joints = self._joint_positions
        self._joint_positions = self._compute_joints()
        if self._joint_positions is not None and prev_joints is None:
            logger.info(
                "Skeleton loaded: pid=%d, %d joints, camera=%s, gl_ready=%s",
                self._person_id, len(self._joint_positions),
                self._camera_mode, self._gl_ready,
            )

        # Compute joints for ALL persons (multi-person view)
        self._all_joint_positions.clear()
        self._all_active_skels.clear()
        if self._show_all_persons and self._session is not None:
            for pid, track in self._session.person_tracks.items():
                ctx = self._resolve_track_render_context(track)
                if ctx is None:
                    continue
                try:
                    joints = None
                    if ctx["source"] == "smplx":
                        joints = self._compute_smplx_model_joints(
                            ctx["params"], self._current_frame
                        )
                    if joints is None:
                        shape_off = (
                            self._get_shape_offsets(pid)
                            if ctx["source"] == "smplx"
                            else None
                        )
                        joints = forward_kinematics(
                            ctx["params"], self._current_frame, shape_off
                        )
                    if joints is None:
                        continue
                    self._all_joint_positions[pid] = joints
                    self._all_active_skels[pid] = ctx["skeleton"]
                except Exception:
                    pass

        # Auto-center orbit camera on skeleton when no mesh is available
        if (
            self._vertices is None
            and (self._joint_positions is not None or self._all_joint_positions)
            and self._camera_mode == "orbit"
            and not self._orbit_auto_centered
        ):
            self._auto_center_orbit()
            self._update_camera()

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

        # Element buffer (face indices) — re-upload when face topology changes
        # (SMPL-X and SOMA have different face arrays)
        current_source = "soma" if (self._soma_faces is not None
                                    and self._faces is self._soma_faces) else "smplx"
        if not self._faces_uploaded or self._uploaded_face_source != current_source:
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
            self._uploaded_face_source = current_source

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
