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
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal, Qt, QRect, QSize
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


def _forward_kinematics_soma(params: dict, frame_idx: int) -> np.ndarray:
    """Compute 3D joint positions for SOMA's unified poses tensor.

    Parameters
    ----------
    params : dict with keys poses (N, 77, 3), transl (N, 3)
    frame_idx : which frame to compute

    Returns
    -------
    positions : (77, 3) float64 — joint positions in camera space
    """
    from scipy.spatial.transform import Rotation
    from models.skeleton import SOMA_SKELETON

    n_joints = SOMA_SKELETON.n_joints
    positions = np.zeros((n_joints, 3))
    accumulated_R = np.zeros((n_joints, 3, 3))

    offsets = np.zeros((n_joints, 3))
    for i, name in enumerate(SOMA_SKELETON.joint_names):
        offsets[i] = SOMA_SKELETON.default_offsets.get(name, [0, 0, 0])

    poses = np.asarray(params["poses"])
    tr = params.get("transl")
    if tr is not None:
        tr = np.asarray(tr)

    # Root (joint 0 = global orient).
    # In orbit mode the viewport swaps params["global_orient"] to world-space;
    # prefer that over poses[:, 0] which is always incam.
    go = params.get("global_orient")
    if go is not None:
        go = np.asarray(go)
        root_aa = go[frame_idx] if go.ndim >= 2 else go
    else:
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


def _rotvec_to_matrix_batch(rotvecs: np.ndarray) -> np.ndarray:
    """Convert (N, 3) axis-angle vectors to (N, 3, 3) rotation matrices.

    Uses Rodrigues' formula directly in numpy — avoids N separate
    scipy.Rotation calls which dominate FK cost in the per-joint loop.
    """
    rotvecs = np.asarray(rotvecs, dtype=np.float64)
    angles = np.linalg.norm(rotvecs, axis=1, keepdims=True)  # (N, 1)
    # Avoid division by zero for near-identity rotations
    safe = np.where(angles > 1e-8, angles, np.ones_like(angles))
    k = rotvecs / safe  # (N, 3) unit axes
    K = np.zeros((len(k), 3, 3), dtype=np.float64)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    sin_a = np.sin(angles)[..., np.newaxis]   # (N, 1, 1)
    cos_a = np.cos(angles)[..., np.newaxis]   # (N, 1, 1)
    I = np.eye(3, dtype=np.float64)[np.newaxis]  # (1, 3, 3)
    R = I + sin_a * K + (1 - cos_a) * (K @ K)
    # Identity for near-zero angles
    near_zero = (angles.ravel() < 1e-8)
    R[near_zero] = np.eye(3, dtype=np.float64)
    return R


def forward_kinematics(params: dict, frame_idx: int) -> np.ndarray:
    """Compute 3D joint positions in camera space for one frame.

    Uses DEFAULT_OFFSETS (model-derived rest-pose, not shape-dependent).
    Vectorized: converts all joint rotations in one batch call, then
    walks the kinematic tree with accumulated rotation matrices.

    Parameters
    ----------
    params : dict with keys global_orient (N,3), body_pose (N,21*3 or N,21,3),
             transl (N,3). Optional: left_hand_pose (N,15,3), right_hand_pose (N,15,3).
    frame_idx : which frame to compute

    Returns
    -------
    positions : (52, 3) float64 — joint positions in camera space
    """
    # SOMA path: unified poses tensor (N, J, 3)
    if "poses" in params and "body_pose" not in params:
        return _forward_kinematics_soma(params, frame_idx)

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
    lwo = _get_array("left_wrist_orient")
    rwo = _get_array("right_wrist_orient")

    if go is None or bp is None:
        return positions

    # Root
    go_frame = go[frame_idx] if go.ndim >= 2 else go
    positions[0] = tr[frame_idx] if tr is not None and tr.ndim >= 2 else (tr if tr is not None else np.zeros(3))

    # Reshape body_pose to (N, 21, 3) if flat
    if bp.ndim == 2 and bp.shape[-1] != 3:
        bp = bp.reshape(bp.shape[0], -1, 3)

    if bp.ndim >= 3:
        bp = bp.copy()
        if lwo is not None:
            if lwo.ndim >= 2 and frame_idx < lwo.shape[0]:
                bp[frame_idx, 19] = np.asarray(lwo[frame_idx]).reshape(3)
            elif lwo.ndim == 1:
                bp[frame_idx, 19] = np.asarray(lwo).reshape(3)
        if rwo is not None:
            if rwo.ndim >= 2 and frame_idx < rwo.shape[0]:
                bp[frame_idx, 20] = np.asarray(rwo[frame_idx]).reshape(3)
            elif rwo.ndim == 1:
                bp[frame_idx, 20] = np.asarray(rwo).reshape(3)

    # Gather all joint axis-angle vectors into (n_joints, 3)
    all_aa = np.zeros((n_joints, 3), dtype=np.float64)
    all_aa[0] = go_frame.ravel()[:3]

    for j in range(1, 22):
        if bp.ndim >= 3 and frame_idx < bp.shape[0]:
            all_aa[j] = bp[frame_idx, j - 1]
        elif bp.ndim == 2:
            all_aa[j] = bp[j - 1]

    if lh is not None:
        if lh.ndim == 3 and frame_idx < lh.shape[0]:
            lh_frame = lh[frame_idx]
        elif lh.ndim == 2:
            lh_r = lh.reshape(-1, 15, 3) if lh.shape[-1] != 3 else lh
            lh_frame = lh_r[frame_idx] if lh_r.ndim == 3 else np.zeros((15, 3))
        else:
            lh_frame = np.zeros((15, 3))
        all_aa[22:37] = np.asarray(lh_frame).reshape(15, 3)

    if rh is not None:
        if rh.ndim == 3 and frame_idx < rh.shape[0]:
            rh_frame = rh[frame_idx]
        elif rh.ndim == 2:
            rh_r = rh.reshape(-1, 15, 3) if rh.shape[-1] != 3 else rh
            rh_frame = rh_r[frame_idx] if rh_r.ndim == 3 else np.zeros((15, 3))
        else:
            rh_frame = np.zeros((15, 3))
        all_aa[37:52] = np.asarray(rh_frame).reshape(15, 3)

    # Batch convert all axis-angle to rotation matrices (single vectorized call)
    all_R = _rotvec_to_matrix_batch(all_aa)  # (52, 3, 3)

    # Walk kinematic tree
    accumulated_R[0] = all_R[0]
    for j in range(1, n_joints):
        parent = JOINT_PARENTS[j]
        accumulated_R[j] = accumulated_R[parent] @ all_R[j]
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


def scale_K_to_viewport(
    K: np.ndarray,
    img_w: int,
    img_h: int,
    vp_w: int,
    vp_h: int,
) -> np.ndarray:
    """Scale intrinsics *K* from image coords to viewport coords (fit, centered).

    Uniform-scales so the full image fits inside the viewport (letterbox/pillarbox),
    then shifts cx/cy so the image is centered in the viewport.
    """
    s = min(vp_w / img_w, vp_h / img_h)
    K_s = K.astype(np.float32).copy()
    K_s[0, 0] *= s   # fx
    K_s[1, 1] *= s   # fy
    K_s[0, 2] = K[0, 2] * s + (vp_w - img_w * s) / 2  # cx
    K_s[1, 2] = K[1, 2] * s + (vp_h - img_h * s) / 2  # cy
    return K_s


def compute_letterbox(
    widget_w: int, widget_h: int, img_w: int, img_h: int,
) -> tuple[int, int, int, int]:
    """Compute letterboxed sub-rect preserving img aspect ratio within widget.

    Returns (x_offset, y_offset, render_width, render_height).
    Falls back to full widget if img dimensions are zero.
    """
    if img_w <= 0 or img_h <= 0:
        return (0, 0, widget_w, widget_h)
    s = min(widget_w / img_w, widget_h / img_h)
    rw = int(img_w * s)
    rh = int(img_h * s)
    x = (widget_w - rw) // 2
    y = (widget_h - rh) // 2
    return (x, y, rw, rh)


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
# ~240KB per entry (10K verts * 3 * 4 bytes * 2 arrays), so 2000 entries ≈ 480MB.
_CACHE_MAX = 2000

# How many frames to prefetch ahead of playback.
_PREFETCH_AHEAD = 120

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

# Joint drag-to-rotate sensitivity.
_JOINT_DRAG_SENSITIVITY = 0.5  # degrees per pixel

# Root drag (Shift+drag joint 0) sensitivity — world units per pixel.
_ROOT_DRAG_SENSITIVITY = 0.003

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


def compute_frustum_lines(
    c2w: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    depth: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Build wireframe frustum line vertices for a single camera pose.

    Parameters
    ----------
    c2w : (4, 4) camera-to-world transform (OpenCV convention: +Z forward)
    K : (3, 3) camera intrinsic matrix
    img_w, img_h : image dimensions in pixels
    depth : frustum depth in metres

    Returns
    -------
    positions : (N, 3) float32 — GL_LINES vertex pairs
    colors : (N, 3) float32 — per-vertex colors (cyan)
    """
    R = c2w[:3, :3]
    t = c2w[:3, 3]

    # Unproject 4 image corners to camera-space rays, scale to depth
    K_inv = np.linalg.inv(K)
    corners_px = np.array([
        [0, 0, 1],
        [img_w, 0, 1],
        [img_w, img_h, 1],
        [0, img_h, 1],
    ], dtype=np.float32)
    rays_cam = (K_inv @ corners_px.T).T  # (4, 3)
    # Normalise so Z=depth
    rays_cam = rays_cam / rays_cam[:, 2:3] * depth
    # Transform to world (OpenCV cam: Z-forward, Y-down)
    corners_world = (R @ rays_cam.T).T + t  # (4, 3)

    # c2w translations are already in Y-up world space
    origin = t.astype(np.float32)
    gl_corners = corners_world.astype(np.float32)

    verts = []
    # 4 pyramid edges (origin → corner)
    for c in gl_corners:
        verts.append(origin)
        verts.append(c)
    # 4 near-plane edges (corner → next corner)
    for i in range(4):
        verts.append(gl_corners[i])
        verts.append(gl_corners[(i + 1) % 4])
    # Up tick on top edge midpoint
    top_mid = (gl_corners[0] + gl_corners[1]) * 0.5
    cam_up = R @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
    cam_up_gl = cam_up.astype(np.float32)
    tick_end = top_mid + cam_up_gl * (depth * 0.15)
    verts.append(top_mid)
    verts.append(tick_end)

    positions = np.array(verts, dtype=np.float32)
    color = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    colors = np.tile(color, (len(positions), 1))
    return positions, colors


def compute_camera_trail(
    c2w_all: np.ndarray,
    current_frame: int,
    trail_frames: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Build GL_LINES pairs tracing camera origin through recent frames.

    Parameters
    ----------
    c2w_all : (N, 4, 4) per-frame camera-to-world
    current_frame : current frame index
    trail_frames : how many past frames to show

    Returns
    -------
    positions : (M, 3) float32 — GL_LINES pairs
    colors : (M, 3) float32 — dim cyan per vertex
    """
    N = c2w_all.shape[0]
    start = max(0, current_frame - trail_frames)
    end = min(current_frame + 1, N)
    if end - start < 2:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    # Extract origins (already in Y-up world space)
    gl_origins = c2w_all[start:end, :3, 3].astype(np.float32)

    # Build line pairs: frame[i] → frame[i+1]
    K = gl_origins.shape[0]
    verts = np.zeros((2 * (K - 1), 3), dtype=np.float32)
    for i in range(K - 1):
        verts[2 * i] = gl_origins[i]
        verts[2 * i + 1] = gl_origins[i + 1]

    color = np.array([0.0, 0.6, 0.6], dtype=np.float32)
    colors = np.tile(color, (len(verts), 1))
    return verts, colors


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
        self._source_text = ""
        self._mesh_text = ""

        self.setFixedSize(200, 100)

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
        if self._source_text:
            lines.append(self._source_text)
        if self._mesh_text:
            lines.append(self._mesh_text)

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
        source: str | None = None,
        mesh: str | None = None,
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
        if source is not None:
            self._source_text = source
        if mesh is not None:
            self._mesh_text = mesh
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
    joint_drag_updated = Signal(int, object)  # (joint_idx, euler_deg ndarray or None)
    root_drag_committed = Signal(int, object)  # (person_id, offset_xyz ndarray)
    camera_changed = Signal(object)
    gl_rendered = Signal()  # emitted after paintGL completes
    source_status_changed = Signal(str)
    mesh_status_changed = Signal(str)

    def __init__(self, gvhmr_root: Path | None = None, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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
        self._body_model_kind: str | None = None
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
        # Background prefetch thread for filling cache ahead of playback
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_stop = threading.Event()
        self._cache_lock = threading.Lock()

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
        self._motion_source: str = "auto"
        self._resolved_motion_source: str = "auto"
        self._world_sources_enabled: bool = True

        # Orbit camera state (used in "orbit" mode)
        self._orbit_yaw: float = _ORBIT_DEFAULT_YAW
        self._orbit_pitch: float = _ORBIT_DEFAULT_PITCH
        self._orbit_distance: float = _ORBIT_DEFAULT_DISTANCE
        self._orbit_center: np.ndarray = _ORBIT_DEFAULT_CENTER.copy()
        self._orbit_auto_centered: bool = False

        # Incam camera offset (manual alignment adjustment in GL camera space)
        self._incam_offset: np.ndarray = np.zeros(3, dtype=np.float32)

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

        # Joint drag-to-rotate state
        self._drag_mode: str = "none"  # "none" | "orbit" | "joint_drag" | "root_drag"
        self._root_drag_offset: np.ndarray = np.zeros(3, dtype=np.float32)
        self._drag_joint: int = -1  # joint index being dragged
        self._drag_start_aa: np.ndarray | None = None  # original axis-angle (3,)
        self._drag_cumulative_R: object | None = None  # scipy Rotation

        # Joint label overlay state (QPainter text over GL)
        self._show_joint_labels: bool = False  # off by default, toggled by user

        # Grid floor state (orbit mode reference plane)
        self._show_grid: bool = True  # visible by default in orbit mode
        self._grid_y: float = 0.0  # Y level of the grid in GL space
        self._grid_center_x: float = 0.0
        self._grid_center_z: float = 0.0

        # Interpolation preview: SLERP between correction keyframes
        self._interpolation_enabled: bool = True

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
        # Playback render mode — what to switch to during scrubbing/playback
        self._playback_render_mode: RenderMode = RenderMode.FAST

        # Position anchor markers (purple crosshairs on ground plane)
        self._pos_anchor_markers: list[np.ndarray] = []  # list of (3,) world positions

        # Camera frustum wireframe (orbit mode)
        self._show_camera_frustum: bool = False
        self._frustum_fov_override: float | None = None  # horizontal FOV in degrees

        # Status message for fallback rendering
        self._status_msg: str = ""
        self._mesh_status: str = "Mesh: no person selected"

        # Letterbox rect: (x_offset, y_offset, render_w, render_h)
        self._letterbox: tuple[int, int, int, int] = (0, 0, 1, 1)

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
        self._hud_enabled = False  # off by default; toggle via View > Toggle HUD (Ctrl+H)
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
        """Reposition HUD overlay and recompute letterbox on resize."""
        super().resizeEvent(event)
        self._recompute_letterbox()
        self._position_hud()

    def _recompute_letterbox(self):
        """Recompute the letterbox rect from widget size and video aspect ratio."""
        w, h = max(self.width(), 1), max(self.height(), 1)
        if self._session and self._session.img_width > 0 and self._session.img_height > 0:
            self._letterbox = compute_letterbox(w, h, self._session.img_width, self._session.img_height)
        else:
            self._letterbox = (0, 0, w, h)

    def _widget_to_viewport(self, wx: float, wy: float) -> tuple[float, float] | None:
        """Convert widget-space mouse coords to viewport-space. None if outside letterbox."""
        lx, ly, lw, lh = self._letterbox
        vx, vy = wx - lx, wy - ly
        if vx < 0 or vy < 0 or vx >= lw or vy >= lh:
            return None
        return (vx, vy)

    def _position_hud(self):
        """Place HUD in bottom-right corner of letterbox with margin."""
        margin = 10
        hud_w = self._hud.width()
        hud_h = self._hud.height()
        lx, ly, lw, lh = self._letterbox
        self._hud.move(lx + lw - hud_w - margin, ly + lh - hud_h - margin)

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
        source: str | None = None,
        mesh: str | None = None,
    ):
        """Update HUD fields and briefly show if enabled."""
        self._hud.update_info(
            frame=frame, total_frames=total_frames,
            speed=speed, person=person, fps=fps,
            source=source, mesh=mesh,
        )
        if self._hud_enabled:
            self._hud.show_with_timer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _invalidate_cache(self):
        """Stop prefetch and clear vertex cache. Thread-safe."""
        self._stop_prefetch()
        with self._cache_lock:
            self._vertex_cache.clear()

    def set_session(self, session: Session):
        """Bind session data source."""
        self._session = session
        self._invalidate_cache()
        self._recompute_letterbox()
        self._resolved_motion_source = "auto"
        self._emit_source_status()
        self._emit_mesh_status()

    def set_person(self, person_id: int):
        """Select which person's mesh to display."""
        if person_id == self._person_id:
            return
        self._invalidate_cache()
        self._person_id = person_id
        # Update active skeleton def and coordinate space based on track data
        if self._session is not None:
            track = self._session.person_tracks.get(person_id)
            if track is not None and track.body_model_type == "soma":
                self._active_skel = _SOMA_SKEL
            else:
                self._active_skel = _SKEL
            self._data_is_global = self._track_supports_world(track)
        else:
            self._data_is_global = False
        self._update_camera()
        self._refresh_mesh()
        self._emit_source_status()
        self._emit_mesh_status()

    def set_motion_source(self, source: str):
        """Set the preferred body-motion source used by the viewport."""
        if source not in ("auto", "camera_baseline", "world_baseline", "world_physics"):
            return
        if source == self._motion_source:
            return
        self._motion_source = source
        self._invalidate_cache()
        if self._session is not None and self._person_id >= 0:
            track = self._session.person_tracks.get(self._person_id)
            self._data_is_global = self._track_supports_world(track)
        self._refresh_mesh()
        self._update_camera()
        self._emit_source_status()
        self._emit_mesh_status()

    def on_frame_changed(self, frame_idx: int):
        """Update displayed frame."""
        if frame_idx == self._current_frame:
            return
        self._current_frame = frame_idx
        if self._camera_mode == "incam" and self._data_is_global:
            self._update_camera()
        self._refresh_mesh()
        self._emit_source_status()
        self._emit_mesh_status()

    @property
    def camera_mode(self) -> str:
        """Current camera mode ('incam' or 'orbit')."""
        return self._camera_mode

    def set_frustum_fov(self, fov_deg: float):
        """Override the horizontal FOV used for frustum rendering."""
        self._frustum_fov_override = fov_deg
        if _HAS_GL:
            self.update()

    def get_frustum_fov(self) -> float:
        """Current horizontal FOV in degrees from camera_K, focal_mm, or default."""
        if self._frustum_fov_override is not None:
            return self._frustum_fov_override
        sess = self._session
        if sess is not None:
            w = sess.img_width or 1920
            if sess.camera_K is not None:
                fx = float(sess.camera_K[0, 0])
                return float(np.degrees(2 * np.arctan(w / (2 * fx))))
            # Derive from focal_mm assuming 36mm sensor width
            f_mm = sess.focal_mm if sess.focal_mm else 24.0
            fx = f_mm / 36.0 * w
            return float(np.degrees(2 * np.arctan(w / (2 * fx))))
        return 60.0

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
        self._incam_offset = np.zeros(3, dtype=np.float32)  # reset stale offset
        self._invalidate_cache()  # vertices depend on camera mode (incam vs world params)
        if mode == "orbit":
            self._orbit_auto_centered = False  # force re-center on mode switch
        self._refresh_mesh()  # recompute joints/vertices in new mode's coordinate space
        self._update_camera()
        self.camera_changed.emit(self._camera_state())
        self._emit_source_status()

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

    def set_show_camera_frustum(self, show: bool):
        """Toggle camera frustum wireframe visibility in orbit mode."""
        if show == self._show_camera_frustum:
            return
        self._show_camera_frustum = show
        if _HAS_GL:
            self.update()

    def set_video_frame(self, frame: np.ndarray | None):
        """Set the video frame to render as background in in-camera mode.

        Parameters
        ----------
        frame : (H, W, 3) uint8 RGB array, or None to clear.
                The array is used by reference — caller must not mutate it
                after this call until the next set_video_frame.
        """
        self._video_frame = frame
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
        """Auto-switch render mode during active scrubbing/playback.

        When *active* is True the current render mode is saved and the
        viewport switches to _playback_render_mode.  When *active* becomes
        False the previous mode is restored.  Also starts/stops the
        background vertex prefetcher.
        """
        if active:
            if self._pre_scrub_mode is None:
                self._pre_scrub_mode = self._render_mode
                if self._render_mode != self._playback_render_mode:
                    self._render_mode = self._playback_render_mode
                    self._refresh_mesh()
            # Start prefetching ahead of playback position
            if self._render_mode != RenderMode.WIREFRAME:
                self._start_prefetch()
        else:
            self._stop_prefetch()
            if self._pre_scrub_mode is not None:
                restore = self._pre_scrub_mode
                self._pre_scrub_mode = None
                if restore != self._render_mode:
                    self._render_mode = restore
                    self._refresh_mesh()

    def set_playback_render_mode(self, mode: RenderMode):
        """Set the render mode used during playback/scrubbing.

        If currently scrubbing, applies the new mode immediately.
        """
        self._playback_render_mode = mode
        if self._pre_scrub_mode is not None and self._render_mode != mode:
            self._render_mode = mode
            self._refresh_mesh()

    # ------------------------------------------------------------------
    # Background vertex prefetch (fills cache ahead of playback)
    # ------------------------------------------------------------------

    def _start_prefetch(self):
        """Launch a daemon thread to pre-compute vertices ahead of playback."""
        self._stop_prefetch()  # stop any existing thread
        if self._session is None or self._person_id < 0:
            return
        track = self._session.person_tracks.get(self._person_id)
        if track is None:
            return
        params, _source = self._params_for_track(track)
        if params is None:
            return
        # Determine total frame count from params
        go = params.get("global_orient")
        if go is None:
            return
        go_arr = np.asarray(go) if not isinstance(go, np.ndarray) else go
        n_frames = go_arr.shape[0] if go_arr.ndim >= 2 else 1

        self._prefetch_stop.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(self._person_id, self._current_frame, n_frames),
            daemon=True,
            name="vertex-prefetch",
        )
        self._prefetch_thread.start()

    def _stop_prefetch(self):
        """Signal the prefetch thread to stop and wait for it."""
        self._prefetch_stop.set()
        t = self._prefetch_thread
        if t is not None and t.is_alive():
            t.join(timeout=0.5)
        self._prefetch_thread = None

    def _prefetch_worker(self, person_id: int, start_frame: int, n_frames: int):
        """Background thread: compute vertices for frames ahead of playback.

        Runs the same _compute_vertices logic but stores results in the
        shared cache.  Stops when _prefetch_stop is set or all frames are done.
        """
        for offset in range(_PREFETCH_AHEAD):
            if self._prefetch_stop.is_set():
                return
            frame = start_frame + offset
            if frame >= n_frames:
                frame = frame % n_frames  # wrap for looping playback
            cache_key = (person_id, frame)
            with self._cache_lock:
                if cache_key in self._vertex_cache:
                    continue
            # Compute on this thread (no GL calls — just torch + numpy)
            try:
                result = self._compute_vertices(person_id, frame)
                if result is not None:
                    with self._cache_lock:
                        if len(self._vertex_cache) >= _CACHE_MAX:
                            oldest = next(iter(self._vertex_cache))
                            del self._vertex_cache[oldest]
                        self._vertex_cache[cache_key] = result
            except Exception as exc:
                logger.debug("prefetch frame %d failed: %s", frame, exc)
                continue

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
            with self._cache_lock:
                self._vertex_cache.pop((self._person_id, frame_idx), None)
        self._refresh_mesh()

    def refresh(self):
        """Force a full mesh + skeleton recompute for the current frame.

        Use when underlying params have changed but the frame index hasn't,
        which would cause ``on_frame_changed`` to early-return.
        """
        self._invalidate_cache()
        self._refresh_mesh()

    def invalidate_cache(self, person_id: int | None = None, frame_idx: int | None = None):
        """Invalidate vertex cache entries.

        If both person_id and frame_idx are given, removes that single entry.
        Otherwise clears the entire cache.
        """
        if person_id is not None and frame_idx is not None:
            with self._cache_lock:
                self._vertex_cache.pop((person_id, frame_idx), None)
        else:
            self._invalidate_cache()

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

        # Root position offset (Shift+drag on pelvis)
        transl_offset = self._pose_override.get("transl_offset")
        if transl_offset is not None:
            key = "transl"  # already mapped from transl_world by caller
            if key in params:
                tr = np.array(params[key], dtype=np.float32)
                if tr.ndim >= 2 and frame_idx < tr.shape[0]:
                    tr = tr.copy()
                    tr[frame_idx] = tr[frame_idx] + transl_offset
                    params[key] = tr

        return params

    def set_interpolation_enabled(self, enabled: bool):
        """Toggle viewport SLERP interpolation preview between correction keyframes."""
        if self._interpolation_enabled == enabled:
            return
        self._interpolation_enabled = enabled
        self._invalidate_cache()
        self._refresh_mesh()

    @property
    def interpolation_enabled(self) -> bool:
        return self._interpolation_enabled

    def _apply_interpolation_to_params(self, params: dict, frame_idx: int) -> dict:
        """Apply SLERP interpolation between adjacent correction keyframes.

        If frame_idx falls between two correction keyframes, builds the full
        corrected pose at both keyframes and SLERP-interpolates between them.
        This matches the export-time behavior of apply_corrections().
        """
        if not self._interpolation_enabled:
            return params
        if self._session is None or self._person_id < 0:
            return params

        ct = self._session.correction_tracks.get(self._person_id)
        if ct is None or not ct.corrections:
            return params

        # If this frame IS a keyframe, apply the correction directly
        corr_at_frame = ct.get_correction(frame_idx)
        if corr_at_frame is not None:
            try:
                from pose_correction import _build_full_correction_pose
            except ImportError:
                return params

            go, bp, tr = _build_full_correction_pose(params, corr_at_frame)
            params = dict(params)
            go_arr = np.array(params["global_orient"], dtype=np.float32)
            if go_arr.ndim >= 2 and frame_idx < go_arr.shape[0]:
                go_arr[frame_idx] = go
                params["global_orient"] = go_arr
            bp_arr = np.array(params["body_pose"], dtype=np.float32)
            if bp_arr.ndim == 2 and bp_arr.shape[-1] != 3:
                bp_arr = bp_arr.reshape(bp_arr.shape[0], -1, 3)
            if bp_arr.ndim >= 3 and frame_idx < bp_arr.shape[0]:
                bp_arr[frame_idx] = bp
                params["body_pose"] = bp_arr
            tr_arr = np.array(params["transl"], dtype=np.float32)
            if tr_arr.ndim >= 2 and frame_idx < tr_arr.shape[0]:
                tr_arr[frame_idx] = tr
                params["transl"] = tr_arr
            return params

        # Check if we're between two correction keyframes
        prev_corr, next_corr = ct.get_surrounding(frame_idx)
        if (prev_corr is None or next_corr is None
                or prev_corr is next_corr
                or prev_corr.frame_index >= frame_idx
                or next_corr.frame_index <= frame_idx):
            return params

        try:
            from pose_correction import _build_full_correction_pose, _slerp_axis_angle
        except ImportError:
            return params

        go_a, bp_a, tr_a = _build_full_correction_pose(params, prev_corr)
        go_b, bp_b, tr_b = _build_full_correction_pose(params, next_corr)

        t = (frame_idx - prev_corr.frame_index) / (next_corr.frame_index - prev_corr.frame_index)

        # SLERP global orient
        go_interp = _slerp_axis_angle(go_a, go_b, t).astype(np.float32)

        # SLERP each body joint
        n_joints = bp_a.shape[0]
        bp_interp = np.empty_like(bp_a)
        for j in range(n_joints):
            bp_interp[j] = _slerp_axis_angle(bp_a[j], bp_b[j], t).astype(np.float32)

        # Lerp translation
        tr_interp = ((1 - t) * tr_a + t * tr_b).astype(np.float32)

        # Apply to params copy
        params = dict(params)
        go_arr = np.array(params["global_orient"], dtype=np.float32)
        if go_arr.ndim >= 2 and frame_idx < go_arr.shape[0]:
            go_arr[frame_idx] = go_interp
            params["global_orient"] = go_arr
        bp_arr = np.array(params["body_pose"], dtype=np.float32)
        if bp_arr.ndim == 2 and bp_arr.shape[-1] != 3:
            bp_arr = bp_arr.reshape(bp_arr.shape[0], -1, 3)
        if bp_arr.ndim >= 3 and frame_idx < bp_arr.shape[0]:
            bp_arr[frame_idx] = bp_interp
            params["body_pose"] = bp_arr
        tr_arr = np.array(params["transl"], dtype=np.float32)
        if tr_arr.ndim >= 2 and frame_idx < tr_arr.shape[0]:
            tr_arr[frame_idx] = tr_interp
            params["transl"] = tr_arr

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

    def _use_world_params(self) -> bool:
        """Whether to use world-space orient/transl for joints and vertices."""
        return self._data_is_global

    def _track_motion_sources(self, track) -> dict[str, dict]:
        if track is None or track.body_model_type == "soma":
            return {}
        return getattr(track, "motion_sources", {}) or {}

    def _params_have_world_payload(self, params: dict | None) -> bool:
        return bool(
            params is not None
            and "global_orient_world" in params
            and "transl_world" in params
        )

    def _source_uses_world_payload(self, source: str, params: dict | None) -> bool:
        if params is None:
            return False
        if source in ("world_baseline", "world_physics"):
            return True
        if source == "soma":
            return self._params_have_world_payload(params)
        return False

    def _canonical_world_source(
        self,
        track,
        sources: dict[str, dict],
    ) -> tuple[dict | None, str | None]:
        world_physics = sources.get("world_physics")
        if world_physics is not None:
            return world_physics, "world_physics"

        world_baseline = sources.get("world_baseline")
        if world_baseline is not None:
            return world_baseline, "world_baseline"

        fallback = getattr(track, "smplx_params", None)
        if self._params_have_world_payload(fallback):
            return fallback, "world_baseline"

        return None, None

    def _track_supports_world(self, track) -> bool:
        params, source = self._params_for_track(track)
        return self._source_uses_world_payload(source, params)

    def _world_source_state(self, track) -> tuple[bool, str]:
        if track is None:
            return False, "not available"
        params, source = self._params_for_track(track)
        if not self._source_uses_world_payload(source, params):
            return False, "camera fallback"
        if self._camera_mode == "orbit":
            return True, "orbit"
        if self._session is not None and self._session.derived_c2w is not None:
            return True, "in-camera"
        return True, "missing derived camera alignment"

    def _motion_source_display_text(self, source: str) -> str:
        labels = {
            "auto": "Auto",
            "soma": "SOMA",
            "camera_baseline": "Camera baseline",
            "world_baseline": "World baseline",
            "world_physics": "World physics",
        }
        return labels.get(source, source.replace("_", " ").title())

    def _requested_motion_source_label_text(self) -> str:
        return self._motion_source_display_text(self._motion_source)

    def _params_for_track(self, track) -> tuple[dict | None, str]:
        """Resolve the currently active params dict for a track."""
        if track is None:
            self._resolved_motion_source = "auto"
            return None, "auto"
        if track.body_model_type == "soma":
            self._resolved_motion_source = "soma"
            return track.soma_params, "soma"

        sources = self._track_motion_sources(track)
        fallback = track.smplx_params
        requested = self._motion_source
        if requested in ("world_physics", "world_baseline"):
            params = sources.get(requested)
            if params is not None:
                self._resolved_motion_source = requested
                return params, requested

        world_params, world_source = self._canonical_world_source(track, sources)
        if world_params is not None and world_source is not None:
            self._resolved_motion_source = world_source
            return world_params, world_source

        if requested == "camera_baseline":
            params = sources.get("camera_baseline") or fallback
            if params is not None:
                self._resolved_motion_source = "camera_baseline"
                return params, "camera_baseline"

        camera_params = sources.get("camera_baseline")
        if camera_params is not None:
            self._resolved_motion_source = "camera_baseline"
            return camera_params, "camera_baseline"

        self._resolved_motion_source = "camera_baseline" if fallback is not None else "auto"
        return fallback, self._resolved_motion_source

    def active_motion_source(self) -> str:
        """Return the resolved motion source for the selected track."""
        if self._session is None or self._person_id < 0:
            return self._resolved_motion_source
        track = self._session.person_tracks.get(self._person_id)
        _params, source = self._params_for_track(track)
        return source

    def _motion_source_label_text(self) -> str:
        source = self.active_motion_source()
        return self._motion_source_display_text(source)

    def current_source_status_text(self) -> str:
        if self._session is None:
            return "Motion: waiting for session"
        if self._person_id < 0:
            return "Motion: no person selected"
        track = self._session.person_tracks.get(self._person_id)
        if track is None:
            return "Motion: no selected person"
        params, source = self._params_for_track(track)
        if params is None:
            return "Motion: unavailable"
        world_active, world_reason = self._world_source_state(track)
        parts = [f"Motion: {self._motion_source_display_text(source)}"]
        sources = self._track_motion_sources(track)
        motion_contract = params.get("motion_contract") if isinstance(params, dict) else None
        source_tag = (
            track.smplx_params.get("source")
            if isinstance(getattr(track, "smplx_params", None), dict)
            else None
        )
        if source == "world_physics":
            if motion_contract == "refined_hmr4d":
                # Loaded from hmr4d_results.pt because no hybrid snapshot was
                # available; annotate the legacy fallback path explicitly.
                parts.append("refined hmr4d")
            elif source_tag == "phc_refined":
                # Happy path: loaded from a phc_refined_hybrid_smplx.pt
                # snapshot.  Source label alone says "World physics", add a
                # concise "physics refined" tag so users can tell at a glance
                # that PHC ran (and not just that baseline world motion is
                # active).
                parts.append("physics refined")
            elif source_tag == "spring_refined":
                # Parallel happy path for the spring-filter refinement:
                # data loaded from spring_refined_hybrid_smplx.pt is stored
                # under the *_world_physics triad for viewport compat, but
                # the source tag lets us report the correct refiner.
                parts.append("spring refined")
            elif source_tag == "spring_refined_pinned":
                # v2 foot-pinned spring output. Same plumbing as
                # ``spring_refined``, just the tag is distinct so users can
                # see at a glance that the contact-aware pin pass ran.
                parts.append("spring refined (pinned)")
        if source == "world_baseline" and sources.get("world_physics") is None:
            if source_tag == "phc_refined":
                parts.append("physics expected but artifact missing")
            elif source_tag in ("spring_refined", "spring_refined_pinned"):
                parts.append("spring expected but artifact missing")
            else:
                parts.append("world physics missing")
        if not world_active and world_reason == "camera fallback":
            parts.append("camera fallback")
        elif world_active and self._camera_mode == "incam" and world_reason == "missing derived camera alignment":
            parts.append("in-camera view missing alignment")

        # Append file artifact name + modification timestamp for freshness
        stamp = getattr(track, "_motion_file_stamp", (None, None))
        if stamp[1] is not None:
            from datetime import datetime
            mtime = datetime.fromtimestamp(stamp[1] / 1e9)
            time_str = mtime.strftime("%H:%M:%S")
            artifact = stamp[0].rsplit("/", 1)[-1] if stamp[0] else "?"
            parts.append(f"{artifact} @ {time_str}")

        return " | ".join(parts)

    def current_source_status_detail_text(self) -> str:
        requested = self._requested_motion_source_label_text()
        if self._session is None:
            return f"Requested: {requested} | Resolved: none | World: no session"
        if self._person_id < 0:
            return f"Requested: {requested} | Resolved: none | World: no person selected"
        track = self._session.person_tracks.get(self._person_id)
        if track is None:
            return f"Requested: {requested} | Resolved: none | World: no selected person"
        params, source = self._params_for_track(track)
        world_active, world_reason = self._world_source_state(track)
        resolved = self._motion_source_display_text(source) if params is not None else "none"
        world_text = (
            "active"
            if world_active and world_reason in ("orbit", "in-camera")
            else world_reason
        )
        parts = [
            f"Requested: {requested}",
            f"Resolved: {resolved}",
            f"World: {world_text}",
        ]
        if params is not None:
            motion_artifact = params.get("motion_artifact")
            if motion_artifact:
                parts.append(f"Artifact: {motion_artifact}")
            wrist_source = params.get("wrist_debug_source")
            if wrist_source:
                parts.append(f"Wrist: {wrist_source}")
        return " | ".join(parts)

    def _emit_source_status(self):
        text = self.current_source_status_text()
        self.source_status_changed.emit(text)
        self._hud.update_info(source=text)

    def current_mesh_status_text(self) -> str:
        reason = self._mesh_draw_gate_reason()
        if reason and reason not in self._mesh_status:
            return f"{self._mesh_status} | {reason}"
        return self._mesh_status

    def current_mesh_status_detail_text(self) -> str:
        base = self.current_mesh_status_text()
        detail = (
            f"mode={self._render_mode.value} "
            f"vertices={'yes' if self._vertices is not None else 'no'} "
            f"n_vertices={self._n_vertices} "
            f"n_indices={self._n_indices} "
            f"gl={'ready' if self._gl_ready else 'not-ready'}"
        )
        return f"{base} | {detail}"

    def _mesh_draw_gate_reason(self) -> str | None:
        if self._render_mode == RenderMode.WIREFRAME:
            return "triangles skipped (Skeleton mode)"
        if self._vertices is None:
            return None
        if not self._gl_ready:
            return "triangles waiting for GL"
        if self._n_indices <= 0:
            return "triangles skipped (no indices)"
        return None

    def _set_mesh_status(self, text: str):
        self._mesh_status = text
        self._status_msg = text
        if hasattr(self, "_fallback_label"):
            self._fallback_label.setText(text)
        self._hud.update_info(mesh=text)
        self._emit_mesh_status()

    def _emit_mesh_status(self):
        self.mesh_status_changed.emit(self.current_mesh_status_text())

    def _incam_transform_points(self, pts: np.ndarray, person_id: int) -> np.ndarray:
        """Transform points from crop camera space to full-frame camera space (incam only)."""
        if self._camera_mode != "incam" or self._session is None:
            return pts
        if self._use_world_params():
            return pts  # world-space data: view matrix handles positioning
        track = self._session.person_tracks.get(person_id)
        if track is None or track.K_crop is None or track.crop_bbox is None:
            return pts
        img_w = self._session.img_width or 1920
        img_h = self._session.img_height or 1080
        K_full = estimate_K(img_w, img_h)
        K_c = track.K_crop
        x1, y1 = track.crop_bbox[0], track.crop_bbox[1]
        X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
        X_f = (K_c[0, 0] * X + (K_c[0, 2] + x1 - K_full[0, 2]) * Z) / K_full[0, 0]
        Y_f = (K_c[1, 1] * Y + (K_c[1, 2] + y1 - K_full[1, 2]) * Z) / K_full[1, 1]
        return np.stack([X_f, Y_f, Z], axis=-1).astype(np.float32)

    def _incam_world_view(self, frame_idx: int) -> np.ndarray | None:
        """Per-frame incam view matrix from camera-to-world data."""
        sess = self._session
        if sess is None:
            return None
        c2w_all = sess.derived_c2w
        if c2w_all is None:
            return None
        idx = min(frame_idx, len(c2w_all) - 1)
        w2c = np.linalg.inv(c2w_all[idx])
        return (_CV_TO_GL @ w2c).astype(np.float32)

    def _incam_effective_K(self) -> np.ndarray | None:
        """Effective intrinsics for incam world-space rendering.

        Returns K_crop with principal point shifted by the crop offset,
        so projecting crop-camera-space coordinates gives correct
        full-frame pixel positions.
        """
        if self._session is None:
            return None
        tracks = self._session.person_tracks
        if not tracks:
            return None
        # Use primary person (first track — the one derived_c2w was computed from)
        track = next(iter(tracks.values()))
        if track.K_crop is None or track.crop_bbox is None:
            return None
        K = track.K_crop.astype(np.float32).copy()
        K[0, 2] += track.crop_bbox[0]   # cx += x1
        K[1, 2] += track.crop_bbox[1]   # cy += y1
        return K

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
            view = self._incam_world_view(self._current_frame)
            if view is None:
                view = _CV_TO_GL.copy()
            # Apply manual alignment offset (pan/zoom)
            if np.any(self._incam_offset != 0):
                T = np.eye(4, dtype=np.float32)
                T[:3, 3] = self._incam_offset
                view = T @ view
            self._view = view
        lx, ly, lw, lh = self._letterbox
        w = lw if lw > 0 else 200
        h = lh if lh > 0 else 150
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
        """Set orbit center to the current mesh/skeleton centroid in GL space."""
        # Use vertices if available, otherwise fall back to joint positions
        pts = self._vertices
        if pts is None:
            pts = self._joint_positions
        # Combine all persons' joints for multi-person centering
        if self._all_joint_positions:
            all_pts = list(self._all_joint_positions.values())
            if pts is not None:
                all_pts.append(pts)
            pts = np.concatenate(all_pts, axis=0) if all_pts else pts
        if pts is None:
            return

        # Global-space data (GEM-X) is already Y-up;
        # camera-space data (GVHMR) needs Y/Z flip to GL convention.
        need_flip = not self._data_is_global
        centroid = pts.mean(axis=0).copy()
        if need_flip:
            centroid[1] *= -1
            centroid[2] *= -1
        self._orbit_center = centroid.astype(np.float32)

        gl_pts = pts.copy()
        if need_flip:
            gl_pts[:, 1] *= -1
            gl_pts[:, 2] *= -1
        extent = np.max(
            np.linalg.norm(gl_pts - self._orbit_center, axis=1)
        )
        self._orbit_distance = max(float(extent) * 2.5, 1.0)
        # Place grid floor at lowest point (feet)
        self._grid_y = float(np.min(gl_pts[:, 1]))
        self._grid_center_x = float(self._orbit_center[0])
        self._grid_center_z = float(self._orbit_center[2])
        self._orbit_auto_centered = True

    def mousePressEvent(self, event):
        """Joint picking on left-click, orbit drag on left-drag in orbit mode."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Convert to viewport coords; ignore clicks on black bars
            wx, wy = event.position().x(), event.position().y()
            vp = self._widget_to_viewport(wx, wy)
            if vp is None:
                logger.warning("click rejected: widget=(%.0f,%.0f) letterbox=%r",
                             wx, wy, self._letterbox)
                event.accept()
                return
            # Try joint picking first
            hit = self._pick_joint(vp[0], vp[1])
            if hit is not None:
                self._selected_joint = hit
                self.joint_clicked.emit(hit)
                if _HAS_GL:
                    self.update()
                # Shift+click on any joint → root drag (position correction)
                if event.modifiers() & Qt.ShiftModifier and 0 <= hit <= 21:
                    self._drag_mode = "root_drag"
                    self._drag_joint = 0
                    self._root_drag_offset = np.zeros(3, dtype=np.float32)
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                # Start joint drag if it's a body joint (0-21)
                elif 0 <= hit <= 21:
                    self._drag_mode = "joint_drag"
                    self._drag_joint = hit
                    self._drag_start_aa = self._get_current_joint_aa(hit)
                    try:
                        from scipy.spatial.transform import Rotation
                        self._drag_cumulative_R = Rotation.identity()
                    except ImportError:
                        self._drag_mode = "none"
                else:
                    self._drag_mode = "orbit"
            else:
                self._drag_mode = "orbit"

        # Track mouse position for drag (orbit, joint drag, root drag, or incam pan)
        if (self._camera_mode == "orbit"
                or self._drag_mode in ("joint_drag", "root_drag")
                or (self._camera_mode == "incam"
                    and event.button() == Qt.MouseButton.MiddleButton)):
            self._mouse_last_pos = (event.position().x(), event.position().y())
        event.accept()

    def mouseMoveEvent(self, event):
        """Update orbit/pan during drag; show HUD on movement."""
        event.accept()
        # Show HUD on any mouse activity over the viewport
        if self._hud_enabled:
            self._hud.show_with_timer()

        # Root drag (Shift+drag pelvis) takes priority
        if self._drag_mode == "root_drag" and self._mouse_last_pos is not None:
            x, y = event.position().x(), event.position().y()
            dx = x - self._mouse_last_pos[0]
            dy = y - self._mouse_last_pos[1]
            self._mouse_last_pos = (x, y)
            if event.buttons() & Qt.MouseButton.LeftButton:
                self._handle_root_drag(dx, dy)
            return

        # Joint drag-to-rotate takes priority over orbit
        if self._drag_mode == "joint_drag" and self._mouse_last_pos is not None:
            x, y = event.position().x(), event.position().y()
            dx = x - self._mouse_last_pos[0]
            dy = y - self._mouse_last_pos[1]
            self._mouse_last_pos = (x, y)
            if event.buttons() & Qt.MouseButton.LeftButton:
                self._handle_joint_drag(dx, dy)
            return

        # Incam pan (middle-drag adjusts offset)
        if self._camera_mode == "incam" and self._mouse_last_pos is not None:
            if event.buttons() & Qt.MouseButton.MiddleButton:
                x, y = event.position().x(), event.position().y()
                dx = x - self._mouse_last_pos[0]
                dy = y - self._mouse_last_pos[1]
                self._mouse_last_pos = (x, y)
                speed = 0.005
                self._incam_offset[0] += float(dx) * speed
                self._incam_offset[1] -= float(dy) * speed  # GL Y is up
                self._update_camera()
                return

        if self._camera_mode != "orbit" or self._mouse_last_pos is None:
            return

        x, y = event.position().x(), event.position().y()
        dx = x - self._mouse_last_pos[0]
        dy = y - self._mouse_last_pos[1]
        self._mouse_last_pos = (x, y)

        buttons = event.buttons()
        if buttons & Qt.MouseButton.LeftButton and self._drag_mode != "joint_drag":
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
        """End orbit/pan drag, joint drag, or root drag."""
        if self._drag_mode == "root_drag":
            offset = self._root_drag_offset.copy()
            if np.linalg.norm(offset) > 1e-6:
                self.root_drag_committed.emit(self._person_id, offset)
            # Clear live preview
            if self._pose_override:
                self._pose_override.pop("transl_offset", None)
                if not any(k != "frame_idx" for k in self._pose_override):
                    self.set_pose_override(None)
                else:
                    self.set_pose_override(self._pose_override)
            self._root_drag_offset = np.zeros(3, dtype=np.float32)
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self._drag_mode = "none"
        self._mouse_last_pos = None
        event.accept()

    def wheelEvent(self, event):
        """Zoom in/out in orbit or incam mode."""
        if self._camera_mode == "incam":
            delta = event.angleDelta().y()
            self._incam_offset[2] += (delta / 120.0) * 0.1
            self._update_camera()
            event.accept()
            return
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
        """Reset camera on double-click."""
        if self._camera_mode == "orbit":
            self._reset_orbit()
            self._update_camera()
            self.camera_changed.emit(self._camera_state())
            event.accept()
            return
        if self._camera_mode == "incam":
            self._incam_offset = np.zeros(3, dtype=np.float32)
            self._update_camera()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # ------------------------------------------------------------------
    # Joint drag-to-rotate
    # ------------------------------------------------------------------

    def _get_current_joint_aa(self, joint_idx: int) -> np.ndarray:
        """Get current axis-angle (3,) for a joint, checking pose override first."""
        # Check active pose override first (for chained drags)
        if self._pose_override is not None:
            ov_frame = self._pose_override.get("frame_idx", self._current_frame)
            if ov_frame == self._current_frame:
                if joint_idx == 0:
                    go = self._pose_override.get("global_orient")
                    if go is not None:
                        return np.asarray(go, dtype=np.float32).ravel()[:3]
                elif 1 <= joint_idx <= 21:
                    bp = self._pose_override.get("body_pose")
                    if bp is not None and (joint_idx - 1) in bp:
                        return np.asarray(bp[joint_idx - 1], dtype=np.float32).ravel()[:3]

        # Fall back to module-level helper from pose_corrector_panel
        from views.pose_corrector_panel import _get_joint_axis_angle
        return _get_joint_axis_angle(
            self._session, self._person_id, self._current_frame, joint_idx
        )

    def _handle_joint_drag(self, dx: float, dy: float):
        """Rotate the dragged joint based on mouse delta in camera space."""
        if self._drag_joint < 0 or self._drag_start_aa is None:
            return
        try:
            from scipy.spatial.transform import Rotation
        except ImportError:
            return

        # Camera right/up vectors from view matrix (world space)
        cam_right = self._view[0, :3].copy().astype(np.float64)
        cam_up = self._view[1, :3].copy().astype(np.float64)
        r_len = np.linalg.norm(cam_right)
        u_len = np.linalg.norm(cam_up)
        if r_len > 1e-8:
            cam_right /= r_len
        if u_len > 1e-8:
            cam_up /= u_len

        # Transform camera vectors from world space to model/parameter space.
        # When _data_is_global=False, _model_mat is _CV_TO_GL which flips Y/Z;
        # without this transform the drag rotation axis is inverted, causing
        # the skeleton to flip upside down on first drag.
        model_R = self._model_mat[:3, :3].astype(np.float64)
        det = np.linalg.det(model_R)
        if abs(det) > 1e-8:
            model_R_inv = np.linalg.inv(model_R)
            cam_right = model_R_inv @ cam_right
            cam_up = model_R_inv @ cam_up

        # Horizontal drag → rotation around camera-up axis
        # Vertical drag → rotation around camera-right axis
        angle_h = np.radians(dx * _JOINT_DRAG_SENSITIVITY)
        angle_v = np.radians(-dy * _JOINT_DRAG_SENSITIVITY)

        R_h = Rotation.from_rotvec(cam_up * angle_h)
        R_v = Rotation.from_rotvec(cam_right * angle_v)
        R_increment = R_v * R_h

        # Accumulate
        self._drag_cumulative_R = R_increment * self._drag_cumulative_R

        # Final rotation = cumulative * original
        R_original = Rotation.from_rotvec(self._drag_start_aa.astype(np.float64))
        R_final = self._drag_cumulative_R * R_original
        aa_final = R_final.as_rotvec().astype(np.float32)

        # Build pose override
        override = {"frame_idx": self._current_frame}
        if self._drag_joint == 0:
            override["global_orient"] = aa_final
        else:
            override["body_pose"] = {self._drag_joint - 1: aa_final}
        self.set_pose_override(override)

        # Emit euler for slider sync
        euler_deg = R_final.as_euler("XYZ", degrees=True).astype(np.float32)
        self.joint_drag_updated.emit(self._drag_joint, euler_deg)

    def _handle_root_drag(self, dx: float, dy: float):
        """Translate root in XZ ground plane from mouse delta."""
        # Camera right/forward projected onto XZ plane
        cam_right = self._view[0, :3].copy().astype(np.float64)
        cam_fwd = -self._view[2, :3].copy().astype(np.float64)
        cam_right[1] = 0.0
        cam_fwd[1] = 0.0
        rn = np.linalg.norm(cam_right)
        fn = np.linalg.norm(cam_fwd)
        if rn > 1e-8:
            cam_right /= rn
        if fn > 1e-8:
            cam_fwd /= fn

        speed = self._orbit_distance * _ROOT_DRAG_SENSITIVITY
        delta = (cam_right * float(dx) + cam_fwd * float(-dy)) * speed
        self._root_drag_offset += delta.astype(np.float32)

        # Live preview via pose_override
        override = dict(self._pose_override or {})
        override["frame_idx"] = self._current_frame
        override["transl_offset"] = self._root_drag_offset.copy()
        self.set_pose_override(override)

    def keyPressEvent(self, event):
        """Esc cancels active root drag or joint drag preview."""
        if event.key() == Qt.Key.Key_Escape:
            # Cancel root drag first
            if self._drag_mode == "root_drag":
                self._root_drag_offset = np.zeros(3, dtype=np.float32)
                if self._pose_override:
                    self._pose_override.pop("transl_offset", None)
                self.set_pose_override(None)
                self._drag_mode = "none"
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                return
            if self._pose_override is not None:
                joint_idx = self._drag_joint
                self.set_pose_override(None)
                self._drag_mode = "none"
                self._drag_joint = -1
                self._drag_start_aa = None
                self._drag_cumulative_R = None
                self.joint_drag_updated.emit(joint_idx, None)
                event.accept()
                return
        super().keyPressEvent(event)

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

        action = menu.exec_(event.globalPos())
        if action is None:
            return

        if action == act_chain:
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

    def _world_params_for_person(self, params: dict, pid: int) -> dict:
        """Return world-space orient/transl for a person.

        Multi-person XZ offsets are already baked into transl_world at load
        time (from assembly/person_offsets.json), so every person just uses
        their own world params directly.
        """
        params = dict(params)
        params["global_orient"] = params["global_orient_world"]
        params["transl"] = params["transl_world"]
        if "body_pose_world" in params:
            params["body_pose"] = params["body_pose_world"]
        return params

    def _compute_joints(self) -> np.ndarray | None:
        """Compute 3D joint positions for current person/frame.

        Returns (J, 3) array in camera space, or None if params unavailable.
        """
        if self._session is None or self._person_id < 0:
            return None
        track = self._session.person_tracks.get(self._person_id)
        if track is None:
            logger.debug("_compute_joints: no track for pid=%d", self._person_id)
            return None
        params, _source = self._params_for_track(track)
        if params is None:
            logger.debug(
                "_compute_joints: no params for pid=%d (model=%s, soma=%s, smplx=%s)",
                self._person_id, track.body_model_type,
                track.soma_params is not None, track.smplx_params is not None,
            )
            return None

        # Use world-space orient/transl if available and appropriate
        # so characters stay grounded while the camera orbits freely
        if ("global_orient_world" in params
                and "transl_world" in params
                and self._use_world_params()):
            params = self._world_params_for_person(params, self._person_id)

        # Apply interpolation between correction keyframes
        params = self._apply_interpolation_to_params(params, self._current_frame)

        # Apply pose override for real-time preview
        if self._pose_override is not None:
            ov_frame = self._pose_override.get("frame_idx", self._current_frame)
            if ov_frame == self._current_frame:
                params = self._apply_override_to_params(params, self._current_frame)

        try:
            joints = forward_kinematics(params, self._current_frame)
            if joints is not None:
                joints = self._incam_transform_points(joints, self._person_id)
            return joints
        except Exception as e:
            import traceback as _tb
            logger.warning("FK failed (pid=%d, f=%d): %s\n%s",
                           self._person_id, self._current_frame, e,
                           _tb.format_exc())
            return None

    def _pick_joint(self, screen_x: float, screen_y: float) -> int | None:
        """Pick the joint at screen position, using the active picking mode.

        When ``_picking_mode`` is ``'fbo'``, renders joints as uniquely
        colored points into an offscreen FBO and reads back the pixel.
        Falls back to screen-space distance if FBO picking fails or is
        disabled.

        screen_x/screen_y are in viewport-space (relative to letterbox origin).
        """
        if self._joint_positions is None:
            logger.warning("pick_joint: no joint_positions")
            return None

        if self._picking_mode == "fbo" and _HAS_GL and self._gl_ready:
            result = self._fbo_pick_joint(screen_x, screen_y)
            if result is not None:
                return result
            # Fall through to screen-space if FBO returned no hit

        # Screen-space distance fallback (viewport-space coords)
        lx, ly, lw, lh = self._letterbox
        w = lw if lw > 0 else 200
        h = lh if lh > 0 else 150
        mvp = self._projection @ self._view @ self._model_mat
        screen = project_joints_to_screen(self._joint_positions, mvp, w, h)
        result = find_nearest_joint(screen_x, screen_y, screen)
        if result is None:
            # Diagnostic: log nearest distance to help debug picking failures
            body = screen[:min(len(screen), 22)]
            dists = np.sqrt((body[:, 0] - screen_x) ** 2 + (body[:, 1] - screen_y) ** 2)
            min_dist = float(np.min(dists)) if len(dists) > 0 else -1
            logger.warning(
                "pick_joint miss: click=(%.0f,%.0f) letterbox=(%d,%d,%d,%d) "
                "nearest_dist=%.1f threshold=%.0f n_joints=%d",
                screen_x, screen_y, lx, ly, lw, lh,
                min_dist, _JOINT_PICK_THRESHOLD, len(self._joint_positions),
            )
        return result

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

        _lx, _ly, lw, lh = self._letterbox
        w = lw if lw > 0 else self.width()
        h = lh if lh > 0 else self.height()
        if w <= 0 or h <= 0:
            return None

        self.makeCurrent()

        if not self._ensure_pick_fbo(w, h):
            self.doneCurrent()
            return None

        # Bind FBO and clear (FBO uses its own viewport at origin)
        # Reset color mask — paintGL's QPainter path leaves alpha writes
        # disabled, which would prevent the FBO clear and joint rendering
        # from writing correct RGBA values for ID-color picking.
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
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

        # Restore default framebuffer + letterbox viewport
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        lx, ly, lw_lb, lh_lb = self._letterbox
        gl_y = self.height() - ly - lh_lb
        gl.glViewport(lx, gl_y, lw_lb, lh_lb)
        gl.glEnable(gl.GL_CULL_FACE)

        self.doneCurrent()

        # Decode pixel → joint index
        if pixel is not None and len(pixel) >= 3:
            r_val = int(pixel[0]) if not isinstance(pixel[0], int) else pixel[0]
            g_val = int(pixel[1]) if not isinstance(pixel[1], int) else pixel[1]
            b_val = int(pixel[2]) if not isinstance(pixel[2], int) else pixel[2]
            result = decode_joint_id(r_val, g_val, b_val)
            if result is not None and result < _N_BODY_JOINTS:
                logger.warning("fbo_pick: hit joint %d at px=(%d,%d) rgba=(%d,%d,%d,%d)",
                             result, px, py, r_val, g_val, b_val,
                             int(pixel[3]) if len(pixel) > 3 else 0)
                return result
            logger.warning("fbo_pick: miss at px=(%d,%d) rgba=(%d,%d,%d,%d) fbo=%dx%d",
                         px, py, r_val, g_val, b_val,
                         int(pixel[3]) if len(pixel) > 3 else 0, w, h)
        else:
            logger.warning("fbo_pick: glReadPixels returned %r", pixel)
        return None

    # ------------------------------------------------------------------
    # SMPL-X model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> bool:
        """Lazily load SMPL-X body model. Returns True on success."""
        if self._model_loaded:
            return self._body_model is not None
        self._model_loaded = True

        model_dir = None
        body_models_root = None
        if self._gvhmr_root:
            model_dir = (
                self._gvhmr_root
                / "inputs"
                / "checkpoints"
                / "body_models"
                / "smplx"
            )
            body_models_root = (
                self._gvhmr_root
                / "inputs"
                / "checkpoints"
                / "body_models"
            )

        try:
            import torch  # noqa: F401
            from hmr4d.utils.body_model.smplx_lite import SmplxLite

            self._body_model = SmplxLite(
                model_path=str(model_dir) if model_dir else None
            )
            self._body_model.cpu().eval()
            self._body_model_kind = "smplx_lite"
            logger.info("SMPL-X model path: %s", model_dir)
        except Exception as lite_error:
            logger.warning("SmplxLite unavailable, trying smplx fallback: %s", lite_error, exc_info=True)
            try:
                import smplx

                if body_models_root is None:
                    raise RuntimeError("GVHMR body model root is not configured")
                self._body_model = smplx.create(
                    str(body_models_root),
                    model_type="smplx",
                    gender="neutral",
                    use_pca=False,
                    flat_hand_mean=True,
                    ext="npz",
                )
                self._body_model.cpu().eval()
                self._body_model_kind = "smplx_pkg"
                logger.info("SMPL-X fallback model path: %s", body_models_root)
            except Exception as smplx_error:
                logger.warning("Could not load SMPL-X model fallback: %s", smplx_error, exc_info=True)
                self._set_mesh_status(f"Mesh: model load failed ({smplx_error})")
                return False

        self._faces = np.asarray(self._body_model.faces, dtype=np.int32)
        self._n_faces = len(self._faces)
        if hasattr(self._body_model, "lbs_weights"):
            try:
                self._lbs_weights = self._body_model.lbs_weights.detach().cpu().numpy()
                logger.info("LBS weights: %s", self._lbs_weights.shape)
            except Exception:
                logger.debug("Could not extract lbs_weights", exc_info=True)
        logger.info("SMPL-X model loaded via %s: %d faces", self._body_model_kind, self._n_faces)
        self._set_mesh_status(f"Mesh: model ready ({self._body_model_kind})")
        return True

    def _body_model_pelvis(self, betas, body_pose, global_orient):
        import torch

        if self._body_model is None:
            return None
        if hasattr(self._body_model, "get_skeleton"):
            return self._body_model.get_skeleton(betas)[:, 0]
        if self._body_model_kind == "smplx_pkg":
            zeros_go = torch.zeros_like(global_orient)
            zeros_bp = torch.zeros_like(body_pose)
            out = self._body_model(
                betas=betas,
                global_orient=zeros_go,
                body_pose=zeros_bp,
                transl=torch.zeros((betas.shape[0], 3), dtype=betas.dtype),
                left_hand_pose=torch.zeros((betas.shape[0], 45), dtype=betas.dtype),
                right_hand_pose=torch.zeros((betas.shape[0], 45), dtype=betas.dtype),
                return_verts=False,
            )
            return out.joints[:, 0]
        return None

    def _tensor_to_numpy(self, value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float32)

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

        # Skip cache when interpolation modifies this frame
        interp_active = False
        if self._interpolation_enabled and self._session is not None:
            ct = self._session.correction_tracks.get(person_id)
            if ct is not None and len(ct.corrections) >= 2:
                if ct.get_correction(frame_idx) is not None:
                    interp_active = True
                else:
                    prev_c, next_c = ct.get_surrounding(frame_idx)
                    if (prev_c is not None and next_c is not None
                            and prev_c is not next_c
                            and prev_c.frame_index < frame_idx < next_c.frame_index):
                        interp_active = True

        if self._session is None:
            self._set_mesh_status("Mesh: no session")
            return None

        track = self._session.person_tracks.get(person_id)
        if track is None:
            self._set_mesh_status("Mesh: no selected person")
            return None

        if track.body_model_type == "soma":
            # SOMA: skeleton-only rendering (no mesh model available yet)
            self._set_mesh_status("Mesh: unavailable for SOMA skeleton")
            return None

        if not override_active and not interp_active:
            with self._cache_lock:
                if cache_key in self._vertex_cache:
                    self._set_mesh_status("Mesh: cached")
                    return self._vertex_cache[cache_key]

        if not self._load_model():
            return None

        params, _source = self._params_for_track(track)
        if params is None:
            self._set_mesh_status("Mesh: no params for active source")
            return None

        # Use world-space orient/transl so the mesh matches
        # the world-grounded skeleton positions
        if ("global_orient_world" in params
                and "transl_world" in params
                and self._use_world_params()):
            params = self._world_params_for_person(params, person_id)

        # Apply interpolation between correction keyframes
        params = self._apply_interpolation_to_params(params, frame_idx)

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
                    logger.warning(
                        "Vertex params incomplete (pid=%d, source=%s): go=%s bp=%s betas=%s",
                        person_id,
                        _source,
                        go is not None,
                        bp is not None,
                        be is not None,
                    )
                    missing = []
                    if go is None:
                        missing.append("global_orient")
                    if bp is None:
                        missing.append("body_pose")
                    if be is None:
                        missing.append("betas")
                    self._set_mesh_status(
                        f"Mesh: missing {', '.join(missing)}"
                    )
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
                    self._set_mesh_status("Mesh: frame index outside pose data")
                    return None

                # SmplxLite expects flat (*, 63), not structured (*, 21, 3)
                bp_frame = bp_frame.reshape(bp_frame.shape[0], -1)

                # betas: take first row (shape is shared across frames)
                be_frame = be_t[:1] if be_t.ndim >= 2 else be_t.unsqueeze(0)

                tr_frame = None
                if tr_t is not None:
                    tr_frame = _frame_slice(tr_t, frame_idx)

                # SMPL-X internally places the pelvis at J[0] (shape-dependent,
                # ~35cm below mesh center for mean shape).  The skeleton FK puts
                # the root at transl directly.  Subtract J[0] from transl so the
                # mesh rotates around transl, matching the FK convention.
                if tr_frame is not None:
                    pelvis = self._body_model_pelvis(be_frame, bp_frame, go_frame)
                    if pelvis is not None:
                        tr_frame = tr_frame - pelvis

                # Hand poses are already normalized in the loader to match export behavior.
                # Export adds the hand mean once in the loader; pass explicit hand pose
                # tensors through the body model without introducing any extra viewport-
                # only hand offset math.
                lh = params.get("left_hand_pose")
                rh = params.get("right_hand_pose")
                lh_frame = None
                rh_frame = None
                if lh is not None:
                    lh_t = _to_tensor(lh)
                    lh_frame = _frame_slice(lh_t, frame_idx)
                    if lh_frame is not None:
                        lh_frame = lh_frame.reshape(lh_frame.shape[0], 45)
                if rh is not None:
                    rh_t = _to_tensor(rh)
                    rh_frame = _frame_slice(rh_t, frame_idx)
                    if rh_frame is not None:
                        rh_frame = rh_frame.reshape(rh_frame.shape[0], 45)
                if self._body_model_kind == "smplx_lite":
                    lh_mean = params.get("_lh_mean")
                    rh_mean = params.get("_rh_mean")
                    if lh_frame is not None and lh_mean is not None:
                        lh_mean_t = _to_tensor(lh_mean).reshape(1, 45)
                        lh_frame = lh_frame - lh_mean_t
                    if rh_frame is not None and rh_mean is not None:
                        rh_mean_t = _to_tensor(rh_mean).reshape(1, 45)
                        rh_frame = rh_frame - rh_mean_t

                lwo = params.get("left_wrist_orient")
                rwo = params.get("right_wrist_orient")
                if lwo is not None or rwo is not None:
                    bp_frame = bp_frame.clone()
                    if lwo is not None:
                        lwo_t = _to_tensor(lwo)
                        lwo_frame = _frame_slice(lwo_t, frame_idx)
                        if lwo_frame is not None:
                            bp_frame[0, 57:60] = lwo_frame.reshape(3)
                    if rwo is not None:
                        rwo_t = _to_tensor(rwo)
                        rwo_frame = _frame_slice(rwo_t, frame_idx)
                        if rwo_frame is not None:
                            bp_frame[0, 60:63] = rwo_frame.reshape(3)

                if self._body_model_kind == "smplx_pkg":
                    output = self._body_model(
                        betas=be_frame,
                        global_orient=go_frame,
                        body_pose=bp_frame,
                        left_hand_pose=lh_frame
                        if lh_frame is not None
                        else torch.zeros((be_frame.shape[0], 45), dtype=be_frame.dtype),
                        right_hand_pose=rh_frame
                        if rh_frame is not None
                        else torch.zeros((be_frame.shape[0], 45), dtype=be_frame.dtype),
                        transl=tr_frame,
                        return_verts=True,
                    )
                    verts = output.vertices
                else:
                    verts = self._body_model(
                        body_pose=bp_frame,
                        left_hand_pose=lh_frame,
                        right_hand_pose=rh_frame,
                        betas=be_frame,
                        global_orient=go_frame,
                        transl=tr_frame,
                    )  # (1, V, 3)

                vertices = self._tensor_to_numpy(verts[0]).astype(np.float32)
                vertices = self._incam_transform_points(vertices, person_id)
                normals = compute_normals(vertices, self._faces)
                self._set_mesh_status(f"Mesh: ok ({self._body_model_kind}, {_source})")

                result = (vertices, normals)

                # Only cache non-overridden and non-interpolated results
                if not override_active and not interp_active:
                    with self._cache_lock:
                        if len(self._vertex_cache) >= _CACHE_MAX:
                            oldest = next(iter(self._vertex_cache))
                            del self._vertex_cache[oldest]
                        self._vertex_cache[cache_key] = result

                return result
        except Exception as e:
            import traceback as _tb
            logger.warning("Vertex computation failed (pid=%d, f=%d): %s\n%s",
                           person_id, frame_idx, e, _tb.format_exc())
            self._set_mesh_status(f"Mesh: vertex compute failed ({e})")
            return None

    # ------------------------------------------------------------------
    # OpenGL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self):
        if not _HAS_GL:
            return
        try:
            # Reset draw state — old GL resources are invalid after context recreation
            self._gl_ready = False
            self._faces_uploaded = False
            self._n_indices = 0
            self._n_vertices = 0
            self._invalidate_cache()

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
            # Re-compute mesh so data is uploaded to the new GL buffers
            self._refresh_mesh()
        except Exception as e:
            logger.error("OpenGL init failed: %s", e)
            self._gl_ready = False
            self._status_msg = f"OpenGL init failed: {e}"
            self._setup_fallback()

    def resizeGL(self, w: int, h: int):
        if not _HAS_GL or not self._gl_ready:
            return
        self._recompute_letterbox()
        lx, ly, lw, lh = self._letterbox
        gl_y = h - ly - lh  # flip Y for GL bottom-left origin
        gl.glViewport(lx, gl_y, lw, lh)
        self._update_projection(lw, lh)

    def _paintGL_pure(self):
        """Pure-GL paint path for WSL2/Wayland (no QPainter).

        QPainter + beginNativePainting() on QOpenGLWidget crashes WSLg's
        Weston compositor.  This path does all GL rendering directly and
        skips 2D text overlays (joint labels, debug info).
        """
        # Clear full widget to black (bars), then letterbox to theme color.
        # glClear ignores glViewport — use glScissor to restrict the second clear.
        ww, wh = self.width(), self.height()
        gl.glViewport(0, 0, ww, wh)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        lx, ly, lw, lh = self._letterbox
        gl_y = wh - ly - lh
        gl.glViewport(lx, gl_y, lw, lh)
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(lx, gl_y, lw, lh)
        gl.glClearColor(0.1, 0.1, 0.12, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_SCISSOR_TEST)

        # Grid floor
        if self._show_grid:
            self._draw_grid()

        # Position anchor markers (after grid, before mesh/skeleton)
        if self._pos_anchor_markers:
            self._draw_pos_anchor_markers()

        # Camera frustum wireframe (orbit mode)
        if self._show_camera_frustum and self._camera_mode == "orbit":
            self._draw_camera_frustum()

        # Mesh triangles
        if (
            self._render_mode != RenderMode.WIREFRAME
            and self._vertices is not None
            and self._n_indices > 0
        ):
            gl.glDisable(gl.GL_CULL_FACE)
            self._shader.bind()
            self._set_mat4("model", self._model_mat)
            self._set_mat4("view", self._view)
            self._set_mat4("projection", self._projection)
            self._set_vec3("light_dir", _LIGHT_DIR)
            if self._render_mode == RenderMode.FAST:
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
            gl.glEnable(gl.GL_CULL_FACE)

        # Skeleton overlay
        if self._show_skeleton:
            if self._show_all_persons and self._all_joint_positions:
                for pid, joints in self._all_joint_positions.items():
                    if pid == self._person_id:
                        continue
                    skel = self._all_active_skels.get(pid, self._active_skel)
                    self._draw_skeleton_for(joints, skel, selected=False)
            if self._joint_positions is not None:
                self._draw_skeleton()

        # Force alpha to 1.0 over full widget (including bars) so Wayland
        # compositor doesn't treat viewport as transparent (click-through).
        gl.glViewport(0, 0, self.width(), self.height())
        gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_TRUE)
        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
        gl.glClearColor(0.1, 0.1, 0.12, 1.0)
        # Restore letterbox viewport
        gl.glViewport(lx, gl_y, lw, lh)

    def paintGL(self):
        if not _HAS_GL:
            return

        if not self._gl_ready:
            return

        # On WSL2/Wayland, QPainter + beginNativePainting() on QOpenGLWidget
        # crashes the Weston compositor (segfault).  Use pure GL rendering
        # on WSL2 and only use QPainter overlays on native Linux/X11.
        if _IS_WSL:
            self._paintGL_pure()
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
        lx, ly, lw, lh = self._letterbox
        # Black bars + letterboxed video background
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if _has_bg:
            fh, fw = self._video_frame.shape[:2]
            bpl = fw * 3  # bytes per line for RGB888
            qimg = QImage(
                self._video_frame.data, fw, fh, bpl,
                QImage.Format.Format_RGB888,
            )
            painter.drawImage(QRect(lx, ly, lw, lh), qimg)

        painter.beginNativePainting()

        # Set GL viewport to letterbox rect
        gl_y = self.height() - ly - lh
        gl.glViewport(lx, gl_y, lw, lh)

        # Clear with alpha=1.0 to ensure the surface is fully opaque,
        # then disable alpha writes so GL draw calls can't make it transparent.
        # Without this, the Wayland compositor on WSL2 treats GL-rendered
        # pixels as transparent → click-through to windows behind.
        # Use scissor to restrict clear to letterbox area (preserve black bars).
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
        gl.glEnable(gl.GL_SCISSOR_TEST)
        gl.glScissor(lx, gl_y, lw, lh)
        if _has_bg:
            gl.glClear(gl.GL_DEPTH_BUFFER_BIT)
        else:
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glDisable(gl.GL_SCISSOR_TEST)
        gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_FALSE)

        # Grid floor (drawn first so mesh occludes it)
        if self._show_grid:
            self._draw_grid()

        # Position anchor markers (after grid, before mesh/skeleton)
        if self._pos_anchor_markers:
            self._draw_pos_anchor_markers()

        # Camera frustum wireframe (orbit mode)
        if self._show_camera_frustum and self._camera_mode == "orbit":
            self._draw_camera_frustum()

        # Mesh triangles — skip entirely in wireframe mode
        if (
            self._render_mode != RenderMode.WIREFRAME
            and self._vertices is not None
            and self._n_indices > 0
        ):
            gl.glDisable(gl.GL_CULL_FACE)
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
            gl.glEnable(gl.GL_CULL_FACE)

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
            painter.drawText(lx + 10, ly + 20, (
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
                    lines.append(f"Motion source: {self._motion_source_label_text()}")
                    params, _source = self._params_for_track(track)
                    if params is not None and params.get("wrist_debug_source"):
                        lines.append(f"Wrist source: {params['wrist_debug_source']}")
            lines.append(f"Camera: {self._camera_mode}  |  GL: {'ready' if self._gl_ready else 'NOT READY'}")
            lines.append("Press V to toggle orbit/incam camera")
            y = ly + lh // 2 - len(lines) * 10
            for line in lines:
                painter.drawText(lx + 10, y, line)
                y += 20

        painter.end()

        # WSL2/Wayland fix: QPainter text rendering writes alpha < 1.0 for
        # antialiasing.  The Wayland compositor treats those pixels as
        # transparent, causing click-through to windows behind.
        # Fix: after QPainter is done, do one final GL pass that overwrites
        # the entire framebuffer alpha channel to 1.0 (fully opaque).
        if _IS_WSL and _HAS_GL:
            self.makeCurrent()
            # Full widget for alpha fix (including bars)
            gl.glViewport(0, 0, self.width(), self.height())
            gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_TRUE)
            gl.glClearColor(0, 0, 0, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
            # Restore clear color and letterbox viewport for next frame
            gl.glClearColor(0.1, 0.1, 0.12, 1.0)
            gl.glViewport(lx, self.height() - ly - lh, lw, lh)

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

        if self._camera_mode == "orbit":
            cx = self._orbit_center[0]
            cz = self._orbit_center[2]
        else:
            cx = self._grid_center_x
            cz = self._grid_center_z
        positions, colors = compute_grid_lines(
            y=self._grid_y,
            center_x=cx,
            center_z=cz,
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

    def set_position_anchor_markers(self, markers: list[np.ndarray]):
        """Set 3D positions for position anchor markers (purple crosshairs)."""
        self._pos_anchor_markers = list(markers)
        self.update()

    def _draw_pos_anchor_markers(self):
        """Draw purple XZ crosshairs at each position anchor's target location."""
        if not _HAS_GL or not self._gl_ready or not self._pos_anchor_markers:
            return

        arm = 0.15  # metres
        color = np.array([0.7, 0.3, 0.9], dtype=np.float32)

        # Build line vertices (4 per marker: ±X arm, ±Z arm)
        line_verts = []
        for pos in self._pos_anchor_markers:
            p = np.asarray(pos, dtype=np.float32)
            line_verts.append(p + np.array([-arm, 0, 0], dtype=np.float32))
            line_verts.append(p + np.array([arm, 0, 0], dtype=np.float32))
            line_verts.append(p + np.array([0, 0, -arm], dtype=np.float32))
            line_verts.append(p + np.array([0, 0, arm], dtype=np.float32))

        positions = np.array(line_verts, dtype=np.float32)
        colors = np.tile(color, (len(positions), 1))
        normals = np.zeros_like(positions)

        # Center dot vertices
        dot_positions = np.array(
            [np.asarray(p, dtype=np.float32) for p in self._pos_anchor_markers],
            dtype=np.float32,
        )
        dot_colors = np.tile(color, (len(dot_positions), 1))
        dot_normals = np.zeros_like(dot_positions)

        self._shader.bind()

        # Unlit (full ambient, zero light — same as grid)
        self._set_mat4("model", np.eye(4, dtype=np.float32))
        self._set_mat4("view", self._view)
        self._set_mat4("projection", self._projection)
        self._set_vec3("light_dir", _LIGHT_DIR)
        self._set_vec3("light_color", np.zeros(3, dtype=np.float32))
        self._set_vec3("ambient", np.ones(3, dtype=np.float32))

        gl.glDisable(gl.GL_CULL_FACE)
        self._draw_primitive(gl.GL_LINES, positions, normals, colors, line_width=2.0)
        self._draw_primitive(gl.GL_POINTS, dot_positions, dot_normals, dot_colors, point_size=6.0)
        gl.glEnable(gl.GL_CULL_FACE)

        self._shader.release()

    def _get_frustum_K(self) -> np.ndarray:
        """Intrinsics matrix for frustum rendering, respecting FOV override."""
        sess = self._session
        w = sess.img_width or 1920
        h = sess.img_height or 1080
        # Priority 1: user FOV override
        if self._frustum_fov_override is not None:
            fx = w / (2 * np.tan(np.radians(self._frustum_fov_override / 2)))
            return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float32)
        # Priority 2: pipeline-computed K
        if sess.camera_K is not None:
            return sess.camera_K.astype(np.float32)
        # Priority 3: derive from focal_mm (NOT max(w,h) fallback)
        f_mm = sess.focal_mm if sess.focal_mm else 24.0
        fx = f_mm / 36.0 * w
        return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float32)

    def _draw_camera_frustum(self):
        """Draw camera frustum wireframe and motion trail in orbit mode."""
        if not _HAS_GL or not self._gl_ready:
            return
        sess = self._session
        if sess is None or sess.derived_c2w is None:
            return

        c2w_all = sess.derived_c2w
        frame = min(self._current_frame, len(c2w_all) - 1)
        c2w = c2w_all[frame]

        K = self._get_frustum_K()
        img_w = sess.img_width or 1920
        img_h = sess.img_height or 1080

        # Frustum wireframe
        frust_pos, frust_col = compute_frustum_lines(c2w, K, img_w, img_h)
        # Trail
        trail_pos, trail_col = compute_camera_trail(c2w_all, frame)
        # Camera origin dot (in GL coords)
        origin_gl = c2w[:3, 3].reshape(1, 3).astype(np.float32)
        dot_col = np.array([[0.0, 1.0, 1.0]], dtype=np.float32)

        self._shader.bind()

        # Unlit (same pattern as grid)
        self._set_mat4("model", np.eye(4, dtype=np.float32))
        self._set_mat4("view", self._view)
        self._set_mat4("projection", self._projection)
        self._set_vec3("light_dir", _LIGHT_DIR)
        self._set_vec3("light_color", np.zeros(3, dtype=np.float32))
        self._set_vec3("ambient", np.ones(3, dtype=np.float32))

        gl.glDisable(gl.GL_CULL_FACE)

        # Frustum lines
        if len(frust_pos) > 0:
            normals = np.zeros_like(frust_pos)
            self._draw_primitive(gl.GL_LINES, frust_pos, normals, frust_col, line_width=2.0)

        # Trail lines
        if len(trail_pos) > 0:
            normals = np.zeros_like(trail_pos)
            self._draw_primitive(gl.GL_LINES, trail_pos, normals, trail_col, line_width=1.0)

        # Camera origin dot
        dot_normals = np.zeros_like(origin_gl)
        self._draw_primitive(gl.GL_POINTS, origin_gl, dot_normals, dot_col, point_size=6.0)

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

        lb_x, lb_y, lb_w, lb_h = self._letterbox
        w = lb_w if lb_w > 0 else 200
        h = lb_h if lb_h > 0 else 150
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
            # Offset from viewport-space to widget-space for QPainter
            sx += lb_x
            sy += lb_y

            text_w = fm.horizontalAdvance(name)
            text_h = fm.height()

            # Position label to the right and slightly above the joint point
            lx = int(sx + _LABEL_OFFSET_X)
            ly = int(sy + _LABEL_OFFSET_Y)

            # Clamp label box within letterbox area
            lx = max(lb_x + _LABEL_MARGIN, min(lx, lb_x + w - text_w - 2 * _LABEL_PADDING_X - _LABEL_MARGIN))
            ly = max(lb_y + _LABEL_MARGIN + text_h, min(ly, lb_y + h - _LABEL_MARGIN))

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
        # ---- Vertices (cached) ----
        if self._person_id < 0:
            self._set_mesh_status("Mesh: no person selected")
        elif self._render_mode == RenderMode.WIREFRAME:
            self._vertices = None
            self._normals = None
            self._n_indices = 0
            self._set_mesh_status("Mesh: hidden in Skeleton mode")
        else:
            verts_result = self._compute_vertices(self._person_id, self._current_frame)
            if verts_result is not None:
                self._vertices, self._normals = verts_result
                self._n_vertices = len(self._vertices)
                if self._gl_ready:
                    self._upload_buffers()
                if self._camera_mode == "orbit" and not self._orbit_auto_centered:
                    self._auto_center_orbit()
                    self._update_camera()
            else:
                self._vertices = None
                self._normals = None
                self._n_indices = 0

        # ---- Joints (lightweight FK — always computed) ----
        prev_joints = self._joint_positions
        self._joint_positions = self._compute_joints()
        if self._joint_positions is not None and prev_joints is None:
            logger.info(
                "Skeleton loaded: pid=%d, %d joints, camera=%s, gl_ready=%s",
                self._person_id, len(self._joint_positions),
                self._camera_mode, self._gl_ready,
            )

        # ---- Grid ----
        if self._joint_positions is not None and self._camera_mode != "orbit":
            pts = self._joint_positions
            if self._use_world_params():
                self._grid_y = float(np.min(pts[:, 1]))
            else:
                self._grid_y = float(np.max(pts[:, 1]))
            self._grid_center_x = float(pts[0, 0])
            self._grid_center_z = float(pts[0, 2])

        # ---- Multi-person joints ----
        # Always recompute FK for every person — it's pure numpy and cheap
        # enough to run per frame during playback.  (A previous optimization
        # skipped recompute during scrubbing, which froze non-selected
        # characters on playback.)
        self._all_joint_positions.clear()
        self._all_active_skels.clear()
        if self._show_all_persons and self._session is not None:
            for pid, track in self._session.person_tracks.items():
                skel = _SOMA_SKEL if track.body_model_type == "soma" else _SKEL
                params, _source = self._params_for_track(track)
                if params is None:
                    continue
                use_world = self._source_uses_world_payload(_source, params)
                if ("global_orient_world" in params
                        and "transl_world" in params
                        and use_world):
                    params = self._world_params_for_person(params, pid)
                try:
                    joints = forward_kinematics(params, self._current_frame)
                    if joints is not None:
                        joints = self._incam_transform_points(joints, pid)
                        self._all_joint_positions[pid] = joints
                        self._all_active_skels[pid] = skel
                except Exception:
                    pass

        # ---- Auto-center orbit on skeleton when no mesh ----
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
        self._emit_source_status()
        self._emit_mesh_status()

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
        if not self._faces_uploaded or self._n_indices <= 0:
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
            img_w = w
            img_h = h
            if self._session is not None:
                img_w = self._session.img_width or w
                img_h = self._session.img_height or h
            K = None
            if self._use_world_params():
                K = self._incam_effective_K()
            if K is None:
                K = estimate_K(img_w, img_h)
            K_vp = scale_K_to_viewport(K, img_w, img_h, w, h)
            self._projection = k_to_projection(K_vp, w, h)

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
