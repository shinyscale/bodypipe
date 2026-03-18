"""SMPL-X mesh viewport — QOpenGLWidget with Phong shading.

Renders the SMPL-X body mesh for the selected person at the current frame.
The body model is loaded once; vertices are recomputed per frame from the
session's SMPL-X parameters (torch forward pass → numpy), uploaded to
dynamic VBOs, and rendered with Phong shading matching shaders/mesh.vert
and shaders/mesh.frag.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal, Qt, QSize
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


# View matrix: flip Y and Z to convert from CV camera space to GL eye space.
_CV_TO_GL = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float32,
)

# Default skin-tone for mesh rendering (warm beige).
_SKIN_COLOR = np.array([0.82, 0.72, 0.63], dtype=np.float32)

# Lighting parameters.
_LIGHT_DIR = np.array([0.0, -0.5, -1.0], dtype=np.float32)
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)
_LIGHT_COLOR = np.array([0.8, 0.8, 0.8], dtype=np.float32)
_AMBIENT = np.array([0.35, 0.35, 0.35], dtype=np.float32)

# Maximum vertex cache size (per person+frame).
_CACHE_MAX = 200


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
        """Set camera mode ('incam' or 'orbit'). Stub for Phase 3.2."""

    def set_color_mode(self, mode: str):
        """Set vertex color mode ('solid', 'joint', 'confidence'). Stub for Phase 3.2."""

    def highlight_joint(self, joint_idx: int):
        """Highlight a joint in accent color. Stub for Phase 3.3."""

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
        if cache_key in self._vertex_cache:
            return self._vertex_cache[cache_key]

        if not self._load_model():
            return None
        if self._session is None:
            return None

        track = self._session.person_tracks.get(person_id)
        if track is None or track.smplx_params is None:
            return None

        params = track.smplx_params

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

                # Cache (evict oldest when full)
                if len(self._vertex_cache) >= _CACHE_MAX:
                    oldest = next(iter(self._vertex_cache))
                    del self._vertex_cache[oldest]
                result = (vertices, normals)
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

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if self._vertices is None or self._n_indices == 0:
            return

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_mesh(self):
        """Recompute vertices for current person/frame and trigger repaint."""
        result = self._compute_vertices(self._person_id, self._current_frame)
        if result is not None:
            self._vertices, self._normals = result
            self._n_vertices = len(self._vertices)
            if self._gl_ready:
                self._upload_buffers()
        else:
            self._vertices = None
            self._normals = None
            self._n_indices = 0
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

        # Color VBO (location 2) — solid skin tone for now
        colors = np.tile(_SKIN_COLOR, (self._n_vertices, 1)).astype(
            np.float32
        )
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
        """Recompute projection from session camera K or an estimate."""
        w = max(width, 1)
        h = max(height, 1)
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
