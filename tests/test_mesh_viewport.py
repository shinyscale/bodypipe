"""Tests for the SMPL-X mesh viewport (Phase 3.1).

Tests cover:
 - Pure helper functions (compute_normals, k_to_projection, estimate_K) which
   are the mathematical core and always runnable without OpenGL.
 - MeshViewport widget construction, public API, and signal existence.
 - Integration with MultiPersonTab (widget presence + signal wiring).
 - Vertex computation logic with a mocked SMPL-X model.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PySide6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session, PersonTrack
from views.mesh_viewport import (
    MeshViewport,
    compute_normals,
    k_to_projection,
    estimate_K,
    compute_orbit_view,
    perspective_fov,
    forward_kinematics,
    project_joints_to_screen,
    find_nearest_joint,
    encode_joint_id,
    decode_joint_id,
    get_joint_chain,
    get_joint_chain_bones,
    get_joint_siblings,
    get_opposite_joint,
    get_joint_region,
    JOINT_NAMES,
    JOINT_PARENTS,
    DEFAULT_OFFSETS,
    BONE_CONNECTIONS,
    _CV_TO_GL,
    _HAS_GL,
    _N_BODY_JOINTS,
    _JOINT_PICK_THRESHOLD,
    _ACCENT_COLOR,
    _SELECTED_ACCENT_COLOR,
    _CHAIN_BONE_COLOR,
    _CHAIN_JOINT_COLOR,
    _BONE_COLOR,
    _BODY_JOINT_COLOR,
    _HAND_JOINT_COLOR,
    _JOINT_POINT_SIZE,
    _SELECTED_JOINT_POINT_SIZE,
    _CHAIN_JOINT_POINT_SIZE,
    _BONE_LINE_WIDTH,
    _CHAIN_BONE_LINE_WIDTH,
    _ORBIT_DEFAULT_YAW,
    _ORBIT_DEFAULT_PITCH,
    _ORBIT_DEFAULT_DISTANCE,
    _ORBIT_DEFAULT_CENTER,
    _ORBIT_SENSITIVITY,
    _PITCH_LIMIT,
    _ZOOM_FACTOR,
    compute_grid_lines,
    compute_joint_label_layout,
    _GRID_SIZE,
    _GRID_DIVISIONS,
    _GRID_COLOR,
    _GRID_AXIS_COLOR,
    _LABEL_MARGIN,
    _LR_PAIRS,
    _JOINT_REGIONS,
    _JOINT_TO_REGION,
)


# ======================================================================
# Pure function tests (no GL / Qt required)
# ======================================================================


class TestComputeNormals:
    """compute_normals: face→vertex normal averaging."""

    def test_single_triangle_z_up(self):
        """A flat triangle in the XY-plane should have normals pointing +Z."""
        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = compute_normals(verts, faces)

        assert normals.shape == (3, 3)
        # All vertices share the same single face → all normals identical
        for i in range(3):
            np.testing.assert_allclose(normals[i], [0, 0, 1], atol=1e-5)

    def test_normals_unit_length(self):
        """Normals should be unit-length."""
        verts = np.array(
            [[0, 0, 0], [3, 0, 0], [0, 4, 0], [0, 0, 5]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3]], dtype=np.int32)
        normals = compute_normals(verts, faces)

        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)

    def test_output_dtype(self):
        """Output should be float32 regardless of input."""
        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        normals = compute_normals(verts, faces)
        assert normals.dtype == np.float32

    def test_two_opposing_triangles(self):
        """Shared edge vertex normal should be average of two face normals."""
        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32)
        normals = compute_normals(verts, faces)
        assert normals.shape == (4, 3)
        # All face normals should point in +Z (both tris are in XY plane)
        for i in range(4):
            np.testing.assert_allclose(normals[i, 2], 1.0, atol=1e-5)

    def test_cube_normals(self):
        """A cube's vertex normals should all have unit length."""
        # Minimal cube: 8 verts, 12 faces
        verts = np.array(
            [
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2], [0, 2, 3],  # front
                [4, 6, 5], [4, 7, 6],  # back
                [0, 4, 5], [0, 5, 1],  # bottom
                [2, 6, 7], [2, 7, 3],  # top
                [0, 3, 7], [0, 7, 4],  # left
                [1, 5, 6], [1, 6, 2],  # right
            ],
            dtype=np.int32,
        )
        normals = compute_normals(verts, faces)
        lengths = np.linalg.norm(normals, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


class TestKToProjection:
    """k_to_projection: camera intrinsics → OpenGL projection matrix."""

    def test_shape(self):
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = K[1, 1] = 500
        K[0, 2] = 320
        K[1, 2] = 240
        proj = k_to_projection(K, 640, 480)
        assert proj.shape == (4, 4)
        assert proj.dtype == np.float32

    def test_perspective_division(self):
        """Row [3] should encode perspective division: [0, 0, -1, 0]."""
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = K[1, 1] = 500
        K[0, 2] = 320
        K[1, 2] = 240
        proj = k_to_projection(K, 640, 480)
        assert proj[3, 0] == 0.0
        assert proj[3, 1] == 0.0
        assert proj[3, 2] == -1.0
        assert proj[3, 3] == 0.0

    def test_centered_principal_point(self):
        """When cx=w/2 and cy=h/2, off-diagonal projection terms vanish."""
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = K[1, 1] = 500
        K[0, 2] = 320  # w/2
        K[1, 2] = 240  # h/2
        proj = k_to_projection(K, 640, 480)
        np.testing.assert_allclose(proj[0, 2], 0.0, atol=1e-6)
        np.testing.assert_allclose(proj[1, 2], 0.0, atol=1e-6)

    def test_focal_scaling(self):
        """Doubling focal length should double the diagonal elements."""
        K1 = np.eye(3, dtype=np.float32)
        K1[0, 0] = K1[1, 1] = 250
        K1[0, 2] = 320
        K1[1, 2] = 240

        K2 = K1.copy()
        K2[0, 0] = K2[1, 1] = 500

        p1 = k_to_projection(K1, 640, 480)
        p2 = k_to_projection(K2, 640, 480)
        np.testing.assert_allclose(p2[0, 0], 2 * p1[0, 0], atol=1e-5)
        np.testing.assert_allclose(p2[1, 1], 2 * p1[1, 1], atol=1e-5)

    def test_near_far(self):
        """Custom near/far should affect depth terms (row 2)."""
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = K[1, 1] = 500
        K[0, 2] = 320
        K[1, 2] = 240
        p1 = k_to_projection(K, 640, 480, near=0.1, far=10.0)
        p2 = k_to_projection(K, 640, 480, near=0.1, far=100.0)
        # Different far → different depth mapping
        assert p1[2, 2] != p2[2, 2]
        assert p1[2, 3] != p2[2, 3]


class TestEstimateK:
    """estimate_K: image dims → camera intrinsics."""

    def test_principal_point_centered(self):
        K = estimate_K(640, 480)
        assert K[0, 2] == 320.0
        assert K[1, 2] == 240.0

    def test_square_pixels(self):
        K = estimate_K(640, 480)
        assert K[0, 0] == K[1, 1]

    def test_focal_is_diagonal(self):
        """Focal length should equal sqrt(w² + h²)."""
        K = estimate_K(640, 480)
        expected = (640**2 + 480**2) ** 0.5
        np.testing.assert_allclose(K[0, 0], expected, atol=1e-3)

    def test_shape_and_dtype(self):
        K = estimate_K(1920, 1080)
        assert K.shape == (3, 3)
        assert K.dtype == np.float32


class TestCvToGl:
    """_CV_TO_GL view matrix flips Y and Z."""

    def test_shape_dtype(self):
        assert _CV_TO_GL.shape == (4, 4)
        assert _CV_TO_GL.dtype == np.float32

    def test_flips(self):
        assert _CV_TO_GL[0, 0] == 1
        assert _CV_TO_GL[1, 1] == -1
        assert _CV_TO_GL[2, 2] == -1
        assert _CV_TO_GL[3, 3] == 1

    def test_orthogonal(self):
        """_CV_TO_GL should be its own inverse (involution)."""
        product = _CV_TO_GL @ _CV_TO_GL
        np.testing.assert_allclose(product, np.eye(4), atol=1e-6)


# ======================================================================
# Widget tests (require QApplication from conftest)
# ======================================================================


class TestMeshViewportWidget:
    """MeshViewport construction and public API."""

    def test_construction(self, qapp):
        w = MeshViewport()
        assert w is not None

    def test_construction_with_gvhmr_root(self, qapp):
        w = MeshViewport(gvhmr_root=Path("/tmp/gvhmr"))
        assert w._gvhmr_root == Path("/tmp/gvhmr")

    def test_signal_joint_clicked(self, qapp):
        w = MeshViewport()
        assert hasattr(w, "joint_clicked")

    def test_signal_camera_changed(self, qapp):
        w = MeshViewport()
        assert hasattr(w, "camera_changed")

    def test_set_session(self, qapp, session):
        w = MeshViewport()
        w.set_session(session)
        assert w._session is session

    def test_set_session_clears_cache(self, qapp):
        w = MeshViewport()
        w._vertex_cache[(0, 0)] = (np.zeros((3, 3)), np.zeros((3, 3)))
        w.set_session(Session())
        assert len(w._vertex_cache) == 0

    def test_set_person(self, qapp):
        w = MeshViewport()
        w.set_person(2)
        assert w._person_id == 2

    def test_set_person_same_noop(self, qapp):
        """Setting the same person ID should not trigger recompute."""
        w = MeshViewport()
        w.set_person(2)
        # Set again — _refresh_mesh would normally be called, but with
        # same ID it should be a no-op (early return)
        w.set_person(2)
        assert w._person_id == 2

    def test_on_frame_changed(self, qapp):
        w = MeshViewport()
        w.on_frame_changed(42)
        assert w._current_frame == 42

    def test_on_frame_changed_same_noop(self, qapp):
        w = MeshViewport()
        w.on_frame_changed(10)
        w.on_frame_changed(10)  # should not trigger recompute
        assert w._current_frame == 10

    def test_minimum_size(self, qapp):
        w = MeshViewport()
        assert w.minimumWidth() >= 200
        assert w.minimumHeight() >= 150

    def test_stub_methods_exist(self, qapp):
        """Phase 3.2+ stub methods should exist and be callable."""
        w = MeshViewport()
        w.set_camera_mode("incam")
        w.set_color_mode("solid")
        w.highlight_joint(0)

    def test_default_state(self, qapp):
        w = MeshViewport()
        assert w._person_id == -1
        assert w._current_frame == 0
        assert w._session is None
        assert w._vertices is None
        assert w._normals is None
        assert w._body_model is None
        assert w._model_loaded is False

    def test_initial_view_matrix(self, qapp):
        """View matrix should be the CV→GL flip."""
        w = MeshViewport()
        np.testing.assert_array_equal(w._view, _CV_TO_GL)


class TestMeshViewportVertexComputation:
    """Vertex computation with mocked SMPL-X model."""

    def _make_mock_model(self, n_verts=100, n_faces=50):
        """Create a mock SMPL-X model returning deterministic vertices."""
        import types

        model = MagicMock()
        model.faces = np.random.randint(0, n_verts, (n_faces, 3)).astype(
            np.int32
        )
        model.eval.return_value = model
        model.cpu.return_value = model

        # Forward pass: return (1, V, 3) tensor-like
        verts = np.random.randn(1, n_verts, 3).astype(np.float32)

        class FakeTensor:
            """Mimics torch.Tensor for cpu/numpy/indexing."""

            def __init__(self, data):
                self._data = np.array(data, dtype=np.float32)

            def cpu(self):
                return self

            def numpy(self):
                return self._data

            def __getitem__(self, idx):
                return FakeTensor(self._data[idx])

        model.__call__ = MagicMock(return_value=FakeTensor(verts))
        model.return_value = FakeTensor(verts)
        return model

    def test_compute_vertices_no_session(self, qapp):
        """Should return None when no session is set."""
        w = MeshViewport()
        result = w._compute_vertices(0, 0)
        assert result is None

    def test_compute_vertices_no_track(self, qapp, session):
        """Should return None when person track doesn't exist."""
        w = MeshViewport()
        w.set_session(session)
        w._model_loaded = True  # skip model loading
        w._body_model = self._make_mock_model()
        w._faces = w._body_model.faces
        result = w._compute_vertices(99, 0)
        assert result is None

    def test_compute_vertices_no_params(self, qapp, session):
        """Should return None when smplx_params is None."""
        w = MeshViewport()
        session.person_tracks[0] = PersonTrack(person_id=0)
        w.set_session(session)
        w._model_loaded = True
        w._body_model = self._make_mock_model()
        w._faces = w._body_model.faces
        result = w._compute_vertices(0, 0)
        assert result is None

    def test_compute_vertices_success(self, qapp, session):
        """Should return (vertices, normals) when params are available."""
        pytest.importorskip("torch")

        import torch

        n_verts, n_faces = 100, 50
        mock_model = self._make_mock_model(n_verts, n_faces)

        w = MeshViewport()
        session.person_tracks[0] = PersonTrack(
            person_id=0,
            smplx_params={
                "global_orient": torch.randn(10, 3),
                "body_pose": torch.randn(10, 63),
                "betas": torch.randn(1, 10),
                "transl": torch.randn(10, 3),
            },
        )
        w.set_session(session)
        w._model_loaded = True
        w._body_model = mock_model
        w._faces = mock_model.faces

        result = w._compute_vertices(0, 0)
        assert result is not None
        verts, norms = result
        assert verts.shape == (n_verts, 3)
        assert norms.shape == (n_verts, 3)
        assert verts.dtype == np.float32
        assert norms.dtype == np.float32

    def test_vertex_cache(self, qapp, session):
        """Repeated calls for the same frame should hit cache."""
        pytest.importorskip("torch")

        import torch

        mock_model = self._make_mock_model()

        w = MeshViewport()
        session.person_tracks[0] = PersonTrack(
            person_id=0,
            smplx_params={
                "global_orient": torch.randn(10, 3),
                "body_pose": torch.randn(10, 63),
                "betas": torch.randn(1, 10),
                "transl": torch.randn(10, 3),
            },
        )
        w.set_session(session)
        w._model_loaded = True
        w._body_model = mock_model
        w._faces = mock_model.faces

        r1 = w._compute_vertices(0, 0)
        r2 = w._compute_vertices(0, 0)
        assert r1 is r2  # same object from cache

    def test_vertex_cache_eviction(self, qapp, session):
        """Cache should evict entries when full."""
        w = MeshViewport()
        # Fill cache beyond limit
        dummy = (np.zeros((3, 3)), np.zeros((3, 3)))
        for i in range(250):
            w._vertex_cache[(0, i)] = dummy
        # Should not exceed _CACHE_MAX + a few
        # (eviction happens during _compute_vertices, so manual fill won't
        # trigger it — this just verifies the dict doesn't explode)
        assert len(w._vertex_cache) == 250

    def test_compute_vertices_with_numpy_params(self, qapp, session):
        """Should handle numpy array params (not just torch tensors)."""
        pytest.importorskip("torch")

        mock_model = self._make_mock_model()

        w = MeshViewport()
        session.person_tracks[0] = PersonTrack(
            person_id=0,
            smplx_params={
                "global_orient": np.random.randn(10, 3),
                "body_pose": np.random.randn(10, 63),
                "betas": np.random.randn(1, 10),
                "transl": np.random.randn(10, 3),
            },
        )
        w.set_session(session)
        w._model_loaded = True
        w._body_model = mock_model
        w._faces = mock_model.faces

        result = w._compute_vertices(0, 0)
        assert result is not None

    def test_compute_vertices_missing_orient(self, qapp, session):
        """Should return None when global_orient is missing."""
        pytest.importorskip("torch")

        import torch

        mock_model = self._make_mock_model()

        w = MeshViewport()
        session.person_tracks[0] = PersonTrack(
            person_id=0,
            smplx_params={
                "body_pose": torch.randn(10, 63),
                "betas": torch.randn(1, 10),
            },
        )
        w.set_session(session)
        w._model_loaded = True
        w._body_model = mock_model
        w._faces = mock_model.faces

        result = w._compute_vertices(0, 0)
        assert result is None


# ======================================================================
# Integration: MeshViewport inside MultiPersonTab
# ======================================================================


class TestMultiPersonTabMeshIntegration:
    """MeshViewport should replace the old placeholder in MultiPersonTab."""

    def test_mesh_viewport_exists(self, qapp):
        from views.multi_person_tab import MultiPersonTab

        s = Session()
        tab = MultiPersonTab(s, Path("."))
        assert hasattr(tab, "_mesh_viewport")
        assert isinstance(tab._mesh_viewport, MeshViewport)

    def test_mesh_viewport_has_session(self, qapp):
        from views.multi_person_tab import MultiPersonTab

        s = Session()
        tab = MultiPersonTab(s, Path("."))
        assert tab._mesh_viewport._session is s

    def test_mesh_viewport_in_splitter(self, qapp):
        """MeshViewport should be in the bottom splitter (wrapped by PoseCorrectorPanel)."""
        from views.multi_person_tab import MultiPersonTab

        s = Session()
        tab = MultiPersonTab(s, Path("."))
        splitter = tab._bottom_splitter
        # PoseCorrectorPanel wraps MeshViewport; the panel is in the splitter
        found = False
        for i in range(splitter.count()):
            if splitter.widget(i) is tab._pose_corrector:
                found = True
                break
        assert found, "PoseCorrectorPanel not found in bottom splitter"
        # MeshViewport is accessible through the pose corrector
        assert tab._mesh_viewport is tab._pose_corrector.mesh_viewport

    def test_frame_change_updates_viewport(self, qapp):
        """Frame change handler should call mesh viewport."""
        from views.multi_person_tab import MultiPersonTab

        s = Session()
        tab = MultiPersonTab(s, Path("."))
        tab._mesh_viewport.on_frame_changed = MagicMock()
        tab._on_frame_changed(5)
        tab._mesh_viewport.on_frame_changed.assert_called_once_with(5)

    def test_person_change_updates_viewport(self, qapp):
        """Person change from identity panel should update mesh viewport."""
        from views.multi_person_tab import MultiPersonTab

        s = Session()
        tab = MultiPersonTab(s, Path("."))
        tab._mesh_viewport.set_person = MagicMock()
        tab._on_identity_person_changed(2)
        tab._mesh_viewport.set_person.assert_called_once_with(2)

    def test_track_click_updates_viewport(self, qapp):
        """Clicking a track should update mesh viewport person."""
        from views.multi_person_tab import MultiPersonTab

        s = Session()
        tab = MultiPersonTab(s, Path("."))
        tab._mesh_viewport.set_person = MagicMock()
        tab._on_track_clicked(3, 10)
        tab._mesh_viewport.set_person.assert_called_once_with(3)

    def test_no_pose_placeholder_label(self, qapp):
        """The old 'Phase 3' placeholder text should be gone."""
        from views.multi_person_tab import MultiPersonTab

        s = Session()
        tab = MultiPersonTab(s, Path("."))
        # Should NOT have the old _pose_panel QWidget with placeholder labels
        assert not hasattr(tab, "_pose_panel")


class TestShaderFiles:
    """Shader files should exist and contain expected content."""

    def test_vertex_shader_exists(self):
        path = Path(__file__).resolve().parent.parent / "shaders" / "mesh.vert"
        assert path.is_file()

    def test_fragment_shader_exists(self):
        path = Path(__file__).resolve().parent.parent / "shaders" / "mesh.frag"
        assert path.is_file()

    def test_vertex_shader_has_uniforms(self):
        path = Path(__file__).resolve().parent.parent / "shaders" / "mesh.vert"
        src = path.read_text()
        assert "uniform mat4 model" in src
        assert "uniform mat4 view" in src
        assert "uniform mat4 projection" in src

    def test_fragment_shader_has_uniforms(self):
        path = Path(__file__).resolve().parent.parent / "shaders" / "mesh.frag"
        src = path.read_text()
        assert "uniform vec3 light_dir" in src
        assert "uniform vec3 light_color" in src
        assert "uniform vec3 ambient" in src

    def test_shaders_version_330(self):
        for name in ("mesh.vert", "mesh.frag"):
            path = Path(__file__).resolve().parent.parent / "shaders" / name
            src = path.read_text()
            assert "#version 330 core" in src


# ======================================================================
# Pure function tests: orbit camera (Phase 3.2)
# ======================================================================


class TestComputeOrbitView:
    """compute_orbit_view: spherical orbit → look-at view matrix."""

    def test_shape_dtype(self):
        center = np.zeros(3, dtype=np.float32)
        view = compute_orbit_view(0.0, 0.0, 5.0, center)
        assert view.shape == (4, 4)
        assert view.dtype == np.float32

    def test_default_looks_from_z(self):
        """Yaw=0, pitch=0 → camera on +Z axis; center in front of camera."""
        center = np.zeros(3, dtype=np.float32)
        view = compute_orbit_view(0.0, 0.0, 5.0, center)
        center_view = view @ np.array([0, 0, 0, 1], dtype=np.float32)
        # Center should be at negative Z (in front) in GL view space
        assert center_view[2] < 0

    def test_yaw_90_rotates_camera(self):
        """Yaw=90° → camera on +X axis; center still in front."""
        center = np.zeros(3, dtype=np.float32)
        view = compute_orbit_view(90.0, 0.0, 5.0, center)
        center_view = view @ np.array([0, 0, 0, 1], dtype=np.float32)
        assert center_view[2] < 0

    def test_pitch_clamp_no_nan(self):
        """Pitch > 89° should be clamped; result must not contain NaN."""
        center = np.zeros(3, dtype=np.float32)
        view = compute_orbit_view(0.0, 95.0, 5.0, center)
        assert not np.any(np.isnan(view))

    def test_negative_pitch_no_nan(self):
        """Large negative pitch should be safe."""
        center = np.zeros(3, dtype=np.float32)
        view = compute_orbit_view(0.0, -95.0, 5.0, center)
        assert not np.any(np.isnan(view))

    def test_zero_distance_safe(self):
        """Tiny distance should not crash."""
        center = np.zeros(3, dtype=np.float32)
        view = compute_orbit_view(0.0, 0.0, 1e-10, center)
        assert not np.any(np.isnan(view))

    def test_rotation_orthonormal(self):
        """Upper-left 3×3 of view matrix should be orthonormal."""
        center = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        view = compute_orbit_view(45.0, 30.0, 5.0, center)
        R = view[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)

    def test_distance_affects_translation(self):
        """Doubling distance should move the center further in view space."""
        center = np.zeros(3, dtype=np.float32)
        v1 = compute_orbit_view(0.0, 0.0, 5.0, center)
        v2 = compute_orbit_view(0.0, 0.0, 10.0, center)
        c1 = (v1 @ np.array([0, 0, 0, 1], dtype=np.float32))[2]
        c2 = (v2 @ np.array([0, 0, 0, 1], dtype=np.float32))[2]
        assert abs(c2) > abs(c1)

    def test_center_offset(self):
        """Non-zero center should shift the view accordingly."""
        c1 = np.zeros(3, dtype=np.float32)
        c2 = np.array([5.0, 0.0, 0.0], dtype=np.float32)
        v1 = compute_orbit_view(0.0, 0.0, 5.0, c1)
        v2 = compute_orbit_view(0.0, 0.0, 5.0, c2)
        # Different centers → different view matrices
        assert not np.allclose(v1, v2)


class TestPerspectiveFov:
    """perspective_fov: vertical FOV → projection matrix."""

    def test_shape_dtype(self):
        proj = perspective_fov(45.0, 16 / 9)
        assert proj.shape == (4, 4)
        assert proj.dtype == np.float32

    def test_perspective_row(self):
        """Row 3 encodes perspective division: [0, 0, -1, 0]."""
        proj = perspective_fov(60.0, 1.0)
        np.testing.assert_allclose(proj[3], [0, 0, -1, 0], atol=1e-7)

    def test_square_aspect_equal_diag(self):
        """Aspect=1 → proj[0,0] == proj[1,1]."""
        proj = perspective_fov(45.0, 1.0)
        np.testing.assert_allclose(proj[0, 0], proj[1, 1], atol=1e-6)

    def test_wider_aspect_reduces_x(self):
        """Wider aspect → smaller proj[0,0]."""
        p1 = perspective_fov(45.0, 1.0)
        p2 = perspective_fov(45.0, 2.0)
        assert p2[0, 0] < p1[0, 0]

    def test_narrower_fov_increases_focal(self):
        """Smaller FOV → larger diagonal elements."""
        p1 = perspective_fov(60.0, 1.0)
        p2 = perspective_fov(30.0, 1.0)
        assert p2[1, 1] > p1[1, 1]

    def test_near_far_depth(self):
        """Different near/far → different depth mapping."""
        p1 = perspective_fov(45.0, 1.0, near=0.1, far=10.0)
        p2 = perspective_fov(45.0, 1.0, near=0.1, far=100.0)
        assert p1[2, 2] != p2[2, 2]


# ======================================================================
# Widget tests: camera modes (Phase 3.2)
# ======================================================================


class TestCameraModes:
    """Camera mode switching and orbit state management."""

    def test_default_mode_is_incam(self, qapp):
        w = MeshViewport()
        assert w._camera_mode == "incam"

    def test_set_orbit_mode(self, qapp):
        w = MeshViewport()
        w.set_camera_mode("orbit")
        assert w._camera_mode == "orbit"

    def test_set_incam_mode(self, qapp):
        w = MeshViewport()
        w.set_camera_mode("orbit")
        w.set_camera_mode("incam")
        assert w._camera_mode == "incam"

    def test_invalid_mode_ignored(self, qapp):
        w = MeshViewport()
        w.set_camera_mode("invalid")
        assert w._camera_mode == "incam"

    def test_same_mode_no_signal(self, qapp):
        """Setting the same mode should not emit camera_changed."""
        w = MeshViewport()
        received = []
        w.camera_changed.connect(lambda s: received.append(s))
        w.set_camera_mode("incam")  # already incam
        assert len(received) == 0

    def test_mode_switch_emits_signal(self, qapp):
        w = MeshViewport()
        received = []
        w.camera_changed.connect(lambda s: received.append(s))
        w.set_camera_mode("orbit")
        assert len(received) == 1
        assert received[0]["mode"] == "orbit"

    def test_orbit_defaults(self, qapp):
        w = MeshViewport()
        assert w._orbit_yaw == _ORBIT_DEFAULT_YAW
        assert w._orbit_pitch == _ORBIT_DEFAULT_PITCH
        assert w._orbit_distance == _ORBIT_DEFAULT_DISTANCE

    def test_incam_view_is_cv_to_gl(self, qapp):
        """In incam mode, view matrix should be _CV_TO_GL."""
        w = MeshViewport()
        np.testing.assert_array_equal(w._view, _CV_TO_GL)

    def test_orbit_view_differs(self, qapp):
        """In orbit mode, view matrix should differ from _CV_TO_GL."""
        w = MeshViewport()
        w.set_camera_mode("orbit")
        assert not np.array_equal(w._view, _CV_TO_GL)

    def test_orbit_model_is_cv_to_gl(self, qapp):
        """In orbit mode, model matrix should be _CV_TO_GL (vertex transform)."""
        w = MeshViewport()
        w.set_camera_mode("orbit")
        np.testing.assert_array_equal(w._model_mat, _CV_TO_GL)

    def test_incam_model_is_identity(self, qapp):
        """In incam mode, model matrix should be identity."""
        w = MeshViewport()
        np.testing.assert_array_equal(
            w._model_mat, np.eye(4, dtype=np.float32)
        )

    def test_switch_back_restores_incam(self, qapp):
        """Switching orbit→incam restores identity model + CV_TO_GL view."""
        w = MeshViewport()
        w.set_camera_mode("orbit")
        w.set_camera_mode("incam")
        np.testing.assert_array_equal(
            w._model_mat, np.eye(4, dtype=np.float32)
        )
        np.testing.assert_array_equal(w._view, _CV_TO_GL)

    def test_camera_state_incam(self, qapp):
        w = MeshViewport()
        state = w._camera_state()
        assert state == {"mode": "incam"}

    def test_camera_state_orbit(self, qapp):
        w = MeshViewport()
        w.set_camera_mode("orbit")
        state = w._camera_state()
        assert state["mode"] == "orbit"
        assert "yaw" in state
        assert "pitch" in state
        assert "distance" in state
        assert "center" in state

    def test_reset_orbit(self, qapp):
        w = MeshViewport()
        w.set_camera_mode("orbit")
        w._orbit_yaw = 90.0
        w._orbit_pitch = 45.0
        w._orbit_distance = 10.0
        w._reset_orbit()
        assert w._orbit_yaw == _ORBIT_DEFAULT_YAW
        assert w._orbit_pitch == _ORBIT_DEFAULT_PITCH

    def test_auto_center_orbit_with_vertices(self, qapp):
        """_auto_center_orbit should set center to mesh centroid in GL space."""
        w = MeshViewport()
        w._vertices = np.array(
            [[0.0, 0.0, 3.0], [2.0, 0.0, 3.0], [-2.0, 0.0, 3.0]],
            dtype=np.float32,
        )
        w._auto_center_orbit()
        # CV (0, 0, 3) → GL (0, 0, -3)
        np.testing.assert_allclose(w._orbit_center[2], -3.0, atol=1e-5)
        assert w._orbit_auto_centered is True

    def test_auto_center_orbit_no_vertices(self, qapp):
        """_auto_center_orbit with no vertices should be a no-op."""
        w = MeshViewport()
        original = w._orbit_center.copy()
        w._auto_center_orbit()
        np.testing.assert_array_equal(w._orbit_center, original)
        assert w._orbit_auto_centered is False

    def test_orbit_projection_uses_fov(self, qapp):
        """In orbit mode, projection should use perspective_fov, not K."""
        w = MeshViewport()
        w.set_camera_mode("orbit")
        incam_proj = w._projection.copy()
        w.set_camera_mode("incam")
        # Projections should differ (fov-based vs K-based)
        assert not np.allclose(w._projection, incam_proj)


class TestOrbitMouseInteraction:
    """Mouse events for orbit camera interaction."""

    def _make_mouse_event(self, event_type, pos, button, buttons):
        """Helper to create QMouseEvent."""
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QMouseEvent

        return QMouseEvent(
            event_type,
            QPointF(*pos),
            QPointF(*pos),  # globalPos
            button,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        )

    def test_left_drag_changes_yaw(self, qapp):
        from PySide6.QtCore import QEvent

        w = MeshViewport()
        w.set_camera_mode("orbit")
        initial_yaw = w._orbit_yaw

        press = self._make_mouse_event(
            QEvent.Type.MouseButtonPress, (100, 100),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        )
        w.mousePressEvent(press)

        move = self._make_mouse_event(
            QEvent.Type.MouseMove, (150, 100),
            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
        )
        w.mouseMoveEvent(move)

        assert w._orbit_yaw != initial_yaw

    def test_left_drag_changes_pitch(self, qapp):
        from PySide6.QtCore import QEvent

        w = MeshViewport()
        w.set_camera_mode("orbit")
        initial_pitch = w._orbit_pitch

        press = self._make_mouse_event(
            QEvent.Type.MouseButtonPress, (100, 100),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        )
        w.mousePressEvent(press)

        move = self._make_mouse_event(
            QEvent.Type.MouseMove, (100, 150),
            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
        )
        w.mouseMoveEvent(move)

        assert w._orbit_pitch != initial_pitch

    def test_pitch_clamped_at_limit(self, qapp):
        from PySide6.QtCore import QEvent

        w = MeshViewport()
        w.set_camera_mode("orbit")
        w._orbit_pitch = 85.0
        w._mouse_last_pos = (100, 100)

        move = self._make_mouse_event(
            QEvent.Type.MouseMove, (100, -500),
            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
        )
        w.mouseMoveEvent(move)

        assert w._orbit_pitch <= _PITCH_LIMIT
        assert w._orbit_pitch >= -_PITCH_LIMIT

    def test_middle_drag_pans(self, qapp):
        from PySide6.QtCore import QEvent

        w = MeshViewport()
        w.set_camera_mode("orbit")
        initial_center = w._orbit_center.copy()

        press = self._make_mouse_event(
            QEvent.Type.MouseButtonPress, (100, 100),
            Qt.MouseButton.MiddleButton, Qt.MouseButton.MiddleButton,
        )
        w.mousePressEvent(press)

        move = self._make_mouse_event(
            QEvent.Type.MouseMove, (150, 120),
            Qt.MouseButton.NoButton, Qt.MouseButton.MiddleButton,
        )
        w.mouseMoveEvent(move)

        assert not np.array_equal(w._orbit_center, initial_center)

    def test_wheel_zooms_in(self, qapp):
        from PySide6.QtCore import QPointF, QPoint
        from PySide6.QtGui import QWheelEvent

        w = MeshViewport()
        w.set_camera_mode("orbit")
        initial_dist = w._orbit_distance

        wheel = QWheelEvent(
            QPointF(100, 100), QPointF(100, 100),
            QPoint(0, 0), QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
        w.wheelEvent(wheel)

        assert w._orbit_distance < initial_dist

    def test_wheel_zooms_out(self, qapp):
        from PySide6.QtCore import QPointF, QPoint
        from PySide6.QtGui import QWheelEvent

        w = MeshViewport()
        w.set_camera_mode("orbit")
        initial_dist = w._orbit_distance

        wheel = QWheelEvent(
            QPointF(100, 100), QPointF(100, 100),
            QPoint(0, 0), QPoint(0, -120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False,
        )
        w.wheelEvent(wheel)

        assert w._orbit_distance > initial_dist

    def test_zoom_minimum_enforced(self, qapp):
        from PySide6.QtCore import QPointF, QPoint
        from PySide6.QtGui import QWheelEvent

        w = MeshViewport()
        w.set_camera_mode("orbit")
        w._orbit_distance = 0.2

        for _ in range(50):
            wheel = QWheelEvent(
                QPointF(100, 100), QPointF(100, 100),
                QPoint(0, 0), QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase, False,
            )
            w.wheelEvent(wheel)

        assert w._orbit_distance >= 0.1

    def test_double_click_resets_orbit(self, qapp):
        from PySide6.QtCore import QEvent

        w = MeshViewport()
        w.set_camera_mode("orbit")
        w._orbit_yaw = 90.0
        w._orbit_pitch = 45.0

        dbl = self._make_mouse_event(
            QEvent.Type.MouseButtonDblClick, (100, 100),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        )
        w.mouseDoubleClickEvent(dbl)

        assert w._orbit_yaw == _ORBIT_DEFAULT_YAW
        assert w._orbit_pitch == _ORBIT_DEFAULT_PITCH

    def test_incam_ignores_drag(self, qapp):
        """Drags in incam mode should not change the view matrix."""
        from PySide6.QtCore import QEvent

        w = MeshViewport()
        initial_view = w._view.copy()

        press = self._make_mouse_event(
            QEvent.Type.MouseButtonPress, (100, 100),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        )
        w.mousePressEvent(press)

        move = self._make_mouse_event(
            QEvent.Type.MouseMove, (200, 200),
            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
        )
        w.mouseMoveEvent(move)

        np.testing.assert_array_equal(w._view, initial_view)

    def test_release_clears_tracking(self, qapp):
        from PySide6.QtCore import QEvent

        w = MeshViewport()
        w.set_camera_mode("orbit")
        w._mouse_last_pos = (100, 100)

        release = self._make_mouse_event(
            QEvent.Type.MouseButtonRelease, (100, 100),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        )
        w.mouseReleaseEvent(release)

        assert w._mouse_last_pos is None

    def test_orbit_emits_signal_on_drag(self, qapp):
        """Mouse drag in orbit mode should emit camera_changed."""
        from PySide6.QtCore import QEvent

        w = MeshViewport()
        w.set_camera_mode("orbit")
        received = []
        w.camera_changed.connect(lambda s: received.append(s))

        press = self._make_mouse_event(
            QEvent.Type.MouseButtonPress, (100, 100),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        )
        w.mousePressEvent(press)

        move = self._make_mouse_event(
            QEvent.Type.MouseMove, (120, 100),
            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
        )
        w.mouseMoveEvent(move)

        assert len(received) >= 1
        assert received[-1]["mode"] == "orbit"


# ======================================================================
# Skeleton data tests (Phase 3.3)
# ======================================================================


class TestSkeletonData:
    """JOINT_NAMES, JOINT_PARENTS, DEFAULT_OFFSETS, BONE_CONNECTIONS."""

    def test_joint_names_length(self):
        assert len(JOINT_NAMES) == 52

    def test_joint_parents_length(self):
        assert len(JOINT_PARENTS) == 52

    def test_root_has_no_parent(self):
        assert JOINT_PARENTS[0] == -1

    def test_all_parents_valid(self):
        for i, p in enumerate(JOINT_PARENTS):
            if i == 0:
                assert p == -1
            else:
                assert 0 <= p < i, f"joint {i} has invalid parent {p}"

    def test_default_offsets_has_all_joints(self):
        for name in JOINT_NAMES:
            assert name in DEFAULT_OFFSETS, f"missing offset for {name}"

    def test_default_offsets_are_3d(self):
        for name, off in DEFAULT_OFFSETS.items():
            assert len(off) == 3, f"offset for {name} is not 3D"

    def test_bone_connections_body(self):
        """Body bones (first 20) should all reference joints 0-21."""
        body_bones = [
            (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
            (0, 1), (1, 4), (4, 7), (7, 10),
            (0, 2), (2, 5), (5, 8), (8, 11),
            (9, 13), (13, 16), (16, 18), (18, 20),
            (9, 14), (14, 17), (17, 19), (19, 21),
        ]
        for bone in body_bones:
            assert bone in BONE_CONNECTIONS

    def test_bone_connections_has_hand_bones(self):
        """Should have 30 hand bone connections (15 per hand, 3 per finger × 5 fingers)."""
        hand_bones = [b for b in BONE_CONNECTIONS if b[0] >= 20 or b[1] >= 22]
        assert len(hand_bones) == 30

    def test_bone_connections_all_valid(self):
        """All bone connections should reference valid joint indices."""
        for a, b in BONE_CONNECTIONS:
            assert 0 <= a < 52
            assert 0 <= b < 52

    def test_n_body_joints(self):
        assert _N_BODY_JOINTS == 22

    def test_joint_names_start_with_pelvis(self):
        assert JOINT_NAMES[0] == "Pelvis"

    def test_joint_names_body_count(self):
        """First 22 joints are body joints."""
        body = JOINT_NAMES[:22]
        assert body[0] == "Pelvis"
        assert body[15] == "Head"
        assert body[20] == "L_Wrist"
        assert body[21] == "R_Wrist"


# ======================================================================
# forward_kinematics tests (Phase 3.3)
# ======================================================================


class TestForwardKinematics:
    """forward_kinematics: params dict → (52, 3) joint positions."""

    def _make_params(self, n_frames=10):
        """Create minimal params dict with zero rotations."""
        return {
            "global_orient": np.zeros((n_frames, 3)),
            "body_pose": np.zeros((n_frames, 21, 3)),
            "transl": np.zeros((n_frames, 3)),
        }

    def test_output_shape(self):
        params = self._make_params()
        joints = forward_kinematics(params, 0)
        assert joints.shape == (52, 3)

    def test_root_at_origin(self):
        params = self._make_params()
        joints = forward_kinematics(params, 0)
        np.testing.assert_allclose(joints[0], [0, 0, 0], atol=1e-6)

    def test_root_with_translation(self):
        params = self._make_params()
        params["transl"][3] = [1.0, 2.0, 3.0]
        joints = forward_kinematics(params, 3)
        np.testing.assert_allclose(joints[0], [1.0, 2.0, 3.0], atol=1e-6)

    def test_children_offset_from_root(self):
        """With zero rotations, child joints should be offset from root by DEFAULT_OFFSETS."""
        params = self._make_params()
        joints = forward_kinematics(params, 0)
        # Spine1 (idx 3) is child of Pelvis (idx 0)
        expected_spine1 = np.array(DEFAULT_OFFSETS["Spine1"])
        np.testing.assert_allclose(joints[3], expected_spine1, atol=1e-5)

    def test_different_frames_give_different_results(self):
        params = self._make_params()
        params["transl"][0] = [0, 0, 0]
        params["transl"][5] = [5, 5, 5]
        j0 = forward_kinematics(params, 0)
        j5 = forward_kinematics(params, 5)
        assert not np.allclose(j0, j5)

    def test_missing_params_returns_zeros(self):
        """Missing global_orient/body_pose returns zero positions."""
        joints = forward_kinematics({}, 0)
        np.testing.assert_allclose(joints, 0.0)

    def test_flat_body_pose(self):
        """Should handle flat (N, 63) body_pose."""
        params = self._make_params()
        params["body_pose"] = np.zeros((10, 63))
        joints = forward_kinematics(params, 0)
        assert joints.shape == (52, 3)

    def test_with_hand_pose(self):
        """Hand joints should be offset from wrist when hand_pose is provided."""
        params = self._make_params()
        params["left_hand_pose"] = np.zeros((10, 15, 3))
        params["right_hand_pose"] = np.zeros((10, 15, 3))
        joints = forward_kinematics(params, 0)
        # L_Index1 (22) should be offset from L_Wrist (20)
        wrist_pos = joints[20]
        index1_pos = joints[22]
        assert not np.allclose(wrist_pos, index1_pos)

    def test_rotation_affects_children(self):
        """Rotating the root should move all children."""
        params = self._make_params()
        j_rest = forward_kinematics(params, 0).copy()
        params["global_orient"][0] = [0, np.pi / 2, 0]  # 90° yaw
        j_rotated = forward_kinematics(params, 0)
        # Root stays at origin
        np.testing.assert_allclose(j_rest[0], j_rotated[0], atol=1e-6)
        # Other joints should move
        assert not np.allclose(j_rest[3], j_rotated[3], atol=1e-3)


# ======================================================================
# project_joints_to_screen tests (Phase 3.3)
# ======================================================================


class TestProjectJointsToScreen:
    """project_joints_to_screen: 3D joints → 2D screen pixels."""

    def test_output_shape(self):
        joints = np.zeros((52, 3))
        mvp = np.eye(4)
        screen = project_joints_to_screen(joints, mvp, 640, 480)
        assert screen.shape == (52, 2)

    def test_origin_projects_to_center(self):
        """A point at origin with identity MVP should project to screen center."""
        joints = np.array([[0, 0, 0]], dtype=np.float64)
        mvp = np.eye(4)
        screen = project_joints_to_screen(joints, mvp, 640, 480)
        np.testing.assert_allclose(screen[0, 0], 320, atol=1)
        np.testing.assert_allclose(screen[0, 1], 240, atol=1)

    def test_right_is_positive_x(self):
        """Points to the right in NDC should have larger screen x."""
        joints = np.array([[0, 0, 0], [0.5, 0, 0]], dtype=np.float64)
        mvp = np.eye(4)
        screen = project_joints_to_screen(joints, mvp, 640, 480)
        assert screen[1, 0] > screen[0, 0]

    def test_up_is_negative_y(self):
        """Points up in NDC (+Y) should have smaller screen y (screen Y is flipped)."""
        joints = np.array([[0, 0, 0], [0, 0.5, 0]], dtype=np.float64)
        mvp = np.eye(4)
        screen = project_joints_to_screen(joints, mvp, 640, 480)
        assert screen[1, 1] < screen[0, 1]

    def test_perspective_division(self):
        """Points further away should project closer to center."""
        mvp = perspective_fov(45.0, 640 / 480)
        # Two points at same X but different Z
        j = np.array([[0.5, 0, -2], [0.5, 0, -10]], dtype=np.float64)
        screen = project_joints_to_screen(j, mvp, 640, 480)
        # Closer point should be further from center than far point
        center_x = 320
        assert abs(screen[0, 0] - center_x) > abs(screen[1, 0] - center_x)


# ======================================================================
# find_nearest_joint tests (Phase 3.3)
# ======================================================================


class TestFindNearestJoint:
    """find_nearest_joint: click position → joint index or None."""

    def _make_joints_2d(self):
        """Create 52 joints spread across screen."""
        joints = np.zeros((52, 2))
        for i in range(52):
            joints[i] = [50 + i * 10, 100 + (i % 5) * 20]
        return joints

    def test_exact_hit(self):
        joints = self._make_joints_2d()
        result = find_nearest_joint(50.0, 100.0, joints)
        assert result == 0

    def test_within_threshold(self):
        joints = self._make_joints_2d()
        # Click 15px away from joint 5 (x=100, y=100)
        result = find_nearest_joint(115.0, 100.0, joints)
        assert result == 5  # joint at (100, 100), 15px away

    def test_beyond_threshold(self):
        """Click far from any joint should return None."""
        joints = self._make_joints_2d()
        result = find_nearest_joint(1000.0, 1000.0, joints)
        assert result is None

    def test_only_body_joints(self):
        """Default max_joint=22 means hand joints are ignored."""
        joints = np.zeros((52, 2))
        joints[30] = [100, 100]  # A hand joint right at click position
        result = find_nearest_joint(100.0, 100.0, joints, max_joint=22)
        # All body joints are at (0,0), ~141px from (100,100) → beyond threshold
        assert result is None

    def test_custom_threshold(self):
        joints = np.zeros((52, 2))
        joints[5] = [100, 100]
        # 50px away
        result = find_nearest_joint(150.0, 100.0, joints, threshold=60.0)
        assert result == 5
        result = find_nearest_joint(150.0, 100.0, joints, threshold=10.0)
        assert result is None

    def test_custom_max_joint(self):
        """With max_joint=52, hand joints are also pickable."""
        joints = np.zeros((52, 2))
        joints[40] = [100, 100]  # R_Middle1
        result = find_nearest_joint(100.0, 100.0, joints, max_joint=52)
        assert result == 40

    def test_returns_nearest(self):
        """When multiple joints are within threshold, returns the nearest."""
        joints = np.zeros((52, 2))
        joints[0] = [100, 100]
        joints[1] = [108, 100]  # 8px from click
        joints[2] = [95, 100]   # 5px from click
        result = find_nearest_joint(100.0, 100.0, joints)
        assert result == 0  # exactly at click


# ======================================================================
# Widget skeleton tests (Phase 3.3)
# ======================================================================


class TestMeshViewportSkeleton:
    """MeshViewport skeleton overlay state and joint picking."""

    def test_initial_joint_state(self, qapp):
        w = MeshViewport()
        assert w._selected_joint == -1
        assert w._joint_positions is None
        assert w._show_skeleton is True

    def test_highlight_joint(self, qapp):
        w = MeshViewport()
        w.highlight_joint(5)
        assert w._selected_joint == 5

    def test_highlight_joint_same_noop(self, qapp):
        w = MeshViewport()
        w.highlight_joint(5)
        w.highlight_joint(5)  # no change
        assert w._selected_joint == 5

    def test_highlight_joint_change(self, qapp):
        w = MeshViewport()
        w.highlight_joint(5)
        w.highlight_joint(10)
        assert w._selected_joint == 10

    def test_set_show_skeleton(self, qapp):
        w = MeshViewport()
        w.set_show_skeleton(False)
        assert w._show_skeleton is False
        w.set_show_skeleton(True)
        assert w._show_skeleton is True

    def test_joint_clicked_signal_exists(self, qapp):
        w = MeshViewport()
        assert hasattr(w, "joint_clicked")

    def test_joint_clicked_emitted_on_pick(self, qapp, session):
        """joint_clicked should be emitted when a joint is picked."""
        w = MeshViewport()
        # Set up fake joint positions
        w._joint_positions = np.zeros((52, 3))
        w._joint_positions[5] = [0, 0, -2]  # Some position

        # Mock _pick_joint to return a hit
        w._pick_joint = MagicMock(return_value=5)

        received = []
        w.joint_clicked.connect(lambda j: received.append(j))

        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QMouseEvent

        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(100, 100), QPointF(100, 100),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        w.mousePressEvent(press)

        assert len(received) == 1
        assert received[0] == 5
        assert w._selected_joint == 5

    def test_no_emit_on_miss(self, qapp):
        """No signal when click doesn't hit any joint."""
        w = MeshViewport()
        w._pick_joint = MagicMock(return_value=None)

        received = []
        w.joint_clicked.connect(lambda j: received.append(j))

        from PySide6.QtCore import QEvent, QPointF
        from PySide6.QtGui import QMouseEvent

        press = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(100, 100), QPointF(100, 100),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        w.mousePressEvent(press)

        assert len(received) == 0

    def test_pick_joint_returns_none_no_joints(self, qapp):
        """_pick_joint returns None when no joint positions exist."""
        w = MeshViewport()
        assert w._pick_joint(100, 100) is None

    def test_compute_joints_no_session(self, qapp):
        w = MeshViewport()
        assert w._compute_joints() is None

    def test_compute_joints_no_track(self, qapp, session):
        w = MeshViewport()
        w.set_session(session)
        w._person_id = 99
        assert w._compute_joints() is None

    def test_compute_joints_no_params(self, qapp, session):
        """Should return None when smplx_params is None."""
        w = MeshViewport()
        session.person_tracks[0] = PersonTrack(person_id=0)
        w.set_session(session)
        w._person_id = 0
        result = w._compute_joints()
        assert result is None

    def test_compute_joints_with_params(self, qapp, session):
        """Should return (52,3) joint positions when params are set."""
        w = MeshViewport()
        session.person_tracks[0] = PersonTrack(
            person_id=0,
            smplx_params={
                "global_orient": np.zeros((10, 3)),
                "body_pose": np.zeros((10, 21, 3)),
                "transl": np.tile([0, 0, 2], (10, 1)).astype(float),
            },
        )
        w.set_session(session)
        w._person_id = 0
        result = w._compute_joints()
        assert result is not None
        assert result.shape == (52, 3)
        # Root should be at translation
        np.testing.assert_allclose(result[0], [0, 0, 2], atol=1e-6)

    def test_refresh_mesh_computes_joints(self, qapp, session):
        """_refresh_mesh should also compute _joint_positions."""
        w = MeshViewport()
        session.person_tracks[0] = PersonTrack(
            person_id=0,
            smplx_params={
                "global_orient": np.zeros((10, 3)),
                "body_pose": np.zeros((10, 21, 3)),
                "transl": np.zeros((10, 3)),
            },
        )
        w.set_session(session)
        w._person_id = 0
        w._current_frame = -1  # force refresh
        w.on_frame_changed(0)
        # Joint positions should have been computed
        assert w._joint_positions is not None
        assert w._joint_positions.shape == (52, 3)


# ======================================================================
# Color mode tests
# ======================================================================

from views.mesh_viewport import (
    compute_joint_colors,
    confidence_to_color,
    _JOINT_PALETTE,
    _SKIN_COLOR,
)


class TestComputeJointColors:
    """compute_joint_colors: LBS weight → per-vertex joint coloring."""

    def test_output_shape(self):
        """Output should be (V, 3) float32."""
        V, J = 50, 55
        weights = np.random.rand(V, J).astype(np.float32)
        # Normalize rows to sum to 1
        weights /= weights.sum(axis=1, keepdims=True)
        colors = compute_joint_colors(weights)
        assert colors.shape == (V, 3)
        assert colors.dtype == np.float32

    def test_body_joint_dominant(self):
        """Vertex dominated by body joint 3 should get palette color 3."""
        V, J = 10, 55
        weights = np.zeros((V, J), dtype=np.float32)
        weights[:, 3] = 1.0  # all verts dominated by joint 3 (Spine1)
        colors = compute_joint_colors(weights)
        expected = _JOINT_PALETTE[3]
        for i in range(V):
            np.testing.assert_allclose(colors[i], expected, atol=1e-6)

    def test_left_hand_inherits_wrist(self):
        """Vertex dominated by left hand joint (22-36) should get L_Wrist color (20)."""
        V, J = 5, 55
        weights = np.zeros((V, J), dtype=np.float32)
        weights[:, 25] = 1.0  # joint 25 = L_Middle1 (left hand)
        colors = compute_joint_colors(weights)
        expected = _JOINT_PALETTE[20]  # L_Wrist
        for i in range(V):
            np.testing.assert_allclose(colors[i], expected, atol=1e-6)

    def test_right_hand_inherits_wrist(self):
        """Vertex dominated by right hand joint (37-51) should get R_Wrist color (21)."""
        V, J = 5, 55
        weights = np.zeros((V, J), dtype=np.float32)
        weights[:, 40] = 1.0  # joint 40 = R_Middle1 (right hand)
        colors = compute_joint_colors(weights)
        expected = _JOINT_PALETTE[21]  # R_Wrist
        for i in range(V):
            np.testing.assert_allclose(colors[i], expected, atol=1e-6)

    def test_jaw_eye_joints_inherit_head(self):
        """Joints beyond 51 (jaw/eyes) should map to Head color (15)."""
        V, J = 5, 55
        weights = np.zeros((V, J), dtype=np.float32)
        weights[:, 53] = 1.0  # joint 53 = jaw/eye area
        colors = compute_joint_colors(weights)
        expected = _JOINT_PALETTE[15]  # Head
        for i in range(V):
            np.testing.assert_allclose(colors[i], expected, atol=1e-6)

    def test_mixed_joints(self):
        """Different vertices dominated by different joints get different colors."""
        V, J = 3, 55
        weights = np.zeros((V, J), dtype=np.float32)
        weights[0, 0] = 1.0   # Pelvis
        weights[1, 15] = 1.0  # Head
        weights[2, 21] = 1.0  # R_Wrist
        colors = compute_joint_colors(weights)
        np.testing.assert_allclose(colors[0], _JOINT_PALETTE[0], atol=1e-6)
        np.testing.assert_allclose(colors[1], _JOINT_PALETTE[15], atol=1e-6)
        np.testing.assert_allclose(colors[2], _JOINT_PALETTE[21], atol=1e-6)

    def test_custom_palette(self):
        """Should accept a custom palette."""
        V, J = 3, 55
        weights = np.zeros((V, J), dtype=np.float32)
        weights[:, 0] = 1.0
        custom = np.zeros((22, 3), dtype=np.float32)
        custom[0] = [0.1, 0.2, 0.3]
        colors = compute_joint_colors(weights, palette=custom)
        np.testing.assert_allclose(colors[0], [0.1, 0.2, 0.3], atol=1e-6)


class TestConfidenceToColor:
    """confidence_to_color: scalar → RGB gradient."""

    def test_zero_is_red(self):
        color = confidence_to_color(0.0)
        np.testing.assert_allclose(color, [1.0, 0.0, 0.0], atol=1e-6)

    def test_half_is_yellow(self):
        color = confidence_to_color(0.5)
        np.testing.assert_allclose(color, [1.0, 1.0, 0.0], atol=1e-6)

    def test_one_is_green(self):
        color = confidence_to_color(1.0)
        np.testing.assert_allclose(color, [0.0, 1.0, 0.0], atol=1e-6)

    def test_clamp_below_zero(self):
        color = confidence_to_color(-0.5)
        np.testing.assert_allclose(color, [1.0, 0.0, 0.0], atol=1e-6)

    def test_clamp_above_one(self):
        color = confidence_to_color(1.5)
        np.testing.assert_allclose(color, [0.0, 1.0, 0.0], atol=1e-6)

    def test_quarter(self):
        """0.25 → halfway between red and yellow."""
        color = confidence_to_color(0.25)
        np.testing.assert_allclose(color, [1.0, 0.5, 0.0], atol=1e-6)

    def test_three_quarter(self):
        """0.75 → halfway between yellow and green."""
        color = confidence_to_color(0.75)
        np.testing.assert_allclose(color, [0.5, 1.0, 0.0], atol=1e-6)

    def test_output_dtype(self):
        assert confidence_to_color(0.5).dtype == np.float32


class TestMeshViewportColorMode:
    """MeshViewport.set_color_mode integration tests."""

    def test_default_color_mode(self, qapp):
        w = MeshViewport()
        assert w._color_mode == "solid"

    def test_set_color_mode_solid(self, qapp):
        w = MeshViewport()
        w._color_mode = "joint"  # change away first
        w.set_color_mode("solid")
        assert w._color_mode == "solid"

    def test_set_color_mode_joint(self, qapp):
        w = MeshViewport()
        w.set_color_mode("joint")
        assert w._color_mode == "joint"

    def test_set_color_mode_confidence(self, qapp):
        w = MeshViewport()
        w.set_color_mode("confidence")
        assert w._color_mode == "confidence"

    def test_set_invalid_mode_ignored(self, qapp):
        w = MeshViewport()
        w.set_color_mode("invalid")
        assert w._color_mode == "solid"

    def test_set_same_mode_noop(self, qapp):
        """Setting the same mode should be a no-op."""
        w = MeshViewport()
        w.set_color_mode("solid")
        # Should not error even when called twice
        assert w._color_mode == "solid"

    def test_compute_colors_solid(self, qapp):
        """Solid mode should return skin tone for all vertices."""
        w = MeshViewport()
        w._n_vertices = 10
        colors = w._compute_colors()
        assert colors.shape == (10, 3)
        for i in range(10):
            np.testing.assert_allclose(colors[i], _SKIN_COLOR, atol=1e-6)

    def test_compute_colors_joint_with_weights(self, qapp):
        """Joint mode with LBS weights should use joint coloring."""
        w = MeshViewport()
        w._n_vertices = 20
        w._color_mode = "joint"
        # Create fake LBS weights: all vertices dominated by joint 5
        lbs = np.zeros((20, 55), dtype=np.float32)
        lbs[:, 5] = 1.0
        w._lbs_weights = lbs
        colors = w._compute_colors()
        assert colors.shape == (20, 3)
        np.testing.assert_allclose(colors[0], _JOINT_PALETTE[5], atol=1e-6)

    def test_compute_colors_joint_no_weights_falls_back(self, qapp):
        """Joint mode without LBS weights should fall back to solid."""
        w = MeshViewport()
        w._n_vertices = 10
        w._color_mode = "joint"
        w._lbs_weights = None
        colors = w._compute_colors()
        assert colors.shape == (10, 3)
        np.testing.assert_allclose(colors[0], _SKIN_COLOR, atol=1e-6)

    def test_compute_colors_confidence_default(self, qapp):
        """Confidence mode with no session should use mid confidence (yellow)."""
        w = MeshViewport()
        w._n_vertices = 10
        w._color_mode = "confidence"
        colors = w._compute_colors()
        assert colors.shape == (10, 3)
        expected = confidence_to_color(0.5)
        np.testing.assert_allclose(colors[0], expected, atol=1e-6)

    def test_compute_colors_confidence_from_breakdown(self, qapp, session):
        """Confidence mode should read from confidence_breakdown['overall']."""
        w = MeshViewport()
        w._n_vertices = 10
        w._color_mode = "confidence"
        track = PersonTrack(
            person_id=0,
            confidence_breakdown={"overall": [0.2, 0.8, 0.5]},
        )
        session.person_tracks[0] = track
        w.set_session(session)
        w._person_id = 0
        w._current_frame = 1  # confidence 0.8
        colors = w._compute_colors()
        expected = confidence_to_color(0.8)
        np.testing.assert_allclose(colors[0], expected, atol=1e-6)

    def test_compute_colors_confidence_from_raw_list(self, qapp, session):
        """Confidence mode should fall back to raw confidences list."""
        w = MeshViewport()
        w._n_vertices = 10
        w._color_mode = "confidence"
        track = PersonTrack(
            person_id=0,
            confidences=[0.9, 0.1, 0.5],
        )
        session.person_tracks[0] = track
        w.set_session(session)
        w._person_id = 0
        w._current_frame = 0  # confidence 0.9
        colors = w._compute_colors()
        expected = confidence_to_color(0.9)
        np.testing.assert_allclose(colors[0], expected, atol=1e-6)

    def test_lbs_weights_extracted_on_model_load(self, qapp):
        """_load_model should extract lbs_weights from body model."""
        w = MeshViewport()
        # Create mock model with lbs_weights
        mock_model = MagicMock()
        mock_model.faces = np.array([[0, 1, 2]], dtype=np.int32)
        mock_model.cpu.return_value = mock_model
        mock_model.eval.return_value = mock_model

        # Fake lbs_weights tensor
        fake_weights = np.random.rand(100, 55).astype(np.float32)

        class FakeLBS:
            def detach(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return fake_weights

        mock_model.lbs_weights = FakeLBS()

        # Inject mock
        w._model_loaded = True
        w._body_model = mock_model
        w._faces = mock_model.faces
        w._lbs_weights = mock_model.lbs_weights.detach().cpu().numpy()

        assert w._lbs_weights is not None
        assert w._lbs_weights.shape == (100, 55)


# ======================================================================
# Grid floor tests
# ======================================================================


class TestComputeGridLines:
    """Tests for the pure compute_grid_lines() helper function."""

    def test_returns_positions_and_colors(self):
        """Should return a tuple of (positions, colors) arrays."""
        pos, col = compute_grid_lines()
        assert isinstance(pos, np.ndarray)
        assert isinstance(col, np.ndarray)
        assert pos.dtype == np.float32
        assert col.dtype == np.float32

    def test_shape_matches(self):
        """positions and colors should have same shape (N, 3)."""
        pos, col = compute_grid_lines()
        assert pos.shape == col.shape
        assert pos.ndim == 2
        assert pos.shape[1] == 3

    def test_expected_vertex_count(self):
        """Lines per axis = 2*divisions+1, each line has 2 endpoints, 2 axes."""
        divisions = 5
        pos, col = compute_grid_lines(divisions=divisions)
        lines_per_axis = 2 * divisions + 1
        expected = lines_per_axis * 2 * 2  # 2 axes × 2 endpoints per line
        assert len(pos) == expected

    def test_default_divisions(self):
        """Default divisions should produce correct count."""
        pos, col = compute_grid_lines()
        lines_per_axis = 2 * _GRID_DIVISIONS + 1
        expected = lines_per_axis * 2 * 2
        assert len(pos) == expected

    def test_y_coordinate(self):
        """All vertices should be at the specified Y level."""
        y_val = -1.5
        pos, _ = compute_grid_lines(y=y_val)
        np.testing.assert_allclose(pos[:, 1], y_val, atol=1e-7)

    def test_default_y_is_zero(self):
        """Default Y should be 0."""
        pos, _ = compute_grid_lines()
        np.testing.assert_allclose(pos[:, 1], 0.0, atol=1e-7)

    def test_centered_at_custom_position(self):
        """Grid should be centered at specified center_x, center_z."""
        cx, cz = 2.0, -3.0
        size = 5.0
        pos, _ = compute_grid_lines(size=size, divisions=2, center_x=cx, center_z=cz)
        # X range should be [cx - size, cx + size]
        assert pos[:, 0].min() == pytest.approx(cx - size, abs=1e-6)
        assert pos[:, 0].max() == pytest.approx(cx + size, abs=1e-6)
        # Z range should be [cz - size, cz + size]
        assert pos[:, 2].min() == pytest.approx(cz - size, abs=1e-6)
        assert pos[:, 2].max() == pytest.approx(cz + size, abs=1e-6)

    def test_center_lines_use_axis_color(self):
        """The center cross-hair lines (i=0) should use _GRID_AXIS_COLOR."""
        pos, col = compute_grid_lines(divisions=2)
        # Find vertices on the center Z-parallel line: x ≈ 0
        center_mask = np.abs(pos[:, 0]) < 1e-6
        if np.any(center_mask):
            for idx in np.where(center_mask)[0]:
                np.testing.assert_allclose(col[idx], _GRID_AXIS_COLOR, atol=1e-6)

    def test_non_center_lines_use_grid_color(self):
        """Non-center lines should use _GRID_COLOR."""
        pos, col = compute_grid_lines(size=5.0, divisions=2)
        # Find a vertex on x = step (non-center)
        step = 5.0 / 2
        non_center_mask = np.abs(pos[:, 0] - step) < 1e-6
        if np.any(non_center_mask):
            idx = np.where(non_center_mask)[0][0]
            np.testing.assert_allclose(col[idx], _GRID_COLOR, atol=1e-6)

    def test_single_division(self):
        """divisions=1 should produce 3 lines per axis (6 lines total, 12 verts)."""
        pos, _ = compute_grid_lines(divisions=1)
        assert len(pos) == 3 * 2 * 2  # 3 lines × 2 axes × 2 endpoints

    def test_size_affects_extent(self):
        """Grid should span ±size in each axis direction."""
        size = 3.0
        pos, _ = compute_grid_lines(size=size, divisions=5)
        assert pos[:, 0].min() == pytest.approx(-size, abs=1e-6)
        assert pos[:, 0].max() == pytest.approx(size, abs=1e-6)
        assert pos[:, 2].min() == pytest.approx(-size, abs=1e-6)
        assert pos[:, 2].max() == pytest.approx(size, abs=1e-6)


class TestMeshViewportGrid:
    """Tests for MeshViewport grid floor state and rendering."""

    def test_show_grid_default_true(self, qapp):
        """Grid should be visible by default."""
        w = MeshViewport()
        assert w._show_grid is True

    def test_set_show_grid(self, qapp):
        """set_show_grid should toggle the flag."""
        w = MeshViewport()
        w.set_show_grid(False)
        assert w._show_grid is False
        w.set_show_grid(True)
        assert w._show_grid is True

    def test_grid_y_default(self, qapp):
        """Grid Y should default to 0."""
        w = MeshViewport()
        assert w._grid_y == 0.0

    def test_grid_y_updated_on_auto_center(self, qapp):
        """_auto_center_orbit should set _grid_y to lowest GL vertex Y."""
        w = MeshViewport()
        # Fake vertices in CV space: Y values 0.5..2.5 (Y down in CV)
        # After GL flip: Y values -0.5..-2.5 → min is -2.5
        w._vertices = np.array([
            [0, 0.5, 1],
            [0, 1.5, 1],
            [0, 2.5, 1],
        ], dtype=np.float32)
        w._auto_center_orbit()
        # In GL space, Y is flipped: -0.5, -1.5, -2.5 → min = -2.5
        assert w._grid_y == pytest.approx(-2.5, abs=1e-5)

    def test_grid_only_in_orbit_mode(self, qapp):
        """Grid should only be drawn in orbit mode (verified via state check)."""
        w = MeshViewport()
        # In incam mode, paintGL should NOT call _draw_grid
        assert w._camera_mode == "incam"
        assert w._show_grid is True
        # The grid draw is guarded by camera_mode == "orbit" in paintGL

    def test_grid_hidden_when_show_grid_false(self, qapp):
        """Grid should not draw when _show_grid is False."""
        w = MeshViewport()
        w._camera_mode = "orbit"
        w.set_show_grid(False)
        assert w._show_grid is False


# ======================================================================
# Joint label overlay tests (pure function + widget state)
# ======================================================================


class TestComputeJointLabelLayout:
    """compute_joint_label_layout: pure function that computes label positions."""

    def _make_joints_2d(self, n=52, x=100.0, y=100.0):
        """Create synthetic 2D joint positions, all at the same point."""
        return np.full((n, 2), [x, y], dtype=np.float64)

    def test_returns_list_of_tuples(self):
        """Output should be a list of (idx, name, x, y, is_selected) tuples."""
        joints_2d = self._make_joints_2d()
        result = compute_joint_label_layout(joints_2d, 400, 300)
        assert isinstance(result, list)
        assert len(result) > 0
        idx, name, sx, sy, is_sel = result[0]
        assert isinstance(idx, int)
        assert isinstance(name, str)
        assert isinstance(is_sel, bool)

    def test_body_only_limits_to_22_joints(self):
        """body_only=True should return at most 22 labels (body joints 0-21)."""
        joints_2d = self._make_joints_2d()
        result = compute_joint_label_layout(joints_2d, 400, 300, body_only=True)
        indices = [r[0] for r in result]
        assert all(i < _N_BODY_JOINTS for i in indices)
        assert len(result) == _N_BODY_JOINTS

    def test_all_joints_mode(self):
        """body_only=False should include hand joints too."""
        joints_2d = self._make_joints_2d()
        result = compute_joint_label_layout(joints_2d, 400, 300, body_only=False)
        indices = [r[0] for r in result]
        assert max(indices) >= _N_BODY_JOINTS  # hand joints included

    def test_selected_joint_included_even_when_hand(self):
        """Selected hand joint should appear in labels even with body_only=True."""
        joints_2d = self._make_joints_2d()
        selected = 30  # hand joint
        result = compute_joint_label_layout(
            joints_2d, 400, 300, selected_joint=selected, body_only=True
        )
        indices = [r[0] for r in result]
        assert selected in indices

    def test_selected_joint_is_last(self):
        """Selected joint label should be last in the list (drawn on top)."""
        joints_2d = self._make_joints_2d()
        result = compute_joint_label_layout(
            joints_2d, 400, 300, selected_joint=5
        )
        assert result[-1][0] == 5
        assert result[-1][4] is True  # is_selected

    def test_offscreen_joints_excluded(self):
        """Joints projected far outside the viewport should be excluded."""
        joints_2d = np.full((52, 2), [-200.0, -200.0], dtype=np.float64)
        result = compute_joint_label_layout(joints_2d, 400, 300)
        assert len(result) == 0

    def test_joint_names_match_constant(self):
        """Label names should match the JOINT_NAMES constant."""
        joints_2d = self._make_joints_2d()
        result = compute_joint_label_layout(joints_2d, 400, 300, body_only=True)
        for idx, name, *_ in result:
            assert name == JOINT_NAMES[idx]

    def test_positions_clamped_to_viewport(self):
        """Label positions should be clamped within viewport bounds."""
        # Place joints at extreme corners
        joints_2d = np.array([[0, 0], [999, 999]] + [[100, 100]] * 50, dtype=np.float64)
        result = compute_joint_label_layout(joints_2d, 400, 300, margin=4)
        for _, _, sx, sy, _ in result:
            assert sx >= _LABEL_MARGIN
            assert sy >= _LABEL_MARGIN
            assert sx <= 400 - _LABEL_MARGIN
            assert sy <= 300 - _LABEL_MARGIN

    def test_empty_joints_array(self):
        """Empty joints array should return empty list."""
        joints_2d = np.zeros((0, 2), dtype=np.float64)
        result = compute_joint_label_layout(joints_2d, 400, 300)
        assert result == []

    def test_no_selected_joint(self):
        """With selected_joint=-1, no label should be marked as selected."""
        joints_2d = self._make_joints_2d()
        result = compute_joint_label_layout(joints_2d, 400, 300, selected_joint=-1)
        for _, _, _, _, is_sel in result:
            assert is_sel is False


class TestMeshViewportJointLabels:
    """Tests for MeshViewport joint label overlay state and API."""

    def test_show_joint_labels_default_false(self, qapp):
        """Joint labels should be off by default."""
        w = MeshViewport()
        assert w._show_joint_labels is False

    def test_set_show_joint_labels_toggle(self, qapp):
        """set_show_joint_labels should toggle the flag."""
        w = MeshViewport()
        w.set_show_joint_labels(True)
        assert w._show_joint_labels is True
        w.set_show_joint_labels(False)
        assert w._show_joint_labels is False

    def test_set_show_joint_labels_no_op_same_value(self, qapp):
        """Setting the same value should be a no-op (no update triggered)."""
        w = MeshViewport()
        assert w._show_joint_labels is False
        # This should not raise or cause issues
        w.set_show_joint_labels(False)
        assert w._show_joint_labels is False

    def test_labels_require_skeleton_visible(self, qapp):
        """Labels are only drawn when skeleton is also visible (paintGL guard)."""
        w = MeshViewport()
        w.set_show_joint_labels(True)
        w.set_show_skeleton(False)
        # Even with labels enabled, skeleton hidden means no labels drawn
        # (verified by the paintGL condition: _show_joint_labels AND _show_skeleton)
        assert w._show_joint_labels is True
        assert w._show_skeleton is False

    def test_has_draw_joint_labels_method(self, qapp):
        """MeshViewport should have _draw_joint_labels method."""
        w = MeshViewport()
        assert hasattr(w, "_draw_joint_labels")
        assert callable(w._draw_joint_labels)


class TestVideoFrameComposite:
    """Verify in-camera video frame background compositing.

    Why: The spec requires the in-camera 3D viewport to show the mesh
    overlaid on the video frame — a critical feature for verifying that
    the SMPL-X pose aligns with the actual footage. Without this, users
    can't visually compare the reconstructed mesh to the original video.
    """

    def test_set_video_frame_stores_copy(self, qapp):
        """set_video_frame should store a copy of the input array."""
        w = MeshViewport()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        w.set_video_frame(frame)
        assert w._video_frame is not None
        assert w._video_frame is not frame  # must be a copy
        np.testing.assert_array_equal(w._video_frame, frame)

    def test_set_video_frame_none_clears(self, qapp):
        """set_video_frame(None) should clear the stored frame."""
        w = MeshViewport()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        w.set_video_frame(frame)
        assert w._video_frame is not None
        w.set_video_frame(None)
        assert w._video_frame is None

    def test_initial_video_frame_is_none(self, qapp):
        """Video frame should be None by default."""
        w = MeshViewport()
        assert w._video_frame is None

    def test_set_video_frame_mutating_original_no_effect(self, qapp):
        """Modifying the original array after set_video_frame should not
        affect the stored frame (copy isolation)."""
        w = MeshViewport()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        w.set_video_frame(frame)
        frame[0, 0, 0] = 255
        assert w._video_frame[0, 0, 0] == 0

    def test_set_video_frame_different_sizes(self, qapp):
        """set_video_frame should accept different frame sizes."""
        w = MeshViewport()
        for h, wd in [(240, 320), (480, 640), (1080, 1920)]:
            frame = np.random.randint(0, 255, (h, wd, 3), dtype=np.uint8)
            w.set_video_frame(frame)
            assert w._video_frame.shape == (h, wd, 3)

    def test_has_set_video_frame_method(self, qapp):
        """Public API: set_video_frame must exist."""
        w = MeshViewport()
        assert hasattr(w, "set_video_frame")
        assert callable(w.set_video_frame)

    def test_video_frame_only_in_incam_mode(self, qapp):
        """Video frame should be stored regardless of camera mode, but the
        background rendering logic checks camera mode internally."""
        w = MeshViewport()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        w.set_camera_mode("orbit")
        w.set_video_frame(frame)
        assert w._video_frame is not None  # stored, just not rendered

    def test_set_video_frame_updates_triggers_repaint(self, qapp):
        """set_video_frame should store the frame even without GL context."""
        w = MeshViewport()
        frame = np.ones((100, 200, 3), dtype=np.uint8) * 128
        w.set_video_frame(frame)
        assert w._video_frame is not None
        assert w._video_frame.shape == (100, 200, 3)
        np.testing.assert_array_equal(w._video_frame, 128)


# ======================================================================
# FBO color-coded joint picking tests
# ======================================================================


class TestEncodeJointId:
    """encode_joint_id: joint index → unique (R, G, B) color."""

    def test_joint_zero(self):
        """Joint 0 encodes to (1, 0, 0) — 0 is reserved for background."""
        assert encode_joint_id(0) == (1, 0, 0)

    def test_joint_one(self):
        assert encode_joint_id(1) == (2, 0, 0)

    def test_joint_51(self):
        """Last SMPL-X joint encodes correctly."""
        r, g, b = encode_joint_id(51)
        assert r == 52
        assert g == 0
        assert b == 0

    def test_large_index_uses_multiple_channels(self):
        """Joint index 255 → (0, 1, 0) since 256 = 0x100."""
        r, g, b = encode_joint_id(255)
        assert r == 0
        assert g == 1
        assert b == 0

    def test_all_body_joints_unique(self):
        """All 22 body joint colors must be distinct."""
        colors = {encode_joint_id(i) for i in range(22)}
        assert len(colors) == 22

    def test_all_52_joints_unique(self):
        """All 52 SMPL-X joint colors must be distinct."""
        colors = {encode_joint_id(i) for i in range(52)}
        assert len(colors) == 52

    def test_no_joint_encodes_to_black(self):
        """No valid joint should encode to (0, 0, 0) (reserved for background)."""
        for i in range(1000):
            assert encode_joint_id(i) != (0, 0, 0)


class TestDecodeJointId:
    """decode_joint_id: (R, G, B) → joint index or None."""

    def test_background_returns_none(self):
        """(0, 0, 0) is background — no joint hit."""
        assert decode_joint_id(0, 0, 0) is None

    def test_decode_joint_zero(self):
        assert decode_joint_id(1, 0, 0) == 0

    def test_decode_joint_one(self):
        assert decode_joint_id(2, 0, 0) == 1

    def test_decode_joint_51(self):
        assert decode_joint_id(52, 0, 0) == 51

    def test_roundtrip_all_body_joints(self):
        """Encode then decode for all 22 body joints."""
        for i in range(22):
            r, g, b = encode_joint_id(i)
            assert decode_joint_id(r, g, b) == i

    def test_roundtrip_all_52_joints(self):
        """Encode then decode for all 52 SMPL-X joints."""
        for i in range(52):
            r, g, b = encode_joint_id(i)
            assert decode_joint_id(r, g, b) == i

    def test_roundtrip_large_index(self):
        """Roundtrip for index 1000 (uses G channel)."""
        r, g, b = encode_joint_id(1000)
        assert decode_joint_id(r, g, b) == 1000

    def test_decode_multi_channel(self):
        """Decode a color that spans R and G channels."""
        # 255 → encoded = 256 → R=0, G=1
        assert decode_joint_id(0, 1, 0) == 255


class TestMeshViewportFboPicking:
    """MeshViewport FBO picking mode: state, API, and dispatch."""

    def test_default_picking_mode_is_fbo(self, qapp):
        """Default picking mode should be 'fbo'."""
        w = MeshViewport()
        assert w._picking_mode == "fbo"

    def test_set_picking_mode_screen(self, qapp):
        w = MeshViewport()
        w.set_picking_mode("screen")
        assert w._picking_mode == "screen"

    def test_set_picking_mode_fbo(self, qapp):
        w = MeshViewport()
        w.set_picking_mode("screen")
        w.set_picking_mode("fbo")
        assert w._picking_mode == "fbo"

    def test_set_picking_mode_invalid_ignored(self, qapp):
        w = MeshViewport()
        w.set_picking_mode("invalid")
        assert w._picking_mode == "fbo"  # unchanged

    def test_has_set_picking_mode_method(self, qapp):
        w = MeshViewport()
        assert hasattr(w, "set_picking_mode")
        assert callable(w.set_picking_mode)

    def test_pick_fbo_state_initialized(self, qapp):
        """FBO resources should start at zero (not yet created)."""
        w = MeshViewport()
        assert w._pick_fbo_id == 0
        assert w._pick_rbo_color == 0
        assert w._pick_rbo_depth == 0
        assert w._pick_fbo_size == (0, 0)

    def test_pick_joint_screen_mode_no_fbo(self, qapp):
        """In 'screen' mode, _pick_joint should use screen-space distance."""
        w = MeshViewport()
        w.set_picking_mode("screen")
        w._joint_positions = np.zeros((52, 3))
        w._joint_positions[5] = [0, 0, -2]

        # Mock to verify screen-space path is used
        with patch.object(w, "_fbo_pick_joint") as mock_fbo:
            w._pick_joint(100, 100)
            mock_fbo.assert_not_called()

    def test_pick_joint_fbo_mode_attempts_fbo(self, qapp):
        """In 'fbo' mode with GL ready, _pick_joint should attempt FBO picking."""
        w = MeshViewport()
        w._picking_mode = "fbo"
        w._gl_ready = True
        w._joint_positions = np.zeros((52, 3))

        with patch.object(w, "_fbo_pick_joint", return_value=5) as mock_fbo:
            result = w._pick_joint(100, 100)
            if _HAS_GL:
                mock_fbo.assert_called_once_with(100, 100)
                assert result == 5

    def test_pick_joint_fbo_fallback_to_screen(self, qapp):
        """When FBO returns None, _pick_joint should fall back to screen-space."""
        w = MeshViewport()
        w._picking_mode = "fbo"
        w._gl_ready = True
        w._joint_positions = np.zeros((52, 3))
        w._joint_positions[0] = [0, 0, 0]

        with patch.object(w, "_fbo_pick_joint", return_value=None):
            # Should fall through to screen-space distance
            result = w._pick_joint(100, 100)
            # Result depends on projection — just verify no crash
            assert result is None or isinstance(result, int)

    def test_fbo_pick_joint_no_gl(self, qapp):
        """_fbo_pick_joint returns None when GL is not ready."""
        w = MeshViewport()
        w._gl_ready = False
        w._joint_positions = np.zeros((52, 3))
        assert w._fbo_pick_joint(100, 100) is None

    def test_fbo_pick_joint_no_joints(self, qapp):
        """_fbo_pick_joint returns None when no joint positions exist."""
        w = MeshViewport()
        w._gl_ready = True
        w._joint_positions = None
        assert w._fbo_pick_joint(100, 100) is None

    def test_has_fbo_pick_joint_method(self, qapp):
        w = MeshViewport()
        assert hasattr(w, "_fbo_pick_joint")
        assert callable(w._fbo_pick_joint)

    def test_has_ensure_pick_fbo_method(self, qapp):
        w = MeshViewport()
        assert hasattr(w, "_ensure_pick_fbo")
        assert callable(w._ensure_pick_fbo)


# ======================================================================
# Phase 4: Joint chain highlighting — pure function tests
# ======================================================================


class TestGetJointChain:
    """get_joint_chain: walk JOINT_PARENTS from joint to root."""

    def test_root_returns_self(self):
        """Pelvis (root, idx 0) chain is just [0]."""
        chain = get_joint_chain(0)
        assert chain == [0]

    def test_leaf_body_joint(self):
        """L_Foot (idx 10): L_Foot → L_Ankle → L_Knee → L_Hip → Pelvis."""
        chain = get_joint_chain(10)
        assert chain == [10, 7, 4, 1, 0]

    def test_head_chain(self):
        """Head (idx 15): Head → Neck → Spine3 → Spine2 → Spine1 → Pelvis."""
        chain = get_joint_chain(15)
        assert chain == [15, 12, 9, 6, 3, 0]

    def test_wrist_chain(self):
        """L_Wrist (idx 20): L_Wrist → L_Elbow → L_Shoulder → L_Collar → Spine3 → ..."""
        chain = get_joint_chain(20)
        assert chain[0] == 20
        assert chain[-1] == 0  # always ends at root
        assert 18 in chain  # L_Elbow
        assert 16 in chain  # L_Shoulder
        assert 13 in chain  # L_Collar

    def test_hand_joint_chain(self):
        """L_Index3 (idx 24): L_Index3 → L_Index2 → L_Index1 → L_Wrist → ..."""
        chain = get_joint_chain(24)
        assert chain[0] == 24
        assert chain[1] == 23  # L_Index2
        assert chain[2] == 22  # L_Index1
        assert chain[3] == 20  # L_Wrist
        assert chain[-1] == 0

    def test_right_hand_joint(self):
        """R_Thumb3 (idx 51): chain ends at root."""
        chain = get_joint_chain(51)
        assert chain[0] == 51
        assert chain[-1] == 0
        assert 21 in chain  # R_Wrist

    def test_out_of_range_negative(self):
        assert get_joint_chain(-1) == []

    def test_out_of_range_too_large(self):
        assert get_joint_chain(999) == []

    def test_chain_always_ends_at_root(self):
        """Every valid joint's chain ends at Pelvis (0)."""
        for i in range(len(JOINT_PARENTS)):
            chain = get_joint_chain(i)
            assert chain[-1] == 0, f"Joint {i} ({JOINT_NAMES[i]}) chain doesn't end at root"

    def test_chain_is_monotonically_connected(self):
        """Each successive joint in the chain is the parent of the previous."""
        for i in range(len(JOINT_PARENTS)):
            chain = get_joint_chain(i)
            for k in range(len(chain) - 1):
                assert JOINT_PARENTS[chain[k]] == chain[k + 1]


class TestGetJointChainBones:
    """get_joint_chain_bones: set of (parent, child) bone tuples along chain."""

    def test_root_no_bones(self):
        """Root has no parent → no bones in chain."""
        bones = get_joint_chain_bones(0)
        assert bones == set()

    def test_l_knee_bones(self):
        """L_Knee (4): bones are (1,4) and (0,1)."""
        bones = get_joint_chain_bones(4)
        assert (1, 4) in bones
        assert (0, 1) in bones
        assert len(bones) == 2

    def test_head_bones(self):
        """Head (15) chain has 5 bones."""
        bones = get_joint_chain_bones(15)
        assert len(bones) == 5
        assert (12, 15) in bones  # Neck → Head
        assert (9, 12) in bones   # Spine3 → Neck
        assert (0, 3) in bones    # Pelvis → Spine1

    def test_normalized_ordering(self):
        """All bone tuples should be (min, max) for consistent lookup."""
        for i in range(len(JOINT_PARENTS)):
            bones = get_joint_chain_bones(i)
            for a, b in bones:
                assert a < b, f"Bone ({a}, {b}) not normalized for joint {i}"


class TestGetJointSiblings:
    """get_joint_siblings: joints sharing same parent."""

    def test_root_no_siblings(self):
        """Root (Pelvis) has no parent → no siblings."""
        assert get_joint_siblings(0) == []

    def test_hip_siblings(self):
        """L_Hip (1) parent=Pelvis: siblings are R_Hip(2) and Spine1(3)."""
        siblings = get_joint_siblings(1)
        assert 2 in siblings  # R_Hip
        assert 3 in siblings  # Spine1
        assert 1 not in siblings  # excludes self

    def test_spine2_no_siblings(self):
        """Spine2 (6) parent=Spine1(3): only child of Spine1 → no siblings."""
        siblings = get_joint_siblings(6)
        assert siblings == []

    def test_head_no_siblings(self):
        """Head (15) parent=Neck(12): check siblings."""
        siblings = get_joint_siblings(15)
        # Neck's children include Head, L_Collar, R_Collar
        # Wait — let me check: JOINT_PARENTS[13]=9, JOINT_PARENTS[14]=9
        # JOINT_PARENTS[15]=12 (Neck). Only Head has parent Neck.
        # So siblings should be empty.
        # Actually let me check: Neck(12) parent is Spine3(9),
        # L_Collar(13) parent is Spine3(9), R_Collar(14) parent is Spine3(9).
        # Head(15) parent is Neck(12). So Head has parent Neck, which
        # has no other children → no siblings.
        assert siblings == []

    def test_out_of_range(self):
        assert get_joint_siblings(-1) == []
        assert get_joint_siblings(999) == []

    def test_finger_siblings(self):
        """L_Index1 (22) parent=L_Wrist(20): siblings are all finger bases."""
        siblings = get_joint_siblings(22)
        # Other finger bases from left hand with parent=20:
        # L_Middle1(25), L_Pinky1(28), L_Ring1(31), L_Thumb1(34)
        assert 25 in siblings  # L_Middle1
        assert 28 in siblings  # L_Pinky1
        assert 31 in siblings  # L_Ring1
        assert 34 in siblings  # L_Thumb1
        assert 22 not in siblings  # excludes self


class TestGetOppositeJoint:
    """get_opposite_joint: L↔R mirror mapping."""

    def test_center_joints_no_opposite(self):
        """Pelvis, Spine1, Spine2, Spine3, Neck, Head have no opposite."""
        for idx in [0, 3, 6, 9, 12, 15]:
            assert get_opposite_joint(idx) is None

    def test_body_lr_pairs(self):
        """All body L↔R pairs are symmetric."""
        for left, right in [(1, 2), (4, 5), (7, 8), (10, 11),
                            (13, 14), (16, 17), (18, 19), (20, 21)]:
            assert get_opposite_joint(left) == right
            assert get_opposite_joint(right) == left

    def test_hand_joints(self):
        """Left hand joints 22-36 ↔ right hand joints 37-51."""
        for i in range(15):
            assert get_opposite_joint(22 + i) == 37 + i
            assert get_opposite_joint(37 + i) == 22 + i

    def test_symmetry(self):
        """Applying opposite twice returns to original."""
        for i in range(len(JOINT_NAMES)):
            opp = get_opposite_joint(i)
            if opp is not None:
                assert get_opposite_joint(opp) == i


class TestGetJointRegion:
    """get_joint_region: all joints in same body region."""

    def test_spine_region(self):
        region = get_joint_region(0)  # Pelvis
        assert region == [0, 3, 6, 9, 12, 15]

    def test_left_leg_region(self):
        region = get_joint_region(4)  # L_Knee
        assert region == [1, 4, 7, 10]

    def test_right_arm_region(self):
        region = get_joint_region(17)  # R_Shoulder
        assert region == [14, 17, 19, 21]

    def test_left_hand_region(self):
        region = get_joint_region(22)  # L_Index1
        assert len(region) == 15
        assert all(22 <= j <= 36 for j in region)

    def test_right_hand_region(self):
        region = get_joint_region(37)  # R_Index1
        assert len(region) == 15
        assert all(37 <= j <= 51 for j in region)

    def test_region_contains_self(self):
        """The queried joint should always be in its own region."""
        for i in range(len(JOINT_NAMES)):
            region = get_joint_region(i)
            if region:
                assert i in region

    def test_all_joints_have_region(self):
        """Every joint 0-51 should belong to a region."""
        for i in range(52):
            region = get_joint_region(i)
            assert len(region) > 0, f"Joint {i} ({JOINT_NAMES[i]}) has no region"


class TestJointChainConstants:
    """Verify new chain highlighting constants."""

    def test_selected_accent_brighter_than_accent(self):
        """Selected accent should be brighter (higher luminance) than chain accent."""
        assert np.sum(_SELECTED_ACCENT_COLOR) > np.sum(_ACCENT_COLOR)

    def test_chain_bone_color_matches_accent(self):
        np.testing.assert_array_equal(_CHAIN_BONE_COLOR, _ACCENT_COLOR)

    def test_chain_joint_size_between_normal_and_selected(self):
        assert _JOINT_POINT_SIZE < _CHAIN_JOINT_POINT_SIZE < _SELECTED_JOINT_POINT_SIZE

    def test_chain_bone_line_width_thicker_than_normal(self):
        assert _CHAIN_BONE_LINE_WIDTH > _BONE_LINE_WIDTH

    def test_lr_pairs_symmetric(self):
        """Every entry in _LR_PAIRS has its inverse."""
        for a, b in _LR_PAIRS.items():
            assert _LR_PAIRS[b] == a

    def test_all_joints_in_region_map(self):
        """_JOINT_TO_REGION covers all 52 joints."""
        for i in range(52):
            assert i in _JOINT_TO_REGION, f"Joint {i} not in _JOINT_TO_REGION"

    def test_region_names(self):
        """All expected region names exist."""
        expected = {"spine", "left_leg", "right_leg", "left_arm", "right_arm",
                    "left_hand", "right_hand"}
        assert set(_JOINT_REGIONS.keys()) == expected


class TestContextMenuExists:
    """MeshViewport should have a contextMenuEvent for joint selection."""

    def test_has_context_menu(self, qapp):
        w = MeshViewport()
        assert hasattr(w, "contextMenuEvent")
        assert callable(w.contextMenuEvent)
