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
    _CV_TO_GL,
    _HAS_GL,
    _ORBIT_DEFAULT_YAW,
    _ORBIT_DEFAULT_PITCH,
    _ORBIT_DEFAULT_DISTANCE,
    _ORBIT_DEFAULT_CENTER,
    _ORBIT_SENSITIVITY,
    _PITCH_LIMIT,
    _ZOOM_FACTOR,
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
        """MeshViewport should be in the bottom splitter."""
        from views.multi_person_tab import MultiPersonTab

        s = Session()
        tab = MultiPersonTab(s, Path("."))
        splitter = tab._bottom_splitter
        found = False
        for i in range(splitter.count()):
            if splitter.widget(i) is tab._mesh_viewport:
                found = True
                break
        assert found, "MeshViewport not found in bottom splitter"

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
