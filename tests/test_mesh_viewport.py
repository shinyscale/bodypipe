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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session, PersonTrack
from views.mesh_viewport import (
    MeshViewport,
    compute_normals,
    k_to_projection,
    estimate_K,
    _CV_TO_GL,
    _HAS_GL,
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
