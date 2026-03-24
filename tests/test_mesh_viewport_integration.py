"""Headless viewport integration tests for world-grounded multi-person paths."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import PersonTrack
from models.skeleton import SOMA_SKELETON
from views.mesh_viewport import MeshViewport, project_joints_to_screen


def _make_world_smplx_track() -> PersonTrack:
    return PersonTrack(
        person_id=0,
        smplx_params={
            "global_orient": np.zeros((1, 3), dtype=np.float32),
            "body_pose": np.zeros((1, 21, 3), dtype=np.float32),
            "betas": np.zeros((1, 10), dtype=np.float32),
            "transl": np.array([[0.0, 1.0, 2.5]], dtype=np.float32),
            "global_orient_world": np.array([[0.0, np.pi / 6, 0.0]], dtype=np.float32),
            "transl_world": np.array([[0.5, 0.0, -1.5]], dtype=np.float32),
        },
    )


def _make_world_soma_track() -> PersonTrack:
    poses = np.zeros((1, SOMA_SKELETON.n_joints, 3), dtype=np.float32)
    return PersonTrack(
        person_id=0,
        body_model_type="soma",
        soma_params={
            "poses": poses,
            "transl": np.array([[0.0, 1.2, 2.0]], dtype=np.float32),
            "global_orient_world": np.array([[0.0, np.pi / 4, 0.0]], dtype=np.float32),
            "transl_world": np.array([[1.0, 0.0, -2.0]], dtype=np.float32),
        },
    )


class TestMeshViewportHeadlessIntegration:
    """Drive the real widget offscreen and assert its end-to-end state."""

    def test_world_smplx_track_projects_on_screen(self, qapp, session):
        w = MeshViewport()
        w.resize(640, 480)
        session.person_tracks[0] = _make_world_smplx_track()
        w.set_session(session)
        w.set_person(0)
        w.set_camera_mode("orbit")
        w.on_frame_changed(0)

        assert w._data_is_global is True
        assert w._joint_positions is not None
        assert w._grid_y == 0.0

        mvp = w._projection @ w._view @ w._model_mat
        screen = project_joints_to_screen(w._joint_positions[:1], mvp, w.width(), w.height())
        assert 0.0 <= screen[0, 0] <= w.width()
        assert 0.0 <= screen[0, 1] <= w.height()

    def test_world_soma_track_keeps_grounded_orbit_state(self, qapp, session):
        w = MeshViewport()
        w.resize(800, 600)
        session.person_tracks[0] = _make_world_soma_track()
        w.set_session(session)
        w.set_person(0)
        w.set_camera_mode("orbit")
        w.on_frame_changed(0)

        assert w._data_is_global is True
        assert w._joint_positions is not None
        np.testing.assert_allclose(w._joint_positions[0], [1.0, 0.0, -2.0], atol=1e-6)
        assert w._grid_y == 0.0

        mvp = w._projection @ w._view @ w._model_mat
        screen = project_joints_to_screen(w._joint_positions[:2], mvp, w.width(), w.height())
        assert np.all(np.isfinite(screen))
        assert np.all(screen[:, 0] >= -1.0)
        assert np.all(screen[:, 0] <= w.width() + 1.0)
        assert np.all(screen[:, 1] >= -1.0)
        assert np.all(screen[:, 1] <= w.height() + 1.0)
