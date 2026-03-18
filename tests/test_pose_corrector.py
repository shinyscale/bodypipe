"""Tests for the Pose Corrector panel (Phase 3.4).

Tests cover:
 - Pure helper functions (get_joint_euler, _get_joint_axis_angle) which
   extract rotation values from session params and corrections.
 - PoseCorrectorPanel widget construction, sub-widget presence, signals.
 - Joint selector: dropdown population, viewport click → dropdown sync.
 - Euler slider: value updates on joint change, slider ↔ spinbox sync.
 - Preview: euler change → pose override on MeshViewport.
 - Apply: correction committed to CorrectionTrack + raw params updated.
 - Reset: joint reset clears single joint, reset all clears entire frame.
 - Person selector: dropdown population from session, person change updates.
 - Integration with MultiPersonTab: PoseCorrectorPanel replaces bare viewport.

Why: The pose corrector is the primary tool for fixing bad GVHMR poses.
These tests verify that joint selection, euler editing, real-time preview,
and correction persistence all work correctly through the Qt signal chain.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session, PersonTrack
from views.mesh_viewport import JOINT_NAMES, JOINT_PARENTS, MeshViewport
from views.pose_corrector_panel import (
    PoseCorrectorPanel,
    get_joint_euler,
    _get_joint_axis_angle,
    _safe_import_pose_correction,
    axis_angle_to_euler_deg_fallback,
    euler_deg_to_axis_angle_fallback,
    _N_BODY_JOINTS,
)


# ======================================================================
# Helper: make a session with mock SMPL-X params
# ======================================================================

def _make_session_with_params(n_frames=100, n_persons=2):
    """Create a session with synthetic SMPL-X params for testing."""
    session = Session(num_frames=n_frames, fps=30.0, img_width=640, img_height=480)
    for pid in range(n_persons):
        params = {
            "global_orient": np.zeros((n_frames, 3), dtype=np.float32),
            "body_pose": np.zeros((n_frames, 21, 3), dtype=np.float32),
            "betas": np.zeros((1, 10), dtype=np.float32),
            "transl": np.zeros((n_frames, 3), dtype=np.float32),
        }
        # Set some non-zero values for testing
        params["global_orient"][0] = [0.1, 0.2, 0.3]
        params["body_pose"][0, 0] = [0.4, 0.5, 0.6]  # L_Hip (joint 1, bp idx 0)
        params["body_pose"][0, 3] = [0.7, 0.8, 0.9]  # L_Knee (joint 4, bp idx 3)

        track = PersonTrack(person_id=pid, smplx_params=params)
        session.person_tracks[pid] = track
    return session


# ======================================================================
# Pure function tests (no Qt required)
# ======================================================================


class TestAxisAngleEulerFallback:
    """Fallback rotation conversion when GVHMR backend is unavailable."""

    def test_zero_roundtrip(self):
        euler = axis_angle_to_euler_deg_fallback(np.zeros(3))
        np.testing.assert_allclose(euler, 0.0, atol=1e-5)

    def test_nonzero_roundtrip(self):
        """axis_angle → euler → axis_angle should roundtrip."""
        aa_orig = np.array([0.5, -0.3, 0.8], dtype=np.float32)
        euler = axis_angle_to_euler_deg_fallback(aa_orig)
        aa_back = euler_deg_to_axis_angle_fallback(euler)
        np.testing.assert_allclose(aa_back, aa_orig, atol=1e-4)

    def test_output_shape(self):
        euler = axis_angle_to_euler_deg_fallback(np.array([0.1, 0.2, 0.3]))
        assert euler.shape == (3,)
        assert euler.dtype == np.float32

    def test_euler_to_aa_output_shape(self):
        aa = euler_deg_to_axis_angle_fallback(np.array([10.0, 20.0, 30.0]))
        assert aa.shape == (3,)
        assert aa.dtype == np.float32


class TestGetJointAxisAngle:
    """_get_joint_axis_angle: extract rotation from session params."""

    def test_global_orient_joint0(self):
        session = _make_session_with_params()
        aa = _get_joint_axis_angle(session, 0, 0, 0)
        np.testing.assert_allclose(aa, [0.1, 0.2, 0.3], atol=1e-6)

    def test_body_joint_l_hip(self):
        """Joint 1 (L_Hip) maps to body_pose index 0."""
        session = _make_session_with_params()
        aa = _get_joint_axis_angle(session, 0, 0, 1)
        np.testing.assert_allclose(aa, [0.4, 0.5, 0.6], atol=1e-6)

    def test_body_joint_l_knee(self):
        """Joint 4 (L_Knee) maps to body_pose index 3."""
        session = _make_session_with_params()
        aa = _get_joint_axis_angle(session, 0, 0, 4)
        np.testing.assert_allclose(aa, [0.7, 0.8, 0.9], atol=1e-6)

    def test_zero_frame(self):
        """Non-zero frame should return zeros for our test data."""
        session = _make_session_with_params()
        aa = _get_joint_axis_angle(session, 0, 50, 0)
        np.testing.assert_allclose(aa, 0.0, atol=1e-6)

    def test_no_session(self):
        aa = _get_joint_axis_angle(None, 0, 0, 0)
        np.testing.assert_allclose(aa, 0.0)

    def test_no_person(self):
        session = _make_session_with_params()
        aa = _get_joint_axis_angle(session, 99, 0, 0)
        np.testing.assert_allclose(aa, 0.0)

    def test_no_params(self):
        session = Session()
        session.person_tracks[0] = PersonTrack(person_id=0)
        aa = _get_joint_axis_angle(session, 0, 0, 0)
        np.testing.assert_allclose(aa, 0.0)

    def test_correction_takes_precedence(self):
        """When a committed correction exists, it should override raw params."""
        session = _make_session_with_params()
        # Create a mock CorrectionTrack
        mock_ct = MagicMock()
        mock_corr = MagicMock()
        mock_corr.global_orient = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mock_corr.body_pose = None
        mock_ct.get_correction.return_value = mock_corr
        session.correction_tracks[0] = mock_ct

        aa = _get_joint_axis_angle(session, 0, 0, 0)
        np.testing.assert_allclose(aa, [1.0, 2.0, 3.0], atol=1e-6)

    def test_body_correction_takes_precedence(self):
        """Body pose correction should override raw params for that joint."""
        session = _make_session_with_params()
        mock_ct = MagicMock()
        mock_corr = MagicMock()
        mock_corr.global_orient = None
        mock_corr.body_pose = {0: np.array([9.0, 8.0, 7.0], dtype=np.float32)}
        mock_ct.get_correction.return_value = mock_corr
        session.correction_tracks[0] = mock_ct

        # Joint 1 (L_Hip) → body_pose index 0
        aa = _get_joint_axis_angle(session, 0, 0, 1)
        np.testing.assert_allclose(aa, [9.0, 8.0, 7.0], atol=1e-6)

    def test_flat_body_pose(self):
        """body_pose stored as (N, 63) instead of (N, 21, 3)."""
        session = _make_session_with_params()
        bp = session.person_tracks[0].smplx_params["body_pose"]
        # Flatten to (N, 63)
        session.person_tracks[0].smplx_params["body_pose"] = bp.reshape(bp.shape[0], -1)
        aa = _get_joint_axis_angle(session, 0, 0, 1)
        np.testing.assert_allclose(aa, [0.4, 0.5, 0.6], atol=1e-6)


class TestGetJointEuler:
    """get_joint_euler: full pipeline from session to euler degrees."""

    def test_zero_returns_zero(self):
        session = _make_session_with_params()
        euler = get_joint_euler(session, 0, 50, 0)
        np.testing.assert_allclose(euler, 0.0, atol=1e-5)

    def test_nonzero_roundtrip(self):
        """Setting params and reading euler should be consistent."""
        session = _make_session_with_params()
        euler = get_joint_euler(session, 0, 0, 0)
        assert euler.shape == (3,)
        # The euler should convert back to approximately [0.1, 0.2, 0.3]
        aa_back = euler_deg_to_axis_angle_fallback(euler)
        np.testing.assert_allclose(aa_back, [0.1, 0.2, 0.3], atol=1e-4)


class TestSafeImport:
    """_safe_import_pose_correction: graceful backend import."""

    def test_returns_tuple(self):
        result = _safe_import_pose_correction()
        assert isinstance(result, tuple)
        assert len(result) == 3


# ======================================================================
# Widget tests (require QApplication from conftest)
# ======================================================================


class TestPoseCorrectorPanelConstruction:
    """PoseCorrectorPanel widget construction and sub-widget presence."""

    def test_construction(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel is not None

    def test_has_mesh_viewport(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert isinstance(panel.mesh_viewport, MeshViewport)

    def test_has_person_combo(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._person_combo is not None

    def test_has_joint_combo(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._joint_combo is not None
        # Should have at least body joints
        assert panel._joint_combo.count() >= _N_BODY_JOINTS

    def test_has_euler_spinboxes(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._euler_x is not None
        assert panel._euler_y is not None
        assert panel._euler_z is not None

    def test_has_sliders(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._slider_x is not None
        assert panel._slider_y is not None
        assert panel._slider_z is not None

    def test_has_action_buttons(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._apply_btn is not None
        assert panel._reset_joint_btn is not None
        assert panel._reset_all_btn is not None

    def test_has_camera_combo(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._camera_combo is not None
        assert panel._camera_combo.count() == 2  # In-camera, Free orbit

    def test_has_color_combo(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._color_combo is not None
        assert panel._color_combo.count() == 3  # Solid, Joint influence, Confidence

    def test_signals_exist(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "joint_selected")
        assert hasattr(panel, "correction_applied")

    def test_euler_range(self, qapp):
        """Euler spinboxes should have -180 to +180 range."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        for spin in [panel._euler_x, panel._euler_y, panel._euler_z]:
            assert spin.minimum() == -180.0
            assert spin.maximum() == 180.0
            assert spin.singleStep() == 0.1

    def test_slider_range(self, qapp):
        """Sliders should have -180 to +180 range (integer)."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        for s in [panel._slider_x, panel._slider_y, panel._slider_z]:
            assert s.minimum() == -180
            assert s.maximum() == 180


class TestJointDropdown:
    """Joint dropdown: population, selection, viewport click sync."""

    def test_body_joints_present(self, qapp):
        """All 22 body joints should be in the dropdown."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        body_items = []
        for i in range(panel._joint_combo.count()):
            data = panel._joint_combo.itemData(i)
            if data is not None and data < _N_BODY_JOINTS:
                body_items.append(data)
        assert len(body_items) == _N_BODY_JOINTS

    def test_hand_joints_present(self, qapp):
        """Hand joints (22-51) should be in the dropdown."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        hand_items = []
        for i in range(panel._joint_combo.count()):
            data = panel._joint_combo.itemData(i)
            if data is not None and data >= _N_BODY_JOINTS:
                hand_items.append(data)
        assert len(hand_items) == len(JOINT_NAMES) - _N_BODY_JOINTS

    def test_joint_selection_updates_current(self, qapp):
        """Selecting a joint in dropdown should update _current_joint."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        # Select joint 4 (L_Knee)
        for i in range(panel._joint_combo.count()):
            if panel._joint_combo.itemData(i) == 4:
                panel._joint_combo.setCurrentIndex(i)
                break
        assert panel._current_joint == 4

    def test_viewport_click_syncs_dropdown(self, qapp):
        """Viewport joint_clicked signal should sync the dropdown."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        # Simulate viewport click on joint 3 (Spine1)
        panel._on_joint_clicked(3)
        assert panel._joint_combo.itemData(panel._joint_combo.currentIndex()) == 3

    def test_joint_selection_emits_signal(self, qapp):
        """Joint selection should emit joint_selected signal."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        received = []
        panel.joint_selected.connect(received.append)
        for i in range(panel._joint_combo.count()):
            if panel._joint_combo.itemData(i) == 5:
                panel._joint_combo.setCurrentIndex(i)
                break
        assert 5 in received


class TestJointInfo:
    """Joint info label updates on selection."""

    def test_pelvis_info(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_joint = 0
        panel._update_joint_info()
        assert "Pelvis" in panel._joint_info.text()
        assert "root" in panel._joint_info.text()

    def test_l_knee_info(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_joint = 4
        panel._update_joint_info()
        assert "L_Knee" in panel._joint_info.text()
        assert "L_Hip" in panel._joint_info.text()

    def test_no_joint_clears_info(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_joint = -1
        panel._update_joint_info()
        assert panel._joint_info.text() == ""


class TestEulerSliders:
    """Euler sliders: value updates, slider↔spinbox sync."""

    def test_sliders_update_on_joint_selection(self, qapp):
        """Selecting a joint with non-zero rotation should update sliders."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0
        panel._current_joint = 0
        panel._update_sliders()
        # global_orient = [0.1, 0.2, 0.3] → should have non-zero euler values
        assert panel._euler_x.value() != 0.0 or panel._euler_y.value() != 0.0 or panel._euler_z.value() != 0.0

    def test_sliders_zero_for_zero_joint(self, qapp):
        """Zero rotation should give zero euler values."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 50  # frame 50 has zero rotation
        panel._current_joint = 0
        panel._update_sliders()
        assert abs(panel._euler_x.value()) < 0.1
        assert abs(panel._euler_y.value()) < 0.1
        assert abs(panel._euler_z.value()) < 0.1

    def test_spinbox_slider_sync(self, qapp):
        """Changing spinbox should update slider (integer approximation)."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_joint = 0
        panel._euler_x.setValue(45.5)
        # Slider should be at 45 (integer truncation from the sync handler)
        # Note: the sync happens via _on_euler_changed which updates sliders
        assert abs(panel._slider_x.value() - 45) <= 1

    def test_slider_spinbox_sync(self, qapp):
        """Changing slider should update spinbox."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_joint = 0
        panel._slider_x.setValue(30)
        assert abs(panel._euler_x.value() - 30.0) < 1.0


class TestPreview:
    """Euler changes should set a pose override on MeshViewport."""

    def test_euler_change_sets_override(self, qapp):
        """Changing euler values should set a pose override for preview."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0
        panel._current_joint = 1  # L_Hip (body joint, bp idx 0)

        panel._euler_x.setValue(15.0)
        # Trigger preview manually (spinbox setValue triggers _on_euler_changed)
        # The viewport should now have a pose override
        override = panel._viewport._pose_override
        assert override is not None
        assert override["frame_idx"] == 0
        assert "body_pose" in override
        assert 0 in override["body_pose"]  # bp index 0 for joint 1

    def test_global_orient_override(self, qapp):
        """Joint 0 should set global_orient override, not body_pose."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0
        panel._current_joint = 0  # Pelvis (global orient)

        panel._euler_x.setValue(10.0)
        override = panel._viewport._pose_override
        assert override is not None
        assert "global_orient" in override
        assert "body_pose" not in override

    def test_no_override_without_selection(self, qapp):
        """No override should be set without person/joint selection."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = -1
        panel._current_joint = -1
        panel._euler_x.setValue(10.0)
        assert panel._viewport._pose_override is None

    def test_frame_change_clears_override(self, qapp):
        """Changing frames should clear the pose override."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_joint = 1
        panel._euler_x.setValue(15.0)
        assert panel._viewport._pose_override is not None

        panel.on_frame_changed(10)
        assert panel._viewport._pose_override is None


class TestApply:
    """Apply button: commit correction to CorrectionTrack."""

    def test_apply_creates_correction_track(self, qapp):
        """Apply should create CorrectionTrack if none exists."""
        aa_to_euler, euler_to_aa, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0
        panel._current_joint = 1  # L_Hip

        panel._euler_x.setValue(15.0)
        panel._on_apply()

        assert 0 in session.correction_tracks
        ct = session.correction_tracks[0]
        assert len(ct.corrections) == 1
        corr = ct.corrections[0]
        assert corr.frame_index == 0
        assert corr.body_pose is not None
        assert 0 in corr.body_pose  # bp idx 0 for joint 1

    def test_apply_global_orient(self, qapp):
        """Apply on joint 0 should set global_orient correction."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 5
        panel._current_joint = 0

        panel._euler_x.setValue(90.0)
        panel._on_apply()

        ct = session.correction_tracks[0]
        corr = ct.get_correction(5)
        assert corr is not None
        assert corr.global_orient is not None
        assert corr.correction_type == "global_orient"

    def test_apply_emits_signal(self, qapp):
        """Apply should emit correction_applied signal."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0
        panel._current_joint = 1

        received = []
        panel.correction_applied.connect(lambda p, f: received.append((p, f)))
        panel._euler_x.setValue(15.0)
        panel._on_apply()

        assert (0, 0) in received

    def test_apply_clears_override(self, qapp):
        """After apply, the pose override should be cleared."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0
        panel._current_joint = 1

        panel._euler_x.setValue(15.0)
        assert panel._viewport._pose_override is not None
        panel._on_apply()
        assert panel._viewport._pose_override is None

    def test_apply_merges_corrections(self, qapp):
        """Applying corrections for different joints at same frame should merge."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0

        # Apply joint 1
        panel._current_joint = 1
        panel._euler_x.setValue(15.0)
        panel._on_apply()

        # Apply joint 4 at same frame
        panel._current_joint = 4
        panel._euler_y.setValue(30.0)
        panel._on_apply()

        ct = session.correction_tracks[0]
        corr = ct.get_correction(0)
        assert corr is not None
        assert corr.body_pose is not None
        # Both joints should be in the correction
        assert 0 in corr.body_pose  # joint 1 → bp idx 0
        assert 3 in corr.body_pose  # joint 4 → bp idx 3

    def test_apply_updates_raw_params(self, qapp):
        """Apply should update the raw smplx_params in session."""
        _, euler_to_aa, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0
        panel._current_joint = 0  # global orient

        panel._euler_x.setValue(45.0)
        panel._euler_y.setValue(0.0)
        panel._euler_z.setValue(0.0)
        panel._on_apply()

        # The raw params should be updated
        go = session.person_tracks[0].smplx_params["global_orient"]
        # Should no longer be the original [0.1, 0.2, 0.3]
        assert not np.allclose(go[0], [0.1, 0.2, 0.3], atol=1e-3)

    def test_no_apply_without_selection(self, qapp):
        """Apply with no joint/person selected should be a no-op."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = -1
        panel._current_joint = -1
        panel._on_apply()
        assert len(session.correction_tracks) == 0


class TestResetJoint:
    """Reset Joint: clear single joint from correction."""

    def test_reset_joint_removes_single(self, qapp):
        """Reset joint should remove only that joint from the correction."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0

        # Apply two joints
        panel._current_joint = 1
        panel._euler_x.setValue(15.0)
        panel._on_apply()

        panel._current_joint = 4
        panel._euler_y.setValue(30.0)
        panel._on_apply()

        # Reset joint 1
        panel._current_joint = 1
        panel._on_reset_joint()

        ct = session.correction_tracks[0]
        corr = ct.get_correction(0)
        assert corr is not None
        # Joint 1 (bp idx 0) should be gone, joint 4 (bp idx 3) should remain
        assert corr.body_pose is not None
        assert 0 not in corr.body_pose
        assert 3 in corr.body_pose

    def test_reset_last_joint_removes_correction(self, qapp):
        """Resetting the only joint should remove the entire correction."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0

        panel._current_joint = 1
        panel._euler_x.setValue(15.0)
        panel._on_apply()

        panel._on_reset_joint()

        ct = session.correction_tracks[0]
        assert ct.get_correction(0) is None


class TestResetAll:
    """Reset All: remove all corrections at current frame."""

    def test_reset_all_removes_frame_correction(self, qapp):
        """Reset all should remove the correction at the current frame."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0

        # Apply corrections
        panel._current_joint = 0
        panel._euler_x.setValue(45.0)
        panel._on_apply()
        panel._current_joint = 1
        panel._euler_y.setValue(30.0)
        panel._on_apply()

        # Reset all
        panel._on_reset_all()

        ct = session.correction_tracks[0]
        assert ct.get_correction(0) is None

    def test_reset_all_preserves_other_frames(self, qapp):
        """Reset all at one frame should not affect corrections at other frames."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0

        # Correction at frame 0
        panel._current_frame = 0
        panel._current_joint = 1
        panel._euler_x.setValue(15.0)
        panel._on_apply()

        # Correction at frame 10
        panel.on_frame_changed(10)
        panel._current_joint = 1
        panel._euler_x.setValue(30.0)
        panel._on_apply()

        # Reset all at frame 0
        panel.on_frame_changed(0)
        panel._on_reset_all()

        ct = session.correction_tracks[0]
        assert ct.get_correction(0) is None
        assert ct.get_correction(10) is not None


class TestPersonSelector:
    """Person dropdown: population from session, person change."""

    def test_refresh_populates_persons(self, qapp):
        """Refresh should populate dropdown from session person_tracks."""
        session = _make_session_with_params(n_persons=3)
        panel = PoseCorrectorPanel(session=session)
        panel.refresh()
        assert panel._person_combo.count() == 3

    def test_inactive_tracks_excluded(self, qapp):
        """Inactive tracks should not appear in the dropdown."""
        session = _make_session_with_params(n_persons=3)
        session.inactive_tracks.add(1)
        panel = PoseCorrectorPanel(session=session)
        panel.refresh()
        assert panel._person_combo.count() == 2

    def test_set_person_syncs_dropdown(self, qapp):
        """External set_person should sync the dropdown."""
        session = _make_session_with_params(n_persons=3)
        panel = PoseCorrectorPanel(session=session)
        panel.refresh()
        panel.set_person(1)
        assert panel._person_combo.itemData(panel._person_combo.currentIndex()) == 1

    def test_set_person_updates_viewport(self, qapp):
        """set_person should propagate to the viewport."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel.set_person(1)
        assert panel._viewport._person_id == 1


class TestFrameChange:
    """Frame changes: slider updates, override clearing."""

    def test_on_frame_changed_updates_viewport(self, qapp):
        """on_frame_changed should propagate to viewport."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel.on_frame_changed(42)
        assert panel._viewport._current_frame == 42
        assert panel._current_frame == 42


# ======================================================================
# MeshViewport pose override tests
# ======================================================================


class TestMeshViewportPoseOverride:
    """MeshViewport.set_pose_override: override management."""

    def test_set_override(self, qapp):
        w = MeshViewport()
        override = {"frame_idx": 0, "global_orient": np.array([1, 0, 0], dtype=np.float32)}
        w.set_pose_override(override)
        assert w._pose_override is override

    def test_clear_override(self, qapp):
        w = MeshViewport()
        w.set_pose_override({"frame_idx": 0})
        w.set_pose_override(None)
        assert w._pose_override is None

    def test_invalidate_cache_specific(self, qapp):
        """invalidate_cache with person+frame should remove just that entry."""
        w = MeshViewport()
        w._vertex_cache[(0, 0)] = (np.zeros((3, 3)), np.zeros((3, 3)))
        w._vertex_cache[(0, 1)] = (np.zeros((3, 3)), np.zeros((3, 3)))
        w.invalidate_cache(0, 0)
        assert (0, 0) not in w._vertex_cache
        assert (0, 1) in w._vertex_cache

    def test_invalidate_cache_all(self, qapp):
        """invalidate_cache without args should clear everything."""
        w = MeshViewport()
        w._vertex_cache[(0, 0)] = (np.zeros((3, 3)), np.zeros((3, 3)))
        w._vertex_cache[(1, 5)] = (np.zeros((3, 3)), np.zeros((3, 3)))
        w.invalidate_cache()
        assert len(w._vertex_cache) == 0

    def test_override_invalidates_cache_on_set(self, qapp):
        """Setting an override should invalidate the cache for that frame."""
        w = MeshViewport()
        w._person_id = 0
        w._vertex_cache[(0, 5)] = (np.zeros((3, 3)), np.zeros((3, 3)))
        w.set_pose_override({"frame_idx": 5})
        assert (0, 5) not in w._vertex_cache

    def test_apply_override_to_params_global_orient(self, qapp):
        """_apply_override_to_params should override global_orient."""
        w = MeshViewport()
        params = {
            "global_orient": np.zeros((10, 3), dtype=np.float32),
            "body_pose": np.zeros((10, 21, 3), dtype=np.float32),
        }
        override_aa = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        w._pose_override = {"frame_idx": 0, "global_orient": override_aa}
        w._current_frame = 0

        result = w._apply_override_to_params(params, 0)
        np.testing.assert_allclose(result["global_orient"][0], [1.0, 2.0, 3.0])
        # Original should be unchanged
        np.testing.assert_allclose(params["global_orient"][0], [0.0, 0.0, 0.0])

    def test_apply_override_to_params_body_pose(self, qapp):
        """_apply_override_to_params should override specific body joints."""
        w = MeshViewport()
        params = {
            "global_orient": np.zeros((10, 3), dtype=np.float32),
            "body_pose": np.zeros((10, 21, 3), dtype=np.float32),
        }
        override_aa = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        w._pose_override = {"frame_idx": 0, "body_pose": {3: override_aa}}
        w._current_frame = 0

        result = w._apply_override_to_params(params, 0)
        np.testing.assert_allclose(result["body_pose"][0, 3], [0.5, 0.6, 0.7])
        # Other joints unchanged
        np.testing.assert_allclose(result["body_pose"][0, 0], [0.0, 0.0, 0.0])
        # Original unchanged
        np.testing.assert_allclose(params["body_pose"][0, 3], [0.0, 0.0, 0.0])

    def test_apply_override_wrong_frame(self, qapp):
        """Override should not apply to a different frame."""
        w = MeshViewport()
        params = {
            "global_orient": np.zeros((10, 3), dtype=np.float32),
        }
        w._pose_override = {"frame_idx": 5, "global_orient": np.ones(3)}
        w._current_frame = 5

        result = w._apply_override_to_params(params, 0)
        assert result is params  # should return unchanged

    def test_apply_override_flat_body_pose(self, qapp):
        """Override should work with flat body_pose (N, 63)."""
        w = MeshViewport()
        params = {
            "global_orient": np.zeros((10, 3), dtype=np.float32),
            "body_pose": np.zeros((10, 63), dtype=np.float32),
        }
        override_aa = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        w._pose_override = {"frame_idx": 0, "body_pose": {5: override_aa}}
        w._current_frame = 0

        result = w._apply_override_to_params(params, 0)
        # After reshape to (10, 21, 3), index 5 should have the override
        bp = result["body_pose"]
        assert bp.shape == (10, 21, 3)
        np.testing.assert_allclose(bp[0, 5], [1.0, 2.0, 3.0])


# ======================================================================
# MultiPersonTab integration
# ======================================================================


class TestMultiPersonTabIntegration:
    """PoseCorrectorPanel should be wired into MultiPersonTab."""

    def test_multi_person_tab_has_pose_corrector(self, qapp, session):
        """MultiPersonTab should use PoseCorrectorPanel, not bare MeshViewport."""
        from views.multi_person_tab import MultiPersonTab
        tab = MultiPersonTab(session=session, gvhmr_root=Path("/tmp/gvhmr"))
        assert hasattr(tab, "_pose_corrector")
        assert isinstance(tab._pose_corrector, PoseCorrectorPanel)

    def test_mesh_viewport_accessible(self, qapp, session):
        """The mesh viewport should still be accessible via _mesh_viewport."""
        from views.multi_person_tab import MultiPersonTab
        tab = MultiPersonTab(session=session, gvhmr_root=Path("/tmp/gvhmr"))
        assert tab._mesh_viewport is tab._pose_corrector.mesh_viewport
