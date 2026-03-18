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
    _safe_import_quick_fix,
    _safe_import_space_override,
    _safe_import_bvh_export,
    _safe_import_fbx_export,
    axis_angle_to_euler_deg_fallback,
    euler_deg_to_axis_angle_fallback,
    flip_global_orient_fallback,
    mirror_lr_pose_fallback,
    _N_BODY_JOINTS,
    _LR_SWAP_PAIRS,
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

    def test_frame_requested_wired(self, qapp, session):
        """Pose corrector frame_requested should be connected to video player."""
        from views.multi_person_tab import MultiPersonTab
        tab = MultiPersonTab(session=session, gvhmr_root=Path("/tmp/gvhmr"))
        # Verify signal exists and is connected
        assert hasattr(tab._pose_corrector, "frame_requested")
        # Emit should not raise (connected to video_player.seek)
        tab._pose_corrector.frame_requested.emit(0)


# ======================================================================
# Phase 3.5: Quick-fix button + corrections table tests
# ======================================================================


class TestQuickFixButtonPresence:
    """Quick-fix buttons should be present in the panel."""

    def test_has_flip_btn(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "_flip_btn")
        assert panel._flip_btn.text() == "Flip Body"

    def test_has_invert_btn(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "_invert_btn")
        assert panel._invert_btn.text() == "Invert"

    def test_has_mirror_btn(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "_mirror_btn")
        assert panel._mirror_btn.text() == "Mirror L/R"

    def test_has_copy_from_btn(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "_copy_from_btn")
        assert panel._copy_from_btn.text() == "Copy From"

    def test_has_frame_requested_signal(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "frame_requested")


class TestCorrectionsTablePresence:
    """Corrections table should be present and configured correctly."""

    def test_has_corrections_table(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "_corrections_table")

    def test_table_columns(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._corrections_table.columnCount() == 5

    def test_table_headers(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        headers = []
        for col in range(panel._corrections_table.columnCount()):
            item = panel._corrections_table.horizontalHeaderItem(col)
            if item:
                headers.append(item.text())
        assert headers == ["Frame", "Type", "Joint", "Go", "Del"]

    def test_table_initially_empty(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._corrections_table.rowCount() == 0


# ======================================================================
# Quick-fix fallback function tests (pure, no Qt)
# ======================================================================


class TestFlipGlobalOrientFallback:
    """flip_global_orient_fallback: 180° rotation of global_orient."""

    def test_yaw_flip(self):
        params = {
            "global_orient": np.zeros((10, 3), dtype=np.float32),
        }
        result = flip_global_orient_fallback(params, 0, "yaw")
        assert result.shape == (3,)
        assert result.dtype == np.float32
        # 180° yaw on identity → should have non-trivial rotation
        # (pi * [0,1,0] = [0, pi, 0] in axis-angle)
        assert np.linalg.norm(result) > 2.0  # ~pi

    def test_pitch_flip(self):
        params = {
            "global_orient": np.zeros((10, 3), dtype=np.float32),
        }
        result = flip_global_orient_fallback(params, 0, "pitch")
        assert result.shape == (3,)
        assert np.linalg.norm(result) > 2.0

    def test_roll_flip(self):
        params = {
            "global_orient": np.zeros((10, 3), dtype=np.float32),
        }
        result = flip_global_orient_fallback(params, 0, "roll")
        assert result.shape == (3,)
        assert np.linalg.norm(result) > 2.0

    def test_nonzero_input(self):
        params = {
            "global_orient": np.array([[0.5, 0.3, -0.2]], dtype=np.float32),
        }
        result = flip_global_orient_fallback(params, 0, "yaw")
        assert result.shape == (3,)
        # Should differ from original
        assert not np.allclose(result, params["global_orient"][0], atol=0.1)

    def test_double_flip_returns_to_original(self):
        """Flipping twice should return to approximately the original pose."""
        orig = np.array([0.5, 0.3, -0.2], dtype=np.float32)
        params = {"global_orient": orig.reshape(1, 3)}
        first = flip_global_orient_fallback(params, 0, "yaw")
        params2 = {"global_orient": first.reshape(1, 3)}
        second = flip_global_orient_fallback(params2, 0, "yaw")
        np.testing.assert_allclose(second, orig, atol=1e-4)


class TestMirrorLrPoseFallback:
    """mirror_lr_pose_fallback: swap left/right body joints."""

    def test_basic_mirror(self):
        params = {
            "body_pose": np.zeros((10, 21, 3), dtype=np.float32),
        }
        # Set L_Hip (idx 0) and R_Hip (idx 1) to different values
        params["body_pose"][0, 0] = [1.0, 0.0, 0.0]  # L_Hip (joint 1, bp 0)
        params["body_pose"][0, 1] = [0.0, 1.0, 0.0]  # R_Hip (joint 2, bp 1)

        mirrored = mirror_lr_pose_fallback(params, 0)
        # L_Hip should have R_Hip's value and vice versa
        np.testing.assert_allclose(mirrored[0], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(mirrored[1], [1.0, 0.0, 0.0])

    def test_all_pairs_swapped(self):
        """All L/R swap pairs should be present in the result."""
        params = {
            "body_pose": np.random.randn(10, 21, 3).astype(np.float32),
        }
        mirrored = mirror_lr_pose_fallback(params, 0)
        for l_idx, r_idx in _LR_SWAP_PAIRS:
            l_bp, r_bp = l_idx - 1, r_idx - 1
            assert l_bp in mirrored
            assert r_bp in mirrored

    def test_returns_sparse_dict(self):
        params = {
            "body_pose": np.zeros((10, 21, 3), dtype=np.float32),
        }
        mirrored = mirror_lr_pose_fallback(params, 0)
        assert isinstance(mirrored, dict)
        # Should have 16 entries (8 pairs × 2)
        assert len(mirrored) == 16

    def test_flat_body_pose(self):
        """Should handle (N, 63) shaped body_pose."""
        params = {
            "body_pose": np.zeros((10, 63), dtype=np.float32),
        }
        params["body_pose"][0, 0:3] = [1.0, 0.0, 0.0]  # bp idx 0
        params["body_pose"][0, 3:6] = [0.0, 1.0, 0.0]  # bp idx 1
        mirrored = mirror_lr_pose_fallback(params, 0)
        assert len(mirrored) > 0
        np.testing.assert_allclose(mirrored[0], [0.0, 1.0, 0.0])


class TestSafeImportQuickFix:
    """_safe_import_quick_fix: graceful backend import."""

    def test_returns_tuple(self):
        result = _safe_import_quick_fix()
        assert isinstance(result, tuple)
        assert len(result) == 3


# ======================================================================
# Quick-fix handler tests (require QApplication + backend)
# ======================================================================


class TestFlipBodyHandler:
    """_on_flip_body: handler for flip body button."""

    def test_flip_creates_correction(self, qapp):
        """Flip body should create a global_orient correction."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0

        # Mock the QInputDialog to avoid modal dialog
        with patch(
            "views.pose_corrector_panel.QInputDialog.getItem",
            return_value=("Yaw (Y-axis)", True),
        ):
            panel._on_flip_body()

        assert 0 in session.correction_tracks
        ct = session.correction_tracks[0]
        corr = ct.get_correction(0)
        assert corr is not None
        assert corr.global_orient is not None
        assert corr.correction_type == "flip"

    def test_flip_emits_signal(self, qapp):
        """Flip should emit correction_applied signal."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 5

        received = []
        panel.correction_applied.connect(lambda p, f: received.append((p, f)))

        with patch(
            "views.pose_corrector_panel.QInputDialog.getItem",
            return_value=("Pitch (X-axis)", True),
        ):
            panel._on_flip_body()

        assert (0, 5) in received

    def test_flip_cancelled(self, qapp):
        """Cancelling the dialog should not create a correction."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0

        with patch(
            "views.pose_corrector_panel.QInputDialog.getItem",
            return_value=("Yaw (Y-axis)", False),
        ):
            panel._on_flip_body()

        assert len(session.correction_tracks) == 0

    def test_flip_no_person_noop(self, qapp):
        """Flip with no person selected should be a no-op."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = -1
        panel._on_flip_body()
        assert len(session.correction_tracks) == 0


class TestInvertUprightHandler:
    """_on_invert_upright: handler for invert button."""

    def test_invert_creates_correction(self, qapp):
        """Invert should create a pitch-flip correction."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0

        panel._on_invert_upright()

        assert 0 in session.correction_tracks
        corr = session.correction_tracks[0].get_correction(0)
        assert corr is not None
        assert corr.global_orient is not None
        assert corr.correction_type == "invert"

    def test_invert_no_person_noop(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = -1
        panel._on_invert_upright()
        assert len(session.correction_tracks) == 0


class TestMirrorLrHandler:
    """_on_mirror_lr: handler for mirror L/R button."""

    def test_mirror_creates_body_pose_correction(self, qapp):
        """Mirror should create body_pose corrections for L/R pairs."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0

        panel._on_mirror_lr()

        assert 0 in session.correction_tracks
        corr = session.correction_tracks[0].get_correction(0)
        assert corr is not None
        assert corr.body_pose is not None
        assert corr.correction_type == "mirror"
        # Should have swapped L/R pairs
        assert len(corr.body_pose) >= 2

    def test_mirror_swaps_values(self, qapp):
        """Mirror should swap L_Hip ↔ R_Hip values."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        # Set distinct L/R hip values
        session.person_tracks[0].smplx_params["body_pose"][0, 0] = [1.0, 0.0, 0.0]
        session.person_tracks[0].smplx_params["body_pose"][0, 1] = [0.0, 2.0, 0.0]

        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 0

        panel._on_mirror_lr()

        corr = session.correction_tracks[0].get_correction(0)
        # bp idx 0 (L_Hip) should now have R_Hip's value
        np.testing.assert_allclose(corr.body_pose[0], [0.0, 2.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(corr.body_pose[1], [1.0, 0.0, 0.0], atol=1e-5)

    def test_mirror_no_person_noop(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = -1
        panel._on_mirror_lr()
        assert len(session.correction_tracks) == 0


class TestCopyFromFrameHandler:
    """_on_copy_from_frame: handler for copy from frame button."""

    def test_copy_creates_full_correction(self, qapp):
        """Copy from frame should create correction with global_orient + body_pose."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 10

        with patch(
            "views.pose_corrector_panel.QInputDialog.getInt",
            return_value=(0, True),
        ):
            panel._on_copy_from_frame()

        assert 0 in session.correction_tracks
        corr = session.correction_tracks[0].get_correction(10)
        assert corr is not None
        assert corr.correction_type == "copy_from_frame"
        assert corr.global_orient is not None
        # Should have copied body_pose from frame 0
        assert corr.body_pose is not None
        assert len(corr.body_pose) == 21  # all body joints

    def test_copy_uses_source_frame_values(self, qapp):
        """Copied correction should have the source frame's rotation values."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        # Source frame 0 has global_orient = [0.1, 0.2, 0.3]
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 50

        with patch(
            "views.pose_corrector_panel.QInputDialog.getInt",
            return_value=(0, True),
        ):
            panel._on_copy_from_frame()

        corr = session.correction_tracks[0].get_correction(50)
        np.testing.assert_allclose(corr.global_orient, [0.1, 0.2, 0.3], atol=1e-5)

    def test_copy_cancelled(self, qapp):
        """Cancelling the dialog should not create a correction."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0

        with patch(
            "views.pose_corrector_panel.QInputDialog.getInt",
            return_value=(0, False),
        ):
            panel._on_copy_from_frame()

        assert len(session.correction_tracks) == 0


# ======================================================================
# Corrections table interaction tests
# ======================================================================


class TestCorrectionsTablePopulation:
    """Corrections table should populate from CorrectionTrack."""

    def test_table_populated_after_apply(self, qapp):
        """Table should show corrections after they are applied."""
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

        assert panel._corrections_table.rowCount() == 1
        assert panel._corrections_table.item(0, 0).text() == "0"  # frame

    def test_table_shows_multiple_corrections(self, qapp):
        """Table should show all corrections for the person."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0

        # Apply at frame 0
        panel._current_frame = 0
        panel._current_joint = 1
        panel._euler_x.setValue(15.0)
        panel._on_apply()

        # Apply at frame 10
        panel.on_frame_changed(10)
        panel._current_joint = 4
        panel._euler_y.setValue(30.0)
        panel._on_apply()

        assert panel._corrections_table.rowCount() == 2
        # Should be sorted by frame
        assert panel._corrections_table.item(0, 0).text() == "0"
        assert panel._corrections_table.item(1, 0).text() == "10"

    def test_table_updates_on_person_change(self, qapp):
        """Changing person should refresh the corrections table."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel.refresh()  # populate dropdown

        # Apply correction for person 0
        panel._current_person = 0
        panel._current_frame = 0
        panel._current_joint = 1
        panel._euler_x.setValue(15.0)
        panel._on_apply()
        assert panel._corrections_table.rowCount() == 1

        # Switch to person 1 (no corrections)
        panel.set_person(1)
        assert panel._corrections_table.rowCount() == 0

    def test_table_clears_after_reset_all(self, qapp):
        """Reset all should clear the table when only one correction exists."""
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
        assert panel._corrections_table.rowCount() == 1

        panel._on_reset_all()
        assert panel._corrections_table.rowCount() == 0


class TestCorrectionsTableGo:
    """Corrections table Go button emits frame_requested."""

    def test_go_emits_frame_requested(self, qapp):
        """Go button should emit frame_requested with the frame index."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)

        received = []
        panel.frame_requested.connect(received.append)
        panel._on_correction_go(42)

        assert 42 in received


class TestCorrectionsTableDelete:
    """Corrections table Delete button removes corrections."""

    def test_delete_removes_correction(self, qapp):
        """Delete should remove the correction at the given frame."""
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
        assert panel._corrections_table.rowCount() == 1

        panel._on_correction_delete(0)
        assert panel._corrections_table.rowCount() == 0

        ct = session.correction_tracks[0]
        assert ct.get_correction(0) is None

    def test_delete_preserves_other_corrections(self, qapp):
        """Deleting one correction should not affect others."""
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0

        # Two corrections
        panel._current_frame = 0
        panel._current_joint = 1
        panel._euler_x.setValue(15.0)
        panel._on_apply()

        panel.on_frame_changed(20)
        panel._current_joint = 4
        panel._euler_y.setValue(30.0)
        panel._on_apply()

        assert panel._corrections_table.rowCount() == 2

        panel._on_correction_delete(0)
        assert panel._corrections_table.rowCount() == 1
        assert panel._corrections_table.item(0, 0).text() == "20"


class TestDescribeCorrectionJoints:
    """_describe_correction_joints: readable joint description."""

    def test_global_orient_only(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        corr = MagicMock()
        corr.global_orient = np.array([1, 0, 0])
        corr.body_pose = None
        result = panel._describe_correction_joints(corr)
        assert "Global" in result

    def test_body_pose_joints(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        corr = MagicMock()
        corr.global_orient = None
        corr.body_pose = {0: np.array([1, 0, 0])}  # bp idx 0 → joint 1 = L_Hip
        result = panel._describe_correction_joints(corr)
        assert "L_Hip" in result

    def test_truncation_with_many_joints(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        corr = MagicMock()
        corr.global_orient = np.array([1, 0, 0])
        corr.body_pose = {i: np.zeros(3) for i in range(10)}
        result = panel._describe_correction_joints(corr)
        assert "…" in result

    def test_empty_correction(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        corr = MagicMock()
        corr.global_orient = None
        corr.body_pose = None
        result = panel._describe_correction_joints(corr)
        assert "—" in result


class TestQuickFixTableIntegration:
    """Quick-fix operations should update the corrections table."""

    def test_invert_updates_table(self, qapp):
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 5

        panel._on_invert_upright()
        assert panel._corrections_table.rowCount() == 1
        assert panel._corrections_table.item(0, 0).text() == "5"
        assert panel._corrections_table.item(0, 1).text() == "invert"

    def test_mirror_updates_table(self, qapp):
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._current_frame = 3

        panel._on_mirror_lr()
        assert panel._corrections_table.rowCount() == 1
        assert panel._corrections_table.item(0, 1).text() == "mirror"


# ======================================================================
# Phase 3.6: Space overrides + BVH/FBX export
#
# Why: Space overrides control per-frame coordinate space (world/camera/
# carried) for complex multi-person scenarios like lifts. BVH/FBX export
# applies all corrections and space overrides to produce clean output.
# These tests verify the UI CRUD operations, table rendering, backend
# wiring, and export signal emission.
# ======================================================================


class TestSpaceOverrideImports:
    """Verify lazy import helpers for space overrides and export."""

    def test_space_override_import(self):
        """FrameSpaceOverride should be importable from backend."""
        FSO = _safe_import_space_override()
        # May be None if backend not on path, but function shouldn't crash
        assert FSO is None or callable(FSO)

    def test_bvh_export_import(self):
        """convert_params_to_bvh should be importable from backend."""
        fn = _safe_import_bvh_export()
        assert fn is None or callable(fn)

    def test_fbx_export_import(self):
        """convert_bvh_to_fbx should be importable from backend."""
        fn = _safe_import_fbx_export()
        assert fn is None or callable(fn)


class TestSpaceOverrideUIWidgets:
    """Space override UI widgets exist with correct properties."""

    def test_space_combo_exists(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "_space_combo")
        assert panel._space_combo.count() == 3
        assert panel._space_combo.itemText(0) == "World"
        assert panel._space_combo.itemText(1) == "Camera"
        assert panel._space_combo.itemText(2) == "Carried"

    def test_space_start_end_spinboxes(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "_space_start")
        assert hasattr(panel, "_space_end")
        assert panel._space_start.minimum() == 0
        assert panel._space_end.minimum() == 0

    def test_space_ref_combo_default(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "_space_ref_combo")
        # Default has "—" placeholder
        assert panel._space_ref_combo.count() >= 1
        assert panel._space_ref_combo.itemData(0) is None

    def test_space_y_offset_default(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "_space_y_offset")
        assert panel._space_y_offset.value() == pytest.approx(0.4, abs=0.01)

    def test_space_table_columns(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._space_table.columnCount() == 5
        headers = [
            panel._space_table.horizontalHeaderItem(i).text()
            for i in range(5)
        ]
        assert headers == ["Start", "End", "Space", "Ref Person", "Y Offset"]

    def test_add_delete_buttons_exist(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "_add_override_btn")
        assert hasattr(panel, "_del_override_btn")
        assert panel._add_override_btn.text() == "Add Override"
        assert panel._del_override_btn.text() == "Delete Selected"

    def test_export_buttons_exist(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "_reexport_bvh_btn")
        assert hasattr(panel, "_reexport_fbx_btn")
        assert "BVH" in panel._reexport_bvh_btn.text()
        assert "FBX" in panel._reexport_fbx_btn.text()

    def test_export_status_label_exists(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "_export_status")
        assert panel._export_status.text() == ""

    def test_export_requested_signal_exists(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "export_requested")


class TestSpaceOverrideCRUD:
    """Add, display, and delete space overrides via the panel."""

    def test_add_override_requires_person(self, qapp):
        """Adding override with no person selected should be a no-op."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = -1
        panel._on_add_space_override()
        assert panel._space_table.rowCount() == 0

    def test_add_override_creates_entry(self, qapp):
        """Adding a space override should populate the table."""
        FSO = _safe_import_space_override()
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if FSO is None or CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0

        panel._space_start.setValue(10)
        panel._space_end.setValue(50)
        panel._space_combo.setCurrentIndex(1)  # Camera

        panel._on_add_space_override()

        assert panel._space_table.rowCount() == 1
        assert panel._space_table.item(0, 0).text() == "10"
        assert panel._space_table.item(0, 1).text() == "50"
        assert panel._space_table.item(0, 2).text() == "camera"

    def test_add_override_persists_to_correction_track(self, qapp):
        """Override should be stored in CorrectionTrack."""
        FSO = _safe_import_space_override()
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if FSO is None or CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0

        panel._space_start.setValue(20)
        panel._space_end.setValue(40)
        panel._space_combo.setCurrentIndex(0)  # World

        panel._on_add_space_override()

        ct = session.correction_tracks.get(0)
        assert ct is not None
        assert len(ct.space_overrides) == 1
        assert ct.space_overrides[0].frame_start == 20
        assert ct.space_overrides[0].frame_end == 40
        assert ct.space_overrides[0].space == "world"

    def test_add_carried_with_reference(self, qapp):
        """Carried override should store reference person and y_offset."""
        FSO = _safe_import_space_override()
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if FSO is None or CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0

        # Populate ref dropdown
        panel._update_space_ref_dropdown()

        panel._space_start.setValue(5)
        panel._space_end.setValue(25)
        panel._space_combo.setCurrentIndex(2)  # Carried
        panel._space_y_offset.setValue(0.6)

        # Select Person 1 as reference
        for i in range(panel._space_ref_combo.count()):
            if panel._space_ref_combo.itemData(i) == 1:
                panel._space_ref_combo.setCurrentIndex(i)
                break

        panel._on_add_space_override()

        ct = session.correction_tracks[0]
        assert len(ct.space_overrides) == 1
        ovr = ct.space_overrides[0]
        assert ovr.space == "carried"
        assert ovr.reference_person == 1
        assert ovr.y_offset == pytest.approx(0.6, abs=0.01)

    def test_add_swapped_range(self, qapp):
        """If start > end, they should be auto-swapped."""
        FSO = _safe_import_space_override()
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if FSO is None or CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0

        panel._space_start.setValue(50)
        panel._space_end.setValue(10)

        panel._on_add_space_override()

        ct = session.correction_tracks[0]
        assert ct.space_overrides[0].frame_start == 10
        assert ct.space_overrides[0].frame_end == 50

    def test_delete_override(self, qapp):
        """Deleting a selected override should remove it."""
        FSO = _safe_import_space_override()
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if FSO is None or CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0

        # Add two overrides
        panel._space_start.setValue(10)
        panel._space_end.setValue(20)
        panel._on_add_space_override()

        panel._space_start.setValue(30)
        panel._space_end.setValue(40)
        panel._on_add_space_override()

        assert panel._space_table.rowCount() == 2

        # Select first row and delete
        panel._space_table.setCurrentCell(0, 0)
        panel._on_delete_space_override()

        assert panel._space_table.rowCount() == 1
        ct = session.correction_tracks[0]
        assert len(ct.space_overrides) == 1
        assert ct.space_overrides[0].frame_start == 30

    def test_delete_no_selection(self, qapp):
        """Delete with no selection should be a no-op."""
        FSO = _safe_import_space_override()
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if FSO is None or CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0

        panel._space_start.setValue(10)
        panel._space_end.setValue(20)
        panel._on_add_space_override()

        # No row selected (deselect)
        panel._space_table.setCurrentCell(-1, -1)
        panel._on_delete_space_override()

        # Should still have the override
        ct = session.correction_tracks[0]
        assert len(ct.space_overrides) == 1

    def test_overlapping_override_replaces(self, qapp):
        """Adding an overlapping override should replace the existing one."""
        FSO = _safe_import_space_override()
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if FSO is None or CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0

        panel._space_start.setValue(10)
        panel._space_end.setValue(30)
        panel._on_add_space_override()

        # Add overlapping
        panel._space_start.setValue(20)
        panel._space_end.setValue(40)
        panel._space_combo.setCurrentIndex(1)  # Camera
        panel._on_add_space_override()

        ct = session.correction_tracks[0]
        assert len(ct.space_overrides) == 1
        assert ct.space_overrides[0].frame_start == 20
        assert ct.space_overrides[0].space == "camera"

    def test_person_change_refreshes_space_table(self, qapp):
        """Switching person should refresh the space overrides table."""
        FSO = _safe_import_space_override()
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if FSO is None or CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0

        # Add override for person 0
        panel._space_start.setValue(10)
        panel._space_end.setValue(20)
        panel._on_add_space_override()
        assert panel._space_table.rowCount() == 1

        # Switch to person 1 (no overrides)
        panel.set_person(1)
        assert panel._space_table.rowCount() == 0

        # Switch back to person 0
        panel.set_person(0)
        assert panel._space_table.rowCount() == 1


class TestSpaceRefDropdown:
    """Reference person dropdown for space overrides."""

    def test_ref_dropdown_populated_on_refresh(self, qapp):
        session = _make_session_with_params(n_persons=3)
        panel = PoseCorrectorPanel(session=session)
        panel.refresh()

        # Should have "—" + 3 persons
        assert panel._space_ref_combo.count() == 4
        assert panel._space_ref_combo.itemData(0) is None
        assert panel._space_ref_combo.itemData(1) == 0
        assert panel._space_ref_combo.itemData(2) == 1
        assert panel._space_ref_combo.itemData(3) == 2

    def test_ref_dropdown_excludes_inactive(self, qapp):
        session = _make_session_with_params(n_persons=3)
        session.inactive_tracks.add(1)
        panel = PoseCorrectorPanel(session=session)
        panel.refresh()

        # Should have "—" + 2 active persons
        assert panel._space_ref_combo.count() == 3
        data = [panel._space_ref_combo.itemData(i)
                for i in range(panel._space_ref_combo.count())]
        assert 1 not in data


class TestBVHExport:
    """BVH re-export button handler."""

    def test_no_person_is_noop(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = -1
        panel._on_reexport_bvh()
        assert panel._export_status.text() == ""

    def test_no_params_shows_error(self, qapp):
        session = _make_session_with_params()
        session.person_tracks[0].smplx_params = None
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        panel._on_reexport_bvh()
        assert "No SMPL-X params" in panel._export_status.text()

    def test_export_with_mock_backend(self, qapp):
        """Mock the BVH converter to verify it gets called correctly."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        session.person_tracks[0].person_dir = Path("/tmp/test_person_0")

        mock_convert = MagicMock(return_value="/tmp/test_person_0/corrected_body.bvh")

        with patch(
            "views.pose_corrector_panel._safe_import_bvh_export",
            return_value=mock_convert,
        ):
            panel._on_reexport_bvh()

        mock_convert.assert_called_once()
        call_kwargs = mock_convert.call_args
        assert call_kwargs[0][0] is session.person_tracks[0].smplx_params
        assert "corrected_body.bvh" in call_kwargs[0][1]
        assert call_kwargs[1]["fps"] == 30.0
        assert call_kwargs[1]["skip_world_grounding"] is True
        assert "exported" in panel._export_status.text().lower()

    def test_export_with_space_overrides_builds_ref_params(self, qapp):
        """BVH export with carried overrides should build reference_params."""
        FSO = _safe_import_space_override()
        _, _, CorrectionTrack = _safe_import_pose_correction()
        if FSO is None or CorrectionTrack is None:
            pytest.skip("pose_correction backend not available")

        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        session.person_tracks[0].person_dir = Path("/tmp/test_person_0")

        # Add carried override referencing person 1
        ct = CorrectionTrack(person_id=0)
        ct.add_space_override(FSO(
            frame_start=10, frame_end=30, space="carried",
            reference_person=1, y_offset=0.5,
        ))
        session.correction_tracks[0] = ct

        mock_convert = MagicMock(return_value="/tmp/corrected.bvh")

        with patch(
            "views.pose_corrector_panel._safe_import_bvh_export",
            return_value=mock_convert,
        ):
            panel._on_reexport_bvh()

        mock_convert.assert_called_once()
        ref_params = mock_convert.call_args[1]["reference_params"]
        assert ref_params is not None
        assert 1 in ref_params

    def test_export_signal_emitted(self, qapp):
        """export_requested signal should be emitted on success."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        session.person_tracks[0].person_dir = Path("/tmp/test_person_0")

        signals = []
        panel.export_requested.connect(lambda s: signals.append(s))

        mock_convert = MagicMock(return_value="/tmp/corrected.bvh")
        with patch(
            "views.pose_corrector_panel._safe_import_bvh_export",
            return_value=mock_convert,
        ):
            panel._on_reexport_bvh()

        assert signals == ["bvh"]

    def test_export_failure_shows_error(self, qapp):
        """Export failure should show error status, not crash."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        session.person_tracks[0].person_dir = Path("/tmp/test_person_0")

        mock_convert = MagicMock(side_effect=RuntimeError("test error"))
        with patch(
            "views.pose_corrector_panel._safe_import_bvh_export",
            return_value=mock_convert,
        ):
            panel._on_reexport_bvh()

        assert "failed" in panel._export_status.text().lower()


class TestFBXExport:
    """FBX re-export button handler."""

    def test_no_person_is_noop(self, qapp):
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = -1
        panel._on_reexport_fbx()
        assert panel._export_status.text() == ""

    def test_fbx_with_mock_backend(self, qapp):
        """Mock both BVH and FBX converters."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        session.person_tracks[0].person_dir = Path("/tmp/test_person_0")

        signals = []
        panel.export_requested.connect(lambda s: signals.append(s))

        mock_bvh = MagicMock(return_value="/tmp/corrected.bvh")
        mock_fbx = MagicMock(return_value="FBX converted")

        with patch(
            "views.pose_corrector_panel._safe_import_bvh_export",
            return_value=mock_bvh,
        ), patch(
            "views.pose_corrector_panel._safe_import_fbx_export",
            return_value=mock_fbx,
        ):
            panel._on_reexport_fbx()

        mock_bvh.assert_called_once()
        mock_fbx.assert_called_once()
        assert signals == ["fbx"]
        assert "fbx" in panel._export_status.text().lower()

    def test_fbx_unavailable_falls_back_to_bvh(self, qapp):
        """If FBX converter unavailable, BVH should still export."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        session.person_tracks[0].person_dir = Path("/tmp/test_person_0")

        signals = []
        panel.export_requested.connect(lambda s: signals.append(s))

        mock_bvh = MagicMock(return_value="/tmp/corrected.bvh")

        with patch(
            "views.pose_corrector_panel._safe_import_bvh_export",
            return_value=mock_bvh,
        ), patch(
            "views.pose_corrector_panel._safe_import_fbx_export",
            return_value=None,
        ):
            panel._on_reexport_fbx()

        mock_bvh.assert_called_once()
        assert signals == ["bvh"]
        assert "unavailable" in panel._export_status.text().lower()

    def test_fbx_failure_shows_error(self, qapp):
        """FBX conversion failure should show error but BVH signal still emitted."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._current_person = 0
        session.person_tracks[0].person_dir = Path("/tmp/test_person_0")

        signals = []
        panel.export_requested.connect(lambda s: signals.append(s))

        mock_bvh = MagicMock(return_value="/tmp/corrected.bvh")
        mock_fbx = MagicMock(side_effect=RuntimeError("blender not found"))

        with patch(
            "views.pose_corrector_panel._safe_import_bvh_export",
            return_value=mock_bvh,
        ), patch(
            "views.pose_corrector_panel._safe_import_fbx_export",
            return_value=mock_fbx,
        ):
            panel._on_reexport_fbx()

        assert signals == ["bvh"]
        assert "failed" in panel._export_status.text().lower()
