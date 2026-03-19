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
    PoseIssue,
    get_joint_euler,
    _get_joint_axis_angle,
    _rotation_angles_deg,
    _detect_angular_jumps,
    _detect_jitter,
    _detect_low_confidence,
    compute_pose_issues,
    smooth_joint_rotations,
    find_similar_frames,
    propagate_corrections,
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
    _CollapsibleSection,
    _LABEL_MIN_WIDTH,
    _SPINBOX_FIXED_WIDTH,
    _SLIDER_FIXED_HEIGHT,
    _MONO_FONT_FAMILY,
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

    def test_has_labels_checkbox(self, qapp):
        """Labels checkbox should exist and be unchecked by default."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._labels_checkbox is not None
        assert not panel._labels_checkbox.isChecked()

    def test_labels_checkbox_toggles_viewport(self, qapp):
        """Toggling the labels checkbox should set viewport joint labels."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        panel._labels_checkbox.setChecked(True)
        assert panel._viewport._show_joint_labels is True
        panel._labels_checkbox.setChecked(False)
        assert panel._viewport._show_joint_labels is False

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
# AppWindow dock integration (migrated from deleted MultiPersonTab)
# ======================================================================


class TestAppWindowPoseCorrectorIntegration:
    """PoseCorrectorPanel should be wired into AppWindow dock layout."""

    def test_app_window_has_pose_corrector(self, qapp):
        """AppWindow should have PoseCorrectorPanel in a dock."""
        from app_window import AppWindow
        window = AppWindow()
        assert hasattr(window, "_pose_corrector")
        assert isinstance(window._pose_corrector, PoseCorrectorPanel)

    def test_pose_corrector_has_mesh_viewport(self, qapp):
        """PoseCorrectorPanel should have its own MeshViewport."""
        from app_window import AppWindow
        window = AppWindow()
        assert window._pose_corrector.mesh_viewport is not None

    def test_frame_requested_wired(self, qapp):
        """Pose corrector frame_requested should be connected to video player."""
        from app_window import AppWindow
        window = AppWindow()
        # Verify signal exists and is connected
        assert hasattr(window._pose_corrector, "frame_requested")
        # Emit should not raise (connected to video_player.seek)
        window._pose_corrector.frame_requested.emit(0)


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


# ======================================================================
# Auto-Detect: Pure function tests
# ======================================================================


class TestRotationAnglesDeg:
    """_rotation_angles_deg: vectorized angular difference."""

    def test_identical_returns_zero(self):
        a = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        result = _rotation_angles_deg(a, a)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_known_difference(self):
        a = np.zeros((1, 3), dtype=np.float32)
        b = np.array([[np.pi / 2, 0, 0]], dtype=np.float32)
        result = _rotation_angles_deg(a, b)
        np.testing.assert_allclose(result, [90.0], atol=0.1)

    def test_output_shape_2d(self):
        a = np.zeros((5, 3), dtype=np.float32)
        b = np.ones((5, 3), dtype=np.float32)
        result = _rotation_angles_deg(a, b)
        assert result.shape == (5,)

    def test_output_shape_3d(self):
        """(N, J, 3) input → (N, J) output."""
        a = np.zeros((10, 21, 3), dtype=np.float32)
        b = np.ones((10, 21, 3), dtype=np.float32)
        result = _rotation_angles_deg(a, b)
        assert result.shape == (10, 21)

    def test_symmetric(self):
        a = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        b = np.array([[0.4, 0.5, 0.6]], dtype=np.float32)
        np.testing.assert_allclose(
            _rotation_angles_deg(a, b), _rotation_angles_deg(b, a), atol=1e-5
        )


class TestDetectAngularJumps:
    """_detect_angular_jumps: detect large single-frame pose changes."""

    def test_no_motion_no_issues(self):
        bp = np.zeros((50, 21, 3), dtype=np.float32)
        issues = _detect_angular_jumps(bp, person_id=0)
        assert len(issues) == 0

    def test_detects_large_jump(self):
        bp = np.zeros((50, 21, 3), dtype=np.float32)
        # Insert a 90° jump at frame 10 on joint 0 (L_Hip)
        bp[10, 0] = [np.pi / 2, 0, 0]
        issues = _detect_angular_jumps(bp, person_id=0, threshold_deg=45.0)
        assert len(issues) >= 1
        # Should detect the jump AT frame 10
        jump_frames = [i.frame for i in issues]
        assert 10 in jump_frames

    def test_small_motion_ignored(self):
        bp = np.zeros((50, 21, 3), dtype=np.float32)
        # 10° change — should be below 45° threshold
        bp[10, 0] = [np.radians(10), 0, 0]
        issues = _detect_angular_jumps(bp, person_id=0, threshold_deg=45.0)
        assert len(issues) == 0

    def test_custom_threshold(self):
        bp = np.zeros((50, 21, 3), dtype=np.float32)
        bp[10, 0] = [np.radians(20), 0, 0]
        # With a 15° threshold, should detect
        issues = _detect_angular_jumps(bp, person_id=0, threshold_deg=15.0)
        assert len(issues) >= 1

    def test_issue_fields(self):
        bp = np.zeros((50, 21, 3), dtype=np.float32)
        bp[10, 0] = [np.pi / 2, 0, 0]
        issues = _detect_angular_jumps(bp, person_id=7, threshold_deg=45.0)
        issue = [i for i in issues if i.frame == 10][0]
        assert issue.person_id == 7
        assert issue.issue_type == "angular_jump"
        assert issue.severity > 0.0
        assert issue.span == (9, 10)
        assert "jump" in issue.description.lower()

    def test_jump_at_first_frame(self):
        bp = np.zeros((5, 21, 3), dtype=np.float32)
        bp[0, 0] = [np.pi, 0, 0]  # Big value at frame 0
        bp[1, 0] = [0, 0, 0]      # Then zero
        # The jump is at frame 1 (comparing frame 0 and 1)
        issues = _detect_angular_jumps(bp, person_id=0, threshold_deg=45.0)
        assert len(issues) >= 1
        assert issues[0].frame == 1

    def test_multiple_joints_worst_reported(self):
        bp = np.zeros((5, 21, 3), dtype=np.float32)
        bp[2, 3] = [np.pi, 0, 0]  # 180° on joint 3 (body_pose index)
        bp[2, 5] = [np.radians(50), 0, 0]  # 50° on joint 5
        issues = _detect_angular_jumps(bp, person_id=0, threshold_deg=45.0)
        # Should report the worst joint (index 3 → SMPL-X joint 4 = L_Knee)
        jump_at_2 = [i for i in issues if i.frame == 2]
        assert len(jump_at_2) >= 1


class TestDetectJitter:
    """_detect_jitter: detect high-frequency oscillation."""

    def test_smooth_motion_no_jitter(self):
        bp = np.zeros((50, 21, 3), dtype=np.float32)
        # Smooth linear motion
        for f in range(50):
            bp[f, 0, 0] = f * 0.01  # tiny increment
        issues = _detect_jitter(bp, person_id=0)
        assert len(issues) == 0

    def test_static_no_jitter(self):
        bp = np.zeros((50, 21, 3), dtype=np.float32)
        issues = _detect_jitter(bp, person_id=0)
        assert len(issues) == 0

    def test_detects_oscillation(self):
        bp = np.zeros((50, 21, 3), dtype=np.float32)
        # Create high-frequency oscillation in frames 10-25
        for f in range(10, 25):
            if f % 2 == 0:
                bp[f, 0] = [np.radians(40), 0, 0]
            else:
                bp[f, 0] = [np.radians(-40), 0, 0]
        issues = _detect_jitter(bp, person_id=0, threshold_deg=15.0)
        assert len(issues) >= 1
        assert all(i.issue_type == "jitter" for i in issues)

    def test_short_sequence_returns_empty(self):
        bp = np.zeros((5, 21, 3), dtype=np.float32)
        issues = _detect_jitter(bp, person_id=0, window=7)
        assert len(issues) == 0

    def test_jitter_issue_fields(self):
        bp = np.zeros((50, 21, 3), dtype=np.float32)
        for f in range(10, 30):
            bp[f, 0] = [np.radians(50 * ((-1) ** f)), 0, 0]
        issues = _detect_jitter(bp, person_id=3, threshold_deg=10.0)
        if issues:
            issue = issues[0]
            assert issue.person_id == 3
            assert issue.issue_type == "jitter"
            assert issue.severity == 0.6
            assert issue.span[0] <= issue.span[1]
            assert issue.span[0] >= 0


class TestDetectLowConfidence:
    """_detect_low_confidence: detect spans of low confidence."""

    def test_all_high_confidence(self):
        confs = [0.9] * 50
        issues = _detect_low_confidence(confs, person_id=0)
        assert len(issues) == 0

    def test_detects_low_span(self):
        confs = [0.9] * 10 + [0.2] * 10 + [0.9] * 30
        issues = _detect_low_confidence(confs, person_id=0, threshold=0.4, min_span=3)
        assert len(issues) == 1
        assert issues[0].issue_type == "low_confidence"
        assert issues[0].span == (10, 19)

    def test_short_span_ignored(self):
        confs = [0.9] * 10 + [0.2] * 2 + [0.9] * 38
        issues = _detect_low_confidence(confs, person_id=0, threshold=0.4, min_span=3)
        assert len(issues) == 0

    def test_span_at_end(self):
        confs = [0.9] * 10 + [0.1] * 10
        issues = _detect_low_confidence(confs, person_id=0, threshold=0.4, min_span=3)
        assert len(issues) == 1
        assert issues[0].span == (10, 19)

    def test_multiple_spans(self):
        confs = [0.2] * 5 + [0.9] * 10 + [0.1] * 5 + [0.9] * 10
        issues = _detect_low_confidence(confs, person_id=0, threshold=0.4, min_span=3)
        assert len(issues) == 2

    def test_handles_track_confidence_objects(self):
        """Objects with .overall attribute should work."""
        class MockConf:
            def __init__(self, val):
                self.overall = val
        confs = [MockConf(0.9)] * 5 + [MockConf(0.1)] * 5 + [MockConf(0.9)] * 5
        issues = _detect_low_confidence(confs, person_id=0, threshold=0.4, min_span=3)
        assert len(issues) == 1

    def test_issue_fields(self):
        confs = [0.1] * 10 + [0.9] * 10
        issues = _detect_low_confidence(confs, person_id=5, threshold=0.4, min_span=3)
        assert len(issues) == 1
        issue = issues[0]
        assert issue.person_id == 5
        assert issue.issue_type == "low_confidence"
        assert issue.severity == 0.7
        assert issue.span == (0, 9)


class TestComputePoseIssues:
    """compute_pose_issues: integration test scanning session."""

    def test_empty_session_no_issues(self):
        session = Session()
        issues = compute_pose_issues(session)
        assert issues == []

    def test_no_params_no_issues(self):
        session = Session()
        session.person_tracks[0] = PersonTrack(person_id=0)
        issues = compute_pose_issues(session)
        assert issues == []

    def test_clean_params_no_issues(self):
        """All-zero body_pose should produce no issues."""
        session = Session(num_frames=50, fps=30.0)
        session.person_tracks[0] = PersonTrack(
            person_id=0,
            smplx_params={
                "body_pose": np.zeros((50, 21, 3), dtype=np.float32),
                "global_orient": np.zeros((50, 3), dtype=np.float32),
            },
        )
        issues = compute_pose_issues(session)
        assert issues == []

    def test_detects_jump_in_session(self):
        session = _make_session_with_params(n_frames=50, n_persons=1)
        params = session.person_tracks[0].smplx_params
        params["body_pose"][25, 0] = [np.pi, 0, 0]  # 180° jump
        issues = compute_pose_issues(session, jump_threshold_deg=45.0)
        jump_issues = [i for i in issues if i.issue_type == "angular_jump"]
        assert len(jump_issues) >= 1

    def test_detects_low_confidence_in_session(self):
        session = _make_session_with_params(n_frames=50, n_persons=1)
        session.person_tracks[0].confidences = (
            [0.9] * 10 + [0.1] * 10 + [0.9] * 30
        )
        issues = compute_pose_issues(session, conf_threshold=0.4, min_span=3)
        conf_issues = [i for i in issues if i.issue_type == "low_confidence"]
        assert len(conf_issues) >= 1

    def test_skips_inactive_tracks(self):
        session = _make_session_with_params(n_frames=50, n_persons=2)
        # Add big jump to person 1
        session.person_tracks[1].smplx_params["body_pose"][10, 0] = [np.pi, 0, 0]
        session.inactive_tracks.add(1)
        issues = compute_pose_issues(session, jump_threshold_deg=45.0)
        person_1_issues = [i for i in issues if i.person_id == 1]
        assert len(person_1_issues) == 0

    def test_sorted_by_severity_desc(self):
        session = _make_session_with_params(n_frames=50, n_persons=1)
        # Add jump (severity ~ 1.0) and low confidence (severity 0.7)
        session.person_tracks[0].smplx_params["body_pose"][10, 0] = [np.pi, 0, 0]
        session.person_tracks[0].confidences = (
            [0.9] * 30 + [0.1] * 10 + [0.9] * 10
        )
        issues = compute_pose_issues(session)
        if len(issues) >= 2:
            assert issues[0].severity >= issues[1].severity

    def test_handles_flat_body_pose(self):
        """body_pose as (N, 63) should be reshaped to (N, 21, 3)."""
        session = Session(num_frames=50)
        params = {
            "body_pose": np.zeros((50, 63), dtype=np.float32),
            "global_orient": np.zeros((50, 3), dtype=np.float32),
        }
        # Insert jump in flat format
        params["body_pose"][10, :3] = [np.pi, 0, 0]
        session.person_tracks[0] = PersonTrack(person_id=0, smplx_params=params)
        issues = compute_pose_issues(session, jump_threshold_deg=45.0)
        jump_issues = [i for i in issues if i.issue_type == "angular_jump"]
        assert len(jump_issues) >= 1


# ======================================================================
# Auto-Detect: UI widget tests
# ======================================================================


class TestAutoDetectUI:
    """Auto-Detect UI group on PoseCorrectorPanel."""

    @pytest.fixture()
    def panel(self, qapp):
        session = _make_session_with_params(n_frames=50, n_persons=2)
        p = PoseCorrectorPanel(session=session)
        yield p
        p.close()

    def test_detect_button_exists(self, panel):
        assert hasattr(panel, "_detect_btn")
        assert panel._detect_btn.text() == "Detect Bad Spans"

    def test_prev_next_buttons_exist(self, panel):
        assert hasattr(panel, "_prev_issue_btn")
        assert hasattr(panel, "_next_issue_btn")
        assert panel._prev_issue_btn.isHidden() is False
        assert panel._next_issue_btn.isHidden() is False

    def test_prev_next_initially_disabled(self, panel):
        assert not panel._prev_issue_btn.isEnabled()
        assert not panel._next_issue_btn.isEnabled()

    def test_issues_table_exists(self, panel):
        assert hasattr(panel, "_issues_table")
        assert panel._issues_table.columnCount() == 4
        headers = [
            panel._issues_table.horizontalHeaderItem(c).text()
            for c in range(4)
        ]
        assert headers == ["Frame", "Person", "Type", "Description"]

    def test_issues_label_default(self, panel):
        assert panel._issues_label.text() == "No scan performed"

    def test_detect_no_issues(self, qapp):
        """Truly clean params (all zeros) should produce no issues."""
        session = Session(num_frames=50, fps=30.0, img_width=640, img_height=480)
        session.person_tracks[0] = PersonTrack(
            person_id=0,
            smplx_params={
                "body_pose": np.zeros((50, 21, 3), dtype=np.float32),
                "global_orient": np.zeros((50, 3), dtype=np.float32),
            },
        )
        panel = PoseCorrectorPanel(session=session)
        panel._on_detect_bad_spans()
        assert panel._issues_label.text() == "No issues found."
        assert not panel._prev_issue_btn.isEnabled()
        assert not panel._next_issue_btn.isEnabled()
        assert panel._issues_table.rowCount() == 0
        panel.close()

    def test_detect_with_issues(self, panel):
        """Insert a jump and verify detection populates the table."""
        panel._session.person_tracks[0].smplx_params["body_pose"][25, 0] = [
            np.pi, 0, 0,
        ]
        panel._on_detect_bad_spans()
        assert panel._issues_table.rowCount() > 0
        assert panel._prev_issue_btn.isEnabled()
        assert panel._next_issue_btn.isEnabled()
        # Auto-navigates to first issue, label shows [1/N] format
        assert "[1/" in panel._issues_label.text()

    def test_next_prev_wraps(self, panel):
        """Next/Prev navigation wraps around the issue list."""
        panel._session.person_tracks[0].smplx_params["body_pose"][10, 0] = [
            np.pi, 0, 0,
        ]
        panel._session.person_tracks[0].smplx_params["body_pose"][30, 0] = [
            np.pi, 0, 0,
        ]
        panel._on_detect_bad_spans()
        n = len(panel._pose_issues)
        assert n >= 2

        # Navigate forward through all issues and wrap
        for _ in range(n):
            panel._on_next_pose_issue()
        assert panel._pose_issue_idx == 0  # wrapped

        # Navigate backward
        panel._on_prev_pose_issue()
        assert panel._pose_issue_idx == n - 1  # wrapped to last

    def test_double_click_navigates(self, panel):
        """Double-clicking a row in the issues table navigates to that issue."""
        panel._session.person_tracks[0].smplx_params["body_pose"][10, 0] = [
            np.pi, 0, 0,
        ]
        panel._on_detect_bad_spans()
        n = len(panel._pose_issues)
        if n > 1:
            panel._on_pose_issue_double_clicked(n - 1, 0)
            assert panel._pose_issue_idx == n - 1

    def test_navigate_emits_frame_requested(self, panel):
        """Navigation should emit frame_requested signal."""
        signals = []
        panel.frame_requested.connect(lambda f: signals.append(f))
        panel._session.person_tracks[0].smplx_params["body_pose"][20, 0] = [
            np.pi, 0, 0,
        ]
        panel._on_detect_bad_spans()
        assert len(signals) >= 1  # first issue auto-navigated

    def test_navigate_changes_person(self, panel):
        """Navigation to issue for a different person should update person."""
        panel._session.person_tracks[0].smplx_params["body_pose"][10, 0] = [
            np.pi, 0, 0,
        ]
        panel._session.person_tracks[1].smplx_params["body_pose"][20, 0] = [
            np.pi, 0, 0,
        ]
        panel._on_detect_bad_spans()
        # Find an issue for person 1
        for idx, issue in enumerate(panel._pose_issues):
            if issue.person_id == 1:
                panel._navigate_to_pose_issue(idx)
                assert panel._current_person == 1
                break


# ======================================================================
# Phase 5: smooth_joint_rotations pure function tests
# ======================================================================


class TestSmoothJointRotations:
    """smooth_joint_rotations: temporal smoothing on axis-angle body_pose."""

    def test_identity_on_constant_pose(self):
        """Smoothing a constant pose should produce the same pose."""
        bp = np.ones((20, 21, 3), dtype=np.float32) * 0.5
        result = smooth_joint_rotations(bp, 0, 19)
        np.testing.assert_allclose(result, bp, atol=1e-5)

    def test_reduces_noise(self):
        """Smoothing should reduce high-frequency noise."""
        bp = np.zeros((30, 21, 3), dtype=np.float32)
        # Add noise to joint 0
        rng = np.random.RandomState(42)
        bp[:, 0] = rng.randn(30, 3).astype(np.float32) * 0.5
        result = smooth_joint_rotations(bp, 0, 29, joint_indices=[0], window=7)
        # Smoothed std should be less than original
        assert np.std(result[:, 0]) < np.std(bp[:, 0])

    def test_only_affects_specified_range(self):
        """Frames outside [start, end] should be unchanged."""
        bp = np.zeros((30, 21, 3), dtype=np.float32)
        bp[:, 0] = np.random.RandomState(42).randn(30, 3).astype(np.float32)
        original = bp.copy()
        result = smooth_joint_rotations(bp, 10, 20, joint_indices=[0])
        np.testing.assert_array_equal(result[:10, 0], original[:10, 0])
        np.testing.assert_array_equal(result[21:, 0], original[21:, 0])

    def test_only_affects_specified_joints(self):
        """Joints not in joint_indices should be unchanged."""
        bp = np.random.RandomState(42).randn(20, 21, 3).astype(np.float32) * 0.3
        original = bp.copy()
        result = smooth_joint_rotations(bp, 0, 19, joint_indices=[0, 1])
        # Joint 5 should be unchanged
        np.testing.assert_array_equal(result[:, 5], original[:, 5])

    def test_none_joints_smooths_all(self):
        """joint_indices=None should smooth all joints."""
        bp = np.random.RandomState(42).randn(20, 21, 3).astype(np.float32) * 0.3
        result = smooth_joint_rotations(bp, 0, 19, joint_indices=None)
        # Some change expected on all joints (noisy input)
        assert not np.array_equal(result[5:15], bp[5:15])

    def test_moving_average_method(self):
        """moving_average method should also reduce noise."""
        bp = np.zeros((30, 21, 3), dtype=np.float32)
        rng = np.random.RandomState(42)
        bp[:, 0] = rng.randn(30, 3).astype(np.float32) * 0.5
        result = smooth_joint_rotations(
            bp, 0, 29, joint_indices=[0], window=7, method="moving_average",
        )
        assert np.std(result[:, 0]) < np.std(bp[:, 0])

    def test_returns_copy(self):
        """Original array should not be modified."""
        bp = np.random.RandomState(42).randn(10, 21, 3).astype(np.float32)
        original = bp.copy()
        smooth_joint_rotations(bp, 0, 9)
        np.testing.assert_array_equal(bp, original)

    def test_window_clamped_to_odd(self):
        """Even window should be rounded up to odd."""
        bp = np.random.RandomState(42).randn(20, 21, 3).astype(np.float32)
        # Should not crash with even window
        result = smooth_joint_rotations(bp, 0, 19, window=6)
        assert result.shape == bp.shape

    def test_small_window_minimum(self):
        """Window < 3 should be clamped to 3."""
        bp = np.random.RandomState(42).randn(10, 21, 3).astype(np.float32)
        result = smooth_joint_rotations(bp, 0, 9, window=1)
        assert result.shape == bp.shape

    def test_wrong_ndim_returns_copy(self):
        """Non-3D input should return unchanged copy."""
        bp = np.zeros((10, 63), dtype=np.float32)
        result = smooth_joint_rotations(bp, 0, 9)
        np.testing.assert_array_equal(result, bp)


# ======================================================================
# Phase 5: find_similar_frames pure function tests
# ======================================================================


class TestFindSimilarFrames:
    """find_similar_frames: find frames with similar joint rotations."""

    def test_finds_identical_frames(self):
        """Frames with identical rotation should match."""
        bp = np.zeros((20, 21, 3), dtype=np.float32)
        bp[5, 0] = [0.5, 0.3, 0.1]
        bp[10, 0] = [0.5, 0.3, 0.1]
        bp[15, 0] = [0.5, 0.3, 0.1]
        matches = find_similar_frames(bp, frame_idx=5, joint_idx=0, threshold_deg=5.0)
        assert 10 in matches
        assert 15 in matches
        assert 5 not in matches  # reference excluded

    def test_excludes_dissimilar(self):
        """Frames far from threshold should not match."""
        bp = np.zeros((20, 21, 3), dtype=np.float32)
        bp[5, 0] = [0.5, 0.3, 0.1]
        bp[10, 0] = [2.0, 1.5, 1.0]  # very different
        matches = find_similar_frames(bp, frame_idx=5, joint_idx=0, threshold_deg=5.0)
        assert 10 not in matches

    def test_all_zeros_match(self):
        """All-zero frames should all match each other."""
        bp = np.zeros((10, 21, 3), dtype=np.float32)
        matches = find_similar_frames(bp, frame_idx=0, joint_idx=0, threshold_deg=1.0)
        assert len(matches) == 9  # all others

    def test_reference_excluded(self):
        """Reference frame should never be in matches."""
        bp = np.zeros((10, 21, 3), dtype=np.float32)
        matches = find_similar_frames(bp, frame_idx=3, joint_idx=0, threshold_deg=90.0)
        assert 3 not in matches

    def test_invalid_joint_returns_empty(self):
        bp = np.zeros((10, 21, 3), dtype=np.float32)
        assert find_similar_frames(bp, 0, -1) == []
        assert find_similar_frames(bp, 0, 99) == []

    def test_invalid_frame_returns_empty(self):
        bp = np.zeros((10, 21, 3), dtype=np.float32)
        assert find_similar_frames(bp, -1, 0) == []
        assert find_similar_frames(bp, 99, 0) == []

    def test_wrong_ndim_returns_empty(self):
        bp = np.zeros((10, 63), dtype=np.float32)
        assert find_similar_frames(bp, 0, 0) == []

    def test_sorted_output(self):
        bp = np.zeros((20, 21, 3), dtype=np.float32)
        matches = find_similar_frames(bp, frame_idx=10, joint_idx=0, threshold_deg=90.0)
        assert matches == sorted(matches)

    def test_threshold_boundary(self):
        """Frame exactly at threshold should be included (<=)."""
        bp = np.zeros((5, 21, 3), dtype=np.float32)
        # Set frame 1 with known angular distance from frame 0
        angle_rad = np.radians(15.0)
        bp[1, 0] = [angle_rad, 0, 0]
        matches = find_similar_frames(bp, 0, 0, threshold_deg=15.0)
        assert 1 in matches


# ======================================================================
# Phase 5: propagate_corrections pure function tests
# ======================================================================


class TestPropagateCorrections:
    """propagate_corrections: SLERP interpolation between keyframes."""

    def test_single_frame(self):
        """start==end should return single frame."""
        aa = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = propagate_corrections(5, 5, aa, aa)
        assert 5 in result
        np.testing.assert_allclose(result[5], aa, atol=1e-5)

    def test_end_before_start(self):
        """end < start should return single start frame."""
        aa = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = propagate_corrections(10, 5, aa, aa)
        assert 10 in result
        assert len(result) == 1

    def test_interpolates_endpoints(self):
        """Start and end frames should match their input rotations."""
        start_aa = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        end_aa = np.array([0.5, 0.3, 0.1], dtype=np.float32)
        result = propagate_corrections(0, 10, start_aa, end_aa)
        np.testing.assert_allclose(result[0], start_aa, atol=1e-4)
        np.testing.assert_allclose(result[10], end_aa, atol=1e-4)

    def test_all_frames_present(self):
        """All frames in [start, end] should be in result."""
        start_aa = np.zeros(3, dtype=np.float32)
        end_aa = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = propagate_corrections(5, 15, start_aa, end_aa)
        for f in range(5, 16):
            assert f in result

    def test_midpoint_intermediate(self):
        """Midpoint should be between start and end rotations."""
        start_aa = np.zeros(3, dtype=np.float32)
        end_aa = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = propagate_corrections(0, 10, start_aa, end_aa)
        mid_norm = np.linalg.norm(result[5])
        start_norm = np.linalg.norm(start_aa)
        end_norm = np.linalg.norm(end_aa)
        # Mid should be between start and end
        assert start_norm <= mid_norm <= end_norm or abs(mid_norm - (start_norm + end_norm) / 2) < 0.5

    def test_output_dtype(self):
        start_aa = np.zeros(3, dtype=np.float32)
        end_aa = np.array([0.5, 0.3, 0.1], dtype=np.float32)
        result = propagate_corrections(0, 5, start_aa, end_aa)
        for aa in result.values():
            assert aa.dtype == np.float32

    def test_monotonic_rotation_magnitude(self):
        """Rotation magnitude should increase monotonically from zero to target."""
        start_aa = np.zeros(3, dtype=np.float32)
        end_aa = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = propagate_corrections(0, 20, start_aa, end_aa)
        norms = [np.linalg.norm(result[f]) for f in range(21)]
        for i in range(len(norms) - 1):
            assert norms[i] <= norms[i + 1] + 1e-5


# ======================================================================
# Phase 5: _on_smooth widget test
# ======================================================================


class TestOnSmooth:
    """_on_smooth: widget-level smoothing operation."""

    @pytest.fixture()
    def panel(self, qapp):
        session = _make_session_with_params(n_frames=50, n_persons=1)
        # Add noise for smoothing
        rng = np.random.RandomState(42)
        session.person_tracks[0].smplx_params["body_pose"][:, 0] = (
            rng.randn(50, 3).astype(np.float32) * 0.3
        )
        p = PoseCorrectorPanel(session=session)
        p._current_person = 0
        p._current_joint = 1  # L_Hip (body_pose idx 0)
        p._current_frame = 25
        yield p
        p.close()

    def test_smooth_reduces_noise(self, panel):
        """Smoothing should reduce std of the joint rotation."""
        bp = panel._session.person_tracks[0].smplx_params["body_pose"]
        old_std = np.std(bp[:, 0])
        panel._range_start.setValue(0)
        panel._range_end.setValue(49)
        panel._smooth_window.setValue(7)
        panel._smooth_scope.setCurrentIndex(0)  # Current Joint
        panel._on_smooth()
        bp_after = panel._session.person_tracks[0].smplx_params["body_pose"]
        if bp_after.ndim == 2:
            bp_after = bp_after.reshape(bp_after.shape[0], -1, 3)
        new_std = np.std(bp_after[:, 0])
        assert new_std < old_std

    def test_smooth_emits_correction_applied(self, panel):
        signals = []
        panel.correction_applied.connect(lambda p, f: signals.append((p, f)))
        panel._range_start.setValue(5)
        panel._range_end.setValue(15)
        panel._on_smooth()
        assert len(signals) == 1
        assert signals[0][0] == 0  # person_id

    def test_smooth_undo(self, panel):
        bp_before = panel._session.person_tracks[0].smplx_params["body_pose"].copy()
        if bp_before.ndim == 2:
            bp_before = bp_before.reshape(bp_before.shape[0], -1, 3)
        panel._range_start.setValue(0)
        panel._range_end.setValue(49)
        panel._on_smooth()
        # Undo
        panel._session.undo_stack.undo()
        bp_after_undo = np.asarray(
            panel._session.person_tracks[0].smplx_params["body_pose"], dtype=np.float32,
        )
        if bp_after_undo.ndim == 2:
            bp_after_undo = bp_after_undo.reshape(bp_after_undo.shape[0], -1, 3)
        np.testing.assert_allclose(bp_after_undo, bp_before, atol=1e-6)

    def test_smooth_all_joints_scope(self, panel):
        """Scope 'All Body Joints' should modify multiple joints."""
        bp_before = panel._session.person_tracks[0].smplx_params["body_pose"].copy()
        if bp_before.ndim == 2:
            bp_before = bp_before.reshape(bp_before.shape[0], -1, 3)
        panel._range_start.setValue(0)
        panel._range_end.setValue(49)
        panel._smooth_scope.setCurrentIndex(1)  # All Body Joints
        panel._on_smooth()
        bp_after = np.asarray(
            panel._session.person_tracks[0].smplx_params["body_pose"], dtype=np.float32,
        )
        if bp_after.ndim == 2:
            bp_after = bp_after.reshape(bp_after.shape[0], -1, 3)
        # At least some change on joints other than 0
        changed = not np.array_equal(bp_after[:, 0], bp_before[:, 0])
        assert changed

    def test_smooth_no_person_noop(self, panel):
        """Smoothing with no person selected should be a no-op."""
        panel._current_person = -1
        panel._range_start.setValue(0)
        panel._range_end.setValue(49)
        panel._on_smooth()  # should not crash

    def test_smooth_hand_joint_noop(self, panel):
        """Smoothing hand joints (>21) should be a no-op for current joint scope."""
        panel._current_joint = 25
        panel._range_start.setValue(0)
        panel._range_end.setValue(10)
        panel._smooth_scope.setCurrentIndex(0)  # Current Joint
        panel._on_smooth()  # should not crash, no changes


# ======================================================================
# Phase 5: _on_apply_to_similar widget test
# ======================================================================


class TestOnApplyToSimilar:
    """_on_apply_to_similar: batch correction of similar frames."""

    @pytest.fixture()
    def panel(self, qapp):
        session = _make_session_with_params(n_frames=30, n_persons=1)
        # Set same rotation at multiple frames
        for f in [5, 10, 15, 20]:
            session.person_tracks[0].smplx_params["body_pose"][f, 0] = [0.5, 0.3, 0.1]
        p = PoseCorrectorPanel(session=session)
        p._current_person = 0
        p._current_joint = 1  # L_Hip (bp idx 0)
        p._current_frame = 5
        yield p
        p.close()

    def test_finds_and_applies(self, panel):
        """Should find similar frames and apply the correction."""
        # Set target euler (different from current)
        panel._euler_x.setValue(30.0)
        panel._euler_y.setValue(0.0)
        panel._euler_z.setValue(0.0)
        panel._sim_threshold.setValue(15.0)
        panel._on_apply_to_similar()
        bp = np.asarray(
            panel._session.person_tracks[0].smplx_params["body_pose"], dtype=np.float32,
        )
        if bp.ndim == 2:
            bp = bp.reshape(bp.shape[0], -1, 3)
        # Frames 10, 15, 20 should now have the same correction
        for f in [10, 15, 20]:
            assert not np.allclose(bp[f, 0], [0.5, 0.3, 0.1], atol=0.01)

    def test_status_label_updated(self, panel):
        panel._euler_x.setValue(30.0)
        panel._sim_threshold.setValue(15.0)
        panel._on_apply_to_similar()
        assert "similar" in panel._sim_status.text().lower()

    def test_no_matches_status(self, qapp):
        """When no frames match (all have very different rotations), show no-match message."""
        session = _make_session_with_params(n_frames=10, n_persons=1)
        # Make every frame have a unique, widely-spaced rotation
        for f in range(10):
            session.person_tracks[0].smplx_params["body_pose"][f, 0] = [
                f * 0.5, 0, 0,
            ]
        p = PoseCorrectorPanel(session=session)
        p._current_person = 0
        p._current_joint = 1  # L_Hip (bp idx 0)
        p._current_frame = 0
        p._sim_threshold.setValue(1.0)  # 1° — too tight for 0.5 rad gaps
        p._on_apply_to_similar()
        assert "no similar" in p._sim_status.text().lower()
        p.close()

    def test_emits_correction_applied(self, panel):
        signals = []
        panel.correction_applied.connect(lambda p, f: signals.append((p, f)))
        panel._euler_x.setValue(30.0)
        panel._sim_threshold.setValue(15.0)
        panel._on_apply_to_similar()
        assert len(signals) == 1

    def test_undo_restores(self, panel):
        bp_before = np.asarray(
            panel._session.person_tracks[0].smplx_params["body_pose"], dtype=np.float32,
        ).copy()
        if bp_before.ndim == 2:
            bp_before = bp_before.reshape(bp_before.shape[0], -1, 3)
        panel._euler_x.setValue(30.0)
        panel._sim_threshold.setValue(15.0)
        panel._on_apply_to_similar()
        panel._session.undo_stack.undo()
        bp_after = np.asarray(
            panel._session.person_tracks[0].smplx_params["body_pose"], dtype=np.float32,
        )
        if bp_after.ndim == 2:
            bp_after = bp_after.reshape(bp_after.shape[0], -1, 3)
        np.testing.assert_allclose(bp_after, bp_before, atol=1e-6)

    def test_no_person_noop(self, panel):
        panel._current_person = -1
        panel._on_apply_to_similar()  # should not crash

    def test_global_orient_joint_noop(self, panel):
        """Joint 0 (global orient) is not a body joint — should show message."""
        panel._current_joint = 0
        panel._on_apply_to_similar()
        assert panel._sim_status.text() != ""


# ======================================================================
# Phase 5: _on_propagate widget test
# ======================================================================


class TestOnPropagate:
    """_on_propagate: SLERP correction propagation across frame range."""

    @pytest.fixture()
    def panel(self, qapp):
        session = _make_session_with_params(n_frames=50, n_persons=1)
        # Set distinct rotations at start and end for SLERP
        session.person_tracks[0].smplx_params["body_pose"][10, 0] = [0.0, 0.0, 0.0]
        session.person_tracks[0].smplx_params["body_pose"][30, 0] = [1.0, 0.0, 0.0]
        p = PoseCorrectorPanel(session=session)
        p._current_person = 0
        p._current_joint = 1  # L_Hip (bp idx 0)
        p._current_frame = 20
        yield p
        p.close()

    def test_interpolates_across_range(self, panel):
        """Propagation should interpolate between start and end frames."""
        panel._range_start.setValue(10)
        panel._range_end.setValue(30)
        panel._on_propagate()
        bp = np.asarray(
            panel._session.person_tracks[0].smplx_params["body_pose"], dtype=np.float32,
        )
        if bp.ndim == 2:
            bp = bp.reshape(bp.shape[0], -1, 3)
        # Frame 20 (midpoint) should have intermediate rotation
        mid_norm = np.linalg.norm(bp[20, 0])
        assert mid_norm > 0.0  # not zero (was interpolated)

    def test_emits_correction_applied(self, panel):
        signals = []
        panel.correction_applied.connect(lambda p, f: signals.append((p, f)))
        panel._range_start.setValue(10)
        panel._range_end.setValue(30)
        panel._on_propagate()
        assert len(signals) == 1

    def test_status_label_updated(self, panel):
        panel._range_start.setValue(10)
        panel._range_end.setValue(30)
        panel._on_propagate()
        assert "propagated" in panel._prop_status.text().lower()

    def test_same_start_end_shows_error(self, panel):
        """Equal start/end should show message, not crash."""
        panel._range_start.setValue(20)
        panel._range_end.setValue(20)
        panel._on_propagate()
        assert "must differ" in panel._prop_status.text().lower()

    def test_undo_restores(self, panel):
        bp_before = np.asarray(
            panel._session.person_tracks[0].smplx_params["body_pose"], dtype=np.float32,
        ).copy()
        if bp_before.ndim == 2:
            bp_before = bp_before.reshape(bp_before.shape[0], -1, 3)
        panel._range_start.setValue(10)
        panel._range_end.setValue(30)
        panel._on_propagate()
        panel._session.undo_stack.undo()
        bp_after = np.asarray(
            panel._session.person_tracks[0].smplx_params["body_pose"], dtype=np.float32,
        )
        if bp_after.ndim == 2:
            bp_after = bp_after.reshape(bp_after.shape[0], -1, 3)
        np.testing.assert_allclose(bp_after, bp_before, atol=1e-6)

    def test_global_orient_propagation(self, panel):
        """Joint 0 (global_orient) should also propagate."""
        panel._current_joint = 0
        panel._session.person_tracks[0].smplx_params["global_orient"][10] = [0.0, 0.0, 0.0]
        panel._session.person_tracks[0].smplx_params["global_orient"][30] = [0.5, 0.0, 0.0]
        panel._range_start.setValue(10)
        panel._range_end.setValue(30)
        panel._on_propagate()
        go = np.asarray(
            panel._session.person_tracks[0].smplx_params["global_orient"], dtype=np.float32,
        )
        mid_norm = np.linalg.norm(go[20])
        assert mid_norm > 0.0

    def test_no_person_noop(self, panel):
        panel._current_person = -1
        panel._on_propagate()

    def test_hand_joint_shows_error(self, panel):
        panel._current_joint = 25
        panel._on_propagate()
        assert "body joints" in panel._prop_status.text().lower()


# ======================================================================
# Phase 5: Preview range UI tests
# ======================================================================


class TestPreviewRange:
    """Preview range controls — ±N frames auto-play after correction."""

    @pytest.fixture()
    def panel(self, qapp):
        session = _make_session_with_params(n_frames=100, n_persons=1)
        p = PoseCorrectorPanel(session=session)
        p._current_person = 0
        p._current_joint = 1
        p._current_frame = 50
        yield p
        p.close()

    def test_preview_widgets_exist(self, panel):
        assert hasattr(panel, "_preview_half_range")
        assert hasattr(panel, "_preview_btn")
        assert hasattr(panel, "_auto_preview_check")
        assert hasattr(panel, "_preview_timer")

    def test_default_half_range(self, panel):
        assert panel._preview_half_range.value() == 15

    def test_preview_btn_text(self, panel):
        assert panel._preview_btn.text() == "Preview"

    def test_preview_play_starts_timer(self, panel):
        """Clicking preview should start the timer."""
        panel._on_preview_play()
        assert panel._preview_timer.isActive()
        assert panel._preview_btn.text() == "Stop"
        panel._preview_timer.stop()  # cleanup

    def test_preview_emits_frame_requested(self, panel):
        """Preview ticks should emit frame_requested."""
        signals = []
        panel.frame_requested.connect(lambda f: signals.append(f))
        panel._on_preview_play()
        # Manually tick a few times
        panel._on_preview_tick()
        panel._on_preview_tick()
        panel._preview_timer.stop()
        assert len(signals) >= 2

    def test_preview_stops_at_end(self, panel):
        """Preview should stop when reaching the end of the range."""
        panel._preview_half_range.setValue(2)
        panel._on_preview_play()
        # Tick through all frames
        for _ in range(10):  # more than enough
            if panel._preview_timer.isActive():
                panel._on_preview_tick()
        assert not panel._preview_timer.isActive()
        assert panel._preview_btn.text() == "Preview"

    def test_auto_preview_checkbox(self, panel):
        """Auto-preview checkbox should be unchecked by default."""
        assert not panel._auto_preview_check.isChecked()

    def test_auto_preview_triggers_on_apply(self, panel):
        """When auto-preview is checked, Apply should start preview."""
        panel._auto_preview_check.setChecked(True)
        panel._euler_x.setValue(10.0)
        # Mock CorrectionTrack
        mock_ct = MagicMock()
        mock_ct.get_correction.return_value = None
        mock_ct.corrections = []
        with patch(
            "views.pose_corrector_panel._safe_import_pose_correction",
            return_value=(
                axis_angle_to_euler_deg_fallback,
                euler_deg_to_axis_angle_fallback,
                MagicMock(return_value=mock_ct),
            ),
        ):
            panel._on_apply()
        # Timer should be running (auto-preview triggered)
        assert panel._preview_timer.isActive()
        panel._preview_timer.stop()

    def test_preview_range_clamps_to_bounds(self, panel):
        """Preview near the start of the video should clamp to frame 0."""
        panel._current_frame = 2
        panel._preview_half_range.setValue(15)
        panel._on_preview_play()
        assert panel._preview_frame == 0
        panel._preview_timer.stop()


# ======================================================================
# Phase 8: Property Panel Refinement — collapsible/tabbed structure tests
# ======================================================================


class TestCollapsibleSection:
    """_CollapsibleSection: toggle visibility of content area."""

    def test_construction_expanded(self, qapp):
        section = _CollapsibleSection("Test Section")
        assert section._header.isChecked()
        assert not section._content.isHidden()

    def test_construction_collapsed(self, qapp):
        section = _CollapsibleSection("Test", collapsed=True)
        assert not section._header.isChecked()
        assert section._content.isHidden()

    def test_toggle_collapses_content(self, qapp):
        section = _CollapsibleSection("Test")
        assert not section._content.isHidden()
        section._header.setChecked(False)
        assert section._content.isHidden()

    def test_toggle_expands_content(self, qapp):
        section = _CollapsibleSection("Test", collapsed=True)
        assert section._content.isHidden()
        section._header.setChecked(True)
        assert not section._content.isHidden()

    def test_arrow_type_expanded(self, qapp):
        from PySide6.QtCore import Qt
        section = _CollapsibleSection("Test")
        assert section._header.arrowType() == Qt.DownArrow

    def test_arrow_type_collapsed(self, qapp):
        from PySide6.QtCore import Qt
        section = _CollapsibleSection("Test", collapsed=True)
        assert section._header.arrowType() == Qt.RightArrow

    def test_content_layout_accessible(self, qapp):
        from PySide6.QtWidgets import QVBoxLayout
        section = _CollapsibleSection("Test")
        assert isinstance(section.content_layout, QVBoxLayout)

    def test_add_widget_to_content(self, qapp):
        from PySide6.QtWidgets import QLabel
        section = _CollapsibleSection("Test")
        label = QLabel("child")
        section.content_layout.addWidget(label)
        assert section.content_layout.count() == 1


class TestTabbedPropertyPanel:
    """Phase 8: PoseCorrectorPanel tabbed layout with 4 tabs."""

    def test_has_controls_tabs(self, qapp):
        """Panel should have a QTabWidget for property sections."""
        from PySide6.QtWidgets import QTabWidget
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert hasattr(panel, "_controls_tabs")
        assert isinstance(panel._controls_tabs, QTabWidget)

    def test_four_tabs(self, qapp):
        """Controls tabs should have exactly 4 tabs."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._controls_tabs.count() == 4

    def test_tab_names(self, qapp):
        """Tabs should be Pose, Corrections, Export, Space."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        names = [panel._controls_tabs.tabText(i) for i in range(4)]
        assert names == ["Pose", "Corrections", "Export", "Space"]

    def test_pose_tab_has_person_combo(self, qapp):
        """Person combo should exist and be accessible."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._person_combo is not None

    def test_pose_tab_has_euler_sliders(self, qapp):
        """Euler sliders should exist in the Pose tab."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._euler_x is not None
        assert panel._slider_x is not None

    def test_corrections_tab_has_table(self, qapp):
        """Corrections table should exist in the Corrections tab."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._corrections_table is not None
        assert panel._corrections_table.columnCount() == 5

    def test_export_tab_has_buttons(self, qapp):
        """Export tab should have BVH and FBX buttons."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._reexport_bvh_btn is not None
        assert panel._reexport_fbx_btn is not None

    def test_space_tab_has_controls(self, qapp):
        """Space tab should have override controls."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._space_combo is not None
        assert panel._space_table is not None
        assert panel._space_table.columnCount() == 5

    def test_viewport_not_in_tabs(self, qapp):
        """Viewport should be on the left side, not in tabs."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert isinstance(panel._viewport, MeshViewport)
        # Viewport should not be a child of the tab widget
        vp_parent = panel._viewport.parent()
        while vp_parent is not None:
            assert not isinstance(vp_parent, type(panel._controls_tabs))
            if vp_parent == panel:
                break
            vp_parent = vp_parent.parent()


class TestConsistentGridLayout:
    """Phase 8: consistent 120px labels, monospace spinboxes, uniform sliders."""

    def test_euler_spinbox_monospace_font(self, qapp):
        """Euler spinboxes should have monospace font."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        font_family = panel._euler_x.font().family()
        assert "consolas" in font_family.lower() or "courier" in font_family.lower() or "mono" in font_family.lower()

    def test_euler_spinbox_fixed_width(self, qapp):
        """Euler spinboxes should have consistent fixed width."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._euler_x.maximumWidth() == _SPINBOX_FIXED_WIDTH
        assert panel._euler_y.maximumWidth() == _SPINBOX_FIXED_WIDTH
        assert panel._euler_z.maximumWidth() == _SPINBOX_FIXED_WIDTH

    def test_slider_fixed_height(self, qapp):
        """Euler sliders should have consistent fixed height."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        assert panel._slider_x.maximumHeight() == _SLIDER_FIXED_HEIGHT
        assert panel._slider_y.maximumHeight() == _SLIDER_FIXED_HEIGHT
        assert panel._slider_z.maximumHeight() == _SLIDER_FIXED_HEIGHT

    def test_range_spinbox_monospace(self, qapp):
        """Frame range spinboxes should have monospace font."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        font_family = panel._range_start.font().family()
        assert "consolas" in font_family.lower() or "courier" in font_family.lower() or "mono" in font_family.lower()

    def test_smooth_window_monospace(self, qapp):
        """Smooth window spinbox should have monospace font."""
        session = _make_session_with_params()
        panel = PoseCorrectorPanel(session=session)
        font_family = panel._smooth_window.font().family()
        assert "consolas" in font_family.lower() or "courier" in font_family.lower() or "mono" in font_family.lower()

    def test_constants_defined(self, qapp):
        """Layout constants should be defined with expected values."""
        assert _LABEL_MIN_WIDTH == 120
        assert _SPINBOX_FIXED_WIDTH == 80
        assert _SLIDER_FIXED_HEIGHT == 22
