"""Tests for two-click bbox editing, interpolation, and correction persistence.

Why: Bbox editing is the core interaction for correcting multi-person tracking
errors. The two-click workflow (top-left → bottom-right) must correctly map
normalized VideoPlayer click coordinates to pixel coordinates, store corrections
in the Session model, handle edge cases (min size, clamping, state cancellation),
and interpolate between keyframes in delta space. These tests verify the complete
editing pipeline in isolation — no GPU, no video, no on-screen rendering needed.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session, PersonTrack
from views.identity_inspector import (
    IdentityInspector,
    interpolate_bbox_corrections,
)
from views.bbox_overlay import render_edit_preview


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_with_tracks(num_frames: int = 100) -> Session:
    """Create a session with two person tracks for testing."""
    session = Session()
    session.num_frames = num_frames
    session.img_width = 640
    session.img_height = 480
    session.person_tracks = {
        0: PersonTrack(
            person_id=0,
            confidences=[0.9 - i * 0.005 for i in range(num_frames)],
            keyframes=[
                {"frame": 10, "verified": True},
                {"frame": 50, "verified": False},
            ],
            bboxes=np.array([[50, 60, 200, 300]] * num_frames, dtype=float),
        ),
        1: PersonTrack(
            person_id=1,
            confidences=[0.7] * num_frames,
            keyframes=[],
            bboxes=np.array([[300, 100, 500, 400]] * num_frames, dtype=float),
        ),
    }
    return session


def _session_with_corrections(num_frames: int = 100) -> Session:
    """Create a session with bbox corrections at frames 20 and 60."""
    session = _session_with_tracks(num_frames)
    corrections = np.zeros((num_frames, 4), dtype=float)
    corrections[20] = [60, 70, 210, 310]
    corrections[60] = [70, 80, 220, 320]
    session.person_tracks[0].bbox_corrections = corrections
    session.person_tracks[0].original_bboxes = session.person_tracks[0].bboxes.copy()
    return session


# ---------------------------------------------------------------------------
# BBox edit state machine
# ---------------------------------------------------------------------------


class TestBBoxEditStateMachine:
    """Verify the two-click state machine transitions."""

    def test_initial_state_is_idle(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        assert panel._bbox_edit_state is None
        assert panel._bbox_edit_corner1 is None

    def test_edit_button_enters_click1(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel._on_edit_bbox()
        assert panel._bbox_edit_state == "click1"

    def test_edit_button_noop_without_person(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        panel._on_edit_bbox()
        assert panel._bbox_edit_state is None

    def test_first_click_transitions_to_click2(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)
        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)
        assert panel._bbox_edit_state == "click2"
        assert panel._bbox_edit_corner1 == (64, 96)

    def test_second_click_returns_to_idle(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)
        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)
        panel.on_frame_click(0.5, 0.8)
        assert panel._bbox_edit_state is None
        assert panel._bbox_edit_corner1 is None

    def test_cancel_resets_from_click1(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel._on_edit_bbox()
        panel._on_cancel_edit()
        assert panel._bbox_edit_state is None

    def test_cancel_resets_from_click2(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)
        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)
        panel._on_cancel_edit()
        assert panel._bbox_edit_state is None
        assert panel._bbox_edit_corner1 is None

    def test_click_ignored_when_idle(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        # No _on_edit_bbox called — click should be ignored
        panel.on_frame_click(0.5, 0.5)
        assert panel._bbox_edit_state is None

    def test_person_change_cancels_edit(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel._on_edit_bbox()
        panel.set_person(1)
        assert panel._bbox_edit_state is None


# ---------------------------------------------------------------------------
# BBox edit UI state
# ---------------------------------------------------------------------------


class TestBBoxEditUIState:
    """Verify button enabled/disabled states and status label updates."""

    def test_has_bbox_editing_widgets(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert panel._edit_bbox_btn is not None
        assert panel._cancel_edit_btn is not None
        assert panel._interpolate_btn is not None
        assert panel._apply_btn is not None
        assert panel._bbox_status is not None

    def test_initial_button_states(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert panel._edit_bbox_btn.isEnabled()
        assert not panel._cancel_edit_btn.isEnabled()

    def test_edit_mode_disables_edit_btn(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel._on_edit_bbox()
        assert not panel._edit_bbox_btn.isEnabled()
        assert panel._cancel_edit_btn.isEnabled()

    def test_cancel_restores_button_states(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel._on_edit_bbox()
        panel._on_cancel_edit()
        assert panel._edit_bbox_btn.isEnabled()
        assert not panel._cancel_edit_btn.isEnabled()

    def test_status_shows_click1_message(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel._on_edit_bbox()
        assert "top-left" in panel._bbox_status.text().lower()

    def test_status_shows_click2_message(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)
        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)
        assert "bottom-right" in panel._bbox_status.text().lower()

    def test_status_shows_idle_after_complete(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)
        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)
        panel.on_frame_click(0.5, 0.8)
        assert "idle" in panel._bbox_status.text().lower()


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------


class TestCoordinateMapping:
    """Verify normalized-to-pixel coordinate conversion."""

    def test_corner_coords_640x480(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)
        panel._on_edit_bbox()
        panel.on_frame_click(0.0, 0.0)  # top-left of image
        assert panel._bbox_edit_corner1 == (0, 0)

    def test_center_coords(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)
        panel._on_edit_bbox()
        panel.on_frame_click(0.5, 0.5)
        assert panel._bbox_edit_corner1 == (320, 240)


# ---------------------------------------------------------------------------
# BBox correction storage
# ---------------------------------------------------------------------------


class TestBBoxCorrectionStorage:
    """Verify that completed edits store corrections in Session."""

    def test_stores_correction_in_track(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)

        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)   # top-left: (64, 96)
        panel.on_frame_click(0.5, 0.8)   # bottom-right: (320, 384)

        track = session.person_tracks[0]
        assert track.bbox_corrections is not None
        bbox = track.bbox_corrections[30]
        assert not np.all(bbox == 0)
        assert bbox[0] == 64   # x1
        assert bbox[1] == 96   # y1
        assert bbox[2] == 320  # x2
        assert bbox[3] == 384  # y2

    def test_swapped_corners_normalized(self, qapp):
        """Clicking bottom-right first, then top-left, still produces valid bbox."""
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)

        panel._on_edit_bbox()
        panel.on_frame_click(0.5, 0.8)   # bottom-right first
        panel.on_frame_click(0.1, 0.2)   # top-left second

        track = session.person_tracks[0]
        bbox = track.bbox_corrections[30]
        assert bbox[0] < bbox[2]  # x1 < x2
        assert bbox[1] < bbox[3]  # y1 < y2

    def test_minimum_bbox_size_enforced(self, qapp):
        """Bboxes smaller than 20px are expanded."""
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)

        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.1)
        panel.on_frame_click(0.1 + 5 / 640, 0.1 + 5 / 480)  # ~5px apart

        track = session.person_tracks[0]
        bbox = track.bbox_corrections[30]
        assert bbox[2] - bbox[0] >= 20
        assert bbox[3] - bbox[1] >= 20

    def test_bbox_clamped_to_image_bounds(self, qapp):
        """Bboxes extending beyond image are clamped."""
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)

        # Force corner1 near edge, then corner2 at max
        panel._on_edit_bbox()
        panel.on_frame_click(0.95, 0.95)
        panel.on_frame_click(1.0, 1.0)

        track = session.person_tracks[0]
        bbox = track.bbox_corrections[30]
        assert bbox[2] <= session.img_width - 1
        assert bbox[3] <= session.img_height - 1

    def test_backup_original_bboxes(self, qapp):
        """First correction should backup original bboxes."""
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)

        assert session.person_tracks[0].original_bboxes is None

        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)
        panel.on_frame_click(0.5, 0.8)

        track = session.person_tracks[0]
        assert track.original_bboxes is not None
        np.testing.assert_array_equal(track.original_bboxes, track.bboxes)

    def test_adds_keyframe_at_edit_frame(self, qapp):
        """Completing a bbox edit adds a keyframe if not already present."""
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)  # No keyframe at frame 30

        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)
        panel.on_frame_click(0.5, 0.8)

        track = session.person_tracks[0]
        kf_frames = [kf["frame"] for kf in track.keyframes]
        assert 30 in kf_frames

    def test_does_not_duplicate_keyframe(self, qapp):
        """Editing at an existing keyframe frame should not add duplicate."""
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(10)  # Existing keyframe

        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)
        panel.on_frame_click(0.5, 0.8)

        track = session.person_tracks[0]
        kf_at_10 = [kf for kf in track.keyframes if kf["frame"] == 10]
        assert len(kf_at_10) == 1

    def test_marks_person_dirty(self, qapp):
        """Completing a bbox edit marks the person as dirty."""
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)

        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)
        panel.on_frame_click(0.5, 0.8)

        assert 0 in session.dirty_persons


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------


class TestBBoxEditSignals:
    """Verify correct signals emitted during bbox editing."""

    def test_emits_person_dirty(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)

        received = []
        panel.person_dirty.connect(received.append)
        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)
        panel.on_frame_click(0.5, 0.8)

        assert received == [0]

    def test_emits_keyframe_changed(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)

        received = []
        panel.keyframe_changed.connect(lambda pid, f: received.append((pid, f)))
        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)
        panel.on_frame_click(0.5, 0.8)

        assert (0, 30) in received

    def test_emits_overlay_on_click1(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)

        received = []
        panel.bbox_overlay_changed.connect(received.append)
        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)

        assert len(received) == 1
        assert "edit_preview" in received[0]
        assert received[0]["edit_preview"]["corner1"] == (64, 96)

    def test_emits_overlay_clear_on_click2(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)

        received = []
        panel.bbox_overlay_changed.connect(received.append)
        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.2)
        panel.on_frame_click(0.5, 0.8)

        # Last emission should clear preview
        assert received[-1] == {"edit_preview": None}

    def test_emits_overlay_clear_on_cancel(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        received = []
        panel.bbox_overlay_changed.connect(received.append)
        panel._on_edit_bbox()
        panel._on_cancel_edit()

        assert {"edit_preview": None} in received


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


class TestInterpolation:
    """Verify delta-space bbox interpolation between corrected keyframes."""

    def test_interpolate_two_corrections(self):
        """Two corrections should interpolate frames between them."""
        n = 100
        original = np.array([[50, 60, 200, 300]] * n, dtype=float)
        corrections = np.zeros((n, 4), dtype=float)
        corrections[20] = [60, 70, 210, 310]  # delta = +10 each
        corrections[60] = [70, 80, 220, 320]  # delta = +20 each

        result = interpolate_bbox_corrections(original, corrections)

        # At corrected frames: exact values
        np.testing.assert_array_equal(result[20], [60, 70, 210, 310])
        np.testing.assert_array_equal(result[60], [70, 80, 220, 320])

        # Midpoint (frame 40): delta should be average of +10 and +20 = +15
        expected_40 = original[40] + 15
        np.testing.assert_array_almost_equal(result[40], expected_40)

    def test_interpolate_constant_extrapolation_before(self):
        """Frames before first correction use constant delta extrapolation."""
        n = 100
        original = np.array([[50, 60, 200, 300]] * n, dtype=float)
        corrections = np.zeros((n, 4), dtype=float)
        corrections[20] = [60, 70, 210, 310]  # delta = +10
        corrections[60] = [70, 80, 220, 320]  # delta = +20

        result = interpolate_bbox_corrections(original, corrections)

        # Frame 0 should have same delta as frame 20 (+10)
        expected_0 = original[0] + 10
        np.testing.assert_array_almost_equal(result[0], expected_0)

    def test_interpolate_constant_extrapolation_after(self):
        """Frames after last correction use constant delta extrapolation."""
        n = 100
        original = np.array([[50, 60, 200, 300]] * n, dtype=float)
        corrections = np.zeros((n, 4), dtype=float)
        corrections[20] = [60, 70, 210, 310]
        corrections[60] = [70, 80, 220, 320]  # delta = +20

        result = interpolate_bbox_corrections(original, corrections)

        # Frame 80 should have same delta as frame 60 (+20)
        expected_80 = original[80] + 20
        np.testing.assert_array_almost_equal(result[80], expected_80)

    def test_interpolate_single_correction_noop(self):
        """Single correction cannot interpolate — returns input unchanged."""
        n = 50
        original = np.array([[50, 60, 200, 300]] * n, dtype=float)
        corrections = np.zeros((n, 4), dtype=float)
        corrections[20] = [60, 70, 210, 310]

        result = interpolate_bbox_corrections(original, corrections)
        np.testing.assert_array_equal(result, corrections)

    def test_interpolate_no_corrections_noop(self):
        """No corrections — returns input unchanged."""
        n = 50
        original = np.array([[50, 60, 200, 300]] * n, dtype=float)
        corrections = np.zeros((n, 4), dtype=float)

        result = interpolate_bbox_corrections(original, corrections)
        np.testing.assert_array_equal(result, corrections)

    def test_interpolate_three_keyframes(self):
        """Three corrections should interpolate piecewise linearly."""
        n = 100
        original = np.array([[100, 100, 200, 200]] * n, dtype=float)
        corrections = np.zeros((n, 4), dtype=float)
        corrections[10] = [110, 100, 200, 200]  # delta = +10, 0, 0, 0
        corrections[30] = [120, 100, 200, 200]  # delta = +20, 0, 0, 0
        corrections[50] = [100, 100, 200, 200]  # delta = 0, 0, 0, 0

        result = interpolate_bbox_corrections(original, corrections)

        # At keyframes
        np.testing.assert_array_equal(result[10], [110, 100, 200, 200])
        np.testing.assert_array_equal(result[30], [120, 100, 200, 200])
        np.testing.assert_array_equal(result[50], [100, 100, 200, 200])

        # Midpoint between 10 and 30: delta = +15
        np.testing.assert_array_almost_equal(result[20], [115, 100, 200, 200])

        # Midpoint between 30 and 50: delta = +10
        np.testing.assert_array_almost_equal(result[40], [110, 100, 200, 200])

    def test_interpolate_via_inspector(self, qapp):
        """IdentityInspector._on_interpolate applies interpolation to track."""
        session = _session_with_corrections()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_interpolate()

        track = session.person_tracks[0]
        # Frame 40 (midpoint between 20 and 60) should be interpolated
        bbox_40 = track.bbox_corrections[40]
        assert not np.all(bbox_40 == 0)

    def test_interpolate_marks_dirty(self, qapp):
        session = _session_with_corrections()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_interpolate()
        assert 0 in session.dirty_persons

    def test_interpolate_noop_without_corrections(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        # Should not crash
        panel._on_interpolate()


# ---------------------------------------------------------------------------
# Apply / save corrections
# ---------------------------------------------------------------------------


class TestApplyCorrections:
    """Verify that Apply All saves corrections to disk."""

    def test_apply_saves_json(self, qapp, tmp_path):
        session = _session_with_corrections()
        person_dir = tmp_path / "person_0"
        person_dir.mkdir()
        session.person_tracks[0].person_dir = person_dir

        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel._on_apply_all()

        out_path = person_dir / "bbox_corrections.json"
        assert out_path.exists()

        data = json.loads(out_path.read_text())
        assert data["person_id"] == 0
        assert "20" in data["corrections"]
        assert "60" in data["corrections"]

    def test_apply_json_format(self, qapp, tmp_path):
        session = _session_with_corrections()
        person_dir = tmp_path / "person_0"
        person_dir.mkdir()
        session.person_tracks[0].person_dir = person_dir

        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel._on_apply_all()

        data = json.loads((person_dir / "bbox_corrections.json").read_text())
        bbox_20 = data["corrections"]["20"]
        assert len(bbox_20) == 4
        assert bbox_20 == [60.0, 70.0, 210.0, 310.0]

    def test_apply_skips_zero_corrections(self, qapp, tmp_path):
        session = _session_with_corrections()
        person_dir = tmp_path / "person_0"
        person_dir.mkdir()
        session.person_tracks[0].person_dir = person_dir

        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel._on_apply_all()

        data = json.loads((person_dir / "bbox_corrections.json").read_text())
        # Only frames 20 and 60 should be in corrections (rest are zeros)
        assert len(data["corrections"]) == 2

    def test_apply_noop_without_corrections(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        # Should not crash
        panel._on_apply_all()

    def test_apply_noop_without_person_dir(self, qapp):
        session = _session_with_corrections()
        # person_dir is None
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        # Should not crash
        panel._on_apply_all()

    def test_apply_emits_overlay_changed(self, qapp, tmp_path):
        session = _session_with_corrections()
        person_dir = tmp_path / "person_0"
        person_dir.mkdir()
        session.person_tracks[0].person_dir = person_dir

        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        received = []
        panel.bbox_overlay_changed.connect(received.append)
        panel._on_apply_all()

        assert {"applied": True} in received


# ---------------------------------------------------------------------------
# Edit preview rendering
# ---------------------------------------------------------------------------


class TestEditPreviewRendering:
    """Verify render_edit_preview draws markers on frame."""

    def test_crosshair_drawn_at_corner1(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = render_edit_preview(frame, {"corner1": (100, 200)})
        assert result is not frame  # copy made
        # Crosshair at (100, 200) should have drawn pixels
        assert result[200, 100].sum() > 0

    def test_none_frame_returns_none(self):
        result = render_edit_preview(None, {"corner1": (100, 200)})
        assert result is None

    def test_empty_state_returns_input(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = render_edit_preview(frame, {})
        assert result is frame  # empty dict is falsy → early return, no copy

    def test_none_state_returns_input(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = render_edit_preview(frame, None)
        assert result is frame  # early return, no copy

    def test_crosshair_clamped_near_edge(self):
        """Crosshair near frame edge should not go out of bounds."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = render_edit_preview(frame, {"corner1": (5, 5)})
        # Should not crash
        assert result is not None


# ---------------------------------------------------------------------------
# AppWindow bbox overlay wiring (migrated from deleted MultiPersonTab)
# ---------------------------------------------------------------------------


class TestAppWindowBBoxWiring:
    """Verify that AppWindow correctly wires bbox editing signals."""

    def test_frame_clicked_wired_to_inspector(self, qapp):
        from app_window import AppWindow

        window = AppWindow()

        # The frame_clicked signal should be connected to identity inspector
        assert window._video_player.frame_clicked is not None
        assert window._identity_inspector.on_frame_click is not None

    def test_edit_preview_stored_in_window(self, qapp):
        from app_window import AppWindow

        window = AppWindow()

        window._on_bbox_overlay_changed({"edit_preview": {"corner1": (100, 200)}})
        assert window._edit_preview == {"corner1": (100, 200)}

    def test_edit_preview_cleared(self, qapp):
        from app_window import AppWindow

        window = AppWindow()

        window._on_bbox_overlay_changed({"edit_preview": {"corner1": (100, 200)}})
        window._on_bbox_overlay_changed({"edit_preview": None})
        assert window._edit_preview is None

    def test_show_all_and_edit_preview_independent(self, qapp):
        """show_all and edit_preview should update independently."""
        from app_window import AppWindow

        window = AppWindow()

        window._on_bbox_overlay_changed({"show_all": True})
        assert window._show_all_tracks is True
        assert window._edit_preview is None

        window._on_bbox_overlay_changed({"edit_preview": {"corner1": (50, 50)}})
        assert window._show_all_tracks is True  # unchanged
        assert window._edit_preview == {"corner1": (50, 50)}


# ---------------------------------------------------------------------------
# Multiple edits
# ---------------------------------------------------------------------------


class TestMultipleEdits:
    """Verify multiple sequential bbox edits work correctly."""

    def test_two_edits_at_different_frames(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        # Edit frame 30
        panel.set_frame(30)
        panel._on_edit_bbox()
        panel.on_frame_click(0.1, 0.1)
        panel.on_frame_click(0.5, 0.5)

        # Edit frame 70
        panel.set_frame(70)
        panel._on_edit_bbox()
        panel.on_frame_click(0.2, 0.2)
        panel.on_frame_click(0.6, 0.6)

        track = session.person_tracks[0]
        assert not np.all(track.bbox_corrections[30] == 0)
        assert not np.all(track.bbox_corrections[70] == 0)

    def test_overwrite_existing_correction(self, qapp):
        session = _session_with_corrections()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(20)

        # Overwrite the existing correction at frame 20
        panel._on_edit_bbox()
        panel.on_frame_click(0.3, 0.3)
        panel.on_frame_click(0.7, 0.7)

        track = session.person_tracks[0]
        bbox = track.bbox_corrections[20]
        # Should be the new coords, not the old [60, 70, 210, 310]
        assert bbox[0] != 60
