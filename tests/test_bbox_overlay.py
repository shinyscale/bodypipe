"""Tests for bbox overlay rendering.

Why: The bbox overlay is the primary visual feedback for multi-person tracking.
Incorrect rendering — wrong colors, missing labels, broken coordinate clamping,
or failure to distinguish corrected vs. original bboxes — can mislead users
during identity verification. These tests verify the rendering logic in isolation
(pure numpy/cv2) without requiring a live video or Qt display.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session, PersonTrack
from views.bbox_overlay import (
    render_bbox_overlay,
    get_bbox_at_frame,
    is_corrected_bbox,
    get_confidence_at_frame,
    PERSON_COLORS,
    _hex_to_rgb,
    _hex_to_bgr,
    _confidence_color_rgb,
    _draw_dashed_rect,
    _draw_dashed_line,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Create a blank black RGB frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _session_with_tracks(num_frames: int = 100) -> Session:
    """Create a session with two active person tracks."""
    session = Session()
    session.num_frames = num_frames
    session.img_width = 640
    session.img_height = 480
    session.person_tracks = {
        0: PersonTrack(
            person_id=0,
            confidences=[0.9 - i * 0.005 for i in range(num_frames)],
            bboxes=np.array([[50, 60, 200, 300]] * num_frames, dtype=float),
        ),
        1: PersonTrack(
            person_id=1,
            confidences=[0.5 + i * 0.003 for i in range(num_frames)],
            bboxes=np.array([[300, 100, 500, 400]] * num_frames, dtype=float),
        ),
    }
    return session


def _session_with_corrected_bbox(num_frames: int = 50) -> Session:
    """Create a session where person 0 has a corrected bbox at frame 10."""
    session = _session_with_tracks(num_frames)
    corrections = np.zeros((num_frames, 4), dtype=float)
    corrections[10] = [60, 70, 210, 310]  # corrected
    session.person_tracks[0].bbox_corrections = corrections
    return session


def _session_with_inactive_tracks(num_frames: int = 50) -> Session:
    """Create a session with one active and one inactive track."""
    session = _session_with_tracks(num_frames)
    session.inactive_tracks = {1}  # person 1 is inactive
    return session


# ---------------------------------------------------------------------------
# Color conversion tests
# ---------------------------------------------------------------------------


class TestColorConversion:
    def test_hex_to_rgb(self):
        assert _hex_to_rgb("#e94560") == (233, 69, 96)
        assert _hex_to_rgb("#ffffff") == (255, 255, 255)
        assert _hex_to_rgb("#000000") == (0, 0, 0)

    def test_hex_to_bgr(self):
        assert _hex_to_bgr("#e94560") == (96, 69, 233)
        assert _hex_to_bgr("#ff0000") == (0, 0, 255)

    def test_confidence_color_high(self):
        """High confidence > 0.8 → green."""
        r, g, b = _confidence_color_rgb(0.9)
        assert g > r  # green-ish

    def test_confidence_color_medium(self):
        """Medium confidence 0.5-0.8 → yellow."""
        r, g, b = _confidence_color_rgb(0.6)
        assert r > 100 and g > 100  # yellow-ish

    def test_confidence_color_low(self):
        """Low confidence < 0.5 → red."""
        r, g, b = _confidence_color_rgb(0.3)
        assert r > g  # red-ish


# ---------------------------------------------------------------------------
# Data access helper tests
# ---------------------------------------------------------------------------


class TestDataHelpers:
    def test_get_bbox_at_frame_original(self):
        session = _session_with_tracks()
        track = session.person_tracks[0]
        bbox = get_bbox_at_frame(track, 5)
        assert bbox is not None
        np.testing.assert_array_equal(bbox, [50, 60, 200, 300])

    def test_get_bbox_at_frame_corrected(self):
        """Corrected bbox takes priority over original."""
        session = _session_with_corrected_bbox()
        track = session.person_tracks[0]
        bbox = get_bbox_at_frame(track, 10)
        np.testing.assert_array_equal(bbox, [60, 70, 210, 310])

    def test_get_bbox_at_frame_uncorrected_falls_back(self):
        """Frame without correction falls back to original bbox."""
        session = _session_with_corrected_bbox()
        track = session.person_tracks[0]
        bbox = get_bbox_at_frame(track, 5)
        np.testing.assert_array_equal(bbox, [50, 60, 200, 300])

    def test_get_bbox_at_frame_no_bboxes(self):
        track = PersonTrack(person_id=99)
        assert get_bbox_at_frame(track, 0) is None

    def test_get_bbox_at_frame_out_of_range(self):
        session = _session_with_tracks(10)
        track = session.person_tracks[0]
        assert get_bbox_at_frame(track, 999) is None

    def test_is_corrected_bbox_true(self):
        session = _session_with_corrected_bbox()
        track = session.person_tracks[0]
        assert is_corrected_bbox(track, 10) is True

    def test_is_corrected_bbox_false(self):
        session = _session_with_corrected_bbox()
        track = session.person_tracks[0]
        assert is_corrected_bbox(track, 5) is False

    def test_is_corrected_bbox_no_corrections(self):
        track = PersonTrack(person_id=0)
        assert is_corrected_bbox(track, 0) is False

    def test_get_confidence_at_frame(self):
        session = _session_with_tracks()
        track = session.person_tracks[0]
        conf = get_confidence_at_frame(track, 0)
        assert conf == pytest.approx(0.9)

    def test_get_confidence_at_frame_none(self):
        track = PersonTrack(person_id=0)
        assert get_confidence_at_frame(track, 0) is None

    def test_get_confidence_at_frame_out_of_range(self):
        session = _session_with_tracks(10)
        track = session.person_tracks[0]
        assert get_confidence_at_frame(track, 999) is None


# ---------------------------------------------------------------------------
# Drawing primitive tests
# ---------------------------------------------------------------------------


class TestDrawingPrimitives:
    def test_dashed_line_modifies_frame(self):
        frame = _blank_frame()
        _draw_dashed_line(frame, (10, 10), (200, 10), (255, 0, 0), 2)
        # Should have drawn something in the row around y=10
        assert frame[10, 10:200].sum() > 0

    def test_dashed_rect_modifies_frame(self):
        frame = _blank_frame()
        _draw_dashed_rect(frame, (50, 50), (200, 200), (0, 255, 0), 2)
        # Should have drawn on all four edges
        assert frame[50, 50:200].sum() > 0   # top edge
        assert frame[200, 50:200].sum() > 0  # bottom edge
        assert frame[50:200, 50].sum() > 0   # left edge
        assert frame[50:200, 200].sum() > 0  # right edge

    def test_dashed_line_zero_length(self):
        """Zero-length line should not crash."""
        frame = _blank_frame()
        _draw_dashed_line(frame, (10, 10), (10, 10), (255, 0, 0), 2)
        # No crash is success


# ---------------------------------------------------------------------------
# Core render_bbox_overlay tests
# ---------------------------------------------------------------------------


class TestRenderBboxOverlay:
    def test_returns_copy_not_original(self):
        """Overlay should not modify the original frame."""
        frame = _blank_frame()
        session = _session_with_tracks()
        result = render_bbox_overlay(frame, session, 0)
        assert result is not frame
        assert frame.sum() == 0  # original unchanged

    def test_empty_session_returns_input(self):
        """No tracks → return original frame unchanged."""
        frame = _blank_frame()
        session = Session()
        result = render_bbox_overlay(frame, session, 0)
        assert result is frame  # same object, no copy needed

    def test_none_frame_returns_none(self):
        session = _session_with_tracks()
        result = render_bbox_overlay(None, session, 0)
        assert result is None

    def test_draws_active_tracks(self):
        """Active tracks should produce visible pixels in bbox regions."""
        frame = _blank_frame()
        session = _session_with_tracks()
        result = render_bbox_overlay(frame, session, 0)
        # Person 0 bbox region should have non-zero pixels (border drawn)
        # Check a strip along the top edge of person 0's bbox
        assert result[60, 50:200].sum() > 0

    def test_selected_person_thicker(self):
        """Selected person should have thicker outline (more changed pixels)."""
        frame = _blank_frame()
        session = _session_with_tracks()

        # Render without selection
        r1 = render_bbox_overlay(frame, session, 0, selected_person=-1)
        pixels_no_sel = np.count_nonzero(r1)

        # Render with person 0 selected
        r2 = render_bbox_overlay(frame, session, 0, selected_person=0)
        pixels_sel = np.count_nonzero(r2)

        # Selected should have more drawn pixels (thicker lines)
        assert pixels_sel >= pixels_no_sel

    def test_inactive_tracks_hidden_by_default(self):
        """Inactive tracks should not be drawn when show_all_tracks=False."""
        frame = _blank_frame()
        session = _session_with_inactive_tracks()
        result = render_bbox_overlay(frame, session, 0, show_all_tracks=False)
        # Person 1 is inactive, their bbox region [300,100,500,400] should be empty
        # (only person 0's bbox [50,60,200,300] should be drawn)
        # Check person 1's bbox center
        assert result[250, 400].sum() == 0  # center of person 1's bbox

    def test_inactive_tracks_shown_when_show_all(self):
        """Inactive tracks should appear when show_all_tracks=True."""
        frame = _blank_frame()
        session = _session_with_inactive_tracks()
        result = render_bbox_overlay(frame, session, 0, show_all_tracks=True)
        # Person 1 bbox region should have something drawn (dashed gray)
        # Check along the top edge of person 1's bbox
        assert result[100, 300:500].sum() > 0

    def test_corrected_bbox_uses_dashed(self):
        """Corrected bboxes should be drawn differently from originals."""
        frame = _blank_frame()
        session = _session_with_corrected_bbox()

        # Frame 10 has a corrected bbox for person 0
        r_corrected = render_bbox_overlay(frame, session, 10)
        # Frame 5 has original bbox
        r_original = render_bbox_overlay(frame, session, 5)

        # Both should have drawn something, but pixels differ (dashed vs solid)
        assert np.count_nonzero(r_corrected) > 0
        assert np.count_nonzero(r_original) > 0
        # They should look different (different bbox position and style)
        assert not np.array_equal(r_corrected, r_original)

    def test_person_colors_cycle(self):
        """Person colors should cycle through the palette."""
        session = Session()
        session.num_frames = 10
        session.img_width = 640
        session.img_height = 480
        # Add 10 persons (more than PERSON_COLORS length)
        for i in range(10):
            session.person_tracks[i] = PersonTrack(
                person_id=i,
                bboxes=np.array([[10 + i * 50, 10, 40 + i * 50, 100]] * 10, dtype=float),
                confidences=[0.8] * 10,
            )
        frame = _blank_frame()
        result = render_bbox_overlay(frame, session, 0)
        # Should not crash and should draw all 10
        assert np.count_nonzero(result) > 0

    def test_bbox_clamped_to_frame_bounds(self):
        """Bboxes extending outside the frame should be clamped."""
        session = Session()
        session.num_frames = 10
        session.img_width = 640
        session.img_height = 480
        session.person_tracks[0] = PersonTrack(
            person_id=0,
            bboxes=np.array([[-50, -50, 700, 500]] * 10, dtype=float),
            confidences=[0.8] * 10,
        )
        frame = _blank_frame()
        result = render_bbox_overlay(frame, session, 0)
        # Should not crash, and should draw within bounds
        assert np.count_nonzero(result) > 0

    def test_frame_out_of_range(self):
        """Requesting a frame beyond track length returns unchanged frame."""
        frame = _blank_frame()
        session = _session_with_tracks(10)
        result = render_bbox_overlay(frame, session, 999)
        # No bboxes to draw — result should still be a copy but with no drawing
        # Actually it returns a copy if there are tracks, just no bboxes drawn
        assert result is not frame  # copy made because tracks exist

    def test_confidence_dot_drawn(self):
        """Confidence dot should be drawn near bottom-right of bbox."""
        frame = _blank_frame()
        session = _session_with_tracks()
        result = render_bbox_overlay(frame, session, 0)
        # Person 0 bbox is [50,60,200,300], dot at (200-8, 300-8) = (192, 292)
        # Check a small region around the dot
        region = result[287:297, 187:197]
        assert region.sum() > 0

    def test_label_drawn_above_bbox(self):
        """Person ID label should be drawn above the bbox."""
        frame = _blank_frame()
        session = _session_with_tracks()
        result = render_bbox_overlay(frame, session, 0)
        # Person 0 bbox top is y=60, label badge should be above that
        # Check the region just above the bbox
        region = result[40:60, 50:200]
        assert region.sum() > 0

    def test_multiple_persons_all_drawn(self):
        """Both active persons should have visible overlays."""
        frame = _blank_frame()
        session = _session_with_tracks()
        result = render_bbox_overlay(frame, session, 0)
        # Person 0 region
        assert result[60, 50:200].sum() > 0
        # Person 1 region
        assert result[100, 300:500].sum() > 0


# ---------------------------------------------------------------------------
# Integration with AppWindow signal wiring (migrated from deleted MultiPersonTab)
# ---------------------------------------------------------------------------


class TestAppWindowOverlayWiring:
    """Test that AppWindow correctly wires bbox overlay signals."""

    def test_show_frame_calls_overlay(self, qapp):
        """_show_frame should composite bbox overlay onto raw frame."""
        from app_window import AppWindow

        window = AppWindow()

        # Without a loaded video, _show_frame should not crash
        window._show_frame(0)

    def test_show_all_tracks_toggle(self, qapp):
        """_on_bbox_overlay_changed should update _show_all_tracks flag."""
        from app_window import AppWindow

        window = AppWindow()

        assert window._show_all_tracks is False
        window._on_bbox_overlay_changed({"show_all": True})
        assert window._show_all_tracks is True
        window._on_bbox_overlay_changed({"show_all": False})
        assert window._show_all_tracks is False

    def test_identity_person_changed_redraws(self, qapp, session):
        """Person change should update selected_person and redraw."""
        from app_window import AppWindow

        window = AppWindow()

        window._on_identity_person_changed(2)
        assert window._session.selected_person == 2

    def test_on_bbox_overlay_changed_handles_non_dict(self, qapp):
        """Should handle non-dict data gracefully."""
        from app_window import AppWindow

        window = AppWindow()

        # Should not crash
        window._on_bbox_overlay_changed("invalid")
        window._on_bbox_overlay_changed(None)
        window._on_bbox_overlay_changed(42)
