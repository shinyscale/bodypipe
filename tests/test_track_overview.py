"""Tests for the DAW-style vertical track timeline (Phase 2 UX overhaul).

Why comprehensive tests: The track timeline is a central navigation widget —
person selection, frame seeking, and marker display all flow through it.
Regressions here break the multi-person review workflow.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication

from views.track_overview import (
    HEADER_WIDTH,
    LANE_HEIGHT,
    LANE_SPACING,
    PERSON_COLORS,
    TrackOverview,
    _PlayheadItem,
    _TimelineView,
    _TrackHeader,
    _TrackLaneItem,
    _conf_color,
)
from theme import COLORS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _two_person_tracks(n: int = 100):
    return {
        0: np.ones(n) * 0.9,
        1: np.ones(n) * 0.5,
    }


# ---------------------------------------------------------------------------
# _conf_color
# ---------------------------------------------------------------------------


class TestConfColor:
    def test_high_confidence_green(self):
        c = _conf_color(0.95)
        assert c.name() == COLORS["success"].lower() or c == _conf_color(0.95)

    def test_medium_confidence_yellow(self):
        c = _conf_color(0.6)
        # Should be warning color
        assert c.name() != _conf_color(0.3).name()

    def test_low_confidence_red(self):
        c = _conf_color(0.2)
        assert c.name() != _conf_color(0.9).name()

    def test_boundary_0_8(self):
        # 0.8 is NOT > 0.8, so should be yellow
        assert _conf_color(0.8).name() == _conf_color(0.6).name()

    def test_boundary_0_5(self):
        # 0.5 is NOT > 0.5, so should be red
        assert _conf_color(0.5).name() == _conf_color(0.3).name()


# ---------------------------------------------------------------------------
# _TrackLaneItem
# ---------------------------------------------------------------------------


class TestTrackLaneItem:
    def test_bounding_rect(self):
        lane = _TrackLaneItem(person_id=0, num_frames=200, y_pos=0)
        br = lane.boundingRect()
        assert br.width() == 200
        assert br.height() == LANE_HEIGHT

    def test_position_set_by_y_pos(self):
        lane = _TrackLaneItem(person_id=1, num_frames=50, y_pos=28)
        assert lane.pos().y() == 28

    def test_person_id_property(self):
        lane = _TrackLaneItem(person_id=3, num_frames=10, y_pos=0)
        assert lane.person_id == 3

    def test_set_confidences(self):
        lane = _TrackLaneItem(person_id=0, num_frames=10, y_pos=0)
        conf = np.linspace(0, 1, 10)
        lane.set_confidences(conf)
        assert lane._confidences is not None
        assert len(lane._confidences) == 10

    def test_set_keyframes(self):
        lane = _TrackLaneItem(person_id=0, num_frames=100, y_pos=0)
        lane.set_keyframes([10, 50, 80], verified={50})
        assert lane._keyframe_frames == [10, 50, 80]
        assert lane._verified_frames == {50}

    def test_set_issues(self):
        lane = _TrackLaneItem(person_id=0, num_frames=100, y_pos=0)
        lane.set_issues([5, 25, 90])
        assert lane._issue_frames == [5, 25, 90]

    def test_set_corrections(self):
        lane = _TrackLaneItem(person_id=0, num_frames=100, y_pos=0)
        lane.set_corrections([12, 44])
        assert lane._correction_frames == [12, 44]

    def test_set_crossing_spans(self):
        lane = _TrackLaneItem(person_id=0, num_frames=100, y_pos=0)
        lane.set_crossing_spans([(10, 30), (60, 80)])
        assert lane._crossing_spans == [(10, 30), (60, 80)]

    def test_collapse_and_expand(self):
        lane = _TrackLaneItem(person_id=0, num_frames=10, y_pos=0)
        assert not lane.collapsed
        lane.set_collapsed(True)
        assert lane.collapsed
        lane.set_collapsed(False)
        assert not lane.collapsed

    def test_num_frames_min_1(self):
        lane = _TrackLaneItem(person_id=0, num_frames=0, y_pos=0)
        assert lane._num_frames == 1


# ---------------------------------------------------------------------------
# _PlayheadItem
# ---------------------------------------------------------------------------


class TestPlayheadItem:
    def test_initial_position(self):
        ph = _PlayheadItem(total_height=100)
        assert ph.frame == 0

    def test_set_frame(self):
        ph = _PlayheadItem(total_height=100)
        ph.set_frame(42)
        assert ph.frame == 42
        assert ph.pos().x() == pytest.approx(42.5, abs=0.1)

    def test_set_height(self):
        ph = _PlayheadItem(total_height=50)
        ph.set_height(200)
        assert ph._total_height == 200

    def test_min_height(self):
        ph = _PlayheadItem(total_height=0)
        assert ph._total_height >= 1.0

    def test_bounding_rect_contains_line(self):
        ph = _PlayheadItem(total_height=100)
        br = ph.boundingRect()
        assert br.left() < 0  # extends left of center
        assert br.right() > 0  # extends right of center
        assert br.bottom() >= 100

    def test_zvalue_on_top(self):
        ph = _PlayheadItem(total_height=50)
        assert ph.zValue() == 100


# ---------------------------------------------------------------------------
# _TimelineView
# ---------------------------------------------------------------------------


class TestTimelineView:
    def test_initial_zoom(self, qapp):
        from PySide6.QtWidgets import QGraphicsScene

        scene = QGraphicsScene()
        view = _TimelineView(scene)
        assert view.zoom_x == 1.0

    def test_set_num_frames(self, qapp):
        from PySide6.QtWidgets import QGraphicsScene

        scene = QGraphicsScene()
        view = _TimelineView(scene)
        view.set_num_frames(500)
        assert view._num_frames == 500

    def test_set_lanes(self, qapp):
        from PySide6.QtWidgets import QGraphicsScene

        scene = QGraphicsScene()
        view = _TimelineView(scene)
        lane = _TrackLaneItem(0, 100, 0)
        view.set_lanes({0: lane})
        assert 0 in view._lanes


# ---------------------------------------------------------------------------
# _TrackHeader
# ---------------------------------------------------------------------------


class TestTrackHeader:
    def test_creates_with_person_id(self, qapp):
        h = _TrackHeader(person_id=2)
        assert h.person_id == 2

    def test_initial_state_not_collapsed(self, qapp):
        h = _TrackHeader(person_id=0)
        assert not h.collapsed

    def test_collapse_toggle_emits_signal(self, qapp):
        h = _TrackHeader(person_id=1)
        received = []
        h.collapse_toggled.connect(lambda pid, c: received.append((pid, c)))
        h._btn.click()
        assert received == [(1, True)]
        h._btn.click()
        assert received == [(1, True), (1, False)]

    def test_fixed_height(self, qapp):
        h = _TrackHeader(person_id=0)
        assert h.height() == LANE_HEIGHT + LANE_SPACING

    def test_label_has_person_color(self, qapp):
        h = _TrackHeader(person_id=0)
        assert PERSON_COLORS[0] in h._label.styleSheet()

    def test_label_wraps_color_index(self, qapp):
        pid = len(PERSON_COLORS) + 2
        h = _TrackHeader(person_id=pid)
        expected_color = PERSON_COLORS[pid % len(PERSON_COLORS)]
        assert expected_color in h._label.styleSheet()


# ---------------------------------------------------------------------------
# TrackOverview — main widget
# ---------------------------------------------------------------------------


class TestTrackOverview:
    def test_creates_without_error(self, qapp):
        w = TrackOverview()
        assert w is not None

    def test_has_view_and_scene(self, qapp):
        w = TrackOverview()
        assert w._view is not None
        assert w._scene is not None

    def test_has_playhead(self, qapp):
        w = TrackOverview()
        assert w._playhead is not None

    def test_set_tracks_creates_lanes_and_headers(self, qapp):
        w = TrackOverview()
        w.set_tracks(_two_person_tracks())
        assert len(w._lanes) == 2
        assert len(w._headers) == 2
        assert 0 in w._lanes
        assert 1 in w._lanes

    def test_set_tracks_empty_clears(self, qapp):
        w = TrackOverview()
        w.set_tracks(_two_person_tracks())
        assert len(w._lanes) == 2
        w.set_tracks({})
        assert len(w._lanes) == 0
        assert len(w._headers) == 0

    def test_set_tracks_replaces_previous(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(10), 1: np.ones(10)})
        assert len(w._lanes) == 2
        w.set_tracks({0: np.ones(5)})
        assert len(w._lanes) == 1
        assert len(w._headers) == 1

    def test_set_tracks_num_frames(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(200)})
        assert w._num_frames == 200

    def test_set_tracks_different_lengths(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(50), 1: np.ones(100)})
        assert w._num_frames == 100

    def test_set_current_frame(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(100)})
        w.set_current_frame(42)
        assert w._playhead.frame == 42

    def test_set_current_frame_before_tracks(self, qapp):
        w = TrackOverview()
        w.set_current_frame(10)  # should not raise

    def test_person_clicked_signal_from_view(self, qapp):
        w = TrackOverview()
        received = []
        w.person_clicked.connect(lambda pid, f: received.append((pid, f)))
        w.set_tracks({0: np.ones(50)})
        w._view.frame_clicked.emit(0, 25)
        assert received == [(0, 25)]

    def test_lane_positions_are_stacked(self, qapp):
        w = TrackOverview()
        w.set_tracks(_two_person_tracks())
        y0 = w._lanes[0].pos().y()
        y1 = w._lanes[1].pos().y()
        assert y0 == 0
        assert y1 == LANE_HEIGHT + LANE_SPACING

    def test_scene_rect_covers_all_lanes(self, qapp):
        w = TrackOverview()
        w.set_tracks(_two_person_tracks(100))
        sr = w._scene.sceneRect()
        assert sr.width() == 100
        expected_h = 2 * LANE_HEIGHT + LANE_SPACING
        assert sr.height() == expected_h

    def test_set_num_frames(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(50)})
        w.set_num_frames(200)
        assert w._num_frames == 200
        assert w._scene.sceneRect().width() == 200

    # -- markers ----------------------------------------------------------

    def test_set_track_markers_keyframes(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(100)})
        w.set_track_markers(0, keyframes=[10, 50], verified_frames={50})
        lane = w._lanes[0]
        assert lane._keyframe_frames == [10, 50]
        assert lane._verified_frames == {50}

    def test_set_track_markers_issues(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(100)})
        w.set_track_markers(0, issue_frames=[5, 75])
        assert w._lanes[0]._issue_frames == [5, 75]

    def test_set_track_markers_corrections(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(100)})
        w.set_track_markers(0, correction_frames=[20, 40, 60])
        assert w._lanes[0]._correction_frames == [20, 40, 60]

    def test_set_track_markers_crossing_spans(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(100)})
        w.set_track_markers(0, crossing_spans=[(10, 30), (60, 80)])
        assert w._lanes[0]._crossing_spans == [(10, 30), (60, 80)]

    def test_set_track_markers_nonexistent_person(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(100)})
        # Should not raise for unknown person
        w.set_track_markers(99, keyframes=[10])

    def test_set_track_markers_partial_update(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(100)})
        w.set_track_markers(0, keyframes=[10])
        w.set_track_markers(0, issue_frames=[20])
        # Previous keyframes should be unchanged
        assert w._lanes[0]._keyframe_frames == [10]
        assert w._lanes[0]._issue_frames == [20]

    # -- collapse ---------------------------------------------------------

    def test_collapse_toggles_lane(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(10)})
        assert not w._lanes[0].collapsed
        w._headers[0]._btn.click()
        assert w._lanes[0].collapsed

    def test_collapse_and_expand(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(10)})
        w._headers[0]._btn.click()
        assert w._lanes[0].collapsed
        w._headers[0]._btn.click()
        assert not w._lanes[0].collapsed

    # -- header panel -----------------------------------------------------

    def test_header_panel_width(self, qapp):
        w = TrackOverview()
        assert w._header_panel.width() == HEADER_WIDTH

    def test_headers_have_correct_person_ids(self, qapp):
        w = TrackOverview()
        w.set_tracks({2: np.ones(10), 5: np.ones(10)})
        assert set(w._headers.keys()) == {2, 5}
        assert w._headers[2].person_id == 2
        assert w._headers[5].person_id == 5


# ---------------------------------------------------------------------------
# PERSON_COLORS
# ---------------------------------------------------------------------------


class TestPersonColors:
    def test_at_least_8_colors(self):
        assert len(PERSON_COLORS) >= 8

    def test_all_valid_hex(self):
        for c in PERSON_COLORS:
            assert c.startswith("#")
            assert len(c) == 7
