"""Tests for Identity Inspector panel.

Why: The Identity Inspector is the most complex panel in the multi-person
workflow — it manages person selection, confidence visualization, and keyframe
CRUD operations. These tests verify that the widget correctly reads from
Session.person_tracks, updates all sub-displays on person/frame changes,
and properly mutates keyframe state through CRUD operations. All tests run
headless (offscreen) without a GPU or video backend.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session, PersonTrack
from views.identity_inspector import (
    IdentityInspector,
    CONFIDENCE_METRICS,
    CONFIDENCE_LABELS,
    PERSON_COLORS,
    _confidence_color,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_with_tracks(num_frames: int = 100) -> Session:
    """Create a session with two person tracks for testing."""
    session = Session()
    session.num_frames = num_frames
    session.person_tracks = {
        0: PersonTrack(
            person_id=0,
            confidences=[0.9 - i * 0.005 for i in range(num_frames)],
            keyframes=[
                {"frame": 10, "verified": True},
                {"frame": 50, "verified": False},
                {"frame": 80, "verified": True},
            ],
            bboxes=np.array([[10, 20, 110, 220]] * num_frames, dtype=float),
        ),
        1: PersonTrack(
            person_id=1,
            confidences=[0.5 + i * 0.003 for i in range(num_frames)],
            keyframes=[
                {"frame": 20, "verified": False},
            ],
        ),
    }
    return session


def _session_with_breakdown(num_frames: int = 50) -> Session:
    """Create a session with confidence breakdown data."""
    session = Session()
    session.num_frames = num_frames
    session.person_tracks = {
        0: PersonTrack(
            person_id=0,
            confidences=[0.85] * num_frames,
            confidence_breakdown={
                "detection": [0.92] * num_frames,
                "visibility": [0.85] * num_frames,
                "overlap": [0.12] * num_frames,
                "shape": [0.95] * num_frames,
                "motion": [0.88] * num_frames,
                "overall": [0.87] * num_frames,
            },
        ),
    }
    return session


# ---------------------------------------------------------------------------
# Widget construction
# ---------------------------------------------------------------------------


class TestIdentityInspectorConstruction:
    """Verify that the widget creates correctly with all expected children."""

    def test_creates_without_error(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert panel is not None

    def test_has_person_combo(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert panel._person_combo is not None

    def test_has_show_all_tracks_checkbox(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert panel._show_all_tracks is not None

    def test_has_confidence_timeline(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert panel._timeline is not None

    def test_has_six_confidence_labels(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert len(panel._conf_labels) == 6
        for metric in CONFIDENCE_METRICS:
            assert metric in panel._conf_labels

    def test_has_keyframe_table_with_5_columns(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert panel._keyframe_table is not None
        assert panel._keyframe_table.columnCount() == 5

    def test_keyframe_table_headers(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        headers = []
        for col in range(5):
            item = panel._keyframe_table.horizontalHeaderItem(col)
            headers.append(item.text() if item else "")
        assert headers == ["Frame", "Verified", "Confidence", "BBox", "Actions"]

    def test_has_verify_button(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert panel._verify_btn is not None

    def test_has_add_kf_button(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert panel._add_kf_btn is not None

    def test_has_remove_kf_button(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert panel._remove_kf_btn is not None

    def test_has_prev_next_buttons(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert panel._prev_kf_btn is not None
        assert panel._next_kf_btn is not None

    def test_has_expected_signals(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert hasattr(panel, "person_changed")
        assert hasattr(panel, "frame_requested")
        assert hasattr(panel, "person_dirty")
        assert hasattr(panel, "bbox_overlay_changed")
        assert hasattr(panel, "keyframe_changed")

    def test_empty_session_no_crash(self, qapp):
        """Empty session should not crash — combo empty, table empty, labels show dashes."""
        session = Session()
        panel = IdentityInspector(session)
        panel.refresh()
        assert panel._person_combo.count() == 0
        assert panel._keyframe_table.rowCount() == 0


# ---------------------------------------------------------------------------
# Person selector
# ---------------------------------------------------------------------------


class TestPersonSelector:
    """Verify person combo box population and selection behavior."""

    def test_empty_when_no_tracks(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        panel.refresh()
        assert panel._person_combo.count() == 0

    def test_populates_with_person_tracks(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        assert panel._person_combo.count() == 2
        assert panel._person_combo.itemData(0) == 0
        assert panel._person_combo.itemData(1) == 1

    def test_excludes_inactive_tracks(self, qapp):
        session = _session_with_tracks()
        session.inactive_tracks.add(1)
        panel = IdentityInspector(session)
        panel.refresh()
        assert panel._person_combo.count() == 1
        assert panel._person_combo.itemData(0) == 0

    def test_set_person_updates_combo(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(1)
        assert panel._person_combo.currentData() == 1

    def test_set_person_idempotent(self, qapp):
        """Setting same person twice should not re-trigger refresh."""
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        # No crash, no duplicate signals
        panel.set_person(0)

    def test_combo_change_emits_person_changed(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()

        received = []
        panel.person_changed.connect(received.append)

        # Simulate selecting person 1
        panel._person_combo.setCurrentIndex(1)

        assert received == [1]

    def test_combo_change_updates_session(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel._person_combo.setCurrentIndex(1)
        assert session.selected_person == 1

    def test_refresh_preserves_selection(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(1)
        # Refresh again — should keep person 1 selected
        panel.refresh()
        assert panel._person_combo.currentData() == 1

    def test_combo_labels(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        assert panel._person_combo.itemText(0) == "Person 0"
        assert panel._person_combo.itemText(1) == "Person 1"


# ---------------------------------------------------------------------------
# Confidence breakdown
# ---------------------------------------------------------------------------


class TestConfidenceBreakdown:
    """Verify the 6 confidence labels update with correct values and colors."""

    def test_shows_dashes_when_no_track(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        panel.set_frame(0)
        for metric in CONFIDENCE_METRICS:
            assert "\u2014" in panel._conf_labels[metric].text()

    def test_shows_values_from_breakdown(self, qapp):
        session = _session_with_breakdown()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(0)
        assert "0.92" in panel._conf_labels["detection"].text()
        assert "0.85" in panel._conf_labels["visibility"].text()
        assert "0.12" in panel._conf_labels["overlap"].text()
        assert "0.95" in panel._conf_labels["shape"].text()
        assert "0.88" in panel._conf_labels["motion"].text()
        assert "0.87" in panel._conf_labels["overall"].text()

    def test_overall_falls_back_to_confidences(self, qapp):
        """When no breakdown exists, 'overall' should use the confidences array."""
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(0)
        # Person 0 has confidences starting at 0.9, no breakdown
        assert "0.90" in panel._conf_labels["overall"].text()

    def test_non_overall_shows_dash_without_breakdown(self, qapp):
        """Non-overall metrics should show dash when no breakdown exists."""
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(0)
        assert "\u2014" in panel._conf_labels["detection"].text()

    def test_color_green_for_high(self, qapp):
        assert _confidence_color(0.9) == "#4ecca3"

    def test_color_yellow_for_medium(self, qapp):
        assert _confidence_color(0.65) == "#ffd93d"

    def test_color_red_for_low(self, qapp):
        assert _confidence_color(0.3) == "#ff6b6b"

    def test_frame_out_of_range_shows_dash(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(9999)  # beyond num_frames
        assert "\u2014" in panel._conf_labels["overall"].text()


# ---------------------------------------------------------------------------
# Keyframe table
# ---------------------------------------------------------------------------


class TestKeyframeTable:
    """Verify keyframe table population and display."""

    def test_empty_when_no_person(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert panel._keyframe_table.rowCount() == 0

    def test_populates_sorted_by_frame(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        # Person 0 has keyframes at 10, 50, 80
        assert panel._keyframe_table.rowCount() == 3
        assert panel._keyframe_table.item(0, 0).text() == "10"
        assert panel._keyframe_table.item(1, 0).text() == "50"
        assert panel._keyframe_table.item(2, 0).text() == "80"

    def test_shows_verified_status(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        # Frame 10: verified=True, Frame 50: verified=False
        assert "\u2713" in panel._keyframe_table.item(0, 1).text()
        assert "\u2717" in panel._keyframe_table.item(1, 1).text()

    def test_shows_confidence_values(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        # Frame 10 confidence: 0.9 - 10*0.005 = 0.85
        conf_text = panel._keyframe_table.item(0, 2).text()
        assert "0.85" in conf_text

    def test_shows_bbox_dimensions(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        # Bbox is [10, 20, 110, 220] → 100x200
        bbox_text = panel._keyframe_table.item(0, 3).text()
        assert "100x200" in bbox_text

    def test_shows_dash_when_no_bbox(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(1)  # Person 1 has no bboxes
        # Frame 20 keyframe
        bbox_text = panel._keyframe_table.item(0, 3).text()
        assert "\u2014" in bbox_text

    def test_has_action_buttons(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        # Check action widgets exist for each row
        for row in range(panel._keyframe_table.rowCount()):
            widget = panel._keyframe_table.cellWidget(row, 4)
            assert widget is not None

    def test_switching_person_updates_table(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        assert panel._keyframe_table.rowCount() == 3
        panel.set_person(1)
        assert panel._keyframe_table.rowCount() == 1


# ---------------------------------------------------------------------------
# Keyframe CRUD
# ---------------------------------------------------------------------------


class TestKeyframeCRUD:
    """Verify add, remove, and verify operations on keyframes."""

    def test_add_keyframe(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)  # Not an existing keyframe

        panel._on_add_keyframe()

        track = session.person_tracks[0]
        frames = [kf["frame"] for kf in track.keyframes]
        assert 30 in frames
        assert panel._keyframe_table.rowCount() == 4

    def test_add_keyframe_is_unverified(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)

        panel._on_add_keyframe()

        track = session.person_tracks[0]
        kf = [k for k in track.keyframes if k["frame"] == 30][0]
        assert kf["verified"] is False

    def test_add_keyframe_noop_if_exists(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(10)  # Already exists

        panel._on_add_keyframe()

        track = session.person_tracks[0]
        assert len(track.keyframes) == 3  # Still 3, not 4

    def test_remove_keyframe(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        panel._on_remove_keyframe()

        track = session.person_tracks[0]
        frames = [kf["frame"] for kf in track.keyframes]
        assert 50 not in frames
        assert panel._keyframe_table.rowCount() == 2

    def test_remove_nonexistent_keyframe_noop(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(99)  # No keyframe here

        panel._on_remove_keyframe()

        track = session.person_tracks[0]
        assert len(track.keyframes) == 3  # Unchanged

    def test_verify_existing_keyframe_toggles(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)  # Existing unverified keyframe

        panel._on_verify()

        track = session.person_tracks[0]
        kf = [k for k in track.keyframes if k["frame"] == 50][0]
        assert kf["verified"] is True

        # Toggle back
        panel._on_verify()
        assert kf["verified"] is False

    def test_verify_creates_keyframe_if_missing(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(25)  # No keyframe here

        panel._on_verify()

        track = session.person_tracks[0]
        kf = [k for k in track.keyframes if k["frame"] == 25]
        assert len(kf) == 1
        assert kf[0]["verified"] is True

    def test_add_emits_keyframe_changed(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)

        received = []
        panel.keyframe_changed.connect(lambda pid, f: received.append((pid, f)))
        panel._on_add_keyframe()

        assert received == [(0, 30)]

    def test_remove_emits_keyframe_changed(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        received = []
        panel.keyframe_changed.connect(lambda pid, f: received.append((pid, f)))
        panel._on_remove_keyframe()

        assert received == [(0, 50)]

    def test_verify_emits_keyframe_changed(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        received = []
        panel.keyframe_changed.connect(lambda pid, f: received.append((pid, f)))
        panel._on_verify()

        assert received == [(0, 50)]

    def test_crud_noop_without_person(self, qapp):
        """CRUD operations should be no-ops when no person is selected."""
        session = Session()
        panel = IdentityInspector(session)
        # Should not crash
        panel._on_add_keyframe()
        panel._on_remove_keyframe()
        panel._on_verify()

    def test_delete_via_table_button(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_keyframe_delete(10)

        track = session.person_tracks[0]
        frames = [kf["frame"] for kf in track.keyframes]
        assert 10 not in frames
        assert panel._keyframe_table.rowCount() == 2


# ---------------------------------------------------------------------------
# Keyframe navigation
# ---------------------------------------------------------------------------


class TestKeyframeNavigation:
    """Verify prev/next keyframe navigation emits correct frame_requested."""

    def test_next_keyframe(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(0)  # Before all keyframes

        received = []
        panel.frame_requested.connect(received.append)
        panel._on_next_keyframe()

        assert received == [10]

    def test_prev_keyframe(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(60)  # Between 50 and 80

        received = []
        panel.frame_requested.connect(received.append)
        panel._on_prev_keyframe()

        assert received == [50]

    def test_next_at_end_noop(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(90)  # Past all keyframes

        received = []
        panel.frame_requested.connect(received.append)
        panel._on_next_keyframe()

        assert received == []

    def test_prev_at_start_noop(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(5)  # Before first keyframe

        received = []
        panel.frame_requested.connect(received.append)
        panel._on_prev_keyframe()

        assert received == []

    def test_go_button_emits_frame_requested(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        received = []
        panel.frame_requested.connect(received.append)
        panel._on_keyframe_go(50)

        assert received == [50]

    def test_nav_noop_without_person(self, qapp):
        session = Session()
        panel = IdentityInspector(session)

        received = []
        panel.frame_requested.connect(received.append)
        panel._on_next_keyframe()
        panel._on_prev_keyframe()

        assert received == []


# ---------------------------------------------------------------------------
# Timeline interaction
# ---------------------------------------------------------------------------


class TestTimelineInteraction:
    """Verify timeline click triggers frame_requested and updates breakdown."""

    def test_timeline_click_emits_frame_requested(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        received = []
        panel.frame_requested.connect(received.append)
        panel._on_timeline_clicked(42)

        assert received == [42]

    def test_timeline_click_updates_breakdown(self, qapp):
        session = _session_with_breakdown()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel._on_timeline_clicked(10)

        assert "0.92" in panel._conf_labels["detection"].text()


# ---------------------------------------------------------------------------
# Show all tracks
# ---------------------------------------------------------------------------


class TestShowAllTracks:
    def test_toggle_emits_bbox_overlay_changed(self, qapp):
        session = Session()
        panel = IdentityInspector(session)

        received = []
        panel.bbox_overlay_changed.connect(received.append)
        panel._show_all_tracks.setChecked(True)

        assert len(received) == 1
        assert received[0] == {"show_all": True}


# ---------------------------------------------------------------------------
# Table double-click
# ---------------------------------------------------------------------------


class TestTableDoubleClick:
    def test_double_click_emits_frame_requested(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        received = []
        panel.frame_requested.connect(received.append)
        # Simulate double-click on row 1 (frame 50)
        panel._on_table_double_clicked(1, 0)

        assert received == [50]


# ---------------------------------------------------------------------------
# set_frame updates
# ---------------------------------------------------------------------------


class TestSetFrame:
    def test_set_frame_updates_timeline(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(42)
        assert panel._timeline._current_frame == 42

    def test_set_frame_updates_breakdown(self, qapp):
        session = _session_with_breakdown()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(0)
        assert "0.92" in panel._conf_labels["detection"].text()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


class TestDataHelpers:
    """Verify internal data access methods."""

    def test_get_current_track_returns_none_without_person(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        assert panel._get_current_track() is None

    def test_get_current_track_returns_track(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        track = panel._get_current_track()
        assert track is not None
        assert track.person_id == 0

    def test_get_confidences_returns_array(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        track = panel._get_current_track()
        confs = panel._get_confidences(track)
        assert isinstance(confs, np.ndarray)
        assert len(confs) == 100

    def test_get_confidences_fallback(self, qapp):
        """Track without confidences returns fallback array."""
        session = Session()
        session.num_frames = 10
        session.person_tracks = {0: PersonTrack(person_id=0)}
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        track = panel._get_current_track()
        confs = panel._get_confidences(track)
        assert len(confs) == 10
        assert all(c == 0.5 for c in confs)

    def test_get_bbox_prefers_corrections(self, qapp):
        session = Session()
        session.num_frames = 5
        session.person_tracks = {
            0: PersonTrack(
                person_id=0,
                bboxes=np.array([[10, 20, 110, 220]] * 5, dtype=float),
                bbox_corrections=np.array(
                    [[0, 0, 0, 0], [15, 25, 115, 225], [0, 0, 0, 0],
                     [0, 0, 0, 0], [0, 0, 0, 0]],
                    dtype=float,
                ),
            ),
        }
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        track = panel._get_current_track()

        # Frame 0: no correction (all zeros), should return original
        bbox0 = panel._get_bbox_at_frame(track, 0)
        assert bbox0 is not None
        assert bbox0[0] == 10

        # Frame 1: has correction
        bbox1 = panel._get_bbox_at_frame(track, 1)
        assert bbox1 is not None
        assert bbox1[0] == 15


# ---------------------------------------------------------------------------
# PersonTrack keyframes serialization
# ---------------------------------------------------------------------------


class TestPersonTrackKeyframes:
    """Verify that keyframes survive serialization round-trip."""

    def test_keyframes_in_to_dict(self, qapp):
        track = PersonTrack(
            person_id=0,
            keyframes=[{"frame": 10, "verified": True}],
        )
        d = track.to_dict()
        assert "keyframes" in d
        assert d["keyframes"] == [{"frame": 10, "verified": True}]

    def test_keyframes_from_dict(self, qapp):
        d = {
            "person_id": 0,
            "keyframes": [
                {"frame": 10, "verified": True},
                {"frame": 50, "verified": False},
            ],
        }
        track = PersonTrack.from_dict(d)
        assert len(track.keyframes) == 2
        assert track.keyframes[0]["frame"] == 10

    def test_keyframes_default_empty(self, qapp):
        track = PersonTrack(person_id=0)
        assert track.keyframes == []

    def test_session_round_trip_preserves_keyframes(self, qapp, tmp_path):
        session = _session_with_tracks()
        path = tmp_path / "test_session.json"
        session.save(path)

        loaded = Session.load(path)
        track = loaded.person_tracks[0]
        assert len(track.keyframes) == 3
        assert track.keyframes[0] == {"frame": 10, "verified": True}


# ---------------------------------------------------------------------------
# PERSON_COLORS constant
# ---------------------------------------------------------------------------


class TestConstants:
    def test_has_8_colors(self, qapp):
        assert len(PERSON_COLORS) == 8

    def test_has_6_metrics(self, qapp):
        assert len(CONFIDENCE_METRICS) == 6

    def test_labels_match_metrics(self, qapp):
        for metric in CONFIDENCE_METRICS:
            assert metric in CONFIDENCE_LABELS
