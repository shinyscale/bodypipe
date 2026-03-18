"""Tests for track operations: swap IDs, split track, merge track, crossing spans.

Why: Track operations are critical for correcting multi-person identity errors.
Swapping IDs fixes tracker ID switches, splitting isolates track segments,
merging recombines fragments, and crossing spans mark occlusion windows.
These tests verify all operations correctly mutate Session state, handle edge
cases, and emit appropriate signals — all headless, no GPU/video needed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session, PersonTrack
from views.identity_inspector import IdentityInspector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_with_tracks(num_frames: int = 100) -> Session:
    """Create a session with two active person tracks for testing."""
    session = Session()
    session.num_frames = num_frames
    session.img_width = 640
    session.img_height = 480
    session.person_tracks = {
        0: PersonTrack(
            person_id=0,
            confidences=[0.9 - i * 0.005 for i in range(num_frames)],
            keyframes=[{"frame": 10, "verified": True}],
            bboxes=np.array([[50, 60, 200, 300]] * num_frames, dtype=float),
        ),
        1: PersonTrack(
            person_id=1,
            confidences=[0.7 + i * 0.002 for i in range(num_frames)],
            keyframes=[{"frame": 20, "verified": False}],
            bboxes=np.array([[300, 100, 500, 400]] * num_frames, dtype=float),
        ),
    }
    return session


def _session_with_inactive(num_frames: int = 100) -> Session:
    """Create a session with two active and one inactive track."""
    session = _session_with_tracks(num_frames)
    inactive_bboxes = np.zeros((num_frames, 4), dtype=float)
    inactive_bboxes[50:80] = [400, 200, 550, 380]
    session.person_tracks[2] = PersonTrack(
        person_id=2,
        confidences=[0.0] * 50 + [0.6] * 30 + [0.0] * 20,
        bboxes=inactive_bboxes,
    )
    session.inactive_tracks.add(2)
    return session


# ---------------------------------------------------------------------------
# Track Operations UI construction
# ---------------------------------------------------------------------------


class TestTrackOpsConstruction:
    """Verify track operations UI elements exist."""

    def test_has_swap_combo(self, qapp):
        panel = IdentityInspector(Session())
        assert panel._swap_combo is not None

    def test_has_swap_button(self, qapp):
        panel = IdentityInspector(Session())
        assert panel._swap_btn is not None

    def test_has_split_button(self, qapp):
        panel = IdentityInspector(Session())
        assert panel._split_btn is not None

    def test_has_merge_combo(self, qapp):
        panel = IdentityInspector(Session())
        assert panel._merge_combo is not None

    def test_has_merge_button(self, qapp):
        panel = IdentityInspector(Session())
        assert panel._merge_btn is not None

    def test_has_crossing_buttons(self, qapp):
        panel = IdentityInspector(Session())
        assert panel._crossing_start_btn is not None
        assert panel._crossing_end_btn is not None

    def test_has_crossing_table(self, qapp):
        panel = IdentityInspector(Session())
        assert panel._crossing_table is not None
        assert panel._crossing_table.columnCount() == 4

    def test_crossing_table_headers(self, qapp):
        panel = IdentityInspector(Session())
        headers = []
        for col in range(4):
            item = panel._crossing_table.horizontalHeaderItem(col)
            headers.append(item.text() if item else "")
        assert headers == ["Person", "Start", "End", "Actions"]

    def test_has_crossing_status(self, qapp):
        panel = IdentityInspector(Session())
        assert panel._crossing_status is not None

    def test_has_track_modified_signal(self, qapp):
        panel = IdentityInspector(Session())
        assert hasattr(panel, "track_modified")

    def test_crossing_end_initially_disabled(self, qapp):
        panel = IdentityInspector(Session())
        assert not panel._crossing_end_btn.isEnabled()
        assert panel._crossing_start_btn.isEnabled()


# ---------------------------------------------------------------------------
# Swap IDs
# ---------------------------------------------------------------------------


class TestSwapIds:
    """Verify swap operation swaps bboxes and confidences from current frame onward."""

    def test_swap_bboxes_from_frame_onward(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        panel._on_swap_ids()

        track_0 = session.person_tracks[0]
        track_1 = session.person_tracks[1]
        # After swap at frame 50, person 0 should have person 1's original bboxes
        np.testing.assert_array_equal(track_0.bboxes[50], [300, 100, 500, 400])
        np.testing.assert_array_equal(track_1.bboxes[50], [50, 60, 200, 300])

    def test_swap_preserves_before_frame(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        original_0_before = session.person_tracks[0].bboxes[30].copy()
        original_1_before = session.person_tracks[1].bboxes[30].copy()

        panel._on_swap_ids()

        np.testing.assert_array_equal(session.person_tracks[0].bboxes[30], original_0_before)
        np.testing.assert_array_equal(session.person_tracks[1].bboxes[30], original_1_before)

    def test_swap_confidences_from_frame_onward(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        original_conf_0_at_50 = session.person_tracks[0].confidences[50]
        original_conf_1_at_50 = session.person_tracks[1].confidences[50]

        panel._on_swap_ids()

        assert session.person_tracks[0].confidences[50] == original_conf_1_at_50
        assert session.person_tracks[1].confidences[50] == original_conf_0_at_50

    def test_swap_adds_verified_keyframes(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        panel._on_swap_ids()

        kf_0 = [kf for kf in session.person_tracks[0].keyframes if kf["frame"] == 50]
        kf_1 = [kf for kf in session.person_tracks[1].keyframes if kf["frame"] == 50]
        assert len(kf_0) == 1
        assert kf_0[0]["verified"] is True
        assert len(kf_1) == 1
        assert kf_1[0]["verified"] is True

    def test_swap_does_not_duplicate_keyframe(self, qapp):
        """If a keyframe already exists at the swap frame, set verified=True instead."""
        session = _session_with_tracks()
        # Person 1 already has a keyframe at frame 20
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(20)

        panel._on_swap_ids()

        kf_1_at_20 = [kf for kf in session.person_tracks[1].keyframes if kf["frame"] == 20]
        assert len(kf_1_at_20) == 1  # not duplicated
        assert kf_1_at_20[0]["verified"] is True  # now verified

    def test_swap_marks_both_dirty(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        panel._on_swap_ids()

        assert 0 in session.dirty_persons
        assert 1 in session.dirty_persons

    def test_swap_emits_signals(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        dirty = []
        modified = []
        panel.person_dirty.connect(dirty.append)
        panel.track_modified.connect(lambda: modified.append(True))

        panel._on_swap_ids()

        assert 0 in dirty
        assert 1 in dirty
        assert len(modified) == 1

    def test_swap_noop_without_person(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        panel._on_swap_ids()  # should not crash

    def test_swap_noop_with_empty_combo(self, qapp):
        """Only one active person — swap combo is empty, no-op."""
        session = Session()
        session.num_frames = 10
        session.person_tracks = {
            0: PersonTrack(person_id=0, bboxes=np.zeros((10, 4))),
        }
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_swap_ids()  # should not crash

    def test_swap_at_frame_0(self, qapp):
        """Swap at frame 0 swaps all frames."""
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(0)

        panel._on_swap_ids()

        # All frames should be swapped
        np.testing.assert_array_equal(session.person_tracks[0].bboxes[0], [300, 100, 500, 400])
        np.testing.assert_array_equal(session.person_tracks[1].bboxes[0], [50, 60, 200, 300])


# ---------------------------------------------------------------------------
# Split track
# ---------------------------------------------------------------------------


class TestSplitTrack:
    """Verify split creates new inactive track from frames after split point."""

    def test_split_creates_new_track(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        panel._on_split_track()

        assert len(session.person_tracks) == 3
        new_id = max(session.person_tracks.keys())
        assert new_id in session.inactive_tracks

    def test_split_new_track_has_data_from_split_onward(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        panel._on_split_track()

        new_id = max(session.person_tracks.keys())
        new_track = session.person_tracks[new_id]
        # Frame 50 onward should have bboxes
        np.testing.assert_array_equal(new_track.bboxes[50], [50, 60, 200, 300])
        # Before split: zeros
        np.testing.assert_array_equal(new_track.bboxes[49], [0, 0, 0, 0])

    def test_split_original_zeroed_after_split(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        panel._on_split_track()

        track_0 = session.person_tracks[0]
        np.testing.assert_array_equal(track_0.bboxes[50], [0, 0, 0, 0])
        np.testing.assert_array_equal(track_0.bboxes[49], [50, 60, 200, 300])

    def test_split_confidences(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        original_conf_50 = session.person_tracks[0].confidences[50]

        panel._on_split_track()

        new_id = max(session.person_tracks.keys())
        assert session.person_tracks[new_id].confidences[50] == original_conf_50
        assert session.person_tracks[0].confidences[50] == 0.0

    def test_split_keyframes(self, qapp):
        """Keyframes before split stay with original; at/after split go to new track."""
        session = _session_with_tracks()
        # Person 0 has keyframe at frame 10
        session.person_tracks[0].keyframes.append({"frame": 60, "verified": False})
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        panel._on_split_track()

        new_id = max(session.person_tracks.keys())
        original_kf_frames = [kf["frame"] for kf in session.person_tracks[0].keyframes]
        new_kf_frames = [kf["frame"] for kf in session.person_tracks[new_id].keyframes]
        assert 10 in original_kf_frames
        assert 60 not in original_kf_frames
        assert 60 in new_kf_frames
        assert 10 not in new_kf_frames

    def test_split_bbox_corrections(self, qapp):
        session = _session_with_tracks()
        corrections = np.zeros((100, 4), dtype=float)
        corrections[30] = [55, 65, 205, 305]
        corrections[70] = [65, 75, 215, 315]
        session.person_tracks[0].bbox_corrections = corrections

        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        panel._on_split_track()

        new_id = max(session.person_tracks.keys())
        # Correction at frame 30 stays with original
        np.testing.assert_array_equal(session.person_tracks[0].bbox_corrections[30], [55, 65, 205, 305])
        np.testing.assert_array_equal(session.person_tracks[0].bbox_corrections[70], [0, 0, 0, 0])
        # Correction at frame 70 goes to new track
        np.testing.assert_array_equal(session.person_tracks[new_id].bbox_corrections[70], [65, 75, 215, 315])
        np.testing.assert_array_equal(session.person_tracks[new_id].bbox_corrections[30], [0, 0, 0, 0])

    def test_split_marks_dirty(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        panel._on_split_track()

        assert 0 in session.dirty_persons

    def test_split_at_frame_0_rejected(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(0)

        panel._on_split_track()

        assert len(session.person_tracks) == 2  # unchanged

    def test_split_at_last_frame_rejected(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(99)  # num_frames - 1

        panel._on_split_track()

        assert len(session.person_tracks) == 2  # unchanged

    def test_split_emits_track_modified(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        received = []
        panel.track_modified.connect(lambda: received.append(True))

        panel._on_split_track()

        assert len(received) == 1

    def test_split_noop_without_person(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        panel._on_split_track()  # should not crash

    def test_split_generates_unique_id(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        panel._on_split_track()

        new_id = max(session.person_tracks.keys())
        assert new_id == 2  # next after 0, 1


# ---------------------------------------------------------------------------
# Merge track
# ---------------------------------------------------------------------------


class TestMergeTrack:
    """Verify merge combines inactive track data into active person."""

    def test_merge_copies_bboxes(self, qapp):
        session = _session_with_inactive()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_merge_track()

        track_0 = session.person_tracks[0]
        # Frame 60 should now have merged bboxes from track 2
        np.testing.assert_array_equal(track_0.bboxes[60], [400, 200, 550, 380])

    def test_merge_preserves_target_where_source_zero(self, qapp):
        session = _session_with_inactive()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        original_at_30 = session.person_tracks[0].bboxes[30].copy()

        panel._on_merge_track()

        # Frame 30 — source track 2 has zeros, so target should be unchanged
        np.testing.assert_array_equal(session.person_tracks[0].bboxes[30], original_at_30)

    def test_merge_removes_inactive_track(self, qapp):
        session = _session_with_inactive()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_merge_track()

        assert 2 not in session.person_tracks
        assert 2 not in session.inactive_tracks

    def test_merge_confidences(self, qapp):
        session = _session_with_inactive()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_merge_track()

        # Frame 60 should have merged confidence from track 2
        assert session.person_tracks[0].confidences[60] == 0.6

    def test_merge_marks_target_dirty(self, qapp):
        session = _session_with_inactive()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_merge_track()

        assert 0 in session.dirty_persons

    def test_merge_emits_track_modified(self, qapp):
        session = _session_with_inactive()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        received = []
        panel.track_modified.connect(lambda: received.append(True))

        panel._on_merge_track()

        assert len(received) == 1

    def test_merge_emits_person_dirty(self, qapp):
        session = _session_with_inactive()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        received = []
        panel.person_dirty.connect(received.append)

        panel._on_merge_track()

        assert 0 in received

    def test_merge_noop_without_inactive(self, qapp):
        session = _session_with_tracks()  # no inactive tracks
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_merge_track()  # should not crash
        assert len(session.person_tracks) == 2

    def test_merge_noop_without_person(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        panel._on_merge_track()  # should not crash

    def test_merge_keyframes(self, qapp):
        """Keyframes from source should be added to target without duplicates."""
        session = _session_with_inactive()
        session.person_tracks[2].keyframes = [
            {"frame": 55, "verified": True},
            {"frame": 10, "verified": False},  # conflicts with person 0's frame 10
        ]

        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_merge_track()

        target_kf_frames = [kf["frame"] for kf in session.person_tracks[0].keyframes]
        assert 55 in target_kf_frames
        # Frame 10 should not be duplicated — person 0 already has it
        assert target_kf_frames.count(10) == 1

    def test_merge_bbox_corrections(self, qapp):
        session = _session_with_inactive()
        corrections = np.zeros((100, 4), dtype=float)
        corrections[60] = [410, 210, 560, 390]
        session.person_tracks[2].bbox_corrections = corrections

        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_merge_track()

        assert session.person_tracks[0].bbox_corrections is not None
        np.testing.assert_array_equal(
            session.person_tracks[0].bbox_corrections[60], [410, 210, 560, 390]
        )


# ---------------------------------------------------------------------------
# Crossing spans
# ---------------------------------------------------------------------------


class TestCrossingSpans:
    """Verify crossing span marking and table updates."""

    def test_mark_start_sets_pending(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(20)

        panel._on_crossing_start()

        assert panel._crossing_start_frame == 20

    def test_mark_start_disables_start_enables_end(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(20)

        panel._on_crossing_start()

        assert not panel._crossing_start_btn.isEnabled()
        assert panel._crossing_end_btn.isEnabled()

    def test_mark_start_updates_status(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(20)

        panel._on_crossing_start()

        assert "20" in panel._crossing_status.text()

    def test_mark_end_creates_span(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(20)

        panel._on_crossing_start()
        panel.set_frame(40)
        panel._on_crossing_end()

        assert 0 in session.crossing_spans
        assert (20, 40) in session.crossing_spans[0]

    def test_mark_end_resets_state(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(20)

        panel._on_crossing_start()
        panel.set_frame(40)
        panel._on_crossing_end()

        assert panel._crossing_start_frame is None
        assert panel._crossing_start_btn.isEnabled()
        assert not panel._crossing_end_btn.isEnabled()

    def test_end_before_start_rejected(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(40)

        panel._on_crossing_start()
        panel.set_frame(20)  # before start
        panel._on_crossing_end()

        assert session.crossing_spans == {}

    def test_end_equal_start_rejected(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(30)

        panel._on_crossing_start()
        # Don't change frame
        panel._on_crossing_end()

        assert session.crossing_spans == {}

    def test_multiple_spans(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        # First span
        panel.set_frame(10)
        panel._on_crossing_start()
        panel.set_frame(20)
        panel._on_crossing_end()

        # Second span
        panel.set_frame(40)
        panel._on_crossing_start()
        panel.set_frame(60)
        panel._on_crossing_end()

        assert len(session.crossing_spans[0]) == 2
        assert (10, 20) in session.crossing_spans[0]
        assert (40, 60) in session.crossing_spans[0]

    def test_crossing_table_populated(self, qapp):
        session = _session_with_tracks()
        session.crossing_spans = {0: [(10, 30), (50, 70)]}
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        assert panel._crossing_table.rowCount() == 2
        assert panel._crossing_table.item(0, 1).text() == "10"
        assert panel._crossing_table.item(0, 2).text() == "30"
        assert panel._crossing_table.item(1, 1).text() == "50"
        assert panel._crossing_table.item(1, 2).text() == "70"

    def test_crossing_table_shows_person_id(self, qapp):
        session = _session_with_tracks()
        session.crossing_spans = {0: [(10, 30)], 1: [(20, 40)]}
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        assert panel._crossing_table.rowCount() == 2
        assert "Person 0" in panel._crossing_table.item(0, 0).text()
        assert "Person 1" in panel._crossing_table.item(1, 0).text()

    def test_crossing_delete(self, qapp):
        session = _session_with_tracks()
        session.crossing_spans = {0: [(10, 30), (50, 70)]}
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_crossing_delete(0, 10, 30)

        assert session.crossing_spans[0] == [(50, 70)]
        assert panel._crossing_table.rowCount() == 1

    def test_crossing_delete_removes_empty_person(self, qapp):
        """Deleting last span for a person removes the person key."""
        session = _session_with_tracks()
        session.crossing_spans = {0: [(10, 30)]}
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_crossing_delete(0, 10, 30)

        assert 0 not in session.crossing_spans

    def test_crossing_noop_without_person(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        panel._on_crossing_start()  # should not crash
        panel._on_crossing_end()  # should not crash

    def test_end_without_start_noop(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        panel._on_crossing_end()  # should not crash
        assert session.crossing_spans == {}

    def test_crossing_has_delete_buttons(self, qapp):
        session = _session_with_tracks()
        session.crossing_spans = {0: [(10, 30)]}
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        widget = panel._crossing_table.cellWidget(0, 3)
        assert widget is not None


# ---------------------------------------------------------------------------
# Swap combo
# ---------------------------------------------------------------------------


class TestSwapCombo:
    """Verify swap target combo excludes current person and inactive tracks."""

    def test_excludes_current_person(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        data = [panel._swap_combo.itemData(i) for i in range(panel._swap_combo.count())]
        assert 0 not in data
        assert 1 in data

    def test_excludes_inactive_tracks(self, qapp):
        session = _session_with_inactive()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        data = [panel._swap_combo.itemData(i) for i in range(panel._swap_combo.count())]
        assert 2 not in data  # track 2 is inactive

    def test_empty_with_single_person(self, qapp):
        session = Session()
        session.num_frames = 10
        session.person_tracks = {0: PersonTrack(person_id=0)}
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        assert panel._swap_combo.count() == 0

    def test_updates_on_person_change(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        data_for_0 = [panel._swap_combo.itemData(i) for i in range(panel._swap_combo.count())]
        assert 1 in data_for_0

        panel.set_person(1)
        data_for_1 = [panel._swap_combo.itemData(i) for i in range(panel._swap_combo.count())]
        assert 0 in data_for_1
        assert 1 not in data_for_1


# ---------------------------------------------------------------------------
# Merge combo
# ---------------------------------------------------------------------------


class TestMergeCombo:
    """Verify merge source combo shows inactive tracks."""

    def test_shows_inactive_tracks(self, qapp):
        session = _session_with_inactive()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        assert panel._merge_combo.count() == 1
        assert panel._merge_combo.itemData(0) == 2

    def test_empty_without_inactive(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        assert panel._merge_combo.count() == 0

    def test_label_includes_frame_info(self, qapp):
        session = _session_with_inactive()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        label = panel._merge_combo.itemText(0)
        assert "Track 2" in label
        assert "50" in label  # first non-zero frame

    def test_updates_after_merge(self, qapp):
        """After merge, the combo should be empty (no more inactive tracks)."""
        session = _session_with_inactive()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        assert panel._merge_combo.count() == 1

        panel._on_merge_track()

        assert panel._merge_combo.count() == 0

    def test_updates_after_split(self, qapp):
        """After split, the combo should show the new inactive track."""
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel.set_frame(50)

        assert panel._merge_combo.count() == 0

        panel._on_split_track()

        assert panel._merge_combo.count() == 1


# ---------------------------------------------------------------------------
# Session crossing_spans serialization
# ---------------------------------------------------------------------------


class TestCrossingSpansSerialization:
    """Verify crossing_spans dict survives serialization round-trip."""

    def test_round_trip(self, qapp, tmp_path):
        session = Session()
        session.crossing_spans = {0: [(10, 30)], 1: [(40, 60)]}
        path = tmp_path / "session.json"
        session.save(path)

        loaded = Session.load(path)
        assert 0 in loaded.crossing_spans
        assert (10, 30) in loaded.crossing_spans[0]
        assert 1 in loaded.crossing_spans
        assert (40, 60) in loaded.crossing_spans[1]

    def test_empty_round_trip(self, qapp, tmp_path):
        session = Session()
        path = tmp_path / "session.json"
        session.save(path)

        loaded = Session.load(path)
        assert loaded.crossing_spans == {}


# ---------------------------------------------------------------------------
# MultiPersonTab track_modified wiring
# ---------------------------------------------------------------------------


class TestMultiPersonTabTrackWiring:
    """Verify MultiPersonTab connects to track_modified signal."""

    def test_track_modified_does_not_crash(self, qapp, session):
        from views.multi_person_tab import MultiPersonTab

        gvhmr_root = Path(__file__).resolve().parent.parent.parent / "GVHMR"
        tab = MultiPersonTab(session, gvhmr_root)

        # Emit track_modified — should call _on_tracks_modified without crash
        tab._identity_panel.track_modified.emit()

    def test_has_on_tracks_modified_handler(self, qapp, session):
        from views.multi_person_tab import MultiPersonTab

        gvhmr_root = Path(__file__).resolve().parent.parent.parent / "GVHMR"
        tab = MultiPersonTab(session, gvhmr_root)

        assert hasattr(tab, "_on_tracks_modified")
