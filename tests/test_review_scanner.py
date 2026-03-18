"""Tests for review scanner + reprocess functionality (Phase 2.5).

Why: The review scanner auto-detects tracking issues (low confidence, potential
swaps, shape drift, detection gaps) so the user doesn't have to manually scrub
through every frame. The reprocess button triggers re-running GVHMR for persons
with corrected bounding boxes. These tests verify both the scanning algorithm
and the UI integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from models.session import Session, PersonTrack
from views.identity_inspector import (
    IdentityInspector,
    ReviewIssue,
    compute_review_issues,
)


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
            keyframes=[{"frame": 10, "verified": True}],
            bboxes=np.array([[50, 60, 200, 300]] * num_frames, dtype=float),
        ),
        1: PersonTrack(
            person_id=1,
            confidences=[0.7 + i * 0.002 for i in range(num_frames)],
            keyframes=[],
            bboxes=np.array([[300, 100, 500, 400]] * num_frames, dtype=float),
        ),
    }
    return session


def _session_with_low_confidence(num_frames: int = 100) -> Session:
    """Create a session where person 0 has a low-confidence span."""
    session = _session_with_tracks(num_frames)
    confs = list(session.person_tracks[0].confidences)
    # Insert a span of 10 frames below 0.4
    for f in range(30, 40):
        confs[f] = 0.2
    session.person_tracks[0].confidences = confs
    return session


def _session_with_high_overlap(num_frames: int = 100) -> Session:
    """Create a session where person 0 has high overlap in some frames."""
    session = _session_with_tracks(num_frames)
    overlaps = [0.1] * num_frames
    # Insert a burst of high overlap
    for f in range(20, 25):
        overlaps[f] = 0.7
    session.person_tracks[0].confidence_breakdown = {
        "detection": [0.9] * num_frames,
        "visibility": [0.9] * num_frames,
        "overlap": overlaps,
        "shape": [0.9] * num_frames,
        "motion": [0.9] * num_frames,
        "overall": [0.9] * num_frames,
    }
    return session


def _session_with_shape_drift(num_frames: int = 100) -> Session:
    """Create a session where person 0 has low shape confidence."""
    session = _session_with_tracks(num_frames)
    shapes = [0.9] * num_frames
    # Low shape confidence for a few frames
    for f in range(50, 55):
        shapes[f] = 0.2
    session.person_tracks[0].confidence_breakdown = {
        "detection": [0.9] * num_frames,
        "visibility": [0.9] * num_frames,
        "overlap": [0.1] * num_frames,
        "shape": shapes,
        "motion": [0.9] * num_frames,
        "overall": [0.9] * num_frames,
    }
    return session


def _session_with_track_gap(num_frames: int = 200) -> Session:
    """Create a session where person 0 has a detection gap."""
    session = _session_with_tracks(num_frames)
    bboxes = session.person_tracks[0].bboxes.copy()
    # Zero out bboxes for 40 frames (above gap_threshold of 30)
    bboxes[80:120] = 0
    session.person_tracks[0].bboxes = bboxes
    return session


def _session_with_multiple_issues() -> Session:
    """Create a session with multiple different issue types."""
    num_frames = 200
    session = Session()
    session.num_frames = num_frames
    session.img_width = 640
    session.img_height = 480

    # Person 0: low confidence span (frames 30-44) + track gap (frames 100-140)
    confs_0 = [0.8] * num_frames
    for f in range(30, 45):
        confs_0[f] = 0.2
    bboxes_0 = np.array([[50, 60, 200, 300]] * num_frames, dtype=float)
    bboxes_0[100:140] = 0

    session.person_tracks[0] = PersonTrack(
        person_id=0,
        confidences=confs_0,
        bboxes=bboxes_0,
    )

    # Person 1: high overlap at frame 60
    overlaps = [0.1] * num_frames
    overlaps[60] = 0.8
    overlaps[61] = 0.6
    session.person_tracks[1] = PersonTrack(
        person_id=1,
        confidences=[0.9] * num_frames,
        bboxes=np.array([[300, 100, 500, 400]] * num_frames, dtype=float),
        confidence_breakdown={
            "detection": [0.9] * num_frames,
            "visibility": [0.9] * num_frames,
            "overlap": overlaps,
            "shape": [0.9] * num_frames,
            "motion": [0.9] * num_frames,
            "overall": [0.9] * num_frames,
        },
    )

    return session


# ===========================================================================
# Tests: compute_review_issues()
# ===========================================================================


class TestComputeReviewIssues:
    """Unit tests for the compute_review_issues() function."""

    def test_no_tracks_returns_empty(self):
        session = Session()
        issues = compute_review_issues(session)
        assert issues == []

    def test_healthy_tracks_no_issues(self):
        session = _session_with_tracks()
        issues = compute_review_issues(session)
        assert issues == []

    def test_low_confidence_span_detected(self):
        session = _session_with_low_confidence()
        issues = compute_review_issues(session)
        low_conf = [i for i in issues if i.issue_type == "low_confidence"]
        assert len(low_conf) == 1
        assert low_conf[0].person_id == 0
        # Mid-point of span 30-39
        assert low_conf[0].frame == (30 + 40) // 2
        assert low_conf[0].severity == 0.7

    def test_low_confidence_short_span_ignored(self):
        """Spans shorter than 5 frames should not be flagged."""
        session = _session_with_tracks()
        confs = list(session.person_tracks[0].confidences)
        for f in range(30, 33):  # Only 3 frames
            confs[f] = 0.2
        session.person_tracks[0].confidences = confs
        issues = compute_review_issues(session)
        low_conf = [i for i in issues if i.issue_type == "low_confidence"]
        assert len(low_conf) == 0

    def test_low_confidence_trailing_span(self):
        """Low confidence at the end of the track should be detected."""
        session = _session_with_tracks(num_frames=50)
        confs = list(session.person_tracks[0].confidences)
        for f in range(40, 50):  # Last 10 frames
            confs[f] = 0.1
        session.person_tracks[0].confidences = confs
        issues = compute_review_issues(session)
        low_conf = [i for i in issues if i.issue_type == "low_confidence"]
        assert len(low_conf) == 1

    def test_potential_swap_detected(self):
        session = _session_with_high_overlap()
        issues = compute_review_issues(session)
        swaps = [i for i in issues if i.issue_type == "potential_swap"]
        assert len(swaps) == 1
        assert swaps[0].person_id == 0
        assert swaps[0].frame == 20  # First frame of burst
        assert swaps[0].severity == 0.9

    def test_potential_swap_only_first_frame_of_burst(self):
        """Only the first frame of a continuous high-overlap burst is flagged."""
        session = _session_with_high_overlap()
        issues = compute_review_issues(session)
        swaps = [i for i in issues if i.issue_type == "potential_swap"]
        assert len(swaps) == 1  # Not 5

    def test_shape_drift_detected(self):
        session = _session_with_shape_drift()
        issues = compute_review_issues(session)
        drifts = [i for i in issues if i.issue_type == "shape_drift"]
        assert len(drifts) == 1
        assert drifts[0].person_id == 0
        assert drifts[0].frame == 50  # First frame of low shape
        assert drifts[0].severity == 0.6

    def test_track_gap_detected(self):
        session = _session_with_track_gap()
        issues = compute_review_issues(session)
        gaps = [i for i in issues if i.issue_type == "track_gap"]
        assert len(gaps) == 1
        assert gaps[0].person_id == 0
        # Mid-point of gap 80-119
        assert gaps[0].frame == (80 + 120) // 2
        assert gaps[0].severity == 0.5

    def test_track_gap_short_ignored(self):
        """Gaps shorter than threshold should not be flagged."""
        session = _session_with_tracks(200)
        bboxes = session.person_tracks[0].bboxes.copy()
        bboxes[80:90] = 0  # Only 10 frames, below default 30
        session.person_tracks[0].bboxes = bboxes
        issues = compute_review_issues(session)
        gaps = [i for i in issues if i.issue_type == "track_gap"]
        assert len(gaps) == 0

    def test_track_gap_trailing(self):
        """Trailing zero bboxes at end of track."""
        session = _session_with_tracks(200)
        bboxes = session.person_tracks[0].bboxes.copy()
        bboxes[160:] = 0  # Last 40 frames
        session.person_tracks[0].bboxes = bboxes
        issues = compute_review_issues(session)
        gaps = [i for i in issues if i.issue_type == "track_gap"]
        assert len(gaps) == 1

    def test_multiple_issues_sorted_by_frame(self):
        session = _session_with_multiple_issues()
        issues = compute_review_issues(session)
        assert len(issues) >= 2
        # Verify sorted by frame
        frames = [i.frame for i in issues]
        assert frames == sorted(frames)

    def test_inactive_tracks_skipped(self):
        session = _session_with_low_confidence()
        session.inactive_tracks.add(0)
        issues = compute_review_issues(session)
        person_0_issues = [i for i in issues if i.person_id == 0]
        assert len(person_0_issues) == 0

    def test_custom_thresholds(self):
        session = _session_with_low_confidence()
        # With a very low threshold, the span won't be flagged
        issues = compute_review_issues(session, low_conf_threshold=0.1)
        low_conf = [i for i in issues if i.issue_type == "low_confidence"]
        assert len(low_conf) == 0

    def test_no_confidence_breakdown_no_swap_or_drift(self):
        """Without confidence_breakdown, only low_conf and gap issues possible."""
        session = _session_with_low_confidence()
        assert session.person_tracks[0].confidence_breakdown is None
        issues = compute_review_issues(session)
        for issue in issues:
            assert issue.issue_type in ("low_confidence", "track_gap")

    def test_issue_description_contains_frame_range(self):
        session = _session_with_low_confidence()
        issues = compute_review_issues(session)
        low_conf = [i for i in issues if i.issue_type == "low_confidence"]
        assert len(low_conf) == 1
        assert "30" in low_conf[0].description
        assert "39" in low_conf[0].description

    def test_empty_confidences_skipped(self):
        session = Session()
        session.person_tracks[0] = PersonTrack(
            person_id=0, confidences=[]
        )
        issues = compute_review_issues(session)
        assert issues == []


# ===========================================================================
# Tests: ReviewIssue dataclass
# ===========================================================================


class TestReviewIssue:
    def test_fields(self):
        issue = ReviewIssue(
            frame=42, person_id=1, issue_type="low_confidence",
            description="Test issue", severity=0.7,
        )
        assert issue.frame == 42
        assert issue.person_id == 1
        assert issue.issue_type == "low_confidence"
        assert issue.description == "Test issue"
        assert issue.severity == 0.7

    def test_equality(self):
        a = ReviewIssue(10, 0, "low_confidence", "desc", 0.5)
        b = ReviewIssue(10, 0, "low_confidence", "desc", 0.5)
        assert a == b


# ===========================================================================
# Tests: IdentityInspector — Review Scanner UI
# ===========================================================================


class TestReviewScannerUI:
    """Test the scanner UI integration in IdentityInspector."""

    def test_scanner_widgets_exist(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        assert hasattr(panel, "_scan_btn")
        assert hasattr(panel, "_prev_issue_btn")
        assert hasattr(panel, "_next_issue_btn")
        assert hasattr(panel, "_issues_label")
        assert hasattr(panel, "_issues_table")

    def test_scan_no_issues(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        panel._on_scan_issues()

        assert len(panel._review_issues) == 0
        assert "No issues" in panel._issues_label.text()
        assert not panel._prev_issue_btn.isEnabled()
        assert not panel._next_issue_btn.isEnabled()

    def test_scan_finds_issues(self, qapp):
        session = _session_with_low_confidence()
        panel = IdentityInspector(session)
        panel.refresh()
        panel._on_scan_issues()

        assert len(panel._review_issues) > 0
        # After scan, navigates to first issue — label shows [1/N] format
        assert "[1/" in panel._issues_label.text()
        assert panel._prev_issue_btn.isEnabled()
        assert panel._next_issue_btn.isEnabled()

    def test_scan_populates_issues_table(self, qapp):
        session = _session_with_low_confidence()
        panel = IdentityInspector(session)
        panel.refresh()
        panel._on_scan_issues()

        assert panel._issues_table.rowCount() == len(panel._review_issues)

    def test_next_issue_cycles(self, qapp):
        session = _session_with_multiple_issues()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        # Capture frame_requested emissions
        requested_frames = []
        panel.frame_requested.connect(requested_frames.append)

        panel._on_scan_issues()
        n = len(panel._review_issues)
        assert n >= 2

        # Navigate through all issues — should wrap around
        for i in range(n):
            panel._on_next_issue()

        # After scan, idx=0. Each _on_next_issue increments by 1 mod n.
        # n calls: (0+1)%n, (1+1)%n, ..., (n-1+1)%n = 0
        assert panel._review_issue_idx == 0

    def test_prev_issue_cycles(self, qapp):
        session = _session_with_multiple_issues()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel._on_scan_issues()

        n = len(panel._review_issues)
        assert n >= 2

        # Going prev from index 0 should wrap to last
        panel._on_prev_issue()
        assert panel._review_issue_idx == n - 1

    def test_next_issue_emits_frame_requested(self, qapp):
        session = _session_with_low_confidence()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        frames = []
        panel.frame_requested.connect(frames.append)

        panel._on_scan_issues()
        initial_frame = panel._review_issues[0].frame
        assert initial_frame in frames  # scan navigates to first issue

        frames.clear()
        panel._on_next_issue()
        # Should navigate to issue's frame
        assert len(frames) > 0

    def test_issue_table_double_click_navigates(self, qapp):
        session = _session_with_low_confidence()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        frames = []
        panel.frame_requested.connect(frames.append)

        panel._on_scan_issues()
        frames.clear()

        # Double-click first row
        panel._on_issue_double_clicked(0, 0)
        assert panel._review_issue_idx == 0
        assert len(frames) > 0

    def test_scan_with_empty_session(self, qapp):
        session = Session()
        panel = IdentityInspector(session)
        panel._on_scan_issues()
        assert len(panel._review_issues) == 0
        assert "No issues" in panel._issues_label.text()

    def test_scan_switches_person_for_issue(self, qapp):
        """When navigating to an issue for a different person, auto-select."""
        session = _session_with_multiple_issues()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)

        person_changes = []
        panel.person_changed.connect(person_changes.append)

        panel._on_scan_issues()

        # Find an issue for person 1
        person_1_issues = [
            (i, iss) for i, iss in enumerate(panel._review_issues)
            if iss.person_id == 1
        ]
        if person_1_issues:
            idx, issue = person_1_issues[0]
            panel._review_issue_idx = idx - 1 if idx > 0 else len(panel._review_issues) - 1
            panel._on_next_issue()
            assert 1 in person_changes

    def test_issues_label_shows_index(self, qapp):
        session = _session_with_multiple_issues()
        panel = IdentityInspector(session)
        panel.refresh()
        panel.set_person(0)
        panel._on_scan_issues()

        n = len(panel._review_issues)
        if n > 0:
            # After scan, label should show [1/N]
            assert f"[1/{n}]" in panel._issues_label.text()

            panel._on_next_issue()
            assert f"[2/{n}]" in panel._issues_label.text()


# ===========================================================================
# Tests: IdentityInspector — Reprocess Button
# ===========================================================================


class TestReprocessButton:
    """Test the reprocess button UI and signal emission."""

    def test_reprocess_button_exists(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        assert hasattr(panel, "_reprocess_btn")

    def test_reprocess_button_disabled_no_dirty(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        assert not panel._reprocess_btn.isEnabled()
        assert "0 dirty" in panel._reprocess_btn.text()

    def test_reprocess_button_enabled_with_dirty(self, qapp):
        session = _session_with_tracks()
        session.dirty_persons.add(0)
        panel = IdentityInspector(session)
        panel.refresh()
        assert panel._reprocess_btn.isEnabled()
        assert "1 dirty" in panel._reprocess_btn.text()

    def test_reprocess_button_updates_on_person_dirty(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()
        assert not panel._reprocess_btn.isEnabled()

        # Mark a person dirty via the signal
        session.dirty_persons.add(0)
        panel.person_dirty.emit(0)
        assert panel._reprocess_btn.isEnabled()
        assert "1 dirty" in panel._reprocess_btn.text()

    def test_reprocess_emits_signal(self, qapp):
        session = _session_with_tracks()
        session.dirty_persons.add(0)
        session.dirty_persons.add(1)
        panel = IdentityInspector(session)
        panel.refresh()

        received = []
        panel.reprocess_requested.connect(received.append)

        panel._on_reprocess()
        assert len(received) == 1
        assert set(received[0]) == {0, 1}

    def test_reprocess_noop_no_dirty(self, qapp):
        session = _session_with_tracks()
        panel = IdentityInspector(session)
        panel.refresh()

        received = []
        panel.reprocess_requested.connect(received.append)

        panel._on_reprocess()
        assert len(received) == 0

    def test_update_reprocess_button_multiple_dirty(self, qapp):
        session = _session_with_tracks()
        session.dirty_persons = {0, 1}
        panel = IdentityInspector(session)
        panel.update_reprocess_button()
        assert "2 dirty" in panel._reprocess_btn.text()
        assert panel._reprocess_btn.isEnabled()

    def test_update_reprocess_button_cleared(self, qapp):
        session = _session_with_tracks()
        session.dirty_persons = {0}
        panel = IdentityInspector(session)
        panel.update_reprocess_button()
        assert panel._reprocess_btn.isEnabled()

        session.dirty_persons.clear()
        panel.update_reprocess_button()
        assert not panel._reprocess_btn.isEnabled()
        assert "0 dirty" in panel._reprocess_btn.text()


# ===========================================================================
# Tests: MultiPersonTab — Reprocess Wiring
# ===========================================================================


class TestMultiPersonTabReprocessWiring:
    """Test that MultiPersonTab correctly wires the reprocess signal."""

    def test_reprocess_signal_connected(self, qapp):
        from views.multi_person_tab import MultiPersonTab
        from pathlib import Path

        session = _session_with_tracks()
        tab = MultiPersonTab(session, gvhmr_root=Path("/tmp/gvhmr"))
        assert tab._reprocess_worker is None

        # The signal should be connected — we verify by checking
        # the method exists and is wired
        assert hasattr(tab, "_on_reprocess_requested")

    def test_reprocess_finished_clears_dirty(self, qapp):
        from views.multi_person_tab import MultiPersonTab
        from pathlib import Path

        session = _session_with_tracks()
        session.dirty_persons = {0, 1}
        tab = MultiPersonTab(session, gvhmr_root=Path("/tmp/gvhmr"))

        # Simulate worker completion
        tab._on_reprocess_finished({"reprocessed": [0, 1]})
        assert len(session.dirty_persons) == 0

    def test_reprocess_person_done_updates_dirty(self, qapp):
        from views.multi_person_tab import MultiPersonTab
        from pathlib import Path

        session = _session_with_tracks()
        session.dirty_persons = {0, 1}
        tab = MultiPersonTab(session, gvhmr_root=Path("/tmp/gvhmr"))

        tab._on_reprocess_person_done(0)
        assert 0 not in session.dirty_persons
        assert 1 in session.dirty_persons

    def test_reprocess_error_clears_worker(self, qapp):
        from views.multi_person_tab import MultiPersonTab
        from pathlib import Path

        session = _session_with_tracks()
        tab = MultiPersonTab(session, gvhmr_root=Path("/tmp/gvhmr"))
        tab._reprocess_worker = "fake"

        tab._on_reprocess_error("test error")
        assert tab._reprocess_worker is None
