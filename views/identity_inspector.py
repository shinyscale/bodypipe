"""Identity Inspector panel — person selection, confidence visualization,
keyframe management, bbox editing, review scanning, and identity verification.

Why: The identity inspector is the primary tool for verifying and correcting
multi-person tracking results. It provides per-person confidence visualization,
keyframe-based annotation, two-click bbox editing with interpolation, review
scanning for automated issue detection, reprocessing of dirty persons, and CRUD
operations for managing tracked identities. All state flows through
Session.person_tracks — no module-level dicts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QComboBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QDoubleSpinBox,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor

from models.session import Session, PersonTrack, UndoEntry
from theme import COLORS, PERSON_COLORS
from views.confidence_timeline import ConfidenceTimeline

log = logging.getLogger(__name__)


# Confidence metric names and their display labels
CONFIDENCE_METRICS = [
    "detection", "visibility", "overlap", "shape", "motion", "overall",
]

CONFIDENCE_LABELS = {
    "detection": "Detection",
    "visibility": "Visibility",
    "overlap": "Overlap",
    "shape": "Shape",
    "motion": "Motion",
    "overall": "Overall",
}


@dataclass
class ReviewIssue:
    """A flagged issue found by the review scanner.

    Why: Automated scanning catches problems that are tedious to find manually —
    low-confidence spans, potential identity swaps from high overlap, shape drift,
    and gaps in detection. The scanner prioritizes issues by severity so the user
    reviews the most critical problems first.
    """

    frame: int
    person_id: int
    issue_type: str  # "low_confidence", "potential_swap", "shape_drift", "track_gap"
    description: str
    severity: float  # 0-1, higher = more urgent


def compute_review_issues(
    session: Session,
    low_conf_threshold: float = 0.4,
    gap_threshold: int = 30,
) -> list[ReviewIssue]:
    """Scan all active persons for issues that need human review.

    Detects four issue types:
    - low_confidence: spans of >= 5 frames where overall confidence < threshold
    - potential_swap: high bbox overlap (> 0.5) suggesting identity switch
    - shape_drift: low shape confidence (< 0.4) suggesting body changed
    - track_gap: detection gaps of >= gap_threshold consecutive zero-bbox frames

    Returns issues sorted by frame number.
    """
    issues: list[ReviewIssue] = []

    for pid, track in session.person_tracks.items():
        if pid in session.inactive_tracks:
            continue

        confs = track.confidences
        if not confs:
            continue

        num_frames = len(confs)

        # --- Low confidence spans (>= 5 frames below threshold) ---
        span_start: int | None = None
        for f in range(num_frames):
            if confs[f] < low_conf_threshold:
                if span_start is None:
                    span_start = f
            else:
                if span_start is not None and f - span_start >= 5:
                    mid = (span_start + f) // 2
                    issues.append(ReviewIssue(
                        frame=mid,
                        person_id=pid,
                        issue_type="low_confidence",
                        description=(
                            f"Low confidence span frames {span_start}-{f - 1} "
                            f"({f - span_start} frames)"
                        ),
                        severity=0.7,
                    ))
                span_start = None
        if span_start is not None and num_frames - span_start >= 5:
            mid = (span_start + num_frames) // 2
            issues.append(ReviewIssue(
                frame=mid,
                person_id=pid,
                issue_type="low_confidence",
                description=(
                    f"Low confidence span frames {span_start}-{num_frames - 1}"
                ),
                severity=0.7,
            ))

        # --- Potential swap (high bbox overlap) ---
        if track.confidence_breakdown and "overlap" in track.confidence_breakdown:
            overlaps = track.confidence_breakdown["overlap"]
            for f in range(len(overlaps)):
                if overlaps[f] > 0.5:
                    if f == 0 or overlaps[f - 1] <= 0.5:
                        issues.append(ReviewIssue(
                            frame=f,
                            person_id=pid,
                            issue_type="potential_swap",
                            description=(
                                f"High bbox overlap ({overlaps[f]:.2f}) "
                                f"— possible identity swap"
                            ),
                            severity=0.9,
                        ))

        # --- Shape drift (low shape confidence) ---
        if track.confidence_breakdown and "shape" in track.confidence_breakdown:
            shapes = track.confidence_breakdown["shape"]
            for f in range(len(shapes)):
                if shapes[f] < 0.4:
                    if f == 0 or shapes[f - 1] >= 0.4:
                        issues.append(ReviewIssue(
                            frame=f,
                            person_id=pid,
                            issue_type="shape_drift",
                            description=(
                                f"Shape confidence low ({shapes[f]:.2f}) "
                                f"— body shape may have changed"
                            ),
                            severity=0.6,
                        ))

        # --- Track gaps (zero bboxes for >= gap_threshold frames) ---
        if track.bboxes is not None:
            gap_start: int | None = None
            for f in range(len(track.bboxes)):
                if np.all(track.bboxes[f] == 0):
                    if gap_start is None:
                        gap_start = f
                else:
                    if gap_start is not None and f - gap_start >= gap_threshold:
                        mid = (gap_start + f) // 2
                        issues.append(ReviewIssue(
                            frame=mid,
                            person_id=pid,
                            issue_type="track_gap",
                            description=(
                                f"Detection gap frames {gap_start}-{f - 1} "
                                f"({f - gap_start} frames)"
                            ),
                            severity=0.5,
                        ))
                    gap_start = None
            if gap_start is not None and len(track.bboxes) - gap_start >= gap_threshold:
                mid = (gap_start + len(track.bboxes)) // 2
                issues.append(ReviewIssue(
                    frame=mid,
                    person_id=pid,
                    issue_type="track_gap",
                    description=(
                        f"Detection gap frames {gap_start}-{len(track.bboxes) - 1}"
                    ),
                    severity=0.5,
                ))

    issues.sort(key=lambda i: i.frame)
    return issues


def _confidence_color(value: float) -> str:
    """Return CSS color string based on confidence value."""
    if value > 0.8:
        return COLORS["success"]
    elif value > 0.5:
        return COLORS["warning"]
    else:
        return COLORS["error"]


class IdentityInspector(QWidget):
    """Interactive per-person identity verification panel.

    Provides:
    - Confidence timeline (ConfidenceTimeline widget) with keyframe markers
    - Confidence breakdown (6 labeled metrics per frame)
    - Keyframe table (QTableWidget) with CRUD operations
    - Navigate between keyframes
    """

    person_changed = Signal(int)           # person_id
    frame_requested = Signal(int)          # seek to frame
    person_dirty = Signal(int)             # person_id needs reprocess
    reprocess_requested = Signal(list)     # list of dirty person_ids
    bbox_overlay_changed = Signal(object)  # updated overlay data for video player
    keyframe_changed = Signal(int, int)    # person_id, frame_index
    track_modified = Signal()              # tracks split/merged/swapped

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self._session = session
        self._current_person_id: int = -1
        self._current_frame: int = 0

        # BBox edit state machine: None → "click1" → "click2" → None
        self._bbox_edit_state: str | None = None
        self._bbox_edit_corner1: tuple[int, int] | None = None

        # Crossing span two-click state
        self._crossing_start_frame: int | None = None

        # Review scanner state
        self._review_issues: list[ReviewIssue] = []
        self._review_issue_idx: int = 0

        # Show-all-tracks state (checkbox now lives in PersonSelectorBar)
        self._show_all: bool = False

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header
        header = QLabel("Identity Inspector")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        # ---- Confidence Timeline ----
        self._timeline = ConfidenceTimeline()
        layout.addWidget(self._timeline)

        # ---- Confidence Breakdown ----
        breakdown_group = QGroupBox("Confidence Breakdown")
        breakdown_layout = QVBoxLayout(breakdown_group)

        self._conf_labels: dict[str, QLabel] = {}

        # Row 1: Detection, Visibility, Overlap
        row1 = QHBoxLayout()
        for metric in ["detection", "visibility", "overlap"]:
            lbl = QLabel(f"{CONFIDENCE_LABELS[metric]}: \u2014")
            lbl.setMinimumWidth(100)
            self._conf_labels[metric] = lbl
            row1.addWidget(lbl)
        row1.addStretch()
        breakdown_layout.addLayout(row1)

        # Row 2: Shape, Motion, Overall
        row2 = QHBoxLayout()
        for metric in ["shape", "motion", "overall"]:
            lbl = QLabel(f"{CONFIDENCE_LABELS[metric]}: \u2014")
            lbl.setMinimumWidth(100)
            self._conf_labels[metric] = lbl
            row2.addWidget(lbl)
        row2.addStretch()
        breakdown_layout.addLayout(row2)

        layout.addWidget(breakdown_group)

        # ---- Keyframe Table ----
        kf_group = QGroupBox("Keyframes")
        kf_layout = QVBoxLayout(kf_group)

        self._keyframe_table = QTableWidget(0, 5)
        self._keyframe_table.setHorizontalHeaderLabels(
            ["Frame", "Verified", "Confidence", "BBox", "Actions"]
        )
        self._keyframe_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._keyframe_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._keyframe_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._keyframe_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._keyframe_table.verticalHeader().hide()
        self._keyframe_table.setMaximumHeight(180)
        kf_layout.addWidget(self._keyframe_table)

        # Keyframe action buttons
        kf_buttons = QHBoxLayout()

        self._verify_btn = QPushButton("\u2713 Verify")
        self._verify_btn.setToolTip("Toggle verified status at current frame")
        kf_buttons.addWidget(self._verify_btn)

        self._add_kf_btn = QPushButton("+ Add KF")
        self._add_kf_btn.setToolTip("Add keyframe at current frame")
        kf_buttons.addWidget(self._add_kf_btn)

        self._remove_kf_btn = QPushButton("- Remove KF")
        self._remove_kf_btn.setToolTip("Remove keyframe at current frame")
        kf_buttons.addWidget(self._remove_kf_btn)

        kf_layout.addLayout(kf_buttons)

        # Keyframe navigation buttons
        kf_nav = QHBoxLayout()

        self._prev_kf_btn = QPushButton("\u25c4 Prev KF")
        self._prev_kf_btn.setToolTip("Navigate to previous keyframe")
        kf_nav.addWidget(self._prev_kf_btn)

        self._next_kf_btn = QPushButton("\u25ba Next KF")
        self._next_kf_btn.setToolTip("Navigate to next keyframe")
        kf_nav.addWidget(self._next_kf_btn)

        kf_layout.addLayout(kf_nav)

        layout.addWidget(kf_group)

        # ---- BBox Editing ----
        bbox_group = QGroupBox("BBox Editing")
        bbox_layout = QVBoxLayout(bbox_group)

        self._bbox_status = QLabel("Idle")
        self._bbox_status.setStyleSheet("font-style: italic;")
        bbox_layout.addWidget(self._bbox_status)

        bbox_btn_row1 = QHBoxLayout()
        self._edit_bbox_btn = QPushButton("Edit BBox")
        self._edit_bbox_btn.setToolTip("Start two-click bbox editing on video frame")
        bbox_btn_row1.addWidget(self._edit_bbox_btn)

        self._cancel_edit_btn = QPushButton("Cancel Edit")
        self._cancel_edit_btn.setToolTip("Abort current bbox edit")
        self._cancel_edit_btn.setEnabled(False)
        bbox_btn_row1.addWidget(self._cancel_edit_btn)
        bbox_layout.addLayout(bbox_btn_row1)

        bbox_btn_row2 = QHBoxLayout()
        self._interpolate_btn = QPushButton("Interpolate")
        self._interpolate_btn.setToolTip(
            "Linear interpolation of bbox corrections between keyframes"
        )
        bbox_btn_row2.addWidget(self._interpolate_btn)

        self._apply_btn = QPushButton("Apply All")
        self._apply_btn.setToolTip("Save bbox corrections to disk")
        bbox_btn_row2.addWidget(self._apply_btn)
        bbox_layout.addLayout(bbox_btn_row2)

        layout.addWidget(bbox_group)

        # ---- Track Operations ----
        track_ops_group = QGroupBox("Track Operations")
        track_ops_layout = QVBoxLayout(track_ops_group)

        # Swap row
        swap_row = QHBoxLayout()
        swap_row.addWidget(QLabel("Swap with:"))
        self._swap_combo = QComboBox()
        self._swap_combo.setMinimumWidth(80)
        swap_row.addWidget(self._swap_combo, 1)
        self._swap_btn = QPushButton("Swap IDs")
        self._swap_btn.setToolTip("Swap person IDs from current frame onward")
        swap_row.addWidget(self._swap_btn)
        track_ops_layout.addLayout(swap_row)

        # Split
        self._split_btn = QPushButton("Split at Frame")
        self._split_btn.setToolTip(
            "Split track at current frame — subsequent frames become a new inactive track"
        )
        track_ops_layout.addWidget(self._split_btn)

        # Merge row
        merge_row = QHBoxLayout()
        merge_row.addWidget(QLabel("Merge from:"))
        self._merge_combo = QComboBox()
        self._merge_combo.setMinimumWidth(80)
        merge_row.addWidget(self._merge_combo, 1)
        self._merge_btn = QPushButton("Merge")
        self._merge_btn.setToolTip("Merge selected inactive track into current person")
        merge_row.addWidget(self._merge_btn)
        track_ops_layout.addLayout(merge_row)

        # Crossing spans
        crossing_label = QLabel("Crossing Spans")
        crossing_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        track_ops_layout.addWidget(crossing_label)

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Proximity threshold:"))
        self._crossing_threshold_spin = QDoubleSpinBox()
        self._crossing_threshold_spin.setRange(0.05, 0.50)
        self._crossing_threshold_spin.setSingleStep(0.05)
        self._crossing_threshold_spin.setDecimals(2)
        self._crossing_threshold_spin.setValue(self._session.crossing_threshold)
        self._crossing_threshold_spin.setToolTip(
            "Bbox overlap IoU that triggers auto-crossing detection. "
            "Higher = less sensitive (fewer bridges). Default 0.15")
        self._crossing_threshold_spin.valueChanged.connect(self._on_crossing_threshold_changed)
        threshold_row.addWidget(self._crossing_threshold_spin)
        track_ops_layout.addLayout(threshold_row)

        crossing_row = QHBoxLayout()
        self._crossing_start_btn = QPushButton("Mark Start")
        self._crossing_start_btn.setToolTip("Mark start of a crossing/occlusion span")
        crossing_row.addWidget(self._crossing_start_btn)
        self._crossing_end_btn = QPushButton("Mark End")
        self._crossing_end_btn.setToolTip("Mark end of the crossing span")
        self._crossing_end_btn.setEnabled(False)
        crossing_row.addWidget(self._crossing_end_btn)
        track_ops_layout.addLayout(crossing_row)

        self._crossing_status = QLabel("")
        self._crossing_status.setStyleSheet("font-style: italic;")
        track_ops_layout.addWidget(self._crossing_status)

        self._crossing_table = QTableWidget(0, 4)
        self._crossing_table.setHorizontalHeaderLabels(
            ["Person", "Start", "End", "Actions"]
        )
        self._crossing_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._crossing_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._crossing_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._crossing_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._crossing_table.verticalHeader().hide()
        self._crossing_table.setMaximumHeight(120)
        track_ops_layout.addWidget(self._crossing_table)

        layout.addWidget(track_ops_group)

        # ---- Review Scanner ----
        scanner_group = QGroupBox("Review Scanner")
        scanner_layout = QVBoxLayout(scanner_group)

        scanner_btn_row = QHBoxLayout()
        self._scan_btn = QPushButton("Scan")
        self._scan_btn.setToolTip(
            "Auto-detect issues: low confidence, overlaps, shape drift, gaps"
        )
        scanner_btn_row.addWidget(self._scan_btn)

        self._prev_issue_btn = QPushButton("\u25c4 Prev")
        self._prev_issue_btn.setToolTip("Navigate to previous issue")
        self._prev_issue_btn.setEnabled(False)
        scanner_btn_row.addWidget(self._prev_issue_btn)

        self._next_issue_btn = QPushButton("\u25ba Next")
        self._next_issue_btn.setToolTip("Navigate to next issue")
        self._next_issue_btn.setEnabled(False)
        scanner_btn_row.addWidget(self._next_issue_btn)
        scanner_layout.addLayout(scanner_btn_row)

        self._issues_label = QLabel("No scan performed")
        self._issues_label.setStyleSheet("font-style: italic;")
        scanner_layout.addWidget(self._issues_label)

        self._issues_table = QTableWidget(0, 4)
        self._issues_table.setHorizontalHeaderLabels(
            ["Frame", "Person", "Type", "Description"]
        )
        self._issues_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._issues_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._issues_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._issues_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._issues_table.verticalHeader().hide()
        self._issues_table.setMaximumHeight(140)
        scanner_layout.addWidget(self._issues_table)

        layout.addWidget(scanner_group)

        # ---- Reprocess ----
        self._reprocess_btn = QPushButton("Reprocess (0 dirty)")
        self._reprocess_btn.setToolTip(
            "Reprocess all persons with pending bbox corrections"
        )
        self._reprocess_btn.setEnabled(False)
        self._reprocess_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 8px; }"
        )
        layout.addWidget(self._reprocess_btn)

        layout.addStretch()

    def _connect_signals(self):
        self._timeline.frame_clicked.connect(self._on_timeline_clicked)
        self._verify_btn.clicked.connect(self._on_verify)
        self._add_kf_btn.clicked.connect(self._on_add_keyframe)
        self._remove_kf_btn.clicked.connect(self._on_remove_keyframe)
        self._prev_kf_btn.clicked.connect(self._on_prev_keyframe)
        self._next_kf_btn.clicked.connect(self._on_next_keyframe)
        self._keyframe_table.cellDoubleClicked.connect(self._on_table_double_clicked)
        self._edit_bbox_btn.clicked.connect(self._on_edit_bbox)
        self._cancel_edit_btn.clicked.connect(self._on_cancel_edit)
        self._interpolate_btn.clicked.connect(self._on_interpolate)
        self._apply_btn.clicked.connect(self._on_apply_all)
        self._swap_btn.clicked.connect(self._on_swap_ids)
        self._split_btn.clicked.connect(self._on_split_track)
        self._merge_btn.clicked.connect(self._on_merge_track)
        self._crossing_start_btn.clicked.connect(self._on_crossing_start)
        self._crossing_end_btn.clicked.connect(self._on_crossing_end)
        self._scan_btn.clicked.connect(self._on_scan_issues)
        self._prev_issue_btn.clicked.connect(self._on_prev_issue)
        self._next_issue_btn.clicked.connect(self._on_next_issue)
        self._issues_table.cellDoubleClicked.connect(self._on_issue_double_clicked)
        self._reprocess_btn.clicked.connect(self._on_reprocess)
        self.person_dirty.connect(lambda _: self.update_reprocess_button())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_person(self, person_id: int):
        """Select a person by ID and update all displays."""
        if person_id == self._current_person_id:
            return
        self._current_person_id = person_id

        # Cancel any in-progress bbox edit when switching person
        if self._bbox_edit_state is not None:
            self._cancel_bbox_edit()

        self._refresh_for_person()

    def set_frame(self, frame_idx: int):
        """Update displays for a new frame position."""
        self._current_frame = frame_idx
        self._timeline.set_current_frame(frame_idx)
        self._update_confidence_breakdown(frame_idx)

    def refresh(self):
        """Rebuild all UI from session state (e.g., after pipeline finishes)."""
        self._update_merge_combo()
        if self._current_person_id >= 0:
            self._refresh_for_person()
        self.update_reprocess_button()

    def on_frame_click(self, x_norm: float, y_norm: float):
        """Handle click on video frame during bbox editing.

        Called by MultiPersonTab when VideoPlayer emits frame_clicked.
        Coordinates are normalized [0, 1] in image space.
        """
        if self._bbox_edit_state is None:
            return

        track = self._get_current_track()
        if track is None:
            return

        # Convert normalized coords to pixel coords
        x_px = int(x_norm * self._session.img_width)
        y_px = int(y_norm * self._session.img_height)

        if self._bbox_edit_state == "click1":
            # First click: store top-left corner
            self._bbox_edit_corner1 = (x_px, y_px)
            self._bbox_edit_state = "click2"
            self._update_bbox_edit_status()
            # Emit overlay to show crosshair at corner1
            self.bbox_overlay_changed.emit({
                "edit_preview": {"corner1": self._bbox_edit_corner1}
            })

        elif self._bbox_edit_state == "click2":
            # Second click: complete the bbox
            x1, y1 = self._bbox_edit_corner1
            x2, y2 = x_px, y_px

            # Normalize: ensure x1 < x2, y1 < y2
            bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

            # Enforce minimum size (20px)
            if bbox[2] - bbox[0] < 20:
                bbox[2] = bbox[0] + 20
            if bbox[3] - bbox[1] < 20:
                bbox[3] = bbox[1] + 20

            # Clamp to image bounds
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(self._session.img_width - 1, bbox[2])
            bbox[3] = min(self._session.img_height - 1, bbox[3])

            # Snapshot for undo (full corrections array + keyframes)
            pid = self._current_person_id
            frame = self._current_frame
            old_corrections = (
                track.bbox_corrections.copy()
                if track.bbox_corrections is not None
                else None
            )
            old_keyframes = list(track.keyframes)

            # Store correction
            self._store_bbox_correction(track, self._current_frame, bbox)

            # Add keyframe at this frame if not present
            if not any(kf["frame"] == self._current_frame for kf in track.keyframes):
                track.keyframes.append({
                    "frame": self._current_frame,
                    "verified": False,
                })

            # Auto-interpolate across all frames
            self._auto_interpolate(track)

            # Undo/redo with full-array snapshots
            new_corrections = track.bbox_corrections.copy()
            new_keyframes = list(track.keyframes)

            def undo(p=pid, f=frame, old_c=old_corrections, old_kf=old_keyframes):
                t = self._session.person_tracks.get(p)
                if t:
                    t.bbox_corrections = old_c.copy() if old_c is not None else None
                    t.keyframes = list(old_kf)
                self._session.dirty_persons.discard(p)
                self._refresh_for_person()
                self.keyframe_changed.emit(p, f)
                self.bbox_overlay_changed.emit({"edit_preview": None})

            def redo(p=pid, f=frame, new_c=new_corrections, new_kf=new_keyframes):
                t = self._session.person_tracks.get(p)
                if t:
                    t.bbox_corrections = new_c.copy()
                    t.keyframes = list(new_kf)
                    self._session.dirty_persons.add(p)
                self._refresh_for_person()
                self.keyframe_changed.emit(p, f)
                self.bbox_overlay_changed.emit({"edit_preview": None})

            self._session.undo_stack.push(UndoEntry("BBox correction", undo, redo))

            # Mark person dirty
            self._session.dirty_persons.add(self._current_person_id)
            self.person_dirty.emit(self._current_person_id)

            log.info(
                "BBox correction applied: person=%d frame=%d bbox=%s",
                self._current_person_id, self._current_frame, bbox,
            )

            # Reset edit state
            self._bbox_edit_state = None
            self._bbox_edit_corner1 = None
            self._update_bbox_edit_status()

            # Refresh displays
            self._update_keyframe_table()
            self._update_timeline()
            self.keyframe_changed.emit(self._current_person_id, self._current_frame)
            self.bbox_overlay_changed.emit({"edit_preview": None})

    def on_bbox_drag(self, x1_norm: float, y1_norm: float,
                      x2_norm: float, y2_norm: float):
        """Handle click-and-drag bbox from video display.

        Directly applies the dragged rectangle as a bbox correction,
        bypassing the two-click state machine. Works regardless of
        whether bbox edit mode is active.
        """
        track = self._get_current_track()
        if track is None:
            return

        # Convert normalized coords to pixel coords
        w, h = self._session.img_width, self._session.img_height
        bbox = [
            max(0, int(x1_norm * w)),
            max(0, int(y1_norm * h)),
            min(w - 1, int(x2_norm * w)),
            min(h - 1, int(y2_norm * h)),
        ]

        # Enforce minimum size (20px)
        if bbox[2] - bbox[0] < 20:
            bbox[2] = bbox[0] + 20
        if bbox[3] - bbox[1] < 20:
            bbox[3] = bbox[1] + 20

        # Snapshot for undo (full corrections array + keyframes)
        pid = self._current_person_id
        frame = self._current_frame
        old_corrections = (
            track.bbox_corrections.copy()
            if track.bbox_corrections is not None
            else None
        )
        old_keyframes = list(track.keyframes)

        # Store correction
        self._store_bbox_correction(track, frame, bbox)

        # Add keyframe at this frame if not present
        if not any(kf["frame"] == frame for kf in track.keyframes):
            track.keyframes.append({"frame": frame, "verified": False})

        # Auto-interpolate across all frames
        self._auto_interpolate(track)

        # Undo/redo with full-array snapshots
        new_corrections = track.bbox_corrections.copy()
        new_keyframes = list(track.keyframes)

        def undo(p=pid, f=frame, old_c=old_corrections, old_kf=old_keyframes):
            t = self._session.person_tracks.get(p)
            if t:
                t.bbox_corrections = old_c.copy() if old_c is not None else None
                t.keyframes = list(old_kf)
            self._session.dirty_persons.discard(p)
            self._refresh_for_person()
            self.keyframe_changed.emit(p, f)
            self.bbox_overlay_changed.emit({"edit_preview": None})

        def redo(p=pid, f=frame, new_c=new_corrections, new_kf=new_keyframes):
            t = self._session.person_tracks.get(p)
            if t:
                t.bbox_corrections = new_c.copy()
                t.keyframes = list(new_kf)
                self._session.dirty_persons.add(p)
            self._refresh_for_person()
            self.keyframe_changed.emit(p, f)
            self.bbox_overlay_changed.emit({"edit_preview": None})

        self._session.undo_stack.push(UndoEntry("BBox drag", undo, redo))

        # Mark dirty
        self._session.dirty_persons.add(pid)
        self.person_dirty.emit(pid)

        log.info("BBox drag correction: person=%d frame=%d bbox=%s", pid, frame, bbox)

        # Cancel any active two-click edit
        self._bbox_edit_state = None
        self._bbox_edit_corner1 = None
        self._update_bbox_edit_status()

        # Refresh displays
        self._update_keyframe_table()
        self._update_timeline()
        self.keyframe_changed.emit(pid, frame)
        self.bbox_overlay_changed.emit({"edit_preview": None})

    # ------------------------------------------------------------------
    # Internal updates
    # ------------------------------------------------------------------



    def _refresh_for_person(self):
        """Update all displays for the current person."""
        self._update_timeline()
        self._update_confidence_breakdown(self._current_frame)
        self._update_keyframe_table()
        self._update_swap_combo()
        self._update_merge_combo()
        self._update_crossing_table()

    def _update_timeline(self):
        """Update the confidence timeline for the current person.

        When "show all tracks" is checked, keyframe markers from every
        person are merged so the user can navigate across all keyframes.
        """
        track = self._get_current_track()
        if track is None:
            self._timeline.set_data(np.zeros(0))
            return

        confidences = self._get_confidences(track)
        keyframe_frames = [kf["frame"] for kf in track.keyframes]
        verified_frames = {
            kf["frame"] for kf in track.keyframes if kf.get("verified", False)
        }

        # Merge keyframes from all persons when "show all tracks" is active
        if self._show_all:
            for pid, t in self._session.person_tracks.items():
                if pid == self._current_person_id:
                    continue
                for kf in t.keyframes:
                    if kf["frame"] not in keyframe_frames:
                        keyframe_frames.append(kf["frame"])
                    if kf.get("verified", False):
                        verified_frames.add(kf["frame"])

        self._timeline.set_data(confidences, keyframe_frames, verified_frames)

    def _update_confidence_breakdown(self, frame_idx: int):
        """Update the 6 confidence labels for a specific frame."""
        track = self._get_current_track()

        for metric in CONFIDENCE_METRICS:
            label = self._conf_labels[metric]
            display_name = CONFIDENCE_LABELS[metric]

            if track is None:
                label.setText(f"{display_name}: \u2014")
                label.setStyleSheet("")
                continue

            value = self._get_confidence_value(track, metric, frame_idx)
            if value is None:
                label.setText(f"{display_name}: \u2014")
                label.setStyleSheet("")
            else:
                color = _confidence_color(value)
                label.setText(f"{display_name}: {value:.2f}")
                label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _update_keyframe_table(self):
        """Rebuild the keyframe table from current person's keyframes."""
        self._keyframe_table.setRowCount(0)

        track = self._get_current_track()
        if track is None:
            return

        keyframes = sorted(track.keyframes, key=lambda kf: kf["frame"])
        self._keyframe_table.setRowCount(len(keyframes))

        for row, kf in enumerate(keyframes):
            frame = kf["frame"]
            verified = kf.get("verified", False)

            # Frame column
            frame_item = QTableWidgetItem(str(frame))
            frame_item.setTextAlignment(Qt.AlignCenter)
            self._keyframe_table.setItem(row, 0, frame_item)

            # Verified column
            verified_item = QTableWidgetItem("\u2713" if verified else "\u2717")
            verified_item.setTextAlignment(Qt.AlignCenter)
            if verified:
                verified_item.setForeground(QColor(COLORS["success"]))
            else:
                verified_item.setForeground(QColor(COLORS["error"]))
            self._keyframe_table.setItem(row, 1, verified_item)

            # Confidence column
            conf = self._get_confidence_at_frame(track, frame)
            conf_item = QTableWidgetItem(
                f"{conf:.2f}" if conf is not None else "\u2014"
            )
            conf_item.setTextAlignment(Qt.AlignCenter)
            if conf is not None:
                conf_item.setForeground(QColor(_confidence_color(conf)))
            self._keyframe_table.setItem(row, 2, conf_item)

            # BBox column
            bbox = self._get_bbox_at_frame(track, frame)
            if bbox is not None:
                bbox_str = f"{int(bbox[2] - bbox[0])}x{int(bbox[3] - bbox[1])}"
            else:
                bbox_str = "\u2014"
            bbox_item = QTableWidgetItem(bbox_str)
            bbox_item.setTextAlignment(Qt.AlignCenter)
            self._keyframe_table.setItem(row, 3, bbox_item)

            # Actions column — Go and Del buttons
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 0, 2, 0)
            actions_layout.setSpacing(2)

            go_btn = QPushButton("Go")
            go_btn.setFixedSize(32, 22)
            go_btn.clicked.connect(
                lambda checked, f=frame: self._on_keyframe_go(f)
            )
            actions_layout.addWidget(go_btn)

            del_btn = QPushButton("Del")
            del_btn.setFixedSize(32, 22)
            del_btn.clicked.connect(
                lambda checked, f=frame: self._on_keyframe_delete(f)
            )
            actions_layout.addWidget(del_btn)

            self._keyframe_table.setCellWidget(row, 4, actions_widget)

    # ------------------------------------------------------------------
    # Data access helpers
    # ------------------------------------------------------------------

    def _get_current_track(self) -> PersonTrack | None:
        """Get the PersonTrack for the currently selected person."""
        if self._current_person_id < 0:
            return None
        return self._session.person_tracks.get(self._current_person_id)

    def _get_confidences(self, track: PersonTrack) -> np.ndarray:
        """Get overall confidence array for a track."""
        if track.confidences is not None:
            return np.array(track.confidences)
        # Fallback: uniform confidence
        return np.ones(max(1, self._session.num_frames)) * 0.5

    def _get_confidence_value(
        self, track: PersonTrack, metric: str, frame_idx: int
    ) -> float | None:
        """Get a specific confidence metric value at a frame."""
        if track.confidence_breakdown and metric in track.confidence_breakdown:
            values = track.confidence_breakdown[metric]
            if 0 <= frame_idx < len(values):
                return float(values[frame_idx])

        # Fallback: use overall confidences for "overall" metric
        if metric == "overall" and track.confidences is not None:
            if 0 <= frame_idx < len(track.confidences):
                return float(track.confidences[frame_idx])

        return None

    def _get_confidence_at_frame(
        self, track: PersonTrack, frame_idx: int
    ) -> float | None:
        """Get overall confidence at a specific frame."""
        if track.confidences is not None and 0 <= frame_idx < len(track.confidences):
            return float(track.confidences[frame_idx])
        return None

    def _get_bbox_at_frame(
        self, track: PersonTrack, frame_idx: int
    ) -> np.ndarray | None:
        """Get bbox [x1, y1, x2, y2] at frame, preferring corrections."""
        if track.bbox_corrections is not None and frame_idx < len(
            track.bbox_corrections
        ):
            bbox = track.bbox_corrections[frame_idx]
            if bbox is not None and not np.all(bbox == 0):
                return bbox
        if track.bboxes is not None and frame_idx < len(track.bboxes):
            return track.bboxes[frame_idx]
        return None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _snapshot_track(track: PersonTrack) -> dict:
        """Capture mutable track data for undo/redo snapshots."""
        return {
            "bboxes": track.bboxes.copy() if track.bboxes is not None else None,
            "confidences": list(track.confidences) if track.confidences is not None else None,
            "bbox_corrections": track.bbox_corrections.copy() if track.bbox_corrections is not None else None,
            "original_bboxes": track.original_bboxes.copy() if track.original_bboxes is not None else None,
            "keyframes": [dict(kf) for kf in track.keyframes],
        }

    @staticmethod
    def _restore_track(track: PersonTrack, snap: dict) -> None:
        """Restore mutable track data from a snapshot."""
        track.bboxes = snap["bboxes"].copy() if snap["bboxes"] is not None else None
        track.confidences = list(snap["confidences"]) if snap["confidences"] is not None else None
        track.bbox_corrections = snap["bbox_corrections"].copy() if snap["bbox_corrections"] is not None else None
        track.original_bboxes = snap["original_bboxes"].copy() if snap["original_bboxes"] is not None else None
        track.keyframes = [dict(kf) for kf in snap["keyframes"]]

    def _on_timeline_clicked(self, frame_idx: int):
        """Handle click on confidence timeline — seek to frame."""
        self._current_frame = frame_idx
        self._update_confidence_breakdown(frame_idx)
        self.frame_requested.emit(frame_idx)

    def _on_verify(self):
        """Toggle verified status at current frame (add keyframe if needed)."""
        track = self._get_current_track()
        if track is None:
            return

        pid = self._current_person_id
        frame = self._current_frame
        old_keyframes = [dict(kf) for kf in track.keyframes]

        # Find existing keyframe at current frame
        existing = None
        for kf in track.keyframes:
            if kf["frame"] == self._current_frame:
                existing = kf
                break

        if existing:
            # Toggle verified
            existing["verified"] = not existing["verified"]
        else:
            # Add new verified keyframe
            track.keyframes.append({
                "frame": self._current_frame,
                "verified": True,
            })

        new_keyframes = [dict(kf) for kf in track.keyframes]

        def undo(old_kf=old_keyframes, p=pid, f=frame):
            t = self._session.person_tracks.get(p)
            if t:
                t.keyframes = [dict(kf) for kf in old_kf]
            self._refresh_for_person()
            self.keyframe_changed.emit(p, f)

        def redo(new_kf=new_keyframes, p=pid, f=frame):
            t = self._session.person_tracks.get(p)
            if t:
                t.keyframes = [dict(kf) for kf in new_kf]
            self._refresh_for_person()
            self.keyframe_changed.emit(p, f)

        self._session.undo_stack.push(UndoEntry("Verify keyframe", undo, redo))

        self._update_timeline()
        self._update_keyframe_table()
        self.keyframe_changed.emit(self._current_person_id, self._current_frame)

    def _on_add_keyframe(self):
        """Add a keyframe at the current frame."""
        track = self._get_current_track()
        if track is None:
            return

        # No-op if keyframe already exists at this frame
        for kf in track.keyframes:
            if kf["frame"] == self._current_frame:
                return

        pid = self._current_person_id
        frame = self._current_frame
        kf_entry = {"frame": frame, "verified": False}
        track.keyframes.append(kf_entry)

        def undo(p=pid, f=frame):
            t = self._session.person_tracks.get(p)
            if t:
                t.keyframes = [kf for kf in t.keyframes if kf["frame"] != f]
            self._refresh_for_person()
            self.keyframe_changed.emit(p, f)

        def redo(p=pid, entry=dict(kf_entry)):
            t = self._session.person_tracks.get(p)
            if t and not any(kf["frame"] == entry["frame"] for kf in t.keyframes):
                t.keyframes.append(dict(entry))
            self._refresh_for_person()
            self.keyframe_changed.emit(p, entry["frame"])

        self._session.undo_stack.push(UndoEntry("Add keyframe", undo, redo))

        self._update_timeline()
        self._update_keyframe_table()
        self.keyframe_changed.emit(self._current_person_id, self._current_frame)

    def _on_remove_keyframe(self):
        """Remove keyframe at the current frame."""
        track = self._get_current_track()
        if track is None:
            return

        pid = self._current_person_id
        frame = self._current_frame
        removed = [dict(kf) for kf in track.keyframes if kf["frame"] == frame]
        if not removed:
            return

        track.keyframes = [
            kf for kf in track.keyframes if kf["frame"] != self._current_frame
        ]

        def undo(p=pid, f=frame, entries=removed):
            t = self._session.person_tracks.get(p)
            if t:
                t.keyframes.extend(dict(e) for e in entries)
            self._refresh_for_person()
            self.keyframe_changed.emit(p, f)

        def redo(p=pid, f=frame):
            t = self._session.person_tracks.get(p)
            if t:
                t.keyframes = [kf for kf in t.keyframes if kf["frame"] != f]
            self._refresh_for_person()
            self.keyframe_changed.emit(p, f)

        self._session.undo_stack.push(UndoEntry("Remove keyframe", undo, redo))

        self._update_timeline()
        self._update_keyframe_table()
        self.keyframe_changed.emit(self._current_person_id, self._current_frame)

    def _all_keyframe_frames(self) -> list[int]:
        """Return sorted keyframe frames — all persons if 'show all' is checked."""
        track = self._get_current_track()
        if track is None:
            return []
        frames = {kf["frame"] for kf in track.keyframes}
        if self._show_all:
            for pid, t in self._session.person_tracks.items():
                if pid == self._current_person_id:
                    continue
                for kf in t.keyframes:
                    frames.add(kf["frame"])
        return sorted(frames)

    def _on_prev_keyframe(self):
        """Navigate to the previous keyframe before current frame."""
        frames = self._all_keyframe_frames()
        prev_frames = [f for f in frames if f < self._current_frame]
        if prev_frames:
            self.frame_requested.emit(prev_frames[-1])

    def _on_next_keyframe(self):
        """Navigate to the next keyframe after current frame."""
        frames = self._all_keyframe_frames()
        next_frames = [f for f in frames if f > self._current_frame]
        if next_frames:
            self.frame_requested.emit(next_frames[0])

    def _on_keyframe_go(self, frame: int):
        """Seek to a specific keyframe's frame."""
        self.frame_requested.emit(frame)

    def _on_keyframe_delete(self, frame: int):
        """Delete a specific keyframe by frame number."""
        track = self._get_current_track()
        if track is None:
            return

        pid = self._current_person_id
        removed = [dict(kf) for kf in track.keyframes if kf["frame"] == frame]

        track.keyframes = [kf for kf in track.keyframes if kf["frame"] != frame]

        if removed:
            def undo(p=pid, f=frame, entries=removed):
                t = self._session.person_tracks.get(p)
                if t:
                    t.keyframes.extend(dict(e) for e in entries)
                self._refresh_for_person()
                self.keyframe_changed.emit(p, f)

            def redo(p=pid, f=frame):
                t = self._session.person_tracks.get(p)
                if t:
                    t.keyframes = [kf for kf in t.keyframes if kf["frame"] != f]
                self._refresh_for_person()
                self.keyframe_changed.emit(p, f)

            self._session.undo_stack.push(UndoEntry("Delete keyframe", undo, redo))

        self._update_timeline()
        self._update_keyframe_table()
        self.keyframe_changed.emit(self._current_person_id, frame)

    def set_show_all_tracks(self, checked: bool):
        """Update show-all-tracks state (called from PersonSelectorBar)."""
        self._show_all = checked
        self.bbox_overlay_changed.emit({"show_all": checked})
        self._update_timeline()

    def _on_table_double_clicked(self, row: int, col: int):
        """Handle double-click on keyframe table row — navigate to frame."""
        item = self._keyframe_table.item(row, 0)  # Frame column
        if item:
            frame = int(item.text())
            self.frame_requested.emit(frame)

    # ------------------------------------------------------------------
    # BBox editing
    # ------------------------------------------------------------------

    def _on_edit_bbox(self):
        """Enter bbox edit mode — next two clicks define top-left and bottom-right."""
        track = self._get_current_track()
        if track is None:
            return
        self._bbox_edit_state = "click1"
        self._bbox_edit_corner1 = None
        self._update_bbox_edit_status()

    def _on_cancel_edit(self):
        """Abort current bbox edit."""
        self._cancel_bbox_edit()

    def _cancel_bbox_edit(self):
        """Reset bbox edit state machine to idle."""
        self._bbox_edit_state = None
        self._bbox_edit_corner1 = None
        self._update_bbox_edit_status()
        self.bbox_overlay_changed.emit({"edit_preview": None})

    def _update_bbox_edit_status(self):
        """Update status label and button enabled states for bbox edit mode."""
        editing = self._bbox_edit_state is not None
        self._edit_bbox_btn.setEnabled(not editing)
        self._cancel_edit_btn.setEnabled(editing)

        if self._bbox_edit_state == "click1":
            self._bbox_status.setText("Click top-left corner on video frame")
            self._bbox_status.setStyleSheet(f"font-style: italic; color: {COLORS['warning']};")
        elif self._bbox_edit_state == "click2":
            self._bbox_status.setText("Click bottom-right corner on video frame")
            self._bbox_status.setStyleSheet(f"font-style: italic; color: {COLORS['warning']};")
        else:
            self._bbox_status.setText("Idle")
            self._bbox_status.setStyleSheet("font-style: italic;")

    def _store_bbox_correction(
        self, track: PersonTrack, frame_idx: int, bbox: list[float]
    ):
        """Store a bbox correction [x1, y1, x2, y2] for a specific frame."""
        # Ensure original_bboxes are backed up before first correction
        if track.original_bboxes is None and track.bboxes is not None:
            track.original_bboxes = track.bboxes.copy()

        # Initialize corrections array if needed
        if track.bbox_corrections is None:
            n = max(self._session.num_frames, frame_idx + 1)
            track.bbox_corrections = np.zeros((n, 4), dtype=float)
        elif frame_idx >= len(track.bbox_corrections):
            # Extend array
            old = track.bbox_corrections
            track.bbox_corrections = np.zeros((frame_idx + 1, 4), dtype=float)
            track.bbox_corrections[: len(old)] = old

        track.bbox_corrections[frame_idx] = bbox

    def _auto_interpolate(self, track: PersonTrack):
        """Auto-interpolate bbox corrections between all corrected keyframes.

        Runs after every bbox edit so corrections blend smoothly across the
        entire track rather than only affecting the keyed frame.
        """
        if track.bbox_corrections is None:
            return
        original = (
            track.original_bboxes if track.original_bboxes is not None
            else track.bboxes
        )
        if original is None:
            return
        kf_frames = [kf["frame"] for kf in track.keyframes]
        track.bbox_corrections = interpolate_bbox_corrections(
            original, track.bbox_corrections, keyframe_frames=kf_frames
        )

    def _on_interpolate(self):
        """Interpolate bbox corrections between keyframes using delta-space blending.

        Works in delta space (correction - original) so that small adjustments
        blend smoothly across frames. Requires at least two corrected keyframes.
        """
        track = self._get_current_track()
        if track is None or track.bbox_corrections is None:
            return

        original = (
            track.original_bboxes if track.original_bboxes is not None
            else track.bboxes
        )
        if original is None:
            return

        pid = self._current_person_id
        frame = self._current_frame
        old_corrections = track.bbox_corrections.copy()

        kf_frames = [kf["frame"] for kf in track.keyframes]
        result = interpolate_bbox_corrections(
            original, track.bbox_corrections, keyframe_frames=kf_frames
        )
        track.bbox_corrections = result

        new_corrections = result.copy()

        def undo(p=pid, f=frame, old_c=old_corrections):
            t = self._session.person_tracks.get(p)
            if t:
                t.bbox_corrections = old_c.copy()
            self._refresh_for_person()
            self.keyframe_changed.emit(p, f)

        def redo(p=pid, f=frame, new_c=new_corrections):
            t = self._session.person_tracks.get(p)
            if t:
                t.bbox_corrections = new_c.copy()
                self._session.dirty_persons.add(p)
            self._refresh_for_person()
            self.keyframe_changed.emit(p, f)

        self._session.undo_stack.push(UndoEntry("Interpolate bboxes", undo, redo))

        self._session.dirty_persons.add(self._current_person_id)
        self.person_dirty.emit(self._current_person_id)
        self.keyframe_changed.emit(self._current_person_id, self._current_frame)
        log.info("BBox interpolation applied for person %d", self._current_person_id)

    def _on_apply_all(self):
        """Save bbox corrections to disk for the current person."""
        track = self._get_current_track()
        if track is None or track.bbox_corrections is None:
            return

        self._save_bbox_corrections(track)
        self.bbox_overlay_changed.emit({"applied": True})
        log.info("BBox corrections saved for person %d", self._current_person_id)

    def _save_bbox_corrections(self, track: PersonTrack):
        """Persist bbox corrections to JSON at person_dir/bbox_corrections.json."""
        if track.person_dir is None:
            log.warning(
                "Cannot save bbox corrections: person_dir is None for person %d",
                track.person_id,
            )
            return

        corrections = track.bbox_corrections
        if corrections is None:
            return

        # Build sparse dict of non-zero corrections
        data: dict = {"person_id": track.person_id, "corrections": {}}
        for f in range(len(corrections)):
            if not np.all(corrections[f] == 0):
                data["corrections"][str(f)] = corrections[f].tolist()

        out_path = Path(track.person_dir) / "bbox_corrections.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fp:
            json.dump(data, fp, indent=2)

    # ------------------------------------------------------------------
    # Track operations
    # ------------------------------------------------------------------

    def _on_swap_ids(self):
        """Swap bboxes and confidences with target from current frame onward.

        Why: When the tracker switches person IDs mid-sequence (e.g., after an
        occlusion), swapping corrects the identity assignment from that point on
        without requiring manual bbox editing of every frame.
        """
        if self._current_person_id < 0:
            return
        target_id = self._swap_combo.currentData()
        if target_id is None or target_id == self._current_person_id:
            return

        track_a = self._session.person_tracks.get(self._current_person_id)
        track_b = self._session.person_tracks.get(target_id)
        if track_a is None or track_b is None:
            return

        pid_a = self._current_person_id
        pid_b = target_id
        frame_idx = self._current_frame

        # Snapshot for undo
        snap_a = self._snapshot_track(track_a)
        snap_b = self._snapshot_track(track_b)

        # Swap numpy arrays from frame_idx onward
        for attr in ("bboxes", "original_bboxes", "bbox_corrections"):
            arr_a = getattr(track_a, attr, None)
            arr_b = getattr(track_b, attr, None)
            if arr_a is not None and arr_b is not None:
                n = min(len(arr_a), len(arr_b))
                for f in range(frame_idx, n):
                    arr_a[f], arr_b[f] = arr_b[f].copy(), arr_a[f].copy()

        # Swap confidences from frame_idx onward
        if track_a.confidences is not None and track_b.confidences is not None:
            n = min(len(track_a.confidences), len(track_b.confidences))
            for f in range(frame_idx, n):
                track_a.confidences[f], track_b.confidences[f] = (
                    track_b.confidences[f], track_a.confidences[f],
                )

        # Add verified keyframes at the swap point
        for track in (track_a, track_b):
            existing = [kf for kf in track.keyframes if kf["frame"] == frame_idx]
            if existing:
                existing[0]["verified"] = True
            else:
                track.keyframes.append({"frame": frame_idx, "verified": True})

        # Snapshot post-state for redo
        snap_a_new = self._snapshot_track(track_a)
        snap_b_new = self._snapshot_track(track_b)

        def undo(pa=pid_a, pb=pid_b, sa=snap_a, sb=snap_b):
            ta = self._session.person_tracks.get(pa)
            tb = self._session.person_tracks.get(pb)
            if ta:
                self._restore_track(ta, sa)
            if tb:
                self._restore_track(tb, sb)
            self._refresh_for_person()
            self.track_modified.emit()

        def redo(pa=pid_a, pb=pid_b, sa=snap_a_new, sb=snap_b_new):
            ta = self._session.person_tracks.get(pa)
            tb = self._session.person_tracks.get(pb)
            if ta:
                self._restore_track(ta, sa)
                self._session.dirty_persons.add(pa)
            if tb:
                self._restore_track(tb, sb)
                self._session.dirty_persons.add(pb)
            self._refresh_for_person()
            self.track_modified.emit()

        self._session.undo_stack.push(UndoEntry("Swap IDs", undo, redo))

        # Mark dirty
        self._session.dirty_persons.add(self._current_person_id)
        self._session.dirty_persons.add(target_id)
        self.person_dirty.emit(self._current_person_id)
        self.person_dirty.emit(target_id)

        self._refresh_for_person()
        self.track_modified.emit()
        log.info(
            "Swapped IDs %d and %d from frame %d onward",
            self._current_person_id, target_id, frame_idx,
        )

    def _on_split_track(self):
        """Split track at current frame — frames onward become new inactive track.

        Why: When a person exits and re-enters the scene, the tracker may assign
        a single long track spanning both appearances. Splitting isolates the
        segments so they can be independently verified or merged with other tracks.
        """
        track = self._get_current_track()
        if track is None:
            return

        frame_idx = self._current_frame
        num_frames = self._session.num_frames

        if frame_idx <= 0 or frame_idx >= num_frames - 1:
            log.warning(
                "Cannot split at frame %d (must be 1..%d)", frame_idx, num_frames - 2
            )
            return

        pid = self._current_person_id
        snap_before = self._snapshot_track(track)

        # Generate new person ID
        all_ids = set(self._session.person_tracks.keys())
        new_id = max(all_ids) + 1 if all_ids else 0
        new_track = PersonTrack(person_id=new_id)

        # Split bboxes
        if track.bboxes is not None:
            new_bboxes = np.zeros_like(track.bboxes)
            new_bboxes[frame_idx:] = track.bboxes[frame_idx:]
            new_track.bboxes = new_bboxes
            track.bboxes[frame_idx:] = 0

        # Split confidences
        if track.confidences is not None:
            new_confs = [0.0] * len(track.confidences)
            for f in range(frame_idx, len(track.confidences)):
                new_confs[f] = track.confidences[f]
                track.confidences[f] = 0.0
            new_track.confidences = new_confs

        # Split bbox_corrections
        if track.bbox_corrections is not None:
            new_corrections = np.zeros_like(track.bbox_corrections)
            new_corrections[frame_idx:] = track.bbox_corrections[frame_idx:]
            new_track.bbox_corrections = new_corrections
            track.bbox_corrections[frame_idx:] = 0

        # Split keyframes
        track.keyframes, new_track.keyframes = (
            [kf for kf in track.keyframes if kf["frame"] < frame_idx],
            [kf for kf in track.keyframes if kf["frame"] >= frame_idx],
        )

        # Add to session as inactive
        self._session.person_tracks[new_id] = new_track
        self._session.inactive_tracks.add(new_id)

        def undo(p=pid, nid=new_id, snap=snap_before):
            # Remove the split-off track
            self._session.person_tracks.pop(nid, None)
            self._session.inactive_tracks.discard(nid)
            t = self._session.person_tracks.get(p)
            if t:
                self._restore_track(t, snap)
            self._refresh_for_person()
            self.track_modified.emit()

        snap_after = self._snapshot_track(track)
        snap_new = self._snapshot_track(new_track)

        def redo(p=pid, nid=new_id, sa=snap_after, sn=snap_new):
            t = self._session.person_tracks.get(p)
            if t:
                self._restore_track(t, sa)
                self._session.dirty_persons.add(p)
            nt = PersonTrack(person_id=nid)
            self._restore_track(nt, sn)
            self._session.person_tracks[nid] = nt
            self._session.inactive_tracks.add(nid)
            self._refresh_for_person()
            self.track_modified.emit()

        self._session.undo_stack.push(UndoEntry("Split track", undo, redo))

        # Mark dirty
        self._session.dirty_persons.add(self._current_person_id)
        self.person_dirty.emit(self._current_person_id)

        self._refresh_for_person()
        self.track_modified.emit()
        log.info(
            "Split person %d at frame %d -> new inactive track %d",
            self._current_person_id, frame_idx, new_id,
        )

    def _on_merge_track(self):
        """Merge selected inactive track into current active person.

        Why: After splitting or when the tracker creates fragmented tracks,
        merging recombines identity segments that belong to the same person.
        Non-zero frames from the source overwrite the target.
        """
        if self._current_person_id < 0:
            return
        source_id = self._merge_combo.currentData()
        if source_id is None:
            return

        source = self._session.person_tracks.get(source_id)
        target = self._get_current_track()
        if source is None or target is None:
            return

        pid = self._current_person_id
        snap_target = self._snapshot_track(target)
        snap_source = self._snapshot_track(source)
        src_was_inactive = source_id in self._session.inactive_tracks

        # Merge bboxes: non-zero source frames overwrite target
        if source.bboxes is not None:
            if target.bboxes is None:
                target.bboxes = source.bboxes.copy()
            else:
                n = min(len(source.bboxes), len(target.bboxes))
                for f in range(n):
                    if not np.all(source.bboxes[f] == 0):
                        target.bboxes[f] = source.bboxes[f].copy()

        # Merge confidences: non-zero source overwrites target
        if source.confidences is not None:
            if target.confidences is None:
                target.confidences = list(source.confidences)
            else:
                n = min(len(source.confidences), len(target.confidences))
                for f in range(n):
                    if source.confidences[f] > 0:
                        target.confidences[f] = source.confidences[f]

        # Merge bbox_corrections
        if source.bbox_corrections is not None:
            if target.bbox_corrections is None:
                target.bbox_corrections = source.bbox_corrections.copy()
            else:
                n = min(len(source.bbox_corrections), len(target.bbox_corrections))
                for f in range(n):
                    if not np.all(source.bbox_corrections[f] == 0):
                        target.bbox_corrections[f] = source.bbox_corrections[f].copy()

        # Merge keyframes
        existing_frames = {kf["frame"] for kf in target.keyframes}
        for kf in source.keyframes:
            if kf["frame"] not in existing_frames:
                target.keyframes.append(kf)

        # Remove source from session
        del self._session.person_tracks[source_id]
        self._session.inactive_tracks.discard(source_id)

        snap_target_after = self._snapshot_track(target)

        def undo(p=pid, sid=source_id, st=snap_target, ss=snap_source, inactive=src_was_inactive):
            t = self._session.person_tracks.get(p)
            if t:
                self._restore_track(t, st)
            # Restore the removed source track
            restored = PersonTrack(person_id=sid)
            self._restore_track(restored, ss)
            self._session.person_tracks[sid] = restored
            if inactive:
                self._session.inactive_tracks.add(sid)
            self._refresh_for_person()
            self.track_modified.emit()

        def redo(p=pid, sid=source_id, sta=snap_target_after):
            t = self._session.person_tracks.get(p)
            if t:
                self._restore_track(t, sta)
                self._session.dirty_persons.add(p)
            self._session.person_tracks.pop(sid, None)
            self._session.inactive_tracks.discard(sid)
            self._refresh_for_person()
            self.track_modified.emit()

        self._session.undo_stack.push(UndoEntry("Merge track", undo, redo))

        # Mark dirty
        self._session.dirty_persons.add(self._current_person_id)
        self.person_dirty.emit(self._current_person_id)

        self._refresh_for_person()
        self.track_modified.emit()
        log.info("Merged track %d into person %d", source_id, self._current_person_id)

    def _on_crossing_threshold_changed(self, val: float):
        self._session.crossing_threshold = val

    def _on_crossing_start(self):
        """Mark start of a crossing/occlusion span at the current frame."""
        if self._current_person_id < 0:
            return
        self._crossing_start_frame = self._current_frame
        self._crossing_status.setText(
            f"Start marked at frame {self._current_frame}. Click 'Mark End'."
        )
        self._crossing_status.setStyleSheet(f"font-style: italic; color: {COLORS['warning']};")
        self._crossing_start_btn.setEnabled(False)
        self._crossing_end_btn.setEnabled(True)

    def _on_crossing_end(self):
        """Complete crossing span and store in session."""
        if self._current_person_id < 0 or self._crossing_start_frame is None:
            return

        start = self._crossing_start_frame
        end = self._current_frame

        if end <= start:
            self._crossing_status.setText(
                f"End frame ({end}) must be after start ({start})"
            )
            self._crossing_status.setStyleSheet(f"font-style: italic; color: {COLORS['error']};")
            return

        pid = self._current_person_id
        if pid not in self._session.crossing_spans:
            self._session.crossing_spans[pid] = []
        self._session.crossing_spans[pid].append((start, end))
        self._session.crossing_spans[pid].sort()

        def undo(p=pid, s=start, e=end):
            if p in self._session.crossing_spans:
                self._session.crossing_spans[p] = [
                    (a, b) for a, b in self._session.crossing_spans[p]
                    if not (a == s and b == e)
                ]
                if not self._session.crossing_spans[p]:
                    del self._session.crossing_spans[p]
            self._update_crossing_table()

        def redo(p=pid, s=start, e=end):
            if p not in self._session.crossing_spans:
                self._session.crossing_spans[p] = []
            self._session.crossing_spans[p].append((s, e))
            self._session.crossing_spans[p].sort()
            self._update_crossing_table()

        self._session.undo_stack.push(UndoEntry("Add crossing span", undo, redo))

        # Reset state
        self._crossing_start_frame = None
        self._crossing_status.setText(f"Span added: frames {start}\u2013{end}")
        self._crossing_status.setStyleSheet(f"font-style: italic; color: {COLORS['success']};")
        self._crossing_start_btn.setEnabled(True)
        self._crossing_end_btn.setEnabled(False)

        self._update_crossing_table()
        log.info("Crossing span for person %d: frames %d-%d", pid, start, end)

    def _on_crossing_delete(self, person_id: int, start: int, end: int):
        """Remove a crossing span from session and refresh table."""
        def undo(p=person_id, s=start, e=end):
            if p not in self._session.crossing_spans:
                self._session.crossing_spans[p] = []
            self._session.crossing_spans[p].append((s, e))
            self._session.crossing_spans[p].sort()
            self._update_crossing_table()

        def redo(p=person_id, s=start, e=end):
            if p in self._session.crossing_spans:
                self._session.crossing_spans[p] = [
                    (a, b) for a, b in self._session.crossing_spans[p]
                    if not (a == s and b == e)
                ]
                if not self._session.crossing_spans[p]:
                    del self._session.crossing_spans[p]
            self._update_crossing_table()

        self._session.undo_stack.push(UndoEntry("Delete crossing span", undo, redo))

        if person_id in self._session.crossing_spans:
            self._session.crossing_spans[person_id] = [
                (s, e)
                for s, e in self._session.crossing_spans[person_id]
                if not (s == start and e == end)
            ]
            if not self._session.crossing_spans[person_id]:
                del self._session.crossing_spans[person_id]
        self._update_crossing_table()

    # ------------------------------------------------------------------
    # Review scanner
    # ------------------------------------------------------------------

    def _on_scan_issues(self):
        """Run the review scanner across all active tracks.

        Why: Automated scanning catches common tracking problems (confidence
        drops, identity swaps, shape changes, detection gaps) that would take
        a human minutes to find manually. Issues are sorted by frame so the
        user can step through them sequentially.
        """
        self._review_issues = compute_review_issues(self._session)
        self._review_issue_idx = 0

        n = len(self._review_issues)
        self._update_issues_table()

        if n == 0:
            self._issues_label.setText("No issues found.")
            self._issues_label.setStyleSheet(f"font-style: italic; color: {COLORS['success']};")
        else:
            self._issues_label.setText(
                f"Found {n} issue{'s' if n != 1 else ''}. Use Next/Prev to navigate."
            )
            self._issues_label.setStyleSheet(f"font-style: italic; color: {COLORS['warning']};")
            # Navigate to first issue
            self._navigate_to_issue(0)

        has_issues = n > 0
        self._prev_issue_btn.setEnabled(has_issues)
        self._next_issue_btn.setEnabled(has_issues)

        log.info("Review scan complete: %d issues found", n)

    def _on_next_issue(self):
        """Navigate to the next review issue (wraps around)."""
        if not self._review_issues:
            return
        self._review_issue_idx = (
            (self._review_issue_idx + 1) % len(self._review_issues)
        )
        self._navigate_to_issue(self._review_issue_idx)

    def _on_prev_issue(self):
        """Navigate to the previous review issue (wraps around)."""
        if not self._review_issues:
            return
        self._review_issue_idx = (
            (self._review_issue_idx - 1) % len(self._review_issues)
        )
        self._navigate_to_issue(self._review_issue_idx)

    def _navigate_to_issue(self, idx: int):
        """Seek to an issue's frame and update the status label."""
        issue = self._review_issues[idx]
        n = len(self._review_issues)
        self._issues_label.setText(
            f"[{idx + 1}/{n}] ID {issue.person_id}: {issue.description}"
        )
        # Select the issue's person if different
        if issue.person_id != self._current_person_id:
            self.set_person(issue.person_id)
            self.person_changed.emit(issue.person_id)
        # Seek to the issue's frame
        self.frame_requested.emit(issue.frame)
        # Highlight row in issues table
        self._issues_table.selectRow(idx)

    def _on_issue_double_clicked(self, row: int, col: int):
        """Navigate to an issue when double-clicking its table row."""
        if 0 <= row < len(self._review_issues):
            self._review_issue_idx = row
            self._navigate_to_issue(row)

    def _update_issues_table(self):
        """Rebuild the issues table from current scan results."""
        self._issues_table.setRowCount(0)
        self._issues_table.setRowCount(len(self._review_issues))

        for row, issue in enumerate(self._review_issues):
            frame_item = QTableWidgetItem(str(issue.frame))
            frame_item.setTextAlignment(Qt.AlignCenter)
            self._issues_table.setItem(row, 0, frame_item)

            person_item = QTableWidgetItem(f"Person {issue.person_id}")
            person_item.setTextAlignment(Qt.AlignCenter)
            self._issues_table.setItem(row, 1, person_item)

            type_item = QTableWidgetItem(issue.issue_type)
            type_item.setTextAlignment(Qt.AlignCenter)
            self._issues_table.setItem(row, 2, type_item)

            desc_item = QTableWidgetItem(issue.description)
            self._issues_table.setItem(row, 3, desc_item)

    def go_to_next_unreviewed(self):
        """Public API: navigate to next unreviewed keyframe (keyboard shortcut).

        Falls back to next review issue if no unreviewed keyframes exist.
        """
        # Try review issues first
        if self._review_issues:
            self._on_next_issue()
            return
        # Otherwise run a scan and go to the first result
        self._on_scan_issues()
        if self._review_issues:
            self._on_next_issue()

    # ------------------------------------------------------------------
    # Reprocess
    # ------------------------------------------------------------------

    def _on_reprocess(self):
        """Emit signal to reprocess all dirty persons.

        Why: After bbox corrections, the GVHMR pipeline must re-run for
        affected persons to produce updated pose parameters. This button
        triggers ReprocessWorker for every person in session.dirty_persons.
        """
        dirty = sorted(self._session.dirty_persons)
        if not dirty:
            return
        log.info("Reprocess requested for %d dirty persons: %s", len(dirty), dirty)
        self.reprocess_requested.emit(dirty)

    def update_reprocess_button(self):
        """Update reprocess button label and enabled state from session.dirty_persons."""
        n = len(self._session.dirty_persons)
        self._reprocess_btn.setText(f"Reprocess ({n} dirty)")
        self._reprocess_btn.setEnabled(n > 0)

    # ------------------------------------------------------------------
    # Track operations helpers
    # ------------------------------------------------------------------

    def _update_swap_combo(self):
        """Rebuild swap target combo — all active persons except current."""
        self._swap_combo.blockSignals(True)
        self._swap_combo.clear()
        for pid in sorted(self._session.person_tracks.keys()):
            if pid != self._current_person_id and pid not in self._session.inactive_tracks:
                self._swap_combo.addItem(f"Person {pid}", pid)
        self._swap_combo.blockSignals(False)

    def _update_merge_combo(self):
        """Rebuild merge source combo — all inactive tracks with frame info."""
        self._merge_combo.blockSignals(True)
        self._merge_combo.clear()
        for pid in sorted(self._session.inactive_tracks):
            track = self._session.person_tracks.get(pid)
            if track is None:
                continue
            label = f"Track {pid}"
            if track.bboxes is not None:
                nonzero = np.any(track.bboxes != 0, axis=1)
                count = int(np.sum(nonzero))
                frames = np.where(nonzero)[0]
                if len(frames) > 0:
                    label += f" (frames {frames[0]}-{frames[-1]}, {count} dets)"
            self._merge_combo.addItem(label, pid)
        self._merge_combo.blockSignals(False)

    def _update_crossing_table(self):
        """Rebuild crossing spans table from session data."""
        self._crossing_table.setRowCount(0)

        rows: list[tuple[int, int, int]] = []
        for pid, spans in sorted(self._session.crossing_spans.items()):
            for start, end in sorted(spans):
                rows.append((pid, start, end))

        self._crossing_table.setRowCount(len(rows))
        for row_idx, (pid, start, end) in enumerate(rows):
            pid_item = QTableWidgetItem(f"Person {pid}")
            pid_item.setTextAlignment(Qt.AlignCenter)
            self._crossing_table.setItem(row_idx, 0, pid_item)

            start_item = QTableWidgetItem(str(start))
            start_item.setTextAlignment(Qt.AlignCenter)
            self._crossing_table.setItem(row_idx, 1, start_item)

            end_item = QTableWidgetItem(str(end))
            end_item.setTextAlignment(Qt.AlignCenter)
            self._crossing_table.setItem(row_idx, 2, end_item)

            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 0, 2, 0)
            del_btn = QPushButton("Del")
            del_btn.setFixedSize(32, 22)
            del_btn.clicked.connect(
                lambda checked, p=pid, s=start, e=end: self._on_crossing_delete(p, s, e)
            )
            actions_layout.addWidget(del_btn)
            self._crossing_table.setCellWidget(row_idx, 3, actions_widget)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def interpolate_bbox_corrections(
    original: np.ndarray,
    corrections: np.ndarray,
    keyframe_frames: list[int] | None = None,
) -> np.ndarray:
    """Linear interpolation of absolute bbox values between keyframes.

    Keyframe bboxes are treated as 100% confidence control points — the
    result passes through each one exactly.  Between keyframes the bbox
    is linearly interpolated.  Before the first / after the last keyframe
    the nearest keyframe value is held constant.

    Args:
        original: (N, 4) original bboxes (unused when keyframe_frames given,
            kept for backwards-compat fallback).
        corrections: (N, 4) correction array — values at keyframe frames are
            the user's absolute bbox edits.
        keyframe_frames: explicit list of frame indices that are user keyframes.
            When None, falls back to non-zero detection.

    Returns:
        (N, 4) interpolated corrections array.
    """
    n = min(len(original), len(corrections))

    if keyframe_frames is not None:
        sorted_frames = sorted(
            f for f in keyframe_frames
            if 0 <= f < n and not np.all(corrections[f] == 0)
        )
    else:
        sorted_frames = [
            f for f in range(n)
            if not np.all(corrections[f] == 0)
        ]

    if len(sorted_frames) < 1:
        return corrections.copy()

    result = np.zeros((len(corrections), 4), dtype=corrections.dtype)

    if len(sorted_frames) == 1:
        # Single keyframe: hold its value for every frame
        val = corrections[sorted_frames[0]].copy()
        for f in range(len(result)):
            result[f] = val
        return result

    first_f = sorted_frames[0]
    last_f = sorted_frames[-1]

    # Before first keyframe: hold first keyframe value
    val_first = corrections[first_f].copy()
    for f in range(0, first_f):
        result[f] = val_first

    # Between keyframes: linear interpolation of absolute bbox
    for i in range(len(sorted_frames) - 1):
        f_a = sorted_frames[i]
        f_b = sorted_frames[i + 1]
        val_a = corrections[f_a]
        val_b = corrections[f_b]
        span = f_b - f_a

        result[f_a] = val_a
        for f in range(f_a + 1, f_b):
            t = (f - f_a) / span
            result[f] = (1 - t) * val_a + t * val_b

    # Last keyframe itself
    result[last_f] = corrections[last_f]

    # After last keyframe: hold last keyframe value
    val_last = corrections[last_f].copy()
    for f in range(last_f + 1, len(result)):
        result[f] = val_last

    return result
