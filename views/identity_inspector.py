"""Identity Inspector panel — person selection, confidence visualization,
keyframe management, bbox editing, and identity verification.

Why: The identity inspector is the primary tool for verifying and correcting
multi-person tracking results. It provides per-person confidence visualization,
keyframe-based annotation, two-click bbox editing with interpolation, and CRUD
operations for managing tracked identities. All state flows through
Session.person_tracks — no module-level dicts.
"""

from __future__ import annotations

import json
import logging
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
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor

from models.session import Session, PersonTrack
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

# Colors for per-person identification (shared with multi_person_tab)
PERSON_COLORS = [
    "#e94560", "#4ecca3", "#ffd93d", "#6c5ce7",
    "#00b894", "#fd79a8", "#0984e3", "#fdcb6e",
]


def _confidence_color(value: float) -> str:
    """Return CSS color string based on confidence value."""
    if value > 0.8:
        return "#4ecca3"  # green
    elif value > 0.5:
        return "#ffd93d"  # yellow
    else:
        return "#ff6b6b"  # red


class IdentityInspector(QWidget):
    """Interactive per-person identity verification panel.

    Provides:
    - Person selector (QComboBox) to switch between tracked persons
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

        # ---- Person Selector ----
        person_group = QGroupBox("Person")
        person_layout = QVBoxLayout(person_group)

        selector_row = QHBoxLayout()
        self._person_combo = QComboBox()
        self._person_combo.setMinimumWidth(120)
        selector_row.addWidget(self._person_combo, 1)

        self._show_all_tracks = QCheckBox("Show all tracks")
        selector_row.addWidget(self._show_all_tracks)

        person_layout.addLayout(selector_row)
        layout.addWidget(person_group)

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

        layout.addStretch()

    def _connect_signals(self):
        self._person_combo.currentIndexChanged.connect(self._on_person_combo_changed)
        self._timeline.frame_clicked.connect(self._on_timeline_clicked)
        self._verify_btn.clicked.connect(self._on_verify)
        self._add_kf_btn.clicked.connect(self._on_add_keyframe)
        self._remove_kf_btn.clicked.connect(self._on_remove_keyframe)
        self._prev_kf_btn.clicked.connect(self._on_prev_keyframe)
        self._next_kf_btn.clicked.connect(self._on_next_keyframe)
        self._show_all_tracks.toggled.connect(self._on_show_all_toggled)
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

        # Update combo box without re-triggering signal
        idx = self._person_combo.findData(person_id)
        if idx >= 0:
            self._person_combo.blockSignals(True)
            self._person_combo.setCurrentIndex(idx)
            self._person_combo.blockSignals(False)

        self._refresh_for_person()

    def set_frame(self, frame_idx: int):
        """Update displays for a new frame position."""
        self._current_frame = frame_idx
        self._timeline.set_current_frame(frame_idx)
        self._update_confidence_breakdown(frame_idx)

    def refresh(self):
        """Rebuild all UI from session state (e.g., after pipeline finishes)."""
        self._update_person_selector()
        self._update_merge_combo()
        if self._current_person_id >= 0:
            self._refresh_for_person()

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

            # Store correction
            self._store_bbox_correction(track, self._current_frame, bbox)

            # Add keyframe at this frame if not present
            if not any(kf["frame"] == self._current_frame for kf in track.keyframes):
                track.keyframes.append({
                    "frame": self._current_frame,
                    "verified": False,
                })

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

    # ------------------------------------------------------------------
    # Internal updates
    # ------------------------------------------------------------------

    def _update_person_selector(self):
        """Rebuild person combo box from session.person_tracks."""
        self._person_combo.blockSignals(True)
        self._person_combo.clear()

        for pid in sorted(self._session.person_tracks.keys()):
            if pid not in self._session.inactive_tracks:
                self._person_combo.addItem(f"Person {pid}", pid)

        # Re-select current person if still valid
        if self._current_person_id >= 0:
            idx = self._person_combo.findData(self._current_person_id)
            if idx >= 0:
                self._person_combo.setCurrentIndex(idx)
            elif self._person_combo.count() > 0:
                self._person_combo.setCurrentIndex(0)
                self._current_person_id = self._person_combo.currentData() or -1
        elif self._person_combo.count() > 0:
            self._person_combo.setCurrentIndex(0)
            self._current_person_id = self._person_combo.currentData() or -1

        self._person_combo.blockSignals(False)

    def _refresh_for_person(self):
        """Update all displays for the current person."""
        self._update_timeline()
        self._update_confidence_breakdown(self._current_frame)
        self._update_keyframe_table()
        self._update_swap_combo()
        self._update_merge_combo()
        self._update_crossing_table()

    def _update_timeline(self):
        """Update the confidence timeline for the current person."""
        track = self._get_current_track()
        if track is None:
            self._timeline.set_data(np.zeros(0))
            return

        confidences = self._get_confidences(track)
        keyframe_frames = [kf["frame"] for kf in track.keyframes]
        verified_frames = {
            kf["frame"] for kf in track.keyframes if kf.get("verified", False)
        }

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
                verified_item.setForeground(QColor("#4ecca3"))
            else:
                verified_item.setForeground(QColor("#ff6b6b"))
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

    def _on_person_combo_changed(self, index: int):
        """Handle person combo box selection change."""
        if index < 0:
            return
        person_id = self._person_combo.currentData()
        if person_id is None:
            return
        self._current_person_id = person_id
        self._session.selected_person = person_id
        self._refresh_for_person()
        self.person_changed.emit(person_id)

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

        track.keyframes.append({
            "frame": self._current_frame,
            "verified": False,
        })

        self._update_timeline()
        self._update_keyframe_table()
        self.keyframe_changed.emit(self._current_person_id, self._current_frame)

    def _on_remove_keyframe(self):
        """Remove keyframe at the current frame."""
        track = self._get_current_track()
        if track is None:
            return

        track.keyframes = [
            kf for kf in track.keyframes if kf["frame"] != self._current_frame
        ]

        self._update_timeline()
        self._update_keyframe_table()
        self.keyframe_changed.emit(self._current_person_id, self._current_frame)

    def _on_prev_keyframe(self):
        """Navigate to the previous keyframe before current frame."""
        track = self._get_current_track()
        if track is None:
            return

        frames = sorted(kf["frame"] for kf in track.keyframes)
        prev_frames = [f for f in frames if f < self._current_frame]
        if prev_frames:
            self.frame_requested.emit(prev_frames[-1])

    def _on_next_keyframe(self):
        """Navigate to the next keyframe after current frame."""
        track = self._get_current_track()
        if track is None:
            return

        frames = sorted(kf["frame"] for kf in track.keyframes)
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

        track.keyframes = [kf for kf in track.keyframes if kf["frame"] != frame]

        self._update_timeline()
        self._update_keyframe_table()
        self.keyframe_changed.emit(self._current_person_id, frame)

    def _on_show_all_toggled(self, checked: bool):
        """Handle show all tracks checkbox toggle."""
        self.bbox_overlay_changed.emit({"show_all": checked})

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
            self._bbox_status.setStyleSheet("font-style: italic; color: #ffd93d;")
        elif self._bbox_edit_state == "click2":
            self._bbox_status.setText("Click bottom-right corner on video frame")
            self._bbox_status.setStyleSheet("font-style: italic; color: #ffd93d;")
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

        result = interpolate_bbox_corrections(original, track.bbox_corrections)
        track.bbox_corrections = result

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

        frame_idx = self._current_frame

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

        # Mark dirty
        self._session.dirty_persons.add(self._current_person_id)
        self.person_dirty.emit(self._current_person_id)

        self._refresh_for_person()
        self.track_modified.emit()
        log.info("Merged track %d into person %d", source_id, self._current_person_id)

    def _on_crossing_start(self):
        """Mark start of a crossing/occlusion span at the current frame."""
        if self._current_person_id < 0:
            return
        self._crossing_start_frame = self._current_frame
        self._crossing_status.setText(
            f"Start marked at frame {self._current_frame}. Click 'Mark End'."
        )
        self._crossing_status.setStyleSheet("font-style: italic; color: #ffd93d;")
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
            self._crossing_status.setStyleSheet("font-style: italic; color: #ff6b6b;")
            return

        pid = self._current_person_id
        if pid not in self._session.crossing_spans:
            self._session.crossing_spans[pid] = []
        self._session.crossing_spans[pid].append((start, end))
        self._session.crossing_spans[pid].sort()

        # Reset state
        self._crossing_start_frame = None
        self._crossing_status.setText(f"Span added: frames {start}\u2013{end}")
        self._crossing_status.setStyleSheet("font-style: italic; color: #4ecca3;")
        self._crossing_start_btn.setEnabled(True)
        self._crossing_end_btn.setEnabled(False)

        self._update_crossing_table()
        log.info("Crossing span for person %d: frames %d-%d", pid, start, end)

    def _on_crossing_delete(self, person_id: int, start: int, end: int):
        """Remove a crossing span from session and refresh table."""
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
    original: np.ndarray, corrections: np.ndarray
) -> np.ndarray:
    """Linear interpolation of bbox corrections in delta space.

    Why delta space: interpolating the difference between corrected and original
    bboxes produces smoother results than interpolating absolute coordinates.
    A 5px nudge at frame 10 and 10px at frame 20 blends naturally between them.

    Args:
        original: (N, 4) original bboxes.
        corrections: (N, 4) correction array — non-zero entries are user edits.

    Returns:
        (N, 4) interpolated corrections array.
    """
    n = len(original)
    result = corrections.copy()

    # Find frames with non-zero corrections
    corrected_frames = [f for f in range(min(n, len(corrections)))
                        if not np.all(corrections[f] == 0)]

    if len(corrected_frames) < 2:
        return result  # Need at least 2 points to interpolate

    # Compute deltas at corrected frames
    deltas = {}
    for f in corrected_frames:
        deltas[f] = corrections[f] - original[f]

    sorted_frames = sorted(corrected_frames)
    first_f = sorted_frames[0]
    last_f = sorted_frames[-1]

    # Before first correction: constant extrapolation
    first_delta = deltas[first_f]
    for f in range(0, first_f):
        if f < n:
            result[f] = original[f] + first_delta

    # Between corrections: linear interpolation of delta
    for i in range(len(sorted_frames) - 1):
        f_a = sorted_frames[i]
        f_b = sorted_frames[i + 1]
        delta_a = deltas[f_a]
        delta_b = deltas[f_b]
        span = f_b - f_a

        for f in range(f_a + 1, f_b):
            t = (f - f_a) / span
            result[f] = original[f] + (1 - t) * delta_a + t * delta_b

    # After last correction: constant extrapolation
    last_delta = deltas[last_f]
    for f in range(last_f + 1, min(n, len(result))):
        result[f] = original[f] + last_delta

    return result
