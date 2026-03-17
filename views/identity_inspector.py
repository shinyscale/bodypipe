"""Identity Inspector panel — person selection, confidence visualization,
keyframe management, and identity verification.

Why: The identity inspector is the primary tool for verifying and correcting
multi-person tracking results. It provides per-person confidence visualization,
keyframe-based annotation, and CRUD operations for managing tracked identities.
All state flows through Session.person_tracks — no module-level dicts.
"""

from __future__ import annotations

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

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self._session = session
        self._current_person_id: int = -1
        self._current_frame: int = 0

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_person(self, person_id: int):
        """Select a person by ID and update all displays."""
        if person_id == self._current_person_id:
            return
        self._current_person_id = person_id

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
        if self._current_person_id >= 0:
            self._refresh_for_person()

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
