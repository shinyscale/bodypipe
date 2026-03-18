"""Per-person confidence timeline overview widget.

Why extracted: _TrackOverview was defined inside multi_person_tab.py but is
used by AppWindow and TrackOverviewDock in the dock-based layout.  Moving it
to its own module eliminates the dependency on the now-deleted tab class and
makes the widget independently testable.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Signal

from views.confidence_timeline import ConfidenceTimeline


# Colors for up to 8 tracked persons — consistent palette across the app
PERSON_COLORS = [
    "#e94560", "#4ecca3", "#ffd93d", "#6c5ce7",
    "#00b894", "#fd79a8", "#0984e3", "#fdcb6e",
]


class TrackOverview(QWidget):
    """Per-person confidence bars showing track quality over time."""

    person_clicked = Signal(int, int)  # (person_id, frame_index)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)
        self._timelines: dict[int, ConfidenceTimeline] = {}
        self._labels: dict[int, QLabel] = {}

    def set_tracks(self, tracks: dict[int, np.ndarray]):
        """Set confidence arrays per person_id. tracks = {pid: conf_array}."""
        self._clear()
        for pid, conf in sorted(tracks.items()):
            label = QLabel(f"Person {pid}")
            color = PERSON_COLORS[pid % len(PERSON_COLORS)]
            label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px;")
            self._labels[pid] = label
            self._layout.addWidget(label)

            timeline = ConfidenceTimeline()
            timeline.set_data(conf)
            timeline.frame_clicked.connect(lambda f, p=pid: self.person_clicked.emit(p, f))
            self._timelines[pid] = timeline
            self._layout.addWidget(timeline)

    def set_current_frame(self, frame: int):
        for tl in self._timelines.values():
            tl.set_current_frame(frame)

    def _clear(self):
        for w in list(self._timelines.values()) + list(self._labels.values()):
            self._layout.removeWidget(w)
            w.deleteLater()
        self._timelines.clear()
        self._labels.clear()
