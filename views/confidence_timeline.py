"""QPainter-based confidence timeline widget."""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal, Qt, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QMouseEvent


class ConfidenceTimeline(QWidget):
    """Interactive confidence timeline with keyframe markers."""

    frame_clicked = Signal(int)  # clicked frame index

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        self.setMaximumHeight(80)
        self._confidences: np.ndarray | None = None  # shape (N,)
        self._keyframe_frames: list[int] = []
        self._verified_frames: set[int] = set()
        self._current_frame: int = 0
        self._num_frames: int = 0

    def set_data(
        self,
        confidences: np.ndarray,
        keyframe_frames: list[int] | None = None,
        verified_frames: set[int] | None = None,
    ):
        self._confidences = confidences
        self._num_frames = len(confidences)
        self._keyframe_frames = keyframe_frames or []
        self._verified_frames = verified_frames or set()
        self.update()

    def set_current_frame(self, frame_idx: int):
        self._current_frame = frame_idx
        self.update()

    def paintEvent(self, event):
        if self._confidences is None or self._num_frames == 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        margin_bottom = 12  # space for keyframe markers

        # Draw confidence bars
        bar_h = h - margin_bottom
        bin_width = max(1.0, w / self._num_frames)

        for i in range(self._num_frames):
            x = i * w / self._num_frames
            conf = float(self._confidences[i])
            bar_height = conf * bar_h

            # Color: green > 0.8, yellow 0.5-0.8, red < 0.5
            if conf > 0.8:
                color = QColor("#4ecca3")
            elif conf > 0.5:
                color = QColor("#ffd93d")
            else:
                color = QColor("#ff6b6b")

            painter.fillRect(QRectF(x, bar_h - bar_height, bin_width + 0.5, bar_height), color)

        # Draw keyframe markers
        for frame in self._keyframe_frames:
            x = frame * w / self._num_frames
            is_verified = frame in self._verified_frames
            color = QColor("#4ecca3") if is_verified else QColor("#ffd93d")
            pen = QPen(color, 2)
            painter.setPen(pen)
            # Small triangle at bottom
            y = h - margin_bottom
            painter.drawLine(int(x), int(y), int(x), int(y + margin_bottom))
            painter.drawLine(int(x) - 3, int(y + margin_bottom), int(x) + 3, int(y + margin_bottom))
            painter.drawLine(int(x) - 3, int(y + margin_bottom), int(x), int(y))
            painter.drawLine(int(x) + 3, int(y + margin_bottom), int(x), int(y))

        # Draw current frame indicator
        if self._num_frames > 0:
            x = self._current_frame * w / self._num_frames
            pen = QPen(QColor("#e94560"), 2)
            painter.setPen(pen)
            painter.drawLine(int(x), 0, int(x), h)

        painter.end()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._num_frames > 0:
            frame = int(event.position().x() * self._num_frames / self.width())
            frame = max(0, min(frame, self._num_frames - 1))
            self.frame_clicked.emit(frame)
        super().mousePressEvent(event)
