"""Video player widget with frame display, seeking, playback, and LRU cache."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QSlider,
    QToolButton,
    QHBoxLayout,
    QVBoxLayout,
    QMenu,
    QFileDialog,
    QApplication,
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QMouseEvent


class FrameCache:
    """LRU cache with read-ahead for sequential video access."""

    READAHEAD = 15
    MAX_SIZE = 200

    def __init__(self, maxsize: int = MAX_SIZE):
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._maxsize = maxsize

    def get(self, idx: int) -> np.ndarray | None:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        return None

    def put(self, idx: int, frame: np.ndarray):
        if idx in self._cache:
            self._cache.move_to_end(idx)
        else:
            self._cache[idx] = frame
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

    def get_frame(self, video_path: Path, frame_idx: int) -> np.ndarray | None:
        """Get frame with read-ahead caching on miss."""
        cached = self.get(frame_idx)
        if cached is not None:
            return cached

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        for i in range(self.READAHEAD + 1):
            ret, frame = cap.read()
            if not ret:
                break
            self.put(frame_idx + i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return self.get(frame_idx)


class FrameDisplay(QLabel):
    """QLabel that displays video frames and emits click positions."""

    clicked = Signal(float, float)  # normalized (x, y) in image space

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #0a0a1a;")
        self._pixmap_size = None
        self._current_frame: np.ndarray | None = None
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def set_frame(self, frame: np.ndarray):
        """Display an RGB numpy array."""
        self._current_frame = frame.copy()
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._pixmap_size = (scaled.width(), scaled.height())
        self.setPixmap(scaled)

    def _frame_to_pixmap(self) -> QPixmap | None:
        """Convert stored frame to full-resolution QPixmap."""
        if self._current_frame is None:
            return None
        h, w, ch = self._current_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(
            self._current_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        return QPixmap.fromImage(qimg)

    def _show_context_menu(self, pos):
        if self._current_frame is None:
            return
        menu = QMenu(self)
        copy_action = menu.addAction("Copy Frame")
        save_action = menu.addAction("Save Frame As...")
        action = menu.exec(self.mapToGlobal(pos))
        if action == copy_action:
            self._copy_frame()
        elif action == save_action:
            self._save_frame()

    def _copy_frame(self):
        pixmap = self._frame_to_pixmap()
        if pixmap:
            QApplication.clipboard().setPixmap(pixmap)

    def _save_frame(self):
        if self._current_frame is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Frame As",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
        )
        if path:
            pixmap = self._frame_to_pixmap()
            if pixmap:
                pixmap.save(path)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._pixmap_size:
            pw, ph = self._pixmap_size
            # Compute offset (pixmap is centered in label)
            ox = (self.width() - pw) / 2
            oy = (self.height() - ph) / 2
            x = (event.position().x() - ox) / pw
            y = (event.position().y() - oy) / ph
            if 0 <= x <= 1 and 0 <= y <= 1:
                self.clicked.emit(x, y)
        super().mousePressEvent(event)


class VideoPlayer(QWidget):
    """Video frame display with transport controls, slider, and playback."""

    frame_changed = Signal(int)
    frame_clicked = Signal(float, float)
    playback_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path: Path | None = None
        self._num_frames = 0
        self._fps = 30.0
        self._current_frame = 0
        self._playing = False
        self._cache = FrameCache()

        self._setup_ui()
        self._connect_signals()

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Frame display
        self._display = FrameDisplay()
        layout.addWidget(self._display, 1)

        # Transport controls
        transport = QHBoxLayout()

        self._btn_first = QToolButton()
        self._btn_first.setText("|<")
        self._btn_first.setToolTip("First frame (Home)")

        self._btn_back10 = QToolButton()
        self._btn_back10.setText("<<")
        self._btn_back10.setToolTip("Back 10 frames (Ctrl+Left)")

        self._btn_back1 = QToolButton()
        self._btn_back1.setText("<")
        self._btn_back1.setToolTip("Back 1 frame (Left)")

        self._btn_play = QToolButton()
        self._btn_play.setText("\u25b6")
        self._btn_play.setToolTip("Play/Pause (Space)")

        self._btn_fwd1 = QToolButton()
        self._btn_fwd1.setText(">")
        self._btn_fwd1.setToolTip("Forward 1 frame (Right)")

        self._btn_fwd10 = QToolButton()
        self._btn_fwd10.setText(">>")
        self._btn_fwd10.setToolTip("Forward 10 frames (Ctrl+Right)")

        self._btn_last = QToolButton()
        self._btn_last.setText(">|")
        self._btn_last.setToolTip("Last frame (End)")

        for btn in [
            self._btn_first, self._btn_back10, self._btn_back1,
            self._btn_play,
            self._btn_fwd1, self._btn_fwd10, self._btn_last,
        ]:
            btn.setFixedSize(32, 28)
            transport.addWidget(btn)

        transport.addStretch()
        layout.addLayout(transport)

        # Slider
        slider_row = QHBoxLayout()
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)
        slider_row.addWidget(self._slider, 1)

        self._frame_label = QLabel("0 / 0")
        self._frame_label.setMinimumWidth(100)
        slider_row.addWidget(self._frame_label)

        self._fps_label = QLabel("")
        self._fps_label.setMinimumWidth(70)
        slider_row.addWidget(self._fps_label)

        layout.addLayout(slider_row)

    def _connect_signals(self):
        self._display.clicked.connect(self.frame_clicked.emit)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._btn_first.clicked.connect(lambda: self.seek(0))
        self._btn_last.clicked.connect(lambda: self.seek(self._num_frames - 1))
        self._btn_back1.clicked.connect(lambda: self.seek(self._current_frame - 1))
        self._btn_fwd1.clicked.connect(lambda: self.seek(self._current_frame + 1))
        self._btn_back10.clicked.connect(lambda: self.seek(self._current_frame - 10))
        self._btn_fwd10.clicked.connect(lambda: self.seek(self._current_frame + 10))
        self._btn_play.clicked.connect(self._toggle_play)

    def set_video(self, path: Path, num_frames: int, fps: float):
        """Load a new video source."""
        self._video_path = path
        self._num_frames = num_frames
        self._fps = fps
        self._current_frame = 0
        self._cache.clear()
        self._slider.setRange(0, max(0, num_frames - 1))
        self._slider.setValue(0)
        self._update_label()
        self._fps_label.setText(f"FPS: {fps:.1f}")
        self._show_current_frame()

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def num_frames(self) -> int:
        return self._num_frames

    def set_frame(self, frame: np.ndarray):
        """Set the displayed frame (already composited with overlays)."""
        self._display.set_frame(frame)

    def seek(self, frame_idx: int):
        """Seek to specific frame (clamps to valid range)."""
        frame_idx = max(0, min(frame_idx, self._num_frames - 1))
        if frame_idx == self._current_frame:
            return
        self._current_frame = frame_idx
        self._slider.blockSignals(True)
        self._slider.setValue(frame_idx)
        self._slider.blockSignals(False)
        self._update_label()
        self.frame_changed.emit(frame_idx)

    def current_frame_index(self) -> int:
        return self._current_frame

    def get_raw_frame(self, frame_idx: int | None = None) -> np.ndarray | None:
        """Get raw (no overlay) frame from cache."""
        if self._video_path is None:
            return None
        idx = frame_idx if frame_idx is not None else self._current_frame
        return self._cache.get_frame(self._video_path, idx)

    def _on_slider_changed(self, value: int):
        self._current_frame = value
        self._update_label()
        self.frame_changed.emit(value)

    def _show_current_frame(self):
        frame = self.get_raw_frame()
        if frame is not None:
            self._display.set_frame(frame)

    def _update_label(self):
        self._frame_label.setText(f"{self._current_frame} / {self._num_frames}")

    def _toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            self._btn_play.setText("\u275a\u275a")
            interval = max(1, int(1000 / self._fps))
            self._play_timer.start(interval)
        else:
            self._btn_play.setText("\u25b6")
            self._play_timer.stop()
        self.playback_toggled.emit(self._playing)

    def _advance_frame(self):
        next_frame = self._current_frame + 1
        if next_frame >= self._num_frames:
            self._toggle_play()  # stop at end
            return
        self.seek(next_frame)

    def keyPressEvent(self, event):
        key = event.key()
        mod = event.modifiers()
        if key == Qt.Key_Space:
            self._toggle_play()
        elif key == Qt.Key_Left:
            step = 10 if mod & Qt.ControlModifier else 1
            self.seek(self._current_frame - step)
        elif key == Qt.Key_Right:
            step = 10 if mod & Qt.ControlModifier else 1
            self.seek(self._current_frame + step)
        elif key == Qt.Key_Home:
            self.seek(0)
        elif key == Qt.Key_End:
            self.seek(self._num_frames - 1)
        else:
            super().keyPressEvent(event)
