"""Video player widget with frame display, seeking, playback, and LRU cache.

Phase 3 additions: semi-transparent transport overlay with auto-hide,
speed toggle chips (0.25x/0.5x/1x/2x/4x), timecode display, and
adaptive frame cache read-ahead (30 during playback, 15 at rest).

Why overlay transport: Mocha Pro-style floating controls maximize the video
display area while keeping transport accessible.  Auto-hide after 2s keeps
the viewport clean during review.  Speed chips replace freeform control
with the 5 most useful scrubbing speeds.
"""

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
    QButtonGroup,
)
from PySide6.QtCore import Signal, Qt, QTimer, QEvent
from PySide6.QtGui import QImage, QPixmap, QMouseEvent, QPainter, QColor


# Speed presets for transport overlay and track timeline footer
SPEED_PRESETS = (0.25, 0.5, 1.0, 2.0, 4.0)


def frame_to_timecode(frame: int, fps: float) -> str:
    """Convert frame index to MM:SS:FF timecode string."""
    fps = max(1.0, fps)
    total_seconds = frame / fps
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    sub_frames = int(frame % fps)
    return f"{minutes:02d}:{seconds:02d}:{sub_frames:02d}"


class FrameCache:
    """LRU cache with read-ahead for sequential video access."""

    DEFAULT_READAHEAD = 15
    PLAYBACK_READAHEAD = 30
    READAHEAD = DEFAULT_READAHEAD  # backward compat class attribute
    MAX_SIZE = 200

    def __init__(self, maxsize: int = MAX_SIZE):
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._maxsize = maxsize
        self._readahead = self.DEFAULT_READAHEAD

    @property
    def readahead(self) -> int:
        return self._readahead

    @readahead.setter
    def readahead(self, n: int):
        self._readahead = max(1, n)

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
        for i in range(self._readahead + 1):
            ret, frame = cap.read()
            if not ret:
                break
            self.put(frame_idx + i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return self.get(frame_idx)


class FrameDisplay(QLabel):
    """QLabel that displays video frames and emits click/drag positions."""

    clicked = Signal(float, float)  # normalized (x, y) in image space
    drag_rect = Signal(float, float, float, float)  # (x1, y1, x2, y2) normalized

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        from theme import COLORS
        self.setStyleSheet(f"background-color: {COLORS['bg_input']};")
        self._pixmap_size = None
        self._current_frame: np.ndarray | None = None
        self._drag_start: tuple[float, float] | None = None
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.setMouseTracking(True)

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

    def _screen_to_norm(self, pos) -> tuple[float, float] | None:
        """Convert screen position to normalized [0,1] image coords."""
        if not self._pixmap_size:
            return None
        pw, ph = self._pixmap_size
        ox = (self.width() - pw) / 2
        oy = (self.height() - ph) / 2
        x = (pos.x() - ox) / pw
        y = (pos.y() - oy) / ph
        if 0 <= x <= 1 and 0 <= y <= 1:
            return (x, y)
        return None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            norm = self._screen_to_norm(event.position())
            if norm:
                self._drag_start = norm
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._drag_start:
            norm = self._screen_to_norm(event.position())
            if norm:
                x1, y1 = self._drag_start
                x2, y2 = norm
                # If drag distance is significant (>2% of image), emit drag_rect
                if abs(x2 - x1) > 0.02 and abs(y2 - y1) > 0.02:
                    self.drag_rect.emit(
                        min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
                    )
                else:
                    # Small movement = click
                    self.clicked.emit(x1, y1)
            self._drag_start = None
        super().mouseReleaseEvent(event)


class _TransportOverlay(QWidget):
    """Semi-transparent overlay bar floating at the bottom of the video display.

    Why overlay: Mocha Pro pattern — keeps the viewport uncluttered during
    review while transport controls remain one mouse-move away.  Auto-hides
    after 2s of inactivity; pauses the timer while the cursor is over the
    overlay so button clicks don't race with the hide.
    """

    HIDE_DELAY_MS = 2000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(self.HIDE_DELAY_MS)
        self._hide_timer.timeout.connect(self.hide)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(0, 0, 0, 180))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(self.rect(), 6, 6)
        p.end()

    def show_with_timer(self):
        """Show the overlay and (re)start the auto-hide timer."""
        self.show()
        self.raise_()
        self._hide_timer.start()

    def enterEvent(self, event):
        """Pause hide timer while cursor is on the overlay."""
        self._hide_timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Resume hide timer when cursor leaves the overlay."""
        self._hide_timer.start()
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        """Restart timer on any movement over the overlay."""
        self._hide_timer.start()
        super().mouseMoveEvent(event)


class VideoPlayer(QWidget):
    """Video frame display with overlay transport controls, slider, and playback.

    The transport overlay floats semi-transparently over the bottom of the
    video display and auto-hides after 2 seconds of mouse inactivity.
    Speed toggle chips (0.25x .. 4x) replace freeform speed control.
    """

    frame_changed = Signal(int)
    frame_clicked = Signal(float, float)
    bbox_dragged = Signal(float, float, float, float)  # (x1, y1, x2, y2) normalized
    playback_toggled = Signal(bool)
    scrub_started = Signal()   # slider press — user is actively scrubbing
    scrub_ended = Signal()     # slider release — scrubbing finished
    speed_changed = Signal(float)  # emitted on user speed chip click

    def __init__(self, parent=None):
        super().__init__(parent)
        self._video_path: Path | None = None
        self._num_frames = 0
        self._fps = 30.0
        self._current_frame = 0
        self._playing = False
        self._playback_speed = 1.0
        self._cache = FrameCache()

        self._setup_ui()
        self._connect_signals()

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance_frame)

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Frame display — mouse tracking for overlay show/hide
        self._display = FrameDisplay()
        self._display.setMouseTracking(True)
        self._display.installEventFilter(self)
        layout.addWidget(self._display, 1)

        # --- Transport Overlay (floats over display bottom) ---
        self._overlay = _TransportOverlay(self)
        self._overlay.hide()

        overlay_layout = QHBoxLayout(self._overlay)
        overlay_layout.setContentsMargins(10, 5, 10, 5)
        overlay_layout.setSpacing(6)

        # Primary transport buttons (in overlay)
        _btn_style = (
            "QToolButton { background: rgba(255,255,255,30); "
            "color: #eff0f1; border: none; border-radius: 3px; font-size: 13px; }"
            "QToolButton:hover { background: rgba(202,149,46,120); }"
            "QToolButton:pressed { background: rgba(202,149,46,180); }"
        )

        self._btn_back1 = QToolButton()
        self._btn_back1.setText("\u25c2")
        self._btn_back1.setToolTip("Back 1 frame (Left)")

        self._btn_play = QToolButton()
        self._btn_play.setText("\u25b6")
        self._btn_play.setToolTip("Play/Pause (Space)")

        self._btn_fwd1 = QToolButton()
        self._btn_fwd1.setText("\u25b8")
        self._btn_fwd1.setToolTip("Forward 1 frame (Right)")

        for btn in (self._btn_back1, self._btn_play, self._btn_fwd1):
            btn.setFixedSize(28, 24)
            btn.setStyleSheet(_btn_style)
            btn.setFocusPolicy(Qt.NoFocus)
            overlay_layout.addWidget(btn)

        # Separator
        _sep = QLabel("\u2502")
        _sep.setStyleSheet("color: rgba(255,255,255,40); background: transparent;")
        _sep.setFixedWidth(10)
        overlay_layout.addWidget(_sep)

        # Frame counter
        self._overlay_frame_label = QLabel("0 / 0")
        self._overlay_frame_label.setStyleSheet(
            "color: #eff0f1; background: transparent; "
            "font-size: 11px; font-family: monospace;"
        )
        overlay_layout.addWidget(self._overlay_frame_label)

        # Timecode
        self._timecode_label = QLabel("00:00:00")
        self._timecode_label.setStyleSheet(
            "color: rgba(239,240,241,160); background: transparent; "
            "font-size: 11px; font-family: monospace;"
        )
        overlay_layout.addWidget(self._timecode_label)

        overlay_layout.addStretch()

        # Speed chips
        self._speed_group = QButtonGroup(self)
        self._speed_group.setExclusive(True)
        self._speed_chips: dict[float, QToolButton] = {}

        _chip_style = (
            "QToolButton { background: transparent; color: #76797C; "
            "border: 1px solid rgba(118,121,124,80); border-radius: 3px; "
            "padding: 1px 4px; font-size: 10px; }"
            "QToolButton:checked { background: rgba(202,149,46,200); "
            "color: #eff0f1; border-color: #ca952e; }"
            "QToolButton:hover { border-color: #ca952e; }"
        )
        for speed in SPEED_PRESETS:
            label = f"{int(speed)}x" if speed == int(speed) else f"{speed}x"
            chip = QToolButton()
            chip.setText(label)
            chip.setCheckable(True)
            chip.setFixedHeight(20)
            chip.setStyleSheet(_chip_style)
            chip.setFocusPolicy(Qt.NoFocus)
            self._speed_group.addButton(chip)
            self._speed_chips[speed] = chip
            overlay_layout.addWidget(chip)
        self._speed_chips[1.0].setChecked(True)

        # --- Hidden transport buttons (keyboard-only, backward compat) ---
        self._btn_first = QToolButton(self)
        self._btn_first.setText("|<")
        self._btn_first.setToolTip("First frame (Home)")
        self._btn_first.setFixedSize(32, 28)
        self._btn_first.hide()

        self._btn_back10 = QToolButton(self)
        self._btn_back10.setText("<<")
        self._btn_back10.setToolTip("Back 10 frames (Ctrl+Left)")
        self._btn_back10.setFixedSize(32, 28)
        self._btn_back10.hide()

        self._btn_fwd10 = QToolButton(self)
        self._btn_fwd10.setText(">>")
        self._btn_fwd10.setToolTip("Forward 10 frames (Ctrl+Right)")
        self._btn_fwd10.setFixedSize(32, 28)
        self._btn_fwd10.hide()

        self._btn_last = QToolButton(self)
        self._btn_last.setText(">|")
        self._btn_last.setToolTip("Last frame (End)")
        self._btn_last.setFixedSize(32, 28)
        self._btn_last.hide()

        # --- Slider row (always visible, below display) ---
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
        self._display.drag_rect.connect(self.bbox_dragged.emit)
        self._slider.valueChanged.connect(self._on_slider_changed)
        self._slider.sliderPressed.connect(self.scrub_started.emit)
        self._slider.sliderReleased.connect(self.scrub_ended.emit)
        self._btn_first.clicked.connect(lambda: self.seek(0))
        self._btn_last.clicked.connect(lambda: self.seek(self._num_frames - 1))
        self._btn_back1.clicked.connect(lambda: self.seek(self._current_frame - 1))
        self._btn_fwd1.clicked.connect(lambda: self.seek(self._current_frame + 1))
        self._btn_back10.clicked.connect(lambda: self.seek(self._current_frame - 10))
        self._btn_fwd10.clicked.connect(lambda: self.seek(self._current_frame + 10))
        self._btn_play.clicked.connect(self._toggle_play)
        self._speed_group.buttonClicked.connect(self._on_speed_chip_clicked)

    # ------------------------------------------------------------------
    # Overlay positioning and mouse tracking
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event):
        """Show overlay on mouse activity over the display."""
        if obj is self._display:
            t = event.type()
            if t == QEvent.Type.MouseMove or t == QEvent.Type.Enter:
                self._overlay.show_with_timer()
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_overlay()

    def _reposition_overlay(self):
        """Position overlay at the bottom of the display area."""
        dg = self._display.geometry()
        oh = self._overlay.sizeHint().height()
        if oh < 20:
            oh = 36
        margin = 10
        self._overlay.setGeometry(
            dg.x() + margin,
            dg.y() + dg.height() - oh - margin,
            max(100, dg.width() - 2 * margin),
            oh,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    @property
    def playback_speed(self) -> float:
        return self._playback_speed

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

    def set_playback_speed(self, speed: float):
        """Set playback speed from external source (no signal emitted).

        Why no signal: prevents infinite loop when synced with track timeline.
        Only user-initiated speed chip clicks emit speed_changed.
        """
        if speed not in self._speed_chips:
            return
        self._playback_speed = speed
        self._speed_group.blockSignals(True)
        self._speed_chips[speed].setChecked(True)
        self._speed_group.blockSignals(False)
        if self._playing:
            interval = max(1, int(1000 / (self._fps * self._playback_speed)))
            self._play_timer.setInterval(interval)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_slider_changed(self, value: int):
        self._current_frame = value
        self._update_label()
        self.frame_changed.emit(value)

    def _show_current_frame(self):
        frame = self.get_raw_frame()
        if frame is not None:
            self._display.set_frame(frame)

    def _update_label(self):
        text = f"{self._current_frame} / {self._num_frames}"
        self._frame_label.setText(text)
        self._overlay_frame_label.setText(text)
        self._timecode_label.setText(
            frame_to_timecode(self._current_frame, self._fps)
        )

    def _on_speed_chip_clicked(self, button):
        """User clicked a speed chip — update playback and emit signal."""
        for speed, chip in self._speed_chips.items():
            if chip is button:
                self._playback_speed = speed
                if self._playing:
                    interval = max(1, int(1000 / (self._fps * self._playback_speed)))
                    self._play_timer.setInterval(interval)
                self.speed_changed.emit(speed)
                break

    def _toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            self._btn_play.setText("\u275a\u275a")
            interval = max(1, int(1000 / (self._fps * self._playback_speed)))
            self._play_timer.start(interval)
            self._cache.readahead = FrameCache.PLAYBACK_READAHEAD
        else:
            self._btn_play.setText("\u25b6")
            self._play_timer.stop()
            self._cache.readahead = FrameCache.DEFAULT_READAHEAD
        self.playback_toggled.emit(self._playing)

    def _advance_frame(self):
        # At speeds > 1x, skip frames so playback is perceptibly faster
        step = max(1, round(self._playback_speed))
        next_frame = self._current_frame + step
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
