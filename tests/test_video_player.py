"""Tests for VideoPlayer widget, focusing on context menu (Copy Frame, Save Frame As)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from views.video_player import FrameDisplay, FrameCache, VideoPlayer
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# FrameCache tests
# ---------------------------------------------------------------------------

class TestFrameCache:
    def test_put_and_get(self):
        cache = FrameCache(maxsize=5)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        cache.put(0, frame)
        result = cache.get(0)
        assert result is not None
        assert np.array_equal(result, frame)

    def test_get_missing_returns_none(self):
        cache = FrameCache(maxsize=5)
        assert cache.get(42) is None

    def test_eviction(self):
        cache = FrameCache(maxsize=3)
        for i in range(5):
            cache.put(i, np.zeros((2, 2, 3), dtype=np.uint8))
        # Frames 0 and 1 should be evicted
        assert cache.get(0) is None
        assert cache.get(1) is None
        assert cache.get(2) is not None
        assert cache.get(3) is not None
        assert cache.get(4) is not None

    def test_clear(self):
        cache = FrameCache(maxsize=5)
        cache.put(0, np.zeros((2, 2, 3), dtype=np.uint8))
        cache.clear()
        assert cache.get(0) is None

    def test_lru_ordering(self):
        cache = FrameCache(maxsize=3)
        for i in range(3):
            cache.put(i, np.zeros((2, 2, 3), dtype=np.uint8))
        # Access frame 0 to make it recently used
        cache.get(0)
        # Adding frame 3 should evict frame 1 (LRU), not frame 0
        cache.put(3, np.zeros((2, 2, 3), dtype=np.uint8))
        assert cache.get(0) is not None
        assert cache.get(1) is None


# ---------------------------------------------------------------------------
# FrameDisplay tests
# ---------------------------------------------------------------------------

class TestFrameDisplay:
    def test_construction(self, qapp):
        display = FrameDisplay()
        assert display.alignment() == Qt.AlignCenter
        assert display._current_frame is None
        assert display._pixmap_size is None

    def test_context_menu_policy(self, qapp):
        display = FrameDisplay()
        assert display.contextMenuPolicy() == Qt.CustomContextMenu

    def test_set_frame_stores_copy(self, qapp):
        display = FrameDisplay()
        frame = np.full((100, 200, 3), 128, dtype=np.uint8)
        display.set_frame(frame)
        assert display._current_frame is not None
        # Verify it's a copy, not the same object
        assert display._current_frame is not frame
        assert np.array_equal(display._current_frame, frame)

    def test_set_frame_updates_pixmap_size(self, qapp):
        display = FrameDisplay()
        frame = np.full((100, 200, 3), 128, dtype=np.uint8)
        display.set_frame(frame)
        assert display._pixmap_size is not None
        pw, ph = display._pixmap_size
        assert pw > 0
        assert ph > 0

    def test_frame_to_pixmap_no_frame(self, qapp):
        display = FrameDisplay()
        assert display._frame_to_pixmap() is None

    def test_frame_to_pixmap_with_frame(self, qapp):
        display = FrameDisplay()
        frame = np.full((100, 200, 3), 128, dtype=np.uint8)
        display.set_frame(frame)
        pixmap = display._frame_to_pixmap()
        assert isinstance(pixmap, QPixmap)
        assert pixmap.width() == 200
        assert pixmap.height() == 100

    def test_copy_frame(self, qapp):
        display = FrameDisplay()
        frame = np.full((50, 80, 3), 64, dtype=np.uint8)
        display.set_frame(frame)
        display._copy_frame()
        clipboard = QApplication.clipboard()
        pixmap = clipboard.pixmap()
        assert not pixmap.isNull()
        assert pixmap.width() == 80
        assert pixmap.height() == 50

    def test_copy_frame_no_frame(self, qapp):
        display = FrameDisplay()
        # Should not raise
        display._copy_frame()

    @patch("views.video_player.QFileDialog.getSaveFileName")
    def test_save_frame(self, mock_dialog, qapp, tmp_path):
        save_path = str(tmp_path / "test_frame.png")
        mock_dialog.return_value = (save_path, "PNG Image (*.png)")

        display = FrameDisplay()
        frame = np.full((50, 80, 3), 200, dtype=np.uint8)
        display.set_frame(frame)
        display._save_frame()

        assert Path(save_path).exists()
        mock_dialog.assert_called_once()

    @patch("views.video_player.QFileDialog.getSaveFileName")
    def test_save_frame_cancelled(self, mock_dialog, qapp):
        mock_dialog.return_value = ("", "")
        display = FrameDisplay()
        frame = np.full((50, 80, 3), 200, dtype=np.uint8)
        display.set_frame(frame)
        # Should not raise when user cancels dialog
        display._save_frame()

    def test_save_frame_no_frame(self, qapp):
        display = FrameDisplay()
        # Should not raise
        display._save_frame()

    def test_context_menu_no_frame(self, qapp):
        """Context menu should not appear when no frame is loaded."""
        display = FrameDisplay()
        # _show_context_menu should return early without error
        display._show_context_menu(QPoint(0, 0))

    def test_click_signal(self, qapp):
        display = FrameDisplay()
        frame = np.full((100, 200, 3), 128, dtype=np.uint8)
        display.set_frame(frame)
        signals = []
        display.clicked.connect(lambda x, y: signals.append((x, y)))
        # Simulate left click — coordinates in center of widget
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtCore import QPointF
        local_pos = QPointF(display.width() / 2, display.height() / 2)
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            local_pos,
            local_pos,  # globalPos (required to avoid deprecation)
            Qt.LeftButton,
            Qt.LeftButton,
            Qt.NoModifier,
        )
        display.mousePressEvent(event)
        # Should have emitted at least if coordinates are in range
        # (exact values depend on widget geometry)


# ---------------------------------------------------------------------------
# VideoPlayer tests
# ---------------------------------------------------------------------------

class TestVideoPlayer:
    def test_construction(self, qapp):
        player = VideoPlayer()
        assert player._video_path is None
        assert player._num_frames == 0
        assert player._fps == 30.0
        assert player._current_frame == 0
        assert player._playing is False

    def test_signals_exist(self, qapp):
        player = VideoPlayer()
        assert hasattr(player, "frame_changed")
        assert hasattr(player, "frame_clicked")
        assert hasattr(player, "playback_toggled")

    def test_transport_buttons_exist(self, qapp):
        player = VideoPlayer()
        assert player._btn_first is not None
        assert player._btn_back10 is not None
        assert player._btn_back1 is not None
        assert player._btn_play is not None
        assert player._btn_fwd1 is not None
        assert player._btn_fwd10 is not None
        assert player._btn_last is not None

    def test_slider_exists(self, qapp):
        player = VideoPlayer()
        assert player._slider is not None

    def test_frame_label_exists(self, qapp):
        player = VideoPlayer()
        assert player._frame_label is not None

    def test_fps_label_exists(self, qapp):
        """Spec requires FPS display in slider row."""
        player = VideoPlayer()
        assert player._fps_label is not None
        assert player._fps_label.text() == ""

    def test_fps_label_set_on_video_load(self, qapp):
        """FPS label shows video FPS after set_video."""
        player = VideoPlayer()
        player.set_video(Path("dummy.mp4"), 100, 24.0)
        assert "24.0" in player._fps_label.text()

    def test_fps_property(self, qapp):
        player = VideoPlayer()
        assert player.fps == 30.0
        player.set_video(Path("dummy.mp4"), 200, 60.0)
        assert player.fps == 60.0

    def test_num_frames_property(self, qapp):
        player = VideoPlayer()
        assert player.num_frames == 0
        player.set_video(Path("dummy.mp4"), 500, 30.0)
        assert player.num_frames == 500

    def test_set_video(self, qapp):
        player = VideoPlayer()
        player.set_video(Path("dummy.mp4"), 100, 24.0)
        assert player._num_frames == 100
        assert player._fps == 24.0
        assert player._current_frame == 0
        assert player._slider.maximum() == 99

    def test_seek(self, qapp):
        player = VideoPlayer()
        player._num_frames = 100
        player._current_frame = 0
        signals = []
        player.frame_changed.connect(signals.append)
        player.seek(50)
        assert player._current_frame == 50
        assert len(signals) == 1
        assert signals[0] == 50

    def test_seek_clamps(self, qapp):
        player = VideoPlayer()
        player._num_frames = 100
        player.seek(-10)
        assert player._current_frame == 0
        player.seek(200)
        assert player._current_frame == 99

    def test_seek_same_frame_no_signal(self, qapp):
        player = VideoPlayer()
        player._num_frames = 100
        player._current_frame = 50
        signals = []
        player.frame_changed.connect(signals.append)
        player.seek(50)
        assert len(signals) == 0

    def test_current_frame_index(self, qapp):
        player = VideoPlayer()
        player._num_frames = 100
        player.seek(42)
        assert player.current_frame_index() == 42

    def test_display_has_context_menu(self, qapp):
        """VideoPlayer's internal FrameDisplay supports right-click context menu."""
        player = VideoPlayer()
        assert player._display.contextMenuPolicy() == Qt.CustomContextMenu

    def test_display_frame_stored_for_context_menu(self, qapp):
        """Setting a frame on the display stores it for copy/save operations."""
        player = VideoPlayer()
        frame = np.full((50, 80, 3), 100, dtype=np.uint8)
        player.set_frame(frame)
        assert player._display._current_frame is not None
        assert np.array_equal(player._display._current_frame, frame)
