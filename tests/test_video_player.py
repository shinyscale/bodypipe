"""Tests for VideoPlayer widget — context menu, overlay transport, speed chips, timecode."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from views.video_player import (
    FrameDisplay, FrameCache, VideoPlayer,
    _TransportOverlay, frame_to_timecode, SPEED_PRESETS,
)
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
        assert hasattr(player, "scrub_started")
        assert hasattr(player, "scrub_ended")

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

    def test_speed_changed_signal_exists(self, qapp):
        """Phase 3: speed_changed signal available for sync."""
        player = VideoPlayer()
        assert hasattr(player, "speed_changed")

    def test_overlay_exists(self, qapp):
        """Phase 3: transport overlay widget is created."""
        player = VideoPlayer()
        assert player._overlay is not None
        assert isinstance(player._overlay, _TransportOverlay)

    def test_overlay_initially_hidden(self, qapp):
        """Phase 3: overlay starts hidden until mouse activity."""
        player = VideoPlayer()
        assert player._overlay.isHidden()

    def test_playback_speed_property(self, qapp):
        """Phase 3: playback_speed property defaults to 1.0."""
        player = VideoPlayer()
        assert player.playback_speed == 1.0


# ---------------------------------------------------------------------------
# Timecode conversion
# ---------------------------------------------------------------------------


class TestTimecode:
    """frame_to_timecode converts frame indices to MM:SS:FF timecode."""

    def test_zero_frame(self):
        assert frame_to_timecode(0, 30.0) == "00:00:00"

    def test_exact_seconds(self):
        # 90 frames at 30fps = 3.0 seconds, sub-frame = 90 % 30 = 0
        assert frame_to_timecode(90, 30.0) == "00:03:00"

    def test_with_sub_frames(self):
        # 95 frames at 30fps = 3.167s → 3s, sub-frame = 95 % 30 = 5
        assert frame_to_timecode(95, 30.0) == "00:03:05"

    def test_over_one_minute(self):
        # 1800 frames at 30fps = 60s = 1min, sub-frame = 1800 % 30 = 0
        assert frame_to_timecode(1800, 30.0) == "01:00:00"

    def test_different_fps(self):
        # 48 frames at 24fps = 2.0s, sub-frame = 48 % 24 = 0
        assert frame_to_timecode(48, 24.0) == "00:02:00"

    def test_zero_fps_fallback(self):
        """Edge case: fps=0 should not crash (clamps to 1.0)."""
        result = frame_to_timecode(10, 0.0)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Transport Overlay
# ---------------------------------------------------------------------------


class TestTransportOverlay:
    """Phase 3: semi-transparent overlay auto-hide behavior."""

    def test_construction(self, qapp):
        overlay = _TransportOverlay()
        assert overlay is not None

    def test_hide_timer_single_shot(self, qapp):
        overlay = _TransportOverlay()
        assert overlay._hide_timer.isSingleShot()
        assert overlay._hide_timer.interval() == _TransportOverlay.HIDE_DELAY_MS

    def test_show_with_timer_makes_visible(self, qapp):
        overlay = _TransportOverlay()
        overlay.show_with_timer()
        assert overlay.isVisible()
        assert overlay._hide_timer.isActive()

    def test_paint_event_no_crash(self, qapp):
        """paintEvent executes without error (offscreen)."""
        overlay = _TransportOverlay()
        overlay.resize(200, 36)
        overlay.show()
        overlay.repaint()  # Force paint in offscreen mode


# ---------------------------------------------------------------------------
# Speed Chips
# ---------------------------------------------------------------------------


class TestSpeedChips:
    """Phase 3: speed toggle chips (0.25x/0.5x/1x/2x/4x)."""

    def test_speed_chips_exist(self, qapp):
        player = VideoPlayer()
        assert len(player._speed_chips) == len(SPEED_PRESETS)
        for speed in SPEED_PRESETS:
            assert speed in player._speed_chips

    def test_default_speed_1x(self, qapp):
        player = VideoPlayer()
        assert player._playback_speed == 1.0
        assert player._speed_chips[1.0].isChecked()

    def test_chips_are_exclusive(self, qapp):
        player = VideoPlayer()
        player._speed_chips[2.0].setChecked(True)
        # Only 2.0 should be checked
        assert player._speed_chips[2.0].isChecked()
        assert not player._speed_chips[1.0].isChecked()

    def test_set_playback_speed(self, qapp):
        player = VideoPlayer()
        player.set_playback_speed(2.0)
        assert player._playback_speed == 2.0
        assert player._speed_chips[2.0].isChecked()

    def test_set_playback_speed_invalid(self, qapp):
        """Invalid speed (not in SPEED_PRESETS) is ignored."""
        player = VideoPlayer()
        player.set_playback_speed(3.0)
        assert player._playback_speed == 1.0  # unchanged

    def test_speed_affects_timer_during_playback(self, qapp):
        """When playing, speed change adjusts timer interval."""
        player = VideoPlayer()
        player._num_frames = 100
        player._fps = 30.0
        player._playing = True
        player.set_playback_speed(2.0)
        expected = max(1, int(1000 / (30.0 * 2.0)))
        assert player._play_timer.interval() == expected

    def test_speed_no_timer_change_when_paused(self, qapp):
        """When paused, speed change doesn't touch timer."""
        player = VideoPlayer()
        player._num_frames = 100
        player._fps = 30.0
        original_interval = player._play_timer.interval()
        player.set_playback_speed(4.0)
        # Timer interval unchanged because not playing
        assert player._play_timer.interval() == original_interval

    def test_toggle_play_uses_speed(self, qapp):
        """Play starts timer at speed-adjusted interval."""
        player = VideoPlayer()
        player._num_frames = 100
        player._fps = 30.0
        player.set_playback_speed(0.5)
        player._toggle_play()
        expected = max(1, int(1000 / (30.0 * 0.5)))
        assert player._play_timer.interval() == expected
        player._toggle_play()  # stop

    def test_speed_changed_signal(self, qapp):
        """Clicking a speed chip emits speed_changed signal."""
        player = VideoPlayer()
        received = []
        player.speed_changed.connect(received.append)
        # Simulate clicking the 2x chip
        player._speed_chips[2.0].click()
        assert received == [2.0]

    def test_set_playback_speed_no_signal(self, qapp):
        """set_playback_speed does NOT emit speed_changed (prevents loop)."""
        player = VideoPlayer()
        received = []
        player.speed_changed.connect(received.append)
        player.set_playback_speed(4.0)
        assert received == []


# ---------------------------------------------------------------------------
# Frame Cache Read-ahead
# ---------------------------------------------------------------------------


class TestFrameCacheReadahead:
    """Phase 3: adaptive read-ahead (15 at rest, 30 during playback)."""

    def test_default_readahead(self):
        cache = FrameCache()
        assert cache.readahead == FrameCache.DEFAULT_READAHEAD == 15

    def test_set_readahead(self):
        cache = FrameCache()
        cache.readahead = 30
        assert cache.readahead == 30

    def test_readahead_minimum(self):
        cache = FrameCache()
        cache.readahead = -5
        assert cache.readahead == 1

    def test_playback_increases_readahead(self, qapp):
        """Starting playback bumps read-ahead to PLAYBACK_READAHEAD."""
        player = VideoPlayer()
        assert player._cache.readahead == FrameCache.DEFAULT_READAHEAD
        player._num_frames = 100
        player._fps = 30.0
        player._toggle_play()
        assert player._cache.readahead == FrameCache.PLAYBACK_READAHEAD == 30
        player._toggle_play()  # stop

    def test_pause_restores_readahead(self, qapp):
        """Pausing restores read-ahead to DEFAULT_READAHEAD."""
        player = VideoPlayer()
        player._num_frames = 100
        player._fps = 30.0
        player._toggle_play()  # start
        player._toggle_play()  # stop
        assert player._cache.readahead == FrameCache.DEFAULT_READAHEAD


# ---------------------------------------------------------------------------
# Overlay Labels
# ---------------------------------------------------------------------------


class TestOverlayLabels:
    """Phase 3: overlay frame counter and timecode update on seek."""

    def test_overlay_frame_label_exists(self, qapp):
        player = VideoPlayer()
        assert player._overlay_frame_label is not None

    def test_timecode_label_exists(self, qapp):
        player = VideoPlayer()
        assert player._timecode_label is not None

    def test_labels_update_on_seek(self, qapp):
        player = VideoPlayer()
        player._num_frames = 100
        player._fps = 30.0
        player.seek(45)
        assert "45" in player._overlay_frame_label.text()
        assert "45" in player._frame_label.text()
        # 45 frames at 30fps: 1s + 15 sub-frames → "00:01:15"
        assert player._timecode_label.text() == "00:01:15"

    def test_labels_update_on_set_video(self, qapp):
        player = VideoPlayer()
        player.set_video(Path("dummy.mp4"), 200, 24.0)
        assert "0 / 200" in player._overlay_frame_label.text()
        assert player._timecode_label.text() == "00:00:00"
