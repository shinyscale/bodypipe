"""Tests for SinglePipelineSettings — the single-person pipeline settings widget.

Why: These tests verify settings assembly, config round-trip, solve_config
persistence, and worker lifecycle for the single-person GVHMR body capture
pipeline — the primary entry point for the application.

History: Originally tested the now-deleted SinglePersonTab; migrated to test
SinglePipelineSettings directly as part of the tab-to-dock cleanup (Commit 1E).
Tests that were unique to the tab's output preview panel (file list, preview
player) were removed since that UI no longer exists in the dock layout.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session
from models.pipeline_config import PipelineConfig
from views.pipeline_settings import SinglePipelineSettings, _DropArea


# ---------------------------------------------------------------------------
# Widget construction
# ---------------------------------------------------------------------------


class TestSinglePipelineSettingsConstruction:
    """Verify that the widget can be created and has the expected child widgets."""

    def test_creates_without_error(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        assert w is not None

    def test_run_button_disabled_initially(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        assert not w._run_btn.isEnabled()

    def test_cancel_button_hidden_initially(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._cancel_btn.isHidden()

    def test_progress_bar_hidden_initially(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._progress_bar.isHidden()

    def test_has_settings_widgets(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._static_cam is not None
        assert w._use_dpvo is not None
        assert w._focal_mm is not None


# ---------------------------------------------------------------------------
# Settings / PipelineConfig
# ---------------------------------------------------------------------------


class TestSettingsConfig:
    """Verify round-trip between UI controls and PipelineConfig."""

    def test_default_config(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        config = w.get_config()
        assert config.mode == "single"
        assert config.static_cam is True
        assert config.use_dpvo is False
        assert config.focal_mm == 24.0

    def test_changed_config(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._static_cam.setChecked(False)
        w._use_dpvo.setChecked(True)
        w._focal_mm.setValue(50.0)

        config = w.get_config()
        assert config.static_cam is False
        assert config.use_dpvo is True
        assert config.focal_mm == 50.0

    def test_set_config(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w.set_config(PipelineConfig(
            static_cam=False, use_dpvo=True, focal_mm=85.0,
        ))
        assert w._static_cam.isChecked() is False
        assert w._use_dpvo.isChecked() is True
        assert w._focal_mm.value() == 85.0

    def test_config_round_trip(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        original = PipelineConfig(static_cam=False, use_dpvo=True, focal_mm=35.0)
        w.set_config(original)
        recovered = w.get_config()
        assert recovered.static_cam == original.static_cam
        assert recovered.use_dpvo == original.use_dpvo
        assert recovered.focal_mm == original.focal_mm


# ---------------------------------------------------------------------------
# Static cam -> DPVO auto-disable
# ---------------------------------------------------------------------------


class TestStaticCamDpvoInterlock:
    """DPVO must be disabled and unchecked when static camera is enabled.

    Matches Gradio GUI behavior: static camera and DPVO are mutually exclusive
    because DPVO estimates camera motion which is meaningless for a static camera.
    """

    def test_dpvo_disabled_by_default(self, qapp):
        """DPVO starts disabled because static_cam defaults to True."""
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._static_cam.isChecked() is True
        assert w._use_dpvo.isChecked() is False
        assert not w._use_dpvo.isEnabled()

    def test_uncheck_static_cam_enables_dpvo(self, qapp):
        """Unchecking static camera re-enables DPVO."""
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._static_cam.setChecked(False)
        assert w._use_dpvo.isEnabled()

    def test_check_static_cam_disables_and_unchecks_dpvo(self, qapp):
        """Checking static camera disables DPVO and forces it unchecked."""
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._static_cam.setChecked(False)
        w._use_dpvo.setChecked(True)
        assert w._use_dpvo.isChecked() is True

        w._static_cam.setChecked(True)
        assert w._use_dpvo.isChecked() is False
        assert not w._use_dpvo.isEnabled()

    def test_dpvo_stays_disabled_after_run_with_static_cam(self, qapp):
        """After pipeline run ends, DPVO remains disabled if static_cam is on."""
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._video_path = Path("/tmp/test.mp4")
        w._set_running(True)
        w._set_running(False)
        assert w._static_cam.isChecked() is True
        assert not w._use_dpvo.isEnabled()

    def test_set_config_respects_interlock(self, qapp):
        """set_config with static_cam=False allows DPVO to be True."""
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w.set_config(PipelineConfig(static_cam=False, use_dpvo=True, focal_mm=24.0))
        assert w._use_dpvo.isEnabled()
        assert w._use_dpvo.isChecked() is True


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------


class TestVideoLoading:
    """Verify that loading a video updates session state and enables run."""

    def test_load_video_updates_session(self, qapp, tmp_path):
        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._load_video(str(video_path))

        assert session.video_path == video_path
        assert session.num_frames > 0
        assert session.img_width > 0
        assert session.img_height > 0
        assert w._run_btn.isEnabled()

    def test_load_nonexistent_video(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))

        messages = []
        w.status_message.connect(messages.append)
        w._load_video("/tmp/nonexistent_video.mp4")

        assert not w._run_btn.isEnabled()
        assert any("not found" in m.lower() or "cannot open" in m.lower() for m in messages)

    def test_load_invalid_file(self, qapp, tmp_path):
        bad_file = tmp_path / "bad.mp4"
        bad_file.write_text("not a video")

        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))

        messages = []
        w.status_message.connect(messages.append)
        w._load_video(str(bad_file))

        assert not w._run_btn.isEnabled()


# ---------------------------------------------------------------------------
# Running state
# ---------------------------------------------------------------------------


class TestRunningState:
    """Verify UI state transitions when running/idle."""

    def test_set_running_true(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._video_path = Path("/tmp/test.mp4")
        w._set_running(True)

        assert not w._run_btn.isEnabled()
        assert not w._cancel_btn.isHidden()
        assert not w._progress_bar.isHidden()
        assert not w._browse_btn.isEnabled()
        assert not w._static_cam.isEnabled()
        assert not w._use_dpvo.isEnabled()
        assert not w._focal_mm.isEnabled()

    def test_set_running_false(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._video_path = Path("/tmp/test.mp4")
        w._set_running(True)
        w._set_running(False)

        assert w._run_btn.isEnabled()
        assert w._cancel_btn.isHidden()
        assert w._progress_bar.isHidden()
        assert w._browse_btn.isEnabled()
        assert w._static_cam.isEnabled()
        # DPVO stays disabled when static_cam is checked (default True)
        assert not w._use_dpvo.isEnabled()
        assert w._focal_mm.isEnabled()


# ---------------------------------------------------------------------------
# Progress updates
# ---------------------------------------------------------------------------


class TestProgressUpdates:
    def test_on_progress(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._progress_bar.show()
        w._on_progress(0.5, "ViTPose")
        assert w._progress_bar.value() == 500
        assert w._progress_label.text() == "ViTPose"

    def test_on_progress_full(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._on_progress(1.0, "Done")
        assert w._progress_bar.value() == 1000


# ---------------------------------------------------------------------------
# Worker lifecycle (mocked)
# ---------------------------------------------------------------------------


class TestWorkerLifecycle:
    def test_on_finished(self, qapp, tmp_path):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._video_path = Path("/tmp/test.mp4")
        w._set_running(True)

        received = []
        w.pipeline_finished.connect(received.append)

        w._on_finished({
            "output_dir": str(tmp_path),
            "video_path": "/tmp/test.mp4",
        })

        assert not w._running
        assert len(received) == 1
        assert received[0]["output_dir"] == str(tmp_path)

    def test_on_error(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._video_path = Path("/tmp/test.mp4")
        w._set_running(True)

        messages = []
        w.status_message.connect(messages.append)
        w._on_error("Something broke")

        assert not w._running
        assert any("error" in m.lower() for m in messages)


# ---------------------------------------------------------------------------
# Solve config save/restore (Gradio parity)
# ---------------------------------------------------------------------------


class TestSolveConfigSaveRestore:
    """Verify solve_config.json is saved on run and restored on video load.

    Why: Gradio saves UI settings to solve_config.json in the output directory
    when a pipeline runs. When the same video is re-entered, settings are
    restored. This provides session continuity -- users don't need to
    reconfigure settings for previously-processed videos.
    """

    def test_output_dir_for_video(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        result = w._output_dir_for_video(Path("/tmp/videos/dance.mp4"))
        assert result == Path("/tmp/GVHMR/outputs/demo/dance")

    def test_on_run_saves_config(self, qapp, tmp_path):
        """_on_run should save solve_config.json to the output directory."""
        gvhmr_root = tmp_path / "GVHMR"
        gvhmr_root.mkdir()
        session = Session()
        w = SinglePipelineSettings(session, gvhmr_root)
        w._video_path = Path("/tmp/test_video.mp4")
        w._static_cam.setChecked(False)
        w._use_dpvo.setChecked(True)
        w._focal_mm.setValue(50.0)

        with patch("views.pipeline_settings.GVHMRWorker") as MockWorker:
            mock_instance = MagicMock()
            MockWorker.return_value = mock_instance
            w._on_run()

        config_path = gvhmr_root / "outputs" / "demo" / "test_video" / "solve_config.json"
        assert config_path.is_file()

        loaded = PipelineConfig.load(config_path)
        assert loaded.static_cam is False
        assert loaded.use_dpvo is True
        assert loaded.focal_mm == 50.0

    def test_load_video_restores_config(self, qapp, tmp_path):
        """Loading a video should restore settings from solve_config.json."""
        gvhmr_root = tmp_path / "GVHMR"
        output_dir = gvhmr_root / "outputs" / "demo" / "test"
        output_dir.mkdir(parents=True)
        PipelineConfig(
            static_cam=False, use_dpvo=True, focal_mm=50.0,
        ).save(output_dir / "solve_config.json")

        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        w = SinglePipelineSettings(session, gvhmr_root)
        w._load_video(str(video_path))

        assert w._static_cam.isChecked() is False
        assert w._use_dpvo.isChecked() is True
        assert w._focal_mm.value() == 50.0

    def test_load_video_no_config_keeps_defaults(self, qapp, tmp_path):
        """Loading a video without solve_config.json keeps default settings."""
        gvhmr_root = tmp_path / "GVHMR"
        gvhmr_root.mkdir()

        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        w = SinglePipelineSettings(session, gvhmr_root)
        w._load_video(str(video_path))

        assert w._static_cam.isChecked() is True
        assert w._use_dpvo.isChecked() is False
        assert w._focal_mm.value() == 24.0

    def test_load_video_corrupt_config_ignored(self, qapp, tmp_path):
        """Corrupt solve_config.json should be silently ignored."""
        gvhmr_root = tmp_path / "GVHMR"
        output_dir = gvhmr_root / "outputs" / "demo" / "test"
        output_dir.mkdir(parents=True)
        (output_dir / "solve_config.json").write_text("not valid json{{{")

        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        w = SinglePipelineSettings(session, gvhmr_root)
        # Should not raise
        w._load_video(str(video_path))
        # Defaults preserved
        assert w._static_cam.isChecked() is True

    def test_restore_emits_log_message(self, qapp, tmp_path):
        """Restoring config should emit a log message."""
        gvhmr_root = tmp_path / "GVHMR"
        output_dir = gvhmr_root / "outputs" / "demo" / "test"
        output_dir.mkdir(parents=True)
        PipelineConfig(static_cam=False).save(output_dir / "solve_config.json")

        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        w = SinglePipelineSettings(session, gvhmr_root)
        logs = []
        w.log_message.connect(lambda text, level: logs.append((text, level)))
        w._load_video(str(video_path))

        assert any("restored" in text.lower() for text, _ in logs)


# ---------------------------------------------------------------------------
# DropArea
# ---------------------------------------------------------------------------


class TestDropArea:
    def test_creates(self, qapp):
        area = _DropArea()
        assert area.acceptDrops()
        assert "Drop video" in area.text()


# ---------------------------------------------------------------------------
# AppWindow integration
# ---------------------------------------------------------------------------


class TestAppWindowIntegration:
    """Verify AppWindow creates SinglePipelineSettings and wires signals."""

    def test_single_settings_is_settings_widget(self, qapp):
        from app_window import AppWindow
        window = AppWindow()
        assert isinstance(window._single_settings, SinglePipelineSettings)

    def test_status_signal_connected(self, qapp):
        from app_window import AppWindow
        window = AppWindow()
        window._single_settings.status_message.emit("test status")
        assert window._status_label.text() == "test status"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_video(path: Path, frames: int = 5, size: tuple = (64, 48)) -> Path:
    """Create a minimal video file for testing."""
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, size)
    for i in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:, :, 1] = i * 40  # green gradient
        writer.write(frame)
    writer.release()
    return path
