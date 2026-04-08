"""Tests for multi-person pipeline settings and track overview.

Why: MultiPipelineSettings manages the most complex pipeline mode — multi-person
tracking with identity management. These tests verify widget construction,
settings round-trip, video loading, running state, and the TrackOverview widget
that displays per-person confidence timelines.

History: Originally tested the now-deleted MultiPersonTab; migrated to test
MultiPipelineSettings + TrackOverview + AppWindow directly as part of the
tab-to-dock cleanup (Commit 1E). Signal hub wiring and reprocess tests moved
to test_app_window.py since AppWindow now owns those responsibilities.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session, PersonTrack
from models.pipeline_config import PipelineConfig
from views.pipeline_settings import MultiPipelineSettings
from views.track_overview import TrackOverview, PERSON_COLORS


# ---------------------------------------------------------------------------
# Widget construction
# ---------------------------------------------------------------------------


class TestMultiPipelineSettingsConstruction:
    """Verify that the widget creates correctly and has expected child widgets."""

    def test_creates_without_error(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w is not None

    def test_run_button_disabled_initially(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert not w._run_btn.isEnabled()

    def test_cancel_button_hidden_initially(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._cancel_btn.isHidden()

    def test_progress_bar_hidden_initially(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._progress_bar.isHidden()

    def test_progress_label_hidden_initially(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._progress_label.isHidden()

    def test_has_pipeline_settings(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._static_cam is not None
        assert w._use_dpvo is not None
        assert w._focal_mm is not None
        assert w._target_fps is not None
        assert w._fbx_naming is not None

    def test_has_multi_person_settings(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._max_persons is not None
        assert w._confidence_threshold is not None
        assert w._use_inpainting is not None
        assert w._render_overlays is not None

    def test_max_persons_defaults(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._max_persons.value() == 8
        assert w._max_persons.minimum() == 1
        assert w._max_persons.maximum() == 20

    def test_confidence_threshold_defaults(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._confidence_threshold.value() == 0.5
        assert w._confidence_threshold.minimum() == 0.0
        assert w._confidence_threshold.maximum() == 1.0

    def test_target_fps_defaults(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._target_fps.value() == 30.0

    def test_fbx_naming_defaults(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._fbx_naming.currentText() == "Mixamo (Cascadeur)"

    def test_inpainting_default(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._use_inpainting.isChecked() is True

    def test_render_overlays_default(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._render_overlays.isChecked() is False



# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestMultiPipelineSettingsConfig:
    def test_default_config(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        config = w.get_config()
        assert config.mode == "multi"
        assert config.static_cam is True
        assert config.use_dpvo is False
        assert config.max_persons == 8
        assert config.confidence_threshold == 0.5
        assert config.target_fps == 30.0
        assert config.fbx_naming == "Mixamo (Cascadeur)"
        assert config.use_inpainting is True
        assert config.render_overlays is False

    def test_changed_config(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        w._static_cam.setChecked(False)
        w._use_dpvo.setChecked(True)
        w._max_persons.setValue(4)
        w._confidence_threshold.setValue(0.8)
        w._target_fps.setValue(24.0)
        w._use_inpainting.setChecked(False)
        w._render_overlays.setChecked(False)

        config = w.get_config()
        assert config.static_cam is False
        assert config.use_dpvo is True
        assert config.max_persons == 4
        assert config.confidence_threshold == 0.8
        assert config.target_fps == 24.0
        assert config.use_inpainting is False
        assert config.render_overlays is False

    def test_set_config(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        w.set_config(PipelineConfig(
            static_cam=False,
            use_dpvo=True,
            focal_mm=50.0,
            max_persons=3,
            confidence_threshold=0.7,
            target_fps=60.0,
            fbx_naming="UE5 Mannequin",
            use_inpainting=False,
            render_overlays=False,
        ))
        assert w._static_cam.isChecked() is False
        assert w._use_dpvo.isChecked() is True
        assert w._focal_mm.value() == 50.0
        assert w._max_persons.value() == 3
        assert w._confidence_threshold.value() == 0.7
        assert w._target_fps.value() == 60.0
        assert w._fbx_naming.currentText() == "UE5 Mannequin"
        assert w._use_inpainting.isChecked() is False
        assert w._render_overlays.isChecked() is False

    def test_config_round_trip(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        original = PipelineConfig(
            mode="multi",
            static_cam=False,
            use_dpvo=True,
            focal_mm=35.0,
            max_persons=5,
            confidence_threshold=0.9,
            target_fps=24.0,
            fbx_naming="UE5 Mannequin",
            use_inpainting=False,
            render_overlays=True,
        )
        w.set_config(original)
        recovered = w.get_config()
        assert recovered.static_cam == original.static_cam
        assert recovered.use_dpvo == original.use_dpvo
        assert recovered.focal_mm == original.focal_mm
        assert recovered.max_persons == original.max_persons
        assert recovered.confidence_threshold == original.confidence_threshold
        assert recovered.target_fps == original.target_fps
        assert recovered.fbx_naming == original.fbx_naming
        assert recovered.use_inpainting == original.use_inpainting
        assert recovered.render_overlays == original.render_overlays


# ---------------------------------------------------------------------------
# Static cam -> DPVO interlock
# ---------------------------------------------------------------------------


class TestStaticCamDpvoInterlock:
    def test_dpvo_disabled_when_static_cam_default(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._static_cam.isChecked() is True
        assert not w._use_dpvo.isEnabled()

    def test_uncheck_static_cam_enables_dpvo(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        w._static_cam.setChecked(False)
        assert w._use_dpvo.isEnabled()

    def test_check_static_cam_disables_and_unchecks_dpvo(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        w._static_cam.setChecked(False)
        w._use_dpvo.setChecked(True)
        w._static_cam.setChecked(True)
        assert w._use_dpvo.isChecked() is False
        assert not w._use_dpvo.isEnabled()


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------


class TestVideoLoading:
    def test_load_video_updates_session(self, qapp, tmp_path):
        video_path = _create_test_video(tmp_path / "test.mp4")
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        w._load_video(str(video_path))

        assert session.video_path == video_path
        assert session.num_frames > 0
        assert w._run_btn.isEnabled()

    def test_load_video_emits_video_loaded(self, qapp, tmp_path):
        video_path = _create_test_video(tmp_path / "test.mp4")
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        received = []
        w.video_loaded.connect(received.append)
        w._load_video(str(video_path))
        assert len(received) == 1

    def test_load_nonexistent_video(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        messages = []
        w.status_message.connect(messages.append)
        w._load_video("/tmp/nonexistent_video.mp4")
        assert not w._run_btn.isEnabled()


# ---------------------------------------------------------------------------
# Running state
# ---------------------------------------------------------------------------


class TestRunningState:
    def test_set_running_true(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        w._video_path = Path("/tmp/test.mp4")
        w._set_running(True)

        assert not w._run_btn.isEnabled()
        assert not w._cancel_btn.isHidden()
        assert not w._progress_bar.isHidden()
        assert not w._browse_btn.isEnabled()
        assert not w._static_cam.isEnabled()
        assert not w._max_persons.isEnabled()
        assert not w._confidence_threshold.isEnabled()
        assert not w._use_inpainting.isEnabled()
        assert not w._render_overlays.isEnabled()

    def test_set_running_false(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        w._video_path = Path("/tmp/test.mp4")
        w._set_running(True)
        w._set_running(False)

        assert w._run_btn.isEnabled()
        assert w._cancel_btn.isHidden()
        assert w._max_persons.isEnabled()
        assert w._confidence_threshold.isEnabled()


# ---------------------------------------------------------------------------
# Worker lifecycle (mocked)
# ---------------------------------------------------------------------------


class TestWorkerLifecycle:
    def test_on_finished_emits_signal(self, qapp, tmp_path):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        w._video_path = Path("/tmp/test.mp4")
        w._set_running(True)

        received = []
        w.pipeline_finished.connect(received.append)

        w._on_finished({
            "output_dir": str(tmp_path),
        })

        assert not w._running
        assert len(received) == 1

    def test_on_error_emits_signal(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        w._set_running(True)

        received = []
        w.pipeline_error.connect(received.append)
        w._on_error("Something broke")

        assert not w._running
        assert len(received) == 1

    def test_on_progress(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        w._progress_bar.show()
        w._on_progress(0.5, "Tracking")
        assert w._progress_bar.value() == 500
        assert w._progress_label.text() == "Tracking"


# ---------------------------------------------------------------------------
# Solve config save/restore
# ---------------------------------------------------------------------------


class TestSolveConfigSaveRestore:
    def test_output_dir_for_video_is_multi_person(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        result = w._output_dir_for_video(Path("/tmp/videos/dance.mp4"))
        assert result == Path("/tmp/GVHMR/outputs/multi_person/dance")

    def test_on_run_saves_config(self, qapp, tmp_path):
        gvhmr_root = tmp_path / "GVHMR"
        gvhmr_root.mkdir()
        session = Session()
        w = MultiPipelineSettings(session, gvhmr_root)
        w._video_path = Path("/tmp/test_video.mp4")
        w._static_cam.setChecked(False)
        w._max_persons.setValue(4)

        with patch("views.pipeline_settings.MultiPersonWorker") as MockWorker:
            mock_instance = MagicMock()
            MockWorker.return_value = mock_instance
            w._on_run()

        config_path = gvhmr_root / "outputs" / "multi_person" / "test_video" / "solve_config.json"
        assert config_path.is_file()

        loaded = PipelineConfig.load(config_path)
        assert loaded.static_cam is False
        assert loaded.max_persons == 4

    def test_load_video_restores_config(self, qapp, tmp_path):
        gvhmr_root = tmp_path / "GVHMR"
        output_dir = gvhmr_root / "outputs" / "multi_person" / "test"
        output_dir.mkdir(parents=True)
        PipelineConfig(
            mode="multi",
            static_cam=False,
            max_persons=3,
            confidence_threshold=0.9,
        ).save(output_dir / "solve_config.json")

        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        w = MultiPipelineSettings(session, gvhmr_root)
        w._load_video(str(video_path))

        assert w._static_cam.isChecked() is False
        assert w._max_persons.value() == 3
        assert w._confidence_threshold.value() == 0.9


# ---------------------------------------------------------------------------
# TrackOverview widget
# ---------------------------------------------------------------------------


class TestTrackOverview:
    """Test TrackOverview widget rendering and interaction."""

    def test_creates_without_error(self, qapp):
        w = TrackOverview()
        assert w is not None

    def test_set_tracks_creates_lanes_and_headers(self, qapp):
        w = TrackOverview()
        tracks = {
            0: np.ones(100) * 0.9,
            1: np.ones(100) * 0.7,
        }
        w.set_tracks(tracks)
        assert len(w._lanes) == 2
        assert len(w._headers) == 2

    def test_set_tracks_headers_have_person_colors(self, qapp):
        w = TrackOverview()
        tracks = {0: np.ones(10) * 0.9}
        w.set_tracks(tracks)
        header = w._headers[0]
        assert PERSON_COLORS[0] in header._label.styleSheet()

    def test_set_current_frame(self, qapp):
        w = TrackOverview()
        tracks = {0: np.ones(100) * 0.8}
        w.set_tracks(tracks)
        # Should not raise
        w.set_current_frame(50)

    def test_clear_on_new_tracks(self, qapp):
        w = TrackOverview()
        w.set_tracks({0: np.ones(10), 1: np.ones(10)})
        assert len(w._lanes) == 2

        w.set_tracks({0: np.ones(5)})
        assert len(w._lanes) == 1
        assert len(w._headers) == 1

    def test_person_clicked_signal(self, qapp):
        w = TrackOverview()
        received = []
        w.person_clicked.connect(lambda pid, f: received.append((pid, f)))
        tracks = {0: np.ones(10)}
        w.set_tracks(tracks)
        # Simulate a click via the underlying timeline view
        w._view.frame_clicked.emit(0, 5)
        assert received == [(0, 5)]

    def test_person_colors_length(self):
        """PERSON_COLORS should have at least 8 colors for multi-person tracking."""
        assert len(PERSON_COLORS) >= 8


# ---------------------------------------------------------------------------
# AppWindow multi-mode integration
# ---------------------------------------------------------------------------


class TestAppWindowMultiMode:
    """Verify AppWindow dock layout for multi-person mode."""

    def test_multi_settings_is_settings_widget(self, qapp):
        from app_window import AppWindow
        window = AppWindow()
        assert isinstance(window._multi_settings, MultiPipelineSettings)

    def test_multi_mode_shows_docks(self, qapp):
        from app_window import AppWindow
        window = AppWindow()
        window._pipeline_dock.set_mode("multi")
        assert not window._person_panel_dock.isHidden()
        assert not window._track_overview_dock.isHidden()

    def test_track_overview_in_dock(self, qapp):
        from app_window import AppWindow
        window = AppWindow()
        assert isinstance(window._track_overview, TrackOverview)

    def test_status_signal_connected(self, qapp):
        from app_window import AppWindow
        window = AppWindow()
        window._multi_settings.status_message.emit("multi test status")
        assert window._status_label.text() == "multi test status"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_test_video(path: Path, frames: int = 5, size: tuple = (64, 48)) -> Path:
    """Create a minimal video file for testing."""
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, size)
    for i in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:, :, 1] = i * 40
        writer.write(frame)
    writer.release()
    return path
