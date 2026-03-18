"""Tests for multi-person capture tab.

Why: The MultiPersonTab is the most complex tab — it orchestrates the
multi-person pipeline, hosts the signal hub for frame/person sync across
sub-panels, and provides the track overview. These tests verify widget
construction, settings round-trip, video loading, running state UI
transitions, worker lifecycle, track overview population, and the
signal hub wiring — all without needing a real GPU or video backend.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session, PersonTrack
from models.pipeline_config import PipelineConfig
from views.multi_person_tab import MultiPersonTab, _TrackOverview, PERSON_COLORS


# ---------------------------------------------------------------------------
# Widget construction
# ---------------------------------------------------------------------------


class TestMultiPersonTabConstruction:
    """Verify that the tab creates correctly and has expected child widgets."""

    def test_creates_without_error(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab is not None

    def test_run_button_disabled_initially(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert not tab._run_btn.isEnabled()

    def test_cancel_button_hidden_initially(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._cancel_btn.isHidden()

    def test_progress_bar_hidden_initially(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._progress_bar.isHidden()

    def test_progress_label_hidden_initially(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._progress_label.isHidden()

    def test_has_pipeline_settings(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._static_cam is not None
        assert tab._use_dpvo is not None
        assert tab._focal_mm is not None
        assert tab._target_fps is not None
        assert tab._fbx_naming is not None

    def test_has_multi_person_settings(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._max_persons is not None
        assert tab._confidence_threshold is not None
        assert tab._use_inpainting is not None
        assert tab._render_overlays is not None

    def test_max_persons_defaults(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._max_persons.value() == 8
        assert tab._max_persons.minimum() == 1
        assert tab._max_persons.maximum() == 20

    def test_confidence_threshold_defaults(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._confidence_threshold.value() == 0.5
        assert tab._confidence_threshold.minimum() == 0.0
        assert tab._confidence_threshold.maximum() == 1.0

    def test_target_fps_defaults(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._target_fps.value() == 30.0
        assert tab._target_fps.minimum() == 1.0
        assert tab._target_fps.maximum() == 120.0

    def test_fbx_naming_defaults(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._fbx_naming.currentText() == "Mixamo (Cascadeur)"
        assert tab._fbx_naming.count() == 2

    def test_use_inpainting_defaults(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._use_inpainting.isChecked() is True

    def test_render_overlays_defaults(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._render_overlays.isChecked() is False

    def test_has_video_player(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._video_player is not None

    def test_has_track_overview(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._track_overview is not None

    def test_has_bottom_splitter(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._bottom_splitter is not None

    def test_has_identity_panel(self, qapp):
        from views.identity_inspector import IdentityInspector
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._identity_panel is not None
        assert isinstance(tab._identity_panel, IdentityInspector)

    def test_has_mesh_viewport(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        from views.mesh_viewport import MeshViewport
        assert isinstance(tab._mesh_viewport, MeshViewport)

    def test_has_vertical_splitter(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._vert_splitter is not None

    def test_has_drop_area(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._drop_area is not None

    def test_has_browse_button(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._browse_btn is not None

    def test_has_video_info_label(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._video_info is not None
        assert tab._video_info.isHidden()

    def test_has_signals(self, qapp):
        """Tab exposes signals for inter-panel communication."""
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert hasattr(tab, "status_message")
        assert hasattr(tab, "log_message")
        assert hasattr(tab, "frame_changed")
        assert hasattr(tab, "person_selected")


# ---------------------------------------------------------------------------
# Settings / PipelineConfig
# ---------------------------------------------------------------------------


class TestMultiPersonSettings:
    """Verify round-trip between UI controls and PipelineConfig."""

    def test_default_config(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        config = tab.get_config()
        assert config.mode == "multi"
        assert config.static_cam is True
        assert config.use_dpvo is False
        assert config.focal_mm == 24.0
        assert config.max_persons == 8
        assert config.confidence_threshold == 0.5
        assert config.target_fps == 30.0
        assert config.fbx_naming == "Mixamo (Cascadeur)"
        assert config.render_overlays is False
        assert config.use_inpainting is True

    def test_changed_config(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._static_cam.setChecked(False)
        tab._use_dpvo.setChecked(True)
        tab._focal_mm.setValue(50.0)
        tab._max_persons.setValue(4)
        tab._confidence_threshold.setValue(0.7)
        tab._target_fps.setValue(24.0)
        tab._fbx_naming.setCurrentText("UE5 Mannequin")
        tab._render_overlays.setChecked(True)
        tab._use_inpainting.setChecked(False)

        config = tab.get_config()
        assert config.static_cam is False
        assert config.use_dpvo is True
        assert config.focal_mm == 50.0
        assert config.max_persons == 4
        assert config.confidence_threshold == 0.7
        assert config.target_fps == 24.0
        assert config.fbx_naming == "UE5 Mannequin"
        assert config.render_overlays is True
        assert config.use_inpainting is False

    def test_set_config(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab.set_config(PipelineConfig(
            static_cam=False,
            use_dpvo=True,
            focal_mm=85.0,
            max_persons=3,
            confidence_threshold=0.8,
            target_fps=60.0,
            fbx_naming="UE5 Mannequin",
            render_overlays=True,
            use_inpainting=False,
        ))
        assert tab._static_cam.isChecked() is False
        assert tab._use_dpvo.isChecked() is True
        assert tab._focal_mm.value() == 85.0
        assert tab._max_persons.value() == 3
        assert tab._confidence_threshold.value() == 0.8
        assert tab._target_fps.value() == 60.0
        assert tab._fbx_naming.currentText() == "UE5 Mannequin"
        assert tab._render_overlays.isChecked() is True
        assert tab._use_inpainting.isChecked() is False

    def test_config_round_trip(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        original = PipelineConfig(
            mode="multi",
            static_cam=False,
            use_dpvo=True,
            focal_mm=35.0,
            max_persons=5,
            confidence_threshold=0.65,
            target_fps=24.0,
            fbx_naming="UE5 Mannequin",
            render_overlays=True,
            use_inpainting=False,
        )
        tab.set_config(original)
        recovered = tab.get_config()
        assert recovered.mode == "multi"
        assert recovered.static_cam == original.static_cam
        assert recovered.use_dpvo == original.use_dpvo
        assert recovered.focal_mm == original.focal_mm
        assert recovered.max_persons == original.max_persons
        assert recovered.confidence_threshold == original.confidence_threshold
        assert recovered.target_fps == original.target_fps
        assert recovered.fbx_naming == original.fbx_naming
        assert recovered.render_overlays == original.render_overlays
        assert recovered.use_inpainting == original.use_inpainting


# ---------------------------------------------------------------------------
# Solve config save/restore (Gradio parity)
# ---------------------------------------------------------------------------


class TestSolveConfigSaveRestore:
    """Verify solve_config.json is saved on run and restored on video load.

    Why: Gradio saves per-video settings to outputs/multi_person/<stem>/solve_config.json.
    When a video is re-entered, settings are restored so users don't need
    to reconfigure for previously-processed videos.
    """

    def test_output_dir_for_video(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        result = tab._output_dir_for_video(Path("/tmp/videos/dance.mp4"))
        assert result == Path("/tmp/GVHMR/outputs/multi_person/dance")

    def test_on_run_saves_config(self, qapp, tmp_path):
        """_on_run should save solve_config.json to the output directory."""
        gvhmr_root = tmp_path / "GVHMR"
        gvhmr_root.mkdir()
        session = Session()
        tab = MultiPersonTab(session, gvhmr_root)
        tab._video_path = Path("/tmp/test_video.mp4")
        tab._static_cam.setChecked(False)
        tab._use_dpvo.setChecked(True)
        tab._focal_mm.setValue(50.0)
        tab._max_persons.setValue(4)
        tab._use_inpainting.setChecked(False)

        with patch("views.pipeline_settings.MultiPersonWorker") as MockWorker:
            mock_instance = MagicMock()
            MockWorker.return_value = mock_instance
            tab._on_run()

        config_path = gvhmr_root / "outputs" / "multi_person" / "test_video" / "solve_config.json"
        assert config_path.is_file()

        loaded = PipelineConfig.load(config_path)
        assert loaded.static_cam is False
        assert loaded.use_dpvo is True
        assert loaded.focal_mm == 50.0
        assert loaded.max_persons == 4
        assert loaded.use_inpainting is False

    def test_load_video_restores_config(self, qapp, tmp_path):
        """Loading a video should restore settings from solve_config.json."""
        gvhmr_root = tmp_path / "GVHMR"
        output_dir = gvhmr_root / "outputs" / "multi_person" / "test"
        output_dir.mkdir(parents=True)
        PipelineConfig(
            mode="multi",
            static_cam=False,
            use_dpvo=True,
            focal_mm=50.0,
            max_persons=4,
            confidence_threshold=0.8,
            target_fps=24.0,
            fbx_naming="UE5 Mannequin",
            render_overlays=True,
            use_inpainting=False,
        ).save(output_dir / "solve_config.json")

        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        tab = MultiPersonTab(session, gvhmr_root)
        tab._load_video(str(video_path))

        assert tab._static_cam.isChecked() is False
        assert tab._use_dpvo.isChecked() is True
        assert tab._focal_mm.value() == 50.0
        assert tab._max_persons.value() == 4
        assert tab._confidence_threshold.value() == 0.8
        assert tab._target_fps.value() == 24.0
        assert tab._fbx_naming.currentText() == "UE5 Mannequin"
        assert tab._render_overlays.isChecked() is True
        assert tab._use_inpainting.isChecked() is False

    def test_load_video_no_config_keeps_defaults(self, qapp, tmp_path):
        """Without solve_config.json, default settings are preserved."""
        gvhmr_root = tmp_path / "GVHMR"
        gvhmr_root.mkdir()

        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        tab = MultiPersonTab(session, gvhmr_root)
        tab._load_video(str(video_path))

        assert tab._static_cam.isChecked() is True
        assert tab._max_persons.value() == 8
        assert tab._use_inpainting.isChecked() is True

    def test_restore_emits_log(self, qapp, tmp_path):
        """Restoring config should emit a log message."""
        gvhmr_root = tmp_path / "GVHMR"
        output_dir = gvhmr_root / "outputs" / "multi_person" / "test"
        output_dir.mkdir(parents=True)
        PipelineConfig(static_cam=False).save(output_dir / "solve_config.json")

        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        tab = MultiPersonTab(session, gvhmr_root)
        logs = []
        tab.log_message.connect(lambda text, level: logs.append((text, level)))
        tab._load_video(str(video_path))

        assert any("restored" in text.lower() for text, _ in logs)


# ---------------------------------------------------------------------------
# Static cam → DPVO auto-disable
# ---------------------------------------------------------------------------


class TestStaticCamDpvoInterlock:
    """DPVO must be disabled and unchecked when static camera is enabled."""

    def test_dpvo_disabled_by_default(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._static_cam.isChecked() is True
        assert tab._use_dpvo.isChecked() is False
        assert not tab._use_dpvo.isEnabled()

    def test_uncheck_static_cam_enables_dpvo(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._static_cam.setChecked(False)
        assert tab._use_dpvo.isEnabled()

    def test_check_static_cam_disables_and_unchecks_dpvo(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._static_cam.setChecked(False)
        tab._use_dpvo.setChecked(True)
        tab._static_cam.setChecked(True)
        assert tab._use_dpvo.isChecked() is False
        assert not tab._use_dpvo.isEnabled()

    def test_dpvo_stays_disabled_after_run_with_static_cam(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._video_path = Path("/tmp/test.mp4")
        tab._set_running(True)
        tab._set_running(False)
        assert not tab._use_dpvo.isEnabled()


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------


class TestVideoLoading:
    """Verify that loading a video updates session state and enables run."""

    def test_load_video_updates_session(self, qapp, tmp_path):
        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._load_video(str(video_path))

        assert session.video_path == video_path
        assert session.num_frames > 0
        assert session.img_width > 0
        assert session.img_height > 0
        assert tab._run_btn.isEnabled()

    def test_load_video_shows_info(self, qapp, tmp_path):
        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._load_video(str(video_path))

        assert not tab._video_info.isHidden()
        assert "test.mp4" in tab._video_info.text()

    def test_load_video_updates_drop_area(self, qapp, tmp_path):
        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._load_video(str(video_path))

        assert "test.mp4" in tab._drop_area.text()

    def test_load_video_emits_status(self, qapp, tmp_path):
        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))

        messages = []
        tab.status_message.connect(messages.append)
        tab._load_video(str(video_path))

        assert any("loaded" in m.lower() for m in messages)

    def test_load_nonexistent_video(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))

        messages = []
        tab.status_message.connect(messages.append)
        tab._load_video("/tmp/nonexistent_video.mp4")

        assert not tab._run_btn.isEnabled()
        assert any("not found" in m.lower() for m in messages)

    def test_load_invalid_file(self, qapp, tmp_path):
        bad_file = tmp_path / "bad.mp4"
        bad_file.write_text("not a video")

        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))

        messages = []
        tab.status_message.connect(messages.append)
        tab._load_video(str(bad_file))

        assert not tab._run_btn.isEnabled()


# ---------------------------------------------------------------------------
# Running state
# ---------------------------------------------------------------------------


class TestRunningState:
    """Verify UI state transitions when running/idle.

    Note: We use ``not widget.isHidden()`` instead of ``widget.isVisible()``
    because Qt's isVisible() requires the entire widget hierarchy to be shown.
    isHidden() correctly reflects the widget's own visibility flag.
    """

    def test_set_running_true(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._video_path = Path("/tmp/test.mp4")
        tab._set_running(True)

        assert not tab._run_btn.isEnabled()
        assert not tab._cancel_btn.isHidden()
        assert not tab._progress_bar.isHidden()
        assert not tab._progress_label.isHidden()
        assert not tab._browse_btn.isEnabled()
        assert not tab._static_cam.isEnabled()
        assert not tab._use_dpvo.isEnabled()
        assert not tab._focal_mm.isEnabled()
        assert not tab._max_persons.isEnabled()
        assert not tab._confidence_threshold.isEnabled()
        assert not tab._target_fps.isEnabled()
        assert not tab._fbx_naming.isEnabled()
        assert not tab._render_overlays.isEnabled()
        assert not tab._use_inpainting.isEnabled()

    def test_set_running_false(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._video_path = Path("/tmp/test.mp4")
        tab._set_running(True)
        tab._set_running(False)

        assert tab._run_btn.isEnabled()
        assert tab._cancel_btn.isHidden()
        assert tab._progress_bar.isHidden()
        assert tab._progress_label.isHidden()
        assert tab._browse_btn.isEnabled()
        assert tab._static_cam.isEnabled()
        # DPVO stays disabled when static_cam is checked (default True)
        assert not tab._use_dpvo.isEnabled()
        assert tab._focal_mm.isEnabled()
        assert tab._max_persons.isEnabled()
        assert tab._confidence_threshold.isEnabled()
        assert tab._target_fps.isEnabled()
        assert tab._fbx_naming.isEnabled()
        assert tab._render_overlays.isEnabled()
        assert tab._use_inpainting.isEnabled()

    def test_set_running_false_resets_progress(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._progress_bar.setValue(500)
        tab._progress_label.setText("Test stage")
        tab._set_running(False)

        assert tab._progress_bar.value() == 0
        assert tab._progress_label.text() == ""

    def test_run_button_disabled_without_video(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._set_running(False)
        assert not tab._run_btn.isEnabled()


# ---------------------------------------------------------------------------
# Progress updates
# ---------------------------------------------------------------------------


class TestProgressUpdates:
    def test_on_progress(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._progress_bar.show()
        tab._on_progress(0.5, "Detection")
        assert tab._progress_bar.value() == 500
        assert tab._progress_label.text() == "Detection"

    def test_on_progress_full(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._on_progress(1.0, "Done")
        assert tab._progress_bar.value() == 1000

    def test_on_progress_emits_status(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))

        messages = []
        tab.status_message.connect(messages.append)
        tab._on_progress(0.3, "Tracking")

        assert any("tracking" in m.lower() for m in messages)
        assert any("30%" in m for m in messages)


# ---------------------------------------------------------------------------
# Worker lifecycle (mocked)
# ---------------------------------------------------------------------------


class TestWorkerLifecycle:
    def test_on_finished(self, qapp, tmp_path):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._video_path = Path("/tmp/test.mp4")
        tab._set_running(True)

        tab._on_finished({"output_dir": str(tmp_path)})

        assert not tab._running
        assert session.output_dir == tmp_path

    def test_on_finished_clears_worker(self, qapp, tmp_path):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._video_path = Path("/tmp/test.mp4")
        tab._worker = MagicMock()
        tab._set_running(True)

        tab._on_finished({"output_dir": str(tmp_path)})

        assert tab._worker is None

    def test_on_error(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._video_path = Path("/tmp/test.mp4")
        tab._set_running(True)

        messages = []
        tab.status_message.connect(messages.append)
        tab._on_error("Something broke")

        assert not tab._running
        assert any("error" in m.lower() for m in messages)

    def test_on_error_clears_worker(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._worker = MagicMock()
        tab._set_running(True)

        tab._on_error("Something broke")

        assert tab._worker is None

    def test_on_cancel_emits_status(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._worker = MagicMock()
        tab._set_running(True)

        messages = []
        tab.status_message.connect(messages.append)
        tab._on_cancel()

        assert any("cancel" in m.lower() for m in messages)
        assert not tab._running

    def test_on_cancel_calls_worker_cancel(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        mock_worker = MagicMock()
        tab._worker = mock_worker
        tab._set_running(True)

        tab._on_cancel()

        mock_worker.cancel.assert_called_once()

    def test_on_log_line_emits_log_message(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))

        messages = []
        tab.log_message.connect(lambda text, level: messages.append((text, level)))
        tab._on_log_line("test log output")

        assert len(messages) == 1
        assert messages[0] == ("test log output", "info")


# ---------------------------------------------------------------------------
# Track Overview widget
# ---------------------------------------------------------------------------


class TestTrackOverview:
    def test_creates(self, qapp):
        overview = _TrackOverview()
        assert overview is not None

    def test_set_tracks(self, qapp):
        overview = _TrackOverview()
        tracks = {
            0: np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
            1: np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
        }
        overview.set_tracks(tracks)
        assert len(overview._timelines) == 2
        assert len(overview._labels) == 2
        assert 0 in overview._timelines
        assert 1 in overview._timelines

    def test_set_tracks_replaces_previous(self, qapp):
        overview = _TrackOverview()
        tracks1 = {0: np.ones(5)}
        overview.set_tracks(tracks1)
        assert len(overview._timelines) == 1

        tracks2 = {0: np.ones(5), 1: np.ones(5), 2: np.ones(5)}
        overview.set_tracks(tracks2)
        assert len(overview._timelines) == 3

    def test_set_current_frame(self, qapp):
        overview = _TrackOverview()
        tracks = {0: np.ones(10)}
        overview.set_tracks(tracks)

        # Should not raise
        overview.set_current_frame(5)

    def test_person_clicked_signal(self, qapp):
        overview = _TrackOverview()
        tracks = {0: np.ones(10)}
        overview.set_tracks(tracks)

        received = []
        overview.person_clicked.connect(lambda pid, f: received.append((pid, f)))

        # Simulate a click on the timeline
        overview._timelines[0].frame_clicked.emit(3)

        assert len(received) == 1
        assert received[0] == (0, 3)

    def test_labels_use_person_colors(self, qapp):
        overview = _TrackOverview()
        tracks = {0: np.ones(5), 1: np.ones(5)}
        overview.set_tracks(tracks)

        for pid, label in overview._labels.items():
            expected_color = PERSON_COLORS[pid % len(PERSON_COLORS)]
            assert expected_color in label.styleSheet()


# ---------------------------------------------------------------------------
# Track population from session
# ---------------------------------------------------------------------------


class TestTrackPopulation:
    """Verify that _populate_tracks correctly builds track overview from session."""

    def test_populate_with_confidences(self, qapp):
        session = Session()
        session.num_frames = 5
        session.person_tracks = {
            0: PersonTrack(person_id=0, confidences=[0.9, 0.8, 0.7, 0.6, 0.5]),
            1: PersonTrack(person_id=1, confidences=[0.5, 0.6, 0.7, 0.8, 0.9]),
        }

        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._populate_tracks()

        assert len(tab._track_overview._timelines) == 2

    def test_populate_without_confidences_uses_placeholder(self, qapp):
        session = Session()
        session.num_frames = 10
        session.person_tracks = {
            0: PersonTrack(person_id=0, confidences=None),
        }

        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._populate_tracks()

        assert len(tab._track_overview._timelines) == 1

    def test_populate_empty_tracks(self, qapp):
        session = Session()
        session.person_tracks = {}

        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._populate_tracks()

        assert len(tab._track_overview._timelines) == 0


# ---------------------------------------------------------------------------
# Signal hub wiring
# ---------------------------------------------------------------------------


class TestSignalHub:
    """Verify frame_changed and person_selected signal propagation."""

    def test_frame_changed_updates_session(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))

        tab._on_frame_changed(42)

        assert session.current_frame == 42

    def test_frame_changed_emits_signal(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))

        received = []
        tab.frame_changed.connect(received.append)
        tab._on_frame_changed(10)

        assert received == [10]

    def test_track_clicked_updates_session(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))

        tab._on_track_clicked(2, 15)

        assert session.selected_person == 2

    def test_track_clicked_emits_person_selected(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))

        received = []
        tab.person_selected.connect(received.append)
        tab._on_track_clicked(3, 20)

        assert received == [3]

    def test_track_clicked_emits_status(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))

        messages = []
        tab.status_message.connect(messages.append)
        tab._on_track_clicked(1, 50)

        assert any("person 1" in m.lower() for m in messages)
        assert any("50" in m for m in messages)


# ---------------------------------------------------------------------------
# Viewport switching
# ---------------------------------------------------------------------------


class TestViewportSwitching:
    """Verify the toolbar toggle between Video and 3D Mesh viewports.

    Why: The multi-person spec requires the main viewport to switch between
    video+bbox overlay and 3D mesh view via toolbar buttons. This is critical
    for users who need to inspect both 2D tracking and 3D pose simultaneously.
    """

    def test_has_viewport_stack(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        from PySide6.QtWidgets import QStackedWidget
        assert isinstance(tab._viewport_stack, QStackedWidget)

    def test_viewport_stack_property(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab.viewport_stack is tab._viewport_stack

    def test_video_mode_is_default(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._viewport_stack.currentIndex() == 0
        assert tab._video_mode_btn.isChecked()
        assert not tab._mesh_mode_btn.isChecked()

    def test_has_toolbar_buttons(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._video_mode_btn is not None
        assert tab._mesh_mode_btn is not None
        assert tab._video_mode_btn.text() == "Video"
        assert tab._mesh_mode_btn.text() == "3D Mesh"

    def test_toolbar_buttons_are_checkable(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._video_mode_btn.isCheckable()
        assert tab._mesh_mode_btn.isCheckable()

    def test_switch_to_mesh(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._switch_to_mesh()
        assert tab._viewport_stack.currentIndex() == 1
        assert tab._mesh_mode_btn.isChecked()
        assert not tab._video_mode_btn.isChecked()

    def test_switch_to_video(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._switch_to_mesh()
        tab._switch_to_video()
        assert tab._viewport_stack.currentIndex() == 0
        assert tab._video_mode_btn.isChecked()
        assert not tab._mesh_mode_btn.isChecked()

    def test_switch_to_mesh_and_back(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._switch_to_mesh()
        assert tab._viewport_stack.currentIndex() == 1
        tab._switch_to_video()
        assert tab._viewport_stack.currentIndex() == 0

    def test_has_main_mesh_viewport(self, qapp):
        """Main viewport has its own MeshViewport instance (separate from pose corrector)."""
        from views.mesh_viewport import MeshViewport
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert isinstance(tab._main_mesh_viewport, MeshViewport)
        # Must be a separate instance from the pose corrector's viewport
        assert tab._main_mesh_viewport is not tab._mesh_viewport

    def test_main_mesh_viewport_has_session(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._main_mesh_viewport._session is session

    def test_viewport_stack_contains_both_widgets(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert tab._viewport_stack.count() == 2
        assert tab._viewport_stack.widget(0) is tab._video_player
        assert tab._viewport_stack.widget(1) is tab._main_mesh_viewport


# ---------------------------------------------------------------------------
# Signal wiring for main mesh viewport
# ---------------------------------------------------------------------------


class TestMeshViewportSignalWiring:
    """Verify that the main mesh viewport receives frame/person/joint signals.

    Why: The spec requires frame sync, person selection, and joint clicking
    to propagate through the main viewport's MeshViewport — not just via
    the PoseCorrectorPanel's embedded viewport.
    """

    def test_frame_changed_propagates_to_main_mesh(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._on_frame_changed(25)
        assert tab._main_mesh_viewport._current_frame == 25

    def test_track_clicked_sets_person_on_main_mesh(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._on_track_clicked(3, 10)
        assert tab._main_mesh_viewport._person_id == 3

    def test_identity_person_changed_sets_person_on_main_mesh(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._on_identity_person_changed(5)
        assert tab._main_mesh_viewport._person_id == 5

    def test_joint_clicked_wired_to_pose_corrector(self, qapp):
        """main mesh viewport joint_clicked → pose_corrector.set_joint."""
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        # Emit joint_clicked from the main mesh viewport
        tab._main_mesh_viewport.joint_clicked.emit(7)
        # The pose corrector should have synced its joint selection
        assert tab._pose_corrector._current_joint == 7


# ---------------------------------------------------------------------------
# Video frame composite wiring
# ---------------------------------------------------------------------------


class TestVideoFrameCompositeWiring:
    """Verify that MultiPersonTab passes video frames to main mesh viewport.

    Why: The spec requires in-camera 3D viewport to composite the mesh
    over the video frame. The tab must wire frame data from the video
    player to the mesh viewport so the background shows the actual footage.
    """

    def test_frame_change_passes_frame_to_mesh_viewport(self, qapp):
        """_on_frame_changed should pass raw frame to main_mesh_viewport."""
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        # Mock the video player's get_raw_frame to return a known frame
        dummy_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        tab._video_player.get_raw_frame = MagicMock(return_value=dummy_frame)
        tab._on_frame_changed(5)
        tab._video_player.get_raw_frame.assert_called_with(5)
        assert tab._main_mesh_viewport._video_frame is not None
        np.testing.assert_array_equal(
            tab._main_mesh_viewport._video_frame, dummy_frame
        )

    def test_frame_change_none_frame_clears_viewport(self, qapp):
        """When no video is loaded, get_raw_frame returns None — viewport
        should clear its video frame."""
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        # Set a frame first
        tab._main_mesh_viewport._video_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        tab._video_player.get_raw_frame = MagicMock(return_value=None)
        tab._on_frame_changed(0)
        assert tab._main_mesh_viewport._video_frame is None

    def test_switch_to_mesh_passes_current_frame(self, qapp):
        """Switching to mesh mode should set the video frame from current frame."""
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        dummy_frame = np.ones((100, 200, 3), dtype=np.uint8) * 128
        tab._video_player.get_raw_frame = MagicMock(return_value=dummy_frame)
        tab._switch_to_mesh()
        assert tab._main_mesh_viewport._video_frame is not None

    def test_mesh_viewport_has_set_video_frame_method(self, qapp):
        """Main mesh viewport must expose set_video_frame."""
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert hasattr(tab._main_mesh_viewport, "set_video_frame")
        assert callable(tab._main_mesh_viewport.set_video_frame)


# ---------------------------------------------------------------------------
# PERSON_COLORS constant
# ---------------------------------------------------------------------------


class TestPersonColors:
    def test_has_8_colors(self, qapp):
        assert len(PERSON_COLORS) == 8

    def test_all_hex_colors(self, qapp):
        for color in PERSON_COLORS:
            assert color.startswith("#")
            assert len(color) == 7


# ---------------------------------------------------------------------------
# AppWindow integration
# ---------------------------------------------------------------------------


class TestAppWindowIntegration:
    """Verify AppWindow uses MultiPipelineSettings via _tab_multi property."""

    def test_tab_multi_is_settings_widget(self, qapp):
        from app_window import AppWindow
        from views.pipeline_settings import MultiPipelineSettings
        window = AppWindow()
        assert isinstance(window._tab_multi, MultiPipelineSettings)

    def test_multi_tab_status_connected(self, qapp):
        from app_window import AppWindow
        window = AppWindow()
        window._multi_settings.status_message.emit("multi test status")
        assert window._status_label.text() == "multi test status"

    def test_mode_combo_has_multi(self, qapp):
        from app_window import AppWindow
        window = AppWindow()
        assert window._pipeline_dock.mode_combo.itemText(2) == "Multi-Person"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Splitter layout persistence
# ---------------------------------------------------------------------------


class TestSplitterPersistence:
    """Verify that splitter sizes persist across tab instances via QSettings.

    Why: The multi-person spec requires 'Splitter layout persists across
    sessions'. Users resize the identity inspector / pose corrector / viewport
    panels and expect them to stay put on next launch. Without persistence,
    every session starts with default ratios — especially frustrating when
    the default doesn't match the user's monitor or workflow.
    """

    def test_has_settings_instance(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        from PySide6.QtCore import QSettings
        assert isinstance(tab._qsettings, QSettings)

    def test_has_main_splitter_as_attribute(self, qapp):
        """main_splitter must be an instance attr for save/restore access."""
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        from PySide6.QtWidgets import QSplitter
        assert isinstance(tab._main_splitter, QSplitter)

    def test_save_splitter_state_writes_settings(self, qapp):
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        tab._save_splitter_state()

        # All three keys should be set
        assert tab._qsettings.value("multi_person/main_splitter") is not None
        assert tab._qsettings.value("multi_person/vert_splitter") is not None
        assert tab._qsettings.value("multi_person/bottom_splitter") is not None

    def test_restore_splitter_state_round_trip(self, qapp):
        """Save → create new tab → verify state is restored."""
        session = Session()
        tab1 = MultiPersonTab(session, Path("/tmp/GVHMR"))

        # Set specific sizes on vert_splitter (viewport : panels)
        tab1._vert_splitter.setSizes([400, 200])
        tab1._bottom_splitter.setSizes([300, 300])
        tab1._save_splitter_state()

        saved_vert = tab1._vert_splitter.saveState()
        saved_bottom = tab1._bottom_splitter.saveState()

        # New tab should restore from QSettings
        tab2 = MultiPersonTab(session, Path("/tmp/GVHMR"))

        assert tab2._vert_splitter.saveState() == saved_vert
        assert tab2._bottom_splitter.saveState() == saved_bottom

    def test_restore_handles_missing_settings(self, qapp):
        """Tab should construct fine when no prior splitter state exists."""
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        # Clear any saved state
        tab._qsettings.remove("multi_person/main_splitter")
        tab._qsettings.remove("multi_person/vert_splitter")
        tab._qsettings.remove("multi_person/bottom_splitter")
        # Restore should not raise
        tab._restore_splitter_state()

    def test_splitter_moved_triggers_save(self, qapp):
        """Moving a splitter should auto-save state."""
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))

        # Clear saved state
        tab._qsettings.remove("multi_person/vert_splitter")

        # Emit splitterMoved signal (pos, index)
        tab._vert_splitter.splitterMoved.emit(200, 0)

        # State should now be saved
        assert tab._qsettings.value("multi_person/vert_splitter") is not None

    def test_all_three_splitters_connected(self, qapp):
        """All three splitters must have splitterMoved wired to save."""
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))

        # Clear all saved state
        for key in ("main_splitter", "vert_splitter", "bottom_splitter"):
            tab._qsettings.remove(f"multi_person/{key}")

        # Fire splitterMoved on each
        tab._main_splitter.splitterMoved.emit(100, 0)
        assert tab._qsettings.value("multi_person/main_splitter") is not None

        tab._qsettings.remove("multi_person/vert_splitter")
        tab._vert_splitter.splitterMoved.emit(200, 0)
        assert tab._qsettings.value("multi_person/vert_splitter") is not None

        tab._qsettings.remove("multi_person/bottom_splitter")
        tab._bottom_splitter.splitterMoved.emit(150, 0)
        assert tab._qsettings.value("multi_person/bottom_splitter") is not None


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
