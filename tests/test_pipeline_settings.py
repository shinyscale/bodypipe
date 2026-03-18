"""Tests for pipeline settings widgets.

Why: Pipeline settings widgets (SinglePipelineSettings, PerfPipelineSettings,
MultiPipelineSettings) are the composable units that will power the dock-based
layout. These tests verify they work correctly as standalone widgets —
independent of the tab shells that embed them. This ensures the dock transition
(Commits 1B/1C) can trust the settings widgets without regression risk.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session
from models.pipeline_config import PipelineConfig
from views.pipeline_settings import (
    SinglePipelineSettings,
    PerfPipelineSettings,
    MultiPipelineSettings,
    _DropArea,
    VIDEO_EXTENSIONS,
    compute_visible_stages,
    map_stage_label,
)


# ===========================================================================
# SinglePipelineSettings
# ===========================================================================


class TestSinglePipelineSettingsConstruction:
    """Verify standalone widget creation and expected child widgets."""

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

    def test_has_drop_area(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._drop_area is not None
        assert w._browse_btn is not None


class TestSinglePipelineSettingsConfig:
    """Verify config round-trip on the standalone widget."""

    def test_default_config(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        config = w.get_config()
        assert config.mode == "single"
        assert config.static_cam is True
        assert config.use_dpvo is False
        assert config.focal_mm == 24.0

    def test_set_config_round_trip(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        original = PipelineConfig(static_cam=False, use_dpvo=True, focal_mm=35.0)
        w.set_config(original)
        recovered = w.get_config()
        assert recovered.static_cam == original.static_cam
        assert recovered.use_dpvo == original.use_dpvo
        assert recovered.focal_mm == original.focal_mm

    def test_static_cam_dpvo_interlock(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._static_cam.isChecked() is True
        assert not w._use_dpvo.isEnabled()
        w._static_cam.setChecked(False)
        assert w._use_dpvo.isEnabled()

    def test_check_static_cam_disables_dpvo(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._static_cam.setChecked(False)
        w._use_dpvo.setChecked(True)
        w._static_cam.setChecked(True)
        assert w._use_dpvo.isChecked() is False
        assert not w._use_dpvo.isEnabled()


class TestSinglePipelineSettingsVideoLoading:
    """Verify video loading on standalone widget."""

    def test_load_video_updates_session(self, qapp, tmp_path):
        video_path = _create_test_video(tmp_path / "test.mp4")
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._load_video(str(video_path))
        assert session.video_path == video_path
        assert session.num_frames > 0
        assert w._run_btn.isEnabled()

    def test_load_video_emits_video_loaded(self, qapp, tmp_path):
        video_path = _create_test_video(tmp_path / "test.mp4")
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        received = []
        w.video_loaded.connect(received.append)
        w._load_video(str(video_path))
        assert len(received) == 1
        assert received[0] == video_path

    def test_load_nonexistent_video(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        messages = []
        w.status_message.connect(messages.append)
        w._load_video("/tmp/nonexistent_video.mp4")
        assert not w._run_btn.isEnabled()
        assert any("not found" in m.lower() for m in messages)


class TestSinglePipelineSettingsRunning:
    """Verify running state transitions."""

    def test_set_running_true(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._video_path = Path("/tmp/test.mp4")
        w._set_running(True)
        assert not w._run_btn.isEnabled()
        assert not w._cancel_btn.isHidden()
        assert not w._progress_bar.isHidden()
        assert not w._browse_btn.isEnabled()

    def test_set_running_false(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._video_path = Path("/tmp/test.mp4")
        w._set_running(True)
        w._set_running(False)
        assert w._run_btn.isEnabled()
        assert w._cancel_btn.isHidden()

    def test_on_progress(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._progress_bar.show()
        w._on_progress(0.5, "ViTPose")
        assert w._progress_bar.value() == 500
        assert w._progress_label.text() == "ViTPose"


class TestSinglePipelineSettingsSignals:
    """Verify signal emission on pipeline events."""

    def test_pipeline_finished_emits_signal(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._video_path = Path("/tmp/test.mp4")
        w._set_running(True)
        received = []
        w.pipeline_finished.connect(received.append)
        w._on_finished({"output_dir": "/tmp/out"})
        assert len(received) == 1
        assert received[0]["output_dir"] == "/tmp/out"

    def test_pipeline_error_emits_signal(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        w._set_running(True)
        received = []
        w.pipeline_error.connect(received.append)
        w._on_error("Something broke")
        assert len(received) == 1
        assert "Something broke" in received[0]


class TestSinglePipelineSettingsConfigPersistence:
    """Verify solve_config.json save/restore on standalone widget."""

    def test_output_dir_for_video(self, qapp):
        session = Session()
        w = SinglePipelineSettings(session, Path("/tmp/GVHMR"))
        result = w._output_dir_for_video(Path("/tmp/videos/dance.mp4"))
        assert result == Path("/tmp/GVHMR/outputs/demo/dance")

    def test_on_run_saves_config(self, qapp, tmp_path):
        gvhmr_root = tmp_path / "GVHMR"
        gvhmr_root.mkdir()
        session = Session()
        w = SinglePipelineSettings(session, gvhmr_root)
        w._video_path = Path("/tmp/test_video.mp4")
        w._static_cam.setChecked(False)
        w._use_dpvo.setChecked(True)
        w._focal_mm.setValue(50.0)

        with patch("views.pipeline_settings.GVHMRWorker") as MockWorker:
            MockWorker.return_value = MagicMock()
            w._on_run()

        config_path = gvhmr_root / "outputs" / "demo" / "test_video" / "solve_config.json"
        assert config_path.is_file()
        loaded = PipelineConfig.load(config_path)
        assert loaded.static_cam is False
        assert loaded.use_dpvo is True

    def test_load_video_restores_config(self, qapp, tmp_path):
        gvhmr_root = tmp_path / "GVHMR"
        output_dir = gvhmr_root / "outputs" / "demo" / "test"
        output_dir.mkdir(parents=True)
        PipelineConfig(static_cam=False, use_dpvo=True, focal_mm=50.0).save(
            output_dir / "solve_config.json"
        )
        video_path = _create_test_video(tmp_path / "test.mp4")
        session = Session()
        w = SinglePipelineSettings(session, gvhmr_root)
        w._load_video(str(video_path))
        assert w._static_cam.isChecked() is False
        assert w._use_dpvo.isChecked() is True
        assert w._focal_mm.value() == 50.0


# ===========================================================================
# PerfPipelineSettings
# ===========================================================================


class TestPerfPipelineSettingsConstruction:
    """Verify PerfPipelineSettings standalone creation."""

    def test_creates_without_error(self, qapp):
        session = Session()
        w = PerfPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w is not None

    def test_has_hand_controls(self, qapp):
        session = Session()
        w = PerfPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._use_hands is not None
        assert w._hand_hybrid is not None
        assert w._hand_smplestx is not None

    def test_has_face_control(self, qapp):
        session = Session()
        w = PerfPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._use_face is not None

    def test_has_pipeline_settings(self, qapp):
        session = Session()
        w = PerfPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._target_fps is not None
        assert w._fbx_naming is not None
        assert w._pitch_adjust is not None
        assert w._body_smooth is not None

    def test_run_button_says_run_pipeline(self, qapp):
        session = Session()
        w = PerfPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._run_btn.text() == "Run Pipeline"


class TestPerfPipelineSettingsConfig:
    """Verify config round-trip on standalone perf settings widget."""

    def test_default_config(self, qapp):
        session = Session()
        w = PerfPipelineSettings(session, Path("/tmp/GVHMR"))
        config = w.get_config()
        assert config.mode == "perf"
        assert config.use_hands is True
        assert config.use_face is False
        assert config.hand_mode == "hybrid"
        assert config.target_fps == 30.0

    def test_config_round_trip(self, qapp):
        session = Session()
        w = PerfPipelineSettings(session, Path("/tmp/GVHMR"))
        original = PipelineConfig(
            mode="perf",
            static_cam=False,
            use_hands=True,
            use_face=True,
            hand_mode="smplestx_only",
            target_fps=24.0,
            fbx_naming="UE5 Mannequin",
            pitch_adjust=5.5,
            hand_source="hamer",
            body_smooth_preset="heavy",
            use_vitpose_face_crops=False,
        )
        w.set_config(original)
        recovered = w.get_config()
        assert recovered.use_hands == original.use_hands
        assert recovered.use_face == original.use_face
        assert recovered.hand_mode == original.hand_mode
        assert recovered.target_fps == original.target_fps
        assert recovered.hand_source == original.hand_source
        assert recovered.body_smooth_preset == original.body_smooth_preset

    def test_output_dir_for_video_is_perfcap(self, qapp):
        session = Session()
        w = PerfPipelineSettings(session, Path("/tmp/GVHMR"))
        result = w._output_dir_for_video(Path("/tmp/videos/dance.mp4"))
        assert result == Path("/tmp/GVHMR/outputs/perfcap/dance")


class TestPerfPipelineSettingsMultiStage:
    """Verify multi-stage progress display on standalone widget."""

    def test_set_running_shows_initial_stage(self, qapp):
        session = Session()
        w = PerfPipelineSettings(session, Path("/tmp/GVHMR"))
        w._set_running(True)
        assert w._progress_label.text() == "Stage 1/3: Body"

    def test_on_progress_hands_stage(self, qapp):
        session = Session()
        w = PerfPipelineSettings(session, Path("/tmp/GVHMR"))
        w._set_running(True)
        w._on_progress(0.35, "SMPLest-X hand solve")
        assert w._progress_label.text() == "Stage 2/3: Hands"

    def test_set_running_disables_hand_face_controls(self, qapp):
        session = Session()
        w = PerfPipelineSettings(session, Path("/tmp/GVHMR"))
        w._set_running(True)
        assert not w._use_hands.isEnabled()
        assert not w._use_face.isEnabled()
        assert not w._target_fps.isEnabled()


# ===========================================================================
# MultiPipelineSettings
# ===========================================================================


class TestMultiPipelineSettingsConstruction:
    """Verify MultiPipelineSettings standalone creation."""

    def test_creates_without_error(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w is not None

    def test_run_button_disabled_initially(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert not w._run_btn.isEnabled()

    def test_has_multi_person_settings(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._max_persons is not None
        assert w._confidence_threshold is not None
        assert w._use_inpainting is not None
        assert w._render_overlays is not None


class TestMultiPipelineSettingsConfig:
    """Verify config round-trip on standalone multi settings widget."""

    def test_default_config(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        config = w.get_config()
        assert config.mode == "multi"
        assert config.max_persons == 8
        assert config.confidence_threshold == 0.5
        assert config.use_inpainting is True

    def test_config_round_trip(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        original = PipelineConfig(
            mode="multi",
            static_cam=False,
            max_persons=4,
            confidence_threshold=0.8,
            target_fps=24.0,
            fbx_naming="UE5 Mannequin",
            use_inpainting=False,
            render_overlays=True,
        )
        w.set_config(original)
        recovered = w.get_config()
        assert recovered.max_persons == original.max_persons
        assert recovered.confidence_threshold == original.confidence_threshold
        assert recovered.use_inpainting == original.use_inpainting
        assert recovered.render_overlays == original.render_overlays

    def test_output_dir_for_video(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        result = w._output_dir_for_video(Path("/tmp/videos/dance.mp4"))
        assert result == Path("/tmp/GVHMR/outputs/multi_person/dance")


class TestMultiPipelineSettingsRunning:
    """Verify running state on standalone multi settings widget."""

    def test_set_running_disables_all_controls(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        w._video_path = Path("/tmp/test.mp4")
        w._set_running(True)
        assert not w._max_persons.isEnabled()
        assert not w._confidence_threshold.isEnabled()
        assert not w._use_inpainting.isEnabled()
        assert not w._render_overlays.isEnabled()
        assert not w._static_cam.isEnabled()

    def test_set_running_false_enables_controls(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        w._video_path = Path("/tmp/test.mp4")
        w._set_running(True)
        w._set_running(False)
        assert w._max_persons.isEnabled()
        assert w._run_btn.isEnabled()


class TestMultiPipelineSettingsVideoLoading:
    """Verify video loading emits signal."""

    def test_load_video_emits_video_loaded(self, qapp, tmp_path):
        video_path = _create_test_video(tmp_path / "test.mp4")
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        received = []
        w.video_loaded.connect(received.append)
        w._load_video(str(video_path))
        assert len(received) == 1

    def test_static_cam_dpvo_interlock(self, qapp):
        session = Session()
        w = MultiPipelineSettings(session, Path("/tmp/GVHMR"))
        assert w._static_cam.isChecked() is True
        assert not w._use_dpvo.isEnabled()
        w._static_cam.setChecked(False)
        assert w._use_dpvo.isEnabled()


# ===========================================================================
# Tab composition tests — verify tabs compose settings correctly
# ===========================================================================


class TestTabComposition:
    """Verify that tab classes correctly compose their settings widgets."""

    def test_single_tab_composes_single_settings(self, qapp):
        from views.single_person_tab import SinglePersonTab
        session = Session()
        tab = SinglePersonTab(session, Path("/tmp/GVHMR"))
        assert isinstance(tab._settings, SinglePipelineSettings)

    def test_perf_tab_composes_perf_settings(self, qapp):
        from views.perf_capture_tab import PerfCaptureTab
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert isinstance(tab._settings, PerfPipelineSettings)

    def test_multi_tab_composes_multi_settings(self, qapp):
        from views.multi_person_tab import MultiPersonTab
        session = Session()
        tab = MultiPersonTab(session, Path("/tmp/GVHMR"))
        assert isinstance(tab._settings, MultiPipelineSettings)

    def test_proxy_reads_settings_attrs(self, qapp):
        """Tab.__getattr__ should proxy attribute reads to settings."""
        from views.single_person_tab import SinglePersonTab
        session = Session()
        tab = SinglePersonTab(session, Path("/tmp/GVHMR"))
        # These attributes live on the settings widget
        assert tab._static_cam is tab._settings._static_cam
        assert tab._run_btn is tab._settings._run_btn

    def test_proxy_writes_video_path(self, qapp):
        """Tab.__setattr__ should forward _video_path to settings."""
        from views.single_person_tab import SinglePersonTab
        session = Session()
        tab = SinglePersonTab(session, Path("/tmp/GVHMR"))
        tab._video_path = Path("/tmp/test.mp4")
        assert tab._settings._video_path == Path("/tmp/test.mp4")

    def test_tab_get_config_delegates(self, qapp):
        from views.single_person_tab import SinglePersonTab
        session = Session()
        tab = SinglePersonTab(session, Path("/tmp/GVHMR"))
        tab._settings._static_cam.setChecked(False)
        config = tab.get_config()
        assert config.static_cam is False

    def test_tab_set_config_delegates(self, qapp):
        from views.single_person_tab import SinglePersonTab
        session = Session()
        tab = SinglePersonTab(session, Path("/tmp/GVHMR"))
        tab.set_config(PipelineConfig(focal_mm=85.0))
        assert tab._settings._focal_mm.value() == 85.0

    def test_signal_forwarding(self, qapp):
        """Tab status_message should forward from settings widget."""
        from views.single_person_tab import SinglePersonTab
        session = Session()
        tab = SinglePersonTab(session, Path("/tmp/GVHMR"))
        received = []
        tab.status_message.connect(received.append)
        tab._settings.status_message.emit("hello")
        assert "hello" in received


# ===========================================================================
# DropArea + re-export tests
# ===========================================================================


class TestDropArea:
    def test_creates(self, qapp):
        area = _DropArea()
        assert area.acceptDrops()
        assert "Drop video" in area.text()

    def test_video_extensions_set(self):
        assert ".mp4" in VIDEO_EXTENSIONS
        assert ".avi" in VIDEO_EXTENSIONS

    def test_importable_from_single_person_tab(self):
        """_DropArea must remain importable from single_person_tab."""
        from views.single_person_tab import _DropArea as DA
        assert DA is _DropArea

    def test_stage_helpers_importable_from_perf_capture_tab(self):
        """compute_visible_stages/map_stage_label must remain importable."""
        from views.perf_capture_tab import compute_visible_stages as cvs
        from views.perf_capture_tab import map_stage_label as msl
        assert cvs is compute_visible_stages
        assert msl is map_stage_label


# ===========================================================================
# Helpers
# ===========================================================================


def _create_test_video(path: Path, frames: int = 5, size: tuple = (64, 48)) -> Path:
    """Create a minimal video file for testing."""
    import cv2
    import numpy as np

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, size)
    for i in range(frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        frame[:, :, 1] = i * 40
        writer.write(frame)
    writer.release()
    return path
