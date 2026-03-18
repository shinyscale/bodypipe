"""Tests for performance capture tab.

Why: The PerfCaptureTab extends SinglePersonTab with hand/face options
and pipeline output settings (FPS, FBX naming, pitch adjust, hand source,
body smoothing, ViTPose face crops). These tests verify the additional UI
controls, correct PipelineConfig generation, settings round-trip, and that
the FullPipelineWorker is used instead of GVHMRWorker.

Multi-stage progress display tests verify the spec requirement that
progress shows "Stage X/N: Name" instead of raw internal labels —
essential for user understanding of long-running pipelines.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pipeline_config import PipelineConfig
from models.session import Session
from views.perf_capture_tab import (
    PerfCaptureTab,
    compute_visible_stages,
    map_stage_label,
)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestPerfCaptureTabConstruction:
    def test_creates_without_error(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert tab is not None

    def test_has_hand_controls(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert hasattr(tab, "_use_hands")
        assert hasattr(tab, "_hand_hybrid")
        assert hasattr(tab, "_hand_smplestx")

    def test_has_face_control(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert hasattr(tab, "_use_face")

    def test_has_pipeline_settings_controls(self, qapp):
        """New Gradio-parity controls: FPS, FBX naming, pitch, smoothing, hand source, vitpose."""
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert hasattr(tab, "_target_fps")
        assert hasattr(tab, "_fbx_naming")
        assert hasattr(tab, "_pitch_adjust")
        assert hasattr(tab, "_body_smooth")
        assert hasattr(tab, "_hand_src_smplestx")
        assert hasattr(tab, "_hand_src_hamer")
        assert hasattr(tab, "_use_vitpose_face")

    def test_run_button_says_run_pipeline(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert tab._run_btn.text() == "Run Pipeline"

    def test_inherits_body_settings(self, qapp):
        """PerfCaptureTab inherits static_cam, use_dpvo, focal_mm from SinglePersonTab."""
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert hasattr(tab, "_static_cam")
        assert hasattr(tab, "_use_dpvo")
        assert hasattr(tab, "_focal_mm")

    def test_inherits_video_input(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert hasattr(tab, "_drop_area")
        assert hasattr(tab, "_browse_btn")


# ---------------------------------------------------------------------------
# Static cam → DPVO auto-disable (inherited from SinglePersonTab)
# ---------------------------------------------------------------------------


class TestStaticCamDpvoInterlock:
    """PerfCaptureTab inherits DPVO interlock from SinglePersonTab."""

    def test_dpvo_disabled_when_static_cam_default(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert tab._static_cam.isChecked() is True
        assert not tab._use_dpvo.isEnabled()

    def test_toggle_static_cam_toggles_dpvo(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab._static_cam.setChecked(False)
        assert tab._use_dpvo.isEnabled()
        tab._static_cam.setChecked(True)
        assert not tab._use_dpvo.isEnabled()
        assert tab._use_dpvo.isChecked() is False


# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------


class TestPerfCaptureDefaults:
    def test_hands_enabled_by_default(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert tab._use_hands.isChecked()

    def test_hybrid_selected_by_default(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert tab._hand_hybrid.isChecked()
        assert not tab._hand_smplestx.isChecked()

    def test_face_disabled_by_default(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert not tab._use_face.isChecked()

    def test_target_fps_default(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert tab._target_fps.value() == 30.0

    def test_fbx_naming_default(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert tab._fbx_naming.currentText() == "Mixamo (Cascadeur)"

    def test_pitch_adjust_default(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert tab._pitch_adjust.value() == 0.0

    def test_body_smooth_default(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert "Moderate" in tab._body_smooth.currentText()

    def test_hand_source_smplestx_default(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert tab._hand_src_smplestx.isChecked()
        assert not tab._hand_src_hamer.isChecked()

    def test_vitpose_face_crops_enabled_by_default(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        assert tab._use_vitpose_face.isChecked()


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPerfCaptureConfig:
    def test_default_config(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        config = tab.get_config()
        assert config.mode == "perf"
        assert config.use_hands is True
        assert config.use_face is False
        assert config.hand_mode == "hybrid"
        assert config.static_cam is True
        assert config.target_fps == 30.0
        assert config.fbx_naming == "Mixamo (Cascadeur)"
        assert config.pitch_adjust == 0.0
        assert config.hand_source == "smplestx"
        assert config.body_smooth_preset == "moderate"
        assert config.use_vitpose_face_crops is True

    def test_smplestx_only_mode(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab._hand_smplestx.setChecked(True)
        config = tab.get_config()
        assert config.hand_mode == "smplestx_only"

    def test_face_enabled(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab._use_face.setChecked(True)
        config = tab.get_config()
        assert config.use_face is True

    def test_hands_disabled(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab._use_hands.setChecked(False)
        config = tab.get_config()
        assert config.use_hands is False

    def test_pipeline_settings_from_ui(self, qapp):
        """Changing UI controls should be reflected in get_config()."""
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab._target_fps.setValue(24.0)
        tab._fbx_naming.setCurrentIndex(1)  # UE5 Mannequin
        tab._pitch_adjust.setValue(15.5)
        tab._body_smooth.setCurrentIndex(2)  # Heavy
        tab._hand_src_hamer.setChecked(True)
        tab._use_vitpose_face.setChecked(False)
        config = tab.get_config()
        assert config.target_fps == 24.0
        assert config.fbx_naming == "UE5 Mannequin"
        assert config.pitch_adjust == 15.5
        assert config.body_smooth_preset == "heavy"
        assert config.hand_source == "hamer"
        assert config.use_vitpose_face_crops is False

    def test_set_config(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab.set_config(PipelineConfig(
            static_cam=False,
            use_dpvo=True,
            focal_mm=85.0,
            use_hands=False,
            use_face=True,
            hand_mode="smplestx_only",
        ))
        assert tab._static_cam.isChecked() is False
        assert tab._use_dpvo.isChecked() is True
        assert tab._focal_mm.value() == 85.0
        assert tab._use_hands.isChecked() is False
        assert tab._use_face.isChecked() is True
        assert tab._hand_smplestx.isChecked() is True

    def test_set_config_pipeline_settings(self, qapp):
        """set_config should restore all pipeline output settings."""
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab.set_config(PipelineConfig(
            target_fps=60.0,
            fbx_naming="UE5 Mannequin",
            pitch_adjust=-10.0,
            hand_source="hamer",
            body_smooth_preset="light",
            use_vitpose_face_crops=False,
        ))
        assert tab._target_fps.value() == 60.0
        assert tab._fbx_naming.currentText() == "UE5 Mannequin"
        assert tab._pitch_adjust.value() == -10.0
        assert tab._hand_src_hamer.isChecked()
        assert tab._body_smooth.currentIndex() == 0  # Light
        assert not tab._use_vitpose_face.isChecked()

    def test_config_round_trip(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        original = PipelineConfig(
            mode="perf",
            static_cam=False,
            use_dpvo=True,
            focal_mm=35.0,
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
        tab.set_config(original)
        recovered = tab.get_config()
        assert recovered.use_hands == original.use_hands
        assert recovered.use_face == original.use_face
        assert recovered.hand_mode == original.hand_mode
        assert recovered.static_cam == original.static_cam
        assert recovered.focal_mm == original.focal_mm
        assert recovered.target_fps == original.target_fps
        assert recovered.fbx_naming == original.fbx_naming
        assert recovered.pitch_adjust == original.pitch_adjust
        assert recovered.hand_source == original.hand_source
        assert recovered.body_smooth_preset == original.body_smooth_preset
        assert recovered.use_vitpose_face_crops == original.use_vitpose_face_crops


# ---------------------------------------------------------------------------
# Solve config save/restore (Gradio parity)
# ---------------------------------------------------------------------------


class TestSolveConfigSaveRestore:
    """Verify perfcap tab uses correct output dir and restores config on load.

    Why: Gradio saves per-video settings to outputs/perfcap/<stem>/solve_config.json.
    When a video is re-entered, settings (hand mode, face, FPS, etc.) are restored
    so users don't need to reconfigure for previously-processed videos.
    """

    def test_output_dir_for_video_is_perfcap(self, qapp):
        """PerfCaptureTab must use perfcap subdirectory, not demo."""
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        result = tab._output_dir_for_video(Path("/tmp/videos/dance.mp4"))
        assert result == Path("/tmp/GVHMR/outputs/perfcap/dance")

    def test_load_video_restores_perf_config(self, qapp, tmp_path):
        """Loading a video should restore all perf capture settings."""
        gvhmr_root = tmp_path / "GVHMR"
        output_dir = gvhmr_root / "outputs" / "perfcap" / "test"
        output_dir.mkdir(parents=True)
        PipelineConfig(
            mode="perf",
            static_cam=False,
            use_dpvo=True,
            focal_mm=50.0,
            use_hands=False,
            use_face=True,
            hand_mode="smplestx_only",
            target_fps=24.0,
            fbx_naming="UE5 Mannequin",
            pitch_adjust=5.5,
            hand_source="hamer",
            body_smooth_preset="heavy",
            use_vitpose_face_crops=False,
        ).save(output_dir / "solve_config.json")

        video_path = _create_test_video(tmp_path / "test.mp4")

        session = Session()
        tab = PerfCaptureTab(session, gvhmr_root)
        tab._load_video(str(video_path))

        assert tab._static_cam.isChecked() is False
        assert tab._use_dpvo.isChecked() is True
        assert tab._focal_mm.value() == 50.0
        assert tab._use_hands.isChecked() is False
        assert tab._use_face.isChecked() is True
        assert tab._hand_smplestx.isChecked() is True
        assert tab._target_fps.value() == 24.0
        assert tab._fbx_naming.currentText() == "UE5 Mannequin"
        assert tab._pitch_adjust.value() == 5.5
        assert tab._hand_src_hamer.isChecked() is True
        assert tab._body_smooth.currentIndex() == 2  # Heavy
        assert not tab._use_vitpose_face.isChecked()


# ---------------------------------------------------------------------------
# Hand mode radio button interaction
# ---------------------------------------------------------------------------


class TestHandModeInteraction:
    def test_radio_buttons_exclusive(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab._hand_smplestx.setChecked(True)
        assert not tab._hand_hybrid.isChecked()
        tab._hand_hybrid.setChecked(True)
        assert not tab._hand_smplestx.isChecked()

    def test_disable_hands_disables_radios(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab._use_hands.setChecked(False)
        assert not tab._hand_hybrid.isEnabled()
        assert not tab._hand_smplestx.isEnabled()

    def test_enable_hands_enables_radios(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab._use_hands.setChecked(False)
        tab._use_hands.setChecked(True)
        assert tab._hand_hybrid.isEnabled()
        assert tab._hand_smplestx.isEnabled()

    def test_hand_source_radios_exclusive(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab._hand_src_hamer.setChecked(True)
        assert not tab._hand_src_smplestx.isChecked()
        tab._hand_src_smplestx.setChecked(True)
        assert not tab._hand_src_hamer.isChecked()

    def test_disable_hands_disables_source_radios(self, qapp):
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab._use_hands.setChecked(False)
        assert not tab._hand_src_smplestx.isEnabled()
        assert not tab._hand_src_hamer.isEnabled()


# ---------------------------------------------------------------------------
# Running state
# ---------------------------------------------------------------------------


class TestPerfCaptureRunningState:
    def test_set_running_disables_new_controls(self, qapp):
        """_set_running(True) should disable all new pipeline settings."""
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab._set_running(True)
        assert not tab._target_fps.isEnabled()
        assert not tab._fbx_naming.isEnabled()
        assert not tab._pitch_adjust.isEnabled()
        assert not tab._body_smooth.isEnabled()
        assert not tab._use_hands.isEnabled()
        assert not tab._use_face.isEnabled()
        assert not tab._use_vitpose_face.isEnabled()

    def test_set_running_false_enables_new_controls(self, qapp):
        """_set_running(False) should re-enable pipeline settings."""
        session = Session()
        tab = PerfCaptureTab(session, Path("/tmp/GVHMR"))
        tab._set_running(True)
        tab._set_running(False)
        assert tab._target_fps.isEnabled()
        assert tab._fbx_naming.isEnabled()
        assert tab._pitch_adjust.isEnabled()
        assert tab._body_smooth.isEnabled()
        assert tab._use_face.isEnabled()
        assert tab._use_vitpose_face.isEnabled()


# ---------------------------------------------------------------------------
# AppWindow integration
# ---------------------------------------------------------------------------


class TestAppWindowPerfTab:
    def test_tab_perf_is_settings_widget(self, qapp):
        from app_window import AppWindow
        from views.pipeline_settings import PerfPipelineSettings
        window = AppWindow()
        assert isinstance(window._tab_perf, PerfPipelineSettings)

    def test_perf_tab_status_connected(self, qapp):
        from app_window import AppWindow
        window = AppWindow()
        window._perf_settings.status_message.emit("perf test status")
        assert window._status_label.text() == "perf test status"


# ---------------------------------------------------------------------------
# Multi-stage progress display — pure functions
# ---------------------------------------------------------------------------


class TestComputeVisibleStages:
    """compute_visible_stages returns the right stage list per config."""

    def test_both_enabled(self):
        assert compute_visible_stages(True, True) == [
            "Body", "Hands", "Face", "Export"
        ]

    def test_hands_only(self):
        assert compute_visible_stages(True, False) == ["Body", "Hands", "Export"]

    def test_face_only(self):
        assert compute_visible_stages(False, True) == ["Body", "Face", "Export"]

    def test_neither(self):
        assert compute_visible_stages(False, False) == ["Body", "Export"]


class TestMapStageLabel:
    """map_stage_label maps internal worker labels to user-visible stages."""

    def test_preprocessing_maps_to_body(self):
        assert map_stage_label("Preprocessing", True, True) == "Body"

    def test_gvhmr_maps_to_body(self):
        assert map_stage_label("GVHMR body solve", True, True) == "Body"

    def test_smplestx_maps_to_hands(self):
        assert map_stage_label("SMPLest-X hand solve", True, True) == "Hands"

    def test_merge_maps_to_hands(self):
        assert map_stage_label("Merging body + hands", True, True) == "Hands"

    def test_face_maps_to_face(self):
        assert map_stage_label("Face pipeline", True, True) == "Face"

    def test_bvh_maps_to_export(self):
        assert map_stage_label("BVH/FBX conversion", True, True) == "Export"

    def test_rendering_maps_to_export(self):
        assert map_stage_label("Rendering", True, True) == "Export"

    def test_unknown_label_returns_none(self):
        assert map_stage_label("some sub-progress message", True, True) is None

    def test_merge_maps_to_export_when_hands_disabled(self):
        """When hands disabled, merge is instant GVHMR-only extraction."""
        assert map_stage_label("Merging body + hands", False, True) == "Export"

    def test_face_returns_none_when_face_disabled(self):
        """When face disabled, swallow the brief 'Face pipeline' emission."""
        assert map_stage_label("Face pipeline", True, False) is None

    def test_smplestx_maps_to_export_when_hands_disabled(self):
        """Edge case: if somehow emitted with hands off, map to Export."""
        assert map_stage_label("SMPLest-X hand solve", False, False) == "Export"


# ---------------------------------------------------------------------------
# Multi-stage progress display — widget behavior
# ---------------------------------------------------------------------------


class TestMultiStageProgress:
    """PerfCaptureTab._on_progress shows stage number/name per spec."""

    def _make_tab(self, qapp):
        session = Session()
        return PerfCaptureTab(session, Path("/tmp/GVHMR"))

    def test_set_running_shows_initial_stage_label(self, qapp):
        tab = self._make_tab(qapp)
        tab._set_running(True)
        assert tab._progress_label.text() == "Stage 1/3: Body"

    def test_set_running_initial_all_enabled(self, qapp):
        tab = self._make_tab(qapp)
        tab._use_face.setChecked(True)
        tab._set_running(True)
        assert tab._progress_label.text() == "Stage 1/4: Body"

    def test_set_running_initial_none_enabled(self, qapp):
        tab = self._make_tab(qapp)
        tab._use_hands.setChecked(False)
        tab._set_running(True)
        assert tab._progress_label.text() == "Stage 1/2: Body"

    def test_on_progress_body_stage(self, qapp):
        tab = self._make_tab(qapp)
        tab._set_running(True)
        tab._on_progress(0.02, "GVHMR body solve")
        assert tab._progress_label.text() == "Stage 1/3: Body"

    def test_on_progress_hands_stage(self, qapp):
        tab = self._make_tab(qapp)
        tab._set_running(True)
        tab._on_progress(0.35, "SMPLest-X hand solve")
        assert tab._progress_label.text() == "Stage 2/3: Hands"

    def test_on_progress_export_stage(self, qapp):
        tab = self._make_tab(qapp)
        tab._set_running(True)
        tab._on_progress(0.72, "BVH/FBX conversion")
        assert tab._progress_label.text() == "Stage 3/3: Export"

    def test_on_progress_face_when_enabled(self, qapp):
        tab = self._make_tab(qapp)
        tab._use_face.setChecked(True)
        tab._set_running(True)
        tab._on_progress(0.52, "Face pipeline")
        assert tab._progress_label.text() == "Stage 3/4: Face"

    def test_on_progress_face_disabled_stays_on_body(self, qapp):
        """When face disabled, 'Face pipeline' emission is swallowed."""
        tab = self._make_tab(qapp)
        tab._set_running(True)
        tab._on_progress(0.02, "GVHMR body solve")
        tab._on_progress(0.52, "Face pipeline")
        # Should still show Body since face was swallowed and no other stage matched
        assert "Body" in tab._progress_label.text()

    def test_on_progress_sub_message_keeps_current_stage(self, qapp):
        """Sub-progress messages (custom text) don't change the stage label."""
        tab = self._make_tab(qapp)
        tab._set_running(True)
        tab._on_progress(0.35, "SMPLest-X hand solve")
        tab._on_progress(0.40, "Processing frame 50/200")
        assert tab._progress_label.text() == "Stage 2/3: Hands"

    def test_on_progress_updates_bar_value(self, qapp):
        tab = self._make_tab(qapp)
        tab._set_running(True)
        tab._on_progress(0.50, "Merging body + hands")
        assert tab._progress_bar.value() == 500

    def test_on_progress_emits_status_message(self, qapp):
        tab = self._make_tab(qapp)
        tab._set_running(True)
        received = []
        tab.status_message.connect(received.append)
        tab._on_progress(0.72, "BVH/FBX conversion")
        assert any("Stage 3/3: Export" in msg for msg in received)

    def test_hands_disabled_merge_maps_to_export(self, qapp):
        """With hands off, 'Merging body + hands' jumps to Export stage."""
        tab = self._make_tab(qapp)
        tab._use_hands.setChecked(False)
        tab._set_running(True)
        tab._on_progress(0.50, "Merging body + hands")
        assert tab._progress_label.text() == "Stage 2/2: Export"

    def test_full_sequence_all_enabled(self, qapp):
        """Walk through complete pipeline with both hands + face enabled."""
        tab = self._make_tab(qapp)
        tab._use_face.setChecked(True)
        tab._set_running(True)

        tab._on_progress(0.00, "Preprocessing")
        assert "1/4: Body" in tab._progress_label.text()

        tab._on_progress(0.02, "GVHMR body solve")
        assert "1/4: Body" in tab._progress_label.text()

        tab._on_progress(0.35, "SMPLest-X hand solve")
        assert "2/4: Hands" in tab._progress_label.text()

        tab._on_progress(0.50, "Merging body + hands")
        assert "2/4: Hands" in tab._progress_label.text()

        tab._on_progress(0.52, "Face pipeline")
        assert "3/4: Face" in tab._progress_label.text()

        tab._on_progress(0.72, "BVH/FBX conversion")
        assert "4/4: Export" in tab._progress_label.text()

        tab._on_progress(0.85, "Rendering")
        assert "4/4: Export" in tab._progress_label.text()

    def test_set_running_false_clears_label(self, qapp):
        """Stopping the pipeline clears the progress label (parent behavior)."""
        tab = self._make_tab(qapp)
        tab._set_running(True)
        tab._on_progress(0.35, "SMPLest-X hand solve")
        tab._set_running(False)
        assert tab._progress_label.text() == ""


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
        frame[:, :, 1] = i * 40
        writer.write(frame)
    writer.release()
    return path
