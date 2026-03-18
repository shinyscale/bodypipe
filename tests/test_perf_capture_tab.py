"""Tests for performance capture tab.

Why: The PerfCaptureTab extends SinglePersonTab with hand/face options
and pipeline output settings (FPS, FBX naming, pitch adjust, hand source,
body smoothing, ViTPose face crops). These tests verify the additional UI
controls, correct PipelineConfig generation, settings round-trip, and that
the FullPipelineWorker is used instead of GVHMRWorker.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pipeline_config import PipelineConfig
from models.session import Session
from views.perf_capture_tab import PerfCaptureTab


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
    def test_second_tab_is_perf_capture(self, qapp):
        from app_window import AppWindow
        window = AppWindow()
        assert isinstance(window._tab_perf, PerfCaptureTab)

    def test_perf_tab_status_connected(self, qapp):
        from app_window import AppWindow
        window = AppWindow()
        window._tab_perf.status_message.emit("perf test status")
        assert window._status_label.text() == "perf test status"
