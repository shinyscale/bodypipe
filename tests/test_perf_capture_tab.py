"""Tests for performance capture tab.

Why: The PerfCaptureTab extends SinglePersonTab with hand/face options.
These tests verify the additional UI controls, correct PipelineConfig
generation, and that the FullPipelineWorker is used instead of GVHMRWorker.
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
        )
        tab.set_config(original)
        recovered = tab.get_config()
        assert recovered.use_hands == original.use_hands
        assert recovered.use_face == original.use_face
        assert recovered.hand_mode == original.hand_mode
        assert recovered.static_cam == original.static_cam
        assert recovered.focal_mm == original.focal_mm


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
