"""Tests for dock widget wrappers.

Why: Dock wrappers must have correct objectNames for QMainWindow state
persistence (saveState/restoreState), correct titles for the UI, and typed
properties for accessing inner widgets.  PipelineSettingsDock must switch
modes correctly and emit mode_changed signals.  These tests verify the dock
API before Commit 1C rewires AppWindow to use them.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session
from views.dock_widgets import (
    VideoDock,
    MeshViewportDock,
    PersonPanelDock,
    TrackOverviewDock,
    PipelineSettingsDock,
)


# ===========================================================================
# VideoDock
# ===========================================================================


class TestVideoDock:
    def test_creates_with_correct_title(self, qapp):
        from views.video_player import VideoPlayer

        dock = VideoDock(VideoPlayer())
        assert dock.windowTitle() == "Video"

    def test_object_name(self, qapp):
        from views.video_player import VideoPlayer

        dock = VideoDock(VideoPlayer())
        assert dock.objectName() == "VideoDock"

    def test_property_returns_inner_widget(self, qapp):
        from views.video_player import VideoPlayer

        player = VideoPlayer()
        dock = VideoDock(player)
        assert dock.video_player is player


# ===========================================================================
# MeshViewportDock
# ===========================================================================


class TestMeshViewportDock:
    def test_creates_with_correct_title(self, qapp):
        from views.mesh_viewport import MeshViewport

        vp = MeshViewport(gvhmr_root=Path("/tmp/GVHMR"))
        dock = MeshViewportDock(vp)
        assert dock.windowTitle() == "3D Viewport"

    def test_object_name(self, qapp):
        from views.mesh_viewport import MeshViewport

        dock = MeshViewportDock(MeshViewport(gvhmr_root=Path("/tmp/GVHMR")))
        assert dock.objectName() == "MeshViewportDock"

    def test_property_returns_inner_widget(self, qapp):
        from views.mesh_viewport import MeshViewport

        vp = MeshViewport(gvhmr_root=Path("/tmp/GVHMR"))
        dock = MeshViewportDock(vp)
        assert dock.mesh_viewport is vp


# ===========================================================================
# PersonPanelDock
# ===========================================================================


class TestPersonPanelDock:
    def _make_dock(self):
        from views.identity_inspector import IdentityInspector
        from views.pose_corrector_panel import PoseCorrectorPanel
        from views.person_selector_bar import PersonSelectorBar

        session = Session()
        bar = PersonSelectorBar()
        bar.set_session(session)
        inspector = IdentityInspector(session)
        pc = PoseCorrectorPanel(session=session, gvhmr_root=Path("/tmp/GVHMR"))
        return PersonPanelDock(bar, inspector, pc), bar, inspector, pc

    def test_creates_with_correct_title(self, qapp):
        dock, *_ = self._make_dock()
        assert dock.windowTitle() == "Person"

    def test_object_name(self, qapp):
        dock, *_ = self._make_dock()
        assert dock.objectName() == "PersonPanelDock"

    def test_properties_return_inner_widgets(self, qapp):
        dock, bar, inspector, pc = self._make_dock()
        assert dock.person_bar is bar
        assert dock.identity_inspector is inspector
        assert dock.pose_corrector is pc

    def test_tabs_has_two_tabs(self, qapp):
        dock, *_ = self._make_dock()
        assert dock.tabs.count() == 2


# ===========================================================================
# TrackOverviewDock
# ===========================================================================


class TestTrackOverviewDock:
    def test_creates_with_correct_title(self, qapp):
        from views.track_overview import TrackOverview

        dock = TrackOverviewDock(TrackOverview())
        assert dock.windowTitle() == "Track Overview"

    def test_object_name(self, qapp):
        from views.track_overview import TrackOverview

        dock = TrackOverviewDock(TrackOverview())
        assert dock.objectName() == "TrackOverviewDock"

    def test_property_returns_inner_widget(self, qapp):
        from views.track_overview import TrackOverview

        to = TrackOverview()
        dock = TrackOverviewDock(to)
        assert dock.track_overview is to

    def test_widget_is_track_overview(self, qapp):
        from views.track_overview import TrackOverview

        to = TrackOverview()
        dock = TrackOverviewDock(to)
        assert dock.widget() is to


# ===========================================================================
# PipelineSettingsDock
# ===========================================================================


class TestPipelineSettingsDock:
    def _make_dock(self):
        from views.pipeline_settings import (
            SinglePipelineSettings,
            PerfPipelineSettings,
            MultiPipelineSettings,
        )

        session = Session()
        root = Path("/tmp/GVHMR")
        single = SinglePipelineSettings(session, root)
        perf = PerfPipelineSettings(session, root)
        multi = MultiPipelineSettings(session, root)
        return PipelineSettingsDock(single, perf, multi), single, perf, multi

    def test_creates_with_correct_title(self, qapp):
        dock, *_ = self._make_dock()
        assert dock.windowTitle() == "Pipeline Settings"

    def test_object_name(self, qapp):
        dock, *_ = self._make_dock()
        assert dock.objectName() == "PipelineSettingsDock"

    def test_initial_mode_is_single(self, qapp):
        dock, *_ = self._make_dock()
        assert dock.current_mode == "single"

    def test_initial_settings_is_single(self, qapp):
        dock, single, _, _ = self._make_dock()
        assert dock.current_settings is single

    def test_switch_to_perf(self, qapp):
        dock, _, perf, _ = self._make_dock()
        dock.set_mode("perf")
        assert dock.current_mode == "perf"
        assert dock.current_settings is perf

    def test_switch_to_multi(self, qapp):
        dock, _, _, multi = self._make_dock()
        dock.set_mode("multi")
        assert dock.current_mode == "multi"
        assert dock.current_settings is multi

    def test_mode_changed_signal(self, qapp):
        dock, *_ = self._make_dock()
        received = []
        dock.mode_changed.connect(received.append)
        dock.set_mode("multi")
        assert received == ["multi"]

    def test_mode_changed_signal_round_trip(self, qapp):
        dock, *_ = self._make_dock()
        received = []
        dock.mode_changed.connect(received.append)
        dock.set_mode("perf")
        dock.set_mode("single")
        assert received == ["perf", "single"]

    def test_settings_widget_accessor(self, qapp):
        dock, single, perf, multi = self._make_dock()
        assert dock.settings_widget("single") is single
        assert dock.settings_widget("perf") is perf
        assert dock.settings_widget("multi") is multi

    def test_mode_combo_has_three_items(self, qapp):
        dock, *_ = self._make_dock()
        assert dock.mode_combo.count() == 3

    def test_mode_combo_labels(self, qapp):
        dock, *_ = self._make_dock()
        assert dock.mode_combo.itemText(0) == "GVHMR Body"
        assert dock.mode_combo.itemText(1) == "Performance Capture"
        assert dock.mode_combo.itemText(2) == "Multi-Person"

    def test_set_same_mode_no_duplicate_signal(self, qapp):
        dock, *_ = self._make_dock()
        received = []
        dock.mode_changed.connect(received.append)
        dock.set_mode("single")  # already at single — no change
        assert received == []
