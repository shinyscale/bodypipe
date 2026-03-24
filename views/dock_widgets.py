"""Dock widget wrappers for the dockable layout.

Why: Thin QDockWidget subclasses isolate the tab-to-dock migration (Commit 1C)
from the dock API.  Each dock wraps a single inner widget, sets a stable
objectName for QMainWindow.saveState()/restoreState() persistence, and exposes
the inner widget via a typed property.  PipelineSettingsDock adds a mode
QComboBox that switches between the 3 settings panels in a QStackedWidget,
emitting mode_changed so AppWindow can show/hide multi-only docks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Signal

if TYPE_CHECKING:
    from views.identity_inspector import IdentityInspector
    from views.mesh_viewport import MeshViewport
    from views.session_library import SessionLibrary
    from views.track_overview import TrackOverview
    from views.pipeline_settings import (
        MultiPipelineSettings,
        PerfPipelineSettings,
        SinglePipelineSettings,
    )
    from views.pose_corrector_panel import PoseCorrectorPanel
    from views.video_player import VideoPlayer


class VideoDock(QDockWidget):
    """Dock wrapping a VideoPlayer widget."""

    def __init__(self, video_player: VideoPlayer, parent: QWidget | None = None):
        super().__init__("Video", parent)
        self.setObjectName("VideoDock")
        self._video_player = video_player
        self.setWidget(video_player)

    @property
    def video_player(self) -> VideoPlayer:
        return self._video_player


class MeshViewportDock(QDockWidget):
    """Dock wrapping a MeshViewport widget."""

    def __init__(self, mesh_viewport: MeshViewport, parent: QWidget | None = None):
        super().__init__("3D Viewport", parent)
        self.setObjectName("MeshViewportDock")
        self._mesh_viewport = mesh_viewport
        self.setWidget(mesh_viewport)

    @property
    def mesh_viewport(self) -> MeshViewport:
        return self._mesh_viewport


class IdentityDock(QDockWidget):
    """Dock wrapping an IdentityInspector widget."""

    def __init__(self, identity_inspector: IdentityInspector, parent: QWidget | None = None):
        super().__init__("Identity Inspector", parent)
        self.setObjectName("IdentityDock")
        self._identity_inspector = identity_inspector
        self.setWidget(identity_inspector)

    @property
    def identity_inspector(self) -> IdentityInspector:
        return self._identity_inspector


class PoseCorrectorDock(QDockWidget):
    """Dock wrapping a PoseCorrectorPanel widget."""

    def __init__(
        self,
        session,
        gvhmr_root=None,
        viewport=None,
        parent: QWidget | None = None,
    ):
        super().__init__("Pose Corrector", parent)
        self.setObjectName("PoseCorrectorDock")
        from views.pose_corrector_panel import PoseCorrectorPanel
        self._pose_corrector = PoseCorrectorPanel(
            session=session, gvhmr_root=gvhmr_root, viewport=viewport, parent=self,
        )
        self.setWidget(self._pose_corrector)

    @property
    def pose_corrector(self) -> PoseCorrectorPanel:
        return self._pose_corrector


class TrackOverviewDock(QDockWidget):
    """Dock wrapping a TrackOverview widget.

    No QScrollArea needed — TrackOverview uses an internal QGraphicsView
    that handles its own horizontal zoom/scroll.
    """

    def __init__(self, track_overview: TrackOverview, parent: QWidget | None = None):
        super().__init__("Track Overview", parent)
        self.setObjectName("TrackOverviewDock")
        self._track_overview = track_overview
        track_overview.setMinimumHeight(80)
        self.setWidget(track_overview)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

    @property
    def track_overview(self) -> TrackOverview:
        return self._track_overview


class SessionLibraryDock(QDockWidget):
    """Dock wrapping a SessionLibrary widget."""

    def __init__(self, session_library: SessionLibrary, parent: QWidget | None = None):
        super().__init__("Session Library", parent)
        self.setObjectName("SessionLibraryDock")
        self._session_library = session_library
        self.setWidget(session_library)

    @property
    def session_library(self) -> SessionLibrary:
        return self._session_library


class PipelineSettingsDock(QDockWidget):
    """Dock with mode QComboBox switching between 3 pipeline settings panels.

    The mode combo emits mode_changed("single"|"perf"|"multi") so AppWindow
    can show/hide multi-only docks (Identity, PoseCorrector, TrackOverview).
    """

    mode_changed = Signal(str)  # "single", "perf", "multi"

    MODE_LABELS = [
        ("single", "GVHMR Body"),
        ("perf", "Performance Capture"),
        ("multi", "Multi-Person"),
    ]

    def __init__(
        self,
        single: SinglePipelineSettings,
        perf: PerfPipelineSettings,
        multi: MultiPipelineSettings,
        parent: QWidget | None = None,
    ):
        super().__init__("Pipeline Settings", parent)
        self.setObjectName("PipelineSettingsDock")

        self._settings = {"single": single, "perf": perf, "multi": multi}

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        self._mode_combo = QComboBox()
        for key, label in self.MODE_LABELS:
            self._mode_combo.addItem(label, key)
        layout.addWidget(self._mode_combo)

        self._stack = QStackedWidget()
        self._stack.addWidget(single)  # index 0
        self._stack.addWidget(perf)    # index 1
        self._stack.addWidget(multi)   # index 2
        layout.addWidget(self._stack, 1)

        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.setWidget(container)

    @property
    def mode_combo(self) -> QComboBox:
        return self._mode_combo

    @property
    def current_mode(self) -> str:
        """The mode key of the currently selected settings panel."""
        return self._mode_combo.currentData()

    @property
    def current_settings(self) -> QWidget:
        """The currently visible settings widget."""
        return self._stack.currentWidget()

    def settings_widget(self, mode: str) -> QWidget:
        """Get the settings widget for a specific mode key."""
        return self._settings[mode]

    def set_mode(self, mode: str):
        """Programmatically switch to the given mode."""
        for i, (key, _) in enumerate(self.MODE_LABELS):
            if key == mode:
                self._mode_combo.setCurrentIndex(i)
                return

    def _on_mode_changed(self, index: int):
        self._stack.setCurrentIndex(index)
        mode = self._mode_combo.currentData()
        self.mode_changed.emit(mode)
