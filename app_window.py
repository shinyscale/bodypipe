"""Main application window with dock-based layout, menu bar, status bar, and log panel.

Why dockable layout: Replaces the fixed 3-tab QTabWidget with QDockWidgets that
users can rearrange, tabify, float, and close/reopen via the View menu.  A mode
selector in PipelineSettingsDock switches the settings panel and shows/hides
multi-person-only docks (Identity Inspector, Pose Corrector, Track Overview).
QMainWindow.saveState()/restoreState() persists the layout across sessions.
"""

import enum
import logging
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStatusBar,
    QDockWidget,
    QPlainTextEdit,
    QWidget,
    QLabel,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QMenu,
)
from PySide6.QtCore import Signal, QSettings, Qt, QByteArray
from PySide6.QtGui import QAction, QPalette, QColor


class InteractionMode(enum.Enum):
    """Context-aware interaction modes that change keyboard shortcuts.

    Why modes: A professional motion capture tool needs different keyboard
    bindings depending on the current task.  Navigate mode uses WASD for
    orbit camera control; Select mode makes click pick a joint; Correct mode
    opens euler sliders; Track mode navigates between persons and unreviewed
    keyframes.  The active mode is shown in the status bar and HUD overlay.
    """
    NAVIGATE = "Navigate"
    SELECT = "Select"
    CORRECT = "Correct"
    TRACK = "Track"

from models.pipeline_config import PipelineConfig
from models.session import Session
from views.video_player import VideoPlayer
from views.mesh_viewport import MeshViewport
from views.identity_inspector import IdentityInspector
from views.track_overview import TrackOverview
from views.pipeline_settings import (
    SinglePipelineSettings,
    PerfPipelineSettings,
    MultiPipelineSettings,
)
from views.session_library import SessionLibrary
from views.dock_widgets import (
    VideoDock,
    MeshViewportDock,
    IdentityDock,
    PoseCorrectorDock,
    TrackOverviewDock,
    PipelineSettingsDock,
    SessionLibraryDock,
)
from views.bbox_overlay import render_bbox_overlay, render_edit_preview
from views.keyboard_shortcuts_dialog import KeyboardShortcutsDialog
from workers.reprocess_worker import ReprocessWorker
from theme import COLORS

log = logging.getLogger(__name__)


def _apply_dark_theme(app):
    """Apply dark palette + Fusion style."""
    from PySide6.QtWidgets import QApplication

    QApplication.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS["bg_primary"]))
    palette.setColor(QPalette.WindowText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.Base, QColor(COLORS["bg_input"]))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS["bg_active_tab"]))
    palette.setColor(QPalette.ToolTipBase, QColor(COLORS["tooltip_bg"]))
    palette.setColor(QPalette.ToolTipText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.Text, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.Button, QColor(COLORS["bg_input"]))
    palette.setColor(QPalette.ButtonText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.BrightText, QColor(COLORS["accent"]))
    palette.setColor(QPalette.Link, QColor(COLORS["accent"]))
    palette.setColor(QPalette.Highlight, QColor(COLORS["accent"]))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(COLORS["text_disabled"]))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(COLORS["text_disabled"]))
    app.setPalette(palette)

    # Fine-grained stylesheet for accent elements
    app.setStyleSheet(f"""
        QTabBar::tab {{
            background: {COLORS["bg_input"]};
            color: {COLORS["text_secondary"]};
            padding: 8px 16px;
            border: 1px solid {COLORS["border"]};
            border-bottom: none;
            border-top-left-radius: 2px;
            border-top-right-radius: 2px;
        }}
        QTabBar::tab:selected {{
            background: {COLORS["bg_active_tab"]};
            color: {COLORS["text_primary"]};
            border-bottom: 2px solid {COLORS["accent"]};
        }}
        QTabBar::tab:hover {{
            background: {COLORS["hover"]};
        }}
        QPushButton {{
            background: {COLORS["bg_input"]};
            color: {COLORS["text_primary"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 2px;
            padding: 6px 12px;
        }}
        QPushButton:hover {{
            background: {COLORS["accent"]};
        }}
        QPushButton:pressed {{
            background: {COLORS["accent_pressed"]};
        }}
        QPushButton:disabled {{
            background: {COLORS["bg_primary"]};
            color: {COLORS["text_disabled"]};
            border-color: {COLORS["border_disabled"]};
        }}
        QGroupBox {{
            border: 1px solid {COLORS["border"]};
            border-radius: 2px;
            margin-top: 8px;
            padding-top: 16px;
            font-weight: bold;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }}
        QSlider::groove:horizontal {{
            background: {COLORS["slider_groove"]};
            height: 6px;
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {COLORS["accent"]};
            width: 14px;
            height: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {COLORS["accent_active"]};
        }}
        QProgressBar {{
            background: {COLORS["bg_input"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 2px;
            text-align: center;
            color: {COLORS["text_primary"]};
        }}
        QProgressBar::chunk {{
            background: {COLORS["accent"]};
            border-radius: 2px;
        }}
        QScrollBar:vertical {{
            background: {COLORS["bg_scroll"]};
            width: 12px;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical {{
            background: {COLORS["slider_grip"]};
            min-height: 20px;
            border-radius: 4px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {COLORS["accent"]};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar:horizontal {{
            background: {COLORS["bg_scroll"]};
            height: 12px;
            border-radius: 4px;
        }}
        QScrollBar::handle:horizontal {{
            background: {COLORS["slider_grip"]};
            min-width: 20px;
            border-radius: 4px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {COLORS["accent"]};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0;
        }}
        QToolTip {{
            background: {COLORS["tooltip_bg"]};
            color: {COLORS["text_primary"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 2px;
            padding: 4px;
        }}
        QHeaderView::section {{
            background: {COLORS["bg_input"]};
            color: {COLORS["text_primary"]};
            border: 1px solid {COLORS["border"]};
            padding: 4px;
        }}
        QHeaderView::section:checked {{
            background: {COLORS["header_checked"]};
        }}
        QDockWidget {{
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }}
        QDockWidget::title {{
            background: {COLORS["bg_input"]};
            padding: 4px;
            border: 1px solid {COLORS["border"]};
        }}
    """)


class LogPanel(QPlainTextEdit):
    """Read-only log panel with color-coded output and line limit."""

    MAX_LINES = 10000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_LINES)
        self.setFont(self.font())  # monospace inherited from app

    def append_line(self, line: str, level: str = "info"):
        color = {
            "info": COLORS["text_primary"],
            "warning": COLORS["warning"],
            "error": COLORS["error"],
        }.get(level, COLORS["text_primary"])
        self.appendHtml(f'<span style="color: {color};">{line}</span>')

    def append_stdout(self, line: str):
        """Auto-detect level from line content."""
        lower = line.lower()
        if "error" in lower or "exception" in lower or "traceback" in lower:
            self.append_line(line, "error")
        elif "warning" in lower or "warn" in lower:
            self.append_line(line, "warning")
        else:
            self.append_line(line, "info")


class AppWindow(QMainWindow):
    """Main application window with dock-based layout.

    All panels live in QDockWidgets that the user can rearrange.  A mode
    selector (in PipelineSettingsDock) switches between single/perf/multi
    settings and shows/hides multi-only docks.  AppWindow acts as the
    central signal hub connecting all panels.
    """

    session_loaded = Signal(object)
    session_saved = Signal(Path)
    tab_changed = Signal(int)   # backward compat — emitted on mode change
    mode_changed = Signal(str)  # "single", "perf", "multi"
    interaction_mode_changed = Signal(str)  # emitted with InteractionMode.value

    MAX_RECENT = 5
    _DOCK_VERSION = 3  # increment when dock layout structure changes

    _MULTI_ONLY_DOCKS = (
        "_identity_dock", "_pose_corrector_dock", "_track_overview_dock",
    )

    # All content docks (excludes _log_dock which is created separately)
    _ALL_CONTENT_DOCKS = (
        "_pipeline_dock", "_video_dock", "_mesh_dock",
        "_identity_dock", "_pose_corrector_dock", "_track_overview_dock",
        "_session_library_dock",
    )

    # Built-in workspace presets: name → (description, set of visible dock attrs)
    _WORKSPACE_PRESETS = {
        "Review": (
            "Video + Inspector + Timeline",
            {"_pipeline_dock", "_video_dock", "_identity_dock",
             "_track_overview_dock"},
        ),
        "Correction": (
            "Video + 3D + Pose Corrector + Inspector",
            {"_pipeline_dock", "_video_dock", "_mesh_dock",
             "_pose_corrector_dock", "_track_overview_dock", "_identity_dock"},
        ),
        "Tracking": (
            "Video + Inspector + Track Overview",
            {"_pipeline_dock", "_video_dock", "_identity_dock",
             "_track_overview_dock"},
        ),
        "Pipeline": (
            "Video + Settings + Library + Log",
            {"_pipeline_dock", "_video_dock", "_session_library_dock"},
        ),
    }

    def __init__(self, gvhmr_root: Path | None = None, parent=None):
        super().__init__(parent)
        self._session = Session()
        self._session_path: Path | None = None
        self._settings = QSettings("GVHMR", "bodypipe")
        self._gvhmr_root = gvhmr_root or Path(__file__).resolve().parent.parent / "GVHMR"
        self._reprocess_worker: ReprocessWorker | None = None
        self._show_all_tracks = False
        self._edit_preview: dict | None = None
        self._interaction_mode: InteractionMode = InteractionMode.NAVIGATE

        self.setWindowTitle("bodypipe \u2014 Motion Capture Studio")
        self.setMinimumSize(1200, 700)

        _apply_dark_theme(QApplication.instance())

        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._setup_status_bar_toggle()
        self._setup_log_panel()
        self._setup_undo_redo()
        self._setup_signal_hub()
        self._setup_interaction_modes()
        self._add_dock_view_toggles()
        # Capture default dock layout before restoring user's saved state
        self._default_state = self.saveState(self._DOCK_VERSION)
        self._restore_geometry()
        self._restore_pipeline_configs()
        self._restore_last_video()
        self._refresh_session_library()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        """Create dock-based layout with mode selector.

        Why empty central widget: All real content lives in docks so users
        can freely rearrange, tabify, and float every panel.  The central
        widget is hidden (zero size) and dock nesting is enabled for
        maximum workspace flexibility.
        """
        # Empty central widget — all content in docks
        central = QWidget()
        central.setMaximumSize(0, 0)
        self.setCentralWidget(central)
        self.setDockNestingEnabled(True)

        # ---- Create inner widgets ----
        self._video_player = VideoPlayer()
        self._mesh_viewport = MeshViewport(gvhmr_root=self._gvhmr_root)
        self._mesh_viewport.set_session(self._session)
        self._identity_inspector = IdentityInspector(self._session)
        self._track_overview = TrackOverview()
        self._session_library = SessionLibrary(gvhmr_root=self._gvhmr_root)

        # ---- Create settings widgets ----
        self._single_settings = SinglePipelineSettings(self._session, self._gvhmr_root)
        self._perf_settings = PerfPipelineSettings(self._session, self._gvhmr_root)
        self._multi_settings = MultiPipelineSettings(self._session, self._gvhmr_root)

        # ---- Create dock widgets ----
        self._video_dock = VideoDock(self._video_player, self)
        self._mesh_dock = MeshViewportDock(self._mesh_viewport, self)
        self._identity_dock = IdentityDock(self._identity_inspector, self)
        self._pose_corrector_dock = PoseCorrectorDock(
            session=self._session, gvhmr_root=self._gvhmr_root,
            viewport=self._mesh_viewport, parent=self,
        )
        self._pose_corrector = self._pose_corrector_dock.pose_corrector
        self._track_overview_dock = TrackOverviewDock(self._track_overview, self)
        self._pipeline_dock = PipelineSettingsDock(
            self._single_settings, self._perf_settings, self._multi_settings, self,
        )
        self._session_library_dock = SessionLibraryDock(self._session_library, self)

        # ---- Arrange docks ----
        # Put dock tabs at the top for side panels; bottom area keeps default (South)
        from PySide6.QtWidgets import QTabWidget
        self.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.North)
        self.setTabPosition(Qt.LeftDockWidgetArea, QTabWidget.North)

        # Left: Pipeline Settings + Session Library (tabified, Settings on top)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._pipeline_dock)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._session_library_dock)
        self.tabifyDockWidget(self._pipeline_dock, self._session_library_dock)
        self._pipeline_dock.raise_()

        # Right: Video, then Identity split to its right (full-height column)
        self.addDockWidget(Qt.RightDockWidgetArea, self._video_dock)
        self.splitDockWidget(self._video_dock, self._identity_dock, Qt.Horizontal)
        self.tabifyDockWidget(self._identity_dock, self._pose_corrector_dock)
        self._identity_dock.raise_()

        # Video/3D stacked: split Video vertically so Mesh goes below Video
        # (Identity stays full-height in the right column)
        self.splitDockWidget(self._video_dock, self._mesh_dock, Qt.Vertical)

        # Bottom: Track overview (log dock added later in _setup_log_panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._track_overview_dock)

    def _setup_menu(self):
        """Create menu bar with File, Edit, View, Help menus."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        open_video_action = QAction("Open &Video...", self)
        open_video_action.setShortcut("Ctrl+O")
        open_video_action.triggered.connect(self._on_open_video)
        file_menu.addAction(open_video_action)

        open_session_action = QAction("Open &Session...", self)
        open_session_action.setShortcut("Ctrl+Shift+O")
        open_session_action.triggered.connect(self._on_open_session)
        file_menu.addAction(open_session_action)

        save_session_action = QAction("&Save Session", self)
        save_session_action.setShortcut("Ctrl+S")
        save_session_action.triggered.connect(self._on_save_session)
        file_menu.addAction(save_session_action)

        file_menu.addSeparator()

        self._recent_menu = QMenu("Recent Sessions", self)
        file_menu.addMenu(self._recent_menu)
        self._update_recent_menu()

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        self._undo_action = QAction("&Undo", self)
        self._undo_action.setShortcut("Ctrl+Z")
        self._undo_action.setEnabled(False)
        self._undo_action.triggered.connect(self._on_undo)
        edit_menu.addAction(self._undo_action)

        self._redo_action = QAction("&Redo", self)
        self._redo_action.setShortcut("Ctrl+Shift+Z")
        self._redo_action.setEnabled(False)
        self._redo_action.triggered.connect(self._on_redo)
        edit_menu.addAction(self._redo_action)

        # View menu (stored for _add_dock_view_toggles)
        self._view_menu = menubar.addMenu("&View")

        self._toggle_statusbar_action = QAction("Toggle &Status Bar", self)
        self._toggle_statusbar_action.setCheckable(True)
        self._toggle_statusbar_action.setChecked(True)
        self._view_menu.addAction(self._toggle_statusbar_action)

        self._toggle_log_action = QAction("Toggle &Log Panel", self)
        self._toggle_log_action.setShortcut("Ctrl+L")
        self._toggle_log_action.setCheckable(True)
        self._toggle_log_action.setChecked(True)
        self._view_menu.addAction(self._toggle_log_action)

        self._toggle_hud_action = QAction("Toggle &HUD Overlay", self)
        self._toggle_hud_action.setShortcut("Ctrl+H")
        self._toggle_hud_action.setCheckable(True)
        self._toggle_hud_action.setChecked(True)
        self._toggle_hud_action.toggled.connect(self._on_toggle_hud)
        self._view_menu.addAction(self._toggle_hud_action)

        self._view_menu.addSeparator()
        self._workspace_menu = QMenu("&Workspace", self)
        self._view_menu.addMenu(self._workspace_menu)
        self._build_workspace_menu()

        # Help menu
        help_menu = menubar.addMenu("&Help")

        shortcuts_action = QAction("&Keyboard Shortcuts", self)
        shortcuts_action.triggered.connect(self._on_keyboard_shortcuts)
        help_menu.addAction(shortcuts_action)

        help_menu.addSeparator()

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _add_dock_view_toggles(self):
        """Add per-dock toggle actions to the View menu."""
        self._view_menu.addSeparator()
        for dock in (
            self._pipeline_dock, self._video_dock, self._mesh_dock,
            self._identity_dock, self._pose_corrector_dock,
            self._track_overview_dock, self._session_library_dock,
        ):
            self._view_menu.addAction(dock.toggleViewAction())

    def _setup_status_bar(self):
        """Create status bar with operation status, mode indicator, frame counter, FPS."""
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("Ready")
        self._mode_label = QLabel("")
        self._mode_label.setStyleSheet(
            f"QLabel {{ background: {COLORS['accent']}; color: #ffffff; "
            f"padding: 1px 8px; border-radius: 3px; font-weight: bold; "
            f"font-size: 11px; }}"
        )
        self._frame_label = QLabel("")
        self._fps_label = QLabel("")

        self._status_bar.addWidget(self._status_label, 1)
        self._status_bar.addPermanentWidget(self._mode_label)
        self._status_bar.addPermanentWidget(self._frame_label)
        self._status_bar.addPermanentWidget(self._fps_label)

    def _setup_status_bar_toggle(self):
        """Wire status bar toggle after both status bar and menu are created."""
        self._toggle_statusbar_action.toggled.connect(self._status_bar.setVisible)

    def _setup_log_panel(self):
        """Create collapsible log dock widget, tabified with track overview."""
        self._log_panel = LogPanel()
        self._log_dock = QDockWidget("Log", self)
        self._log_dock.setObjectName("LogDock")
        self._log_dock.setWidget(self._log_panel)
        self._log_dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._log_dock)
        self.tabifyDockWidget(self._track_overview_dock, self._log_dock)
        self._log_dock.raise_()

        self._toggle_log_action.toggled.connect(self._log_dock.setVisible)
        self._log_dock.visibilityChanged.connect(self._toggle_log_action.setChecked)

    def _restore_geometry(self):
        """Restore window geometry and dock state from settings."""
        geometry = self._settings.value("geometry")
        if geometry and isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        else:
            self.resize(1600, 900)

        state = self._settings.value("windowState")
        if state and isinstance(state, QByteArray):
            self.restoreState(state, self._DOCK_VERSION)

        # Enforce mode-based dock visibility — restoreState may have made
        # multi-only docks visible from a previous session.
        self._update_dock_visibility()

    # ------------------------------------------------------------------
    # Workspace presets
    # ------------------------------------------------------------------

    def _build_workspace_menu(self):
        """Populate the View > Workspace submenu with presets and actions."""
        self._workspace_menu.clear()
        for name, (desc, _visible) in self._WORKSPACE_PRESETS.items():
            action = QAction(f"{name}  —  {desc}", self)
            action.triggered.connect(lambda checked, n=name: self._apply_preset(n))
            self._workspace_menu.addAction(action)

        self._workspace_menu.addSeparator()

        # Custom saved layouts
        custom_names = self._get_custom_workspace_names()
        if custom_names:
            for cname in custom_names:
                action = QAction(cname, self)
                action.triggered.connect(
                    lambda checked, n=cname: self._apply_custom_workspace(n),
                )
                self._workspace_menu.addAction(action)
            self._workspace_menu.addSeparator()

        save_action = QAction("Save Current Layout...", self)
        save_action.triggered.connect(self._on_save_workspace)
        self._workspace_menu.addAction(save_action)

        reset_action = QAction("Reset to Default", self)
        reset_action.triggered.connect(self._on_reset_workspace)
        self._workspace_menu.addAction(reset_action)

    def _apply_preset(self, name: str):
        """Apply a built-in workspace preset by showing/hiding docks."""
        _desc, visible_attrs = self._WORKSPACE_PRESETS[name]

        # Restore default dock arrangement first
        self.restoreState(self._default_state, self._DOCK_VERSION)

        # Show/hide content docks per preset
        for attr in self._ALL_CONTENT_DOCKS:
            dock = getattr(self, attr, None)
            if dock:
                dock.setVisible(attr in visible_attrs)

        # Log dock: visible only in Pipeline preset
        if hasattr(self, "_log_dock"):
            show_log = name == "Pipeline"
            self._log_dock.setVisible(show_log)
            self._toggle_log_action.setChecked(show_log)

        # Raise video dock in tabified groups
        self._video_dock.raise_()

        self.set_status(f"Workspace: {name}")

    def _get_custom_workspace_names(self) -> list[str]:
        """Return names of user-saved custom workspace layouts."""
        raw = self._settings.value("workspace/custom_names")
        if raw and isinstance(raw, list):
            return raw
        return []

    def _on_save_workspace(self):
        """Prompt for a name and save the current dock layout."""
        name, ok = QInputDialog.getText(
            self, "Save Workspace", "Layout name:",
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        # Save the current state bytes
        state = self.saveState(self._DOCK_VERSION)
        self._settings.setValue(f"workspace/state/{name}", state)

        # Update the custom names list
        names = self._get_custom_workspace_names()
        if name not in names:
            names.append(name)
        self._settings.setValue("workspace/custom_names", names)

        # Rebuild menu to include the new entry
        self._build_workspace_menu()
        self.set_status(f"Workspace saved: {name}")

    def _apply_custom_workspace(self, name: str):
        """Restore a user-saved custom workspace layout."""
        state = self._settings.value(f"workspace/state/{name}")
        if state and isinstance(state, QByteArray):
            self.restoreState(state, self._DOCK_VERSION)
            self._update_dock_visibility()
            self.set_status(f"Workspace: {name}")

    def _on_reset_workspace(self):
        """Restore the hardcoded initial dock layout."""
        self.restoreState(self._default_state, self._DOCK_VERSION)
        self._update_dock_visibility()
        self.set_status("Workspace reset to default")

    # ------------------------------------------------------------------
    # Pipeline config persistence
    # ------------------------------------------------------------------

    _TAB_CONFIG_MAP = {
        "single": "_single_settings",
        "perf": "_perf_settings",
        "multi": "_multi_settings",
    }

    def _save_pipeline_configs(self):
        """Save each settings panel's pipeline config to QSettings."""
        import json

        for mode, attr in self._TAB_CONFIG_MAP.items():
            widget = getattr(self, attr, None)
            if widget and hasattr(widget, "get_config"):
                config = widget.get_config()
                self._settings.setValue(
                    f"pipeline_config/{mode}",
                    json.dumps(config.to_dict()),
                )

    def _restore_pipeline_configs(self):
        """Restore each settings panel's pipeline config from QSettings."""
        import json

        for mode, attr in self._TAB_CONFIG_MAP.items():
            widget = getattr(self, attr, None)
            if not widget or not hasattr(widget, "set_config"):
                continue
            raw = self._settings.value(f"pipeline_config/{mode}")
            if not raw or not isinstance(raw, str):
                continue
            try:
                data = json.loads(raw)
                config = PipelineConfig.from_dict(data)
                widget.set_config(config)
            except Exception:
                pass  # Ignore corrupt/stale settings

    def closeEvent(self, event):
        """Save window geometry, dock state, and pipeline configs on close."""
        self._save_pipeline_configs()
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState(self._DOCK_VERSION))
        # Wait for any running worker threads to prevent QThread destruction crash
        for settings in (self._single_settings, self._perf_settings, self._multi_settings):
            worker = getattr(settings, "_worker", None)
            if worker is not None and worker.isRunning():
                worker.cancel()
                worker.wait(5000)
        # Clean up reprocess worker
        if self._reprocess_worker is not None and self._reprocess_worker.isRunning():
            self._reprocess_worker.cancel()
            self._reprocess_worker.wait(5000)
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Signal Hub
    # ------------------------------------------------------------------

    def _setup_signal_hub(self):
        """Wire all cross-panel signals.

        Why centralised: In the tab layout, MultiPersonTab was the signal
        hub connecting sub-panels.  With docks, AppWindow owns all panels
        directly and takes over the hub role so panels stay decoupled.
        """
        # Settings widgets → status bar + log
        for settings in (self._single_settings, self._perf_settings, self._multi_settings):
            settings.status_message.connect(self.set_status)
            settings.log_message.connect(
                lambda text, level: self._log_panel.append_line(text, level)
            )

        # Video loaded → shared VideoPlayer
        self._single_settings.video_loaded.connect(self._on_video_loaded)
        self._perf_settings.video_loaded.connect(self._on_video_loaded)
        self._multi_settings.video_loaded.connect(self._on_video_loaded)

        # Pipeline finished handlers
        self._single_settings.pipeline_finished.connect(self._on_pipeline_output)
        self._perf_settings.pipeline_finished.connect(self._on_pipeline_output)
        self._multi_settings.pipeline_finished.connect(self._on_multi_pipeline_finished)

        # Shared VideoPlayer → status bar + multi-mode broadcast
        self._video_player.frame_changed.connect(self._on_video_frame_changed)

        # Auto-switch 3D viewport to wireframe during scrubbing/playback
        self._video_player.scrub_started.connect(
            lambda: self._mesh_viewport.set_scrubbing(True)
        )
        self._video_player.scrub_ended.connect(
            lambda: self._mesh_viewport.set_scrubbing(False)
        )
        self._video_player.playback_toggled.connect(
            lambda playing: self._mesh_viewport.set_scrubbing(playing)
        )

        # Multi-mode signal hub (signals only fire when panels are visible)
        self._video_player.frame_clicked.connect(self._identity_inspector.on_frame_click)
        self._video_player.bbox_dragged.connect(self._identity_inspector.on_bbox_drag)
        self._track_overview.person_clicked.connect(self._on_track_clicked)
        self._identity_inspector.frame_requested.connect(self._video_player.seek)
        self._identity_inspector.person_changed.connect(self._on_identity_person_changed)
        self._identity_inspector.bbox_overlay_changed.connect(self._on_bbox_overlay_changed)
        self._identity_inspector.keyframe_changed.connect(self._on_keyframe_changed)
        self._identity_inspector.track_modified.connect(self._on_tracks_modified)
        self._identity_inspector.reprocess_requested.connect(self._on_reprocess_requested)
        self._pose_corrector.frame_requested.connect(self._video_player.seek)

        # Auto-raise PoseCorrector dock when a joint is clicked in 3D viewport
        self._mesh_viewport.joint_clicked.connect(self._on_joint_clicked_auto_raise)

        # Speed sync between video player and track timeline
        self._video_player.speed_changed.connect(self._track_overview.set_speed)
        self._track_overview.speed_changed.connect(self._video_player.set_playback_speed)

        # Session library → load session on double-click
        self._session_library.session_load_requested.connect(
            lambda path: self._load_session(Path(path))
        )
        # Refresh library when a session is saved
        self.session_saved.connect(lambda _: self._refresh_session_library())

        # Mode selector
        self._pipeline_dock.mode_changed.connect(self._on_mode_changed)

    # ------------------------------------------------------------------
    # Interaction modes (Phase 10)
    # ------------------------------------------------------------------

    # Mode → number key mapping
    _MODE_KEYS = {
        Qt.Key_1: InteractionMode.NAVIGATE,
        Qt.Key_2: InteractionMode.SELECT,
        Qt.Key_3: InteractionMode.CORRECT,
        Qt.Key_4: InteractionMode.TRACK,
    }

    def _setup_interaction_modes(self):
        """Initialise interaction mode state and update the status bar indicator."""
        self._update_mode_indicator()

    def set_interaction_mode(self, mode: InteractionMode):
        """Switch the active interaction mode and update UI."""
        if self._interaction_mode == mode:
            return
        self._interaction_mode = mode
        self._update_mode_indicator()
        self.interaction_mode_changed.emit(mode.value)
        self._mesh_viewport.set_hud_mode(mode.value)
        log.info("Interaction mode → %s", mode.value)

    def _update_mode_indicator(self):
        """Update the status bar mode pill label."""
        mode = self._interaction_mode
        # Color-code the mode pill for quick visual identification
        mode_colors = {
            InteractionMode.NAVIGATE: COLORS["accent"],
            InteractionMode.SELECT:   "#3d85c6",  # blue
            InteractionMode.CORRECT:  "#cc4125",  # red
            InteractionMode.TRACK:    "#6aa84f",  # green
        }
        bg = mode_colors.get(mode, COLORS["accent"])
        self._mode_label.setStyleSheet(
            f"QLabel {{ background: {bg}; color: #ffffff; "
            f"padding: 1px 8px; border-radius: 3px; font-weight: bold; "
            f"font-size: 11px; }}"
        )
        self._mode_label.setText(f"{mode.value} (#{list(InteractionMode).index(mode) + 1})")

    def keyPressEvent(self, event):
        """Route key events based on active interaction mode.

        Why centralised: Each mode remaps the same physical keys to different
        actions — e.g. G means "go-to-frame" in Navigate but "next unreviewed"
        in Track mode.  Processing here before child widgets see the event
        ensures mode-awareness without each widget needing to know about modes.
        """
        key = event.key()
        mod = event.modifiers()

        # Mode switching: 1-4 keys (without modifiers)
        if not mod and key in self._MODE_KEYS:
            self.set_interaction_mode(self._MODE_KEYS[key])
            return

        # Global shortcuts (mode-independent)
        if not mod and key == Qt.Key_V:
            # Toggle camera mode between incam and orbit
            vp = self._mesh_viewport
            new_mode = "orbit" if vp._camera_mode == "incam" else "incam"
            vp.set_camera_mode(new_mode)
            self.set_status(f"Camera: {new_mode}")
            return

        # Dispatch to mode-specific handler
        handled = False
        if self._interaction_mode == InteractionMode.NAVIGATE:
            handled = self._key_navigate(key, mod)
        elif self._interaction_mode == InteractionMode.SELECT:
            handled = self._key_select(key, mod)
        elif self._interaction_mode == InteractionMode.CORRECT:
            handled = self._key_correct(key, mod)
        elif self._interaction_mode == InteractionMode.TRACK:
            handled = self._key_track(key, mod)

        if not handled:
            super().keyPressEvent(event)

    def _key_navigate(self, key, mod) -> bool:
        """Navigate mode: WASD orbit camera, G go-to-frame."""
        if key == Qt.Key_W:
            self._orbit_nudge(pitch=-5)
            return True
        if key == Qt.Key_S:
            self._orbit_nudge(pitch=5)
            return True
        if key == Qt.Key_A:
            self._orbit_nudge(yaw=-5)
            return True
        if key == Qt.Key_D:
            self._orbit_nudge(yaw=5)
            return True
        if key == Qt.Key_G:
            self._go_to_frame_dialog()
            return True
        return False

    def _key_select(self, key, mod) -> bool:
        """Select mode: G go-to-frame, Escape deselect."""
        if key == Qt.Key_G:
            self._go_to_frame_dialog()
            return True
        if key == Qt.Key_Escape:
            self._mesh_viewport._selected_joint = -1
            self._mesh_viewport.joint_clicked.emit(-1)
            if hasattr(self._mesh_viewport, 'update'):
                self._mesh_viewport.update()
            return True
        return False

    def _key_correct(self, key, mod) -> bool:
        """Correct mode: G open euler, R reset joint, Escape deselect."""
        if key == Qt.Key_G:
            # Focus the pose corrector dock and raise it
            self._pose_corrector_dock.setVisible(True)
            self._pose_corrector_dock.raise_()
            return True
        if key == Qt.Key_R:
            # Reset the currently selected joint rotation
            self._pose_corrector.reset_current_joint()
            return True
        if key == Qt.Key_Escape:
            self._mesh_viewport._selected_joint = -1
            self._mesh_viewport.joint_clicked.emit(-1)
            if hasattr(self._mesh_viewport, 'update'):
                self._mesh_viewport.update()
            return True
        return False

    def _key_track(self, key, mod) -> bool:
        """Track mode: G next unreviewed, Tab next person, Shift+Tab prev, E edit bbox."""
        if key == Qt.Key_G:
            self._identity_inspector.go_to_next_unreviewed()
            return True
        if key == Qt.Key_Tab:
            if mod & Qt.ShiftModifier:
                self._cycle_person(-1)
            else:
                self._cycle_person(1)
            return True
        if key == Qt.Key_E:
            self._identity_dock.setVisible(True)
            self._identity_dock.raise_()
            self._identity_inspector._on_edit_bbox()
            return True
        return False

    def _orbit_nudge(self, yaw: float = 0, pitch: float = 0):
        """Nudge the orbit camera by the given yaw/pitch degrees."""
        vp = self._mesh_viewport
        if vp._camera_mode != "orbit":
            vp.set_camera_mode("orbit")
        vp._orbit_yaw += yaw
        vp._orbit_pitch = float(np.clip(vp._orbit_pitch + pitch, -89, 89))
        vp._update_camera()
        vp.camera_changed.emit(vp._camera_state())

    def _go_to_frame_dialog(self):
        """Show a go-to-frame input dialog."""
        max_frame = max(0, self._video_player.num_frames - 1)
        frame, ok = QInputDialog.getInt(
            self, "Go to Frame", f"Frame (0–{max_frame}):",
            self._video_player._current_frame, 0, max_frame,
        )
        if ok:
            self._video_player.seek(frame)

    def _cycle_person(self, direction: int):
        """Cycle through person tracks by direction (+1 = next, -1 = prev)."""
        pids = sorted(self._session.person_tracks.keys())
        if not pids:
            return
        current = self._session.selected_person
        if current in pids:
            idx = pids.index(current)
            idx = (idx + direction) % len(pids)
        else:
            idx = 0
        new_pid = pids[idx]
        self._session.selected_person = new_pid
        self._identity_inspector.set_person(new_pid)
        self._pose_corrector.set_person(new_pid)
        self._mesh_viewport.set_person(new_pid)
        self.set_status(f"Selected Person {new_pid}")

    def _on_toggle_hud(self, visible: bool):
        """Toggle the HUD overlay on the mesh viewport."""
        self._mesh_viewport.set_hud_visible(visible)

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def _on_mode_changed(self, mode: str):
        """Show/hide multi-only docks and emit signals."""
        self._update_dock_visibility()
        mode_to_index = {"single": 0, "perf": 1, "multi": 2}
        self.tab_changed.emit(mode_to_index.get(mode, 0))
        self.mode_changed.emit(mode)

    def _update_dock_visibility(self):
        """Show/hide docks based on current pipeline mode."""
        is_multi = self._pipeline_dock.current_mode == "multi"
        for attr in self._MULTI_ONLY_DOCKS:
            dock = getattr(self, attr, None)
            if dock:
                dock.setVisible(is_multi)

    # ------------------------------------------------------------------
    # Video frame handling
    # ------------------------------------------------------------------

    def _on_video_frame_changed(self, frame_idx: int):
        """Update status bar, HUD, and broadcast frame change in multi mode."""
        self.set_frame_info(frame_idx, self._video_player.num_frames)
        self.set_fps_info(self._video_player.fps)
        self._session.current_frame = frame_idx
        # Update HUD overlay with current state
        self._mesh_viewport.update_hud(
            frame=frame_idx,
            total_frames=self._video_player.num_frames,
            speed=self._video_player._playback_speed,
            person=self._session.selected_person,
        )

        if self._pipeline_dock.current_mode == "multi":
            self._track_overview.set_current_frame(frame_idx)
            self._identity_inspector.set_frame(frame_idx)
            self._pose_corrector.on_frame_changed(frame_idx)
            raw = self._video_player.get_raw_frame(frame_idx)
            self._mesh_viewport.set_video_frame(raw)
            self._mesh_viewport.on_frame_changed(frame_idx)
            self._show_frame(frame_idx)

    def _on_video_loaded(self, video_path):
        """Load video into the shared VideoPlayer and restore cached results."""
        self._video_player.set_video(
            video_path, self._session.num_frames, self._session.fps,
        )
        # Persist last video path for startup restore
        self._settings.setValue("last_video_path", str(video_path))
        # Check for existing pipeline results on disk
        self._try_restore_results(video_path)

    def _show_frame(self, frame_idx: int):
        """Display the current frame with bbox overlays and edit preview."""
        frame = self._video_player.get_raw_frame(frame_idx)
        if frame is not None:
            composited = render_bbox_overlay(
                frame,
                self._session,
                frame_idx,
                selected_person=self._session.selected_person,
                show_all_tracks=self._show_all_tracks,
            )
            if self._edit_preview:
                composited = render_edit_preview(composited, self._edit_preview)
            self._video_player.set_frame(composited)

    # ------------------------------------------------------------------
    # Startup restore & result loading
    # ------------------------------------------------------------------

    def _restore_last_video(self):
        """On startup, restore the last-loaded video into the current settings panel.

        Why: Without this, the app starts with all docks empty even if the user
        was working on a video in the previous session. Pipeline configs are
        restored by _restore_pipeline_configs(), but the video itself was not.
        """
        last_path = self._settings.value("last_video_path")
        if not last_path or not isinstance(last_path, str):
            return
        path = Path(last_path)
        if not path.is_file():
            return
        settings = self._pipeline_dock.current_settings
        if hasattr(settings, "_load_video"):
            settings._load_video(str(path))

    def _try_restore_results(self, video_path):
        """Load cached pipeline results if a previous run exists for this video.

        Why: When a previously-processed video is loaded (on startup or via
        File > Open), the user expects to see their results immediately — not
        a blank 3D viewport and empty inspector.  This checks for existing
        person tracks (from a loaded session) or scans the expected output
        directory for the current pipeline mode.
        """
        video_path = Path(video_path)

        # If session already has person tracks (loaded from session JSON),
        # hydrate heavy data from disk and refresh all panels.
        if self._session.person_tracks:
            self._hydrate_person_tracks()
            self._refresh_all_panels()
            return

        # Otherwise, check for cached results on disk
        mode = self._pipeline_dock.current_mode

        # Use existing output_dir from session if valid, else derive from mode
        if self._session.output_dir and self._session.output_dir.is_dir():
            output_dir = self._session.output_dir
        elif mode == "multi":
            output_dir = self._gvhmr_root / "outputs" / "multi_person" / video_path.stem
        else:
            output_dir = self._gvhmr_root / "outputs" / "demo" / video_path.stem

        if not output_dir.is_dir():
            return

        self._session.output_dir = output_dir

        if mode == "multi":
            self._load_results_from_output_dir(output_dir)
        else:
            # Single/perf: load output preview video if available
            for name in ("side_by_side.mp4", "incam.mp4"):
                preview = output_dir / name
                if preview.is_file():
                    self._load_output_preview(preview)
                    break

    def _load_results_from_output_dir(self, output_dir: Path):
        """Load person tracks from an existing multi-person output directory.

        Why: After the tab→dock migration, pipeline results were only loaded
        when the pipeline finished (via _on_multi_pipeline_finished). This
        method enables loading cached results from disk on startup or when
        a previously-processed video is opened, without re-running the pipeline.
        """
        from models.session import PersonTrack

        # Discover person directories on disk
        person_dirs = sorted(
            [d for d in output_dir.iterdir()
             if d.is_dir() and d.name.startswith("person_")],
            key=lambda d: d.name,
        )
        if not person_dirs:
            return

        for pdir in person_dirs:
            try:
                pid = int(pdir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            confidences, confidence_breakdown = self._load_confidences_csv(pdir)
            smplx_params, soma_params, body_model_type = self._load_motion_params(pdir)

            pt = PersonTrack(
                person_id=pid,
                person_dir=pdir,
                confidences=confidences,
                confidence_breakdown=confidence_breakdown,
                smplx_params=smplx_params,
                soma_params=soma_params,
                body_model_type=body_model_type,
            )
            self._session.person_tracks[pid] = pt

        # Load crossing spans
        import json as _json
        for pid, pt in self._session.person_tracks.items():
            if pt.person_dir:
                spans_path = pt.person_dir / "crossing_spans.json"
                if spans_path.is_file():
                    try:
                        spans = _json.loads(spans_path.read_text())
                        self._session.crossing_spans[pid] = [
                            tuple(s) for s in spans
                        ]
                    except Exception:
                        pass

        self._refresh_all_panels()

    def _hydrate_person_tracks(self):
        """Load heavy data (smplx_params, confidences) from disk for existing tracks.

        Why: Session JSON stores lightweight fields (person_id, person_dir,
        keyframes) but not heavy data like SMPL-X parameters or per-frame
        confidence arrays.  This method fills in the gaps from disk so that
        the 3D viewport, identity inspector, and track overview can render.
        """
        for _pid, track in self._session.person_tracks.items():
            if not track.person_dir or not track.person_dir.is_dir():
                continue
            if track.smplx_params is None and track.soma_params is None:
                smplx, soma, bmt = self._load_motion_params(track.person_dir)
                track.smplx_params = smplx
                track.soma_params = soma
                track.body_model_type = bmt
            if track.confidences is None:
                track.confidences, track.confidence_breakdown = (
                    self._load_confidences_csv(track.person_dir)
                )
            # Fall back to GEM-X embedded confidences from soma_params
            if track.confidences is None and track.soma_params is not None:
                gemx_conf = track.soma_params.get("confidences")
                if gemx_conf is not None:
                    track.confidences = gemx_conf.tolist()

    def _refresh_all_panels(self):
        """Refresh all panels from current session state after results are loaded.

        Why: Multiple code paths need the same "populate everything" logic —
        startup restore, session load, pipeline finish, reprocess complete.
        Centralising it here prevents duplication and ensures nothing is missed.
        """
        if not self._session.person_tracks:
            return

        # Populate track overview and markers
        self._populate_tracks()

        # Refresh identity inspector
        self._identity_inspector.refresh()

        # Auto-select first person if none selected
        if self._session.selected_person < 0:
            first_pid = min(self._session.person_tracks.keys())
            self._session.selected_person = first_pid

        pid = self._session.selected_person
        self._identity_inspector.set_person(pid)
        self._mesh_viewport.set_person(pid)
        self._pose_corrector.set_person(pid)

        # Auto-switch to orbit camera for skeleton-only tracks (SOMA/GEM-X)
        track = self._session.person_tracks.get(pid)
        if track is not None and track.body_model_type == "soma":
            self._mesh_viewport.set_camera_mode("orbit")

        # Broadcast current frame to all panels
        frame = self._session.current_frame
        self._track_overview.set_current_frame(frame)
        self._identity_inspector.set_frame(frame)
        self._pose_corrector.on_frame_changed(frame)
        raw = self._video_player.get_raw_frame(frame)
        if raw is not None:
            self._mesh_viewport.set_video_frame(raw)
            self._mesh_viewport.on_frame_changed(frame)

        # Show composited frame with overlays in multi mode
        if self._pipeline_dock.current_mode == "multi":
            self._show_frame(frame)

        log.info(
            "Panels refreshed: %d person tracks, selected person %d",
            len(self._session.person_tracks), pid,
        )

    # ------------------------------------------------------------------
    # Track / person interaction (signal hub for multi mode)
    # ------------------------------------------------------------------

    def _on_track_clicked(self, person_id: int, frame_idx: int):
        """Select person and seek to frame from track overview."""
        self._session.selected_person = person_id
        self._identity_inspector.set_person(person_id)
        self._pose_corrector.set_person(person_id)
        self._mesh_viewport.set_person(person_id)
        self._video_player.seek(frame_idx)
        self.set_status(f"Selected Person {person_id} at frame {frame_idx}")

    def _on_identity_person_changed(self, person_id: int):
        """Handle person change from identity inspector."""
        self._session.selected_person = person_id
        self._pose_corrector.set_person(person_id)
        self._mesh_viewport.set_person(person_id)
        self._show_frame(self._session.current_frame)

    def _on_bbox_overlay_changed(self, data: object):
        """Handle overlay changes (show-all-tracks toggle, edit preview)."""
        if isinstance(data, dict):
            if "show_all" in data:
                self._show_all_tracks = data["show_all"]
            if "edit_preview" in data:
                self._edit_preview = data["edit_preview"]
        self._show_frame(self._session.current_frame)

    def _on_keyframe_changed(self, person_id: int, frame_idx: int):
        """Redraw overlay when keyframes change."""
        self._show_frame(self._session.current_frame)

    def _on_tracks_modified(self):
        """Handle track modifications (swap, split, merge)."""
        self._populate_tracks()
        self._identity_inspector.refresh()
        self._show_frame(self._session.current_frame)

    def _on_joint_clicked_auto_raise(self, joint_idx: int):
        """Auto-raise PoseCorrector dock when a joint is clicked in 3D viewport."""
        if joint_idx >= 0 and self._pipeline_dock.current_mode == "multi":
            self._pose_corrector_dock.setVisible(True)
            self._pose_corrector_dock.raise_()

    # ------------------------------------------------------------------
    # Pipeline output handling
    # ------------------------------------------------------------------

    def _on_pipeline_output(self, result: dict):
        """Handle single/perf pipeline completion — load output preview and switch workspace."""
        output_dir = result.get("output_dir")
        if output_dir:
            self._session.output_dir = Path(output_dir)
        loaded = False
        for key in ("side_by_side", "incam"):
            path = result.get(key)
            if path and Path(path).is_file():
                self._load_output_preview(Path(path))
                loaded = True
                break
        # Auto-switch to Review workspace if we loaded results
        if loaded or result.get("soma_params") or result.get("merged_pt"):
            self._apply_preset("Review")

    def _load_output_preview(self, video_path: Path):
        """Load an output video into the shared VideoPlayer."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        self._video_player.set_video(video_path, num_frames, fps)

    def _on_multi_pipeline_finished(self, result: dict):
        """Handle multi pipeline completion — load tracks, auto-detect issues, switch workspace."""
        output_dir = result.get("output_dir")
        if output_dir:
            self._session.output_dir = Path(output_dir)

        multi_result = result.get("result")
        if multi_result is not None:
            self._load_person_tracks_from_result(multi_result)

        self._refresh_all_panels()

        # Auto-detect pose issues and switch to Correction workspace
        if self._session.person_tracks:
            try:
                self._pose_corrector.run_auto_detection()
            except Exception as exc:
                log.warning("Auto-detection failed: %s", exc)
            self._apply_preset("Correction")

    # ------------------------------------------------------------------
    # Reprocess
    # ------------------------------------------------------------------

    def _on_reprocess_requested(self, person_ids: list):
        """Launch ReprocessWorker for dirty persons."""
        if self._reprocess_worker is not None:
            self.set_status("Reprocess already running")
            return

        self._reprocess_worker = ReprocessWorker(
            session=self._session,
            person_ids=person_ids,
        )
        self._reprocess_worker.progress.connect(self._on_reprocess_progress)
        self._reprocess_worker.person_done.connect(self._on_reprocess_person_done)
        self._reprocess_worker.finished.connect(self._on_reprocess_finished)
        self._reprocess_worker.error.connect(self._on_reprocess_error)
        self._reprocess_worker.start()

        self._multi_settings._progress_bar.show()
        self._multi_settings._progress_label.show()
        self.set_status(f"Reprocessing {len(person_ids)} person(s)...")
        self._log_panel.append_line(
            f"Reprocess started for persons: {person_ids}", "info",
        )

    def _on_reprocess_progress(self, fraction: float, stage: str):
        self._multi_settings._progress_bar.setValue(int(fraction * 1000))
        self._multi_settings._progress_label.setText(stage)
        self.set_status(f"{stage} ({fraction:.0%})")

    def _on_reprocess_person_done(self, person_id: int):
        self._session.dirty_persons.discard(person_id)
        self._identity_inspector.update_reprocess_button()
        self._log_panel.append_line(f"Person {person_id} reprocessed", "info")

    def _on_reprocess_finished(self, result: dict):
        if self._reprocess_worker is not None:
            self._reprocess_worker.wait()
            self._reprocess_worker = None
        self._multi_settings._progress_bar.hide()
        self._multi_settings._progress_label.hide()
        self._multi_settings._progress_bar.setValue(0)

        reprocessed = result.get("reprocessed", [])
        self._session.dirty_persons -= set(reprocessed)

        self._populate_tracks()
        self._identity_inspector.refresh()
        self._show_frame(self._session.current_frame)

        self.set_status(f"Reprocess complete: {len(reprocessed)} person(s) updated")
        self._log_panel.append_line(f"Reprocess finished: {reprocessed}", "info")

    def _on_reprocess_error(self, message: str):
        if self._reprocess_worker is not None:
            self._reprocess_worker.wait()
            self._reprocess_worker = None
        self._multi_settings._progress_bar.hide()
        self._multi_settings._progress_label.hide()
        self._multi_settings._progress_bar.setValue(0)

        self.set_status(f"Reprocess error: {message}")
        self._log_panel.append_line(f"Reprocess error: {message}", "error")

    # ------------------------------------------------------------------
    # Person track loading (from multi pipeline results)
    # ------------------------------------------------------------------

    def _load_person_tracks_from_result(self, multi_result):
        """Convert MultiPersonResult into session.person_tracks."""
        from models.session import PersonTrack

        self._session.person_tracks.clear()
        self._session.inactive_tracks.clear()

        all_tracks = getattr(multi_result, "all_tracks", [])
        person_dirs = getattr(multi_result, "person_dirs", [])
        identity_tracks = getattr(multi_result, "identity_tracks", [])

        for i, track in enumerate(all_tracks):
            tid = track.get("track_id", i)
            bboxes_raw = track["bbx_xyxy"]
            if hasattr(bboxes_raw, "numpy"):
                bboxes = bboxes_raw.cpu().numpy()
            else:
                bboxes = np.asarray(bboxes_raw)

            person_dir = Path(person_dirs[i]) if i < len(person_dirs) else None
            id_track = identity_tracks[i] if i < len(identity_tracks) else None

            keyframes = []
            if id_track and hasattr(id_track, "keyframes"):
                for kf in id_track.keyframes:
                    keyframes.append({
                        "frame": kf.frame_index,
                        "verified": kf.verified,
                        "confidence": getattr(kf, "confidence", None),
                    })

            confidences = None
            confidence_breakdown = None
            if person_dir:
                confidences, confidence_breakdown = self._load_confidences_csv(
                    person_dir
                )

            smplx_params = None
            soma_params = None
            body_model_type = "smplx"
            if person_dir:
                smplx_params, soma_params, body_model_type = self._load_motion_params(person_dir)

            # Use GEM-X embedded confidences if no CSV exists
            if confidences is None and soma_params is not None:
                gemx_conf = soma_params.get("confidences")
                if gemx_conf is not None:
                    confidences = gemx_conf.tolist()

            pt = PersonTrack(
                person_id=tid,
                person_dir=person_dir,
                identity_track=id_track,
                confidences=confidences,
                bboxes=bboxes,
                keyframes=keyframes,
                confidence_breakdown=confidence_breakdown,
                smplx_params=smplx_params,
                soma_params=soma_params,
                body_model_type=body_model_type,
            )
            self._session.person_tracks[tid] = pt

        # Mark inactive tracks
        for track in getattr(multi_result, "inactive_tracks", []):
            tid = track.get("track_id", -1) if isinstance(track, dict) else -1
            if tid >= 0:
                self._session.inactive_tracks.add(tid)

        # Load crossing spans from person dirs
        for pid, pt in self._session.person_tracks.items():
            if pt.person_dir:
                spans_path = pt.person_dir / "crossing_spans.json"
                if spans_path.is_file():
                    import json
                    try:
                        spans = json.loads(spans_path.read_text())
                        self._session.crossing_spans[pid] = [
                            tuple(s) for s in spans
                        ]
                    except Exception:
                        pass

    def _load_motion_params(self, person_dir: Path) -> tuple[dict | None, dict | None, str]:
        """Load motion params — tries GEM-X (hpe_results.pt / SOMA) first, falls back to GVHMR (SMPL-X)."""
        # GEM-X primary: hpe_results.pt (torch dict with body_params_global)
        try:
            from workers.gemx_worker import load_gemx_soma_output
            soma_params = load_gemx_soma_output(person_dir)
            if soma_params is not None:
                return None, soma_params, "soma"
            # Also check gemx_demo subdirectory
            gemx_demo = person_dir / "gemx_demo"
            if gemx_demo.is_dir():
                soma_params = load_gemx_soma_output(gemx_demo)
                if soma_params is not None:
                    return None, soma_params, "soma"
        except Exception:
            pass
        smplx_params = self._load_smplx_params_legacy(person_dir)
        if smplx_params is not None:
            return smplx_params, None, "smplx"
        return None, None, "smplx"

    def _load_smplx_params_legacy(self, person_dir: Path) -> dict | None:
        """Load SMPL-X parameters from hmr4d_results.pt for mesh rendering."""
        hmr4d_pt = person_dir / "demo" / "isolated_video" / "hmr4d_results.pt"
        if not hmr4d_pt.is_file():
            return None
        try:
            import torch

            results = torch.load(hmr4d_pt, map_location="cpu", weights_only=False)
            params = results.get("smpl_params_incam")
            if params and "body_pose" in params:
                K = results.get("K_fullimg")
                if K is not None and self._session.camera_K is None:
                    self._session.camera_K = K[0].numpy()
                return params
        except Exception:
            pass
        return None

    def _load_confidences_csv(self, person_dir: Path):
        """Load confidence.csv -> (overall_list, breakdown_dict)."""
        import csv

        csv_path = person_dir / "confidence.csv"
        if not csv_path.is_file():
            return None, None

        overall = []
        breakdown = {
            m: [] for m in [
                "detection", "visibility", "overlap",
                "shape", "motion", "overall",
            ]
        }
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                overall.append(float(row["overall"]))
                breakdown["detection"].append(float(row["detection"]))
                breakdown["visibility"].append(float(row["visible_kp"]))
                breakdown["overlap"].append(float(row["bbox_overlap"]))
                breakdown["shape"].append(float(row["shape_dist"]))
                breakdown["motion"].append(float(row["motion_dist"]))
                breakdown["overall"].append(float(row["overall"]))
        return overall if overall else None, breakdown if overall else None

    def _populate_tracks(self):
        """Populate track overview from session person_tracks."""
        if not self._session.person_tracks:
            return

        tracks: dict[int, np.ndarray] = {}
        for pid, track in self._session.person_tracks.items():
            if track.confidences is not None:
                tracks[pid] = np.array(track.confidences)
            else:
                tracks[pid] = np.ones(max(1, self._session.num_frames)) * 0.8
        self._track_overview.set_tracks(tracks)
        self._populate_track_markers()

    def _populate_track_markers(self):
        """Push keyframe, issue, correction, and crossing markers to track overview."""
        for pid, track in self._session.person_tracks.items():
            kf_frames = [kf["frame"] for kf in (track.keyframes or [])]
            verified = {
                kf["frame"] for kf in (track.keyframes or [])
                if kf.get("verified", False)
            }
            corr_frames: list[int] = []
            ct = self._session.correction_tracks.get(pid)
            if ct is not None and hasattr(ct, "corrections") and ct.corrections:
                corr_frames = [c.frame_index for c in ct.corrections]
            spans = self._session.crossing_spans.get(pid, [])
            self._track_overview.set_track_markers(
                pid,
                keyframes=kf_frames,
                verified_frames=verified,
                correction_frames=corr_frames,
                crossing_spans=spans,
            )
        # Issue flags from review scanner
        try:
            from views.identity_inspector import compute_review_issues

            issues = compute_review_issues(self._session)
            by_person: dict[int, list[int]] = {}
            for issue in issues:
                by_person.setdefault(issue.person_id, []).append(issue.frame)
            for pid, frames in by_person.items():
                self._track_overview.set_track_markers(pid, issue_frames=frames)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Open video
    # ------------------------------------------------------------------

    def _on_open_video(self):
        """Open Video menu action — load video into the active settings panel."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv);;All Files (*)",
        )
        if path:
            settings = self._pipeline_dock.current_settings
            if hasattr(settings, "_load_video"):
                settings._load_video(path)

    @property
    def session(self) -> Session:
        return self._session

    @property
    def log_panel(self) -> LogPanel:
        return self._log_panel

    def set_status(self, text: str):
        self._status_label.setText(text)

    def set_frame_info(self, current: int, total: int):
        self._frame_label.setText(f"Frame {current} / {total}")

    def set_fps_info(self, fps: float):
        self._fps_label.setText(f"{fps:.1f} FPS")

    # --- Undo / Redo ---

    def _setup_undo_redo(self):
        """Wire undo stack on_changed callback to keep Edit menu in sync."""
        self._session.undo_stack.on_changed = self._update_undo_redo_state

    def _update_undo_redo_state(self):
        """Enable/disable and label the Edit > Undo/Redo actions."""
        stack = self._session.undo_stack
        self._undo_action.setEnabled(stack.can_undo())
        self._redo_action.setEnabled(stack.can_redo())

        undo_desc = stack.peek_undo()
        self._undo_action.setText(
            f"&Undo {undo_desc}" if undo_desc else "&Undo"
        )
        redo_desc = stack.peek_redo()
        self._redo_action.setText(
            f"&Redo {redo_desc}" if redo_desc else "&Redo"
        )

    def _on_undo(self):
        desc = self._session.undo_stack.undo()
        if desc:
            self.set_status(f"Undo: {desc}")

    def _on_redo(self):
        desc = self._session.undo_stack.redo()
        if desc:
            self.set_status(f"Redo: {desc}")

    # --- Session I/O ---

    def _on_save_session(self):
        """Save session to JSON — auto-path if output_dir exists, else prompt."""
        if self._session_path:
            save_path = self._session_path
        elif self._session.output_dir:
            save_path = self._session.output_dir / "bodypipe_session.json"
        else:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Session", "", "Session Files (*.json);;All Files (*)"
            )
            if not path:
                return
            save_path = Path(path)

        try:
            self._session.save(save_path)
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save session:\n{e}")
            return

        self._session_path = save_path
        self._add_recent(save_path)
        self.set_status(f"Session saved to {save_path.name}")
        self.session_saved.emit(save_path)

    def _on_open_session(self):
        """Open session from JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Session", "", "Session Files (*.json);;All Files (*)"
        )
        if not path:
            return
        self._load_session(Path(path))

    def _load_session(self, session_path: Path):
        """Load session from path, update shared session, refresh UI."""
        try:
            loaded = Session.load(session_path)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load session:\n{e}")
            return

        # Copy all fields to shared session object (panels hold a reference)
        # Preserve the existing undo stack (with its on_changed callback)
        saved_undo_stack = self._session.undo_stack
        for attr in vars(loaded):
            setattr(self._session, attr, getattr(loaded, attr))
        self._session.undo_stack = saved_undo_stack
        self._session.undo_stack.clear()

        self._session_path = session_path
        self._add_recent(session_path)

        # If video exists, load into current settings panel
        if self._session.video_path and self._session.video_path.is_file():
            settings = self._pipeline_dock.current_settings
            if hasattr(settings, "_load_video"):
                settings._load_video(str(self._session.video_path))

        self.set_status(f"Session loaded from {session_path.name}")
        self.session_loaded.emit(self._session)

    # --- Session Library ---

    def _refresh_session_library(self):
        """Scan for sessions and refresh the library panel.

        Passes recent session paths as extra scan targets so sessions saved
        outside the standard GVHMR output directories still appear.
        """
        recent = self._get_recent()
        extra_paths = [Path(p) for p in recent]
        self._session_library.scan(extra_paths=extra_paths)

    # --- Recent Sessions ---

    def _get_recent(self) -> list[str]:
        """Get recent session paths from QSettings."""
        val = self._settings.value("recent_sessions", [])
        if isinstance(val, str):
            return [val] if val else []
        return list(val) if val else []

    def _add_recent(self, path: Path):
        """Add path to recent sessions list."""
        recent = self._get_recent()
        path_str = str(path)
        if path_str in recent:
            recent.remove(path_str)
        recent.insert(0, path_str)
        recent = recent[: self.MAX_RECENT]
        self._settings.setValue("recent_sessions", recent)
        self._update_recent_menu()

    def _update_recent_menu(self):
        """Rebuild the Recent Sessions submenu from QSettings."""
        self._recent_menu.clear()
        recent = self._get_recent()
        if not recent:
            action = self._recent_menu.addAction("(No recent sessions)")
            action.setEnabled(False)
            return
        for path_str in recent:
            action = self._recent_menu.addAction(Path(path_str).name)
            action.setData(path_str)
            action.triggered.connect(
                lambda checked, p=path_str: self._open_recent(p)
            )

    def _open_recent(self, path_str: str):
        """Open a session from the recent sessions list."""
        path = Path(path_str)
        if not path.is_file():
            QMessageBox.warning(
                self, "File Not Found", f"Session file not found:\n{path}"
            )
            recent = self._get_recent()
            if path_str in recent:
                recent.remove(path_str)
                self._settings.setValue("recent_sessions", recent)
                self._update_recent_menu()
            return
        self._load_session(path)

    # --- Keyboard Shortcuts ---

    def _on_keyboard_shortcuts(self):
        """Show the Keyboard Shortcuts dialog."""
        dlg = KeyboardShortcutsDialog(self)
        dlg.exec()

    # --- About ---

    def _on_about(self):
        """Show About dialog."""
        QMessageBox.about(
            self,
            "About bodypipe",
            "<h3>bodypipe — Motion Capture Studio</h3>"
            "<p>PySide6 interface for GVHMR body, hand, and face capture.</p>"
            "<p>Multi-person tracking with identity verification "
            "and pose correction.</p>",
        )
