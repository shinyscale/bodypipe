"""Main application window with tab widget, menu bar, status bar, and log panel."""

from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QStatusBar,
    QMenuBar,
    QDockWidget,
    QPlainTextEdit,
    QWidget,
    QLabel,
    QVBoxLayout,
    QFileDialog,
    QMessageBox,
    QMenu,
)
from PySide6.QtCore import Signal, QSettings, Qt, QByteArray
from PySide6.QtGui import QAction, QPalette, QColor

from models.session import Session
from views.single_person_tab import SinglePersonTab
from views.perf_capture_tab import PerfCaptureTab
from views.multi_person_tab import MultiPersonTab

# Theme colors matching Gradio dark mode
COLORS = {
    "bg_primary": "#1a1a2e",
    "bg_secondary": "#16213e",
    "bg_tertiary": "#0f3460",
    "accent": "#e94560",
    "accent_hover": "#ff6b6b",
    "text_primary": "#eaeaea",
    "text_secondary": "#a0a0a0",
    "border": "#2a2a4a",
    "success": "#4ecca3",
    "warning": "#ffd93d",
    "error": "#ff6b6b",
}


def _apply_dark_theme(app):
    """Apply dark palette + Fusion style."""
    from PySide6.QtWidgets import QApplication

    QApplication.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS["bg_primary"]))
    palette.setColor(QPalette.WindowText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.Base, QColor(COLORS["bg_secondary"]))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS["bg_tertiary"]))
    palette.setColor(QPalette.ToolTipBase, QColor(COLORS["bg_secondary"]))
    palette.setColor(QPalette.ToolTipText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.Text, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.Button, QColor(COLORS["bg_secondary"]))
    palette.setColor(QPalette.ButtonText, QColor(COLORS["text_primary"]))
    palette.setColor(QPalette.BrightText, QColor(COLORS["accent"]))
    palette.setColor(QPalette.Link, QColor(COLORS["accent"]))
    palette.setColor(QPalette.Highlight, QColor(COLORS["accent"]))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(COLORS["text_secondary"]))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(COLORS["text_secondary"]))
    app.setPalette(palette)

    # Fine-grained stylesheet for accent elements
    app.setStyleSheet(f"""
        QTabBar::tab {{
            background: {COLORS["bg_secondary"]};
            color: {COLORS["text_secondary"]};
            padding: 8px 16px;
            border: 1px solid {COLORS["border"]};
            border-bottom: none;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        QTabBar::tab:selected {{
            background: {COLORS["bg_primary"]};
            color: {COLORS["text_primary"]};
            border-bottom: 2px solid {COLORS["accent"]};
        }}
        QTabBar::tab:hover {{
            background: {COLORS["bg_tertiary"]};
        }}
        QPushButton {{
            background: {COLORS["bg_tertiary"]};
            color: {COLORS["text_primary"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 4px;
            padding: 6px 12px;
        }}
        QPushButton:hover {{
            background: {COLORS["accent"]};
        }}
        QPushButton:pressed {{
            background: {COLORS["accent_hover"]};
        }}
        QPushButton:disabled {{
            background: {COLORS["bg_secondary"]};
            color: {COLORS["text_secondary"]};
        }}
        QGroupBox {{
            border: 1px solid {COLORS["border"]};
            border-radius: 4px;
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
            background: {COLORS["bg_tertiary"]};
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
        QProgressBar {{
            background: {COLORS["bg_secondary"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 4px;
            text-align: center;
            color: {COLORS["text_primary"]};
        }}
        QProgressBar::chunk {{
            background: {COLORS["accent"]};
            border-radius: 3px;
        }}
        QDockWidget {{
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }}
        QDockWidget::title {{
            background: {COLORS["bg_secondary"]};
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
    """Main application window."""

    session_loaded = Signal(object)
    session_saved = Signal(Path)
    tab_changed = Signal(int)

    MAX_RECENT = 5

    def __init__(self, gvhmr_root: Path | None = None, parent=None):
        super().__init__(parent)
        self._session = Session()
        self._session_path: Path | None = None
        self._settings = QSettings("GVHMR", "bodypipe")
        self._gvhmr_root = gvhmr_root or Path(__file__).resolve().parent.parent / "GVHMR"

        self.setWindowTitle("bodypipe \u2014 Motion Capture Studio")
        self.setMinimumSize(1200, 700)

        _apply_dark_theme(self.parent() if self.parent() else self)

        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._setup_status_bar_toggle()
        self._setup_log_panel()
        self._restore_geometry()
        self._connect_tab_signals()

    def _setup_ui(self):
        """Create tab widget with real and placeholder tabs."""
        self._tabs = QTabWidget()
        self._tabs.currentChanged.connect(self.tab_changed.emit)

        # Tab 1: Single-person GVHMR body capture
        self._tab_single = SinglePersonTab(self._session, self._gvhmr_root)

        # Tab 2: Performance capture (body + hands + face)
        self._tab_perf = PerfCaptureTab(self._session, self._gvhmr_root)

        # Tab 3: Multi-person capture
        self._tab_multi = MultiPersonTab(self._session, self._gvhmr_root)

        self._tabs.addTab(self._tab_single, "GVHMR Body")
        self._tabs.addTab(self._tab_perf, "Performance Capture")
        self._tabs.addTab(self._tab_multi, "Multi-Person")

        self.setCentralWidget(self._tabs)

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

        undo_action = QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut("Ctrl+Shift+Z")
        edit_menu.addAction(redo_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        self._toggle_statusbar_action = QAction("Toggle &Status Bar", self)
        self._toggle_statusbar_action.setCheckable(True)
        self._toggle_statusbar_action.setChecked(True)
        view_menu.addAction(self._toggle_statusbar_action)

        self._toggle_log_action = QAction("Toggle &Log Panel", self)
        self._toggle_log_action.setShortcut("Ctrl+L")
        self._toggle_log_action.setCheckable(True)
        self._toggle_log_action.setChecked(True)
        view_menu.addAction(self._toggle_log_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_status_bar(self):
        """Create status bar with operation status, frame counter, FPS."""
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("Ready")
        self._frame_label = QLabel("")
        self._fps_label = QLabel("")

        self._status_bar.addWidget(self._status_label, 1)
        self._status_bar.addPermanentWidget(self._frame_label)
        self._status_bar.addPermanentWidget(self._fps_label)

    def _setup_status_bar_toggle(self):
        """Wire status bar toggle after both status bar and menu are created."""
        self._toggle_statusbar_action.toggled.connect(self._status_bar.setVisible)

    def _setup_log_panel(self):
        """Create collapsible log dock widget."""
        self._log_panel = LogPanel()
        self._log_dock = QDockWidget("Log", self)
        self._log_dock.setObjectName("LogDock")
        self._log_dock.setWidget(self._log_panel)
        self._log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._log_dock)

        self._toggle_log_action.toggled.connect(self._log_dock.setVisible)
        self._log_dock.visibilityChanged.connect(self._toggle_log_action.setChecked)

    def _restore_geometry(self):
        """Restore window geometry from settings."""
        geometry = self._settings.value("geometry")
        if geometry and isinstance(geometry, QByteArray):
            self.restoreGeometry(geometry)
        else:
            self.resize(1600, 900)

        state = self._settings.value("windowState")
        if state and isinstance(state, QByteArray):
            self.restoreState(state)

    def closeEvent(self, event):
        """Save window geometry on close."""
        self._settings.setValue("geometry", self.saveGeometry())
        self._settings.setValue("windowState", self.saveState())
        super().closeEvent(event)

    def _connect_tab_signals(self):
        """Wire tab signals to main window status bar and log panel."""
        for tab in (self._tab_single, self._tab_perf, self._tab_multi):
            tab.status_message.connect(self.set_status)
            tab.log_message.connect(
                lambda text, level: self._log_panel.append_line(text, level)
            )

    def _on_open_video(self):
        """Open Video menu action — load video into the active tab."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv);;All Files (*)",
        )
        if path:
            current = self._tabs.currentWidget()
            if hasattr(current, "_load_video"):
                current._load_video(path)

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

        # Copy all fields to shared session object (tabs hold a reference)
        for attr in vars(loaded):
            setattr(self._session, attr, getattr(loaded, attr))

        self._session_path = session_path
        self._add_recent(session_path)

        # If video exists, load into current tab to set up the UI
        if self._session.video_path and self._session.video_path.is_file():
            current = self._tabs.currentWidget()
            if hasattr(current, "_load_video"):
                current._load_video(str(self._session.video_path))

        self.set_status(f"Session loaded from {session_path.name}")
        self.session_loaded.emit(self._session)

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
