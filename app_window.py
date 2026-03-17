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
)
from PySide6.QtCore import Signal, QSettings, Qt, QByteArray
from PySide6.QtGui import QAction, QPalette, QColor

from models.session import Session
from views.single_person_tab import SinglePersonTab

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

    def __init__(self, gvhmr_root: Path | None = None, parent=None):
        super().__init__(parent)
        self._session = Session()
        self._settings = QSettings("GVHMR", "bodypipe")
        self._gvhmr_root = gvhmr_root or Path(__file__).resolve().parent.parent / "GVHMR"

        self.setWindowTitle("bodypipe \u2014 Motion Capture Studio")
        self.setMinimumSize(1200, 700)

        _apply_dark_theme(self.parent() if self.parent() else self)

        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        self._setup_log_panel()
        self._restore_geometry()
        self._connect_tab_signals()

    def _setup_ui(self):
        """Create tab widget with real and placeholder tabs."""
        self._tabs = QTabWidget()
        self._tabs.currentChanged.connect(self.tab_changed.emit)

        # Tab 1: Single-person GVHMR body capture
        self._tab_single = SinglePersonTab(self._session, self._gvhmr_root)

        # Placeholder tabs — will be replaced with real widgets in later tasks
        self._tab_perf = QWidget()
        self._tab_perf.setLayout(QVBoxLayout())
        self._tab_perf.layout().addWidget(QLabel("Full Performance Capture (coming soon)"))

        self._tab_multi = QWidget()
        self._tab_multi.setLayout(QVBoxLayout())
        self._tab_multi.layout().addWidget(QLabel("Multi-Person Capture (coming soon)"))

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
        file_menu.addAction(open_session_action)

        save_session_action = QAction("&Save Session", self)
        save_session_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_session_action)

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

        self._toggle_log_action = QAction("Toggle &Log Panel", self)
        self._toggle_log_action.setShortcut("Ctrl+L")
        self._toggle_log_action.setCheckable(True)
        self._toggle_log_action.setChecked(True)
        view_menu.addAction(self._toggle_log_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
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
        self._tab_single.status_message.connect(self.set_status)
        self._tab_single.log_message.connect(
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
