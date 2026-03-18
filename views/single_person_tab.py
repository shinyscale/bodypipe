"""Single-person GVHMR body capture tab.

Video in, motion out. Composes SinglePipelineSettings for the left panel
(video input, settings, run/cancel/progress) and adds output preview +
file list on the right.

Why composition: Self-contained settings widgets enable the transition from
tab-based to dock-based layout (Commits 1B/1C). The tab is a thin shell
that owns the right panel (output display) and delegates pipeline concerns
to the settings widget.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QMenu,
    QApplication,
)
from PySide6.QtCore import Signal, Qt

from models.pipeline_config import PipelineConfig
from models.session import Session
from views.video_player import VideoPlayer
from views.pipeline_settings import SinglePipelineSettings

# Re-export for backward compat — tests import these from here
from views.pipeline_settings import _DropArea, VIDEO_EXTENSIONS  # noqa: F401


class SinglePersonTab(QWidget):
    """First tab — single-person GVHMR body capture.

    Composes SinglePipelineSettings (left panel) with output preview and
    file list (right panel). The settings widget manages video input,
    pipeline settings, run/cancel/progress, and worker lifecycle.
    """

    status_message = Signal(str)
    log_message = Signal(str, str)  # (text, level)

    # Attributes that tests set directly and must be forwarded to _settings
    _SETTINGS_ATTRS = frozenset({'_video_path'})

    def __init__(self, session: Session, gvhmr_root: Path, parent=None):
        super().__init__(parent)
        self._session = session
        self._gvhmr_root = gvhmr_root

        self._setup_ui()
        self._connect_signals()

    def __getattr__(self, name):
        """Proxy attribute access to settings widget for backward compat.

        Why: Existing tests and AppWindow access internal attributes like
        _static_cam, _run_btn, _worker directly on the tab. This proxy
        transparently forwards those lookups to the composed settings widget.
        """
        settings = self.__dict__.get('_settings')
        if settings is not None and hasattr(settings, name):
            return getattr(settings, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Forward writes of settings-owned attrs to the settings widget."""
        if name in type(self)._SETTINGS_ATTRS:
            settings = self.__dict__.get('_settings')
            if settings is not None:
                setattr(settings, name, value)
                return
        super().__setattr__(name, value)

    def _create_settings(self):
        """Factory for the settings widget. Override in subclasses."""
        return SinglePipelineSettings(self._session, self._gvhmr_root)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        outer = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Left panel — pipeline settings widget
        self._settings = self._create_settings()
        splitter.addWidget(self._settings)

        # Right panel — output preview + file list
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)

        preview_group = QGroupBox("Output Preview")
        preview_layout = QVBoxLayout(preview_group)
        self._preview_player = VideoPlayer()
        preview_layout.addWidget(self._preview_player)
        right_layout.addWidget(preview_group, 1)

        files_group = QGroupBox("Output Files")
        files_layout = QVBoxLayout(files_group)
        self._file_list = QListWidget()
        self._file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        files_layout.addWidget(self._file_list)
        self._open_folder_btn = QPushButton("Download All")
        self._open_folder_btn.setEnabled(False)
        files_layout.addWidget(self._open_folder_btn)
        right_layout.addWidget(files_group)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter)

    @property
    def video_player(self) -> VideoPlayer:
        """Public access to the tab's VideoPlayer for status bar wiring."""
        return self._preview_player

    def _connect_signals(self):
        # Forward settings signals to tab signals (signal-to-signal)
        self._settings.status_message.connect(self.status_message)
        self._settings.log_message.connect(self.log_message)
        self._settings.pipeline_finished.connect(self._on_tab_pipeline_finished)

        # File list signals
        self._file_list.itemDoubleClicked.connect(self._on_file_double_click)
        self._file_list.customContextMenuRequested.connect(self._on_file_context_menu)
        self._open_folder_btn.clicked.connect(self._on_open_folder)

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    def get_config(self) -> PipelineConfig:
        """Return current settings as PipelineConfig."""
        return self._settings.get_config()

    def set_config(self, config: PipelineConfig):
        """Apply settings from PipelineConfig."""
        self._settings.set_config(config)

    def _load_video(self, path: str):
        """Load video — delegates to settings widget."""
        self._settings._load_video(path)

    # ------------------------------------------------------------------
    # Pipeline result handling
    # ------------------------------------------------------------------

    def _on_tab_pipeline_finished(self, result: dict):
        """Handle pipeline completion — populate output files and load preview."""
        output_dir = result.get("output_dir")
        if output_dir:
            self._session.output_dir = Path(output_dir)
            self._populate_output_files(Path(output_dir))

        sbs = result.get("side_by_side")
        if sbs and Path(sbs).is_file():
            self._load_preview(Path(sbs))
        else:
            incam = result.get("incam")
            if incam and Path(incam).is_file():
                self._load_preview(Path(incam))

    # ------------------------------------------------------------------
    # Output display
    # ------------------------------------------------------------------

    def _populate_output_files(self, output_dir: Path):
        """List all output files in the QListWidget."""
        self._file_list.clear()
        self._output_dir = output_dir
        self._open_folder_btn.setEnabled(True)

        if not output_dir.is_dir():
            return

        for f in sorted(output_dir.rglob("*")):
            if f.is_file():
                item = QListWidgetItem(str(f.relative_to(output_dir)))
                item.setData(Qt.UserRole, str(f))
                item.setToolTip(str(f))
                self._file_list.addItem(item)

    def _load_preview(self, video_path: Path):
        """Load a result video into the preview player."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        self._preview_player.set_video(video_path, num_frames, fps)

    def _on_file_double_click(self, item: QListWidgetItem):
        """Open file with system default application."""
        path = item.data(Qt.UserRole)
        if path and Path(path).is_file():
            os.startfile(path)

    def _on_file_context_menu(self, pos):
        item = self._file_list.itemAt(pos)
        if not item:
            return
        path = item.data(Qt.UserRole)
        if not path:
            return

        menu = QMenu(self)
        copy_action = menu.addAction("Copy Path")
        open_folder_action = menu.addAction("Open Containing Folder")

        action = menu.exec(self._file_list.mapToGlobal(pos))
        if action == copy_action:
            clipboard = QApplication.clipboard()
            if clipboard:
                clipboard.setText(path)
        elif action == open_folder_action:
            self._open_containing_folder(Path(path))

    def _on_open_folder(self):
        output_dir = getattr(self, "_output_dir", None)
        if output_dir and output_dir.is_dir():
            os.startfile(str(output_dir))

    @staticmethod
    def _open_containing_folder(file_path: Path):
        """Open the folder containing the file in the system file manager."""
        folder = file_path.parent
        if folder.is_dir():
            os.startfile(str(folder))
