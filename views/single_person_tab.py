"""Single-person GVHMR body capture tab.

Video in, motion out. Left panel for input/settings/run, right panel for
output preview and file list.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import cv2
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QProgressBar,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QMenu,
    QSizePolicy,
    QApplication,
)
from PySide6.QtCore import Signal, Qt, QMimeData
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent

from models.pipeline_config import PipelineConfig
from models.session import Session
from views.video_player import VideoPlayer
from workers.gvhmr_worker import GVHMRWorker


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


class _DropArea(QLabel):
    """Label that accepts drag-and-drop of video files."""

    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 120)
        self.setStyleSheet(
            "border: 2px dashed #555; border-radius: 8px; padding: 16px;"
        )
        self.setText("Drop video here\nor click Browse")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in VIDEO_EXTENSIONS:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in VIDEO_EXTENSIONS:
                self.file_dropped.emit(path)
                return


class SinglePersonTab(QWidget):
    """First tab — single-person GVHMR body capture."""

    status_message = Signal(str)
    log_message = Signal(str, str)  # (text, level)

    def __init__(self, session: Session, gvhmr_root: Path, parent=None):
        super().__init__(parent)
        self._session = session
        self._gvhmr_root = gvhmr_root
        self._video_path: Path | None = None
        self._worker: GVHMRWorker | None = None
        self._running = False

        self._setup_ui()
        self._connect_signals()
        self._set_running(False)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        outer = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # ---- Left panel ----
        left = QWidget()
        self._left_layout = left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Video input
        input_group = QGroupBox("Video Input")
        input_layout = QVBoxLayout(input_group)

        self._drop_area = _DropArea()
        input_layout.addWidget(self._drop_area)

        self._browse_btn = QPushButton("Browse...")
        input_layout.addWidget(self._browse_btn)

        self._thumbnail = QLabel()
        self._thumbnail.setAlignment(Qt.AlignCenter)
        self._thumbnail.setFixedHeight(120)
        self._thumbnail.hide()
        input_layout.addWidget(self._thumbnail)

        self._video_info = QLabel("")
        self._video_info.setWordWrap(True)
        self._video_info.hide()
        input_layout.addWidget(self._video_info)

        left_layout.addWidget(input_group)

        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)

        self._static_cam = QCheckBox("Static camera")
        self._static_cam.setChecked(True)
        self._static_cam.setToolTip(
            "Enable if the camera is fixed (tripod). "
            "Improves accuracy for static scenes."
        )
        settings_layout.addWidget(self._static_cam)

        self._use_dpvo = QCheckBox("Use DPVO")
        self._use_dpvo.setChecked(False)
        self._use_dpvo.setToolTip(
            "Use DPVO for camera estimation (more accurate, slower). "
            "Falls back to SimpleVO if disabled."
        )
        settings_layout.addWidget(self._use_dpvo)

        focal_row = QHBoxLayout()
        focal_row.addWidget(QLabel("Focal length (mm):"))
        self._focal_mm = QDoubleSpinBox()
        self._focal_mm.setRange(10.0, 200.0)
        self._focal_mm.setValue(24.0)
        self._focal_mm.setSingleStep(1.0)
        self._focal_mm.setToolTip("Camera focal length in mm (default 24)")
        focal_row.addWidget(self._focal_mm)
        settings_layout.addLayout(focal_row)

        left_layout.addWidget(settings_group)

        # Run / Cancel / Progress
        self._run_btn = QPushButton("Run GVHMR")
        self._run_btn.setEnabled(False)
        self._run_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 10px; }"
        )
        left_layout.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.hide()
        left_layout.addWidget(self._cancel_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.hide()
        left_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.hide()
        left_layout.addWidget(self._progress_label)

        left_layout.addStretch()

        # ---- Right panel ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)

        # Output preview
        preview_group = QGroupBox("Output Preview")
        preview_layout = QVBoxLayout(preview_group)
        self._preview_player = VideoPlayer()
        preview_layout.addWidget(self._preview_player)
        right_layout.addWidget(preview_group, 1)

        # Output files
        files_group = QGroupBox("Output Files")
        files_layout = QVBoxLayout(files_group)

        self._file_list = QListWidget()
        self._file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        files_layout.addWidget(self._file_list)

        self._open_folder_btn = QPushButton("Open Output Folder")
        self._open_folder_btn.setEnabled(False)
        files_layout.addWidget(self._open_folder_btn)

        right_layout.addWidget(files_group)

        # Assemble splitter
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter)

    @property
    def video_player(self) -> VideoPlayer:
        """Public access to the tab's VideoPlayer for status bar wiring."""
        return self._preview_player

    def _connect_signals(self):
        self._drop_area.file_dropped.connect(self._load_video)
        self._browse_btn.clicked.connect(self._on_browse)
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._file_list.itemDoubleClicked.connect(self._on_file_double_click)
        self._file_list.customContextMenuRequested.connect(self._on_file_context_menu)
        self._open_folder_btn.clicked.connect(self._on_open_folder)
        self._static_cam.toggled.connect(self._on_static_cam_toggled)

    def _on_static_cam_toggled(self, checked: bool):
        """Disable and uncheck DPVO when static camera is enabled."""
        if checked:
            self._use_dpvo.setChecked(False)
        self._use_dpvo.setEnabled(not checked and not self._running)

    # ------------------------------------------------------------------
    # Output directory mapping
    # ------------------------------------------------------------------

    def _output_dir_for_video(self, video_path: Path) -> Path:
        """Return the expected output directory for a given video."""
        return self._gvhmr_root / "outputs" / "demo" / video_path.stem

    def _try_restore_config(self, video_path: Path):
        """Restore settings from solve_config.json if a previous run exists."""
        config_path = self._output_dir_for_video(video_path) / "solve_config.json"
        if config_path.is_file():
            try:
                config = PipelineConfig.load(config_path)
                self.set_config(config)
                self.log_message.emit(
                    f"Restored settings from previous run: {config_path}",
                    "info",
                )
            except Exception:
                pass  # Ignore corrupt config files

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv);;All Files (*)",
        )
        if path:
            self._load_video(path)

    def _load_video(self, path: str):
        video_path = Path(path)
        if not video_path.is_file():
            self.status_message.emit(f"File not found: {path}")
            return

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.status_message.emit(f"Cannot open video: {path}")
            return

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = num_frames / fps if fps > 0 else 0

        # Read first frame for thumbnail
        ret, frame = cap.read()
        cap.release()

        self._video_path = video_path
        self._session.video_path = video_path
        self._session.num_frames = num_frames
        self._session.fps = fps
        self._session.img_width = width
        self._session.img_height = height

        # Show thumbnail
        if ret and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled = pixmap.scaled(200, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self._thumbnail.setPixmap(scaled)
            self._thumbnail.show()

        # Show video info
        self._video_info.setText(
            f"{video_path.name}\n"
            f"{width}x{height} | {num_frames} frames | {fps:.1f} FPS | {duration:.1f}s"
        )
        self._video_info.show()

        self._drop_area.setText(video_path.name)
        self._run_btn.setEnabled(True)
        self.status_message.emit(f"Loaded: {video_path.name}")
        self.log_message.emit(
            f"Loaded video: {video_path} ({width}x{height}, {num_frames} frames, {fps:.1f} FPS)",
            "info",
        )

        # Restore settings from previous run if available
        self._try_restore_config(video_path)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _on_run(self):
        if not self._video_path:
            return

        config = PipelineConfig(
            mode="single",
            static_cam=self._static_cam.isChecked(),
            use_dpvo=self._use_dpvo.isChecked(),
            focal_mm=self._focal_mm.value(),
        )

        # Save config to output directory for session restore
        output_dir = self._output_dir_for_video(self._video_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        config.save(output_dir / "solve_config.json")

        self._worker = GVHMRWorker(self._video_path, config, self._gvhmr_root)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self._set_running(True)
        self.status_message.emit("Running GVHMR pipeline...")
        self.log_message.emit("Starting GVHMR pipeline...", "info")

    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()
            self._set_running(False)
            self.status_message.emit("Pipeline cancelled")
            self.log_message.emit("Pipeline cancelled by user", "warning")

    def _set_running(self, running: bool):
        self._running = running
        self._run_btn.setEnabled(not running and self._video_path is not None)
        self._cancel_btn.setVisible(running)
        self._progress_bar.setVisible(running)
        self._progress_label.setVisible(running)
        self._browse_btn.setEnabled(not running)
        self._static_cam.setEnabled(not running)
        # DPVO is only enabled when not running AND static_cam is unchecked
        self._use_dpvo.setEnabled(not running and not self._static_cam.isChecked())
        self._focal_mm.setEnabled(not running)
        if not running:
            self._progress_bar.setValue(0)
            self._progress_label.setText("")

    def _on_progress(self, fraction: float, stage: str):
        self._progress_bar.setValue(int(fraction * 1000))
        self._progress_label.setText(stage)
        self.status_message.emit(f"{stage} ({fraction:.0%})")

    def _on_log_line(self, line: str):
        self.log_message.emit(line, "info")

    def _on_finished(self, result: dict):
        self._set_running(False)
        self._worker = None
        self.status_message.emit("GVHMR pipeline complete")
        self.log_message.emit("Pipeline finished successfully", "info")

        output_dir = result.get("output_dir")
        if output_dir:
            self._session.output_dir = Path(output_dir)
            self._populate_output_files(Path(output_dir))

        # Load side-by-side preview if available
        sbs = result.get("side_by_side")
        if sbs and Path(sbs).is_file():
            self._load_preview(Path(sbs))
        else:
            # Try incam as fallback
            incam = result.get("incam")
            if incam and Path(incam).is_file():
                self._load_preview(Path(incam))

    def _on_error(self, message: str):
        self._set_running(False)
        self._worker = None
        self.status_message.emit(f"Error: {message}")
        self.log_message.emit(f"Pipeline error: {message}", "error")

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

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def get_config(self) -> PipelineConfig:
        """Return current settings as PipelineConfig."""
        return PipelineConfig(
            mode="single",
            static_cam=self._static_cam.isChecked(),
            use_dpvo=self._use_dpvo.isChecked(),
            focal_mm=self._focal_mm.value(),
        )

    def set_config(self, config: PipelineConfig):
        """Apply settings from PipelineConfig."""
        self._static_cam.setChecked(config.static_cam)
        self._use_dpvo.setChecked(config.use_dpvo)
        self._focal_mm.setValue(config.focal_mm)
