"""Multi-person capture tab shell.

Left sidebar with pipeline controls + multi-person settings, main viewport
with VideoPlayer, track overview, and horizontal splitter for identity
inspector / pose corrector placeholder panels.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QProgressBar,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QScrollArea,
    QSizePolicy,
    QToolButton,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QImage

from models.pipeline_config import PipelineConfig
from models.session import Session
from views.video_player import VideoPlayer
from views.confidence_timeline import ConfidenceTimeline
from views.identity_inspector import IdentityInspector
from views.single_person_tab import _DropArea, VIDEO_EXTENSIONS
from workers.pipeline_orchestrator import MultiPersonWorker


# Colors for up to 8 tracked persons — consistent palette across the app
PERSON_COLORS = [
    "#e94560", "#4ecca3", "#ffd93d", "#6c5ce7",
    "#00b894", "#fd79a8", "#0984e3", "#fdcb6e",
]


class _TrackOverview(QWidget):
    """Per-person confidence bars showing track quality over time."""

    person_clicked = Signal(int, int)  # (person_id, frame_index)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)
        self._timelines: dict[int, ConfidenceTimeline] = {}
        self._labels: dict[int, QLabel] = {}

    def set_tracks(self, tracks: dict[int, np.ndarray]):
        """Set confidence arrays per person_id. tracks = {pid: conf_array}."""
        self._clear()
        for pid, conf in sorted(tracks.items()):
            label = QLabel(f"Person {pid}")
            color = PERSON_COLORS[pid % len(PERSON_COLORS)]
            label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px;")
            self._labels[pid] = label
            self._layout.addWidget(label)

            timeline = ConfidenceTimeline()
            timeline.set_data(conf)
            timeline.frame_clicked.connect(lambda f, p=pid: self.person_clicked.emit(p, f))
            self._timelines[pid] = timeline
            self._layout.addWidget(timeline)

    def set_current_frame(self, frame: int):
        for tl in self._timelines.values():
            tl.set_current_frame(frame)

    def _clear(self):
        for w in list(self._timelines.values()) + list(self._labels.values()):
            self._layout.removeWidget(w)
            w.deleteLater()
        self._timelines.clear()
        self._labels.clear()


class MultiPersonTab(QWidget):
    """Third tab — multi-person capture with identity + pose correction shell."""

    status_message = Signal(str)
    log_message = Signal(str, str)  # (text, level)
    frame_changed = Signal(int)     # broadcast from video player
    person_selected = Signal(int)   # broadcast person selection

    def __init__(self, session: Session, gvhmr_root: Path, parent=None):
        super().__init__(parent)
        self._session = session
        self._gvhmr_root = gvhmr_root
        self._video_path: Path | None = None
        self._worker: MultiPersonWorker | None = None
        self._running = False

        self._setup_ui()
        self._connect_signals()
        self._set_running(False)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        outer = QHBoxLayout(self)

        # Main horizontal splitter: sidebar | content
        main_splitter = QSplitter(Qt.Horizontal)

        # ---- Left sidebar ----
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar.setMaximumWidth(320)
        sidebar.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Expanding)

        # Video input
        input_group = QGroupBox("Video Input")
        input_layout = QVBoxLayout(input_group)

        self._drop_area = _DropArea()
        input_layout.addWidget(self._drop_area)

        self._browse_btn = QPushButton("Browse...")
        input_layout.addWidget(self._browse_btn)

        self._video_info = QLabel("")
        self._video_info.setWordWrap(True)
        self._video_info.hide()
        input_layout.addWidget(self._video_info)

        sidebar_layout.addWidget(input_group)

        # Pipeline settings
        settings_group = QGroupBox("Pipeline Settings")
        settings_layout = QVBoxLayout(settings_group)

        self._static_cam = QCheckBox("Static camera")
        self._static_cam.setChecked(True)
        settings_layout.addWidget(self._static_cam)

        self._use_dpvo = QCheckBox("Use DPVO")
        self._use_dpvo.setChecked(False)
        settings_layout.addWidget(self._use_dpvo)

        focal_row = QHBoxLayout()
        focal_row.addWidget(QLabel("Focal length (mm):"))
        self._focal_mm = QDoubleSpinBox()
        self._focal_mm.setRange(10.0, 200.0)
        self._focal_mm.setValue(24.0)
        self._focal_mm.setSingleStep(1.0)
        focal_row.addWidget(self._focal_mm)
        settings_layout.addLayout(focal_row)

        sidebar_layout.addWidget(settings_group)

        # Multi-person specific settings
        mp_group = QGroupBox("Multi-Person")
        mp_layout = QVBoxLayout(mp_group)

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max persons:"))
        self._max_persons = QSpinBox()
        self._max_persons.setRange(1, 20)
        self._max_persons.setValue(8)
        self._max_persons.setToolTip("Maximum number of tracked persons")
        max_row.addWidget(self._max_persons)
        mp_layout.addLayout(max_row)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Confidence threshold:"))
        self._confidence_threshold = QDoubleSpinBox()
        self._confidence_threshold.setRange(0.0, 1.0)
        self._confidence_threshold.setValue(0.5)
        self._confidence_threshold.setSingleStep(0.05)
        self._confidence_threshold.setToolTip("Minimum confidence to keep a track")
        thresh_row.addWidget(self._confidence_threshold)
        mp_layout.addLayout(thresh_row)

        sidebar_layout.addWidget(mp_group)

        # Run / Cancel / Progress
        self._run_btn = QPushButton("Run Multi-Person Pipeline")
        self._run_btn.setEnabled(False)
        self._run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; }")
        sidebar_layout.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.hide()
        sidebar_layout.addWidget(self._cancel_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.hide()
        sidebar_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.hide()
        sidebar_layout.addWidget(self._progress_label)

        # Track overview
        track_group = QGroupBox("Track Overview")
        track_layout = QVBoxLayout(track_group)
        self._track_overview = _TrackOverview()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._track_overview)
        scroll.setMinimumHeight(80)
        track_layout.addWidget(scroll)
        sidebar_layout.addWidget(track_group)

        sidebar_layout.addStretch()

        main_splitter.addWidget(sidebar)

        # ---- Content area ----
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Vertical splitter: viewport on top, panels on bottom
        self._vert_splitter = QSplitter(Qt.Vertical)

        # Main viewport — VideoPlayer (with bbox overlay in future)
        self._video_player = VideoPlayer()
        self._vert_splitter.addWidget(self._video_player)

        # Bottom horizontal splitter: identity inspector | pose corrector
        self._bottom_splitter = QSplitter(Qt.Horizontal)

        # Identity Inspector (Phase 2)
        self._identity_panel = IdentityInspector(self._session)
        self._bottom_splitter.addWidget(self._identity_panel)

        # Placeholder: Pose Corrector
        self._pose_panel = QWidget()
        pose_layout = QVBoxLayout(self._pose_panel)
        pose_layout.addWidget(QLabel("Pose Corrector"))
        pose_layout.addWidget(QLabel("(will be implemented in Phase 3)"))
        pose_layout.addStretch()
        self._bottom_splitter.addWidget(self._pose_panel)

        self._bottom_splitter.setStretchFactor(0, 1)
        self._bottom_splitter.setStretchFactor(1, 1)

        self._vert_splitter.addWidget(self._bottom_splitter)
        self._vert_splitter.setStretchFactor(0, 2)
        self._vert_splitter.setStretchFactor(1, 1)

        content_layout.addWidget(self._vert_splitter)
        main_splitter.addWidget(content)

        main_splitter.setStretchFactor(0, 0)  # sidebar fixed
        main_splitter.setStretchFactor(1, 1)  # content stretches

        outer.addWidget(main_splitter)

    def _connect_signals(self):
        self._drop_area.file_dropped.connect(self._load_video)
        self._browse_btn.clicked.connect(self._on_browse)
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn.clicked.connect(self._on_cancel)

        # Frame sync: video player → track overview + identity inspector
        self._video_player.frame_changed.connect(self._on_frame_changed)

        # Track overview → seek + select person
        self._track_overview.person_clicked.connect(self._on_track_clicked)

        # Identity inspector → video player seek, person selection
        self._identity_panel.frame_requested.connect(self._video_player.seek)
        self._identity_panel.person_changed.connect(self._on_identity_person_changed)

    def _on_frame_changed(self, frame_idx: int):
        """Broadcast frame change to all sub-panels."""
        self._session.current_frame = frame_idx
        self._track_overview.set_current_frame(frame_idx)
        self._identity_panel.set_frame(frame_idx)
        self._show_frame(frame_idx)
        self.frame_changed.emit(frame_idx)

    def _on_track_clicked(self, person_id: int, frame_idx: int):
        """Select person and seek to frame from track overview."""
        self._session.selected_person = person_id
        self._identity_panel.set_person(person_id)
        self._video_player.seek(frame_idx)
        self.person_selected.emit(person_id)
        self.status_message.emit(f"Selected Person {person_id} at frame {frame_idx}")

    def _on_identity_person_changed(self, person_id: int):
        """Handle person change from identity inspector."""
        self._session.selected_person = person_id
        self.person_selected.emit(person_id)

    def _show_frame(self, frame_idx: int):
        """Display the current frame (with overlays in future)."""
        frame = self._video_player.get_raw_frame(frame_idx)
        if frame is not None:
            self._video_player.set_frame(frame)

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
        cap.release()

        self._video_path = video_path
        self._session.video_path = video_path
        self._session.num_frames = num_frames
        self._session.fps = fps
        self._session.img_width = width
        self._session.img_height = height

        self._video_info.setText(
            f"{video_path.name}\n"
            f"{width}x{height} | {num_frames} frames | {fps:.1f} FPS"
        )
        self._video_info.show()
        self._drop_area.setText(video_path.name)

        # Load into video player
        self._video_player.set_video(video_path, num_frames, fps)

        self._run_btn.setEnabled(True)
        self.status_message.emit(f"Loaded: {video_path.name}")
        self.log_message.emit(
            f"Loaded video: {video_path} ({width}x{height}, {num_frames} frames)",
            "info",
        )

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _on_run(self):
        if not self._video_path:
            return

        config = self.get_config()
        output_dir = self._gvhmr_root / "outputs" / "multi_person" / self._video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        self._worker = MultiPersonWorker(
            video_path=self._video_path,
            config=config,
            gvhmr_root=self._gvhmr_root,
            output_dir=output_dir,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self._set_running(True)
        self.status_message.emit("Running multi-person pipeline...")
        self.log_message.emit("Starting multi-person pipeline...", "info")

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
        self._use_dpvo.setEnabled(not running)
        self._focal_mm.setEnabled(not running)
        self._max_persons.setEnabled(not running)
        self._confidence_threshold.setEnabled(not running)
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
        self.status_message.emit("Multi-person pipeline complete")
        self.log_message.emit("Multi-person pipeline finished successfully", "info")

        output_dir = result.get("output_dir")
        if output_dir:
            self._session.output_dir = Path(output_dir)

        # Populate track overview and identity inspector from results
        self._populate_tracks()
        self._identity_panel.refresh()

    def _on_error(self, message: str):
        self._set_running(False)
        self._worker = None
        self.status_message.emit(f"Error: {message}")
        self.log_message.emit(f"Pipeline error: {message}", "error")

    def _populate_tracks(self):
        """Populate track overview from session person_tracks."""
        if not self._session.person_tracks:
            return

        tracks: dict[int, np.ndarray] = {}
        for pid, track in self._session.person_tracks.items():
            if track.confidences is not None:
                tracks[pid] = np.array(track.confidences)
            else:
                # Placeholder: uniform confidence
                tracks[pid] = np.ones(max(1, self._session.num_frames)) * 0.8
        self._track_overview.set_tracks(tracks)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_config(self) -> PipelineConfig:
        return PipelineConfig(
            mode="multi",
            static_cam=self._static_cam.isChecked(),
            use_dpvo=self._use_dpvo.isChecked(),
            focal_mm=self._focal_mm.value(),
            max_persons=self._max_persons.value(),
            confidence_threshold=self._confidence_threshold.value(),
        )

    def set_config(self, config: PipelineConfig):
        self._static_cam.setChecked(config.static_cam)
        self._use_dpvo.setChecked(config.use_dpvo)
        self._focal_mm.setValue(config.focal_mm)
        self._max_persons.setValue(config.max_persons)
        self._confidence_threshold.setValue(config.confidence_threshold)
