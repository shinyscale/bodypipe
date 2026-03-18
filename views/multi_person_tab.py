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
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QProgressBar,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
)
from PySide6.QtCore import Signal, Qt, QByteArray, QSettings
from PySide6.QtGui import QPixmap, QImage

from models.pipeline_config import PipelineConfig
from models.session import Session
from views.video_player import VideoPlayer
from views.confidence_timeline import ConfidenceTimeline
from views.identity_inspector import IdentityInspector
from views.single_person_tab import _DropArea, VIDEO_EXTENSIONS
from views.bbox_overlay import render_bbox_overlay, render_edit_preview
from views.mesh_viewport import MeshViewport
from views.pose_corrector_panel import PoseCorrectorPanel
from workers.pipeline_orchestrator import MultiPersonWorker
from workers.reprocess_worker import ReprocessWorker


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
        self._reprocess_worker: ReprocessWorker | None = None
        self._running = False
        self._show_all_tracks = False
        self._edit_preview: dict | None = None

        self._settings = QSettings("bodypipe", "bodypipe")

        self._setup_ui()
        self._connect_signals()
        self._set_running(False)
        self._restore_splitter_state()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        outer = QHBoxLayout(self)

        # Main horizontal splitter: sidebar | content
        self._main_splitter = QSplitter(Qt.Horizontal)

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

        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Target FPS:"))
        self._target_fps = QDoubleSpinBox()
        self._target_fps.setRange(1.0, 120.0)
        self._target_fps.setValue(30.0)
        self._target_fps.setSingleStep(1.0)
        self._target_fps.setDecimals(1)
        self._target_fps.setToolTip("Target frame rate for preprocessing")
        fps_row.addWidget(self._target_fps)
        settings_layout.addLayout(fps_row)

        naming_row = QHBoxLayout()
        naming_row.addWidget(QLabel("FBX naming:"))
        self._fbx_naming = QComboBox()
        self._fbx_naming.addItems(["Mixamo (Cascadeur)", "UE5 Mannequin"])
        self._fbx_naming.setToolTip("Bone naming convention for FBX export")
        naming_row.addWidget(self._fbx_naming)
        settings_layout.addLayout(naming_row)

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

        self._use_inpainting = QCheckBox("SAM2 + ProPainter inpainting")
        self._use_inpainting.setChecked(True)
        self._use_inpainting.setToolTip(
            "Pixel-accurate isolation via segmentation + video inpainting. "
            "Required for front-crossings / heavy occlusion. Very slow (~5min/person)."
        )
        mp_layout.addWidget(self._use_inpainting)

        self._render_overlays = QCheckBox("Render per-person overlays")
        self._render_overlays.setChecked(False)
        self._render_overlays.setToolTip(
            "Render in-camera mesh overlay per person (slower)"
        )
        mp_layout.addWidget(self._render_overlays)

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

        self._main_splitter.addWidget(sidebar)

        # ---- Content area ----
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Vertical splitter: viewport on top, panels on bottom
        self._vert_splitter = QSplitter(Qt.Vertical)

        # Main viewport — switchable between VideoPlayer and MeshViewport
        viewport_container = QWidget()
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(2)

        # Toolbar row for viewport switching
        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(4, 2, 4, 0)

        self._video_mode_btn = QToolButton()
        self._video_mode_btn.setText("Video")
        self._video_mode_btn.setCheckable(True)
        self._video_mode_btn.setChecked(True)
        self._video_mode_btn.setToolTip("Show video with bbox overlay")
        toolbar_row.addWidget(self._video_mode_btn)

        self._mesh_mode_btn = QToolButton()
        self._mesh_mode_btn.setText("3D Mesh")
        self._mesh_mode_btn.setCheckable(True)
        self._mesh_mode_btn.setChecked(False)
        self._mesh_mode_btn.setToolTip("Show 3D mesh viewport")
        toolbar_row.addWidget(self._mesh_mode_btn)

        toolbar_row.addStretch()
        viewport_layout.addLayout(toolbar_row)

        # Stacked widget holding both viewport modes
        self._viewport_stack = QStackedWidget()
        self._video_player = VideoPlayer()
        self._viewport_stack.addWidget(self._video_player)  # index 0

        self._main_mesh_viewport = MeshViewport(gvhmr_root=self._gvhmr_root)
        self._main_mesh_viewport.set_session(self._session)
        self._viewport_stack.addWidget(self._main_mesh_viewport)  # index 1

        self._viewport_stack.setCurrentIndex(0)
        viewport_layout.addWidget(self._viewport_stack, stretch=1)

        self._vert_splitter.addWidget(viewport_container)

        # Bottom horizontal splitter: identity inspector | pose corrector
        self._bottom_splitter = QSplitter(Qt.Horizontal)

        # Identity Inspector (Phase 2)
        self._identity_panel = IdentityInspector(self._session)
        self._bottom_splitter.addWidget(self._identity_panel)

        # Pose Corrector Panel (Phase 3.4 — wraps MeshViewport with joint controls)
        self._pose_corrector = PoseCorrectorPanel(
            session=self._session, gvhmr_root=self._gvhmr_root,
        )
        self._mesh_viewport = self._pose_corrector.mesh_viewport
        self._bottom_splitter.addWidget(self._pose_corrector)

        self._bottom_splitter.setStretchFactor(0, 1)
        self._bottom_splitter.setStretchFactor(1, 1)

        self._vert_splitter.addWidget(self._bottom_splitter)
        self._vert_splitter.setStretchFactor(0, 2)
        self._vert_splitter.setStretchFactor(1, 1)

        content_layout.addWidget(self._vert_splitter)
        self._main_splitter.addWidget(content)

        self._main_splitter.setStretchFactor(0, 0)  # sidebar fixed
        self._main_splitter.setStretchFactor(1, 1)  # content stretches

        outer.addWidget(self._main_splitter)

    @property
    def video_player(self) -> "VideoPlayer":
        """Public access to the tab's VideoPlayer for status bar wiring."""
        return self._video_player

    @property
    def viewport_stack(self) -> QStackedWidget:
        """Public access to the viewport stack for testing."""
        return self._viewport_stack

    def _connect_signals(self):
        self._drop_area.file_dropped.connect(self._load_video)
        self._browse_btn.clicked.connect(self._on_browse)
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._static_cam.toggled.connect(self._on_static_cam_toggled)

        # Viewport mode switching
        self._video_mode_btn.clicked.connect(self._switch_to_video)
        self._mesh_mode_btn.clicked.connect(self._switch_to_mesh)

        # Frame sync: video player → track overview + identity inspector
        self._video_player.frame_changed.connect(self._on_frame_changed)

        # Frame click → identity inspector bbox editing
        self._video_player.frame_clicked.connect(self._identity_panel.on_frame_click)

        # Main mesh viewport → pose corrector joint selection
        self._main_mesh_viewport.joint_clicked.connect(
            self._pose_corrector.set_joint
        )

        # Track overview → seek + select person
        self._track_overview.person_clicked.connect(self._on_track_clicked)

        # Identity inspector → video player seek, person selection, overlay refresh
        self._identity_panel.frame_requested.connect(self._video_player.seek)
        self._identity_panel.person_changed.connect(self._on_identity_person_changed)
        self._identity_panel.bbox_overlay_changed.connect(self._on_bbox_overlay_changed)
        self._identity_panel.keyframe_changed.connect(self._on_keyframe_changed)
        self._identity_panel.track_modified.connect(self._on_tracks_modified)
        self._identity_panel.reprocess_requested.connect(self._on_reprocess_requested)

        # Pose corrector → video player seek
        self._pose_corrector.frame_requested.connect(self._video_player.seek)

        # Splitter layout persistence — save on any splitter move
        self._main_splitter.splitterMoved.connect(self._save_splitter_state)
        self._vert_splitter.splitterMoved.connect(self._save_splitter_state)
        self._bottom_splitter.splitterMoved.connect(self._save_splitter_state)

    # ------------------------------------------------------------------
    # Splitter layout persistence
    # ------------------------------------------------------------------

    def _save_splitter_state(self):
        """Persist all splitter sizes to QSettings."""
        self._settings.setValue(
            "multi_person/main_splitter", self._main_splitter.saveState()
        )
        self._settings.setValue(
            "multi_person/vert_splitter", self._vert_splitter.saveState()
        )
        self._settings.setValue(
            "multi_person/bottom_splitter", self._bottom_splitter.saveState()
        )

    def _restore_splitter_state(self):
        """Restore splitter sizes from QSettings."""
        state = self._settings.value("multi_person/main_splitter")
        if state and isinstance(state, QByteArray):
            self._main_splitter.restoreState(state)

        state = self._settings.value("multi_person/vert_splitter")
        if state and isinstance(state, QByteArray):
            self._vert_splitter.restoreState(state)

        state = self._settings.value("multi_person/bottom_splitter")
        if state and isinstance(state, QByteArray):
            self._bottom_splitter.restoreState(state)

    def _switch_to_video(self):
        """Switch main viewport to video + bbox overlay mode."""
        self._viewport_stack.setCurrentIndex(0)
        self._video_mode_btn.setChecked(True)
        self._mesh_mode_btn.setChecked(False)
        # Refresh video overlay for current frame
        self._show_frame(self._session.current_frame)

    def _switch_to_mesh(self):
        """Switch main viewport to 3D mesh mode."""
        self._viewport_stack.setCurrentIndex(1)
        self._mesh_mode_btn.setChecked(True)
        self._video_mode_btn.setChecked(False)
        # Pass current video frame for in-camera composite background
        frame = self._video_player.get_raw_frame(self._session.current_frame)
        self._main_mesh_viewport.set_video_frame(frame)

    def _on_frame_changed(self, frame_idx: int):
        """Broadcast frame change to all sub-panels."""
        self._session.current_frame = frame_idx
        self._track_overview.set_current_frame(frame_idx)
        self._identity_panel.set_frame(frame_idx)
        self._pose_corrector.on_frame_changed(frame_idx)
        # Pass raw video frame for in-camera composite background
        raw = self._video_player.get_raw_frame(frame_idx)
        self._main_mesh_viewport.set_video_frame(raw)
        self._main_mesh_viewport.on_frame_changed(frame_idx)
        self._show_frame(frame_idx)
        self.frame_changed.emit(frame_idx)

    def _on_track_clicked(self, person_id: int, frame_idx: int):
        """Select person and seek to frame from track overview."""
        self._session.selected_person = person_id
        self._identity_panel.set_person(person_id)
        self._pose_corrector.set_person(person_id)
        self._main_mesh_viewport.set_person(person_id)
        self._video_player.seek(frame_idx)
        self.person_selected.emit(person_id)
        self.status_message.emit(f"Selected Person {person_id} at frame {frame_idx}")

    def _on_identity_person_changed(self, person_id: int):
        """Handle person change from identity inspector — redraw overlay."""
        self._session.selected_person = person_id
        self._pose_corrector.set_person(person_id)
        self._main_mesh_viewport.set_person(person_id)
        self.person_selected.emit(person_id)
        self._show_frame(self._session.current_frame)

    def _on_bbox_overlay_changed(self, data: object):
        """Handle show-all-tracks toggle, edit preview, or other overlay changes."""
        if isinstance(data, dict):
            if "show_all" in data:
                self._show_all_tracks = data["show_all"]
            if "edit_preview" in data:
                self._edit_preview = data["edit_preview"]
        self._show_frame(self._session.current_frame)

    def _on_keyframe_changed(self, person_id: int, frame_idx: int):
        """Redraw overlay when keyframes change (may affect bbox corrections)."""
        self._show_frame(self._session.current_frame)

    def _on_tracks_modified(self):
        """Handle track modifications (swap, split, merge) — refresh everything."""
        self._populate_tracks()
        self._identity_panel.refresh()
        self._show_frame(self._session.current_frame)

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
    # Reprocess
    # ------------------------------------------------------------------

    def _on_reprocess_requested(self, person_ids: list):
        """Launch ReprocessWorker for dirty persons."""
        if self._reprocess_worker is not None:
            self.status_message.emit("Reprocess already running")
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

        self._progress_bar.show()
        self._progress_label.show()
        self.status_message.emit(
            f"Reprocessing {len(person_ids)} person(s)..."
        )
        self.log_message.emit(
            f"Reprocess started for persons: {person_ids}", "info"
        )

    def _on_reprocess_progress(self, fraction: float, stage: str):
        self._progress_bar.setValue(int(fraction * 1000))
        self._progress_label.setText(stage)
        self.status_message.emit(f"{stage} ({fraction:.0%})")

    def _on_reprocess_person_done(self, person_id: int):
        """Handle completion of a single person reprocess."""
        self._session.dirty_persons.discard(person_id)
        self._identity_panel.update_reprocess_button()
        self.log_message.emit(f"Person {person_id} reprocessed", "info")

    def _on_reprocess_finished(self, result: dict):
        """Handle reprocess worker completion — refresh all UI."""
        self._reprocess_worker = None
        self._progress_bar.hide()
        self._progress_label.hide()
        self._progress_bar.setValue(0)

        reprocessed = result.get("reprocessed", [])
        self._session.dirty_persons -= set(reprocessed)

        self._populate_tracks()
        self._identity_panel.refresh()
        self._show_frame(self._session.current_frame)

        self.status_message.emit(
            f"Reprocess complete: {len(reprocessed)} person(s) updated"
        )
        self.log_message.emit(
            f"Reprocess finished: {reprocessed}", "info"
        )

    def _on_reprocess_error(self, message: str):
        """Handle reprocess worker error."""
        self._reprocess_worker = None
        self._progress_bar.hide()
        self._progress_label.hide()
        self._progress_bar.setValue(0)

        self.status_message.emit(f"Reprocess error: {message}")
        self.log_message.emit(f"Reprocess error: {message}", "error")

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
        return self._gvhmr_root / "outputs" / "multi_person" / video_path.stem

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

        # Restore settings from previous run if available
        self._try_restore_config(video_path)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _on_run(self):
        if not self._video_path:
            return

        config = self.get_config()
        output_dir = self._gvhmr_root / "outputs" / "multi_person" / self._video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config to output directory for session restore
        config.save(output_dir / "solve_config.json")

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
        # DPVO is only enabled when not running AND static_cam is unchecked
        self._use_dpvo.setEnabled(not running and not self._static_cam.isChecked())
        self._focal_mm.setEnabled(not running)
        self._max_persons.setEnabled(not running)
        self._confidence_threshold.setEnabled(not running)
        self._target_fps.setEnabled(not running)
        self._fbx_naming.setEnabled(not running)
        self._render_overlays.setEnabled(not running)
        self._use_inpainting.setEnabled(not running)
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
            target_fps=self._target_fps.value(),
            fbx_naming=self._fbx_naming.currentText(),
            render_overlays=self._render_overlays.isChecked(),
            use_inpainting=self._use_inpainting.isChecked(),
        )

    def set_config(self, config: PipelineConfig):
        self._static_cam.setChecked(config.static_cam)
        self._use_dpvo.setChecked(config.use_dpvo)
        self._focal_mm.setValue(config.focal_mm)
        self._max_persons.setValue(config.max_persons)
        self._confidence_threshold.setValue(config.confidence_threshold)
        self._target_fps.setValue(config.target_fps)
        idx = self._fbx_naming.findText(config.fbx_naming)
        if idx >= 0:
            self._fbx_naming.setCurrentIndex(idx)
        self._render_overlays.setChecked(config.render_overlays)
        self._use_inpainting.setChecked(config.use_inpainting)
