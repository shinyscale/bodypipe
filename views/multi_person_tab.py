"""Multi-person capture tab shell.

Left sidebar with MultiPipelineSettings + track overview, main viewport
with VideoPlayer, track overview, and horizontal splitter for identity
inspector / pose corrector panels.

Why composition: The pipeline settings (video input, solve settings,
multi-person params, run/cancel/progress) are extracted into
MultiPipelineSettings for reuse in the dock-based layout (Commits 1B/1C).
The tab remains the signal hub and owns viewport, panels, and track overview.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QSplitter,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QToolButton,
)
from PySide6.QtCore import Signal, Qt, QByteArray, QSettings

from models.pipeline_config import PipelineConfig
from models.session import Session
from views.video_player import VideoPlayer
from views.confidence_timeline import ConfidenceTimeline
from views.identity_inspector import IdentityInspector
from views.bbox_overlay import render_bbox_overlay, render_edit_preview
from views.mesh_viewport import MeshViewport
from views.pose_corrector_panel import PoseCorrectorPanel
from views.pipeline_settings import MultiPipelineSettings
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

    # Attributes that tests set directly and must be forwarded to _settings
    _SETTINGS_ATTRS = frozenset({'_video_path', '_worker'})

    def __init__(self, session: Session, gvhmr_root: Path, parent=None):
        super().__init__(parent)
        self._session = session
        self._gvhmr_root = gvhmr_root
        self._reprocess_worker: ReprocessWorker | None = None
        self._show_all_tracks = False
        self._edit_preview: dict | None = None

        self._qsettings = QSettings("bodypipe", "bodypipe")

        self._setup_ui()
        self._connect_signals()
        self._restore_splitter_state()

    def __getattr__(self, name):
        """Proxy attribute access to settings widget for backward compat."""
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

        # Pipeline settings (video input + settings + multi-person + run/cancel/progress)
        self._settings = MultiPipelineSettings(self._session, self._gvhmr_root)
        sidebar_layout.addWidget(self._settings)

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
        # Forward settings signals to tab signals
        self._settings.status_message.connect(self.status_message)
        self._settings.log_message.connect(self.log_message)
        self._settings.pipeline_finished.connect(self._on_tab_pipeline_finished)
        self._settings.video_loaded.connect(self._on_video_loaded)

        # Viewport mode switching
        self._video_mode_btn.clicked.connect(self._switch_to_video)
        self._mesh_mode_btn.clicked.connect(self._switch_to_mesh)

        # Frame sync: video player -> track overview + identity inspector
        self._video_player.frame_changed.connect(self._on_frame_changed)

        # Frame click -> identity inspector bbox editing
        self._video_player.frame_clicked.connect(self._identity_panel.on_frame_click)

        # Main mesh viewport -> pose corrector joint selection
        self._main_mesh_viewport.joint_clicked.connect(
            self._pose_corrector.set_joint
        )

        # Track overview -> seek + select person
        self._track_overview.person_clicked.connect(self._on_track_clicked)

        # Identity inspector -> video player seek, person selection, overlay refresh
        self._identity_panel.frame_requested.connect(self._video_player.seek)
        self._identity_panel.person_changed.connect(self._on_identity_person_changed)
        self._identity_panel.bbox_overlay_changed.connect(self._on_bbox_overlay_changed)
        self._identity_panel.keyframe_changed.connect(self._on_keyframe_changed)
        self._identity_panel.track_modified.connect(self._on_tracks_modified)
        self._identity_panel.reprocess_requested.connect(self._on_reprocess_requested)

        # Pose corrector -> video player seek
        self._pose_corrector.frame_requested.connect(self._video_player.seek)

        # Splitter layout persistence — save on any splitter move
        self._main_splitter.splitterMoved.connect(self._save_splitter_state)
        self._vert_splitter.splitterMoved.connect(self._save_splitter_state)
        self._bottom_splitter.splitterMoved.connect(self._save_splitter_state)

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
    # Video loaded handler
    # ------------------------------------------------------------------

    def _on_video_loaded(self, video_path):
        """Load video into video player after settings widget processes it."""
        self._video_player.set_video(
            video_path, self._session.num_frames, self._session.fps
        )

    # ------------------------------------------------------------------
    # Pipeline result handling
    # ------------------------------------------------------------------

    def _on_tab_pipeline_finished(self, result: dict):
        """Handle pipeline completion — load tracks and refresh UI."""
        output_dir = result.get("output_dir")
        if output_dir:
            self._session.output_dir = Path(output_dir)

        # Convert pipeline result to session.person_tracks
        multi_result = result.get("result")
        if multi_result is not None:
            self._load_person_tracks_from_result(multi_result)

        # Populate track overview and identity inspector from results
        self._populate_tracks()
        self._identity_panel.refresh()

    # ------------------------------------------------------------------
    # Splitter layout persistence
    # ------------------------------------------------------------------

    def _save_splitter_state(self):
        """Persist all splitter sizes to QSettings."""
        self._qsettings.setValue(
            "multi_person/main_splitter", self._main_splitter.saveState()
        )
        self._qsettings.setValue(
            "multi_person/vert_splitter", self._vert_splitter.saveState()
        )
        self._qsettings.setValue(
            "multi_person/bottom_splitter", self._bottom_splitter.saveState()
        )

    def _restore_splitter_state(self):
        """Restore splitter sizes from QSettings."""
        state = self._qsettings.value("multi_person/main_splitter")
        if state and isinstance(state, QByteArray):
            self._main_splitter.restoreState(state)

        state = self._qsettings.value("multi_person/vert_splitter")
        if state and isinstance(state, QByteArray):
            self._vert_splitter.restoreState(state)

        state = self._qsettings.value("multi_person/bottom_splitter")
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

        # Show progress using settings widget's progress bar
        self._settings._progress_bar.show()
        self._settings._progress_label.show()
        self.status_message.emit(
            f"Reprocessing {len(person_ids)} person(s)..."
        )
        self.log_message.emit(
            f"Reprocess started for persons: {person_ids}", "info"
        )

    def _on_reprocess_progress(self, fraction: float, stage: str):
        self._settings._progress_bar.setValue(int(fraction * 1000))
        self._settings._progress_label.setText(stage)
        self.status_message.emit(f"{stage} ({fraction:.0%})")

    def _on_reprocess_person_done(self, person_id: int):
        """Handle completion of a single person reprocess."""
        self._session.dirty_persons.discard(person_id)
        self._identity_panel.update_reprocess_button()
        self.log_message.emit(f"Person {person_id} reprocessed", "info")

    def _on_reprocess_finished(self, result: dict):
        """Handle reprocess worker completion — refresh all UI."""
        if self._reprocess_worker is not None:
            self._reprocess_worker.wait()
            self._reprocess_worker = None
        self._settings._progress_bar.hide()
        self._settings._progress_label.hide()
        self._settings._progress_bar.setValue(0)

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
        if self._reprocess_worker is not None:
            self._reprocess_worker.wait()
            self._reprocess_worker = None
        self._settings._progress_bar.hide()
        self._settings._progress_label.hide()
        self._settings._progress_bar.setValue(0)

        self.status_message.emit(f"Reprocess error: {message}")
        self.log_message.emit(f"Reprocess error: {message}", "error")

    # ------------------------------------------------------------------
    # Person track loading
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

            # Convert IdentityKeyframe objects to the dict format expected
            # by IdentityInspector
            keyframes = []
            if id_track and hasattr(id_track, "keyframes"):
                for kf in id_track.keyframes:
                    keyframes.append({
                        "frame": kf.frame_index,
                        "verified": kf.verified,
                        "confidence": getattr(kf, "confidence", None),
                    })

            # Load confidences from CSV if available
            confidences = None
            confidence_breakdown = None
            if person_dir:
                confidences, confidence_breakdown = self._load_confidences_csv(
                    person_dir
                )

            # Load SMPL-X parameters from hmr4d_results.pt
            smplx_params = None
            if person_dir:
                smplx_params = self._load_smplx_params(person_dir)

            pt = PersonTrack(
                person_id=tid,
                person_dir=person_dir,
                identity_track=id_track,
                confidences=confidences,
                bboxes=bboxes,
                keyframes=keyframes,
                confidence_breakdown=confidence_breakdown,
                smplx_params=smplx_params,
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

    def _load_smplx_params(self, person_dir: Path) -> dict | None:
        """Load SMPL-X parameters from hmr4d_results.pt for mesh rendering."""
        hmr4d_pt = person_dir / "demo" / "isolated_video" / "hmr4d_results.pt"
        if not hmr4d_pt.is_file():
            return None
        try:
            import torch

            results = torch.load(hmr4d_pt, map_location="cpu", weights_only=False)
            params = results.get("smpl_params_incam")
            if params and "body_pose" in params:
                # Also store camera intrinsics on session if available
                K = results.get("K_fullimg")
                if K is not None and self._session.camera_K is None:
                    # Use first frame's K (they're typically constant)
                    self._session.camera_K = K[0].numpy()
                return params
        except Exception:
            pass
        return None

    def _load_confidences_csv(self, person_dir: Path):
        """Load confidence.csv -> (overall_list, breakdown_dict)."""
        csv_path = person_dir / "confidence.csv"
        if not csv_path.is_file():
            return None, None

        import csv
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
                # Placeholder: uniform confidence
                tracks[pid] = np.ones(max(1, self._session.num_frames)) * 0.8
        self._track_overview.set_tracks(tracks)
