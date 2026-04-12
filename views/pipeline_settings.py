"""Pipeline settings widgets — extracted from tab left panels for dock-based layout.

Why: Self-contained settings widgets enable the transition from tab-based to
dock-based layout (Commits 1B/1C). Each widget manages video input, pipeline
settings, run/cancel/progress, and worker lifecycle as a single composable unit.
Extracting them first ensures the refactor is incremental and testable.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QProgressBar,
    QFileDialog,
    QRadioButton,
    QButtonGroup,
    QComboBox,
    QFrame,
    QSizePolicy,
    QScrollArea,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent

from models.pipeline_config import PipelineConfig
from models.session import Session
from views.widgets import CollapsibleSection
from workers.gvhmr_worker import GVHMRWorker
from workers.gemx_worker import GEMXWorker
from workers.pipeline_orchestrator import FullPipelineWorker, MultiPersonWorker


# Sensitivity combo entries (display text, internal key).
_SENSITIVITY_OPTIONS = [
    ("Low (strict)", "low"),
    ("Medium (default)", "medium"),
    ("High (lenient)", "high"),
]


def _sensitivity_index_for_key(key: str) -> int:
    for i, (_, k) in enumerate(_SENSITIVITY_OPTIONS):
        if k == key:
            return i
    return 1  # medium


def _sensitivity_key_for_index(idx: int) -> str:
    if 0 <= idx < len(_SENSITIVITY_OPTIONS):
        return _SENSITIVITY_OPTIONS[idx][1]
    return "medium"

# Estimation backend display names → config values
_BACKEND_OPTIONS = [
    ("GVHMR (SMPL-X)", "gvhmr", "smplx"),
    ("GEM-X (SOMA)", "gemx", "soma"),
]


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


class _DropArea(QLabel):
    """Label that accepts drag-and-drop of video files."""

    file_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 120)
        from theme import COLORS
        self.setStyleSheet(
            f"border: 2px dashed {COLORS['border']}; border-radius: 8px; padding: 16px;"
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


# =========================================================================
# Single-person pipeline settings
# =========================================================================


class SinglePipelineSettings(QWidget):
    """Pipeline settings for single-person GVHMR body capture.

    Contains video input, body capture settings, run/cancel/progress, and
    worker lifecycle. Emits signals for status updates and pipeline results.
    """

    status_message = Signal(str)
    log_message = Signal(str, str)  # (text, level)
    pipeline_finished = Signal(dict)
    pipeline_error = Signal(str)
    video_loaded = Signal(object)  # Path

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
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Outer layout: scroll body (stretch=1) + sticky footer (stretch=0).
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.NoFrame)
        self._scroll_area.setMinimumHeight(200)
        scroll_inner = QWidget()
        self._left_layout = left_layout = QVBoxLayout(scroll_inner)
        left_layout.setContentsMargins(8, 8, 8, 8)
        self._scroll_area.setWidget(scroll_inner)
        outer.addWidget(self._scroll_area, 1)

        # ------------------------------------------------------------------
        # Scrollable body
        # ------------------------------------------------------------------

        # Video input — not collapsible; drop area is the primary affordance.
        input_container = QWidget()
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 4)
        input_title = QLabel("Video Input")
        input_title.setStyleSheet("font-weight: bold; padding: 4px 2px;")
        input_layout.addWidget(input_title)

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

        left_layout.addWidget(input_container)

        # Pipeline Settings — collapsible, expanded.
        self._settings_section = CollapsibleSection("Pipeline Settings", collapsed=False)
        settings_layout = self._settings_section.content_layout

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self._backend_combo = QComboBox()
        for label, _be, _bm in _BACKEND_OPTIONS:
            self._backend_combo.addItem(label)
        self._backend_combo.setToolTip(
            "GVHMR: multi-stage SMPL-X body capture (body, hands, face separate)\n"
            "GEM-X: single-pass SOMA-77 capture (body + hands + face unified)"
        )
        backend_row.addWidget(self._backend_combo)
        settings_layout.addLayout(backend_row)

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

        left_layout.addWidget(self._settings_section)

        # Trailing stretch keeps the sections pinned to the top while
        # the scroll area grows to fill any available vertical space.
        # Subclasses insert further sections before this stretch.
        left_layout.addStretch()

        # ------------------------------------------------------------------
        # Sticky footer — run / cancel / progress are never scrolled away.
        # ------------------------------------------------------------------
        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(8, 4, 8, 8)
        footer_layout.setSpacing(4)

        self._run_btn = QPushButton("Run GVHMR")
        self._run_btn.setEnabled(False)
        self._run_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 10px; }"
        )
        footer_layout.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.hide()
        footer_layout.addWidget(self._cancel_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.hide()
        footer_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.hide()
        footer_layout.addWidget(self._progress_label)

        outer.addWidget(footer, 0)

    def _insert_before_stretch(self, widget: QWidget) -> None:
        """Insert ``widget`` into ``_left_layout`` just before the trailing stretch."""
        self._left_layout.insertWidget(self._left_layout.count() - 1, widget)

    def _selected_backend(self) -> tuple[str, str]:
        """Return (estimation_backend, body_model) from combo selection."""
        _, be, bm = _BACKEND_OPTIONS[self._backend_combo.currentIndex()]
        return be, bm

    def _connect_signals(self):
        self._drop_area.file_dropped.connect(self._load_video)
        self._browse_btn.clicked.connect(self._on_browse)
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._static_cam.toggled.connect(self._on_static_cam_toggled)
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)

    def _on_backend_changed(self, _index: int):
        """Update run button label when backend changes."""
        label, _be, _bm = _BACKEND_OPTIONS[self._backend_combo.currentIndex()]
        backend_name = label.split(" ")[0]  # "GVHMR" or "GEM-X"
        self._run_btn.setText(f"Run {backend_name}")

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

        # Notify listeners (e.g., tab may load video into player)
        self.video_loaded.emit(video_path)

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def _on_run(self):
        if not self._video_path:
            return

        be, bm = self._selected_backend()
        config = PipelineConfig(
            mode="single",
            static_cam=self._static_cam.isChecked(),
            use_dpvo=self._use_dpvo.isChecked(),
            focal_mm=self._focal_mm.value(),
            estimation_backend=be,
            body_model=bm,
        )

        # Save config to output directory for session restore
        output_dir = self._output_dir_for_video(self._video_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        config.save(output_dir / "solve_config.json")

        if be == "gemx":
            gemx_root = self._gvhmr_root.parent / "GEM-X"
            self._worker = GEMXWorker(
                video_path=self._video_path,
                config=config,
                gemx_root=gemx_root,
                output_dir=output_dir,
                fps=config.target_fps,
            )
            backend_label = "GEM-X"
        else:
            self._worker = GVHMRWorker(self._video_path, config, self._gvhmr_root)
            backend_label = "GVHMR"

        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self._set_running(True)
        self.status_message.emit(f"Running {backend_label} pipeline...")
        self.log_message.emit(f"Starting {backend_label} pipeline...", "info")
        self.log_message.emit(
            f"  Backend: {be} | Body model: {bm} | Worker: {type(self._worker).__name__}",
            "info",
        )

    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()
            self._worker.wait()
            self._worker = None
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
        self._backend_combo.setEnabled(not running)
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
        if self._worker is not None:
            self._worker.wait()
            self._worker = None
        # Log which backend actually produced results
        if "soma_params" in result:
            actual = "GEM-X (SOMA)"
            poses = result["soma_params"].get("poses")
            n = poses.shape[0] if poses is not None and hasattr(poses, "shape") else "?"
            self.log_message.emit(
                f"Pipeline finished — backend confirmed: {actual}, {n} frames", "info",
            )
        else:
            actual = "GVHMR (SMPL-X)"
            self.log_message.emit(
                f"Pipeline finished — backend confirmed: {actual}", "info",
            )
        self.status_message.emit(f"{actual} pipeline complete")
        self.pipeline_finished.emit(result)

    def _on_error(self, message: str):
        self._set_running(False)
        if self._worker is not None:
            self._worker.wait()
            self._worker = None
        self.status_message.emit(f"Error: {message}")
        self.log_message.emit(f"Pipeline error: {message}", "error")
        self.pipeline_error.emit(message)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def get_config(self) -> PipelineConfig:
        """Return current settings as PipelineConfig."""
        be, bm = self._selected_backend()
        return PipelineConfig(
            mode="single",
            static_cam=self._static_cam.isChecked(),
            use_dpvo=self._use_dpvo.isChecked(),
            focal_mm=self._focal_mm.value(),
            estimation_backend=be,
            body_model=bm,
        )

    def set_config(self, config: PipelineConfig):
        """Apply settings from PipelineConfig."""
        self._static_cam.setChecked(config.static_cam)
        self._use_dpvo.setChecked(config.use_dpvo)
        self._focal_mm.setValue(config.focal_mm)
        # Restore backend selection
        for i, (_, be, _bm) in enumerate(_BACKEND_OPTIONS):
            if be == config.estimation_backend:
                self._backend_combo.setCurrentIndex(i)
                break


# =========================================================================
# Perf capture stage mapping
# =========================================================================


# Mapping from internal FullPipelineWorker stage labels to user-visible names.
_INTERNAL_TO_USER_STAGE: dict[str, str] = {
    "Preprocessing": "Body",
    "GVHMR body solve": "Body",
    "SMPLest-X hand solve": "Hands",
    "Merging body + hands": "Hands",
    "Motion refinement": "Motion",
    "Face pipeline": "Face",
    "BVH/FBX conversion": "Export",
    "Rendering": "Export",
}


def compute_visible_stages(
    use_hands: bool, use_face: bool, use_motion: bool = False
) -> list[str]:
    """Return ordered list of user-visible pipeline stage names.

    Stages are dynamic — Hands, Motion, and Face only appear when enabled.
    Body and Export are always present. ``use_motion`` controls the
    motion-refinement stage (spring filter in v1; dormant physics path
    also lives at the same slot).
    """
    stages = ["Body"]
    if use_hands:
        stages.append("Hands")
    if use_motion:
        stages.append("Motion")
    if use_face:
        stages.append("Face")
    stages.append("Export")
    return stages


def map_stage_label(
    internal_label: str,
    use_hands: bool,
    use_face: bool,
    use_motion: bool = False,
) -> str | None:
    """Map internal worker stage label to user-visible stage name.

    Returns ``None`` when the label doesn't correspond to a stage
    transition (sub-progress message or disabled-stage emission that
    should be swallowed).
    """
    user_stage = _INTERNAL_TO_USER_STAGE.get(internal_label)
    if user_stage is None:
        return None
    # When hands disabled, merge is instant GVHMR-only extraction -> Export
    if user_stage == "Hands" and not use_hands:
        return "Export"
    # When motion refinement disabled, swallow the stage emission
    if user_stage == "Motion" and not use_motion:
        return None
    # When face disabled, the worker briefly emits "Face pipeline" then skips
    if user_stage == "Face" and not use_face:
        return None
    return user_stage


# =========================================================================
# Performance capture pipeline settings
# =========================================================================


class PerfPipelineSettings(SinglePipelineSettings):
    """Pipeline settings for full body + hands + face performance capture.

    Extends SinglePipelineSettings with hand capture mode, face capture toggle,
    and pipeline output settings. Uses FullPipelineWorker for multi-stage
    orchestration.
    """

    def _setup_ui(self):
        super()._setup_ui()
        self._add_hand_face_settings()

    def _add_hand_face_settings(self):
        """Insert hand/face/motion/output sections into the scroll body.

        Appended before the trailing stretch that the base class placed at
        the bottom of ``_left_layout`` so sections remain top-aligned.
        """
        # Hand capture group — collapsible, expanded.
        hand_section = CollapsibleSection("Hand Capture", collapsed=False)
        hand_layout = hand_section.content_layout

        self._use_hands = QCheckBox("Enable hand capture")
        self._use_hands.setChecked(True)
        self._use_hands.setToolTip("Capture hand pose using SMPLest-X")
        hand_layout.addWidget(self._use_hands)

        self._hand_mode_group = QButtonGroup(self)
        self._hand_hybrid = QRadioButton("Hybrid (GVHMR body + SMPLest-X hands)")
        self._hand_hybrid.setChecked(True)
        self._hand_hybrid.setToolTip(
            "Use GVHMR for body capture, SMPLest-X for hands only (recommended)"
        )
        self._hand_smplestx = QRadioButton("SMPLest-X only")
        self._hand_smplestx.setToolTip(
            "Use SMPLest-X for full body + hands (single framework)"
        )
        self._hand_mode_group.addButton(self._hand_hybrid)
        self._hand_mode_group.addButton(self._hand_smplestx)
        hand_layout.addWidget(self._hand_hybrid)
        hand_layout.addWidget(self._hand_smplestx)

        # Hand source (SMPLest-X vs HaMeR)
        self._hand_source_group = QButtonGroup(self)
        self._hand_src_smplestx = QRadioButton("SMPLest-X (default)")
        self._hand_src_smplestx.setChecked(True)
        self._hand_src_smplestx.setToolTip("Use SMPLest-X for hand reconstruction")
        self._hand_src_hamer = QRadioButton("HaMeR")
        self._hand_src_hamer.setToolTip(
            "Use HaMeR for dedicated hand mesh recovery (better fingers)"
        )
        self._hand_source_group.addButton(self._hand_src_smplestx)
        self._hand_source_group.addButton(self._hand_src_hamer)
        hand_layout.addWidget(QLabel("Hand source:"))
        hand_layout.addWidget(self._hand_src_smplestx)
        hand_layout.addWidget(self._hand_src_hamer)

        # Enable/disable radio buttons based on hand checkbox
        self._use_hands.toggled.connect(self._hand_hybrid.setEnabled)
        self._use_hands.toggled.connect(self._hand_smplestx.setEnabled)
        self._use_hands.toggled.connect(self._hand_src_smplestx.setEnabled)
        self._use_hands.toggled.connect(self._hand_src_hamer.setEnabled)

        self._insert_before_stretch(hand_section)

        # Face capture group — collapsed by default (most users don't touch it).
        face_section = CollapsibleSection("Face Capture", collapsed=True)
        face_layout = face_section.content_layout

        self._use_face = QCheckBox("Enable face capture")
        self._use_face.setChecked(False)
        self._use_face.setToolTip(
            "Capture facial expressions using MediaPipe (ARKit blendshapes)"
        )
        face_layout.addWidget(self._use_face)

        self._use_vitpose_face = QCheckBox("Use ViTPose face crops")
        self._use_vitpose_face.setChecked(True)
        self._use_vitpose_face.setToolTip(
            "Use ViTPose keypoints for tight face crops (better for full-body shots)"
        )
        face_layout.addWidget(self._use_vitpose_face)

        self._insert_before_stretch(face_section)

        # Motion refinement group (replaces disabled Physics Refinement).
        motion_section = CollapsibleSection("Motion Refinement", collapsed=False)
        motion_layout = motion_section.content_layout

        self._use_camera_stabilize = QCheckBox("Stabilize camera drift")
        self._use_camera_stabilize.setChecked(True)
        self._use_camera_stabilize.setToolTip(
            "Re-derive world-space body params using smoothed camera trajectory.\n"
            "Removes drift caused by camera movement (dolly, orbit). Uses the\n"
            "Camera Smoothing preset to control smoothing strength."
        )
        motion_layout.addWidget(self._use_camera_stabilize)

        _stab_sep = QFrame()
        _stab_sep.setFrameShape(QFrame.HLine)
        _stab_sep.setFrameShadow(QFrame.Sunken)
        motion_layout.addWidget(_stab_sep)

        self._use_spring = QCheckBox("Enable spring-based refinement")
        self._use_spring.setChecked(False)
        self._use_spring.setToolTip(
            "Per-joint critically-damped spring filter applied to body motion in\n"
            "quaternion log space. Adds weight and follow-through (Lieberman-style)\n"
            "without rigid-body simulation. CPU-only, deterministic, fast."
        )
        motion_layout.addWidget(self._use_spring)

        motion_layout.addWidget(QLabel("Spring preset:"))
        self._spring_preset = QComboBox()
        self._spring_preset.addItems(
            [
                "Light (snappier)",
                "Moderate (balanced)",
                "Heavy (loose follow-through)",
            ]
        )
        self._spring_preset.setCurrentIndex(1)
        self._spring_preset.setToolTip(
            "Light: higher stiffness, less filtering, snappier response.\n"
            "Moderate: baseline gains from the per-joint proximal->distal table.\n"
            "Heavy: softer stiffness, more follow-through, ghost/puppet aesthetic."
        )
        motion_layout.addWidget(self._spring_preset)

        # Contact-aware foot pinning (v2). Independent of the spring
        # preset combo — preset only affects the rotation filter.
        _pin_sep = QFrame()
        _pin_sep.setFrameShape(QFrame.HLine)
        _pin_sep.setFrameShadow(QFrame.Sunken)
        motion_layout.addWidget(_pin_sep)
        self._use_foot_pin = QCheckBox("Pin feet during contact")
        self._use_foot_pin.setChecked(False)
        self._use_foot_pin.setToolTip(
            "Detect foot contact frames and snap the foot to a fixed world\n"
            "position during each stance episode. Fixes foot sliding under\n"
            "camera moves. Applied after the spring filter. Heuristic contact\n"
            "detection — works independently of the spring filter, so can\n"
            "be enabled alone."
        )
        motion_layout.addWidget(self._use_foot_pin)

        # Contact sensitivity
        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Contact sensitivity:"))
        self._foot_pin_sensitivity = QComboBox()
        for label, _key in _SENSITIVITY_OPTIONS:
            self._foot_pin_sensitivity.addItem(label)
        self._foot_pin_sensitivity.setCurrentIndex(1)
        self._foot_pin_sensitivity.setToolTip(
            "Controls how aggressively the heuristic marks a frame as in-contact.\n"
            "Low: tight 4 cm / 0.3 m/s window — use when swing phases get wrongly pinned.\n"
            "Medium: 7 cm / 0.5 m/s — good starting point for most walking clips.\n"
            "High: 12 cm / 1.0 m/s — use when the pin misses real contacts on fast footwork."
        )
        sens_row.addWidget(self._foot_pin_sensitivity)
        motion_layout.addLayout(sens_row)

        # Pin strength
        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("Pin strength:"))
        self._foot_pin_strength = QDoubleSpinBox()
        self._foot_pin_strength.setRange(0.0, 1.0)
        self._foot_pin_strength.setSingleStep(0.1)
        self._foot_pin_strength.setDecimals(2)
        self._foot_pin_strength.setValue(1.0)
        self._foot_pin_strength.setToolTip(
            "Scales the pin correction. 1.0 = full pinning, 0.5 = half, 0.0 = no correction.\n"
            "Lower values help when pinning over-corrects mild drift."
        )
        strength_row.addWidget(self._foot_pin_strength)
        motion_layout.addLayout(strength_row)

        self._insert_before_stretch(motion_section)

        # Pipeline output settings section — collapsed by default.
        output_section = CollapsibleSection("Pipeline Output", collapsed=True)
        output_layout = output_section.content_layout

        # Target FPS
        fps_row = QHBoxLayout()
        fps_row.addWidget(QLabel("Target FPS:"))
        self._target_fps = QDoubleSpinBox()
        self._target_fps.setRange(1.0, 120.0)
        self._target_fps.setValue(30.0)
        self._target_fps.setSingleStep(1.0)
        self._target_fps.setDecimals(1)
        self._target_fps.setToolTip("Output frame rate (video will be resampled)")
        fps_row.addWidget(self._target_fps)
        output_layout.addLayout(fps_row)

        # FBX Bone Naming
        naming_row = QHBoxLayout()
        naming_row.addWidget(QLabel("FBX naming:"))
        self._fbx_naming = QComboBox()
        self._fbx_naming.addItems(["Mixamo (Cascadeur)", "UE5 Mannequin"])
        self._fbx_naming.setToolTip("Bone naming convention for FBX export")
        naming_row.addWidget(self._fbx_naming)
        output_layout.addLayout(naming_row)

        # Pitch Adjust
        pitch_row = QHBoxLayout()
        pitch_row.addWidget(QLabel("Pitch adjust:"))
        self._pitch_adjust = QDoubleSpinBox()
        self._pitch_adjust.setRange(-30.0, 30.0)
        self._pitch_adjust.setValue(0.0)
        self._pitch_adjust.setSingleStep(0.5)
        self._pitch_adjust.setDecimals(1)
        self._pitch_adjust.setSuffix("°")
        self._pitch_adjust.setToolTip(
            "Manual pitch correction on top of auto-tilt removal (-30° to +30°)"
        )
        pitch_row.addWidget(self._pitch_adjust)
        output_layout.addLayout(pitch_row)

        # Body Smoothing
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Body smoothing:"))
        self._body_smooth = QComboBox()
        self._body_smooth.addItems(["Light", "Moderate (default)", "Heavy"])
        self._body_smooth.setCurrentIndex(1)  # Moderate
        self._body_smooth.setToolTip(
            "Temporal smoothing strength for body rotations"
        )
        smooth_row.addWidget(self._body_smooth)
        output_layout.addLayout(smooth_row)

        # Camera Smoothing (fallback when SLAM unavailable)
        cam_smooth_row = QHBoxLayout()
        cam_smooth_row.addWidget(QLabel("Camera smoothing:"))
        self._cam_smooth = QComboBox()
        self._cam_smooth.addItems(["Light", "Moderate (default)", "Heavy"])
        self._cam_smooth.setCurrentIndex(1)  # Moderate
        self._cam_smooth.setToolTip(
            "Temporal smoothing for incam camera (fallback when SLAM unavailable)"
        )
        cam_smooth_row.addWidget(self._cam_smooth)
        output_layout.addLayout(cam_smooth_row)

        self._insert_before_stretch(output_section)

        # Update run button text
        self._run_btn.setText("Run Pipeline")

    def _on_backend_changed(self, _index: int):
        """Update run button label when backend changes."""
        label, _be, _bm = _BACKEND_OPTIONS[self._backend_combo.currentIndex()]
        backend_name = label.split(" ")[0]
        self._run_btn.setText(f"Run Pipeline ({backend_name})")

    # ------------------------------------------------------------------
    # Output directory mapping (override for perfcap subdirectory)
    # ------------------------------------------------------------------

    def _output_dir_for_video(self, video_path: Path) -> Path:
        return self._gvhmr_root / "outputs" / "perfcap" / video_path.stem

    # ------------------------------------------------------------------
    # SMPLest-X discovery
    # ------------------------------------------------------------------

    def _discover_smplestx(self) -> tuple[str | None, str | None]:
        """Discover SMPLest-X Python binary and working directory.

        Returns (python_path, working_dir) or (None, None).
        """
        smplestx_script = self._gvhmr_root / "smplestx_inference.py"
        if not smplestx_script.is_file():
            self.log_message.emit(
                "SMPLest-X: smplestx_inference.py not found — hands disabled", "warning")
            return None, None

        smplestx_dir = str(self._gvhmr_root)

        candidates = [
            Path.home() / "miniconda3" / "envs" / "smplestx" / "bin" / "python",
            Path.home() / "anaconda3" / "envs" / "smplestx" / "bin" / "python",
            Path.home() / ".conda" / "envs" / "smplestx" / "bin" / "python",
        ]
        for c in candidates:
            if c.is_file():
                self.log_message.emit(f"SMPLest-X env: {c}", "info")
                return str(c), smplestx_dir

        self.log_message.emit(
            "SMPLest-X: script found but no smplestx conda env — hands disabled",
            "warning")
        return None, None

    # ------------------------------------------------------------------
    # Pipeline execution (override to use FullPipelineWorker)
    # ------------------------------------------------------------------

    def _on_run(self):
        if not self._video_path:
            return

        config = self.get_config()
        be, _bm = self._selected_backend()
        output_dir = self._gvhmr_root / "outputs" / "perfcap" / self._video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config to output directory for session restore
        config.save(output_dir / "solve_config.json")

        if be == "gemx":
            gemx_root = self._gvhmr_root.parent / "GEM-X"
            self._worker = GEMXWorker(
                video_path=self._video_path,
                config=config,
                gemx_root=gemx_root,
                output_dir=output_dir,
                fps=config.target_fps,
            )
            backend_label = "GEM-X"
        else:
            smplestx_python, smplestx_dir = self._discover_smplestx()
            self._worker = FullPipelineWorker(
                video_path=self._video_path,
                config=config,
                gvhmr_root=self._gvhmr_root,
                output_dir=output_dir,
                fps=config.target_fps,
                smplestx_python=smplestx_python,
                smplestx_dir=smplestx_dir,
            )
            backend_label = "GVHMR"

        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self._set_running(True)
        self.status_message.emit(f"Running {backend_label} performance capture pipeline...")
        self.log_message.emit(f"Starting {backend_label} performance capture pipeline...", "info")
        self.log_message.emit(
            f"  Backend: {be} | Body model: {config.body_model} | Worker: {type(self._worker).__name__}",
            "info",
        )

    # ------------------------------------------------------------------
    # Multi-stage progress display
    # ------------------------------------------------------------------

    def _on_progress(self, fraction: float, stage: str):
        """Show stage number/name (e.g. 'Stage 2/4: Hands') per spec."""
        self._progress_bar.setValue(int(fraction * 1000))

        use_hands = self._use_hands.isChecked()
        use_face = self._use_face.isChecked()
        use_motion = self._use_spring.isChecked() or self._use_foot_pin.isChecked()

        visible = map_stage_label(stage, use_hands, use_face, use_motion)
        if visible is not None:
            self._current_stage = visible

        stages = compute_visible_stages(use_hands, use_face, use_motion)
        current = getattr(self, "_current_stage", stages[0])
        idx = stages.index(current) + 1 if current in stages else len(stages)
        total = len(stages)

        stage_text = f"Stage {idx}/{total}: {current}"
        self._progress_label.setText(stage_text)
        self.status_message.emit(f"{stage_text} ({fraction:.0%})")

    # ------------------------------------------------------------------
    # Running state (extend to disable new controls)
    # ------------------------------------------------------------------

    def _set_running(self, running: bool):
        super()._set_running(running)
        if running:
            self._current_stage = "Body"
            stages = compute_visible_stages(
                self._use_hands.isChecked(),
                self._use_face.isChecked(),
                self._use_spring.isChecked() or self._use_foot_pin.isChecked(),
            )
            self._progress_label.setText(f"Stage 1/{len(stages)}: Body")
        self._use_hands.setEnabled(not running)
        self._use_face.setEnabled(not running)
        self._use_spring.setEnabled(not running)
        self._spring_preset.setEnabled(not running)
        self._use_camera_stabilize.setEnabled(not running)
        self._use_foot_pin.setEnabled(not running)
        self._foot_pin_sensitivity.setEnabled(not running)
        self._foot_pin_strength.setEnabled(not running)
        self._use_vitpose_face.setEnabled(not running)
        self._hand_hybrid.setEnabled(not running and self._use_hands.isChecked())
        self._hand_smplestx.setEnabled(not running and self._use_hands.isChecked())
        self._hand_src_smplestx.setEnabled(not running and self._use_hands.isChecked())
        self._hand_src_hamer.setEnabled(not running and self._use_hands.isChecked())
        self._target_fps.setEnabled(not running)
        self._fbx_naming.setEnabled(not running)
        self._pitch_adjust.setEnabled(not running)
        self._body_smooth.setEnabled(not running)
        self._cam_smooth.setEnabled(not running)

    # ------------------------------------------------------------------
    # Settings persistence (extend with hand/face + pipeline settings)
    # ------------------------------------------------------------------

    def get_config(self) -> PipelineConfig:
        """Return current settings including hand/face/pipeline options."""
        # Map body smoothing combo text to internal key
        smooth_text = self._body_smooth.currentText()
        if "Light" in smooth_text:
            smooth_key = "light"
        elif "Heavy" in smooth_text:
            smooth_key = "heavy"
        else:
            smooth_key = "moderate"

        # Map camera smoothing combo text to internal key
        cam_smooth_text = self._cam_smooth.currentText()
        if "Light" in cam_smooth_text:
            cam_smooth_key = "light"
        elif "Heavy" in cam_smooth_text:
            cam_smooth_key = "heavy"
        else:
            cam_smooth_key = "moderate"

        # Map spring preset combo text -> internal key
        spring_text = self._spring_preset.currentText()
        if "Light" in spring_text:
            spring_key = "light"
        elif "Heavy" in spring_text:
            spring_key = "heavy"
        else:
            spring_key = "moderate"

        be, bm = self._selected_backend()
        return PipelineConfig(
            mode="perf",
            static_cam=self._static_cam.isChecked(),
            use_dpvo=self._use_dpvo.isChecked(),
            focal_mm=self._focal_mm.value(),
            use_hands=self._use_hands.isChecked(),
            use_face=self._use_face.isChecked(),
            hand_mode="hybrid" if self._hand_hybrid.isChecked() else "smplestx_only",
            target_fps=self._target_fps.value(),
            fbx_naming=self._fbx_naming.currentText(),
            pitch_adjust=self._pitch_adjust.value(),
            hand_source="hamer" if self._hand_src_hamer.isChecked() else "smplestx",
            body_smooth_preset=smooth_key,
            cam_smooth_preset=cam_smooth_key,
            use_vitpose_face_crops=self._use_vitpose_face.isChecked(),
            estimation_backend=be,
            body_model=bm,
            use_camera_stabilize=self._use_camera_stabilize.isChecked(),
            use_spring_refine=self._use_spring.isChecked(),
            spring_refine_preset=spring_key,
            use_foot_pin=self._use_foot_pin.isChecked(),
            foot_pin_sensitivity=_sensitivity_key_for_index(
                self._foot_pin_sensitivity.currentIndex()
            ),
            foot_pin_strength=float(self._foot_pin_strength.value()),
        )

    def set_config(self, config: PipelineConfig):
        """Apply settings including hand/face/pipeline options."""
        super().set_config(config)
        self._use_hands.setChecked(config.use_hands)
        self._use_face.setChecked(config.use_face)
        self._use_camera_stabilize.setChecked(config.use_camera_stabilize)
        self._use_spring.setChecked(config.use_spring_refine)
        self._use_foot_pin.setChecked(config.use_foot_pin)
        self._foot_pin_sensitivity.setCurrentIndex(
            _sensitivity_index_for_key(config.foot_pin_sensitivity)
        )
        self._foot_pin_strength.setValue(float(config.foot_pin_strength))
        spring_map = {"light": 0, "moderate": 1, "heavy": 2}
        self._spring_preset.setCurrentIndex(
            spring_map.get(config.spring_refine_preset, 1)
        )
        if config.hand_mode == "smplestx_only":
            self._hand_smplestx.setChecked(True)
        else:
            self._hand_hybrid.setChecked(True)

        # Hand source
        if config.hand_source == "hamer":
            self._hand_src_hamer.setChecked(True)
        else:
            self._hand_src_smplestx.setChecked(True)

        # Pipeline settings
        self._target_fps.setValue(config.target_fps)
        idx = self._fbx_naming.findText(config.fbx_naming)
        if idx >= 0:
            self._fbx_naming.setCurrentIndex(idx)
        self._pitch_adjust.setValue(config.pitch_adjust)
        self._use_vitpose_face.setChecked(config.use_vitpose_face_crops)

        # Body smoothing
        smooth_map = {"light": 0, "moderate": 1, "heavy": 2}
        self._body_smooth.setCurrentIndex(smooth_map.get(config.body_smooth_preset, 1))

        # Camera smoothing
        self._cam_smooth.setCurrentIndex(smooth_map.get(config.cam_smooth_preset, 1))


# =========================================================================
# Multi-person pipeline settings
# =========================================================================


class MultiPipelineSettings(QWidget):
    """Pipeline settings for multi-person capture.

    Contains video input, pipeline settings, multi-person settings,
    run/cancel/progress, and worker lifecycle.
    """

    status_message = Signal(str)
    log_message = Signal(str, str)  # (text, level)
    pipeline_finished = Signal(dict)
    pipeline_error = Signal(str)
    video_loaded = Signal(object)  # Path

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
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.NoFrame)
        self._scroll_area.setMinimumHeight(200)
        scroll_inner = QWidget()
        layout = QVBoxLayout(scroll_inner)
        layout.setContentsMargins(8, 8, 8, 8)
        self._scroll_area.setWidget(scroll_inner)
        outer.addWidget(self._scroll_area, 1)

        # Video input — not collapsible.
        input_container = QWidget()
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 4)
        input_title = QLabel("Video Input")
        input_title.setStyleSheet("font-weight: bold; padding: 4px 2px;")
        input_layout.addWidget(input_title)

        self._drop_area = _DropArea()
        input_layout.addWidget(self._drop_area)

        self._browse_btn = QPushButton("Browse...")
        input_layout.addWidget(self._browse_btn)

        self._video_info = QLabel("")
        self._video_info.setWordWrap(True)
        self._video_info.hide()
        input_layout.addWidget(self._video_info)

        layout.addWidget(input_container)

        # Pipeline settings — collapsible, expanded.
        settings_section = CollapsibleSection("Pipeline Settings", collapsed=False)
        settings_layout = settings_section.content_layout

        backend_row = QHBoxLayout()
        backend_row.addWidget(QLabel("Backend:"))
        self._backend_combo = QComboBox()
        for label, _be, _bm in _BACKEND_OPTIONS:
            self._backend_combo.addItem(label)
        self._backend_combo.setToolTip(
            "GVHMR: multi-stage SMPL-X body capture\n"
            "GEM-X: single-pass SOMA-77 capture (body + hands + face)"
        )
        backend_row.addWidget(self._backend_combo)
        settings_layout.addLayout(backend_row)

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

        layout.addWidget(settings_section)

        # Multi-person specific settings — collapsible, expanded.
        mp_section = CollapsibleSection("Multi-Person", collapsed=False)
        mp_layout = mp_section.content_layout

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

        self._use_hands = QCheckBox("Enable hand capture")
        self._use_hands.setChecked(True)
        self._use_hands.setToolTip(
            "Estimate hand poses per person, then merge with GVHMR body.\n"
            "Only applies to GVHMR backend — GEM-X already includes hands."
        )
        mp_layout.addWidget(self._use_hands)

        # Hand source (SMPLest-X vs HaMeR)
        self._hand_source_group = QButtonGroup(self)
        self._hand_src_smplestx = QRadioButton("SMPLest-X (default)")
        self._hand_src_smplestx.setChecked(True)
        self._hand_src_smplestx.setToolTip("Use SMPLest-X for hand reconstruction")
        self._hand_src_hamer = QRadioButton("HaMeR")
        self._hand_src_hamer.setToolTip(
            "Use HaMeR for dedicated hand mesh recovery (better fingers)"
        )
        self._hand_source_group.addButton(self._hand_src_smplestx)
        self._hand_source_group.addButton(self._hand_src_hamer)
        mp_layout.addWidget(QLabel("Hand source:"))
        mp_layout.addWidget(self._hand_src_smplestx)
        mp_layout.addWidget(self._hand_src_hamer)

        self._use_hands.toggled.connect(self._hand_src_smplestx.setEnabled)
        self._use_hands.toggled.connect(self._hand_src_hamer.setEnabled)

        # Motion refinement (replaces disabled Physics Refinement)
        mp_layout.addWidget(QLabel("Motion refinement:"))

        self._use_camera_stabilize = QCheckBox("Stabilize camera drift")
        self._use_camera_stabilize.setChecked(True)
        self._use_camera_stabilize.setToolTip(
            "Re-derive world-space body params using smoothed camera trajectory.\n"
            "Removes drift caused by camera movement (dolly, orbit). Uses the\n"
            "Camera Smoothing preset to control smoothing strength."
        )
        mp_layout.addWidget(self._use_camera_stabilize)

        _mp_stab_sep = QFrame()
        _mp_stab_sep.setFrameShape(QFrame.HLine)
        _mp_stab_sep.setFrameShadow(QFrame.Sunken)
        mp_layout.addWidget(_mp_stab_sep)

        self._use_spring = QCheckBox("Enable spring-based refinement")
        self._use_spring.setChecked(False)
        self._use_spring.setToolTip(
            "Per-joint critically-damped spring filter applied to body motion in\n"
            "quaternion log space. Adds weight and follow-through (Lieberman-style)\n"
            "without rigid-body simulation. CPU-only, deterministic, fast."
        )
        mp_layout.addWidget(self._use_spring)

        mp_layout.addWidget(QLabel("Spring preset:"))
        self._spring_preset = QComboBox()
        self._spring_preset.addItems(
            [
                "Light (snappier)",
                "Moderate (balanced)",
                "Heavy (loose follow-through)",
            ]
        )
        self._spring_preset.setCurrentIndex(1)
        self._spring_preset.setToolTip(
            "Light: higher stiffness, less filtering, snappier response.\n"
            "Moderate: baseline gains from the per-joint proximal->distal table.\n"
            "Heavy: softer stiffness, more follow-through, ghost/puppet aesthetic."
        )
        mp_layout.addWidget(self._spring_preset)

        _mp_pin_sep = QFrame()
        _mp_pin_sep.setFrameShape(QFrame.HLine)
        _mp_pin_sep.setFrameShadow(QFrame.Sunken)
        mp_layout.addWidget(_mp_pin_sep)
        self._use_foot_pin = QCheckBox("Pin feet during contact")
        self._use_foot_pin.setChecked(False)
        self._use_foot_pin.setToolTip(
            "Detect foot contact frames and snap the foot to a fixed world\n"
            "position during each stance episode. Fixes foot sliding under\n"
            "camera moves. Applied after the spring filter. Heuristic contact\n"
            "detection — works independently of the spring filter, so can\n"
            "be enabled alone."
        )
        mp_layout.addWidget(self._use_foot_pin)

        # Contact sensitivity
        mp_sens_row = QHBoxLayout()
        mp_sens_row.addWidget(QLabel("Contact sensitivity:"))
        self._foot_pin_sensitivity = QComboBox()
        for label, _key in _SENSITIVITY_OPTIONS:
            self._foot_pin_sensitivity.addItem(label)
        self._foot_pin_sensitivity.setCurrentIndex(1)
        self._foot_pin_sensitivity.setToolTip(
            "Controls how aggressively the heuristic marks a frame as in-contact.\n"
            "Low: tight 4 cm / 0.3 m/s window — use when swing phases get wrongly pinned.\n"
            "Medium: 7 cm / 0.5 m/s — good starting point for most walking clips.\n"
            "High: 12 cm / 1.0 m/s — use when the pin misses real contacts on fast footwork."
        )
        mp_sens_row.addWidget(self._foot_pin_sensitivity)
        mp_layout.addLayout(mp_sens_row)

        # Pin strength
        mp_strength_row = QHBoxLayout()
        mp_strength_row.addWidget(QLabel("Pin strength:"))
        self._foot_pin_strength = QDoubleSpinBox()
        self._foot_pin_strength.setRange(0.0, 1.0)
        self._foot_pin_strength.setSingleStep(0.1)
        self._foot_pin_strength.setDecimals(2)
        self._foot_pin_strength.setValue(1.0)
        self._foot_pin_strength.setToolTip(
            "Scales the pin correction. 1.0 = full pinning, 0.5 = half, 0.0 = no correction.\n"
            "Lower values help when pinning over-corrects mild drift."
        )
        mp_strength_row.addWidget(self._foot_pin_strength)
        mp_layout.addLayout(mp_strength_row)

        layout.addWidget(mp_section)
        layout.addStretch()

        # ------------------------------------------------------------------
        # Sticky footer — run/cancel/progress are never scrolled out of view.
        # ------------------------------------------------------------------
        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(8, 4, 8, 8)
        footer_layout.setSpacing(4)

        self._run_btn = QPushButton("Run Multi-Person Pipeline")
        self._run_btn.setEnabled(False)
        self._run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; }")
        footer_layout.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.hide()
        footer_layout.addWidget(self._cancel_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.hide()
        footer_layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.hide()
        footer_layout.addWidget(self._progress_label)

        outer.addWidget(footer, 0)

    def _selected_backend(self) -> tuple[str, str]:
        """Return (estimation_backend, body_model) from combo selection."""
        _, be, bm = _BACKEND_OPTIONS[self._backend_combo.currentIndex()]
        return be, bm

    def _connect_signals(self):
        self._drop_area.file_dropped.connect(self._load_video)
        self._browse_btn.clicked.connect(self._on_browse)
        self._run_btn.clicked.connect(self._on_run)
        self._cancel_btn.clicked.connect(self._on_cancel)
        self._static_cam.toggled.connect(self._on_static_cam_toggled)
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)

    def _on_backend_changed(self, _index: int):
        """Update run button label when backend changes."""
        label, _be, _bm = _BACKEND_OPTIONS[self._backend_combo.currentIndex()]
        backend_name = label.split(" ")[0]
        self._run_btn.setText(f"Run Multi-Person ({backend_name})")

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

        self._run_btn.setEnabled(True)
        self.status_message.emit(f"Loaded: {video_path.name}")
        self.log_message.emit(
            f"Loaded video: {video_path} ({width}x{height}, {num_frames} frames)",
            "info",
        )

        # Restore settings from previous run if available
        self._try_restore_config(video_path)

        # Notify listeners (e.g., tab loads video into player)
        self.video_loaded.emit(video_path)

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

        be, _bm = self._selected_backend()
        if be == "gemx":
            backend_label = "GEM-X"
            self.log_message.emit(
                "NOTE: GEM-X multi-person uses GVHMR tracking + GEM-X per-person estimation",
                "info",
            )
        else:
            backend_label = "GVHMR"

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
        self.status_message.emit(f"Running multi-person pipeline ({backend_label})...")
        self.log_message.emit(f"Starting multi-person pipeline ({backend_label})...", "info")
        self.log_message.emit(
            f"  Backend: {be} | Body model: {config.body_model} | Worker: {type(self._worker).__name__}",
            "info",
        )

    def _on_cancel(self):
        if self._worker:
            self._worker.cancel()
            self._worker.wait()
            self._worker = None
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
        self._backend_combo.setEnabled(not running)
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
        self._use_hands.setEnabled(not running)
        self._use_camera_stabilize.setEnabled(not running)
        self._use_spring.setEnabled(not running)
        self._spring_preset.setEnabled(not running)
        self._use_foot_pin.setEnabled(not running)
        self._foot_pin_sensitivity.setEnabled(not running)
        self._foot_pin_strength.setEnabled(not running)
        self._hand_src_smplestx.setEnabled(not running and self._use_hands.isChecked())
        self._hand_src_hamer.setEnabled(not running and self._use_hands.isChecked())
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
        if self._worker is not None:
            self._worker.wait()
            self._worker = None
        self.status_message.emit("Multi-person pipeline complete")
        self.log_message.emit("Multi-person pipeline finished successfully", "info")
        self.pipeline_finished.emit(result)

    def _on_error(self, message: str):
        self._set_running(False)
        if self._worker is not None:
            self._worker.wait()
            self._worker = None
        self.status_message.emit(f"Error: {message}")
        self.log_message.emit(f"Pipeline error: {message}", "error")
        self.pipeline_error.emit(message)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_config(self) -> PipelineConfig:
        spring_text = self._spring_preset.currentText()
        if "Light" in spring_text:
            spring_key = "light"
        elif "Heavy" in spring_text:
            spring_key = "heavy"
        else:
            spring_key = "moderate"

        be, bm = self._selected_backend()
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
            estimation_backend=be,
            body_model=bm,
            use_hands=self._use_hands.isChecked(),
            hand_source="hamer" if self._hand_src_hamer.isChecked() else "smplestx",
            use_camera_stabilize=self._use_camera_stabilize.isChecked(),
            use_spring_refine=self._use_spring.isChecked(),
            spring_refine_preset=spring_key,
            use_foot_pin=self._use_foot_pin.isChecked(),
            foot_pin_sensitivity=_sensitivity_key_for_index(
                self._foot_pin_sensitivity.currentIndex()
            ),
            foot_pin_strength=float(self._foot_pin_strength.value()),
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
        self._use_hands.setChecked(config.use_hands)
        self._use_camera_stabilize.setChecked(config.use_camera_stabilize)
        self._use_spring.setChecked(config.use_spring_refine)
        self._use_foot_pin.setChecked(config.use_foot_pin)
        self._foot_pin_sensitivity.setCurrentIndex(
            _sensitivity_index_for_key(config.foot_pin_sensitivity)
        )
        self._foot_pin_strength.setValue(float(config.foot_pin_strength))
        spring_map = {"light": 0, "moderate": 1, "heavy": 2}
        self._spring_preset.setCurrentIndex(
            spring_map.get(config.spring_refine_preset, 1)
        )
        if config.hand_source == "hamer":
            self._hand_src_hamer.setChecked(True)
        else:
            self._hand_src_smplestx.setChecked(True)
        for i, (_, be, _bm) in enumerate(_BACKEND_OPTIONS):
            if be == config.estimation_backend:
                self._backend_combo.setCurrentIndex(i)
                break
