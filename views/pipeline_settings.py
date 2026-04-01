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
    QGroupBox,
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
    QSizePolicy,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent

from models.pipeline_config import PipelineConfig
from models.session import Session
from workers.gvhmr_worker import GVHMRWorker
from workers.gemx_worker import GEMXWorker
from workers.pipeline_orchestrator import FullPipelineWorker, MultiPersonWorker

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
        self._left_layout = left_layout = QVBoxLayout(self)
        left_layout.setContentsMargins(8, 8, 8, 8)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

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

        # Pipeline Settings
        settings_group = QGroupBox("Pipeline Settings")
        settings_layout = QVBoxLayout(settings_group)

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
    "Face pipeline": "Face",
    "BVH/FBX conversion": "Export",
    "Rendering": "Export",
}


def compute_visible_stages(use_hands: bool, use_face: bool) -> list[str]:
    """Return ordered list of user-visible pipeline stage names.

    Stages are dynamic — Hands and Face only appear when enabled.
    Body and Export are always present.
    """
    stages = ["Body"]
    if use_hands:
        stages.append("Hands")
    if use_face:
        stages.append("Face")
    stages.append("Export")
    return stages


def map_stage_label(
    internal_label: str, use_hands: bool, use_face: bool
) -> str | None:
    """Map internal worker stage label to user-visible stage name.

    Returns ``None`` when the label doesn't correspond to a stage
    transition (sub-progress message or disabled-stage emission that
    should be swallowed).
    """
    user_stage = _INTERNAL_TO_USER_STAGE.get(internal_label)
    if user_stage is None:
        return None
    # When hands disabled, merge is instant GVHMR-only extraction → Export
    if user_stage == "Hands" and not use_hands:
        return "Export"
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
        """Insert hand/face/pipeline settings between body settings and run button."""
        # Find the run button index in the left layout
        run_idx = self._left_layout.indexOf(self._run_btn)

        # Hand capture group
        hand_group = QGroupBox("Hand Capture")
        hand_layout = QVBoxLayout(hand_group)

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

        self._left_layout.insertWidget(run_idx, hand_group)

        # Face capture group
        face_group = QGroupBox("Face Capture")
        face_layout = QVBoxLayout(face_group)

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

        self._left_layout.insertWidget(run_idx + 1, face_group)

        # Pipeline output settings group
        output_group = QGroupBox("Pipeline Settings")
        output_layout = QVBoxLayout(output_group)

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

        self._left_layout.insertWidget(run_idx + 2, output_group)

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

        visible = map_stage_label(stage, use_hands, use_face)
        if visible is not None:
            self._current_stage = visible

        stages = compute_visible_stages(use_hands, use_face)
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
                self._use_hands.isChecked(), self._use_face.isChecked()
            )
            self._progress_label.setText(f"Stage 1/{len(stages)}: Body")
        self._use_hands.setEnabled(not running)
        self._use_face.setEnabled(not running)
        self._use_vitpose_face.setEnabled(not running)
        self._hand_hybrid.setEnabled(not running and self._use_hands.isChecked())
        self._hand_smplestx.setEnabled(not running and self._use_hands.isChecked())
        self._hand_src_smplestx.setEnabled(not running and self._use_hands.isChecked())
        self._hand_src_hamer.setEnabled(not running and self._use_hands.isChecked())
        self._target_fps.setEnabled(not running)
        self._fbx_naming.setEnabled(not running)
        self._pitch_adjust.setEnabled(not running)
        self._body_smooth.setEnabled(not running)

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
            use_vitpose_face_crops=self._use_vitpose_face.isChecked(),
            estimation_backend=be,
            body_model=bm,
        )

    def set_config(self, config: PipelineConfig):
        """Apply settings including hand/face/pipeline options."""
        super().set_config(config)
        self._use_hands.setChecked(config.use_hands)
        self._use_face.setChecked(config.use_face)
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

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

        layout.addWidget(input_group)

        # Pipeline settings
        settings_group = QGroupBox("Pipeline Settings")
        settings_layout = QVBoxLayout(settings_group)

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

        layout.addWidget(settings_group)

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

        layout.addWidget(mp_group)

        # Run / Cancel / Progress
        self._run_btn = QPushButton("Run Multi-Person Pipeline")
        self._run_btn.setEnabled(False)
        self._run_btn.setStyleSheet("QPushButton { font-weight: bold; padding: 10px; }")
        layout.addWidget(self._run_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.hide()
        layout.addWidget(self._cancel_btn)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 1000)
        self._progress_bar.setValue(0)
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        self._progress_label = QLabel("")
        self._progress_label.hide()
        layout.addWidget(self._progress_label)

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
        if config.hand_source == "hamer":
            self._hand_src_hamer.setChecked(True)
        else:
            self._hand_src_smplestx.setChecked(True)
        for i, (_, be, _bm) in enumerate(_BACKEND_OPTIONS):
            if be == config.estimation_backend:
                self._backend_combo.setCurrentIndex(i)
                break
