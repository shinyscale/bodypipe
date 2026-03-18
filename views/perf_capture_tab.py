"""Performance capture tab — body + hands + face pipeline.

Extends SinglePersonTab with hand capture mode selection, face capture
toggle, and pipeline output settings (FPS, FBX naming, pitch adjust,
hand source, body smoothing, ViTPose face crops). Uses FullPipelineWorker
for multi-stage orchestration.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
)

from models.pipeline_config import PipelineConfig
from models.session import Session
from views.single_person_tab import SinglePersonTab
from workers.pipeline_orchestrator import FullPipelineWorker

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


class PerfCaptureTab(SinglePersonTab):
    """Second tab — full body + hands + face performance capture."""

    def __init__(self, session: Session, gvhmr_root: Path, parent=None):
        # _hand_face_added guard prevents double-init from super().__init__
        self._hand_face_added = False
        super().__init__(session, gvhmr_root, parent)

    def _setup_ui(self):
        super()._setup_ui()
        self._add_hand_face_settings()

    def _add_hand_face_settings(self):
        """Insert hand/face/pipeline settings between body settings and run button."""
        if self._hand_face_added:
            return
        self._hand_face_added = True

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

    # ------------------------------------------------------------------
    # Output directory mapping (override for perfcap subdirectory)
    # ------------------------------------------------------------------

    def _output_dir_for_video(self, video_path: Path) -> Path:
        return self._gvhmr_root / "outputs" / "perfcap" / video_path.stem

    # ------------------------------------------------------------------
    # Pipeline execution (override to use FullPipelineWorker)
    # ------------------------------------------------------------------

    def _on_run(self):
        if not self._video_path:
            return

        config = self.get_config()
        output_dir = self._gvhmr_root / "outputs" / "perfcap" / self._video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config to output directory for session restore
        config.save(output_dir / "solve_config.json")

        self._worker = FullPipelineWorker(
            video_path=self._video_path,
            config=config,
            gvhmr_root=self._gvhmr_root,
            output_dir=output_dir,
            fps=config.target_fps,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log_line.connect(self._on_log_line)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        self._set_running(True)
        self.status_message.emit("Running performance capture pipeline...")
        self.log_message.emit("Starting full performance capture pipeline...", "info")

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
