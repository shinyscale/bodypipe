"""Performance capture tab — body + hands + face pipeline.

Extends SinglePersonTab with hand capture mode selection and face capture
toggle. Uses FullPipelineWorker for multi-stage orchestration.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
)

from models.pipeline_config import PipelineConfig
from models.session import Session
from views.single_person_tab import SinglePersonTab
from workers.pipeline_orchestrator import FullPipelineWorker


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
        """Insert hand and face capture settings between body settings and run button."""
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

        # Enable/disable radio buttons based on hand checkbox
        self._use_hands.toggled.connect(self._hand_hybrid.setEnabled)
        self._use_hands.toggled.connect(self._hand_smplestx.setEnabled)

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

        self._left_layout.insertWidget(run_idx + 1, face_group)

        # Update run button text
        self._run_btn.setText("Run Pipeline")

    # ------------------------------------------------------------------
    # Pipeline execution (override to use FullPipelineWorker)
    # ------------------------------------------------------------------

    def _on_run(self):
        if not self._video_path:
            return

        config = self.get_config()
        output_dir = self._gvhmr_root / "outputs" / "perfcap" / self._video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        self._worker = FullPipelineWorker(
            video_path=self._video_path,
            config=config,
            gvhmr_root=self._gvhmr_root,
            output_dir=output_dir,
            fps=self._session.fps,
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
    # Settings persistence (extend with hand/face)
    # ------------------------------------------------------------------

    def get_config(self) -> PipelineConfig:
        """Return current settings including hand/face options."""
        return PipelineConfig(
            mode="perf",
            static_cam=self._static_cam.isChecked(),
            use_dpvo=self._use_dpvo.isChecked(),
            focal_mm=self._focal_mm.value(),
            use_hands=self._use_hands.isChecked(),
            use_face=self._use_face.isChecked(),
            hand_mode="hybrid" if self._hand_hybrid.isChecked() else "smplestx_only",
        )

    def set_config(self, config: PipelineConfig):
        """Apply settings including hand/face options."""
        super().set_config(config)
        self._use_hands.setChecked(config.use_hands)
        self._use_face.setChecked(config.use_face)
        if config.hand_mode == "smplestx_only":
            self._hand_smplestx.setChecked(True)
        else:
            self._hand_hybrid.setChecked(True)
