"""Physics refinement worker — QThread wrapper for the PHC pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


class PhysicsRefineWorker(QThread):
    """Run PHC physics refinement on SMPL params.

    Stages:
        0.0  — Convert params to AMASS NPZ
        0.1  — Run PHC
        0.8  — Convert back to SMPL params
        0.9  — Evaluate quality metrics
        1.0  — Done
    """

    progress = Signal(float, str)
    log_line = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        params: dict,
        output_dir: Path,
        fps: float = 30.0,
        mode: str = "local",
        phc_root: str | Path | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._params = params
        self._output_dir = Path(output_dir)
        self._fps = fps
        self._mode = mode  # "local" or "ssh"
        self._phc_root = phc_root or "~/mocap-filter/PHC"
        self._cancelled = False

    def run(self):
        import numpy as np

        try:
            from workers.physics.gvhmr_to_amass import params_to_amass_npz, amass_to_params
            from workers.physics.phc_runner import run_phc_local, run_phc_ssh
            from workers.physics.phc_to_smpl import phc_output_to_params
            from workers.physics.evaluate import evaluate_refinement

            self._output_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: Convert to AMASS
            self.progress.emit(0.0, "Converting to AMASS format...")
            self.log_line.emit("Physics: converting params to AMASS NPZ...")
            input_npz = self._output_dir / "phc_input.npz"
            params_to_amass_npz(self._params, input_npz, fps=int(self._fps))
            self.log_line.emit(f"Physics: saved AMASS input -> {input_npz}")

            if self._cancelled:
                return

            # Stage 2: Run PHC
            self.progress.emit(0.1, "Running PHC physics simulation...")
            self.log_line.emit(f"Physics: running PHC ({self._mode} mode)...")

            phc_output_dir = self._output_dir / "phc_output"

            def phc_progress(frac, msg):
                if frac >= 0:
                    overall = 0.1 + frac * 0.7  # Map to [0.1, 0.8]
                    self.progress.emit(overall, msg)
                if msg:
                    self.log_line.emit(f"PHC: {msg}")

            if self._mode == "ssh":
                result = run_phc_ssh(
                    input_npz,
                    phc_output_dir,
                    progress_cb=phc_progress,
                )
            else:
                result = run_phc_local(
                    input_npz,
                    phc_output_dir,
                    phc_root=self._phc_root,
                    progress_cb=phc_progress,
                )

            if self._cancelled:
                return

            # Graceful degradation: if PHC fails, return original params
            if not result.success:
                self.log_line.emit(
                    f"WARNING: PHC failed, returning original params. "
                    f"Log:\n{result.log[-500:]}"
                )
                self.progress.emit(1.0, "Physics refinement skipped (PHC unavailable)")
                self.finished.emit(self._params)
                return

            self.log_line.emit(f"Physics: PHC output -> {result.output_path}")

            # Stage 3: Convert back
            self.progress.emit(0.8, "Converting PHC output to SMPL params...")
            self.log_line.emit("Physics: converting PHC output to SMPL params...")
            refined_params = phc_output_to_params(result.output_path, self._params)
            self.log_line.emit(
                f"Physics: refined params — {refined_params['num_frames']} frames"
            )

            if self._cancelled:
                return

            # Stage 4: Evaluate
            self.progress.emit(0.9, "Evaluating refinement quality...")
            self.log_line.emit("Physics: computing quality metrics...")
            try:
                metrics = evaluate_refinement(
                    self._params, refined_params, fps=self._fps
                )
                for key, val in metrics.items():
                    if isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            self.log_line.emit(f"  {key}.{sub_key}: {sub_val:.4f}")
                    elif isinstance(val, float):
                        self.log_line.emit(f"  {key}: {val:.4f}")
                    else:
                        self.log_line.emit(f"  {key}: {val}")
                refined_params["_physics_metrics"] = metrics
            except Exception as exc:
                self.log_line.emit(f"WARNING: Metric evaluation failed: {exc}")

            # Done
            self.progress.emit(1.0, "Physics refinement complete")
            self.finished.emit(refined_params)

        except Exception as exc:
            import traceback

            self.error.emit(
                f"Physics refinement failed: {exc}\n{traceback.format_exc()}"
            )

    def cancel(self):
        self._cancelled = True
