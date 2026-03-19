"""GEM-X estimation worker — single-pass video → SOMA-77 params.

Why: GEM-X replaces the multi-stage GVHMR+SMPLest-X+merge pipeline with a
single model that estimates body, hands, and face simultaneously in SOMA
format. This eliminates the merge step and produces higher-quality results
for hand and face tracking.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal

from models.pipeline_config import PipelineConfig
from workers._base import SubprocessWorkerBase

logger = logging.getLogger(__name__)

# Stage fraction ranges for GEM-X pipeline (simpler than GVHMR — single pass)
_GEMX_STAGES: list[tuple[float, float, str]] = [
    (0.00, 0.05, "Preprocessing"),
    (0.05, 0.80, "GEM-X estimation"),
    (0.80, 0.90, "BVH/FBX conversion"),
    (0.90, 1.00, "Rendering"),
]


def load_gemx_soma_output(output_dir: Path) -> dict | None:
    """Load SOMA params from GEM-X output directory.

    GEM-X saves results as .npz with keys:
    - poses: (N, 77, 3) axis-angle per joint
    - transl: (N, 3)
    - global_orient: (N, 3)
    - identity_coeffs: (1, 45)
    - scale_params: (1, 68)
    - identity_model_type: str

    Returns dict matching PersonTrack.soma_params format, or None.
    """
    # GEM-X outputs to {output_dir}/soma_results.npz
    for pattern in ["soma_results.npz", "*.npz"]:
        for f in sorted(output_dir.glob(pattern)):
            try:
                data = dict(np.load(str(f), allow_pickle=True))
                if "poses" in data and data["poses"].ndim == 3:
                    result = {
                        "poses": data["poses"].astype(np.float32),
                        "transl": data.get("transl", np.zeros((data["poses"].shape[0], 3))).astype(np.float32),
                        "global_orient": data.get("global_orient", data["poses"][:, 0]).astype(np.float32),
                    }
                    if "identity_coeffs" in data:
                        result["identity_coeffs"] = data["identity_coeffs"].astype(np.float32)
                    if "scale_params" in data:
                        result["scale_params"] = data["scale_params"].astype(np.float32)
                    result["identity_model_type"] = str(data.get("identity_model_type", "mhr"))
                    return result
            except Exception:
                continue
    return None


class GEMXWorker(SubprocessWorkerBase):
    """Run GEM-X estimation as a subprocess, output SOMA-77 params."""

    def __init__(
        self,
        video_path: Path,
        config: PipelineConfig,
        gemx_root: Path,
        output_dir: Path,
        fps: float = 30.0,
        parent=None,
    ):
        super().__init__(parent)
        self._video_path = video_path
        self._config = config
        self._gemx_root = gemx_root
        self._output_dir = output_dir
        self._fps = fps

    def run(self):
        try:
            results: dict = {"video_path": str(self._video_path)}
            self._output_dir.mkdir(parents=True, exist_ok=True)

            # Stage 0: Preprocessing
            self._emit_stage(0)
            if self._cancelled:
                return

            # Stage 1: GEM-X estimation
            self._emit_stage(1)
            cmd = self._gemx_command()
            self.log_line.emit(f"$ {' '.join(cmd)}")
            rc, lines = self._run_subprocess(cmd, self._gemx_root)
            if self._cancelled:
                return
            if rc != 0:
                self.error.emit(f"GEM-X estimation failed (exit {rc})")
                return
            results["gemx_log"] = "\n".join(lines)

            # Load SOMA output
            soma_params = load_gemx_soma_output(self._output_dir)
            if soma_params is None:
                self.error.emit("GEM-X produced no SOMA output")
                return
            n_frames = soma_params["poses"].shape[0]
            self.log_line.emit(f"GEM-X output: {n_frames} frames, 77 joints (SOMA)")
            results["soma_params"] = soma_params
            results["n_frames"] = n_frames

            # Stage 2: BVH/FBX conversion
            self._emit_stage(2)
            self._run_bvh_fbx(results, soma_params)
            if self._cancelled:
                return

            # Stage 3: Rendering
            self._emit_stage(3)
            self.log_line.emit("Rendering skipped (SOMA renderer not yet integrated).")

            self.progress.emit(1.0, "GEM-X pipeline complete")
            results["output_dir"] = str(self._output_dir)
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))

    def _run_bvh_fbx(self, results: dict, soma_params: dict) -> None:
        """Convert SOMA params to BVH, then BVH to FBX."""
        stem = self._video_path.stem
        bvh_path = str(self._output_dir / f"{stem}_soma.bvh")

        try:
            from workers.soma_bvh_export import convert_soma_to_bvh

            convert_soma_to_bvh(
                soma_params=soma_params,
                output_path=bvh_path,
                fps=self._fps,
            )
            results["bvh"] = bvh_path
            self.log_line.emit(f"SOMA BVH written: {bvh_path}")
        except Exception as exc:
            self.log_line.emit(f"WARNING: SOMA BVH conversion failed: {exc}")
            return

        # FBX via Blender bridge (same as SMPL-X path)
        fbx_path = str(self._output_dir / f"{stem}_soma.fbx")
        try:
            from bvh_to_fbx import convert_bvh_to_fbx

            naming_key = "ue5" if "ue5" in self._config.fbx_naming.lower() else "mixamo"
            fbx_log = convert_bvh_to_fbx(bvh_path, fbx_path, fps=self._fps, naming=naming_key)
            self.log_line.emit(fbx_log)
            if "ERROR" not in fbx_log:
                results["fbx"] = fbx_path
        except ImportError:
            self.log_line.emit("WARNING: bvh_to_fbx not available, skipping FBX.")
        except Exception as exc:
            self.log_line.emit(f"WARNING: FBX conversion failed: {exc}")

    def _emit_stage(self, idx: int):
        start, _end, label = _GEMX_STAGES[idx]
        self.progress.emit(start, label)

    def _gemx_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "scripts/demo/demo_soma.py",
            f"--video={self._video_path}",
            f"--output_root={self._output_dir}",
        ]
        if self._config.static_cam:
            cmd.append("--static_cam")
        return cmd
