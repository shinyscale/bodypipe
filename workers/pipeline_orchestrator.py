"""Orchestration workers for multi-stage pipelines."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from models.pipeline_config import PipelineConfig
from workers._base import SubprocessWorkerBase


# Stage fraction ranges for FullPipelineWorker — (start, end, label).
# Ranges are contiguous and span [0, 1].
_FULL_STAGES: list[tuple[float, float, str]] = [
    (0.00, 0.02, "Preprocessing"),
    (0.02, 0.35, "GVHMR body solve"),
    (0.35, 0.50, "SMPLest-X hand solve"),
    (0.50, 0.52, "Merging body + hands"),
    (0.52, 0.72, "Face pipeline"),
    (0.72, 0.85, "BVH/FBX conversion"),
    (0.85, 1.00, "Rendering"),
]


class FullPipelineWorker(SubprocessWorkerBase):
    """Orchestrates GVHMR + SMPLest-X + merge + BVH/FBX sequentially."""

    def __init__(
        self,
        video_path: Path,
        config: PipelineConfig,
        gvhmr_root: Path,
        output_dir: Path,
        fps: float = 30.0,
        smplestx_python: str | Path | None = None,
        smplestx_dir: str | Path | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._video_path = video_path
        self._config = config
        self._gvhmr_root = gvhmr_root
        self._output_dir = output_dir
        self._fps = fps
        self._smplestx_python = str(smplestx_python) if smplestx_python else None
        self._smplestx_dir = str(smplestx_dir) if smplestx_dir else None

    def run(self):
        try:
            results: dict = {"video_path": str(self._video_path)}

            # Stage 0: Preprocessing
            self._emit_stage(0)
            if self._cancelled:
                return

            # Stage 1: GVHMR body solve
            self._emit_stage(1)
            gvhmr_cmd = self._gvhmr_command()
            self.log_line.emit(f"$ {' '.join(gvhmr_cmd)}")
            rc, lines = self._run_subprocess(gvhmr_cmd, self._gvhmr_root)
            if self._cancelled:
                return
            if rc != 0:
                self.error.emit(f"GVHMR body solve failed (exit {rc})")
                return
            results["gvhmr_log"] = "\n".join(lines)

            # Stage 2: SMPLest-X hand solve (if enabled and configured)
            if self._config.use_hands and self._smplestx_python and self._smplestx_dir:
                self._emit_stage(2)
                smplx_cmd = self._smplestx_command()
                self.log_line.emit(f"$ {' '.join(smplx_cmd)}")
                rc, lines = self._run_subprocess(smplx_cmd, self._smplestx_dir)
                if self._cancelled:
                    return
                if rc != 0:
                    self.error.emit(f"SMPLest-X hand solve failed (exit {rc})")
                    return
                results["smplestx_log"] = "\n".join(lines)

            # Stage 3: Merge body + hands
            self._emit_stage(3)
            self.log_line.emit("Merging GVHMR and SMPLest-X parameters...")
            # merge_gvhmr_smplestx_params() call will be wired when Tab 2 UI
            # is built (Feature 07).
            if self._cancelled:
                return

            # Stages 4-6: Face pipeline, BVH/FBX conversion, Rendering
            # These stages will be wired by Feature 07.
            for stage_idx in (4, 5, 6):
                self._emit_stage(stage_idx)
                if self._cancelled:
                    return

            self.progress.emit(1.0, "Pipeline complete")
            results["output_dir"] = str(self._output_dir)
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))

    def _emit_stage(self, idx: int):
        """Emit progress for the start of stage *idx*."""
        start, _end, label = _FULL_STAGES[idx]
        self.progress.emit(start, label)

    def _gvhmr_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "tools/demo/demo.py",
            f"--video={self._video_path}",
        ]
        if self._config.static_cam:
            cmd.append("--static_cam")
        if self._config.use_dpvo:
            cmd.append("--use_dpvo")
        if self._config.focal_mm != 24.0:
            cmd.append(f"--f_mm={self._config.focal_mm}")
        return cmd

    def _smplestx_command(self) -> list[str]:
        return [
            self._smplestx_python,
            "smplestx_inference.py",
            f"--video={self._video_path}",
            f"--fps={self._fps}",
            f"--output_dir={self._output_dir}",
            "--no_render",
        ]


class MultiPersonWorker(QThread):
    """Wraps ``split_multi_person_video()`` with progress mapped to signals."""

    progress = Signal(float, str)
    log_line = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        video_path: Path,
        config: PipelineConfig,
        gvhmr_root: Path,
        output_dir: Path,
        parent=None,
    ):
        super().__init__(parent)
        self._video_path = video_path
        self._config = config
        self._gvhmr_root = gvhmr_root
        self._output_dir = output_dir
        self._cancelled = False

    def run(self):
        try:
            from multi_person_split import split_multi_person_video

            def progress_callback(frac: float, msg: str):
                overall = 0.05 + frac * 0.85
                self.progress.emit(overall, msg)
                self.log_line.emit(f"[MultiPerson] {msg}")

            self.progress.emit(0.02, "Starting multi-person pipeline...")

            result = split_multi_person_video(
                video_path=str(self._video_path),
                output_dir=str(self._output_dir),
                gvhmr_dir=str(self._gvhmr_root),
                static_cam=self._config.static_cam,
                use_dpvo=self._config.use_dpvo,
                max_persons=self._config.max_persons,
                progress_callback=progress_callback,
            )

            if self._cancelled:
                return

            self.progress.emit(1.0, "Multi-person pipeline complete")
            self.finished.emit({
                "output_dir": str(self._output_dir),
                "result": result,
            })

        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True
