"""QThread worker for running GVHMR pipeline subprocesses."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from models.pipeline_config import PipelineConfig


# Progress stage patterns (from gvhmr_gui.py stdout parsing)
_STAGE_PATTERNS = [
    (re.compile(r"preprocess", re.I), 0.05, "Preprocessing video..."),
    (re.compile(r"vitpose|pose.estimation", re.I), 0.15, "Running ViTPose..."),
    (re.compile(r"dpvo|droid|slam", re.I), 0.30, "Running SLAM..."),
    (re.compile(r"gvhmr|hmr4d|Running model", re.I), 0.55, "Running GVHMR..."),
    (re.compile(r"render|visualiz", re.I), 0.80, "Rendering outputs..."),
    (re.compile(r"done|finish|complete|saved", re.I), 1.0, "Done"),
]


class GVHMRWorker(QThread):
    """Runs GVHMR demo.py as a subprocess with progress tracking."""

    progress = Signal(float, str)
    log_line = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, video_path: Path, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._config = config
        self._cancelled = False

    def run(self):
        try:
            cmd = self._build_command()
            self.log_line.emit(f"$ {' '.join(str(c) for c in cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                if self._cancelled:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return

                line = line.rstrip()
                self.log_line.emit(line)
                self._parse_progress(line)

            returncode = process.wait()
            if returncode != 0:
                self.error.emit(f"GVHMR exited with code {returncode}")
                return

            result = self._collect_results()
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True

    def _build_command(self) -> list[str]:
        cmd = [
            "python", "demo.py",
            "--video", str(self._video_path),
        ]
        if self._config.static_cam:
            cmd.append("--static_cam")
        if self._config.use_dpvo:
            cmd.append("--use_dpvo")
        if self._config.focal_mm != 24.0:
            cmd.extend(["--f_mm", str(self._config.focal_mm)])
        return cmd

    def _parse_progress(self, line: str):
        for pattern, fraction, stage in _STAGE_PATTERNS:
            if pattern.search(line):
                self.progress.emit(fraction, stage)
                break

    def _collect_results(self) -> dict:
        # Scan for output files in expected locations
        video_stem = self._video_path.stem
        output_base = Path("outputs") / "gvhmr" / video_stem
        result = {
            "output_dir": str(output_base),
            "video_path": str(self._video_path),
        }
        # Look for standard output files
        for pattern in ["*incam*.mp4", "*global*.mp4", "*side_by_side*.mp4", "hmr4d_results.pt"]:
            matches = list(output_base.glob(pattern))
            if matches:
                key = pattern.replace("*", "").replace(".mp4", "").replace(".pt", "").strip("_")
                result[key] = str(matches[0])
        return result
