"""QThread worker for running GVHMR demo.py as a subprocess."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from models.pipeline_config import PipelineConfig
from workers._base import SubprocessWorkerBase


# Progress stage patterns — ported from gvhmr_gui.py:79-87
STAGE_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    (re.compile(r"preprocess|loading video|reading video", re.I), 0.05, "Preprocessing"),
    (re.compile(r"yolo|tracking|detection", re.I), 0.15, "YOLO Tracking"),
    (re.compile(r"vitpose|pose estimation|2d pose", re.I), 0.30, "ViTPose"),
    (re.compile(r"hmr2|hmr4d_feature|feature extraction", re.I), 0.45, "HMR2 Features"),
    (re.compile(r"dpvo|simple_vo|camera estimation|slam", re.I), 0.60, "Camera Estimation"),
    (re.compile(r"gvhmr|predicting|prediction|diffusion", re.I), 0.80, "GVHMR Prediction"),
    (re.compile(r"render|saving|visualization", re.I), 0.95, "Rendering"),
]


def find_output_dir(video_path: Path, gvhmr_root: Path) -> Path | None:
    """Locate GVHMR output directory for a given video.

    Search order (matches gvhmr_gui.py:105-120):
      1. ``outputs/demo/{stem}``
      2. ``outputs/{stem}``
      3. Most recently modified directory under ``outputs/demo/``
    """
    stem = video_path.stem
    outputs = gvhmr_root / "outputs"

    candidate = outputs / "demo" / stem
    if candidate.is_dir():
        return candidate

    candidate = outputs / stem
    if candidate.is_dir():
        return candidate

    demo_dir = outputs / "demo"
    if demo_dir.is_dir():
        dirs = [p for p in demo_dir.iterdir() if p.is_dir()]
        if dirs:
            return max(dirs, key=lambda p: p.stat().st_mtime)

    return None


def find_file(output_dir: Path, pattern: str) -> Path | None:
    """Find the first file matching *pattern* under *output_dir*."""
    matches = list(output_dir.rglob(pattern))
    return matches[0] if matches else None


def match_stage_progress(line: str) -> tuple[float, str] | None:
    """Match a stdout line against ``STAGE_PATTERNS``.

    Returns ``(fraction, stage_name)`` on match, else ``None``.
    """
    for pattern, fraction, stage in STAGE_PATTERNS:
        if pattern.search(line):
            return fraction, stage
    return None


class GVHMRWorker(SubprocessWorkerBase):
    """Runs ``tools/demo/demo.py`` as a subprocess with progress tracking."""

    def __init__(
        self,
        video_path: Path,
        config: PipelineConfig,
        gvhmr_root: Path,
        extra_args: list[str] | None = None,
        parse_progress: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self._video_path = video_path
        self._config = config
        self._gvhmr_root = gvhmr_root
        self._extra_args = extra_args or []
        self._parse_progress = parse_progress

    def run(self):
        try:
            cmd = self._build_command()
            self.log_line.emit(f"$ {' '.join(cmd)}")
            self.progress.emit(0.02, "Starting GVHMR pipeline...")

            t0 = time.monotonic()
            returncode, log_lines = self._run_subprocess(cmd, self._gvhmr_root)
            elapsed = time.monotonic() - t0

            if self._cancelled:
                return
            if returncode != 0:
                self.error.emit(f"GVHMR exited with code {returncode}")
                return

            self.log_line.emit("")
            self.log_line.emit(f"── GVHMR completed in {elapsed:.1f}s ──")
            self.log_line.emit("")

            result = self._collect_results()
            result["log"] = "\n".join(log_lines)
            result["stage_timings"] = {"GVHMR body solve": round(elapsed, 2)}
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))

    def _on_stdout_line(self, line: str):
        if self._parse_progress:
            match = match_stage_progress(line)
            if match:
                self.progress.emit(match[0], match[1])

    def _build_command(self) -> list[str]:
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
        cmd.extend(self._extra_args)
        return cmd

    def _collect_results(self) -> dict:
        """Scan for output files in the expected GVHMR output directory."""
        output_dir = find_output_dir(self._video_path, self._gvhmr_root)
        result: dict = {
            "output_dir": str(output_dir) if output_dir else None,
            "video_path": str(self._video_path),
        }
        if output_dir:
            for key, pat in [
                ("side_by_side", "*side_by_side*.mp4"),
                ("incam", "*incam*.mp4"),
                ("global_view", "*global*.mp4"),
                ("pt_file", "hmr4d_results.pt"),
            ]:
                found = find_file(output_dir, pat)
                if found:
                    result[key] = str(found)
        return result
