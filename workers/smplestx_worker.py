"""QThread worker for running SMPLest-X inference subprocess."""

from __future__ import annotations

from pathlib import Path

from workers._base import SubprocessWorkerBase


class SMPLestXWorker(SubprocessWorkerBase):
    """Runs SMPLest-X inference as a child process."""

    def __init__(
        self,
        video_path: Path,
        fps: float,
        output_dir: Path,
        smplestx_python: str | Path,
        smplestx_dir: str | Path,
        parent=None,
    ):
        super().__init__(parent)
        self._video_path = video_path
        self._fps = fps
        self._output_dir = output_dir
        self._smplestx_python = str(smplestx_python)
        self._smplestx_dir = str(smplestx_dir)

    def run(self):
        try:
            cmd = self._build_command()
            self.log_line.emit(f"$ {' '.join(cmd)}")
            self.progress.emit(0.1, "Starting SMPLest-X inference...")

            returncode, log_lines = self._run_subprocess(cmd, self._smplestx_dir)

            if self._cancelled:
                return
            if returncode != 0:
                self.error.emit(f"SMPLest-X exited with code {returncode}")
                return

            result_file = self._find_result()
            self.progress.emit(1.0, "SMPLest-X complete")
            self.finished.emit({
                "output_dir": str(self._output_dir),
                "result_file": str(result_file) if result_file else None,
                "log": "\n".join(log_lines),
            })

        except Exception as e:
            self.error.emit(str(e))

    def _build_command(self) -> list[str]:
        return [
            self._smplestx_python,
            "smplestx_inference.py",
            f"--video={self._video_path}",
            f"--fps={self._fps}",
            f"--output_dir={self._output_dir}",
            "--no_render",
        ]

    def _find_result(self) -> Path | None:
        """Return the most recently modified .pt or .npz in output_dir."""
        out = Path(self._output_dir)
        candidates = list(out.glob("*.pt")) + list(out.glob("*.npz"))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)
