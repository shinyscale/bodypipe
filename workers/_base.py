"""Base class for workers that manage subprocesses."""

from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal


class SubprocessWorkerBase(QThread):
    """Base QThread worker with subprocess management and cancellation."""

    progress = Signal(float, str)
    log_line = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False
        self._proc: subprocess.Popen | None = None

    def cancel(self):
        """Request cancellation and terminate any running subprocess."""
        self._cancelled = True
        self._terminate_proc()

    def _run_subprocess(
        self, cmd: list[str], cwd: str | Path, env: dict | None = None,
    ) -> tuple[int, list[str]]:
        """Run a subprocess, streaming stdout line-by-line.

        Emits ``log_line`` for each line and calls the ``_on_stdout_line`` hook.
        Returns ``(returncode, log_lines)``.  *returncode* is ``-1`` when cancelled.

        If *env* is provided, it is merged into ``os.environ`` (overrides only
        the specified keys, preserving PATH and other essentials).
        """
        import os

        full_env = None
        if env:
            full_env = {**os.environ, **env}

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(cwd),
            env=full_env,
        )

        log_lines: list[str] = []
        for raw_line in self._proc.stdout:
            if self._cancelled:
                self._terminate_proc()
                return -1, log_lines

            line = raw_line.rstrip()
            log_lines.append(line)
            self.log_line.emit(line)
            self._on_stdout_line(line)

        returncode = self._proc.wait()
        self._proc = None
        return returncode, log_lines

    def _on_stdout_line(self, line: str):
        """Subclass hook called for each stdout line after ``log_line`` is emitted."""

    def _terminate_proc(self):
        """Gracefully terminate the subprocess, force-kill after 5 s."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
