"""QThread worker for scene preview rendering."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from models.session import Session


class RenderWorker(QThread):
    """Renders multi-person in-camera scene preview."""

    progress = Signal(float, str)
    finished = Signal(object)  # Path to rendered video
    error = Signal(str)

    def __init__(self, session: Session, parent=None):
        super().__init__(parent)
        self._session = session
        self._cancelled = False

    def run(self):
        try:
            from multi_person_split import render_multi_person_incam

            self.progress.emit(0.1, "Rendering scene preview...")

            output_dir = self._session.output_dir
            if output_dir is None:
                self.error.emit("No output directory set")
                return

            person_dirs = [
                str(t.person_dir)
                for t in self._session.person_tracks.values()
                if t.person_dir is not None
            ]

            result_path = render_multi_person_incam(
                video_path=str(self._session.video_path),
                person_dirs=person_dirs,
                output_dir=str(output_dir),
            )

            self.progress.emit(1.0, "Done")
            self.finished.emit(Path(result_path))

        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True
