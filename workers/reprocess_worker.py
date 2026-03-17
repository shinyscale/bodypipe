"""QThread worker for incremental per-person reprocessing."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from models.session import Session


class ReprocessWorker(QThread):
    """Runs reprocess_person() for each dirty person."""

    progress = Signal(float, str)
    person_done = Signal(int)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, session: Session, person_ids: list[int], parent=None):
        super().__init__(parent)
        self._session = session
        self._person_ids = person_ids
        self._cancelled = False

    def run(self):
        try:
            from multi_person_split import reprocess_person

            total = len(self._person_ids)
            reprocessed = []

            for i, pid in enumerate(self._person_ids):
                if self._cancelled:
                    return

                track = self._session.person_tracks.get(pid)
                if track is None or track.person_dir is None:
                    continue

                self.progress.emit(i / total, f"Reprocessing person {pid}...")

                reprocess_person(
                    person_dir=str(track.person_dir),
                    video_path=str(self._session.video_path),
                    static_cam=self._session.static_cam,
                    use_dpvo=self._session.use_dpvo,
                )

                reprocessed.append(pid)
                self.person_done.emit(pid)

            self.progress.emit(1.0, "Done")
            self.finished.emit({"reprocessed": reprocessed})

        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True
