"""QThread worker for incremental per-person reprocessing."""

from __future__ import annotations

import time

import numpy as np
from pathlib import Path

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
            import torch
            from multi_person_split import reprocess_person

            output_dir = self._session.output_dir
            if output_dir is None:
                self.error.emit("No output directory set on session")
                return

            # Load cached all_tracks from detection stage
            tracks_path = output_dir / "detection" / "all_tracks.pt"
            if not tracks_path.is_file():
                self.error.emit(f"Track cache not found: {tracks_path}")
                return
            all_tracks_data = torch.load(str(tracks_path), map_location="cpu", weights_only=False)
            # all_tracks.pt stores {"tracks": [...], "metadata": {...}}
            if isinstance(all_tracks_data, dict) and "tracks" in all_tracks_data:
                all_tracks = all_tracks_data["tracks"]
            else:
                all_tracks = all_tracks_data  # legacy format: bare list

            slam_path = str(output_dir / "shared_slam.pt")
            masks_dir = str(output_dir / "masks")

            total = len(self._person_ids)
            reprocessed = []
            person_timings: list[tuple[int, float]] = []
            pipeline_t0 = time.monotonic()

            for i, pid in enumerate(self._person_ids):
                if self._cancelled:
                    return

                track = self._session.person_tracks.get(pid)
                if track is None or track.person_dir is None:
                    continue

                person_t0 = time.monotonic()
                self.progress.emit(i / total, f"Reprocessing person {pid}...")

                # Find person_index in all_tracks by matching track_id
                person_index = None
                for idx, t in enumerate(all_tracks):
                    if t.get("track_id", idx) == pid:
                        person_index = idx
                        break
                if person_index is None:
                    self.progress.emit(
                        (i + 1) / total,
                        f"Person {pid}: no matching track in all_tracks",
                    )
                    continue

                # Build fully-interpolated bboxes from keyframe corrections
                orig = track.original_bboxes if track.original_bboxes is not None else track.bboxes
                if orig is None:
                    self.progress.emit(
                        (i + 1) / total,
                        f"Person {pid}: no bbox data for reprocess",
                    )
                    continue

                if track.bbox_corrections is not None:
                    from views.identity_inspector import interpolate_bbox_corrections
                    # Interpolate keyframe corrections across all frames
                    kf_frames = [kf["frame"] for kf in track.keyframes]
                    interpolated = interpolate_bbox_corrections(
                        np.array(orig), track.bbox_corrections,
                        keyframe_frames=kf_frames,
                    )
                    # Build final bboxes: use interpolated where non-zero, else original
                    updated = np.array(orig, dtype=float).copy()
                    non_zero = ~np.all(interpolated == 0, axis=1)
                    updated[non_zero] = interpolated[non_zero]
                else:
                    updated = np.array(orig, dtype=float)

                # Build manual_bbox_keyframes for DP track selection
                manual_bbox_keyframes = None
                if track.bbox_corrections is not None and track.keyframes:
                    manual_bbox_keyframes = {}
                    for kf in track.keyframes:
                        f = kf["frame"]
                        if f < len(track.bbox_corrections):
                            bbox = track.bbox_corrections[f]
                            if not np.all(bbox == 0):
                                manual_bbox_keyframes[f] = bbox.astype(np.float32)
                    if not manual_bbox_keyframes:
                        manual_bbox_keyframes = None

                def _progress_cb(frac, msg, _i=i, _total=total):
                    overall = (i + frac) / _total
                    self.progress.emit(overall, msg)

                # Use the same backend the track was originally processed with
                backend = "gemx" if track.body_model_type == "soma" else "gvhmr"

                reprocess_person(
                    video_path=str(self._session.video_path),
                    person_index=person_index,
                    person_dir=str(track.person_dir),
                    original_bboxes=updated.astype(np.float32),
                    all_tracks=all_tracks,
                    slam_path=slam_path,
                    masks_dir=masks_dir,
                    static_cam=self._session.static_cam,
                    use_dpvo=self._session.use_dpvo,
                    progress_callback=_progress_cb,
                    estimation_backend=backend,
                    manual_bbox_keyframes=manual_bbox_keyframes,
                    crossing_threshold=self._session.crossing_threshold,
                )

                person_elapsed = time.monotonic() - person_t0
                person_timings.append((pid, person_elapsed))
                reprocessed.append(pid)
                self.person_done.emit(pid)

            total_elapsed = time.monotonic() - pipeline_t0
            self.progress.emit(1.0, "Done")

            # Timing summary
            stage_timings = {}
            if person_timings:
                self.log_line.emit("")
                self.log_line.emit("── Reprocess Timings ──")
                for pid, secs in person_timings:
                    label = f"Person {pid}"
                    self.log_line.emit(f"  {label:<28s} {secs:6.1f}s")
                    stage_timings[label] = round(secs, 2)
                self.log_line.emit(f"  {'TOTAL':<28s} {total_elapsed:6.1f}s")
                self.log_line.emit("")

            self.finished.emit({"reprocessed": reprocessed, "stage_timings": stage_timings})

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def cancel(self):
        self._cancelled = True
