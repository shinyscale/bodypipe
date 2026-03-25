"""QThread worker for incremental per-person reprocessing."""

from __future__ import annotations

import json
import logging
import numpy as np
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from models.session import Session

log = logging.getLogger(__name__)


def _load_person_meta(person_dir: Path | None) -> dict | None:
    if person_dir is None:
        return None
    meta_path = Path(person_dir) / "person_meta.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def _resolve_person_index(person_id: int, track, all_tracks: list[dict]) -> int | None:
    meta = _load_person_meta(track.person_dir if track is not None else None)
    source_index = meta.get("source_index") if isinstance(meta, dict) else None
    track_id = meta.get("track_id") if isinstance(meta, dict) else None

    if source_index is not None:
        source_index = int(source_index)
        if 0 <= source_index < len(all_tracks):
            candidate = all_tracks[source_index]
            candidate_track_id = candidate.get("track_id", source_index)
            if track_id is None or int(candidate_track_id) == int(track_id):
                return source_index

    if track_id is not None:
        wanted_track_id = int(track_id)
        for idx, candidate in enumerate(all_tracks):
            if int(candidate.get("track_id", idx)) == wanted_track_id:
                return idx

    for idx, candidate in enumerate(all_tracks):
        if candidate.get("track_id", idx) == person_id:
            return idx

    if 0 <= person_id < len(all_tracks):
        return person_id
    return None


def build_dense_rerun_bboxes(
    original_bboxes: np.ndarray,
    bbox_corrections: np.ndarray | None,
) -> np.ndarray:
    """Build a dense bbox prior from sparse keyframe edits.

    A single corrected frame now constrains the whole rerun with a constant
    delta, while multiple corrected frames interpolate deltas across time and
    hold the end deltas outside the keyed span.
    """
    original = np.asarray(original_bboxes, dtype=np.float32)
    if original.ndim != 2 or original.shape[1] != 4:
        raise ValueError("original_bboxes must have shape (N, 4)")

    updated = original.copy()
    if bbox_corrections is None:
        return updated

    corrections = np.asarray(bbox_corrections, dtype=np.float32)
    if corrections.ndim != 2 or corrections.shape[1] != 4:
        raise ValueError("bbox_corrections must have shape (N, 4)")

    dense_corrections = np.zeros_like(original)
    copy_len = min(len(original), len(corrections))
    dense_corrections[:copy_len] = corrections[:copy_len]

    edited_frames = np.flatnonzero(~np.all(dense_corrections == 0, axis=1))
    if len(edited_frames) == 0:
        return updated

    keyed = dense_corrections[edited_frames]
    deltas = keyed - original[edited_frames]

    if len(edited_frames) == 1:
        interpolated_deltas = np.repeat(deltas, len(original), axis=0)
    else:
        sample_x = edited_frames.astype(np.float32)
        target_x = np.arange(len(original), dtype=np.float32)
        interpolated_deltas = np.stack(
            [
                np.interp(
                    target_x,
                    sample_x,
                    deltas[:, coord],
                    left=float(deltas[0, coord]),
                    right=float(deltas[-1, coord]),
                )
                for coord in range(4)
            ],
            axis=1,
        ).astype(np.float32)

    updated = original + interpolated_deltas
    updated[edited_frames] = keyed
    return updated


def extract_verified_identity_keyframes(track) -> list[dict]:
    """Build verified identity anchors from UI keyframes and bbox edits."""
    if track is None or not track.keyframes:
        return []

    original = track.original_bboxes if track.original_bboxes is not None else track.bboxes
    if original is None:
        return []
    original = np.asarray(original, dtype=np.float32)

    corrections = None
    if track.bbox_corrections is not None:
        corrections = np.asarray(track.bbox_corrections, dtype=np.float32)

    verified = []
    seen_frames = set()
    for keyframe in track.keyframes:
        if not keyframe.get("verified", False):
            continue
        frame_idx = int(keyframe.get("frame", -1))
        if frame_idx in seen_frames or frame_idx < 0 or frame_idx >= len(original):
            continue
        bbox = original[frame_idx].copy()
        if (
            corrections is not None
            and frame_idx < len(corrections)
            and not np.all(corrections[frame_idx] == 0)
        ):
            bbox = corrections[frame_idx].copy()
        verified.append(
            {
                "frame": frame_idx,
                "bbox": bbox.tolist(),
                "verified": True,
            }
        )
        seen_frames.add(frame_idx)

    verified.sort(key=lambda item: item["frame"])
    return verified


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

            log.info(
                "[ReprocessWorker] Starting reprocess for persons %s, output_dir=%s",
                self._person_ids, output_dir,
            )

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
            failed = []
            for i, pid in enumerate(self._person_ids):
                if self._cancelled:
                    return

                track = self._session.person_tracks.get(pid)
                if track is None or track.person_dir is None:
                    log.warning("[ReprocessWorker] Person %d: no track or person_dir, skipping", pid)
                    continue
                verified_identity_keyframes = extract_verified_identity_keyframes(track)

                self.progress.emit(i / total, f"Reprocessing person {pid}...")

                # Resolve source_index via person_meta first, then fall back.
                person_index = _resolve_person_index(pid, track, all_tracks)
                if person_index is None:
                    log.warning("[ReprocessWorker] Person %d: no matching track in all_tracks", pid)
                    self.progress.emit(
                        (i + 1) / total,
                        f"Person {pid}: no matching track in all_tracks",
                    )
                    continue

                # Build fully-interpolated bboxes from keyframe corrections
                orig = track.original_bboxes if track.original_bboxes is not None else track.bboxes
                if orig is None:
                    log.warning("[ReprocessWorker] Person %d: no bbox data", pid)
                    self.progress.emit(
                        (i + 1) / total,
                        f"Person {pid}: no bbox data for reprocess",
                    )
                    continue

                if track.bbox_corrections is not None:
                    corrections = np.asarray(track.bbox_corrections, dtype=np.float32)
                    num_corrected = int(np.sum(~np.all(corrections == 0, axis=1)))
                    log.info(
                        "[ReprocessWorker] Person %d: %d keyframe corrections out of %d frames",
                        pid, num_corrected, len(orig),
                    )

                    updated = build_dense_rerun_bboxes(orig, corrections)
                    num_changed = int(np.sum(np.any(np.abs(updated - np.asarray(orig, dtype=np.float32)) > 1e-4, axis=1)))

                    # Compute how much the bboxes actually changed
                    diff = np.abs(updated - np.asarray(orig, dtype=np.float32))
                    max_diff = float(diff.max()) if diff.size > 0 else 0.0
                    log.info(
                        "[ReprocessWorker] Person %d: dense rerun prior updated %d/%d frames, "
                        "max bbox delta=%.1fpx",
                        pid, num_changed, len(orig), max_diff,
                    )
                else:
                    log.info("[ReprocessWorker] Person %d: no bbox_corrections, using original bboxes", pid)
                    updated = np.asarray(orig, dtype=np.float32)

                def _progress_cb(frac, msg, _i=i, _total=total):
                    overall = (i + frac) / _total
                    self.progress.emit(overall, msg)

                # Use the same backend the track was originally processed with
                backend = "gemx" if track.body_model_type == "soma" else "gvhmr"
                log.info(
                    "[ReprocessWorker] Person %d: calling reprocess_person "
                    "(backend=%s, bboxes shape=%s)",
                    pid, backend, updated.shape,
                )

                result = reprocess_person(
                    video_path=str(self._session.video_path),
                    person_index=person_index,
                    person_dir=str(track.person_dir),
                    updated_bboxes=updated.astype(np.float32),
                    all_tracks=all_tracks,
                    slam_path=slam_path,
                    masks_dir=masks_dir,
                    static_cam=self._session.static_cam,
                    use_dpvo=self._session.use_dpvo,
                    progress_callback=_progress_cb,
                    estimation_backend=backend,
                    verified_identity_keyframes=verified_identity_keyframes,
                )

                if not result or not result.get("pt_path"):
                    log.error(
                        "[ReprocessWorker] Person %d: pipeline returned no results "
                        "(result=%s)", pid, result,
                    )
                    self.progress.emit(
                        (i + 1) / total,
                        f"Person {pid}: pipeline failed — check terminal for details",
                    )
                    failed.append(pid)
                    continue

                log.info(
                    "[ReprocessWorker] Person %d: success, pt_path=%s",
                    pid, result.get("pt_path"),
                )
                reprocessed.append(pid)
                self.person_done.emit(pid)

            self.progress.emit(1.0, "Done")
            summary = {
                "reprocessed": reprocessed,
            }
            if failed:
                log.warning("[ReprocessWorker] Failed persons: %s", failed)
                summary["failed"] = failed
            self.finished.emit(summary)

        except Exception as e:
            import traceback
            log.error("[ReprocessWorker] Crashed: %s\n%s", e, traceback.format_exc())
            self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True
