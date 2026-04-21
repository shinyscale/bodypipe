"""Re-run SAM2 segmentation + ProPainter inpainting on an existing multi-person output.

Usage:
    python -m tools.reinpaint <output_dir> --video <video_path>

Reuses cached detection tracks and SLAM. Produces new isolated videos with
inpainting for overlapping frames. After this, re-run the GVHMR solve
(on WSL2 if pytorch3d is not available on the current platform).

Example:
    python -m tools.reinpaint F:\\GVHMR\\GVHMR\\outputs\\multi_person\\Hit_Em_01 ^
        --video "C:\\Users\\Zachary Andrews\\Downloads\\Hit_Em_01.mp4"
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-isolate persons with SAM2 + ProPainter inpainting.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Multi-person output directory (contains detection/, person_N/, etc.).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base_plus",
        help="SAM2 model size: tiny, small, base_plus, large (default: base_plus).",
    )
    parser.add_argument(
        "--reprompt-interval",
        type=int,
        default=30,
        help="Re-prompt SAM2 from tracking bboxes every N frames (default: 30).",
    )
    parser.add_argument(
        "--max-long-side",
        type=int,
        default=576,
        help="Max long side for ProPainter inpainting resolution (default: 576).",
    )
    parser.add_argument(
        "--skip-sam2",
        action="store_true",
        help="Skip SAM2 segmentation (use cached masks).",
    )
    parser.add_argument(
        "--skip-inpaint",
        action="store_true",
        help="Skip ProPainter inpainting (only run SAM2 segmentation).",
    )
    parser.add_argument(
        "--persons",
        type=int,
        nargs="*",
        default=None,
        help="Only re-isolate specific person indices (default: all).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = args.output_dir.expanduser().resolve()
    video_path = args.video.expanduser().resolve()

    if not video_path.is_file():
        log.error("Video not found: %s", video_path)
        return 1
    if not output_dir.is_dir():
        log.error("Output directory not found: %s", output_dir)
        return 1

    # --- Load detection tracks ---
    tracks_path = output_dir / "detection" / "all_tracks.pt"
    if not tracks_path.is_file():
        log.error("Detection tracks not found: %s", tracks_path)
        return 1

    data = torch.load(str(tracks_path), map_location="cpu", weights_only=False)
    all_tracks = data["tracks"]
    log.info("Loaded %d tracks from %s", len(all_tracks), tracks_path)

    # --- Load session manifest ---
    manifest_path = output_dir / "session_manifest.json"
    if not manifest_path.is_file():
        log.error("No session_manifest.json in %s", output_dir)
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    person_bindings = manifest.get("person_bindings", [])
    if len(person_bindings) < 2:
        log.error("Need at least 2 persons, found %d", len(person_bindings))
        return 1

    # Build track list matching person order
    tid_to_track = {int(t["track_id"]): t for t in all_tracks}
    ordered_tracks = []
    for pb in person_bindings:
        tid = pb["track_id"]
        if tid not in tid_to_track:
            log.error("Track %d from manifest not found in all_tracks.pt", tid)
            return 1
        ordered_tracks.append(tid_to_track[tid])
    log.info("Person bindings: %s", [pb["track_id"] for pb in person_bindings])

    # --- Step 1: SAM2 Segmentation ---
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    masks: dict[int, np.ndarray] = {}

    if args.skip_sam2:
        # Load cached masks
        cached_count = 0
        for t in ordered_tracks:
            tid = int(t["track_id"])
            mask_file = masks_dir / f"person_{tid}_masks.npz"
            if mask_file.exists():
                masks[tid] = np.load(str(mask_file))["masks"]
                cached_count += 1
        if cached_count < len(ordered_tracks):
            log.error(
                "Only %d/%d cached masks found. Run without --skip-sam2 first.",
                cached_count, len(ordered_tracks),
            )
            return 1
        log.info("Loaded %d cached mask files", cached_count)
    else:
        # Add GVHMR to path for SAM2 imports
        gvhmr_dir = Path(__file__).resolve().parent.parent.parent / "GVHMR"
        if str(gvhmr_dir) not in sys.path:
            sys.path.insert(0, str(gvhmr_dir))

        try:
            from sam2_segmenter import SAM2Segmenter
        except ImportError as e:
            log.error("SAM2 not available: %s", e)
            log.error("Install with: pip install segment-anything-2")
            return 1

        log.info("Starting SAM2 segmentation (model=%s)...", args.model_size)
        t0 = time.monotonic()

        segmenter = SAM2Segmenter(model_size=args.model_size)
        masks = segmenter.segment_all_persons(
            str(video_path),
            ordered_tracks,
            reprompt_interval=args.reprompt_interval,
        )
        del segmenter
        torch.cuda.empty_cache()

        # Save masks
        for tid, mask_array in masks.items():
            np.savez_compressed(
                str(masks_dir / f"person_{tid}_masks.npz"),
                masks=mask_array,
            )

        elapsed = time.monotonic() - t0
        log.info("SAM2 segmentation done in %.1fs, %d person masks saved", elapsed, len(masks))

        # Compute overlap map
        track_ids = list(masks.keys())
        if len(track_ids) >= 2:
            num_frames = masks[track_ids[0]].shape[0]
            overlap = np.zeros(num_frames, dtype=bool)
            for f in range(num_frames):
                for i, tid_a in enumerate(track_ids):
                    for tid_b in track_ids[i + 1:]:
                        if np.logical_and(masks[tid_a][f], masks[tid_b][f]).any():
                            overlap[f] = True
                            break
                    if overlap[f]:
                        break
            np.savez_compressed(str(masks_dir / "overlap_map.npz"), overlap=overlap)
            overlap_frames = int(overlap.sum())
            log.info("Overlap: %d/%d frames (%.1f%%)",
                     overlap_frames, num_frames, 100 * overlap_frames / num_frames)

    if args.skip_inpaint:
        log.info("Skipping inpainting (--skip-inpaint). Masks saved to %s", masks_dir)
        return 0

    if not masks:
        log.error("No masks available for inpainting")
        return 1

    # --- Step 2: ProPainter isolation per person ---
    gvhmr_dir = Path(__file__).resolve().parent.parent.parent / "GVHMR"
    if str(gvhmr_dir) not in sys.path:
        sys.path.insert(0, str(gvhmr_dir))

    try:
        from propainter_inpaint import ProPainterInpainter, isolate_person
    except ImportError as e:
        log.error("ProPainter not available: %s", e)
        return 1

    inpainter = ProPainterInpainter()
    persons_to_process = args.persons if args.persons is not None else list(range(len(person_bindings)))

    for pidx in persons_to_process:
        if pidx >= len(person_bindings):
            log.warning("Person index %d out of range (max %d), skipping", pidx, len(person_bindings) - 1)
            continue

        pb = person_bindings[pidx]
        tid = pb["track_id"]
        person_dir = output_dir / f"person_{pidx}"
        person_dir.mkdir(parents=True, exist_ok=True)

        track = tid_to_track[tid]
        bboxes = np.array(track["bbx_xyxy"], dtype=np.float32)
        det_mask = np.array(track.get("detection_mask", np.ones(len(bboxes), dtype=bool)), dtype=bool)

        isolated_video = person_dir / "isolated_video.mp4"

        # Back up existing isolation
        if isolated_video.exists():
            bak = person_dir / "isolated_video.mp4.bak"
            if bak.exists():
                bak.unlink()
            shutil.copy2(str(isolated_video), str(bak))
            log.info("  Backed up existing isolated video for person %d", pidx)

        log.info("Isolating person %d (track %d)...", pidx, tid)
        t0 = time.monotonic()

        isolation_result = isolate_person(
            video_path=str(video_path),
            target_person_id=tid,
            target_track_bboxes=bboxes,
            target_detection_mask=det_mask,
            all_masks=masks,
            output_path=str(isolated_video),
            inpainter=inpainter,
            propainter_max_long_side=args.max_long_side,
        )

        elapsed = time.monotonic() - t0
        num_inpainted = isolation_result.get("num_inpainted", 0)
        num_crop = isolation_result.get("num_crop_only", 0)
        log.info(
            "  Person %d done in %.1fs: %d inpainted, %d crop-only",
            pidx, elapsed, num_inpainted, num_crop,
        )

        # Save isolation metadata
        with open(person_dir / "isolation_mode.json", "w") as f:
            json.dump(isolation_result.get("isolation_modes", []), f)
        with open(person_dir / "isolation_debug.json", "w") as f:
            json.dump(isolation_result.get("debug", {}), f, indent=2)

    inpainter.cleanup()
    torch.cuda.empty_cache()

    log.info("Re-isolation complete. Next step: re-run GVHMR solve on WSL2.")
    log.info("  cd /mnt/f/GVHMR/GVHMR")
    log.info("  conda activate gvhmr")
    for pidx in persons_to_process:
        if pidx < len(person_bindings):
            pdir = output_dir / f"person_{pidx}"
            log.info("  python demo.py --video=%s/isolated_video.mp4 --output_root=%s/demo ...",
                     pdir, pdir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
