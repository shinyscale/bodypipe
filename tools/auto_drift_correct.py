"""Auto-correct SLAM drift using a local VLM via Ollama.

Usage (single-person):
    python -m tools.auto_drift_correct <person_dir> \\
        --video <video_path>

Usage (multi-person):
    python -m tools.auto_drift_correct <output_dir> \\
        --video <video_path> --multi-person

Samples keyframes from the source video, sends them to a local VLM (via Ollama)
to estimate scene-relative person displacement, compares against GVHMR's root
trajectory, and writes correction anchors compatible with ``compute_position_offsets()``.

In multi-person mode, draws detection bboxes on frames for person identification,
estimates per-person screen positions, and enforces inter-person spatial consistency.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


def _load_results(results_path: Path, person_id: int) -> dict:
    """Load transl_world, transl_incam, and K_fullimg from a results file.

    Returns dict with keys ``transl_world``, and optionally ``transl_incam``
    and ``K`` (for per-frame scale computation).
    """
    results = torch.load(str(results_path), map_location="cpu", weights_only=False)
    out: dict = {}

    # --- transl_world ---
    person_key = f"person_{person_id}"
    if person_key in results and isinstance(results[person_key], dict):
        person = results[person_key]
        if "transl_world" in person:
            out["transl_world"] = np.array(person["transl_world"], dtype=np.float64)
        elif "transl" in person:
            out["transl_world"] = np.array(person["transl"], dtype=np.float64)
    elif "transl_world" in results:
        out["transl_world"] = np.array(results["transl_world"], dtype=np.float64)
    elif "smpl_params_global" in results:
        g = results["smpl_params_global"]
        key = "transl_world" if "transl_world" in g else "transl"
        out["transl_world"] = np.array(g[key], dtype=np.float64)

    if "transl_world" not in out:
        raise KeyError(
            f"Cannot find transl_world for person {person_id} in {results_path}. "
            f"Available keys: {list(results.keys())}"
        )

    # --- transl_incam (camera-space depth for scale computation) ---
    if "smpl_params_incam" in results and "transl" in results["smpl_params_incam"]:
        out["transl_incam"] = np.array(
            results["smpl_params_incam"]["transl"], dtype=np.float64
        )

    # --- K_fullimg (camera intrinsics) ---
    if "K_fullimg" in results:
        out["K"] = np.array(results["K_fullimg"], dtype=np.float64)

    return out


def _run_multi_person(args, output_dir: Path, video_path: Path) -> int:
    """Multi-person drift correction using inter-person VLM constraints."""
    import cv2

    from workers.drift_correct import (
        annotate_frame_with_persons,
        check_ollama,
        compute_drift_severity,
        compute_multi_person_drift,
        estimate_person_positions,
        compute_per_frame_scale,
        extract_frames,
        find_slam,
        load_detection_bboxes,
        load_slam_w2c,
        sample_keyframe_indices,
    )

    # --- Load session manifest ---
    manifest_path = output_dir / "session_manifest.json"
    if not manifest_path.is_file():
        log.error("No session_manifest.json in %s — is this a multi-person output?", output_dir)
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    person_bindings = manifest.get("person_bindings", [])
    if len(person_bindings) < 2:
        log.error("Need at least 2 persons for multi-person mode, found %d", len(person_bindings))
        return 1

    person_dirs = [Path(pb["person_dir"]) for pb in person_bindings]
    n_persons = len(person_dirs)
    log.info("Multi-person mode: %d persons", n_persons)

    # --- Load per-person results ---
    per_person_transl: dict[int, np.ndarray] = {}
    per_person_incam: dict[int, np.ndarray] = {}
    K_shared: np.ndarray | None = None
    num_frames = 0

    for pidx, pdir in enumerate(person_dirs):
        results_path = pdir / "demo" / "isolated_video" / "hmr4d_results.pt"
        if not results_path.exists():
            # Try hybrid file
            for cand_name in ["spring_refined_hybrid_smplx.pt", "hmr4d_results.pt"]:
                cand = pdir / cand_name
                if cand.exists():
                    results_path = cand
                    break
        if not results_path.exists():
            log.warning("No results for person %d at %s, skipping", pidx, pdir)
            continue

        try:
            rdata = _load_results(results_path, 0)
        except KeyError:
            log.warning("Cannot load transl_world for person %d, skipping", pidx)
            continue

        per_person_transl[pidx] = rdata["transl_world"]
        if "transl_incam" in rdata:
            per_person_incam[pidx] = rdata["transl_incam"]
        if K_shared is None and "K" in rdata:
            K_shared = rdata["K"]

        num_frames = max(num_frames, len(rdata["transl_world"]))
        log.info("  Person %d: %d frames from %s", pidx, len(rdata["transl_world"]), results_path.name)

    if len(per_person_transl) < 2:
        log.error("Need results for at least 2 persons, got %d", len(per_person_transl))
        return 1

    # --- Apply world offsets + floor normalization ---
    # Replicate bodypipe's _normalize_world_source so corrections match viewport.
    offsets_raw = manifest.get("offsets", {})
    for pidx, pdir in enumerate(person_dirs):
        if pidx not in per_person_transl:
            continue
        tw = per_person_transl[pidx]

        # Floor Y normalization: same as app_window (floor_y = tw[0,1] - 0.933)
        if tw.shape[0] > 0:
            floor_y = float(tw[0, 1]) - 0.933
            tw[:, 1] -= floor_y

        # World offset from person_offsets.json (via session_manifest)
        tid = person_bindings[pidx]["track_id"]
        offset_cam = offsets_raw.get(str(tid))
        if offset_cam is not None:
            # Transform camera-frame offset to world frame using frame-0 orientations
            rdata = _load_results(
                pdir / "demo" / "isolated_video" / "hmr4d_results.pt", 0
            )
            results_pt = torch.load(
                str(pdir / "demo" / "isolated_video" / "hmr4d_results.pt"),
                map_location="cpu", weights_only=False,
            )
            go_incam_0 = np.array(results_pt["smpl_params_incam"]["global_orient"][0])
            go_world_key = "global_orient_world"
            if go_world_key in results_pt:
                go_world_0 = np.array(results_pt[go_world_key][0])
            elif "smpl_params_global" in results_pt and "global_orient" in results_pt["smpl_params_global"]:
                go_world_0 = np.array(results_pt["smpl_params_global"]["global_orient"][0])
            else:
                go_world_0 = go_incam_0  # fallback

            from scipy.spatial.transform import Rotation
            R_incam = Rotation.from_rotvec(go_incam_0).as_matrix()
            R_world = Rotation.from_rotvec(go_world_0).as_matrix()
            R_c2w = R_world @ R_incam.T

            off_3d = np.array(
                [offset_cam[0], 0.0, offset_cam[2] if len(offset_cam) > 2 else 0.0],
                dtype=np.float64,
            )
            off_w = R_c2w @ off_3d
            tw[:, 0] += off_w[0]
            tw[:, 2] += off_w[2]
            log.info("  Person %d (track %d): applied world offset (%.3f, %.3f)",
                     pidx, tid, off_w[0], off_w[2])

    # --- Load SLAM ---
    slam_path = find_slam(output_dir)
    if slam_path is None:
        log.error("Cannot find SLAM file")
        return 1
    log.info("Loading SLAM from %s", slam_path)
    slam_w2c = load_slam_w2c(slam_path)

    # --- Load detection bboxes ---
    try:
        all_bboxes, detection_masks = load_detection_bboxes(output_dir, manifest)
    except FileNotFoundError as e:
        log.error("%s", e)
        return 1
    log.info("Loaded detection bboxes for %d persons", len(all_bboxes))
    for pidx, mask in sorted(detection_masks.items()):
        detected = int(mask.sum())
        total = len(mask)
        if detected < total:
            log.info("  Person %d: %d/%d frames detected (%.0f%%)",
                     pidx, detected, total, 100 * detected / total)

    # --- Video info + frame sampling ---
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    log.info("Video: %.1f FPS, %d frames", fps, video_frame_count)

    # Detection bboxes are in GVHMR frame space (same count as SLAM/results)
    det_frame_count = max(len(b) for b in all_bboxes.values()) if all_bboxes else num_frames
    frame_ratio = video_frame_count / det_frame_count if det_frame_count > 0 else 1.0
    if abs(frame_ratio - 1.0) > 0.01:
        log.info("Frame ratio (video/detection): %.3f", frame_ratio)

    gvhmr_indices = sample_keyframe_indices(
        det_frame_count, fps,
        interval_sec=args.interval,
        max_samples=args.max_samples,
    )

    # Nudge keyframes away from undetected spans — prefer frames where all
    # persons have real detections.  Search within ±half-step for a frame
    # where every person is detected; keep the original if none found.
    step = max(1, int(round(fps * args.interval)))
    half_step = step // 2
    nudged = 0
    for ki, gvi in enumerate(gvhmr_indices):
        all_detected = all(
            detection_masks.get(pidx, np.ones(1, dtype=bool))[min(gvi, len(detection_masks.get(pidx, np.ones(1, dtype=bool))) - 1)]
            for pidx in range(n_persons)
        )
        if all_detected:
            continue
        # Search outward from gvi for a nearby frame with full detection
        best = None
        for offset in range(1, half_step + 1):
            for candidate in [gvi + offset, gvi - offset]:
                if candidate < 0 or candidate >= det_frame_count:
                    continue
                if all(
                    detection_masks.get(pidx, np.ones(1, dtype=bool))[min(candidate, len(detection_masks.get(pidx, np.ones(1, dtype=bool))) - 1)]
                    for pidx in range(n_persons)
                ):
                    best = candidate
                    break
            if best is not None:
                break
        if best is not None:
            gvhmr_indices[ki] = best
            nudged += 1
    if nudged:
        log.info("Nudged %d keyframes to frames with full detection coverage", nudged)

    video_indices = [min(int(round(i * frame_ratio)), video_frame_count - 1) for i in gvhmr_indices]
    log.info("Sampled %d keyframes (GVHMR): %s", len(gvhmr_indices), gvhmr_indices)

    # --- Extract and annotate frames ---
    raw_frames = extract_frames(video_path, video_indices)
    vid_to_gvhmr = dict(zip(video_indices, gvhmr_indices))

    frames: dict[int, np.ndarray] = {}
    visible_persons_per_frame: dict[int, list[int]] = {}
    for vi, gvi in vid_to_gvhmr.items():
        if vi not in raw_frames:
            continue
        frame = raw_frames[vi]

        # Get bboxes at this GVHMR frame, scaled to the extracted frame's resolution
        if K_shared is not None:
            orig_w = int(K_shared[0, 0, 2] * 2)
            orig_h = int(K_shared[0, 1, 2] * 2)
        else:
            orig_w, orig_h = 1920, 1080
        frame_h, frame_w = frame.shape[:2]
        sx, sy = frame_w / orig_w, frame_h / orig_h

        frame_bboxes: dict[int, tuple] = {}
        for pidx, bbox_arr in all_bboxes.items():
            if gvi < len(bbox_arr):
                # Skip persons not actually detected at this frame
                mask = detection_masks.get(pidx)
                if mask is not None and gvi < len(mask) and not mask[gvi]:
                    continue
                x1, y1, x2, y2 = bbox_arr[gvi]
                frame_bboxes[pidx] = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)

        frames[gvi] = annotate_frame_with_persons(frame, frame_bboxes)
        visible_persons_per_frame[gvi] = sorted(frame_bboxes.keys())

    log.info("Annotated %d frames with person bboxes", len(frames))
    if len(frames) < 2:
        log.error("Need at least 2 frames, got %d", len(frames))
        return 1

    # --- Check Ollama ---
    ollama_url = args.ollama_url
    if not check_ollama(ollama_url):
        log.error("Ollama not reachable at %s", ollama_url)
        return 1

    # --- VLM: estimate per-person positions ---
    t0 = time.monotonic()
    person_positions = estimate_person_positions(
        frames, gvhmr_indices, n_persons,
        interval_sec=args.interval,
        model=args.model,
        ollama_url=ollama_url,
        visible_persons_per_frame=visible_persons_per_frame,
    )
    log.info("VLM inference done in %.1fs, got %d frame estimates", time.monotonic() - t0, len(person_positions))

    # --- Compute per-frame scale from intrinsics ---
    # Use reference person's depth for scale (all persons share similar depth)
    ref_pid = sorted(per_person_incam.keys())[0] if per_person_incam else 0
    if ref_pid in per_person_incam and K_shared is not None:
        scale_per_frame = compute_per_frame_scale(per_person_incam[ref_pid], K_shared)
        log.info(
            "Per-frame scale: min=%.2f, max=%.2f, mean=%.2f m/frac",
            scale_per_frame.min(), scale_per_frame.max(), scale_per_frame.mean(),
        )
    else:
        log.warning("No incam/intrinsics — using fixed scale=%.1f", args.scale)
        scale_per_frame = args.scale

    # Debug: verify per_person_transl has offsets applied
    for pidx, tw in per_person_transl.items():
        log.info("  per_person_transl[%d] frame 0: (%.3f, %.3f, %.3f)",
                 pidx, tw[0][0], tw[0][1], tw[0][2])

    # --- Compute per-person drift anchors from relative spacing ---
    all_anchors = compute_multi_person_drift(
        per_person_transl, person_positions, slam_w2c,
        scale_per_frame=scale_per_frame,
        reference_person=None,  # auto-select least-drifted person
        detection_masks=detection_masks,
    )

    # --- Compute drift severity spans (no VLM needed) ---
    drift_spans = compute_drift_severity(
        per_person_transl, detection_masks=detection_masks,
    )
    for pidx, spans in sorted(drift_spans.items()):
        log.info("  Person %d: %d drift spans", pidx, len(spans))
        for s, e, sev in spans:
            log.info("    frames %d–%d  severity %.3fm", s, e, sev)

    # --- Write output ---
    out_path = args.out or (output_dir / "drift_corrections.json")
    persons_out = []
    for pidx in sorted(all_anchors.keys()):
        anchors = all_anchors[pidx]
        drift_m = 0.0
        if anchors and pidx in per_person_transl:
            tw = per_person_transl[pidx]
            last_f, last_t = anchors[-1]
            if last_f < len(tw):
                drift_m = float(np.linalg.norm(tw[last_f].astype(np.float64) - last_t))

        # Detection range: first/last frame where this person was actually detected
        det_range = None
        mask = detection_masks.get(pidx)
        if mask is not None and mask.any():
            first_det = int(np.argmax(mask))
            last_det = int(len(mask) - 1 - np.argmax(mask[::-1]))
            detected_count = int(mask.sum())
            det_range = {
                "first_frame": first_det,
                "last_frame": last_det,
                "detected_frames": detected_count,
                "total_frames": len(mask),
            }

        persons_out.append({
            "person_id": pidx,
            "person_dir": str(person_dirs[pidx]) if pidx < len(person_dirs) else None,
            "anchors": [
                {"frame": int(f), "target": [round(float(x), 6) for x in t]}
                for f, t in anchors
            ],
            "estimated_drift_m": round(drift_m, 4),
            "detection_range": det_range,
        })

    out_data = {
        "video_path": str(video_path),
        "mode": "multi_person",
        "num_persons": n_persons,
        "model_used": args.model,
        "persons": persons_out,
        "drift_spans": {
            str(pidx): [[s, e, sev] for s, e, sev in spans]
            for pidx, spans in drift_spans.items()
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out_data, fh, indent=2)

    log.info("Wrote multi-person corrections to %s", out_path)
    for p in persons_out:
        log.info("  Person %d: %d anchors, estimated drift %.3fm",
                 p["person_id"], len(p["anchors"]), p["estimated_drift_m"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Auto-correct SLAM drift using a local VLM (Gemma 4).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="GVHMR output directory (or person directory) containing results .pt files.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--person",
        type=int,
        default=0,
        help="Person index (default: 0).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between sampled keyframes (default: 2.0).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10,
        help="Maximum number of keyframes to sample (default: 10).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="Meters per fractional screen width (default: 3.0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma4:31b",
        help="Ollama model name (default: gemma4:31b).",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--multi-person",
        action="store_true",
        help="Multi-person mode: operate on top-level output dir with all person_N dirs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <output_dir>/drift_corrections.json).",
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

    if args.multi_person:
        return _run_multi_person(args, output_dir, video_path)

    # --- Load GVHMR root translation ---
    results_path = output_dir / "hmr4d_results.pt"
    if not results_path.exists():
        results_path = output_dir / "hpe_results.pt"
    if not results_path.exists():
        # Try nested GVHMR structure: person_N/demo/isolated_video/hmr4d_results.pt
        cand = output_dir / "demo" / "isolated_video" / "hmr4d_results.pt"
        if cand.exists():
            results_path = cand
    if not results_path.exists():
        log.error("No hmr4d_results.pt or hpe_results.pt in %s", output_dir)
        return 1

    log.info("Loading results from %s", results_path)
    try:
        rdata = _load_results(results_path, args.person)
    except KeyError as exc:
        log.error("%s", exc)
        return 1

    transl_world = rdata["transl_world"]
    num_frames = len(transl_world)
    log.info("Loaded %d frames of root translation for person %d", num_frames, args.person)

    # --- Load SLAM ---
    from workers.drift_correct import (
        camera_displacements_to_world,
        check_ollama,
        compute_drift_anchors,
        compute_per_frame_scale,
        estimate_displacements,
        extract_frames,
        find_slam,
        load_slam_w2c,
        sample_keyframe_indices,
    )

    slam_path = find_slam(output_dir)
    if slam_path is None:
        log.error("Cannot find SLAM file (shared_slam.pt or demo/preprocess/slam.pt)")
        return 1

    log.info("Loading SLAM from %s", slam_path)
    slam_w2c = load_slam_w2c(slam_path)

    # --- Get video FPS and frame count ---
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    log.info("Video FPS: %.1f, frames: %d", fps, video_frame_count)

    # Map GVHMR frame indices → video frame indices (may differ due to
    # re-encoding at target_fps or person isolation cropping).
    frame_ratio = video_frame_count / num_frames if num_frames > 0 else 1.0
    if abs(frame_ratio - 1.0) > 0.01:
        log.info("Frame ratio (video/results): %.3f — will remap indices", frame_ratio)

    # --- Sample keyframes and extract frames ---
    # Sample in GVHMR frame space, then map to video frame space for extraction
    gvhmr_indices = sample_keyframe_indices(
        num_frames, fps,
        interval_sec=args.interval,
        max_samples=args.max_samples,
    )
    video_indices = [min(int(round(i * frame_ratio)), video_frame_count - 1) for i in gvhmr_indices]
    log.info("Sampled %d keyframes (GVHMR): %s", len(gvhmr_indices), gvhmr_indices)
    if abs(frame_ratio - 1.0) > 0.01:
        log.info("Mapped to video indices: %s", video_indices)

    raw_frames = extract_frames(video_path, video_indices)
    # Re-key from video indices → GVHMR indices so downstream uses GVHMR frame space
    vid_to_gvhmr = dict(zip(video_indices, gvhmr_indices))
    frames = {vid_to_gvhmr[vi]: raw_frames[vi] for vi in raw_frames if vi in vid_to_gvhmr}
    log.info("Extracted %d frames", len(frames))
    if len(frames) < 2:
        log.error("Need at least 2 frames, got %d", len(frames))
        return 1

    # --- Check Ollama connectivity ---
    ollama_url = args.ollama_url
    if not check_ollama(ollama_url):
        log.error("Ollama not reachable at %s — is it running?", ollama_url)
        return 1

    # --- Run VLM inference ---
    log.info("Using Ollama model %s at %s", args.model, ollama_url)
    t0 = time.monotonic()
    displacements = estimate_displacements(
        frames, gvhmr_indices, fps,
        interval_sec=args.interval,
        model=args.model,
        ollama_url=ollama_url,
    )
    log.info(
        "VLM inference done in %.1fs, got %d displacement estimates",
        time.monotonic() - t0, len(displacements),
    )

    # --- Compute per-frame scale from camera intrinsics + depth ---
    if "transl_incam" in rdata and "K" in rdata:
        scale_per_frame = compute_per_frame_scale(rdata["transl_incam"], rdata["K"])
        log.info(
            "Per-frame scale: min=%.2f, max=%.2f, mean=%.2f m/frac",
            scale_per_frame.min(), scale_per_frame.max(), scale_per_frame.mean(),
        )
    else:
        log.warning("No incam transl or K — falling back to fixed scale=%.1f", args.scale)
        scale_per_frame = args.scale

    # --- Convert to world-space and compute anchors ---
    vision_world = camera_displacements_to_world(
        displacements, slam_w2c,
        scale_per_frame=scale_per_frame,
    )

    anchors = compute_drift_anchors(transl_world, vision_world)
    log.info("Generated %d correction anchors", len(anchors))

    # --- Summary stats ---
    total_drift_m = 0.0
    drift_desc = "negligible"
    if len(anchors) >= 2:
        last_frame = anchors[-1][0]
        last_target = anchors[-1][1]
        total_drift_vec = transl_world[last_frame].astype(np.float64) - last_target
        total_drift_m = float(np.linalg.norm(total_drift_vec))

        dx, dz = float(total_drift_vec[0]), float(total_drift_vec[2])
        parts = []
        if abs(dx) > 0.01:
            parts.append(f"{'+' if dx > 0 else '-'}X")
        if abs(dz) > 0.01:
            parts.append(f"{'+' if dz > 0 else '-'}Z")
        drift_desc = "mostly " + "/".join(parts) if parts else "negligible"

    # --- Write output ---
    out_path = args.out or (output_dir / "drift_corrections.json")
    out_data = {
        "video_path": str(video_path),
        "person_id": args.person,
        "num_frames": num_frames,
        "fps": fps,
        "model_used": args.model,
        "scale_source": "per_frame_intrinsics" if isinstance(scale_per_frame, np.ndarray) else "fixed",
        "scale_fallback": args.scale,
        "anchors": [
            {"frame": int(f), "target": [round(float(x), 6) for x in t]}
            for f, t in anchors
        ],
        "estimated_total_drift_m": round(total_drift_m, 4),
        "drift_direction_desc": drift_desc,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out_data, fh, indent=2)

    log.info("Wrote %d anchors to %s", len(anchors), out_path)
    log.info("Estimated total drift: %.3fm (%s)", total_drift_m, drift_desc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
