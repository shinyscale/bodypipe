#!/usr/bin/env python3
"""Comprehensive headless validation of multi-person GEM-X pipeline output.

Loads pipeline output, applies crop→original camera transform, runs FK
and SOMA forward pass, and produces a detailed report with pass/fail
verdicts. Designed to run without the GUI.

Usage:
    cd ~/bodypipe && source .venv/bin/activate
    python tests/validate_pipeline_output.py /path/to/output_dir [vid_w] [vid_h]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np

# Setup paths
GVHMR_ROOT = Path("/home/zacharymandrews/GVHMR")
GEMX_ROOT = Path("/home/zacharymandrews/GEM-X")
BODYPIPE_ROOT = Path("/home/zacharymandrews/bodypipe")
for p in [
    str(GVHMR_ROOT),
    str(GVHMR_ROOT / ".venv/lib/python3.12/site-packages"),
    str(GEMX_ROOT / ".venv/lib/python3.12/site-packages"),
    str(GEMX_ROOT),
    str(BODYPIPE_ROOT),
]:
    if p not in sys.path:
        sys.path.append(p)

import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from scipy.spatial.transform import Rotation
from workers.gemx_worker import load_gemx_soma_output
from views.mesh_viewport import _forward_kinematics_soma, forward_kinematics


def crop_to_original(params, person_dir, vid_w, vid_h):
    """Apply crop→original camera transform to transl."""
    meta = json.loads((person_dir / "person_meta.json").read_text())
    crop_bbox = meta["crop_bbox"]
    x1, y1 = float(crop_bbox[0]), float(crop_bbox[1])
    cx_orig, cy_orig = vid_w / 2.0, vid_h / 2.0
    f_orig = float(max(vid_w, vid_h))

    K = np.asarray(params["K_fullimg"])
    if K.ndim == 3:
        K = K[0]
    f_crop_x = float(K[0, 0])
    f_crop_y = float(K[1, 1])
    cx_crop = float(K[0, 2])
    cy_crop = float(K[1, 2])

    tr = np.asarray(params["transl"], dtype=np.float32).copy()
    X, Y, Z = tr[:, 0], tr[:, 1], tr[:, 2]
    tr[:, 0] = (f_crop_x * X + (cx_crop + x1 - cx_orig) * Z) / f_orig
    tr[:, 1] = (f_crop_y * Y + (cy_crop + y1 - cy_orig) * Z) / f_orig
    params["transl"] = tr
    return meta


def run_validation(output_dir, vid_w=1920, vid_h=1080):
    output_dir = Path(output_dir)
    person_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("person_")]
    )

    if not person_dirs:
        print("FAIL: No person directories found")
        return False

    print(f"=== Validating {output_dir.name} ({vid_w}x{vid_h}) ===")
    print(f"Found {len(person_dirs)} persons\n")

    all_params = {}
    all_meta = {}
    passes = 0
    fails = 0

    def check(name, condition, detail=""):
        nonlocal passes, fails
        if condition:
            passes += 1
            print(f"  PASS: {name}")
        else:
            fails += 1
            print(f"  FAIL: {name} — {detail}")
        return condition

    # ── Load all persons ──
    for pdir in person_dirs:
        pid = int(pdir.name.split("_")[1])
        params = load_gemx_soma_output(pdir)
        if params is None:
            print(f"  FAIL: Person {pid} — no GEM-X output found")
            fails += 1
            continue

        meta = crop_to_original(params, pdir, vid_w, vid_h)
        all_params[pid] = params
        all_meta[pid] = meta

        n_frames = len(params["transl"])
        bmt = params.get("body_model_type", "?")
        print(f"--- Person {pid} ({bmt}, {n_frames} frames, crop={meta['crop_bbox']}) ---")

        tr = params["transl"]

        # Basic sanity
        check("transl finite", np.all(np.isfinite(tr)))
        check("Z depth > 0", np.all(tr[:, 2] > 0), f"min Z = {tr[:, 2].min():.3f}")
        check("Z depth < 10m", np.all(tr[:, 2] < 10), f"max Z = {tr[:, 2].max():.3f}")

        # Orientation stability
        go = np.asarray(params["global_orient"])
        max_delta = 0
        for i in range(1, len(go)):
            max_delta = max(max_delta, np.linalg.norm(go[i] - go[i - 1]))
        check("orient stable (< 0.5 rad/frame)", max_delta < 0.5,
              f"max delta = {max_delta:.3f}")

        # FK orientation check — head should be above pelvis on MOST frames.
        # Dancers may bend forward, so allow up to 30% "head below" frames.
        is_soma = "poses" in params and "body_pose" not in params
        fk_func = _forward_kinematics_soma if is_soma else forward_kinematics
        n_ok = 0
        for frame in range(n_frames):
            joints = fk_func(params, frame)
            if joints is not None and joints[15, 1] < joints[0, 1]:
                n_ok += 1
        pct = n_ok / n_frames * 100
        check(f"head above pelvis (>70% frames)", pct > 70,
              f"{pct:.0f}% of frames")

        # Translation smoothness (no teleporting)
        diffs = np.linalg.norm(np.diff(tr, axis=0), axis=1)
        max_jump = diffs.max()
        check("no teleporting (< 0.5m/frame)", max_jump < 0.5,
              f"max jump = {max_jump:.3f}m")

        # Confidence
        conf = params.get("confidences")
        if conf is not None:
            check("confidence in [0,1]", np.all(conf >= 0) and np.all(conf <= 1))
            mean_conf = conf.mean()
            check(f"mean confidence > 0.2", mean_conf > 0.2,
                  f"mean = {mean_conf:.3f}")
            print(f"        confidence: min={conf.min():.3f} mean={mean_conf:.3f} max={conf.max():.3f}")

        print()

    # ── Multi-person checks ──
    pids = sorted(all_params.keys())
    if len(pids) >= 2:
        print("--- Multi-person positioning ---")
        t0 = all_params[pids[0]]["transl"]
        t1 = all_params[pids[1]]["transl"]
        n = min(len(t0), len(t1))

        x_seps = t0[:n, 0] - t1[:n, 0]
        check("X separation > 0.1m", abs(x_seps[0]) > 0.1,
              f"frame 0 sep = {abs(x_seps[0]):.3f}m")
        check("X separation stable", x_seps.std() < 0.15,
              f"std = {x_seps.std():.3f}")

        y_seps = t0[:n, 1] - t1[:n, 1]
        check("Y separation small (same floor)", abs(y_seps.mean()) < 0.3,
              f"mean Y sep = {y_seps.mean():.3f}")

        # Both persons should be at similar depth
        z0_mean = t0[:n, 2].mean()
        z1_mean = t1[:n, 2].mean()
        check("similar depth", abs(z0_mean - z1_mean) < 1.0,
              f"Z means: {z0_mean:.2f} vs {z1_mean:.2f}")

        print(f"\n        Frame  0: P{pids[0]}=({t0[0,0]:+.3f},{t0[0,1]:+.3f},{t0[0,2]:.2f})  "
              f"P{pids[1]}=({t1[0,0]:+.3f},{t1[0,1]:+.3f},{t1[0,2]:.2f})")
        mid = n // 2
        print(f"        Frame {mid:2d}: P{pids[0]}=({t0[mid,0]:+.3f},{t0[mid,1]:+.3f},{t0[mid,2]:.2f})  "
              f"P{pids[1]}=({t1[mid,0]:+.3f},{t1[mid,1]:+.3f},{t1[mid,2]:.2f})")
        print(f"        Frame {n-1:2d}: P{pids[0]}=({t0[n-1,0]:+.3f},{t0[n-1,1]:+.3f},{t0[n-1,2]:.2f})  "
              f"P{pids[1]}=({t1[n-1,0]:+.3f},{t1[n-1,1]:+.3f},{t1[n-1,2]:.2f})")

    print(f"\n{'='*50}")
    print(f"RESULTS: {passes} passed, {fails} failed")
    return fails == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Multi-person pipeline output directory")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()

    success = run_validation(args.output_dir, args.width, args.height)
    sys.exit(0 if success else 1)
