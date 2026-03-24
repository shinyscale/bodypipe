#!/usr/bin/env python3
"""Compare GVHMR (SMPL-X) vs GEM-X (SOMA) pipeline outputs side-by-side.

Loads both pipeline outputs for the same video, applies crop→original camera
transform, runs FK, and compares positioning, orientation, smoothness, and
confidence. Produces a detailed comparison report.

Usage:
    cd ~/bodypipe && source .venv/bin/activate
    python tests/compare_pipelines.py \
        /path/to/gvhmr_output \
        /path/to/gemx_output \
        [--width 1920] [--height 1080]
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np

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


def load_gvhmr_person(person_dir, vid_w, vid_h):
    """Load GVHMR SMPL-X output and apply crop→original transform."""
    person_dir = Path(person_dir)
    pt_files = list(person_dir.rglob("hmr4d_results.pt"))
    if not pt_files:
        return None

    data = torch.load(str(pt_files[0]), map_location="cpu")
    incam = data.get("smpl_params_incam", {})
    glob = data.get("smpl_params_global", {})
    if "body_pose" not in incam:
        return None

    params = {
        "body_pose": np.array(incam["body_pose"]),
        "global_orient": np.array(incam["global_orient"]),
        "transl": np.array(incam["transl"]),
        "betas": np.array(incam.get("betas", np.zeros((1, 10)))),
        "body_model_type": "smplx",
    }
    if "K_fullimg" in data:
        params["K_fullimg"] = np.array(data["K_fullimg"])

    # Confidence from static_conf_logits
    net_out = data.get("net_outputs", {})
    conf_logits = net_out.get("static_conf_logits")
    if conf_logits is not None:
        cl = np.array(conf_logits)
        if cl.ndim == 3:
            cl = cl[0]
        probs = 1.0 / (1.0 + np.exp(-cl.astype(np.float64)))
        params["confidences"] = probs.max(axis=1).astype(np.float32)

    _apply_crop_transform(params, person_dir, vid_w, vid_h)
    return params


def load_gemx_person(person_dir, vid_w, vid_h):
    """Load GEM-X SOMA output and apply crop→original transform."""
    from workers.gemx_worker import load_gemx_soma_output
    person_dir = Path(person_dir)
    params = load_gemx_soma_output(person_dir)
    if params is None:
        return None
    _apply_crop_transform(params, person_dir, vid_w, vid_h)
    return params


def _apply_crop_transform(params, person_dir, vid_w, vid_h):
    """Transform transl from crop camera → original video camera."""
    meta_path = Path(person_dir) / "person_meta.json"
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text())
    crop_bbox = meta.get("crop_bbox")
    if not crop_bbox:
        return

    x1, y1 = float(crop_bbox[0]), float(crop_bbox[1])
    cx_orig, cy_orig = vid_w / 2.0, vid_h / 2.0
    f_orig = float(max(vid_w, vid_h))

    K = params.get("K_fullimg")
    if K is None:
        return
    K_arr = np.asarray(K)
    if K_arr.ndim == 3:
        K_arr = K_arr[0]

    tr = np.asarray(params["transl"], dtype=np.float32).copy()
    X, Y, Z = tr[:, 0], tr[:, 1], tr[:, 2]
    tr[:, 0] = (K_arr[0, 0] * X + (K_arr[0, 2] + x1 - cx_orig) * Z) / f_orig
    tr[:, 1] = (K_arr[1, 1] * Y + (K_arr[1, 2] + y1 - cy_orig) * Z) / f_orig
    params["transl"] = tr


def fk_joints(params, frame):
    """Run FK for either SMPL-X or SOMA params."""
    from views.mesh_viewport import forward_kinematics, _forward_kinematics_soma
    is_soma = "poses" in params and "body_pose" not in params
    if is_soma:
        return _forward_kinematics_soma(params, frame)
    return forward_kinematics(params, frame)


def orientation_smoothness(global_orient):
    """Compute frame-to-frame angular velocity (rad/frame)."""
    go = np.asarray(global_orient)
    deltas = np.linalg.norm(np.diff(go, axis=0), axis=1)
    return deltas


def translation_smoothness(transl):
    """Compute frame-to-frame position change (m/frame)."""
    tr = np.asarray(transl)
    deltas = np.linalg.norm(np.diff(tr, axis=0), axis=1)
    return deltas


def head_above_ratio(params, n_frames):
    """Fraction of frames where head is above pelvis in camera space."""
    ok = 0
    for f in range(n_frames):
        j = fk_joints(params, f)
        if j is not None and j[15, 1] < j[0, 1]:
            ok += 1
    return ok / n_frames


def compare(gvhmr_dir, gemx_dir, vid_w=1920, vid_h=1080):
    gvhmr_dir = Path(gvhmr_dir)
    gemx_dir = Path(gemx_dir)

    gvhmr_persons = sorted(
        [d for d in gvhmr_dir.iterdir() if d.is_dir() and d.name.startswith("person_")]
    )
    gemx_persons = sorted(
        [d for d in gemx_dir.iterdir() if d.is_dir() and d.name.startswith("person_")]
    )

    n_persons = min(len(gvhmr_persons), len(gemx_persons))
    if n_persons == 0:
        print("ERROR: No person directories found in one or both outputs")
        return

    print(f"{'='*70}")
    print(f"PIPELINE COMPARISON: GVHMR vs GEM-X ({vid_w}x{vid_h})")
    print(f"  GVHMR: {gvhmr_dir}")
    print(f"  GEM-X: {gemx_dir}")
    print(f"  Persons: {n_persons}")
    print(f"{'='*70}\n")

    all_gvhmr = {}
    all_gemx = {}

    for i in range(n_persons):
        gp = load_gvhmr_person(gvhmr_persons[i], vid_w, vid_h)
        ep = load_gemx_person(gemx_persons[i], vid_w, vid_h)

        if gp is None:
            print(f"Person {i}: GVHMR output missing/failed")
            continue
        if ep is None:
            print(f"Person {i}: GEM-X output missing/failed")
            continue

        pid = int(gvhmr_persons[i].name.split("_")[1])
        all_gvhmr[pid] = gp
        all_gemx[pid] = ep

        n_g = len(gp["transl"])
        n_e = len(ep["transl"])
        n = min(n_g, n_e)

        print(f"--- Person {pid} ({n} frames) ---")
        print(f"  {'Metric':<35} {'GVHMR':>12} {'GEM-X':>12} {'Winner':>8}")
        print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*8}")

        # Body model type
        g_bmt = gp.get("body_model_type", "smplx")
        e_bmt = ep.get("body_model_type", "soma")
        print(f"  {'Body model':<35} {g_bmt:>12} {e_bmt:>12}")

        # Orientation smoothness
        g_orient = orientation_smoothness(gp["global_orient"][:n])
        e_orient = orientation_smoothness(ep["global_orient"][:n])
        g_om, e_om = g_orient.mean(), e_orient.mean()
        winner = "GVHMR" if g_om < e_om else "GEM-X"
        print(f"  {'Orient smoothness (rad/f, lower=↑)':<35} {g_om:>12.4f} {e_om:>12.4f} {winner:>8}")

        g_omax, e_omax = g_orient.max(), e_orient.max()
        winner = "GVHMR" if g_omax < e_omax else "GEM-X"
        print(f"  {'Orient max jump (rad)':<35} {g_omax:>12.4f} {e_omax:>12.4f} {winner:>8}")

        # Translation smoothness
        g_trans = translation_smoothness(gp["transl"][:n])
        e_trans = translation_smoothness(ep["transl"][:n])
        g_tm, e_tm = g_trans.mean(), e_trans.mean()
        winner = "GVHMR" if g_tm < e_tm else "GEM-X"
        print(f"  {'Transl smoothness (m/f, lower=↑)':<35} {g_tm:>12.4f} {e_tm:>12.4f} {winner:>8}")

        g_tmax, e_tmax = g_trans.max(), e_trans.max()
        winner = "GVHMR" if g_tmax < e_tmax else "GEM-X"
        print(f"  {'Transl max jump (m)':<35} {g_tmax:>12.4f} {e_tmax:>12.4f} {winner:>8}")

        # Head above pelvis ratio
        g_har = head_above_ratio(gp, n) * 100
        e_har = head_above_ratio(ep, n) * 100
        winner = "GVHMR" if g_har > e_har else "GEM-X"
        print(f"  {'Head above pelvis (%)':<35} {g_har:>11.0f}% {e_har:>11.0f}% {winner:>8}")

        # Z depth range
        g_zmin, g_zmax = gp["transl"][:n, 2].min(), gp["transl"][:n, 2].max()
        e_zmin, e_zmax = ep["transl"][:n, 2].min(), ep["transl"][:n, 2].max()
        print(f"  {'Z depth range (m)':<35} {g_zmin:.2f}-{g_zmax:.2f}     {e_zmin:.2f}-{e_zmax:.2f}")

        # Confidence
        g_conf = gp.get("confidences")
        e_conf = ep.get("confidences")
        if g_conf is not None and e_conf is not None:
            g_cm, e_cm = g_conf[:n].mean(), e_conf[:n].mean()
            winner = "GVHMR" if g_cm > e_cm else "GEM-X"
            print(f"  {'Mean confidence':<35} {g_cm:>12.3f} {e_cm:>12.3f} {winner:>8}")

        # Position agreement — how far apart are the two pipelines' estimates?
        pos_diff = np.linalg.norm(gp["transl"][:n] - ep["transl"][:n], axis=1)
        print(f"  {'Position agreement (m, lower=↑)':<35} {pos_diff.mean():>12.3f}")
        print(f"  {'Position agree range':<35} {pos_diff.min():.3f}-{pos_diff.max():.3f}")

        print()

    # Multi-person comparison
    pids = sorted(set(all_gvhmr.keys()) & set(all_gemx.keys()))
    if len(pids) >= 2:
        p0, p1 = pids[0], pids[1]
        n = min(len(all_gvhmr[p0]["transl"]), len(all_gvhmr[p1]["transl"]),
                len(all_gemx[p0]["transl"]), len(all_gemx[p1]["transl"]))

        g_xsep = all_gvhmr[p0]["transl"][:n, 0] - all_gvhmr[p1]["transl"][:n, 0]
        e_xsep = all_gemx[p0]["transl"][:n, 0] - all_gemx[p1]["transl"][:n, 0]

        print(f"--- Multi-Person Positioning ---")
        print(f"  {'Metric':<35} {'GVHMR':>12} {'GEM-X':>12}")
        print(f"  {'-'*35} {'-'*12} {'-'*12}")
        print(f"  {'Mean X separation (m)':<35} {g_xsep.mean():>12.3f} {e_xsep.mean():>12.3f}")
        print(f"  {'X sep std (lower=↑)':<35} {g_xsep.std():>12.3f} {e_xsep.std():>12.3f}")
        print(f"  {'X sep sign consistent':<35} {'yes' if np.all(g_xsep > 0) or np.all(g_xsep < 0) else 'NO':>12} {'yes' if np.all(e_xsep > 0) or np.all(e_xsep < 0) else 'NO':>12}")

        g_ysep = all_gvhmr[p0]["transl"][:n, 1] - all_gvhmr[p1]["transl"][:n, 1]
        e_ysep = all_gemx[p0]["transl"][:n, 1] - all_gemx[p1]["transl"][:n, 1]
        print(f"  {'Mean Y separation (m)':<35} {g_ysep.mean():>12.3f} {e_ysep.mean():>12.3f}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gvhmr_dir", help="GVHMR multi-person output directory")
    parser.add_argument("gemx_dir", help="GEM-X multi-person output directory")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    args = parser.parse_args()
    compare(args.gvhmr_dir, args.gemx_dir, args.width, args.height)
