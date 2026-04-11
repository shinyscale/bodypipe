"""Diagnostic CLI: verify a PHC physics refinement run.

Usage:
    python -m tools.verify_physics <output_dir_or_person_dir> [--recompute]

What it does:
    1. Locates ``physics/phc_input.npz`` and ``physics/phc_output/phc_refined.npz``
       under the given directory.
    2. Loads both NPZs (Z-up AMASS format) and applies the same Z-up→Y-up
       conversion the runtime pipeline uses, so metric units match the rest
       of the codebase.
    3. Reports shapes, dtypes, frame counts, NaN counts, and per-axis
       translation/rotation diffs.
    4. Runs ``evaluate_refinement`` and prints an OK/WARN/FAIL verdict using
       the default thresholds documented in ``workers/physics/evaluate.py``.
    5. If ``physics/metrics.json`` already exists (written by the orchestrator
       on a fresh run), the verifier prints that file verbatim instead of
       recomputing — pass ``--recompute`` to force re-evaluation.

This is intentionally a single-file script with no UI dependencies so it can
be run on any output directory, including older runs that predate the
metrics.json scaffolding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# Z-up (PHC/MuJoCo) → Y-up (GVHMR): rotate +90° around X. Mirrors
# workers/physics/phc_to_smpl.py:_ZUP_TO_YUP — kept inline so this script has
# no dependency on the workers package and can be vendored elsewhere.
_ZUP_TO_YUP = np.array(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ],
    dtype=np.float64,
)


def _rotate_root_orient(global_orient_aa: np.ndarray, R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation as sRot

    return (sRot.from_matrix(R) * sRot.from_rotvec(global_orient_aa)).as_rotvec()


def _load_phc_npz(path: Path) -> dict:
    """Load a PHC-format NPZ (Z-up) and convert to a GVHMR Y-up params dict."""
    data = np.load(str(path), allow_pickle=True)

    if "poses" in data:
        poses = np.asarray(data["poses"], dtype=np.float64)
        n = poses.shape[0]
        global_orient_zup = poses[:, :3]
        body_pose = poses[:, 3:66].reshape(n, 21, 3)
    elif "body_pose" in data and "global_orient" in data:
        global_orient_zup = np.asarray(data["global_orient"], dtype=np.float64)
        body_pose = np.asarray(data["body_pose"], dtype=np.float64).reshape(
            -1, 21, 3
        )
        n = body_pose.shape[0]
    else:
        raise ValueError(
            f"Unrecognised NPZ format at {path}; keys = {list(data.keys())}"
        )

    trans_zup = np.asarray(
        data.get("trans", data.get("transl", np.zeros((n, 3)))),
        dtype=np.float64,
    )

    transl = trans_zup @ _ZUP_TO_YUP.T
    global_orient = _rotate_root_orient(global_orient_zup, _ZUP_TO_YUP)

    fps = float(
        data["mocap_framerate"] if "mocap_framerate" in data else 30.0
    )

    return {
        "num_frames": n,
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": transl,
        "fps": fps,
        "_raw_zup_poses": (
            data["poses"] if "poses" in data else None
        ),
        "_raw_zup_trans": trans_zup,
    }


def _find_npz_pair(target: Path) -> tuple[Path, Path]:
    """Locate input/refined NPZs given an output dir or person dir."""
    target = Path(target).expanduser().resolve()
    candidates: list[tuple[Path, Path]] = []

    # Direct: target *is* a physics dir
    direct_in = target / "phc_input.npz"
    direct_out = target / "phc_output" / "phc_refined.npz"
    if direct_in.is_file() and direct_out.is_file():
        return direct_in, direct_out

    # One level: target/physics/...
    one_in = target / "physics" / "phc_input.npz"
    one_out = target / "physics" / "phc_output" / "phc_refined.npz"
    if one_in.is_file() and one_out.is_file():
        return one_in, one_out

    # Recursive search (e.g. multi-person output dir)
    for input_npz in sorted(target.rglob("phc_input.npz")):
        out_npz = input_npz.parent / "phc_output" / "phc_refined.npz"
        if out_npz.is_file():
            candidates.append((input_npz, out_npz))

    if not candidates:
        raise FileNotFoundError(
            f"No phc_input.npz / phc_refined.npz pair found under {target}"
        )
    if len(candidates) > 1:
        print(
            f"  (found {len(candidates)} pairs under {target}, "
            "using first; pass a more specific path to disambiguate)"
        )
    return candidates[0]


def _print_summary(label: str, p: dict) -> None:
    print(f"  {label}:")
    print(f"    frames        : {p['num_frames']}")
    print(f"    global_orient : {p['global_orient'].shape} {p['global_orient'].dtype}")
    print(f"    body_pose     : {p['body_pose'].shape} {p['body_pose'].dtype}")
    print(f"    transl        : {p['transl'].shape} {p['transl'].dtype}")
    nans = (
        int(np.isnan(p["global_orient"]).sum())
        + int(np.isnan(p["body_pose"]).sum())
        + int(np.isnan(p["transl"]).sum())
    )
    print(f"    NaN count     : {nans}")


def _print_diff(raw: dict, ref: dict) -> None:
    n = min(raw["num_frames"], ref["num_frames"])
    if n == 0:
        print("  (no frames to diff)")
        return

    raw_t = raw["transl"][:n]
    ref_t = ref["transl"][:n]
    dt = ref_t - raw_t
    horiz = np.linalg.norm(dt[:, [0, 2]], axis=-1)
    vert = np.abs(dt[:, 1])

    raw_p = raw["body_pose"][:n].reshape(n, -1)
    ref_p = ref["body_pose"][:n].reshape(n, -1)
    dp = np.abs(ref_p - raw_p)

    print("  Per-axis translation drift (refined − raw, meters):")
    print(
        f"    horizontal (X,Z): mean {horiz.mean():.3f}, "
        f"max {horiz.max():.3f}"
    )
    print(
        f"    vertical (Y)    : mean {vert.mean():.3f}, "
        f"max {vert.max():.3f}"
    )

    raw_g = raw["global_orient"][:n]
    ref_g = ref["global_orient"][:n]
    dg = np.abs(ref_g - raw_g)
    print("  Root rotation diff (axis-angle, radians):")
    print(f"    mean {dg.mean():.3f}, max {dg.max():.3f}")
    print(
        "    NOTE: large values may indicate axis-angle sign flip, not real "
        "rotation difference."
    )

    print("  Body pose diff (radians):")
    print(f"    mean {dp.mean():.3f}, max {dp.max():.3f}, RMS {np.sqrt((dp**2).mean()):.3f}")


def _print_metrics(metrics: dict) -> None:
    print("  Quality metrics:")
    for key in sorted(metrics):
        val = metrics[key]
        if isinstance(val, dict):
            for sk, sv in val.items():
                if isinstance(sv, (int, float)):
                    print(f"    {key}.{sk}: {sv:.4f}")
        elif isinstance(val, (int, float, bool)):
            print(f"    {key}: {val}")


def _print_verdict(status: str, reason: str) -> None:
    label_map = {
        "ok": "OK",
        "warn": "WARN",
        "fail": "FAIL",
    }
    label = label_map.get(status, status.upper())
    print()
    if reason:
        print(f"  Verdict: {label} — {reason}")
    else:
        print(f"  Verdict: {label}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a PHC physics refinement run."
    )
    parser.add_argument(
        "target",
        type=Path,
        help="Output dir, person dir, or physics dir to verify.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore any existing metrics.json and re-evaluate from NPZs.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override FPS for metrics (default: read from input NPZ).",
    )
    args = parser.parse_args(argv)

    try:
        input_npz, refined_npz = _find_npz_pair(args.target)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Verifying physics run:")
    print(f"  input   : {input_npz}")
    print(f"  refined : {refined_npz}")

    metrics_json = input_npz.parent / "metrics.json"
    if metrics_json.is_file() and not args.recompute:
        print(f"  metrics : {metrics_json} (loaded from disk)")
        with open(metrics_json) as fh:
            payload = json.load(fh)
        print()
        _print_metrics({k: v for k, v in payload.items() if k not in ("status", "reason", "thresholds")})
        _print_verdict(payload.get("status", "?"), payload.get("reason", ""))
        return 0 if payload.get("status") == "ok" else 1

    # Compute fresh
    raw = _load_phc_npz(input_npz)
    ref = _load_phc_npz(refined_npz)

    print()
    _print_summary("input  (raw)", raw)
    _print_summary("refined     ", ref)
    print()
    _print_diff(raw, ref)
    print()

    fps = args.fps if args.fps is not None else raw.get("fps", 30.0)
    # The refined dict needs the SMPL-X-style fields evaluate_refinement
    # expects; copy hand placeholders so FK doesn't choke.
    n_ref = ref["num_frames"]
    ref_for_eval = dict(ref)
    ref_for_eval.setdefault("left_hand_pose", np.zeros((n_ref, 15, 3)))
    ref_for_eval.setdefault("right_hand_pose", np.zeros((n_ref, 15, 3)))
    n_raw = raw["num_frames"]
    raw_for_eval = dict(raw)
    raw_for_eval.setdefault("left_hand_pose", np.zeros((n_raw, 15, 3)))
    raw_for_eval.setdefault("right_hand_pose", np.zeros((n_raw, 15, 3)))

    try:
        from workers.physics.evaluate import (
            compute_verdict,
            evaluate_refinement,
        )
    except ImportError as exc:
        print(f"ERROR: cannot import evaluate module: {exc}", file=sys.stderr)
        print(
            "(run from the bodypipe project root: "
            "`python -m tools.verify_physics ...`)",
            file=sys.stderr,
        )
        return 3

    metrics = evaluate_refinement(raw_for_eval, ref_for_eval, fps=fps)
    _print_metrics(metrics)
    status, reason = compute_verdict(metrics)
    _print_verdict(status, reason)

    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
