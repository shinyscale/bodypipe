"""Pelvis-translation quality metrics for bodypipe outputs.

Measures world-grounding quality of GVHMR pelvis translation using
proxy metrics (no ground-truth required) and optional reference
comparison (if ground-truth trajectory is provided).

Proxy metrics (always computed):
  - pelvis_drift_xz_total_cm: total horizontal displacement over sequence
  - pelvis_drift_xz_rate_cm_s: horizontal drift per second
  - pelvis_height_std_cm: std of pelvis Y during stance (should be low)
  - pelvis_jitter_cm: mean acceleration magnitude (high-freq noise)
  - pelvis_ldlj: log dimensionless jerk on pelvis trajectory
  - pelvis_stance_drift_cm: mean XZ displacement during foot contact

Reference metrics (when ground-truth supplied):
  - pelvis_mae_cm: mean absolute error vs reference trajectory
  - pelvis_mae_xz_cm: horizontal-only MAE
  - pelvis_mae_y_cm: vertical-only MAE

Clinical bar (from OpenCap-monocular): 3.4 cm pelvis MAE.

Usage:
  python -m tools.eval_pelvis_mae <pt_file> [--reference <ref.npy>] [--fps 30]
  python -m tools.eval_pelvis_mae <output_dir> --all-stages
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# OpenCap-monocular clinical bar (cm)
OPENCAP_PELVIS_MAE_CM = 3.4

# Joint indices (SMPL-X body ordering)
_L_ANKLE, _R_ANKLE = 7, 8


def extract_pelvis_transl(params: dict) -> np.ndarray:
    """Extract ``(N, 3)`` pelvis translation from a params dict.

    Tries keys in priority order: ``transl_world`` > ``transl``.
    """
    for key in ("transl_world", "transl"):
        if key in params:
            return np.asarray(params[key], dtype=np.float64)
    raise KeyError("No pelvis translation found (need 'transl' or 'transl_world')")


def compute_proxy_metrics(
    transl: np.ndarray,
    fps: float = 30.0,
    ankle_positions: np.ndarray | None = None,
) -> dict:
    """Compute proxy pelvis-quality metrics (no ground truth needed).

    Parameters
    ----------
    transl : ``(N, 3)`` pelvis world-space translation. Y-up convention.
    fps : Frame rate.
    ankle_positions : ``(N, 2, 3)`` L/R ankle world positions (optional,
        enables stance-phase metrics). If None, stance metrics are skipped.

    Returns
    -------
    dict : Metric name → value.
    """
    N = transl.shape[0]
    dt = 1.0 / fps
    duration_s = (N - 1) * dt
    metrics: dict = {"num_frames": N, "fps": fps, "duration_s": round(duration_s, 3)}

    # --- Total XZ drift ---
    xz = transl[:, [0, 2]]
    xz_displacements = np.linalg.norm(np.diff(xz, axis=0), axis=-1)
    total_path_xz = float(xz_displacements.sum())
    endpoint_drift_xz = float(np.linalg.norm(xz[-1] - xz[0]))

    metrics["pelvis_path_xz_cm"] = round(total_path_xz * 100, 2)
    metrics["pelvis_drift_xz_total_cm"] = round(endpoint_drift_xz * 100, 2)
    if duration_s > 0:
        metrics["pelvis_drift_xz_rate_cm_s"] = round(
            endpoint_drift_xz * 100 / duration_s, 2
        )
    else:
        metrics["pelvis_drift_xz_rate_cm_s"] = 0.0

    # --- Height consistency (Y std) ---
    metrics["pelvis_height_mean_cm"] = round(float(transl[:, 1].mean()) * 100, 2)
    metrics["pelvis_height_std_cm"] = round(float(transl[:, 1].std()) * 100, 2)

    # --- Jitter (mean acceleration magnitude) ---
    if N >= 3:
        vel = np.diff(transl, axis=0) / dt
        acc = np.diff(vel, axis=0) / dt
        jitter = float(np.linalg.norm(acc, axis=-1).mean())
        metrics["pelvis_jitter_m_s2"] = round(jitter, 3)
        metrics["pelvis_jitter_cm"] = round(
            float(np.linalg.norm(np.diff(transl, n=2, axis=0), axis=-1).mean()) * 100,
            3,
        )
    else:
        metrics["pelvis_jitter_m_s2"] = 0.0
        metrics["pelvis_jitter_cm"] = 0.0

    # --- LDLJ on pelvis ---
    if N >= 4:
        vel = np.diff(transl, axis=0) / dt
        acc = np.diff(vel, axis=0) / dt
        jerk = np.diff(acc, axis=0) / dt
        displacement = max(float(np.linalg.norm(transl[-1] - transl[0])), 1e-6)
        jerk_sq_sum = float(np.sum(jerk**2)) * dt
        ldlj = -np.log(duration_s**5 / displacement**2 * jerk_sq_sum + 1e-10)
        metrics["pelvis_ldlj"] = round(float(ldlj), 3)
    else:
        metrics["pelvis_ldlj"] = 0.0

    # --- Stance-phase drift (requires ankle positions) ---
    if ankle_positions is not None and N >= 2:
        from workers.physics.evaluate import detect_foot_contacts

        contacts = detect_foot_contacts(ankle_positions, fps)  # (N, 2)
        # Either foot in contact → stance frame
        stance_mask = contacts.any(axis=1)  # (N,)
        stance_count = int(stance_mask.sum())
        metrics["stance_frames"] = stance_count
        metrics["stance_fraction"] = round(stance_count / N, 3)

        if stance_count >= 2:
            # XZ displacement during stance
            stance_xz = xz.copy()
            stance_delta = np.zeros(N - 1)
            for i in range(N - 1):
                if stance_mask[i] and stance_mask[i + 1]:
                    stance_delta[i] = np.linalg.norm(stance_xz[i + 1] - stance_xz[i])
            n_stance_trans = max(int((stance_mask[:-1] & stance_mask[1:]).sum()), 1)
            metrics["pelvis_stance_drift_cm"] = round(
                float(stance_delta.sum() / n_stance_trans) * 100, 3
            )
        else:
            metrics["pelvis_stance_drift_cm"] = None
    else:
        metrics["stance_frames"] = None
        metrics["pelvis_stance_drift_cm"] = None

    return metrics


def compute_reference_metrics(
    transl: np.ndarray,
    reference: np.ndarray,
) -> dict:
    """Compute MAE metrics against a ground-truth reference trajectory.

    Parameters
    ----------
    transl : ``(N, 3)`` estimated pelvis translation.
    reference : ``(M, 3)`` ground-truth pelvis translation. Truncated/padded
        to match ``N``.

    Returns
    -------
    dict : Reference metric name → value.
    """
    N = transl.shape[0]
    M = reference.shape[0]
    n = min(N, M)
    est = transl[:n]
    ref = reference[:n]

    diff = est - ref
    mae_3d = float(np.linalg.norm(diff, axis=-1).mean())
    mae_xz = float(np.linalg.norm(diff[:, [0, 2]], axis=-1).mean())
    mae_y = float(np.abs(diff[:, 1]).mean())

    return {
        "pelvis_mae_cm": round(mae_3d * 100, 2),
        "pelvis_mae_xz_cm": round(mae_xz * 100, 2),
        "pelvis_mae_y_cm": round(mae_y * 100, 2),
        "reference_frames_used": n,
        "meets_opencap_bar": mae_3d * 100 <= OPENCAP_PELVIS_MAE_CM,
    }


def load_params(path: Path) -> dict:
    """Load a ``.pt`` file and return it as a dict with numpy arrays."""
    import torch

    data = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data)}")
    return {
        k: v.numpy() if hasattr(v, "numpy") else v
        for k, v in data.items()
    }


def _get_ankle_positions(params: dict) -> np.ndarray | None:
    """Try to compute ankle positions via FK. Returns None on failure."""
    try:
        from workers.spring_refine.fk import forward_kinematics_body

        joints = forward_kinematics_body(
            np.asarray(params["body_pose"]),
            np.asarray(params["global_orient"]),
            np.asarray(params["transl"]),
        )
        return joints[:, [_L_ANKLE, _R_ANKLE], :]
    except Exception:
        return None


def evaluate_pt_file(
    pt_path: Path,
    fps: float = 30.0,
    reference: np.ndarray | None = None,
    stage_name: str | None = None,
) -> dict:
    """Full evaluation of a single .pt file.

    Parameters
    ----------
    pt_path : Path to a ``hmr4d_results.pt`` or ``*_hybrid_smplx.pt``.
    fps : Frame rate.
    reference : Optional ``(N, 3)`` ground-truth pelvis trajectory.
    stage_name : Label for this stage (e.g. "baseline", "spring_refined").

    Returns
    -------
    dict : All computed metrics.
    """
    params = load_params(pt_path)
    transl = extract_pelvis_transl(params)
    ankles = _get_ankle_positions(params)

    result = {"file": str(pt_path), "stage": stage_name or pt_path.stem}
    result.update(compute_proxy_metrics(transl, fps, ankles))

    if reference is not None:
        result.update(compute_reference_metrics(transl, reference))

    result["opencap_bar_cm"] = OPENCAP_PELVIS_MAE_CM
    return result


def find_stage_files(output_dir: Path) -> list[tuple[str, Path]]:
    """Discover all evaluable .pt files in a bodypipe output directory.

    Returns list of (stage_name, path) tuples in pipeline order.
    """
    stages: list[tuple[str, Path]] = []

    # GVHMR raw output
    hmr4d = output_dir / "demo" / "isolated_video" / "hmr4d_results.pt"
    if hmr4d.is_file():
        stages.append(("gvhmr_raw", hmr4d))

    # Hybrid SMPLX files (may contain multiple motion sources)
    for hybrid in sorted(output_dir.glob("*_hybrid_smplx.pt")):
        stages.append(("hybrid_" + hybrid.stem.replace("_hybrid_smplx", ""), hybrid))

    # Spring refine output
    spring = output_dir / "spring_refine"
    if spring.is_dir():
        for pt in sorted(spring.glob("*.pt")):
            stages.append(("spring_" + pt.stem, pt))

    return stages


def evaluate_all_stages(
    output_dir: Path,
    fps: float = 30.0,
    reference: np.ndarray | None = None,
) -> list[dict]:
    """Evaluate all pipeline stages found in an output directory."""
    stages = find_stage_files(output_dir)
    if not stages:
        logger.warning("No .pt files found in %s", output_dir)
        return []

    results = []
    for stage_name, pt_path in stages:
        try:
            result = evaluate_pt_file(pt_path, fps, reference, stage_name)
            results.append(result)
        except Exception as exc:
            logger.warning("Failed to evaluate %s (%s): %s", stage_name, pt_path, exc)
            results.append({"stage": stage_name, "file": str(pt_path), "error": str(exc)})

    return results


def _extract_hybrid_stages(params: dict, fps: float) -> list[dict]:
    """Extract and evaluate individual motion sources from a hybrid file.

    A hybrid_smplx.pt may contain baseline, spring, and pinned variants
    stored under separate key prefixes.
    """
    stage_prefixes = [
        ("baseline", "world_baseline"),
        ("spring_filtered", "world_spring"),
        # pinned is the default transl/global_orient if source == "spring_refined_pinned"
    ]
    results = []
    for label, suffix in stage_prefixes:
        go_key = f"global_orient_{suffix}"
        bp_key = f"body_pose_{suffix}"
        tr_key = f"transl_{suffix}"
        if go_key in params and tr_key in params:
            sub = {
                "global_orient": params[go_key],
                "body_pose": params.get(bp_key, params.get("body_pose")),
                "transl": params[tr_key],
            }
            transl = np.asarray(sub["transl"], dtype=np.float64)
            ankles = _get_ankle_positions(sub)
            result = {"stage": label}
            result.update(compute_proxy_metrics(transl, fps, ankles))
            results.append(result)
    return results


def format_report(results: list[dict]) -> str:
    """Format evaluation results as a human-readable table."""
    if not results:
        return "No results to report."

    lines = []
    lines.append("=" * 78)
    lines.append("PELVIS TRANSLATION QUALITY REPORT")
    lines.append(f"OpenCap-monocular clinical bar: {OPENCAP_PELVIS_MAE_CM} cm MAE")
    lines.append("=" * 78)

    for r in results:
        if "error" in r:
            lines.append(f"\n  {r['stage']:30s}  ERROR: {r['error']}")
            continue

        lines.append(f"\n  Stage: {r.get('stage', '?')}")
        lines.append(f"  File:  {r.get('file', '?')}")
        lines.append(f"  Frames: {r.get('num_frames', '?')}  Duration: {r.get('duration_s', '?')}s")
        lines.append("")

        # Proxy metrics
        lines.append("  Proxy metrics (no ground truth):")
        lines.append(f"    XZ endpoint drift:     {r.get('pelvis_drift_xz_total_cm', '?'):>8} cm")
        lines.append(f"    XZ drift rate:         {r.get('pelvis_drift_xz_rate_cm_s', '?'):>8} cm/s")
        lines.append(f"    XZ total path:         {r.get('pelvis_path_xz_cm', '?'):>8} cm")
        lines.append(f"    Height std:            {r.get('pelvis_height_std_cm', '?'):>8} cm")
        lines.append(f"    Jitter (accel):        {r.get('pelvis_jitter_m_s2', '?'):>8} m/s²")
        lines.append(f"    LDLJ:                  {r.get('pelvis_ldlj', '?'):>8}")

        stance_drift = r.get("pelvis_stance_drift_cm")
        if stance_drift is not None:
            stance_frac = r.get("stance_fraction", "?")
            lines.append(f"    Stance XZ drift:       {stance_drift:>8} cm/frame  ({stance_frac} stance)")

        # Reference metrics
        mae = r.get("pelvis_mae_cm")
        if mae is not None:
            lines.append("")
            lines.append("  Reference metrics (vs ground truth):")
            lines.append(f"    3D MAE:                {mae:>8} cm")
            lines.append(f"    XZ MAE:                {r.get('pelvis_mae_xz_cm', '?'):>8} cm")
            lines.append(f"    Y MAE:                 {r.get('pelvis_mae_y_cm', '?'):>8} cm")
            bar = "PASS" if r.get("meets_opencap_bar") else "FAIL"
            lines.append(f"    OpenCap bar (≤3.4cm):  {bar}")

    lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


def _to_jsonable(obj):
    """Recursively convert numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(
        description="Pelvis-translation quality evaluation for bodypipe outputs."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a .pt file or output directory (with --all-stages).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Path to ground-truth pelvis trajectory (.npy, shape (N,3)).",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Frame rate.")
    parser.add_argument(
        "--all-stages",
        action="store_true",
        help="Evaluate all pipeline stages in an output directory.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write results as JSON to this path.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress human-readable report."
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)

    ref = None
    if args.reference is not None:
        ref = np.load(str(args.reference)).astype(np.float64)
        if ref.ndim != 2 or ref.shape[1] != 3:
            print(f"ERROR: reference must be (N, 3), got {ref.shape}", file=sys.stderr)
            sys.exit(1)

    if args.all_stages:
        results = evaluate_all_stages(args.path, args.fps, ref)
    else:
        results = [evaluate_pt_file(args.path, args.fps, ref)]

    if not args.quiet:
        print(format_report(results))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(_to_jsonable(results), f, indent=2)
        print(f"Wrote {args.json}")


if __name__ == "__main__":
    main()
