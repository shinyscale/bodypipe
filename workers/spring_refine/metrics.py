"""Before/after foot-skating metrics for spring-refine v2.

Writes ``spring_refine/metrics.json`` alongside the pipeline output so
runs are measurable without re-running the clip. Reuses the physics
pipeline's ``compute_foot_skating`` + ``detect_foot_contacts`` so the
numbers are directly comparable with the (dormant) physics path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from workers.physics.evaluate import detect_foot_contacts, _to_jsonable

from .fk import forward_kinematics_body

logger = logging.getLogger(__name__)

# Foot skating delta threshold (meters) for verdict classification.
# Metric is mean per-frame horizontal toe displacement during contact —
# a 3 mm per-frame drop corresponds to noticeable de-sliding over a
# typical stance episode, so 3 mm is the inflection point between
# "neutral" and "improved" rather than the 1 cm the plan initially cited
# (that figure assumed the pre-existing evaluate.py metric, which is
# erroneously m*s and not comparable to a true displacement threshold).
_VERDICT_THRESHOLD_M = 0.003


def _toe_positions(params: dict) -> np.ndarray:
    """Return ``(N, 2, 3)`` toe (L_Foot, R_Foot) world positions."""
    joints = forward_kinematics_body(
        np.asarray(params["body_pose"]),
        np.asarray(params["global_orient"]),
        np.asarray(params["transl"]),
    )
    return joints[:, [10, 11], :]


def _ankle_positions(params: dict) -> np.ndarray:
    joints = forward_kinematics_body(
        np.asarray(params["body_pose"]),
        np.asarray(params["global_orient"]),
        np.asarray(params["transl"]),
    )
    return joints[:, [7, 8], :]


def compute_foot_skating_meters(params: dict, fps: float) -> float:
    """Mean per-frame horizontal toe displacement during foot contact, in m.

    Unlike ``workers.physics.evaluate.compute_foot_skating`` this does
    not multiply the frame-to-frame position delta by ``dt``, so the
    result is actually in meters. Both metrics monotone together — this
    version just has physically meaningful units for the verdict
    threshold comparison.
    """
    ankles = _ankle_positions(params)
    toes = _toe_positions(params)
    contacts = detect_foot_contacts(ankles, fps)
    if toes.shape[0] < 2:
        return 0.0
    delta = np.diff(toes, axis=0)  # (N-1, 2, 3)
    horiz = np.linalg.norm(delta[:, :, [0, 2]], axis=-1)  # (N-1, 2)
    mask = contacts[:-1]
    total_contact = mask.sum()
    if total_contact < 1:
        return 0.0
    return float((horiz * mask).sum() / total_contact)


def classify_verdict(delta_m: float) -> str:
    """Classify a skating delta against the 1 cm threshold."""
    if delta_m > _VERDICT_THRESHOLD_M:
        return "improved"
    if delta_m < -_VERDICT_THRESHOLD_M:
        return "worse"
    return "neutral"


def write_spring_metrics_json(
    baseline_params: dict,
    refined_params: dict,
    fps: float,
    preset: str,
    pin_enabled: bool,
    out_path: Path,
    pin_stats: dict | None = None,
) -> dict:
    """Compute before/after foot skating and write metrics JSON.

    Parameters
    ----------
    baseline_params : dict — pre-refine params (baseline world motion).
    refined_params : dict — post-refine params (possibly pinned).
    fps : float
    preset : spring preset string, recorded verbatim.
    pin_enabled : bool
    out_path : Path to write the JSON file (parent dirs created).
    pin_stats : optional dict of stance/pin statistics from
        ``footpin.apply_foot_pin``.

    Returns
    -------
    dict : the payload that was written (also includes derived fields
    like ``verdict`` and ``foot_skating_delta_m``).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        baseline_skating = compute_foot_skating_meters(baseline_params, fps)
    except Exception as exc:
        logger.warning("baseline foot skating failed: %s", exc)
        baseline_skating = float("nan")

    try:
        refined_skating = compute_foot_skating_meters(refined_params, fps)
    except Exception as exc:
        logger.warning("refined foot skating failed: %s", exc)
        refined_skating = float("nan")

    if np.isnan(baseline_skating) or np.isnan(refined_skating):
        delta = float("nan")
        verdict = "unknown"
    else:
        delta = baseline_skating - refined_skating
        verdict = classify_verdict(delta)

    payload: dict = {
        "num_frames": int(refined_params.get("num_frames", 0)),
        "fps": float(fps),
        "preset": preset,
        "pin_enabled": bool(pin_enabled),
        "foot_skating_baseline_m": baseline_skating,
        "foot_skating_refined_m": refined_skating,
        "foot_skating_delta_m": delta,
        "verdict": verdict,
        "verdict_threshold_m": _VERDICT_THRESHOLD_M,
    }
    if pin_stats is not None:
        payload["pin_stats"] = pin_stats

    serialised = _to_jsonable(payload)
    with open(out_path, "w") as fh:
        json.dump(serialised, fh, indent=2)
    return payload
