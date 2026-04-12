"""Top-level spring refinement entry points for the pipeline orchestrator.

``run_spring_refine`` mirrors the (refined_params, ok) contract used by
``workers.physics.phc_runner.run_phc_local`` so the orchestrator can
swap it in at the same stage 4 hook point.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .filter import critically_damped_filter
from .gains import get_preset_gains
from .rotations import aa_sequence_to_log, log_sequence_to_aa


def refine_body_sequence(
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    fps: float,
    preset: str = "moderate",
    pad_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter body_pose + global_orient per joint in quaternion log space.

    Args:
        body_pose: (N, 63) or (N, 21, 3) axis-angle body pose.
        global_orient: (N, 3) or (N, 1, 3) axis-angle root orientation.
        fps: frames per second (drives the filter timestep).
        preset: "light" | "moderate" | "heavy".
        pad_frames: replicate-pad length at each end of the sequence to
            mitigate the warm-up transient. Defaults to ``int(fps / 2)``.

    Returns:
        ``(refined_body_pose, refined_global_orient)`` with shapes
        matching the inputs (``(N, 63)`` vs ``(N, 21, 3)`` preserved).
    """
    body_pose_in = np.asarray(body_pose, dtype=np.float32)
    go_in = np.asarray(global_orient, dtype=np.float32)

    if body_pose_in.ndim == 2:
        if body_pose_in.shape[1] != 63:
            raise ValueError(
                f"2D body_pose must have 63 columns; got {body_pose_in.shape}"
            )
        body_pose_jnt = body_pose_in.reshape(-1, 21, 3)
        body_pose_flat = True
    elif body_pose_in.ndim == 3 and body_pose_in.shape[1:] == (21, 3):
        body_pose_jnt = body_pose_in
        body_pose_flat = False
    else:
        raise ValueError(
            f"body_pose must be (N,63) or (N,21,3); got {body_pose_in.shape}"
        )

    N = body_pose_jnt.shape[0]
    root_in_shape = go_in.shape
    go_reshaped = go_in.reshape(N, 1, 3)

    if fps <= 0:
        raise ValueError(f"fps must be positive; got {fps}")
    dt = 1.0 / float(fps)
    if pad_frames is None:
        pad_frames = max(1, int(fps / 2))

    body_kp, body_kv, root_kp, root_kv = get_preset_gains(preset)

    # Body joints
    body_log = aa_sequence_to_log(body_pose_jnt)
    body_log_filt = critically_damped_filter(
        body_log, body_kp, body_kv, dt, zero_phase=True, pad_frames=pad_frames
    )
    body_pose_filt = log_sequence_to_aa(body_log_filt)

    # Root (single-joint case)
    root_log = aa_sequence_to_log(go_reshaped)
    root_log_filt = critically_damped_filter(
        root_log,
        np.array([root_kp], dtype=np.float32),
        np.array([root_kv], dtype=np.float32),
        dt,
        zero_phase=True,
        pad_frames=pad_frames,
    )
    go_filt = log_sequence_to_aa(root_log_filt).reshape(root_in_shape)

    if body_pose_flat:
        body_pose_filt = body_pose_filt.reshape(N, 63)

    return body_pose_filt.astype(np.float32), go_filt.astype(np.float32)


def run_spring_refine(
    world_params: dict,
    preset: str = "moderate",
    fps: float = 30.0,
    progress_cb: Callable[[float], None] | None = None,
    pin_feet: bool = False,
    filter_rotations: bool = True,
    foot_pin_sensitivity: str = "medium",
    foot_pin_strength: float = 1.0,
) -> tuple[dict, bool]:
    """Orchestrator-facing entry. Returns ``(refined_params, ok)``.

    On any missing required key, returns ``(world_params, False)`` —
    matching the contract used by ``workers.physics.phc_runner.run_phc_local``.

    ``filter_rotations`` (v1): run the per-joint critically-damped IIR on
    ``body_pose`` + ``global_orient``. When ``False``, the rotation
    channels pass through unchanged — useful when the caller only wants
    the foot-pin pass.

    ``pin_feet`` (v2): after the rotation filter (if any), run the
    contact-aware foot-pin pass. This modifies ``transl`` to anchor the
    feet during stance episodes, fixing foot slide under camera moves.
    Pin failure is non-fatal — we log into ``foot_pin_stats.error`` and
    return the rotation-refined result with ``source="spring_refined"``.
    """
    try:
        body_pose = np.asarray(world_params["body_pose"])
        global_orient = np.asarray(world_params["global_orient"])
    except (KeyError, TypeError):
        return world_params, False

    # If the caller disabled both passes there is nothing to do — return
    # a failure so the orchestrator can label the snapshot as baseline.
    if not filter_rotations and not pin_feet:
        return world_params, False

    if progress_cb is not None:
        progress_cb(0.1)

    if filter_rotations:
        try:
            body_pose_filt, go_filt = refine_body_sequence(
                body_pose, global_orient, fps=fps, preset=preset
            )
        except Exception:
            return world_params, False
    else:
        body_pose_filt = np.asarray(body_pose, dtype=np.float32)
        go_filt = np.asarray(global_orient, dtype=np.float32)

    refined = dict(world_params)
    refined["body_pose"] = body_pose_filt
    refined["global_orient"] = go_filt
    refined["num_frames"] = int(body_pose_filt.reshape(-1, 63).shape[0])
    # Source tag reflects what we've actually done so far. Overwritten
    # below if the foot-pin pass also runs successfully.
    refined["source"] = "spring_refined"

    pin_ok = False
    if pin_feet:
        if progress_cb is not None:
            progress_cb(0.6)
        try:
            from .footpin import apply_foot_pin

            refined, pin_stats = apply_foot_pin(
                refined,
                fps=float(fps),
                preset=preset,
                sensitivity=foot_pin_sensitivity,
                pin_strength=foot_pin_strength,
            )
            refined["foot_pin_stats"] = pin_stats
            refined["source"] = "spring_refined_pinned"
            pin_ok = True
        except Exception as exc:  # pragma: no cover - defensive
            refined["foot_pin_stats"] = {"error": repr(exc)}
            # Pin failure is non-fatal — keep the rotation-refined result.

    if progress_cb is not None:
        progress_cb(1.0)

    # ok == True when we produced *something* different from the raw
    # baseline (either filter ran, or pin ran, or both).
    return refined, bool(filter_rotations or pin_ok)
