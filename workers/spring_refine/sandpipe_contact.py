"""Sandpipe-derived foot contact detection for spring-refine.

Replaces the kinematic contact detection in contact.py with physics
signals extracted from sandpipe's GPU sand simulation. The sand sim
reveals weight distribution and grounding from 2D video, independently
of 3D pose estimation quality — breaking the circular dependency where
contact is detected from already-wrong joint positions.

Architecture:
    Sandpipe answers "IS the body grounded?" (video-derived, not circular).
    Kinematics answer "WHICH foot?" (relative ankle comparison, robust even
    when absolute height is wrong).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fk import forward_kinematics_body

_L_ANKLE = 7
_R_ANKLE = 8
_L_FOOT = 10
_R_FOOT = 11


def load_sandpipe_physics(json_path: Path | str) -> dict:
    """Load sandpipe physics JSON exported from sandpipe-body-webgpu-v2."""
    with open(json_path) as f:
        return json.load(f)


def _get_person_frame(frame_data: dict | None, person_idx: int = 0) -> dict | None:
    """Extract per-person data from a frame dict.

    For multi-person exports, ``frame_data["persons"][person_idx]`` holds
    per-person stats. For single-person (or missing persons key), the
    top-level frame dict is returned directly (backward compatible).
    """
    if frame_data is None:
        return None
    persons = frame_data.get("persons")
    if persons and 0 <= person_idx < len(persons):
        return persons[person_idx]
    # Single-person fallback: top-level fields
    return frame_data


def compute_grounding_confidence(frame_data: dict | None) -> float:
    """Compute per-frame grounding confidence from sandpipe signals.

    Returns 0.0 (airborne / no data) to 1.0 (firmly grounded).
    """
    if frame_data is None:
        return 0.0

    total = frame_data.get("insideTotal", 0)
    if total == 0:
        return 0.0

    # Weight center: 1.0 = sand settled at feet, 0.0 = suspended at top
    wcy = frame_data.get("weightCenterY", 0.5)

    # Bottom quarter ratio: high = weight concentrated at feet
    bottom_ratio = frame_data.get("insideBottomQuarter", 0) / total

    # Contact intensity: outside sand piling at the base of the body
    outside_total = max(frame_data.get("outsideTotal", 0), 1)
    contact_ratio = min(1.0, frame_data.get("contactBottom", 0) / (outside_total * 0.1))

    # Settled flag: weight center hasn't moved significantly between frames
    settled = 1.0 if frame_data.get("isSettled", False) else 0.0

    confidence = (
        0.35 * wcy
        + 0.30 * bottom_ratio
        + 0.20 * contact_ratio
        + 0.15 * settled
    )
    return float(np.clip(confidence, 0.0, 1.0))


def _map_sandpipe_frames(
    sandpipe_data: dict, n_frames: int, fps: float, person_idx: int = 0,
) -> list:
    """Map bodypipe frame indices to sandpipe frame dicts (or None).

    When ``person_idx`` is given and the export contains per-person data,
    the returned dicts are the person-specific slices (with the same
    field names as single-person: loadBalance, weightCenterY, etc.).
    """
    frames = sandpipe_data.get("frames", [])
    sp_fps = sandpipe_data.get("fps", fps)
    sp_total = len(frames)
    remap = abs(sp_fps - fps) > 0.1

    result = []
    for t in range(n_frames):
        sp_idx = int(round(t * sp_fps / fps)) if remap else t
        sp_idx = min(sp_idx, sp_total - 1)
        raw = frames[sp_idx] if 0 <= sp_idx < sp_total else None
        result.append(_get_person_frame(raw, person_idx))
    return result


def compute_grounding_curve(
    sandpipe_data: dict,
    n_frames: int,
    fps: float,
    smooth_sigma: float = 3.0,
    person_idx: int = 0,
) -> np.ndarray:
    """Map sandpipe frames to bodypipe frames and return a smoothed grounding curve.

    Parameters
    ----------
    person_idx : which sandpipe person to read stats for (0-based).
        For single-person exports this is ignored.

    Returns
    -------
    curve : ``(n_frames,)`` float64 in ``[0, 1]``.
        Per-frame grounding confidence, Gaussian-smoothed.
    """
    from scipy.ndimage import gaussian_filter1d

    mapped = _map_sandpipe_frames(sandpipe_data, n_frames, fps, person_idx)

    # 1. Base grounding from inside sand (instantaneous)
    raw = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        raw[t] = compute_grounding_confidence(mapped[t])

    # 2. Outside accumulation rate: derivative of contactBottom.
    # Rising pile = sustained grounding. Falling/flat = not accumulating.
    contact_bottom = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        fd = mapped[t]
        contact_bottom[t] = fd.get("contactBottom", 0) if fd else 0.0

    # Smooth the raw contact signal before differentiating (noisy atomics)
    if n_frames > 3:
        contact_bottom_smooth = gaussian_filter1d(contact_bottom, sigma=2.0, mode="nearest")
    else:
        contact_bottom_smooth = contact_bottom

    accum_rate = np.zeros(n_frames, dtype=np.float64)
    accum_rate[1:] = np.diff(contact_bottom_smooth)
    accum_rate = np.clip(accum_rate, 0.0, None)  # only care about growth

    # Normalize to [0, 1] — peak accumulation rate varies by sim settings
    accum_max = accum_rate.max()
    if accum_max > 0:
        accum_signal = np.clip(accum_rate / accum_max, 0.0, 1.0)
    else:
        accum_signal = accum_rate

    # 3. Body velocity dampening: fast mask movement = likely airborne/dynamic.
    # Velocity is in grid-cell units per frame.
    vel_mag = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        fd = mapped[t]
        if fd:
            vx = fd.get("bodyVelX", 0.0)
            vy = fd.get("bodyVelY", 0.0)
            vel_mag[t] = (vx ** 2 + vy ** 2) ** 0.5

    # Map velocity to a dampening factor: 0 velocity = 1.0 (no dampening),
    # high velocity = 0.0 (full dampening). Threshold at ~5 cells/frame.
    vel_threshold = 5.0
    vel_damp = np.clip(1.0 - vel_mag / vel_threshold, 0.0, 1.0)

    # 4. Contact L/R stability: outside sand piles asymmetrically when the
    # body moves laterally. A stable L/R ratio = planted. Rapid shift in
    # the ratio = lateral step or weight transfer (the "wake" of accumulated
    # sand on the side she came FROM, runoff on the side she moved TO).
    contact_l = np.zeros(n_frames, dtype=np.float64)
    contact_r = np.zeros(n_frames, dtype=np.float64)
    for t in range(n_frames):
        fd = mapped[t]
        if fd:
            contact_l[t] = fd.get("contactLeft", 0.0)
            contact_r[t] = fd.get("contactRight", 0.0)

    # L/R ratio: 0.5 = symmetric, deviations = asymmetric pile
    lr_total = contact_l + contact_r
    lr_ratio = np.where(lr_total > 0, contact_l / lr_total, 0.5)

    # Smooth before differentiating
    if n_frames > 3:
        lr_ratio_smooth = gaussian_filter1d(lr_ratio, sigma=2.0, mode="nearest")
    else:
        lr_ratio_smooth = lr_ratio

    # Rate of change in L/R ratio — high = lateral movement in progress
    lr_shift = np.zeros(n_frames, dtype=np.float64)
    lr_shift[1:] = np.abs(np.diff(lr_ratio_smooth))

    # Normalize: peak shift varies by video, so scale relative to observed max
    lr_shift_max = lr_shift.max()
    if lr_shift_max > 0:
        lr_shift_norm = np.clip(lr_shift / lr_shift_max, 0.0, 1.0)
    else:
        lr_shift_norm = lr_shift

    # Convert to stability: 1.0 = stable (no lateral movement), 0.0 = mid-step
    lr_stability = 1.0 - lr_shift_norm

    # Combine: base confidence + accumulation boost, dampened by velocity
    # and lateral stability
    combined = (0.70 * raw + 0.20 * accum_signal + 0.10 * lr_stability) * vel_damp

    if smooth_sigma > 0 and n_frames > 1:
        combined = gaussian_filter1d(combined, sigma=smooth_sigma, mode="nearest")

    return np.clip(combined, 0.0, 1.0)


def estimate_ground_level_sandpipe(
    sandpipe_data: dict,
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
    fps: float,
    min_grounded_frames: int = 10,
    grounding_threshold: float = 0.6,
    person_idx: int = 0,
) -> float | None:
    """Estimate ground-plane Y from ankle heights at sandpipe-confirmed grounded frames.

    More reliable than the kinematic percentile approach because it only
    measures ankle Y when the body is genuinely grounded (video-derived),
    rather than trusting already-wrong 3D joint positions.

    Returns
    -------
    ground_y : float or None — median ankle Y at grounded frames,
        or None if fewer than ``min_grounded_frames`` qualify.
    """
    joints = forward_kinematics_body(body_pose, global_orient, transl)
    ankles_y = joints[:, [_L_ANKLE, _R_ANKLE], 1]  # (N, 2)
    N = int(ankles_y.shape[0])

    gc = compute_grounding_curve(sandpipe_data, N, fps, smooth_sigma=1.0, person_idx=person_idx)
    grounded_mask = gc >= grounding_threshold

    if grounded_mask.sum() < min_grounded_frames:
        return None

    # Min ankle Y per frame (whichever foot is lower)
    min_ankle_y = ankles_y[grounded_mask].min(axis=1)
    return float(np.median(min_ankle_y))


def detect_contacts_sandpipe(
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
    fps: float,
    sandpipe_data: dict,
    grounding_threshold: float = 0.4,
    velocity_threshold: float = 0.8,
    person_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect foot contacts using sandpipe physics + kinematic hints.

    Sandpipe provides grounding confidence (is the body planted?).
    Kinematics provide L/R disambiguation (which foot is the support foot?).

    Parameters
    ----------
    body_pose, global_orient, transl : SMPL-X / SOMA body params.
    fps : frame rate.
    sandpipe_data : dict loaded from sandpipe-physics.json.
    grounding_threshold : minimum grounding confidence for contact.
    velocity_threshold : ankle speed (m/s) below which foot is still.

    Returns
    -------
    contacts : ``(N, 2)`` bool — ``[L, R]`` contact mask.
    toes : ``(N, 2, 3)`` float64 — L_Foot / R_Foot world positions.
    """
    joints = forward_kinematics_body(body_pose, global_orient, transl)
    ankles = joints[:, [_L_ANKLE, _R_ANKLE], :]
    toes = joints[:, [_L_FOOT, _R_FOOT], :]
    n = int(ankles.shape[0])
    contacts = np.zeros((n, 2), dtype=bool)

    if n < 2:
        return contacts, toes

    # Ground-level estimation for secondary validation
    ground_y = estimate_ground_level_sandpipe(
        sandpipe_data, body_pose, global_orient, transl, fps,
        person_idx=person_idx,
    )
    # Max ankle distance above ground to still count as grounded (meters)
    ground_tolerance = 0.15

    frames = sandpipe_data.get("frames", [])
    sp_fps = sandpipe_data.get("fps", fps)
    sp_total = len(frames)

    # Per-foot velocities for L/R disambiguation
    dt = 1.0 / float(fps)
    ankle_vel = np.zeros_like(ankles)
    ankle_vel[1:] = (ankles[1:] - ankles[:-1]) / dt
    ankle_vel[0] = ankle_vel[1]
    ankle_speed = np.linalg.norm(ankle_vel, axis=-1)  # (N, 2)

    for t in range(n):
        # Map bodypipe frame index to sandpipe frame index
        sp_idx = int(round(t * sp_fps / fps)) if abs(sp_fps - fps) > 0.1 else t
        sp_idx = min(sp_idx, sp_total - 1)

        raw_frame = frames[sp_idx] if 0 <= sp_idx < sp_total else None
        frame_data = _get_person_frame(raw_frame, person_idx)

        gc = compute_grounding_confidence(frame_data)
        if gc < grounding_threshold:
            continue

        # Ground-level validation: if ankles are far above the estimated
        # ground plane, reduce confidence even when sandpipe says grounded.
        if ground_y is not None:
            min_ankle_y = min(ankles[t, 0, 1], ankles[t, 1, 1])
            above = min_ankle_y - ground_y
            if above > ground_tolerance:
                # Scale gc down proportionally — fully suppressed at 2x tolerance
                gc *= float(np.clip(1.0 - (above - ground_tolerance) / ground_tolerance, 0.0, 1.0))
                if gc < grounding_threshold:
                    continue

        load_bal = frame_data.get("loadBalance", 0.5) if frame_data else 0.5

        l_still = ankle_speed[t, 0] < velocity_threshold
        r_still = ankle_speed[t, 1] < velocity_threshold

        # Relative ankle height: lower = more likely planted.
        # This comparison is robust even when absolute height is wrong.
        l_lower = ankles[t, 0, 1] <= ankles[t, 1, 1]
        r_lower = ankles[t, 1, 1] <= ankles[t, 0, 1]

        # Weight hints from sandpipe (image-space L/R)
        l_weight = 1.0 - load_bal
        r_weight = load_bal

        # Contact: grounded AND (foot is still OR foot carries weight + is lower)
        if l_still or (l_weight > 0.6 and l_lower):
            contacts[t, 0] = True
        if r_still or (r_weight > 0.6 and r_lower):
            contacts[t, 1] = True

        # Strong grounding + balanced weight → both feet likely planted
        if gc > 0.7 and abs(load_bal - 0.5) < 0.15:
            if l_still:
                contacts[t, 0] = True
            if r_still:
                contacts[t, 1] = True

    return contacts, toes
