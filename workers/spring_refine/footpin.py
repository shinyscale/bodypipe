"""Foot-pin correction for spring-refine v2.

Given a refined params dict (post rotation filter), detect per-frame
foot contacts, lock anchor positions at the start of each stance, and
compute a ``(N, 3)`` correction vector applied to root ``transl`` so
that feet stay pinned during stance. Smoothed with a short Gaussian so
stance transitions don't pop.

v2 scope: correct root trajectory only. Analytic leg IK (keeping the
non-stance leg's world pose fixed when only one foot is in contact) is
deferred to v3.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .contact import detect_contacts, find_episodes, track_stance_anchors


# Contact-heuristic sensitivity presets. Dialed from walking-gait
# observation: planted ankles dip/rise within ~7 cm and lift off around
# 1 m/s. "low" is strict (swing-friendly), "high" is lenient (catches
# fast footwork but may pin some swing frames).
_SENSITIVITY_PRESETS: dict[str, dict[str, float]] = {
    "low":    {"height_threshold": 0.04, "velocity_threshold": 0.3},
    "medium": {"height_threshold": 0.07, "velocity_threshold": 0.5},
    "high":   {"height_threshold": 0.12, "velocity_threshold": 1.0},
}


def compute_pin_offset(
    toes: np.ndarray,
    anchors: np.ndarray,
    valid_mask: np.ndarray,
    fps: float,
    smoothing_sigma_sec: float = 0.1,
    taper_sec: float = 0.067,
    per_frame_strength: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a per-frame translation correction that pins feet.

    Per frame:
    - Both feet valid → offset = mean(anchor_L - toe_L, anchor_R - toe_R).
    - One foot valid → offset = that foot's (anchor - toe).
    - Neither valid → interpolated from neighboring valid frames (linear
      on each component). Beyond the first/last valid frame we hold the
      boundary value. A fully-invalid sequence yields all zeros.

    At stance boundaries a cosine taper blends between the smoothed
    (interpolated) offset and the exact per-frame offset over
    ``taper_frames = max(1, round(taper_sec * fps))`` frames.  This
    eliminates the visible lurch that the previous hard switch caused.

    Returns
    -------
    offset : ``(N, 3)`` float64.
    """
    N = int(toes.shape[0])
    if N == 0:
        return np.zeros((0, 3), dtype=np.float64)

    per_frame = np.zeros((N, 3), dtype=np.float64)
    has = np.zeros(N, dtype=bool)

    for t in range(N):
        l_ok = bool(valid_mask[t, 0])
        r_ok = bool(valid_mask[t, 1])
        if l_ok and r_ok:
            per_frame[t] = 0.5 * (
                (anchors[t, 0] - toes[t, 0]) + (anchors[t, 1] - toes[t, 1])
            )
            has[t] = True
        elif l_ok:
            per_frame[t] = anchors[t, 0] - toes[t, 0]
            has[t] = True
        elif r_ok:
            per_frame[t] = anchors[t, 1] - toes[t, 1]
            has[t] = True

    if not has.any():
        return per_frame

    # Linearly interpolate per component across invalid spans, holding
    # the boundary values at the ends.
    valid_idx = np.flatnonzero(has)
    all_idx = np.arange(N, dtype=np.float64)
    interpolated = np.empty((N, 3), dtype=np.float64)
    for c in range(3):
        interpolated[:, c] = np.interp(
            all_idx, valid_idx.astype(np.float64), per_frame[valid_idx, c]
        )

    sigma = max(0.0, float(smoothing_sigma_sec) * float(fps))
    if sigma > 0.0 and N > 1:
        smoothed = gaussian_filter1d(interpolated, sigma=sigma, axis=0, mode="nearest")
    else:
        smoothed = interpolated

    # Smooth per-frame offsets *within* each stance episode to eliminate
    # jumps at source transitions (e.g., both-feet average → one-foot-
    # only).  Smoothing is restricted to each episode so it never bleeds
    # across non-contact gaps.  A short Gaussian (~1.2 frames) only
    # affects frames where the contributing-foot set changes.
    transition_sigma = max(0.5, 0.04 * float(fps))  # ~1.2 frames at 30 fps
    per_frame_smooth = per_frame.copy()

    # Cosine taper at stance boundaries: blend from the long-range
    # smoothed curve to the transition-smoothed per-frame offset over
    # ``taper_frames`` at each edge of a contact episode.  Interior
    # frames use the transition-smoothed offset (no hard overwrite).
    taper_frames = max(1, round(float(taper_sec) * float(fps)))
    out = smoothed.copy()

    # Find contiguous stance episodes from the combined valid mask
    any_valid = has  # True where at least one foot has a valid anchor
    episodes = find_episodes(any_valid)

    for start, end in episodes:
        length = end - start + 1

        # Per-episode transition smoothing (isolated from other episodes)
        if length > 2 and transition_sigma > 0:
            chunk = per_frame[start : end + 1].copy()
            chunk = gaussian_filter1d(chunk, sigma=transition_sigma, axis=0, mode="nearest")
            per_frame_smooth[start : end + 1] = chunk

        # Half-taper at each end, clamped to not exceed half the episode
        half = min(taper_frames, length // 2) if length > 1 else 0

        # Leading taper: frames [start, start + half)
        for k in range(half):
            # Cosine ramp from 0 to 1
            alpha = 0.5 * (1.0 - np.cos(np.pi * (k + 1) / (half + 1)))
            t = start + k
            out[t] = (1.0 - alpha) * smoothed[t] + alpha * per_frame_smooth[t]

        # Interior: transition-smoothed offset
        interior_start = start + half
        interior_end = end - half + 1  # exclusive
        if interior_start < interior_end:
            out[interior_start:interior_end] = per_frame_smooth[interior_start:interior_end]

        # Trailing taper: frames (end - half, end]
        for k in range(half):
            # Cosine ramp from 1 to 0
            alpha = 0.5 * (1.0 - np.cos(np.pi * (half - k) / (half + 1)))
            t = end - k
            out[t] = (1.0 - alpha) * smoothed[t] + alpha * per_frame_smooth[t]

    # Modulate by per-frame strength when provided
    if per_frame_strength is not None:
        out *= per_frame_strength[:, np.newaxis]

    return out


def apply_foot_pin(
    refined: dict,
    fps: float,
    preset: str = "moderate",
    sensitivity: str = "medium",
    pin_strength: float = 1.0,
    min_stance_frames: int = 2,
    anchor_window: int = 3,
    smoothing_sigma_sec: float = 0.1,
    height_threshold: float | None = None,
    velocity_threshold: float | None = None,
    sandpipe_data: dict | None = None,
    sandpipe_person_idx: int = 0,
) -> tuple[dict, dict]:
    """Run the full foot-pin pass on a refined params dict.

    Parameters
    ----------
    sandpipe_data : optional dict loaded from sandpipe-physics.json.
    sandpipe_person_idx : which person in the sandpipe export (0-based).
    """
    body_pose = np.asarray(refined["body_pose"])
    global_orient = np.asarray(refined["global_orient"])
    transl = np.asarray(refined["transl"], dtype=np.float64)

    sens_key = str(sensitivity).lower()
    if sens_key not in _SENSITIVITY_PRESETS:
        sens_key = "medium"
    sens_cfg = _SENSITIVITY_PRESETS[sens_key]
    h_thr = (
        float(height_threshold)
        if height_threshold is not None
        else sens_cfg["height_threshold"]
    )
    v_thr = (
        float(velocity_threshold)
        if velocity_threshold is not None
        else sens_cfg["velocity_threshold"]
    )

    if sandpipe_data is not None:
        from .sandpipe_contact import detect_contacts_sandpipe

        contacts, toes = detect_contacts_sandpipe(
            body_pose,
            global_orient,
            transl,
            fps=fps,
            sandpipe_data=sandpipe_data,
            velocity_threshold=v_thr,
            person_idx=sandpipe_person_idx,
        )
    else:
        contacts, toes = detect_contacts(
            body_pose,
            global_orient,
            transl,
            fps=fps,
            height_threshold=h_thr,
            velocity_threshold=v_thr,
        )
    anchors, valid, episode_count = track_stance_anchors(
        toes,
        contacts,
        min_stance_frames=min_stance_frames,
        anchor_window=anchor_window,
    )

    strength = float(np.clip(pin_strength, 0.0, 1.0))

    # Build per-frame strength from sandpipe grounding curve when available.
    # gc modulates pin intensity: hard pin when grounded (gc~1), soft during
    # transitions (gc~0.5), no pin when airborne (gc~0).
    pf_strength = None
    if sandpipe_data is not None:
        from .sandpipe_contact import compute_grounding_curve
        N = int(toes.shape[0])
        curve = compute_grounding_curve(sandpipe_data, N, fps, person_idx=sandpipe_person_idx)
        pf_strength = strength * curve  # (N,)

    offset = compute_pin_offset(
        toes,
        anchors,
        valid,
        fps=fps,
        smoothing_sigma_sec=smoothing_sigma_sec,
        per_frame_strength=pf_strength,
    )

    if pf_strength is not None:
        # Strength is already baked into the offset via per_frame_strength
        pinned_transl = (transl + offset).astype(np.float32)
        magnitudes = np.linalg.norm(offset, axis=1) if offset.size else np.zeros(0)
    else:
        pinned_transl = (transl + strength * offset).astype(np.float32)
        magnitudes = np.linalg.norm(offset * strength, axis=1) if offset.size else np.zeros(0)

    out = dict(refined)
    out["transl"] = pinned_transl
    stats = {
        "preset": preset,
        "sensitivity": sens_key,
        "pin_strength": strength,
        "height_threshold": h_thr,
        "velocity_threshold": v_thr,
        "contact_frame_count": int(valid.any(axis=1).sum()),
        "stance_episode_count": int(episode_count),
        "mean_pin_magnitude_m": float(magnitudes.mean()) if magnitudes.size else 0.0,
        "max_pin_magnitude_m": float(magnitudes.max()) if magnitudes.size else 0.0,
        "min_stance_frames": int(min_stance_frames),
        "anchor_window": int(anchor_window),
        "smoothing_sigma_sec": float(smoothing_sigma_sec),
    }
    return out, stats
