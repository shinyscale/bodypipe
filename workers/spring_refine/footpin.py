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

from .contact import detect_contacts, track_stance_anchors


def compute_pin_offset(
    toes: np.ndarray,
    anchors: np.ndarray,
    valid_mask: np.ndarray,
    fps: float,
    smoothing_sigma_sec: float = 0.1,
) -> np.ndarray:
    """Compute a per-frame translation correction that pins feet.

    Per frame:
    - Both feet valid → offset = mean(anchor_L - toe_L, anchor_R - toe_R).
    - One foot valid → offset = that foot's (anchor - toe).
    - Neither valid → interpolated from neighboring valid frames (linear
      on each component). Beyond the first/last valid frame we hold the
      boundary value. A fully-invalid sequence yields all zeros.

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

    return smoothed


def apply_foot_pin(
    refined: dict,
    fps: float,
    preset: str = "moderate",
    min_stance_frames: int = 2,
    anchor_window: int = 3,
    smoothing_sigma_sec: float = 0.1,
    height_threshold: float = 0.05,
    velocity_threshold: float = 0.3,
) -> tuple[dict, dict]:
    """Run the full foot-pin pass on a refined params dict.

    Modifies a shallow copy of ``refined`` in place: ``transl`` receives
    the pin correction, other keys are preserved. Returns ``(out, stats)``
    where ``stats`` summarises the contact detection (frame/episode
    counts, mean correction magnitude) for the metrics JSON.

    Parameters
    ----------
    refined : dict with ``body_pose``, ``global_orient``, ``transl`` and
        ``num_frames`` — post rotation filter (or raw baseline when the
        filter is disabled).
    fps : frame rate for contact velocity + smoothing sigma.
    preset : spring preset string, passed through for logging only.
    min_stance_frames, anchor_window, smoothing_sigma_sec,
    height_threshold, velocity_threshold : tuning knobs — see
        ``contact.detect_contacts`` and ``track_stance_anchors``.

    Returns
    -------
    out : dict — shallow copy of ``refined`` with pinned ``transl``.
    stats : dict — ``contact_frame_count``, ``stance_episode_count``,
        ``mean_pin_magnitude_m``, ``max_pin_magnitude_m``, ``preset``.
    """
    body_pose = np.asarray(refined["body_pose"])
    global_orient = np.asarray(refined["global_orient"])
    transl = np.asarray(refined["transl"], dtype=np.float64)

    contacts, toes = detect_contacts(
        body_pose,
        global_orient,
        transl,
        fps=fps,
        height_threshold=height_threshold,
        velocity_threshold=velocity_threshold,
    )
    anchors, valid, episode_count = track_stance_anchors(
        toes,
        contacts,
        min_stance_frames=min_stance_frames,
        anchor_window=anchor_window,
    )
    offset = compute_pin_offset(
        toes,
        anchors,
        valid,
        fps=fps,
        smoothing_sigma_sec=smoothing_sigma_sec,
    )

    pinned_transl = (transl + offset).astype(np.float32)
    out = dict(refined)
    out["transl"] = pinned_transl

    magnitudes = np.linalg.norm(offset, axis=1) if offset.size else np.zeros(0)
    stats = {
        "preset": preset,
        "contact_frame_count": int(valid.any(axis=1).sum()),
        "stance_episode_count": int(episode_count),
        "mean_pin_magnitude_m": float(magnitudes.mean()) if magnitudes.size else 0.0,
        "max_pin_magnitude_m": float(magnitudes.max()) if magnitudes.size else 0.0,
        "min_stance_frames": int(min_stance_frames),
        "anchor_window": int(anchor_window),
        "smoothing_sigma_sec": float(smoothing_sigma_sec),
    }
    return out, stats
