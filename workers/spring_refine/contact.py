"""Foot contact detection + stance anchor tracking for spring-refine v2.

Contact detection estimates a per-foot ground level from a low percentile
of each ankle's Y trajectory so one noisy low frame cannot anchor the
whole clip. This replaces an earlier reuse of the physics evaluator's
``detect_foot_contacts`` which used ``min(ankle_y)`` plus a tight 5 cm
window and caused the v2 pin pass to silently no-op on real footage.
The physics code is intentionally left alone so its before/after
metrics stay apples-to-apples against older runs.

Anchor tracking splits the contact mask into stance episodes and locks
each episode's pin target to the median of the first few toe positions —
robust to noisy first-contact frames.
"""

from __future__ import annotations

import numpy as np

from .fk import forward_kinematics_body

# SMPL-X body-joint indices (first 22 joints).
_L_ANKLE = 7
_R_ANKLE = 8
_L_FOOT = 10
_R_FOOT = 11


def detect_contacts(
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
    fps: float,
    height_threshold: float = 0.07,
    velocity_threshold: float = 0.5,
    ground_percentile: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run FK and detect per-frame contact for both feet.

    Uses a per-foot low-percentile ground estimate so mild asymmetry in
    the rest pose or gait does not force one foot to be treated as
    never-planted. A frame is "in contact" when its ankle is within
    ``height_threshold`` of that foot's percentile ground level AND the
    ankle's 3-D speed is below ``velocity_threshold``.

    Parameters
    ----------
    body_pose, global_orient, transl : SMPL-X body params.
    fps : frame rate for the velocity criterion.
    height_threshold : ankle height (m) above the per-foot percentile
        ground that still counts as contact.
    velocity_threshold : ankle speed (m/s) below which we count as still.
    ground_percentile : low percentile (0-100) used to estimate each
        foot's ground level. ``10.0`` tolerates single noisy low frames
        and still anchors to the planted band when a foot is stationary
        for ``>=`` ~10 % of the clip.

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

    dt = 1.0 / float(fps)
    for foot in range(2):
        y = ankles[:, foot, 1]
        ground = float(np.percentile(y, ground_percentile))
        height_ok = (y - ground) < height_threshold

        vel = np.zeros_like(ankles[:, foot, :])
        vel[1:] = (ankles[1:, foot, :] - ankles[:-1, foot, :]) / dt
        vel[0] = vel[1]
        speed = np.linalg.norm(vel, axis=-1)
        vel_ok = speed < velocity_threshold

        contacts[:, foot] = height_ok & vel_ok
    return contacts, toes


def _find_episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[(start, end_inclusive), ...]`` for contiguous True runs."""
    episodes: list[tuple[int, int]] = []
    n = int(mask.shape[0])
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            episodes.append((i, j))
            i = j + 1
        else:
            i += 1
    return episodes


def track_stance_anchors(
    foot_positions: np.ndarray,
    contacts: np.ndarray,
    min_stance_frames: int = 2,
    anchor_window: int = 3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Lock a pin target for each stance episode and hold it.

    Parameters
    ----------
    foot_positions : ``(N, 2, 3)`` toe world positions.
    contacts : ``(N, 2)`` bool contact mask.
    min_stance_frames : stance episodes shorter than this are discarded
        as detection noise.
    anchor_window : anchor = median of the first ``K`` contact frames in
        the episode.

    Returns
    -------
    anchors : ``(N, 2, 3)`` anchor positions (undefined where
        ``valid_mask`` is False; filled with zeros).
    valid_mask : ``(N, 2)`` bool — True during live stance episodes.
    episode_count : total accepted stance episodes across both feet.
    """
    N = int(foot_positions.shape[0])
    anchors = np.zeros((N, 2, 3), dtype=np.float64)
    valid = np.zeros((N, 2), dtype=bool)
    total_episodes = 0

    for foot in range(2):
        episodes = _find_episodes(contacts[:, foot])
        for start, end in episodes:
            length = end - start + 1
            if length < min_stance_frames:
                continue
            k = min(anchor_window, length)
            window = foot_positions[start : start + k, foot, :]
            anchor = np.median(window, axis=0)
            anchors[start : end + 1, foot, :] = anchor
            valid[start : end + 1, foot] = True
            total_episodes += 1

    return anchors, valid, total_episodes
