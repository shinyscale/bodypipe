"""Foot contact detection + stance anchor tracking for spring-refine v2.

Contact detection reuses the height/velocity heuristic from
``workers.physics.evaluate.detect_foot_contacts`` so before/after foot
skating metrics are apples-to-apples with the (now dormant) physics
code. Anchor tracking splits the contact mask into stance episodes and
locks each episode's pin target to the median of the first few toe
positions — robust to noisy first-contact frames.
"""

from __future__ import annotations

import numpy as np

from workers.physics.evaluate import detect_foot_contacts

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
    height_threshold: float = 0.05,
    velocity_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Run FK, detect per-frame contact for both feet.

    Parameters
    ----------
    body_pose, global_orient, transl : SMPL-X body params.
    fps : frame rate for the velocity criterion.
    height_threshold : ankle height (m) above the sequence minimum that
        still counts as contact.
    velocity_threshold : ankle speed (m/s) below which we count as still.

    Returns
    -------
    contacts : ``(N, 2)`` bool — ``[L, R]`` contact mask.
    toes : ``(N, 2, 3)`` float64 — L_Foot / R_Foot world positions.
    """
    joints = forward_kinematics_body(body_pose, global_orient, transl)
    ankles = joints[:, [_L_ANKLE, _R_ANKLE], :]
    toes = joints[:, [_L_FOOT, _R_FOOT], :]
    contacts = detect_foot_contacts(
        ankles, fps, height_threshold, velocity_threshold
    )
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
