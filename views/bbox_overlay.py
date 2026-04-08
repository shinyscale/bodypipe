"""Bbox overlay rendering — draw tracked person bboxes on video frames.

Why: The multi-person pipeline produces per-person bounding boxes at each frame.
Visualizing these on the video player is essential for identity verification —
users need to see who is being tracked, at what confidence, and whether corrected
bboxes differ from originals. This module provides a pure function that composites
bbox overlays onto raw video frames using OpenCV drawing primitives.
"""

from __future__ import annotations

import cv2
import numpy as np

from models.session import Session, PersonTrack
from theme import COLORS, PERSON_COLORS


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to (B, G, R) for OpenCV."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert '#RRGGBB' to (R, G, B) for RGB frame drawing."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _draw_dashed_rect(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
    gap_len: int = 6,
):
    """Draw a dashed rectangle on frame (RGB)."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Four edges: top, right, bottom, left
    edges = [
        ((x1, y1), (x2, y1)),  # top
        ((x2, y1), (x2, y2)),  # right
        ((x2, y2), (x1, y2)),  # bottom
        ((x1, y2), (x1, y1)),  # left
    ]
    for (sx, sy), (ex, ey) in edges:
        _draw_dashed_line(frame, (sx, sy), (ex, ey), color, thickness, dash_len, gap_len)


def _draw_dashed_line(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 10,
    gap_len: int = 6,
):
    """Draw a dashed line between two points."""
    x1, y1 = pt1
    x2, y2 = pt2
    dx = x2 - x1
    dy = y2 - y1
    length = max(1, int(np.sqrt(dx * dx + dy * dy)))
    segment = dash_len + gap_len

    for start in range(0, length, segment):
        end = min(start + dash_len, length)
        t0 = start / length
        t1 = end / length
        sx = int(x1 + dx * t0)
        sy = int(y1 + dy * t0)
        ex = int(x1 + dx * t1)
        ey = int(y1 + dy * t1)
        cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)


def _confidence_color_rgb(value: float) -> tuple[int, int, int]:
    """Return RGB color based on confidence value."""
    if value > 0.8:
        return _hex_to_rgb("#4ecca3")
    elif value > 0.5:
        return _hex_to_rgb("#ffd93d")
    else:
        return _hex_to_rgb("#ff6b6b")


def get_bbox_at_frame(
    track: PersonTrack, frame_idx: int
) -> np.ndarray | None:
    """Get bbox [x1, y1, x2, y2] at frame, preferring corrections over originals."""
    if track.bbox_corrections is not None and frame_idx < len(track.bbox_corrections):
        bbox = track.bbox_corrections[frame_idx]
        if bbox is not None and not np.all(bbox == 0):
            return bbox
    if track.bboxes is not None and frame_idx < len(track.bboxes):
        return track.bboxes[frame_idx]
    return None


def is_corrected_bbox(track: PersonTrack, frame_idx: int) -> bool:
    """Check if the bbox at this frame is a user correction (not original)."""
    if track.bbox_corrections is None:
        return False
    if frame_idx >= len(track.bbox_corrections):
        return False
    bbox = track.bbox_corrections[frame_idx]
    return bbox is not None and not np.all(bbox == 0)


def get_confidence_at_frame(
    track: PersonTrack, frame_idx: int
) -> float | None:
    """Get overall confidence at a specific frame."""
    if track.confidences is not None and 0 <= frame_idx < len(track.confidences):
        return float(track.confidences[frame_idx])
    return None


def render_bbox_overlay(
    frame: np.ndarray,
    session: Session,
    frame_idx: int,
    selected_person: int = -1,
    show_all_tracks: bool = False,
) -> np.ndarray:
    """Render all tracked person bboxes onto a video frame.

    Args:
        frame: RGB numpy array (H, W, 3) — will be copied, not modified in-place.
        session: Session containing person_tracks data.
        frame_idx: Current frame index.
        selected_person: Person ID of the selected person (drawn with thicker outline).
        show_all_tracks: If True, also draw inactive tracks with dashed gray outlines.

    Returns:
        RGB numpy array with bbox overlays composited.
    """
    if frame is None or len(session.person_tracks) == 0:
        return frame

    out = frame.copy()
    h, w = out.shape[:2]

    # Draw inactive tracks first (behind active ones) when show_all is on
    if show_all_tracks:
        for pid in sorted(session.inactive_tracks):
            track = session.person_tracks.get(pid)
            if track is None:
                continue
            bbox = get_bbox_at_frame(track, frame_idx)
            if bbox is None:
                continue
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            gray = (128, 128, 128)
            _draw_dashed_rect(out, (x1, y1), (x2, y2), gray, thickness=1)
            # Label
            label = f"Track {pid} (inactive)"
            cv2.putText(
                out, label, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, gray, 1, cv2.LINE_AA,
            )

    # Draw active tracks
    for pid, track in sorted(session.person_tracks.items()):
        if pid in session.inactive_tracks:
            continue

        bbox = get_bbox_at_frame(track, frame_idx)
        if bbox is None:
            continue

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color_hex = PERSON_COLORS[pid % len(PERSON_COLORS)]
        color_rgb = _hex_to_rgb(color_hex)
        is_selected = pid == selected_person
        thickness = 3 if is_selected else 2

        corrected = is_corrected_bbox(track, frame_idx)

        if corrected:
            # Dashed outline for corrected bboxes
            _draw_dashed_rect(out, (x1, y1), (x2, y2), color_rgb, thickness)
        else:
            # Solid rectangle for original bboxes
            cv2.rectangle(out, (x1, y1), (x2, y2), color_rgb, thickness, cv2.LINE_AA)

        # Confidence badge above the bbox
        confidence = get_confidence_at_frame(track, frame_idx)
        if confidence is not None:
            prefix = "\u25b6 " if is_selected else ""
            label = f"{prefix}ID {pid} ({confidence:.2f})"
        else:
            prefix = "\u25b6 " if is_selected else ""
            label = f"{prefix}ID {pid}"

        # Measure text for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Badge background
        badge_y = max(y1 - th - 8, 0)
        badge_x = x1
        # Semi-transparent background rectangle
        overlay = out[badge_y:badge_y + th + 6, badge_x:badge_x + tw + 8].copy()
        if overlay.size > 0:
            bg_color = np.array(color_rgb, dtype=np.float32)
            alpha = 0.6
            blended = (overlay.astype(np.float32) * (1 - alpha) + bg_color * alpha).astype(np.uint8)
            out[badge_y:badge_y + th + 6, badge_x:badge_x + tw + 8] = blended

        # Text on badge
        text_color = (255, 255, 255)
        cv2.putText(
            out, label, (badge_x + 4, badge_y + th + 2),
            font, font_scale, text_color, font_thickness, cv2.LINE_AA,
        )

        # Confidence dot in bottom-right of bbox
        if confidence is not None:
            dot_color = _confidence_color_rgb(confidence)
            dot_center = (x2 - 8, y2 - 8)
            cv2.circle(out, dot_center, 5, dot_color, -1, cv2.LINE_AA)

    return out


def render_edit_preview(
    frame: np.ndarray,
    edit_state: dict,
) -> np.ndarray:
    """Render bbox edit preview markers on frame.

    Draws a crosshair at corner1 while waiting for the second click.
    The frame is assumed to already be a copy (from render_bbox_overlay).

    Args:
        frame: RGB numpy array — modified in-place if already a copy.
        edit_state: Dict with 'corner1' key → (x, y) pixel coords.

    Returns:
        RGB numpy array with preview drawn.
    """
    if frame is None or not edit_state:
        return frame

    out = frame.copy()
    corner1 = edit_state.get("corner1")
    accent = _hex_to_rgb(COLORS["accent"])

    if corner1:
        x, y = int(corner1[0]), int(corner1[1])
        h, w = out.shape[:2]
        # Draw crosshair at first corner
        x1_line = max(0, x - 15)
        x2_line = min(w - 1, x + 15)
        y1_line = max(0, y - 15)
        y2_line = min(h - 1, y + 15)
        cv2.line(out, (x1_line, y), (x2_line, y), accent, 2, cv2.LINE_AA)
        cv2.line(out, (x, y1_line), (x, y2_line), accent, 2, cv2.LINE_AA)

    return out
