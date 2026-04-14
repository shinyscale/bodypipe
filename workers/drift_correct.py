"""Vision-based auto drift correction using a local VLM via Ollama.

Samples keyframes from a source video, sends them to a local VLM (Gemma 4
via Ollama) to estimate scene-relative displacement, compares against GVHMR's
root trajectory, and outputs correction anchors compatible with
``compute_position_offsets()`` in ``GVHMR/pose_correction.py``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import requests
import torch

log = logging.getLogger(__name__)

OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "gemma4:31b"


# ---------------------------------------------------------------------------
# Frame sampling & extraction
# ---------------------------------------------------------------------------


def sample_keyframe_indices(
    num_frames: int,
    fps: float,
    interval_sec: float = 2.0,
    max_samples: int = 10,
) -> list[int]:
    """Return evenly-spaced frame indices, always including first and last."""
    if num_frames <= 1:
        return [0]

    step = max(1, int(round(fps * interval_sec)))
    indices = list(range(0, num_frames, step))

    # Always include last frame
    if indices[-1] != num_frames - 1:
        indices.append(num_frames - 1)

    # Cap at max_samples, re-spacing evenly if needed
    if len(indices) > max_samples:
        indices = [
            int(round(i * (num_frames - 1) / (max_samples - 1)))
            for i in range(max_samples)
        ]

    return indices


def extract_frames(
    video_path: Path,
    frame_indices: list[int],
    max_long_edge: int = 768,
) -> dict[int, np.ndarray]:
    """Extract specific frames from a video as RGB uint8 arrays.

    Frames are resized so the long edge does not exceed *max_long_edge*
    to keep VLM inference efficient.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: dict[int, np.ndarray] = {}
    try:
        for idx in sorted(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                log.warning("Failed to read frame %d from %s", idx, video_path)
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w = frame.shape[:2]
            if max(h, w) > max_long_edge:
                scale = max_long_edge / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            frames[idx] = frame
    finally:
        cap.release()

    return frames


# ---------------------------------------------------------------------------
# VLM displacement estimation
# ---------------------------------------------------------------------------

_DISPLACEMENT_PROMPT = """\
You are analyzing motion capture drift in a video sequence. These {n} frames \
are sampled at regular intervals ({interval:.1f}s apart) from a single camera.

For each frame after Frame 1, estimate how far the person moved on the ground \
plane relative to their position in Frame 1. Use fixed scene elements (walls, \
floor edges, furniture, doorframes) as reference — NOT the camera position.

Report displacement as fractions of the image width:
- right_frac: positive = person moved rightward in the image
- forward_frac: positive = person moved deeper/away from camera

Scale reference: a standard doorframe is ~0.9m wide, ceiling height ~2.4m.

Reply with ONLY a JSON array:
[{{"frame": 2, "right_frac": 0.05, "forward_frac": -0.02}}, ...]
Frame numbers are 1-indexed (Frame 1 is the reference with zero displacement).\
"""


def _frame_to_base64(frame: np.ndarray) -> str:
    """Encode an RGB uint8 numpy array as a base64 PNG string."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode frame as PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def check_ollama(ollama_url: str = OLLAMA_DEFAULT_URL) -> bool:
    """Return True if Ollama is reachable."""
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def estimate_displacements(
    frames: dict[int, np.ndarray],
    frame_indices: list[int],
    fps: float,
    interval_sec: float = 2.0,
    model: str = OLLAMA_DEFAULT_MODEL,
    ollama_url: str = OLLAMA_DEFAULT_URL,
) -> list[dict]:
    """Send sampled frames to Ollama VLM and parse displacement estimates.

    Returns list of ``{"frame_idx": int, "right_frac": float, "forward_frac": float}``
    where *frame_idx* is the actual video frame index (0-based).
    """
    ordered_indices = [i for i in frame_indices if i in frames]
    if len(ordered_indices) < 2:
        raise ValueError("Need at least 2 frames for displacement estimation")

    n = len(ordered_indices)
    images_b64 = [_frame_to_base64(frames[i]) for i in ordered_indices]

    # Build prompt with frame labels
    frame_labels = "\n".join(f"Image {i}: Frame {i}" for i in range(1, n + 1))
    prompt_text = (
        frame_labels + "\n\n"
        + _DISPLACEMENT_PROMPT.format(n=n, interval=interval_sec)
    )

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt_text,
            "images": images_b64,
        }],
        "stream": False,
        "options": {"temperature": 0},
    }

    log.info("Sending %d images to Ollama (%s, model=%s)...", n, ollama_url, model)
    resp = requests.post(
        f"{ollama_url}/api/chat",
        json=payload,
        timeout=300,
    )
    resp.raise_for_status()

    response_text = resp.json()["message"]["content"]
    log.info("VLM response:\n%s", response_text)

    return _parse_displacements(response_text, ordered_indices)


def _parse_displacements(
    response: str,
    ordered_indices: list[int],
) -> list[dict]:
    """Parse VLM JSON response into displacement dicts with real frame indices."""
    # Try JSON array extraction first
    try:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            result = []
            for item in data:
                seq_frame = item["frame"]  # 1-indexed sequence number
                if seq_frame < 2 or seq_frame > len(ordered_indices):
                    continue
                result.append({
                    "frame_idx": ordered_indices[seq_frame - 1],
                    "right_frac": float(item["right_frac"]),
                    "forward_frac": float(item["forward_frac"]),
                })
            if result:
                return result
    except (json.JSONDecodeError, KeyError, IndexError):
        pass

    # Fallback: regex extraction
    log.warning("JSON parse failed, falling back to regex extraction")
    pattern = (
        r'"frame"\s*:\s*(\d+)\s*,\s*'
        r'"right_frac"\s*:\s*([+-]?\d*\.?\d+)\s*,\s*'
        r'"forward_frac"\s*:\s*([+-]?\d*\.?\d+)'
    )
    matches = re.findall(pattern, response)
    if not matches:
        raise ValueError(f"Could not parse VLM response:\n{response}")

    result = []
    for seq_str, right_str, fwd_str in matches:
        seq_frame = int(seq_str)
        if seq_frame < 2 or seq_frame > len(ordered_indices):
            continue
        result.append({
            "frame_idx": ordered_indices[seq_frame - 1],
            "right_frac": float(right_str),
            "forward_frac": float(fwd_str),
        })

    if not result:
        raise ValueError(f"No valid displacements parsed from VLM response:\n{response}")
    return result


# ---------------------------------------------------------------------------
# SLAM loading
# ---------------------------------------------------------------------------


def load_slam_w2c(slam_path: Path) -> np.ndarray:
    """Load SLAM poses as ``(N, 4, 4)`` W2C matrices.

    Handles both SimpleVO ``(N, 4, 4)`` and DPVO ``(N, 7)`` formats.
    Same normalization logic as ``app_window.py:_normalize_slam_w2c()``.
    """
    slam = torch.load(str(slam_path), map_location="cpu", weights_only=False)
    arr = np.array(slam, dtype=np.float32)

    if arr.ndim == 2 and arr.shape[1] == 7:
        from scipy.spatial.transform import Rotation as _R

        quats_xyzw = arr[:, 3:7]
        R_c2w = _R.from_quat(quats_xyzw).as_matrix()
        t_c2w = arr[:, :3].astype(np.float32)
        R_w2c = R_c2w.transpose(0, 2, 1)
        t_w2c = -np.einsum("nij,nj->ni", R_w2c, t_c2w)
        T = np.tile(np.eye(4, dtype=np.float32), (len(arr), 1, 1))
        T[:, :3, :3] = R_w2c
        T[:, :3, 3] = t_w2c
        arr = T

    return arr


def find_slam(output_dir: Path, person_dir: Path | None = None) -> Path | None:
    """Auto-discover SLAM file following the project's search convention.

    Checks ``shared_slam.pt`` at various parent levels, then per-person
    ``demo/preprocess/slam.pt``.
    """
    candidates: list[Path] = [output_dir / "shared_slam.pt"]
    if person_dir is not None:
        candidates.append(person_dir / "demo" / "preprocess" / "slam.pt")

    parent = output_dir
    for _ in range(4):
        candidates.append(parent / "shared_slam.pt")
        candidates.append(parent / "demo" / "preprocess" / "slam.pt")
        parent = parent.parent

    for p in candidates:
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Camera-to-world coordinate mapping
# ---------------------------------------------------------------------------


def compute_per_frame_scale(
    transl_incam: np.ndarray,  # (N, 3) person position in camera space
    K: np.ndarray,             # (N, 3, 3) or (3, 3) camera intrinsics
) -> np.ndarray:
    """Compute meters-per-screen-fraction at person depth for each frame.

    Uses the pinhole model: at depth Z with focal length f and image width W,
    one full image width spans ``Z * W / f`` meters.

    Returns ``(N,)`` array of scale factors.
    """
    depths = transl_incam[:, 2]  # Z in camera space = depth
    if K.ndim == 3:
        fx = K[:, 0, 0]
        img_w = K[:, 0, 2] * 2  # cx ≈ W/2
    else:
        fx = np.full(len(depths), K[0, 0])
        img_w = np.full(len(depths), K[0, 2] * 2)

    scale = depths * img_w / fx
    # Clamp to reasonable range (0.5–20m per screen-width)
    scale = np.clip(scale, 0.5, 20.0)
    return scale.astype(np.float64)


def camera_displacements_to_world(
    displacements: list[dict],
    slam_w2c: np.ndarray,  # (N, 4, 4)
    scale_per_frame: np.ndarray | float = 3.0,
) -> dict[int, np.ndarray]:
    """Convert camera-relative fractional displacements to world-space vectors.

    Uses SLAM camera orientation at each frame to map image-plane right/forward
    to world-space ground-plane directions (XZ, Y zeroed).

    *scale_per_frame* is either a scalar fallback or a ``(N,)`` array from
    ``compute_per_frame_scale()`` giving meters-per-screen-fraction at each frame.

    Returns ``{frame_idx: (3,) world_displacement}`` relative to frame 0.
    """
    result: dict[int, np.ndarray] = {}
    is_array = isinstance(scale_per_frame, np.ndarray)

    for d in displacements:
        fidx = d["frame_idx"]
        if fidx >= len(slam_w2c):
            log.warning("Frame %d beyond SLAM range (%d), skipping", fidx, len(slam_w2c))
            continue

        scale = float(scale_per_frame[fidx]) if is_array else float(scale_per_frame)

        # Camera-to-world rotation: R_c2w = R_w2c^T
        R_w2c = slam_w2c[fidx, :3, :3].copy()
        R_c2w = R_w2c.T

        # Camera axes in world (OpenCV: X=right, Z=forward)
        cam_right = R_c2w[:, 0].copy()
        cam_forward = R_c2w[:, 2].copy()

        # Project to ground plane (zero Y, renormalize)
        cam_right[1] = 0.0
        norm = np.linalg.norm(cam_right)
        if norm > 1e-6:
            cam_right /= norm

        cam_forward[1] = 0.0
        norm = np.linalg.norm(cam_forward)
        if norm > 1e-6:
            cam_forward /= norm

        right_m = d["right_frac"] * scale
        forward_m = d["forward_frac"] * scale

        world_disp = right_m * cam_right + forward_m * cam_forward
        world_disp[1] = 0.0
        result[fidx] = world_disp.astype(np.float64)

    return result


# ---------------------------------------------------------------------------
# Drift anchor computation
# ---------------------------------------------------------------------------


def compute_drift_anchors(
    transl_world: np.ndarray,  # (N, 3)
    vision_displacements: dict[int, np.ndarray],  # {frame: world_disp from frame 0}
    frame_0: int = 0,
) -> list[tuple[int, np.ndarray]]:
    """Compute correction anchors from GVHMR trajectory vs vision estimates.

    Returns ``list[tuple[int, np.ndarray]]`` — direct input for
    ``compute_position_offsets()`` in ``GVHMR/pose_correction.py``.

    Each anchor is ``(frame_index, target_root_position)`` where the target
    is the GVHMR position minus the estimated drift (XZ only, Y preserved).
    """
    anchors: list[tuple[int, np.ndarray]] = []

    # Frame 0: keep original position (zero drift by definition)
    anchors.append((frame_0, transl_world[frame_0].copy().astype(np.float64)))

    gvhmr_origin = transl_world[frame_0].astype(np.float64)

    for fidx, vision_disp in sorted(vision_displacements.items()):
        if fidx == frame_0:
            continue

        gvhmr_disp = transl_world[fidx].astype(np.float64) - gvhmr_origin
        drift = gvhmr_disp - vision_disp
        drift[1] = 0.0

        target = transl_world[fidx].astype(np.float64) - drift
        target[1] = transl_world[fidx][1]  # Preserve original Y
        anchors.append((fidx, target))

    return anchors


# ---------------------------------------------------------------------------
# Multi-person: detection bbox loading & frame annotation
# ---------------------------------------------------------------------------

_PERSON_COLORS = [
    (255, 50, 50),    # red
    (50, 120, 255),   # blue
    (50, 200, 50),    # green
    (255, 180, 0),    # orange
    (180, 50, 255),   # purple
    (0, 200, 200),    # cyan
    (255, 100, 180),  # pink
    (160, 160, 0),    # olive
]


def load_detection_bboxes(
    output_dir: Path,
    session_manifest: dict,
) -> dict[int, np.ndarray]:
    """Load per-person bboxes from ``detection/all_tracks.pt``.

    Returns ``{person_idx: (N_frames, 4) xyxy}`` keyed by person index
    (0, 1, ...) matching the order in ``session_manifest["person_bindings"]``.
    """
    tracks_path = output_dir / "detection" / "all_tracks.pt"
    if not tracks_path.is_file():
        raise FileNotFoundError(f"Detection tracks not found: {tracks_path}")

    data = torch.load(str(tracks_path), map_location="cpu", weights_only=False)
    tracks_list = data["tracks"]

    # Build track_id → bbox array mapping
    tid_to_bbox: dict[int, np.ndarray] = {}
    for t in tracks_list:
        tid = int(t["track_id"])
        tid_to_bbox[tid] = np.array(t["bbx_xyxy"], dtype=np.float32)

    # Map person_idx → track_id via person_bindings
    result: dict[int, np.ndarray] = {}
    for pidx, pb in enumerate(session_manifest["person_bindings"]):
        tid = pb["track_id"]
        if tid in tid_to_bbox:
            result[pidx] = tid_to_bbox[tid]
        else:
            log.warning("Track %d for person %d not found in all_tracks.pt", tid, pidx)

    return result


def annotate_frame_with_persons(
    frame: np.ndarray,
    bboxes: dict[int, tuple[float, float, float, float]],
) -> np.ndarray:
    """Draw labeled bounding boxes on a frame for VLM person identification.

    *bboxes* maps ``{person_idx: (x1, y1, x2, y2)}`` in the frame's pixel space.
    Returns a copy with colored rectangles and "Person N" labels drawn.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for pidx, (x1, y1, x2, y2) in sorted(bboxes.items()):
        color = _PERSON_COLORS[pidx % len(_PERSON_COLORS)]
        ix1, iy1 = int(round(x1)), int(round(y1))
        ix2, iy2 = int(round(x2)), int(round(y2))

        # Draw rectangle (RGB, cv2 wants BGR for putText but we're in RGB)
        cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), color, 2)

        # Label above the box
        label = f"Person {pidx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(h, w) / 1200)
        thickness = max(1, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        # Background rectangle for label
        cv2.rectangle(
            annotated,
            (ix1, max(0, iy1 - th - 6)),
            (ix1 + tw + 4, iy1),
            color, -1,
        )
        cv2.putText(
            annotated, label,
            (ix1 + 2, max(th + 2, iy1 - 4)),
            font, font_scale, (255, 255, 255), thickness,
        )

    return annotated


# ---------------------------------------------------------------------------
# Multi-person: VLM position estimation
# ---------------------------------------------------------------------------

_MULTI_PERSON_PROMPT = """\
You are analyzing a multi-person motion capture video. Each frame has colored \
bounding boxes labeling {n_persons} people (Person 0, Person 1, etc.).

These {n} frames are sampled at regular intervals ({interval:.1f}s apart).

For EACH frame, estimate where each labeled person's feet touch the ground, \
as fractions of the image dimensions:
- x_frac: 0.0 = left edge, 1.0 = right edge
- y_frac: 0.0 = top edge, 1.0 = bottom edge

Reply with ONLY a JSON array:
[
  {{"frame": 1, "persons": [
    {{"id": 0, "x_frac": 0.35, "y_frac": 0.85}},
    {{"id": 1, "x_frac": 0.55, "y_frac": 0.83}}
  ]}},
  ...
]
Frame numbers are 1-indexed. Include ALL frames and ALL persons.\
"""


def estimate_person_positions(
    frames: dict[int, np.ndarray],
    frame_indices: list[int],
    n_persons: int,
    interval_sec: float = 2.0,
    model: str = OLLAMA_DEFAULT_MODEL,
    ollama_url: str = OLLAMA_DEFAULT_URL,
) -> list[dict]:
    """Send annotated multi-person frames to VLM, get per-person positions.

    Returns list of ``{"frame_idx": int, "persons": [{"id": int, "x_frac": float, "y_frac": float}, ...]}``
    where *frame_idx* is in GVHMR frame space.
    """
    ordered_indices = [i for i in frame_indices if i in frames]
    if len(ordered_indices) < 2:
        raise ValueError("Need at least 2 frames")

    n = len(ordered_indices)
    images_b64 = [_frame_to_base64(frames[i]) for i in ordered_indices]

    frame_labels = "\n".join(f"Image {i}: Frame {i}" for i in range(1, n + 1))
    prompt_text = (
        frame_labels + "\n\n"
        + _MULTI_PERSON_PROMPT.format(n=n, n_persons=n_persons, interval=interval_sec)
    )

    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": prompt_text,
            "images": images_b64,
        }],
        "stream": False,
        "options": {"temperature": 0},
    }

    log.info(
        "Sending %d annotated frames (%d persons) to Ollama (%s, model=%s)...",
        n, n_persons, ollama_url, model,
    )
    resp = requests.post(
        f"{ollama_url}/api/chat",
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()

    response_text = resp.json()["message"]["content"]
    log.info("VLM response:\n%s", response_text)

    return _parse_person_positions(response_text, ordered_indices, n_persons)


def _parse_person_positions(
    response: str,
    ordered_indices: list[int],
    n_persons: int,
) -> list[dict]:
    """Parse VLM JSON into per-frame, per-person position dicts."""
    try:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            result = []
            for item in data:
                seq_frame = item["frame"]
                if seq_frame < 1 or seq_frame > len(ordered_indices):
                    continue
                actual_idx = ordered_indices[seq_frame - 1]
                persons = []
                for p in item.get("persons", []):
                    persons.append({
                        "id": int(p["id"]),
                        "x_frac": float(p["x_frac"]),
                        "y_frac": float(p["y_frac"]),
                    })
                result.append({"frame_idx": actual_idx, "persons": persons})
            if result:
                return result
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass

    raise ValueError(f"Could not parse multi-person VLM response:\n{response}")


# ---------------------------------------------------------------------------
# Multi-person: screen→world conversion & drift solver
# ---------------------------------------------------------------------------


def screen_to_world_positions(
    person_positions: list[dict],
    slam_w2c: np.ndarray,
    per_person_incam: dict[int, np.ndarray],  # {pidx: (N,3) incam transl}
    K: np.ndarray,                             # (N,3,3) or (3,3)
) -> dict[int, dict[int, np.ndarray]]:
    """Convert VLM screen-fraction positions to world-space XZ.

    Returns ``{frame_idx: {person_idx: (3,) world_pos}}``.
    """
    result: dict[int, dict[int, np.ndarray]] = {}

    for frame_data in person_positions:
        fidx = frame_data["frame_idx"]
        if fidx >= len(slam_w2c):
            continue

        # Camera pose at this frame
        R_w2c = slam_w2c[fidx, :3, :3]
        t_w2c = slam_w2c[fidx, :3, 3]
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c

        # Intrinsics at this frame
        Kf = K[fidx] if K.ndim == 3 else K
        K_inv = np.linalg.inv(Kf)
        img_w = Kf[0, 2] * 2
        img_h = Kf[1, 2] * 2

        frame_positions: dict[int, np.ndarray] = {}
        for p in frame_data["persons"]:
            pidx = p["id"]
            if pidx not in per_person_incam:
                continue

            incam = per_person_incam[pidx]
            if fidx >= len(incam):
                continue

            # Screen fraction → pixel
            px = p["x_frac"] * img_w
            py = p["y_frac"] * img_h

            # Pixel → camera ray
            ray_cam = K_inv @ np.array([px, py, 1.0])

            # Scale by person's depth
            depth = float(incam[fidx, 2])
            if depth < 0.1:
                continue
            pos_cam = ray_cam * depth

            # Camera → world
            pos_world = R_c2w @ pos_cam + t_c2w
            pos_world[1] = 0.0  # Ground plane
            frame_positions[pidx] = pos_world.astype(np.float64)

        if frame_positions:
            result[fidx] = frame_positions

    return result


def compute_multi_person_drift(
    per_person_transl: dict[int, np.ndarray],   # {pidx: (N,3) transl_world}
    vlm_world_positions: dict[int, dict[int, np.ndarray]],  # {frame: {pidx: pos}}
    reference_person: int = 0,
) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Compute per-person correction anchors from VLM inter-person constraints.

    Uses *reference_person* as the anchor — applies single-person global drift
    correction to them, then corrects all others relative to reference.

    Returns ``{person_idx: [(frame, target_pos), ...]}`` — each list is
    compatible with ``compute_position_offsets()``.
    """
    person_ids = sorted(per_person_transl.keys())
    if reference_person not in person_ids:
        reference_person = person_ids[0]

    keyframes = sorted(vlm_world_positions.keys())
    if not keyframes:
        return {}

    ref_transl = per_person_transl[reference_person]

    # Step 1: Global drift correction for reference person
    # VLM positions give us scene-anchored estimates at each keyframe.
    # Compute displacement of reference person: VLM vs GVHMR
    ref_vision_disp: dict[int, np.ndarray] = {}
    f0 = keyframes[0]
    if f0 in vlm_world_positions and reference_person in vlm_world_positions[f0]:
        vlm_ref_origin = vlm_world_positions[f0][reference_person]
        for fidx in keyframes[1:]:
            fp = vlm_world_positions.get(fidx, {})
            if reference_person in fp:
                disp = fp[reference_person] - vlm_ref_origin
                disp[1] = 0.0
                ref_vision_disp[fidx] = disp

    ref_anchors = compute_drift_anchors(ref_transl, ref_vision_disp, frame_0=f0)

    # Step 2: Relative corrections for other persons
    all_anchors: dict[int, list[tuple[int, np.ndarray]]] = {
        reference_person: ref_anchors,
    }

    for pidx in person_ids:
        if pidx == reference_person:
            continue

        p_transl = per_person_transl[pidx]
        anchors: list[tuple[int, np.ndarray]] = []

        for fidx in keyframes:
            fp = vlm_world_positions.get(fidx, {})
            if reference_person not in fp or pidx not in fp:
                continue

            # VLM says the inter-person vector should be:
            vlm_delta = fp[pidx] - fp[reference_person]
            vlm_delta[1] = 0.0

            # GVHMR says the inter-person vector is:
            gvhmr_delta = p_transl[fidx] - ref_transl[fidx]
            gvhmr_delta = gvhmr_delta.astype(np.float64)
            gvhmr_delta[1] = 0.0

            # Relative drift: how much this person has diverged from reference
            rel_drift = gvhmr_delta - vlm_delta
            rel_drift[1] = 0.0

            # Also apply the global drift correction (from reference person)
            # Find reference person's correction at this frame
            ref_correction = np.zeros(3, dtype=np.float64)
            for rf, rt in ref_anchors:
                if rf == fidx:
                    ref_correction = rt - ref_transl[fidx].astype(np.float64)
                    ref_correction[1] = 0.0
                    break

            target = p_transl[fidx].astype(np.float64) - rel_drift + ref_correction
            target[1] = p_transl[fidx][1]  # Preserve Y
            anchors.append((fidx, target))

        all_anchors[pidx] = anchors

    return all_anchors
