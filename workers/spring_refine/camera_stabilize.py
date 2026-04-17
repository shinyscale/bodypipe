"""Camera-stabilized world grounding for GVHMR body estimation.

Re-derives world-space body params (global_orient, transl) using a
smoothed camera trajectory instead of the raw SLAM cam_angvel that
GVHMR ingested at inference time. The network's pose predictions
(global_orient_gv, local_transl_vel, global_orient_c) are stored in
hmr4d_results.pt under ``net_outputs.decode_dict``; we load them
unchanged, smooth the SLAM W2C, recompute cam_angvel, and re-run
``get_smpl_params_w_Rt_v2``.

This removes camera-induced drift at the source rather than trying to
pin it downstream. Measured on HopeYouDo_14: person 1 XZ drift
256 → 26 cm (90 % reduction).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def stabilize_world_params(
    pt_path: Path,
    slam_path: Path | None = None,
    cam_smooth_preset: str = "moderate",
    fps: float = 30.0,
) -> dict | None:
    """Re-derive world-space body params using smoothed camera trajectory.

    Parameters
    ----------
    pt_path : Path to ``hmr4d_results.pt`` (must contain ``net_outputs``
        with ``decode_dict``).
    slam_path : Explicit path to SLAM .pt. If None, auto-discovers
        ``shared_slam.pt`` or per-person ``demo/preprocess/slam.pt``.
    cam_smooth_preset : One Euro filter preset (``"light"`` / ``"moderate"``
        / ``"heavy"``).
    fps : Frame rate for the smoothing filter.

    Returns
    -------
    dict with ``"global_orient"`` and ``"transl"`` as numpy float32 arrays
    of shape ``(N, 3)``, or ``None`` if required data is missing.
    """
    import torch

    # 1. Load hmr4d_results.pt, extract decode_dict
    pt_path = Path(pt_path)
    if not pt_path.is_file():
        logger.warning("stabilize: %s not found", pt_path)
        return None

    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    net_outputs = data.get("net_outputs")
    if not isinstance(net_outputs, dict):
        logger.info("stabilize: no net_outputs in %s (older file?), skipping", pt_path.name)
        return None

    decode_dict = net_outputs.get("decode_dict")
    if not isinstance(decode_dict, dict):
        logger.info("stabilize: no decode_dict in net_outputs, skipping")
        return None

    # Required keys from the decoder
    global_orient_gv = decode_dict.get("global_orient_gv")  # (B, L, 3)
    local_transl_vel = decode_dict.get("local_transl_vel")  # (B, L, 3)
    global_orient_c = decode_dict.get("global_orient")       # (B, L, 3)
    if global_orient_gv is None or local_transl_vel is None or global_orient_c is None:
        logger.info("stabilize: decode_dict missing required keys, skipping")
        return None

    # 2. Find and load SLAM W2C
    slam_w2c = _load_slam(pt_path, slam_path)
    if slam_w2c is None:
        logger.info("stabilize: no SLAM data found, skipping")
        return None

    # 3. Smooth SLAM trajectory
    slam_c2w = np.linalg.inv(slam_w2c)
    slam_c2w_smooth = _smooth_c2w(slam_c2w, fps=fps, preset=cam_smooth_preset)
    slam_w2c_smooth = np.linalg.inv(slam_c2w_smooth)

    # 4. Compute cam_angvel from smoothed rotations
    R_w2c = torch.from_numpy(slam_w2c_smooth[:, :3, :3]).float()
    cam_angvel = _compute_cam_angvel(R_w2c)  # (N, 6)

    # Match frame count to decode_dict sequence length
    L = global_orient_gv.shape[1] if global_orient_gv.ndim >= 3 else global_orient_gv.shape[0]
    N_slam = cam_angvel.shape[0]
    if N_slam < L:
        # Pad by repeating last frame
        pad = cam_angvel[-1:].expand(L - N_slam, -1)
        cam_angvel = torch.cat([cam_angvel, pad], dim=0)
    elif N_slam > L:
        cam_angvel = cam_angvel[:L]

    # Ensure batch dimension (B=1)
    if cam_angvel.ndim == 2:
        cam_angvel = cam_angvel.unsqueeze(0)  # (1, L, 6)

    # Ensure decode_dict tensors have batch dim
    if global_orient_gv.ndim == 2:
        global_orient_gv = global_orient_gv.unsqueeze(0)
        local_transl_vel = local_transl_vel.unsqueeze(0)
        global_orient_c = global_orient_c.unsqueeze(0)

    # 5. Re-run world grounding with smoothed cam_angvel
    from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import get_smpl_params_w_Rt_v2

    result = get_smpl_params_w_Rt_v2(
        global_orient_gv=global_orient_gv,
        local_transl_vel=local_transl_vel,
        global_orient_c=global_orient_c,
        cam_angvel=cam_angvel,
    )

    # 6. Extract and return as numpy
    go = result["global_orient"][0].cpu().numpy().astype(np.float32)  # (L, 3)
    tr = result["transl"][0].cpu().numpy().astype(np.float32)         # (L, 3)

    logger.info(
        "stabilize: re-derived %d frames with preset=%s", go.shape[0], cam_smooth_preset
    )
    return {"global_orient": go, "transl": tr}


def _load_slam(pt_path: Path, slam_path: Path | None) -> np.ndarray | None:
    """Find and load SLAM W2C matrices, normalized to (N, 4, 4)."""
    import torch

    candidates: list[Path] = []
    if slam_path is not None:
        candidates.append(Path(slam_path))

    # Auto-discover: shared_slam.pt in various parent locations
    # Multi-person: pt_path is inside person_*/demo/, SLAM is at output_dir level
    # Single-person: pt_path is in GVHMR output, SLAM is nearby
    parent = pt_path.parent
    for _ in range(5):
        candidates.append(parent / "shared_slam.pt")
        candidates.append(parent / "demo" / "preprocess" / "slam.pt")
        parent = parent.parent

    for path in candidates:
        if not path.is_file():
            continue
        try:
            slam = torch.load(str(path), map_location="cpu", weights_only=False)
            arr = _normalize_slam_w2c(slam)
            logger.info("stabilize: loaded SLAM from %s (%d frames)", path, arr.shape[0])
            return arr
        except Exception as exc:
            logger.warning("stabilize: failed to load SLAM from %s: %s", path, exc)
            continue
    return None


def _normalize_slam_w2c(slam) -> np.ndarray:
    """Return SLAM poses as W2C 4x4 matrices regardless of on-disk format.

    Handles both SimpleVO (4x4 matrices) and DPVO (tx,ty,tz,qx,qy,qz,qw rows).
    Same logic as ``app_window._normalize_slam_w2c``.
    """
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


# --- Camera smoothing (One Euro filter) ---
# Duplicated from app_window to avoid importing the Qt-dependent module.

_CAM_SMOOTH_PRESETS = {
    "light":    {"min_cutoff": 0.8,  "beta": 0.05,  "med_kernel": 5},
    "moderate": {"min_cutoff": 0.15, "beta": 0.01,  "med_kernel": 7},
    "heavy":    {"min_cutoff": 0.04, "beta": 0.005, "med_kernel": 9},
}


class _LowPassFilter:
    def __init__(self):
        self.s = None

    def __call__(self, value, alpha):
        if self.s is None:
            self.s = value
        else:
            self.s = alpha * value + (1.0 - alpha) * self.s
        return self.s


class _OneEuroFilter:
    def __init__(self, freq, min_cutoff=0.5, beta=0.007, d_cutoff=1.0):
        import math
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._math = math
        self.x_filt = _LowPassFilter()
        self.dx_filt = _LowPassFilter()

    def _alpha(self, cutoff, freq):
        tau = 1.0 / (2.0 * self._math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x):
        prev = self.x_filt.s
        dx = 0.0 if prev is None else (x - prev) * self.freq
        edx = self.dx_filt(dx, self._alpha(self.d_cutoff, self.freq))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filt(x, self._alpha(cutoff, self.freq))


def _smooth_c2w(c2w, fps=30.0, preset="moderate"):
    """Temporal smoothing of camera-to-world matrices via One Euro filter."""
    from scipy.ndimage import median_filter
    from scipy.spatial.transform import Rotation

    N = c2w.shape[0]
    if N < 3:
        return c2w.copy()

    p = _CAM_SMOOTH_PRESETS.get(preset, _CAM_SMOOTH_PRESETS["moderate"])
    med_k = min(p["med_kernel"], N | 1)
    out = c2w.copy()

    # Smooth translation
    for axis in range(3):
        t = out[:, axis, 3].copy()
        t = median_filter(t, size=med_k)
        filt = _OneEuroFilter(fps, min_cutoff=p["min_cutoff"], beta=p["beta"])
        for i in range(N):
            t[i] = filt(float(t[i]))
        out[:, axis, 3] = t

    # Smooth rotation via quaternions
    R_mats = out[:, :3, :3].copy()
    quats = Rotation.from_matrix(R_mats).as_quat()  # (N, 4) xyzw

    for i in range(1, N):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]

    for c in range(4):
        q = quats[:, c].copy()
        q = median_filter(q, size=med_k)
        filt = _OneEuroFilter(fps, min_cutoff=p["min_cutoff"], beta=p["beta"])
        for i in range(N):
            q[i] = filt(float(q[i]))
        quats[:, c] = q

    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats = quats / np.where(norms > 1e-8, norms, np.ones_like(norms))
    out[:, :3, :3] = Rotation.from_quat(quats).as_matrix()
    return out


def _compute_cam_angvel(R_w2c):
    """Compute cam_angvel from W2C rotation matrices.

    Replicates ``hmr4d.utils.geo_transform.compute_cam_angvel``.

    Parameters
    ----------
    R_w2c : torch.Tensor of shape ``(N, 3, 3)``

    Returns
    -------
    cam_angvel : torch.Tensor of shape ``(N, 6)``
    """
    import torch
    from rotation_utils import matrix_to_rotation_6d

    # R @ R0 = R1, so R = R1 @ R0^T
    rel = R_w2c[1:] @ R_w2c[:-1].transpose(-1, -2)  # (N-1, 3, 3)
    cam_angvel = matrix_to_rotation_6d(rel)           # (N-1, 6)
    # Pad last frame (same convention as GVHMR)
    cam_angvel = torch.cat([cam_angvel, cam_angvel[-1:]], dim=0)  # (N, 6)
    return cam_angvel.float()
