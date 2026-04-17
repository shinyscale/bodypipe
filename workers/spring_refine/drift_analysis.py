"""Stance-gated velocity debiasing for GVHMR network-sourced drift.

GVHMR's ``local_transl_vel`` predictions carry a DC bias (~2.5 mm/frame)
that integrates to >1 m of horizontal drift over 20 seconds. Camera
stabilize can't fix this because it passes ``local_transl_vel`` through
unchanged.

This module measures the bias during foot-contact frames (when horizontal
velocity should be near-zero) and subtracts it before re-deriving world
grounding. Inserted between camera_stabilize and spring_refine in the
pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def analyze_and_correct_drift(
    pt_path: Path,
    pre_correction_params: dict,
    fps: float = 30.0,
    cam_smooth_preset: str = "moderate",
    debias_window_sec: float = 2.5,
) -> tuple[dict | None, dict]:
    """Stance-gated velocity debiasing for network-sourced drift.

    Parameters
    ----------
    pt_path : Path to ``hmr4d_results.pt`` (must contain
        ``net_outputs.decode_dict``).
    pre_correction_params : Post-camera-stabilize world params dict with
        ``body_pose``, ``global_orient``, ``transl`` keys.
    fps : Frame rate.
    cam_smooth_preset : SLAM smoothing preset (passed through for
        cam_angvel re-derivation).
    debias_window_sec : Half-window size in seconds for the windowed
        median bias estimator.

    Returns
    -------
    ``(corrected_params, diagnostics)`` where ``corrected_params`` is
    ``None`` if no correction was needed or possible.
    """
    import torch

    diag: dict = {
        "correction_applied": False,
        "drift_source": "unknown",
    }

    pt_path = Path(pt_path)
    if not pt_path.is_file():
        return None, diag

    # --- Load decode_dict ---
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    net_outputs = data.get("net_outputs")
    if not isinstance(net_outputs, dict):
        return None, diag
    decode_dict = net_outputs.get("decode_dict")
    if not isinstance(decode_dict, dict):
        return None, diag

    global_orient_gv = decode_dict.get("global_orient_gv")
    local_transl_vel = decode_dict.get("local_transl_vel")
    global_orient_c = decode_dict.get("global_orient")

    if any(v is None for v in [global_orient_gv, local_transl_vel, global_orient_c]):
        return None, diag

    # --- Shape handling (match camera_stabilize conventions) ---
    had_batch = global_orient_gv.ndim >= 3

    # Squeeze to (L, 3) for numpy processing
    ltv_np = local_transl_vel.cpu().numpy()
    if ltv_np.ndim == 3:
        ltv_np = ltv_np[0]  # (L, 3)

    N = ltv_np.shape[0]

    # --- Short clip guard ---
    if N < 5:
        diag["drift_source"] = "minimal"
        return None, diag

    # --- Contact detection on post-stabilize trajectory ---
    body_pose = np.asarray(pre_correction_params["body_pose"])
    global_orient = np.asarray(pre_correction_params["global_orient"])
    transl = np.asarray(pre_correction_params["transl"])

    from .contact import detect_contacts

    contacts, _ = detect_contacts(body_pose, global_orient, transl, fps)

    any_stance = contacts[:, 0] | contacts[:, 1]
    double_support = contacts[:, 0] & contacts[:, 1]

    stance_frac = float(any_stance.sum()) / N
    diag["stance_fraction"] = round(stance_frac, 3)

    # --- Pre-correction drift measurement ---
    pre_drift_xz = _measure_xz_drift_cm(transl)
    diag["pre_correction_drift_xz_cm"] = round(pre_drift_xz, 1)

    # --- Compute velocity bias ---
    bias = _compute_velocity_bias(
        ltv_np, any_stance, double_support, fps, debias_window_sec
    )
    diag["local_vel_bias_mean_mm"] = [
        round(float(np.mean(np.abs(bias[:, i]))) * 1000, 2) for i in range(3)
    ]

    # Check if bias is significant enough to correct
    mean_bias_xz = float(np.sqrt(np.mean(bias[:, 0]) ** 2 + np.mean(bias[:, 2]) ** 2))
    if mean_bias_xz < 0.0005:  # < 0.5 mm/frame → negligible
        diag["drift_source"] = "minimal"
        diag["correction_applied"] = False
        return None, diag

    # --- Subtract bias (X and Z only, Y untouched) ---
    ltv_corrected = ltv_np.copy()
    ltv_corrected[:, 0] -= bias[:, 0]
    ltv_corrected[:, 2] -= bias[:, 2]

    # --- Re-derive world params with corrected velocity ---
    from .camera_stabilize import _compute_cam_angvel, _load_slam, _smooth_c2w

    slam_w2c = _load_slam(pt_path, None)
    if slam_w2c is None:
        # Fallback: identity cam_angvel (static camera assumption)
        from rotation_utils import matrix_to_rotation_6d

        identity_6d = matrix_to_rotation_6d(torch.eye(3).unsqueeze(0))  # (1, 6)
        cam_angvel = identity_6d.expand(N, -1).float()
    else:
        slam_c2w = np.linalg.inv(slam_w2c)
        slam_c2w_smooth = _smooth_c2w(slam_c2w, fps=fps, preset=cam_smooth_preset)
        slam_w2c_smooth = np.linalg.inv(slam_c2w_smooth)
        R_w2c = torch.from_numpy(slam_w2c_smooth[:, :3, :3]).float()
        cam_angvel = _compute_cam_angvel(R_w2c)
        # Match frame count
        N_slam = cam_angvel.shape[0]
        if N_slam < N:
            pad = cam_angvel[-1:].expand(N - N_slam, -1)
            cam_angvel = torch.cat([cam_angvel, pad], dim=0)
        elif N_slam > N:
            cam_angvel = cam_angvel[:N]

    # Restore batch dimensions for get_smpl_params_w_Rt_v2
    ltv_t = torch.from_numpy(ltv_corrected).float().unsqueeze(0)  # (1, L, 3)
    if cam_angvel.ndim == 2:
        cam_angvel = cam_angvel.unsqueeze(0)

    if had_batch:
        go_gv = global_orient_gv
        go_c = global_orient_c
    else:
        go_gv = global_orient_gv.unsqueeze(0) if global_orient_gv.ndim == 2 else global_orient_gv
        go_c = global_orient_c.unsqueeze(0) if global_orient_c.ndim == 2 else global_orient_c

    from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import get_smpl_params_w_Rt_v2

    result = get_smpl_params_w_Rt_v2(
        global_orient_gv=go_gv,
        local_transl_vel=ltv_t,
        global_orient_c=go_c,
        cam_angvel=cam_angvel,
    )

    go_new = result["global_orient"][0].cpu().numpy().astype(np.float32)
    tr_new = result["transl"][0].cpu().numpy().astype(np.float32)

    # --- Post-correction drift measurement ---
    post_drift_xz = _measure_xz_drift_cm(tr_new)
    diag["post_correction_drift_xz_cm"] = round(post_drift_xz, 1)

    reduction = 0.0
    if pre_drift_xz > 1.0:
        reduction = (1.0 - post_drift_xz / pre_drift_xz) * 100
    diag["drift_reduction_percent"] = round(max(0.0, reduction), 1)

    # --- Classify drift source ---
    if pre_drift_xz < 20:
        diag["drift_source"] = "minimal"
    elif reduction > 30:
        diag["drift_source"] = "network"
    else:
        diag["drift_source"] = "camera"

    # --- Pin strength recommendation ---
    pin_strength = _recommended_pin_strength(post_drift_xz)
    diag["recommended_pin_strength"] = round(pin_strength, 2)

    # Only apply if we actually improved things
    if post_drift_xz < pre_drift_xz:
        diag["correction_applied"] = True
        logger.info(
            "drift_analysis: %.1f cm → %.1f cm (%.0f%% reduction, source=%s)",
            pre_drift_xz,
            post_drift_xz,
            reduction,
            diag["drift_source"],
        )
        return {"global_orient": go_new, "transl": tr_new}, diag
    else:
        diag["correction_applied"] = False
        logger.info(
            "drift_analysis: no improvement (%.1f cm → %.1f cm), skipping",
            pre_drift_xz,
            post_drift_xz,
        )
        return None, diag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _measure_xz_drift_cm(transl: np.ndarray) -> float:
    """XZ displacement from first to last frame, in cm."""
    delta = transl[-1] - transl[0]
    return float(np.sqrt(delta[0] ** 2 + delta[2] ** 2)) * 100


def _compute_velocity_bias(
    ltv: np.ndarray,
    any_stance: np.ndarray,
    double_support: np.ndarray,
    fps: float,
    debias_window_sec: float,
) -> np.ndarray:
    """Compute per-frame velocity bias estimate.

    Uses global mean as the primary estimator: over a full clip,
    oscillatory real motion (forward during swing, backward during
    stance) cancels out, leaving only the DC bias. Stance-gated
    median underestimates the bias because real pelvis motion during
    stance (weight transfer over the planted foot) is nonzero.

    For clips longer than 2× the debias window, uses a windowed mean
    so the bias estimate can track slow changes.

    Returns ``(N, 3)`` bias array. Y channel is always zero.
    """
    N = ltv.shape[0]
    bias = np.zeros((N, 3), dtype=np.float64)
    clip_duration_sec = N / fps

    if clip_duration_sec > debias_window_sec * 2:
        # Long clip: windowed mean allows bias to vary over time
        half_window = max(1, int(debias_window_sec * fps))
        bias[:, 0] = _windowed_mean(ltv[:, 0], half_window)
        bias[:, 2] = _windowed_mean(ltv[:, 2], half_window)
        # Smooth to remove noise
        sigma_frames = max(1, int(fps * 1.0))
        bias[:, 0] = _gaussian_smooth(bias[:, 0], sigma_frames)
        bias[:, 2] = _gaussian_smooth(bias[:, 2], sigma_frames)
    else:
        # Short clip: global mean is the best estimate
        bias[:, 0] = float(np.mean(ltv[:, 0]))
        bias[:, 2] = float(np.mean(ltv[:, 2]))

    # Y is always zero — no vertical debiasing
    return bias


def _windowed_mean(values: np.ndarray, half_window: int) -> np.ndarray:
    """Per-frame windowed mean of ``values``."""
    N = len(values)
    result = np.zeros(N, dtype=np.float64)
    for t in range(N):
        lo = max(0, t - half_window)
        hi = min(N, t + half_window + 1)
        result[t] = np.mean(values[lo:hi])
    return result


def _windowed_stance_median(
    values: np.ndarray,
    stance_mask: np.ndarray,
    half_window: int,
) -> np.ndarray:
    """Per-frame windowed median of ``values`` at stance frames.

    Falls back to the global mean when no stance frames exist.
    """
    N = len(values)
    result = np.zeros(N, dtype=np.float64)

    if stance_mask.any():
        global_fallback = float(np.median(values[stance_mask]))
    else:
        global_fallback = float(np.mean(values))

    for t in range(N):
        lo = max(0, t - half_window)
        hi = min(N, t + half_window + 1)
        window_mask = stance_mask[lo:hi]
        if window_mask.any():
            result[t] = np.median(values[lo:hi][window_mask])
        else:
            result[t] = global_fallback

    return result


def _lowpass_dc_estimate(
    signal: np.ndarray, fps: float, cutoff: float = 0.15
) -> np.ndarray:
    """Extract DC / very-low-frequency component via Butterworth lowpass."""
    from scipy.signal import butter, filtfilt

    N = len(signal)
    if N < 10:
        return np.full(N, np.mean(signal))

    nyquist = fps / 2.0
    if cutoff >= nyquist:
        return np.full(N, np.mean(signal))

    b, a = butter(2, cutoff / nyquist, btype="low")
    pad_len = min(3 * max(len(a), len(b)), N - 1)
    try:
        return filtfilt(b, a, signal, padlen=pad_len).astype(np.float64)
    except Exception:
        return np.full(N, np.mean(signal))


def _gaussian_smooth(signal: np.ndarray, sigma_frames: int) -> np.ndarray:
    """1-D Gaussian smoothing."""
    from scipy.ndimage import gaussian_filter1d

    if len(signal) < 3 or sigma_frames < 1:
        return signal
    return gaussian_filter1d(signal, sigma=sigma_frames, mode="nearest")


def _recommended_pin_strength(residual_drift_cm: float) -> float:
    """Map residual XZ drift to recommended foot-pin strength.

    - < 20 cm residual → 1.0 (full pin)
    - 20–50 cm → linear ramp 1.0 → 0.3
    - > 50 cm → 0.0 (skip pin entirely)
    """
    if residual_drift_cm < 20:
        return 1.0
    elif residual_drift_cm > 50:
        return 0.0
    else:
        t = (residual_drift_cm - 20) / 30  # 0..1
        return 1.0 - t * 0.7  # 1.0..0.3


def write_drift_diagnostics(diag: dict, out_path: Path) -> None:
    """Write drift analysis diagnostics to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(diag, f, indent=2)
