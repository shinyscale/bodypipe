"""Unit tests for pelvis-translation quality metrics.

Pure numpy — no Qt, no torch, no pipeline state.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tools.eval_pelvis_mae import (
    OPENCAP_PELVIS_MAE_CM,
    compute_proxy_metrics,
    compute_reference_metrics,
    extract_pelvis_transl,
    format_report,
)


# ------------------------------------------------------------------
# extract_pelvis_transl
# ------------------------------------------------------------------


def test_extract_transl_prefers_transl_world():
    params = {
        "transl": np.zeros((10, 3)),
        "transl_world": np.ones((10, 3)),
    }
    t = extract_pelvis_transl(params)
    assert np.allclose(t, 1.0)


def test_extract_transl_falls_back_to_transl():
    params = {"transl": np.ones((5, 3)) * 2.0}
    t = extract_pelvis_transl(params)
    assert np.allclose(t, 2.0)


def test_extract_transl_raises_on_missing():
    with pytest.raises(KeyError):
        extract_pelvis_transl({"body_pose": np.zeros((5, 63))})


# ------------------------------------------------------------------
# compute_proxy_metrics — stationary subject
# ------------------------------------------------------------------


def _stationary_transl(n: int = 100, noise_std: float = 0.001) -> np.ndarray:
    """Stationary subject at origin with small noise."""
    rng = np.random.default_rng(42)
    t = np.zeros((n, 3), dtype=np.float64)
    t[:, 1] = 0.9  # ~90 cm pelvis height
    t += rng.normal(scale=noise_std, size=(n, 3))
    return t


def test_stationary_low_drift():
    t = _stationary_transl(200, noise_std=0.0005)
    m = compute_proxy_metrics(t, fps=30.0)
    assert m["pelvis_drift_xz_total_cm"] < 1.0
    assert m["pelvis_drift_xz_rate_cm_s"] < 0.5
    assert m["pelvis_height_std_cm"] < 0.2


def test_stationary_low_jitter():
    t = _stationary_transl(200, noise_std=0.0005)
    m = compute_proxy_metrics(t, fps=30.0)
    assert m["pelvis_jitter_m_s2"] < 5.0


# ------------------------------------------------------------------
# compute_proxy_metrics — walking subject
# ------------------------------------------------------------------


def _walking_transl(n: int = 300, speed_m_s: float = 1.4) -> np.ndarray:
    """Walking subject along +X at constant speed, slight vertical bob."""
    fps = 30.0
    dt = 1.0 / fps
    t = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        t[i, 0] = speed_m_s * i * dt  # forward
        t[i, 1] = 0.9 + 0.01 * np.sin(2 * np.pi * 2.0 * i * dt)  # ~2 Hz bob
    return t


def test_walking_drift_proportional_to_speed():
    t = _walking_transl(300, speed_m_s=1.4)
    m = compute_proxy_metrics(t, fps=30.0)
    # Walking 1.4 m/s for ~10s → ~14m XZ displacement
    assert m["pelvis_drift_xz_total_cm"] > 1000  # > 10m
    assert m["pelvis_path_xz_cm"] > 1000


def test_walking_smooth_trajectory():
    t = _walking_transl(300, speed_m_s=1.4)
    m = compute_proxy_metrics(t, fps=30.0)
    # Constant speed → low jitter (only from the sinusoidal bob)
    assert m["pelvis_jitter_m_s2"] < 20.0


# ------------------------------------------------------------------
# compute_proxy_metrics — edge cases
# ------------------------------------------------------------------


def test_short_sequence():
    t = np.array([[0, 0.9, 0], [0.01, 0.9, 0]], dtype=np.float64)
    m = compute_proxy_metrics(t, fps=30.0)
    assert m["num_frames"] == 2
    assert m["pelvis_jitter_cm"] == 0.0
    assert m["pelvis_ldlj"] == 0.0


def test_single_frame():
    t = np.array([[0, 0.9, 0]], dtype=np.float64)
    m = compute_proxy_metrics(t, fps=30.0)
    assert m["num_frames"] == 1
    assert m["pelvis_drift_xz_total_cm"] == 0.0


# ------------------------------------------------------------------
# compute_proxy_metrics — with ankle positions (stance metrics)
# ------------------------------------------------------------------


def test_stance_metrics_with_ankles():
    n = 100
    fps = 30.0
    t = _stationary_transl(n, noise_std=0.001)
    # Fake ankles near ground level, low velocity → should detect as stance
    ankles = np.zeros((n, 2, 3), dtype=np.float64)
    ankles[:, 0, 1] = 0.03  # left ankle ~3cm above min
    ankles[:, 1, 1] = 0.03  # right ankle
    m = compute_proxy_metrics(t, fps, ankle_positions=ankles)
    assert m["stance_frames"] is not None
    assert m["stance_frames"] > 0
    assert m["pelvis_stance_drift_cm"] is not None


def test_stance_metrics_none_without_ankles():
    t = _stationary_transl(50)
    m = compute_proxy_metrics(t, fps=30.0, ankle_positions=None)
    assert m["stance_frames"] is None
    assert m["pelvis_stance_drift_cm"] is None


# ------------------------------------------------------------------
# compute_reference_metrics
# ------------------------------------------------------------------


def test_reference_exact_match():
    t = _walking_transl(100)
    m = compute_reference_metrics(t, t)
    assert m["pelvis_mae_cm"] == 0.0
    assert m["pelvis_mae_xz_cm"] == 0.0
    assert m["pelvis_mae_y_cm"] == 0.0
    assert m["meets_opencap_bar"] is True


def test_reference_constant_offset():
    t = _walking_transl(100)
    # 2 cm offset in X
    ref = t.copy()
    ref[:, 0] += 0.02
    m = compute_reference_metrics(t, ref)
    assert abs(m["pelvis_mae_cm"] - 2.0) < 0.1  # ~2 cm
    assert m["meets_opencap_bar"] is True  # 2 < 3.4


def test_reference_fails_opencap_bar():
    t = _walking_transl(100)
    ref = t.copy()
    ref[:, 0] += 0.05  # 5 cm offset
    m = compute_reference_metrics(t, ref)
    assert m["pelvis_mae_cm"] > 3.4
    assert m["meets_opencap_bar"] is False


def test_reference_length_mismatch():
    t = _walking_transl(100)
    ref = _walking_transl(80)
    m = compute_reference_metrics(t, ref)
    assert m["reference_frames_used"] == 80


# ------------------------------------------------------------------
# format_report
# ------------------------------------------------------------------


def test_format_report_empty():
    assert "No results" in format_report([])


def test_format_report_renders():
    t = _stationary_transl(60)
    m = compute_proxy_metrics(t, fps=30.0)
    result = {"stage": "test", "file": "test.pt", **m}
    report = format_report([result])
    assert "PELVIS TRANSLATION QUALITY REPORT" in report
    assert "test" in report
    assert "3.4" in report  # OpenCap bar


def test_format_report_with_reference():
    t = _walking_transl(60)
    ref = t.copy()
    ref[:, 0] += 0.02
    proxy = compute_proxy_metrics(t, fps=30.0)
    refm = compute_reference_metrics(t, ref)
    result = {"stage": "test", "file": "test.pt", **proxy, **refm}
    report = format_report([result])
    assert "Reference metrics" in report
    assert "PASS" in report


def test_format_report_error_entry():
    results = [{"stage": "broken", "file": "x.pt", "error": "boom"}]
    report = format_report(results)
    assert "ERROR" in report
    assert "boom" in report


# ------------------------------------------------------------------
# Regression: noisy GVHMR-like output
# ------------------------------------------------------------------


def test_noisy_gvhmr_proxy_metrics():
    """Simulate a noisy GVHMR output and verify metrics are in sane ranges."""
    rng = np.random.default_rng(123)
    n = 300
    fps = 30.0
    # Walking forward with GVHMR-like noise
    t = _walking_transl(n, speed_m_s=1.2)
    t += rng.normal(scale=0.005, size=(n, 3))  # 5mm noise
    m = compute_proxy_metrics(t, fps)

    # Sanity: should have measured something
    assert m["pelvis_drift_xz_total_cm"] > 0
    assert m["pelvis_height_std_cm"] > 0
    assert m["pelvis_jitter_m_s2"] > 0
    assert isinstance(m["pelvis_ldlj"], float)
    assert not math.isnan(m["pelvis_ldlj"])
