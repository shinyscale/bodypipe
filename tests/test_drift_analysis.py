"""Unit tests for workers.spring_refine.drift_analysis.

All tests use synthetic data — no torch, Qt, or GPU required.
"""

from __future__ import annotations

import numpy as np
import pytest

from workers.spring_refine.drift_analysis import (
    _compute_velocity_bias,
    _gaussian_smooth,
    _lowpass_dc_estimate,
    _measure_xz_drift_cm,
    _recommended_pin_strength,
    _windowed_mean,
    _windowed_stance_median,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_ltv(n: int, bias_x: float = 0.0, bias_z: float = 0.0) -> np.ndarray:
    """Create (N, 3) local_transl_vel with optional DC bias."""
    ltv = np.zeros((n, 3), dtype=np.float64)
    ltv[:, 0] = bias_x
    ltv[:, 2] = bias_z
    return ltv


def _all_stance(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def _no_stance(n: int) -> np.ndarray:
    return np.zeros(n, dtype=bool)


# ---------------------------------------------------------------------------
# 1. Constant bias removed
# ---------------------------------------------------------------------------


class TestConstantBiasRemoved:
    def test_z_bias_removed(self):
        """3mm/frame Z bias → drift reduced >70%."""
        n = 600  # 20 s @ 30 fps
        ltv = _make_ltv(n, bias_z=0.003)
        bias = _compute_velocity_bias(ltv, _all_stance(n), _all_stance(n), fps=30.0, debias_window_sec=2.5)

        corrected_z = ltv[:, 2] - bias[:, 2]
        original_drift = abs(np.sum(ltv[:, 2]))
        corrected_drift = abs(np.sum(corrected_z))
        assert corrected_drift < original_drift * 0.30

    def test_x_bias_removed(self):
        """Lateral bias → removed."""
        n = 600
        ltv = _make_ltv(n, bias_x=0.002)
        bias = _compute_velocity_bias(ltv, _all_stance(n), _all_stance(n), fps=30.0, debias_window_sec=2.5)

        corrected_x = ltv[:, 0] - bias[:, 0]
        original_drift = abs(np.sum(ltv[:, 0]))
        corrected_drift = abs(np.sum(corrected_x))
        assert corrected_drift < original_drift * 0.30


# ---------------------------------------------------------------------------
# 2. Walking velocity preserved
# ---------------------------------------------------------------------------


class TestWalkingVelocityPreserved:
    def test_stride_variability_preserved(self):
        """DC bias + alternating stance/swing → overall drift reduced, stride shape preserved.

        The global-mean estimator subtracts a constant from all frames.
        This removes drift and preserves per-stride velocity variation
        (std dev). For standing/dancing (our primary use case), swing
        velocity averages out and the mean accurately captures bias.
        """
        n = 600
        fps = 30.0
        dc_bias = 0.003

        ltv = np.zeros((n, 3), dtype=np.float64)
        for cycle_start in range(0, n, 60):
            stance_end = min(cycle_start + 30, n)
            ltv[cycle_start:stance_end, 2] = dc_bias
            swing_end = min(cycle_start + 60, n)
            swing_len = swing_end - stance_end
            if swing_len > 0:
                t = np.linspace(0, np.pi, swing_len)
                ltv[stance_end:swing_end, 2] = dc_bias + 0.05 * np.sin(t)

        any_stance = np.zeros(n, dtype=bool)
        bias = _compute_velocity_bias(ltv, any_stance, any_stance, fps=fps, debias_window_sec=2.5)
        corrected_z = ltv[:, 2] - bias[:, 2]

        # Integrated drift (sum of velocity) should be reduced
        original_drift = abs(np.sum(ltv[:, 2]))
        corrected_drift = abs(np.sum(corrected_z))
        assert corrected_drift < original_drift * 0.30

        # Per-stride variability (std) preserved — the mean subtraction
        # doesn't change the shape of the velocity curve
        std_original = np.std(ltv[:, 2])
        std_corrected = np.std(corrected_z)
        assert std_corrected > std_original * 0.95


# ---------------------------------------------------------------------------
# 3. Lateral bias with oscillation
# ---------------------------------------------------------------------------


class TestLateralBiasWithOscillation:
    def test_dc_removed_ac_preserved(self):
        """Lateral DC bias removed, high-freq oscillation preserved."""
        n = 300
        fps = 30.0
        t = np.arange(n) / fps

        ltv = np.zeros((n, 3), dtype=np.float64)
        ltv[:, 0] = 0.002 + 0.01 * np.sin(2 * np.pi * 2.0 * t)

        bias = _compute_velocity_bias(ltv, _all_stance(n), _all_stance(n), fps=fps, debias_window_sec=2.5)
        corrected_x = ltv[:, 0] - bias[:, 0]

        # DC removed
        dc_corrected = abs(np.mean(corrected_x))
        dc_original = abs(np.mean(ltv[:, 0]))
        assert dc_corrected < dc_original * 0.50

        # High-freq oscillation survives
        ac_corrected = np.std(corrected_x)
        ac_original = np.std(ltv[:, 0])
        assert ac_corrected > ac_original * 0.60


# ---------------------------------------------------------------------------
# 4. No stance frames — should still work (global mean doesn't use stance)
# ---------------------------------------------------------------------------


class TestNoStanceFallback:
    def test_no_contacts_still_reduces_drift(self):
        """Zero contact frames → global mean still reduces drift >60%."""
        n = 600
        ltv = _make_ltv(n, bias_z=0.003)
        bias = _compute_velocity_bias(ltv, _no_stance(n), _no_stance(n), fps=30.0, debias_window_sec=2.5)

        corrected_z = ltv[:, 2] - bias[:, 2]
        original_drift = abs(np.sum(ltv[:, 2]))
        corrected_drift = abs(np.sum(corrected_z))
        assert corrected_drift < original_drift * 0.40


# ---------------------------------------------------------------------------
# 5. Short clip
# ---------------------------------------------------------------------------


class TestShortClip:
    def test_very_short_no_crash(self):
        """< 5 frames → no crash."""
        ltv = _make_ltv(3, bias_z=0.01)
        bias = _compute_velocity_bias(ltv, _all_stance(3), _all_stance(3), fps=30.0, debias_window_sec=2.5)
        assert bias.shape == (3, 3)

    def test_short_clip_uses_global_mean(self):
        """Short clip (< 2× window) uses global mean exactly."""
        n = 30  # 1 second — well below 2 × 2.5s window
        ltv = _make_ltv(n, bias_z=0.005)
        bias = _compute_velocity_bias(ltv, _all_stance(n), _all_stance(n), fps=30.0, debias_window_sec=2.5)
        np.testing.assert_allclose(bias[:, 2], 0.005, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. Y-axis untouched
# ---------------------------------------------------------------------------


class TestYAxisUntouched:
    def test_vertical_velocity_never_modified(self):
        """Y channel of bias is always zero."""
        n = 200
        ltv = np.random.randn(n, 3) * 0.01
        any_stance = np.random.rand(n) > 0.5
        double_support = np.random.rand(n) > 0.7
        bias = _compute_velocity_bias(ltv, any_stance, double_support, fps=30.0, debias_window_sec=2.5)
        np.testing.assert_array_equal(bias[:, 1], 0.0)


# ---------------------------------------------------------------------------
# 7. Drift measurement
# ---------------------------------------------------------------------------


class TestDriftMeasurement:
    def test_measure_xz_drift(self):
        transl = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]])
        assert abs(_measure_xz_drift_cm(transl) - 100.0) < 0.1

    def test_measure_xz_drift_diagonal(self):
        transl = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.4]])
        assert abs(_measure_xz_drift_cm(transl) - 50.0) < 0.1


# ---------------------------------------------------------------------------
# 8. Pin strength mapping
# ---------------------------------------------------------------------------


class TestPinStrength:
    def test_low_drift_full_pin(self):
        assert _recommended_pin_strength(5.0) == 1.0
        assert _recommended_pin_strength(19.9) == 1.0

    def test_high_drift_no_pin(self):
        assert _recommended_pin_strength(51.0) == 0.0
        assert _recommended_pin_strength(100.0) == 0.0

    def test_mid_drift_ramp(self):
        assert _recommended_pin_strength(20.0) == 1.0
        assert abs(_recommended_pin_strength(35.0) - 0.65) < 0.01
        assert abs(_recommended_pin_strength(50.0) - 0.3) < 0.01

    def test_boundary_values(self):
        assert _recommended_pin_strength(0.0) == 1.0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_lowpass_dc_extracts_mean(self):
        signal = np.full(100, 3.5)
        result = _lowpass_dc_estimate(signal, fps=30.0, cutoff=0.15)
        np.testing.assert_allclose(result, 3.5, atol=0.01)

    def test_lowpass_dc_strips_high_freq(self):
        n = 300
        fps = 30.0
        t = np.arange(n) / fps
        dc = 2.0
        signal = dc + 1.0 * np.sin(2 * np.pi * 5.0 * t)
        result = _lowpass_dc_estimate(signal, fps=fps, cutoff=0.15)
        np.testing.assert_allclose(np.mean(result), dc, atol=0.2)
        assert np.std(result) < np.std(signal) * 0.3

    def test_gaussian_smooth_short(self):
        signal = np.array([1.0, 2.0])
        result = _gaussian_smooth(signal, sigma_frames=5)
        np.testing.assert_array_equal(result, signal)

    def test_windowed_mean_constant(self):
        values = np.ones(50) * 3.0
        result = _windowed_mean(values, half_window=10)
        np.testing.assert_allclose(result, 3.0)

    def test_windowed_mean_captures_dc(self):
        """Windowed mean of DC + oscillation preserves DC."""
        n = 300
        t = np.arange(n, dtype=np.float64)
        values = 2.0 + 0.5 * np.sin(2 * np.pi * t / 30)
        result = _windowed_mean(values, half_window=75)  # 2.5s window
        # Interior points should be close to DC=2.0
        np.testing.assert_allclose(result[100:200], 2.0, atol=0.1)

    def test_windowed_stance_median_uniform(self):
        values = np.ones(50) * 3.0
        mask = np.ones(50, dtype=bool)
        result = _windowed_stance_median(values, mask, half_window=10)
        np.testing.assert_allclose(result, 3.0)
