"""Unit tests for the spring-refine filter math.

Pure numpy/scipy — no Qt, no torch, no pipeline state. Exercises the
rotation helpers, the IIR filter, the gain tables, and the top-level
``refine_body_sequence`` / ``run_spring_refine`` entry points.
"""

from __future__ import annotations

import numpy as np
import pytest

from workers.spring_refine.filter import critically_damped_filter
from workers.spring_refine.gains import (
    _BASE_BODY_GAINS,
    _PRESET_SCALES,
    get_preset_gains,
)
from workers.spring_refine.rotations import (
    aa_sequence_to_log,
    log_sequence_to_aa,
)
from workers.spring_refine.runner import (
    refine_body_sequence,
    run_spring_refine,
)


# ----------------------------------------------------------------------
# Rotation helpers
# ----------------------------------------------------------------------


def test_log_roundtrip_identity():
    aa = np.zeros((100, 21, 3), dtype=np.float32)
    log = aa_sequence_to_log(aa)
    assert log.shape == aa.shape
    assert np.allclose(log, 0.0, atol=1e-7)
    aa_back = log_sequence_to_aa(log)
    assert np.allclose(aa_back, 0.0, atol=1e-7)


def test_log_roundtrip_random_small_angles():
    rng = np.random.default_rng(42)
    aa = rng.uniform(-0.5, 0.5, size=(64, 21, 3)).astype(np.float32)
    log = aa_sequence_to_log(aa)
    aa_back = log_sequence_to_aa(log)
    assert np.max(np.abs(aa - aa_back)) < 1e-5


def test_log_roundtrip_large_angles():
    rng = np.random.default_rng(7)
    axis = rng.normal(size=(40, 5, 3)).astype(np.float32)
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    angles = rng.uniform(0.1, 2.8, size=(40, 5, 1)).astype(np.float32)
    aa = axis * angles
    log = aa_sequence_to_log(aa)
    aa_back = log_sequence_to_aa(log)
    assert np.max(np.abs(aa - aa_back)) < 1e-4


def test_hemisphere_continuity():
    # Build a smoothly-varying rotation sequence around Z (angles stay well
    # below pi so the log-map itself is continuous), then simulate a
    # pathological source that flips the quaternion sign on every other
    # frame. The hemisphere-continuity pass inside aa_sequence_to_log
    # should unflip them and leave the log smooth.
    from scipy.spatial.transform import Rotation

    T = 120
    angles = np.linspace(0.0, 2.0, T, dtype=np.float32)  # [0, 2] rad
    aa = np.zeros((T, 1, 3), dtype=np.float32)
    aa[:, 0, 2] = angles
    # Reference log (no flipping) for comparison
    ref_log = aa_sequence_to_log(aa)
    ref_diffs = np.linalg.norm(ref_log[1:] - ref_log[:-1], axis=-1)
    assert ref_diffs.max() < 0.1  # smooth input stays smooth

    # Now synthesize a sign-flipped sequence by going via quat and
    # negating alternate frames, then pushing it back as axis-angle so we
    # exercise the continuity-fixup path inside aa_sequence_to_log.
    q = Rotation.from_rotvec(aa.reshape(T, 3)).as_quat()
    q[1::2] *= -1.0
    aa_flipped = Rotation.from_quat(q).as_rotvec().reshape(T, 1, 3).astype(np.float32)
    log_flipped = aa_sequence_to_log(aa_flipped)
    diffs = np.linalg.norm(log_flipped[1:] - log_flipped[:-1], axis=-1)
    assert diffs.max() < 0.5


# ----------------------------------------------------------------------
# Filter behaviour
# ----------------------------------------------------------------------


def test_filter_identity_sequence():
    target = np.full((60, 21, 3), 0.3, dtype=np.float32)
    kp, kv, _, _ = get_preset_gains("moderate")
    out = critically_damped_filter(target, kp, kv, dt=1 / 30.0, pad_frames=0)
    assert np.allclose(out, target, atol=1e-6)


def test_filter_step_response_underdamped():
    T = 120
    fps = 30.0
    dt = 1.0 / fps
    target = np.zeros((T, 1, 3), dtype=np.float32)
    target[20:, 0, 0] = 1.0  # step
    kp = np.array([300.0], dtype=np.float32)
    kv = np.array([15.0], dtype=np.float32)
    # Single-pass (forward-only) filter — intentional underdamping for
    # weight injection: the overshoot IS the follow-through.
    out = critically_damped_filter(
        target, kp, kv, dt=dt, zero_phase=False, pad_frames=0
    )
    x = out[:, 0, 0]
    # Underdamped overshoot should be visible but bounded (< 30%).
    assert x.max() <= 1.30
    # Converges to ~95% of the target within ~0.3s after the step.
    steady_start = 20 + int(0.3 * fps)
    assert x[steady_start] > 0.9


def test_filter_frame_count_preserved():
    rng = np.random.default_rng(123)
    for T in (1, 7, 30, 257):
        target = rng.normal(scale=0.1, size=(T, 21, 3)).astype(np.float32)
        kp, kv, _, _ = get_preset_gains("moderate")
        out = critically_damped_filter(target, kp, kv, dt=1 / 30.0, pad_frames=5)
        assert out.shape == target.shape


# ----------------------------------------------------------------------
# Gain table
# ----------------------------------------------------------------------


def test_gain_table_shape():
    assert _BASE_BODY_GAINS.shape == (21, 2)
    assert (_BASE_BODY_GAINS > 0).all()
    kp = _BASE_BODY_GAINS[:, 0]
    kv = _BASE_BODY_GAINS[:, 1]
    zeta = kv / (2.0 * np.sqrt(kp))
    # Underdamped for weight injection: zeta ~0.33–0.50 across all joints.
    assert (zeta > 0.30).all() and (zeta < 0.55).all()


def test_preset_scaling_preserves_damping_ratio():
    ref_kp, ref_kv, _, _ = get_preset_gains("moderate")
    ref_zeta = ref_kv / (2.0 * np.sqrt(ref_kp))
    for preset in _PRESET_SCALES:
        kp, kv, _, _ = get_preset_gains(preset)
        zeta = kv / (2.0 * np.sqrt(kp))
        rel_err = np.abs(zeta - ref_zeta) / ref_zeta
        assert rel_err.max() < 0.05, f"preset {preset} damping ratio drifted"


def test_get_preset_gains_rejects_unknown():
    with pytest.raises(ValueError):
        get_preset_gains("nope")


# ----------------------------------------------------------------------
# Top-level refine entry points
# ----------------------------------------------------------------------


def test_refine_body_sequence_shapes_flat():
    rng = np.random.default_rng(0)
    N = 50
    bp = rng.normal(scale=0.1, size=(N, 63)).astype(np.float32)
    go = rng.normal(scale=0.1, size=(N, 3)).astype(np.float32)
    bp_f, go_f = refine_body_sequence(bp, go, fps=30.0, preset="moderate")
    assert bp_f.shape == bp.shape
    assert go_f.shape == go.shape
    assert bp_f.dtype == np.float32


def test_refine_body_sequence_shapes_jointed():
    rng = np.random.default_rng(1)
    N = 25
    bp = rng.normal(scale=0.1, size=(N, 21, 3)).astype(np.float32)
    go = rng.normal(scale=0.1, size=(N, 3)).astype(np.float32)
    bp_f, go_f = refine_body_sequence(bp, go, fps=30.0, preset="light")
    assert bp_f.shape == (N, 21, 3)
    assert go_f.shape == (N, 3)


def test_run_spring_refine_missing_key():
    out, ok = run_spring_refine({"transl": np.zeros((10, 3))})
    assert ok is False
    assert out == {"transl": out["transl"]}


def test_run_spring_refine_happy_path():
    N = 60
    rng = np.random.default_rng(99)
    params = {
        "body_pose": rng.normal(scale=0.1, size=(N, 63)).astype(np.float32),
        "global_orient": rng.normal(scale=0.05, size=(N, 3)).astype(np.float32),
        "transl": np.zeros((N, 3), dtype=np.float32),
        "num_frames": N,
        "source": "hybrid",
    }
    refined, ok = run_spring_refine(params, preset="moderate", fps=30.0)
    assert ok is True
    assert refined["source"] == "spring_refined"
    assert refined["num_frames"] == N
    assert refined["body_pose"].shape == (N, 63)
    assert refined["global_orient"].shape == (N, 3)
    # transl passed through unchanged
    assert np.allclose(refined["transl"], params["transl"])
    # Output should differ from input (filter did something)
    assert not np.allclose(refined["body_pose"], params["body_pose"])


def test_run_spring_refine_progress_callback():
    N = 30
    params = {
        "body_pose": np.zeros((N, 63), dtype=np.float32),
        "global_orient": np.zeros((N, 3), dtype=np.float32),
        "transl": np.zeros((N, 3), dtype=np.float32),
    }
    calls: list[float] = []
    refined, ok = run_spring_refine(
        params, preset="moderate", fps=30.0, progress_cb=calls.append
    )
    assert ok is True
    assert calls[0] == pytest.approx(0.1)
    assert calls[-1] == pytest.approx(1.0)
