"""Unit tests for spring-refine v2 foot pinning.

Covers headless FK, contact detection, anchor tracking, pin offset
computation, the end-to-end ``apply_foot_pin`` pass, runner integration,
and metrics-JSON verdict classification. Pure numpy/scipy — no Qt, no
torch, no pipeline state.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from workers.spring_refine.fk import forward_kinematics_body
from workers.spring_refine.contact import (
    detect_contacts,
    track_stance_anchors,
)
from workers.spring_refine.footpin import apply_foot_pin, compute_pin_offset
from workers.spring_refine.metrics import (
    classify_verdict,
    write_spring_metrics_json,
)
from workers.spring_refine.runner import run_spring_refine


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _zero_params(n: int, transl: np.ndarray | None = None) -> dict:
    """Rest pose for n frames. Optional translation override."""
    if transl is None:
        transl = np.zeros((n, 3), dtype=np.float32)
    return {
        "num_frames": n,
        "body_pose": np.zeros((n, 21, 3), dtype=np.float32),
        "global_orient": np.zeros((n, 3), dtype=np.float32),
        "transl": transl.astype(np.float32),
    }


# ----------------------------------------------------------------------
# Forward kinematics
# ----------------------------------------------------------------------


def test_fk_matches_mesh_viewport():
    """Headless FK must agree with the Qt-linked mesh_viewport FK on a fixed pose."""
    from views.mesh_viewport import forward_kinematics as mv_fk

    rng = np.random.default_rng(42)
    n = 3
    body_pose = rng.normal(scale=0.2, size=(n, 21, 3)).astype(np.float32)
    global_orient = rng.normal(scale=0.1, size=(n, 3)).astype(np.float32)
    transl = rng.normal(scale=0.5, size=(n, 3)).astype(np.float32)
    params = {
        "body_pose": body_pose,
        "global_orient": global_orient,
        "transl": transl,
    }

    ours = forward_kinematics_body(body_pose, global_orient, transl)
    for frame in range(n):
        theirs = mv_fk(params, frame)[:22]
        assert np.max(np.abs(ours[frame] - theirs)) < 1e-4


def test_fk_identity_pose_foot_height():
    n = 2
    joints = forward_kinematics_body(
        np.zeros((n, 21, 3), dtype=np.float32),
        np.zeros((n, 3), dtype=np.float32),
        np.zeros((n, 3), dtype=np.float32),
    )
    # Y of L_Foot vs R_Foot should match (symmetric rest pose).
    assert abs(joints[0, 10, 1] - joints[0, 11, 1]) < 5e-2


# ----------------------------------------------------------------------
# Contact detection
# ----------------------------------------------------------------------


def test_detect_contacts_stationary_standing():
    n = 30
    contacts, toes = detect_contacts(
        np.zeros((n, 21, 3), dtype=np.float32),
        np.zeros((n, 3), dtype=np.float32),
        np.zeros((n, 3), dtype=np.float32),
        fps=30.0,
    )
    # Fully static -> every frame, both feet, contact = True.
    assert contacts.shape == (n, 2)
    assert contacts.all()


def test_detect_contacts_waving_arms():
    n = 30
    rng = np.random.default_rng(0)
    body_pose = np.zeros((n, 21, 3), dtype=np.float32)
    # Joint 15 (L_Shoulder) moves, feet stay put.
    body_pose[:, 15, 0] = np.linspace(0.0, 1.0, n, dtype=np.float32)
    contacts, _ = detect_contacts(
        body_pose,
        np.zeros((n, 3), dtype=np.float32),
        np.zeros((n, 3), dtype=np.float32),
        fps=30.0,
    )
    # Feet static → all contact True even while arms swing.
    assert contacts.all()


def test_detect_contacts_walking_alternating():
    """Synthetic ankle trajectory that alternates high/low; contacts should track."""
    # Bypass FK and call detect_foot_contacts directly through the same
    # path: build joint trajectories where the L_Ankle lifts first half
    # and R_Ankle lifts second half.
    from workers.physics.evaluate import detect_foot_contacts

    n = 40
    ankles = np.zeros((n, 2, 3), dtype=np.float64)
    # L lifted for first half, R lifted for second half.
    ankles[:20, 0, 1] = 0.30  # L high
    ankles[20:, 0, 1] = 0.00  # L down
    ankles[:20, 1, 1] = 0.00  # R down
    ankles[20:, 1, 1] = 0.30  # R high

    contacts = detect_foot_contacts(ankles, fps=30.0)
    assert contacts.shape == (n, 2)
    # L is in contact on the second half only, R on the first half only
    # (ignore a few transition frames where the velocity criterion may
    # flip).
    assert contacts[25:, 0].sum() > 10  # L down in the back
    assert contacts[:15, 1].sum() > 10  # R down in the front


# ----------------------------------------------------------------------
# Anchor tracking
# ----------------------------------------------------------------------


def test_track_stance_anchors_single_episode():
    n = 20
    toes = np.zeros((n, 2, 3), dtype=np.float64)
    toes[:, 0, :] = [1.0, 0.05, 0.5]
    toes[:, 1, :] = [-1.0, 0.05, 0.5]
    contacts = np.zeros((n, 2), dtype=bool)
    contacts[5:15, 0] = True  # Left foot contact frames 5..14 (10 frames)
    anchors, valid, episodes = track_stance_anchors(toes, contacts)
    assert episodes == 1
    # Anchor held across the full episode.
    assert valid[5:15, 0].all()
    assert not valid[:5, 0].any()
    assert not valid[15:, 0].any()
    assert np.allclose(anchors[5, 0], [1.0, 0.05, 0.5])
    assert np.allclose(anchors[14, 0], [1.0, 0.05, 0.5])


def test_track_stance_anchors_rejects_single_frame_stance():
    n = 10
    toes = np.zeros((n, 2, 3), dtype=np.float64)
    contacts = np.zeros((n, 2), dtype=bool)
    contacts[3, 0] = True  # single-frame blip
    _, valid, episodes = track_stance_anchors(toes, contacts, min_stance_frames=2)
    assert episodes == 0
    assert not valid.any()


def test_track_stance_anchors_median_window():
    n = 20
    toes = np.zeros((n, 2, 3), dtype=np.float64)
    # First three contact frames have noisy X values: 0.0, 0.5, 0.1
    contacts = np.zeros((n, 2), dtype=bool)
    contacts[5:15, 0] = True
    toes[5:15, 0, :] = [0.1, 0.0, 0.0]
    toes[5, 0, 0] = 0.0   # noise frame 1
    toes[6, 0, 0] = 0.5   # noise frame 2 (outlier)
    toes[7, 0, 0] = 0.1   # noise frame 3
    anchors, _, _ = track_stance_anchors(toes, contacts, anchor_window=3)
    # Median of [0.0, 0.5, 0.1] == 0.1 — not the outlier.
    assert abs(anchors[10, 0, 0] - 0.1) < 1e-6


# ----------------------------------------------------------------------
# Pin offset
# ----------------------------------------------------------------------


def test_pin_offset_constant_when_both_feet_anchored():
    n = 10
    anchors = np.zeros((n, 2, 3), dtype=np.float64)
    anchors[:, 0] = [1.0, 0.0, 0.0]
    anchors[:, 1] = [-1.0, 0.0, 0.0]

    # Feet drift linearly in X.
    toes = np.zeros((n, 2, 3), dtype=np.float64)
    drift = np.linspace(0.0, 0.3, n)
    toes[:, 0, 0] = 1.0 + drift
    toes[:, 1, 0] = -1.0 + drift

    valid = np.ones((n, 2), dtype=bool)
    offset = compute_pin_offset(toes, anchors, valid, fps=30.0, smoothing_sigma_sec=0.0)
    # Expected: negate the drift.
    assert np.allclose(offset[:, 0], -drift, atol=1e-6)


def test_pin_offset_smooth_at_stance_transitions():
    n = 60
    anchors = np.zeros((n, 2, 3), dtype=np.float64)
    toes = np.zeros((n, 2, 3), dtype=np.float64)
    valid = np.zeros((n, 2), dtype=bool)

    # Two disjoint stance episodes. First: offset ~ (0,0,0).
    # Second: offset ~ (0,0,0.15) so the 15-frame interpolation ramp
    # stays within ~1 cm/frame linearly, and gets further smoothed.
    anchors[5:20, 0] = [1.0, 0.0, 0.0]
    toes[5:20, 0] = [1.0, 0.0, 0.0]
    valid[5:20, 0] = True

    anchors[35:55, 0] = [0.0, 0.0, 0.15]
    toes[35:55, 0] = [0.0, 0.0, 0.0]
    valid[35:55, 0] = True

    offset = compute_pin_offset(toes, anchors, valid, fps=30.0, smoothing_sigma_sec=0.1)
    # After smoothing, per-frame step is bounded; confirm no pop.
    diffs = np.linalg.norm(np.diff(offset, axis=0), axis=-1)
    assert diffs.max() < 0.02


def test_pin_offset_no_valid_frames_returns_zero():
    n = 10
    anchors = np.zeros((n, 2, 3), dtype=np.float64)
    toes = np.zeros((n, 2, 3), dtype=np.float64)
    valid = np.zeros((n, 2), dtype=bool)
    offset = compute_pin_offset(toes, anchors, valid, fps=30.0)
    assert offset.shape == (n, 3)
    assert np.allclose(offset, 0.0)


# ----------------------------------------------------------------------
# apply_foot_pin
# ----------------------------------------------------------------------


def test_apply_foot_pin_fixes_synthesized_slide():
    n = 60
    # Rest pose, root slides linearly in +X over 60 frames.
    slide = np.zeros((n, 3), dtype=np.float32)
    # 0.4 m over 60 frames at 30 fps -> 0.2 m/s, below velocity threshold
    slide[:, 0] = np.linspace(0.0, 0.4, n, dtype=np.float32)
    params = _zero_params(n, transl=slide)

    out, stats = apply_foot_pin(params, fps=30.0)
    # Compute toe positions from pinned output and measure residual horizontal drift.
    joints = forward_kinematics_body(
        out["body_pose"], out["global_orient"], out["transl"]
    )
    toes = joints[:, [10, 11], :]
    horiz = np.linalg.norm(np.diff(toes, axis=0)[..., [0, 2]], axis=-1)
    # Horizontal displacement during stance after pin should be tiny.
    assert horiz.max() < 0.02
    assert stats["stance_episode_count"] >= 1


def test_apply_foot_pin_noop_on_constant_pose():
    n = 30
    params = _zero_params(n)
    out, stats = apply_foot_pin(params, fps=30.0)
    # Static input → zero pin offset, output transl matches input.
    assert np.allclose(out["transl"], params["transl"], atol=1e-5)


def test_apply_foot_pin_no_stance_returns_unchanged():
    n = 20
    # Lift the whole body high enough that height-threshold fails
    # throughout — no contact frames, no stance episodes.
    transl = np.zeros((n, 3), dtype=np.float32)
    transl[:, 1] = 5.0
    # Make the feet move fast so velocity criterion also fails.
    body_pose = np.zeros((n, 21, 3), dtype=np.float32)
    body_pose[:, 3, 0] = np.linspace(0, 2.0, n, dtype=np.float32)  # L_Knee rotating fast
    body_pose[:, 4, 0] = np.linspace(0, 2.0, n, dtype=np.float32)  # R_Knee
    params = {
        "num_frames": n,
        "body_pose": body_pose,
        "global_orient": np.zeros((n, 3), dtype=np.float32),
        "transl": transl,
    }
    out, stats = apply_foot_pin(params, fps=30.0)
    # With no valid stances, pin offset is all zeros → transl unchanged.
    assert np.allclose(out["transl"], transl, atol=1e-5)
    assert stats["stance_episode_count"] == 0


# ----------------------------------------------------------------------
# Runner integration
# ----------------------------------------------------------------------


def test_run_spring_refine_with_pin_integration():
    n = 60
    slide = np.zeros((n, 3), dtype=np.float32)
    # 0.4 m over 60 frames at 30 fps -> 0.2 m/s, below velocity threshold
    slide[:, 0] = np.linspace(0.0, 0.4, n, dtype=np.float32)
    params = _zero_params(n, transl=slide)

    refined, ok = run_spring_refine(
        params, preset="moderate", fps=30.0, pin_feet=True
    )
    assert ok is True
    assert refined["source"] == "spring_refined_pinned"
    assert "foot_pin_stats" in refined
    assert refined["foot_pin_stats"]["stance_episode_count"] >= 1


def test_run_spring_refine_pin_failure_falls_back(monkeypatch):
    n = 20
    params = _zero_params(n)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated pin failure")

    import workers.spring_refine.footpin as footpin_mod

    monkeypatch.setattr(footpin_mod, "apply_foot_pin", _boom)

    refined, ok = run_spring_refine(
        params, preset="moderate", fps=30.0, pin_feet=True
    )
    # Rotation filter still ran → ok=True, source unchanged, error recorded.
    assert ok is True
    assert refined["source"] == "spring_refined"
    assert "error" in refined["foot_pin_stats"]


def test_run_spring_refine_filter_off_pin_on():
    n = 60
    slide = np.zeros((n, 3), dtype=np.float32)
    # 0.4 m over 60 frames at 30 fps -> 0.2 m/s, below velocity threshold
    slide[:, 0] = np.linspace(0.0, 0.4, n, dtype=np.float32)
    params = _zero_params(n, transl=slide)

    refined, ok = run_spring_refine(
        params,
        preset="moderate",
        fps=30.0,
        pin_feet=True,
        filter_rotations=False,
    )
    assert ok is True
    assert refined["source"] == "spring_refined_pinned"
    # Body pose passed through unchanged (filter disabled).
    np.testing.assert_allclose(refined["body_pose"], params["body_pose"])


def test_run_spring_refine_both_flags_off_returns_failure():
    n = 10
    params = _zero_params(n)
    refined, ok = run_spring_refine(
        params, filter_rotations=False, pin_feet=False
    )
    assert ok is False


# ----------------------------------------------------------------------
# Metrics JSON
# ----------------------------------------------------------------------


def test_classify_verdict_thresholds():
    assert classify_verdict(0.05) == "improved"
    assert classify_verdict(-0.05) == "worse"
    assert classify_verdict(0.001) == "neutral"
    assert classify_verdict(-0.001) == "neutral"


def test_write_spring_metrics_json_verdict_improved(tmp_path):
    n = 60
    slide = np.zeros((n, 3), dtype=np.float32)
    slide[:, 0] = np.linspace(0.0, 0.4, n, dtype=np.float32)
    baseline = _zero_params(n, transl=slide)

    refined, _ = run_spring_refine(baseline, pin_feet=True, filter_rotations=False)
    out_path = tmp_path / "spring_refine" / "metrics.json"
    payload = write_spring_metrics_json(
        baseline_params=baseline,
        refined_params=refined,
        fps=30.0,
        preset="moderate",
        pin_enabled=True,
        out_path=out_path,
        pin_stats=refined.get("foot_pin_stats"),
    )
    assert out_path.is_file()
    assert payload["verdict"] == "improved"
    assert payload["foot_skating_delta_m"] > 0.003
    loaded = json.loads(out_path.read_text())
    assert loaded["verdict"] == "improved"
    assert loaded["pin_enabled"] is True


def test_write_spring_metrics_json_verdict_neutral(tmp_path):
    n = 30
    baseline = _zero_params(n)
    refined = _zero_params(n)
    out_path = tmp_path / "metrics.json"
    payload = write_spring_metrics_json(
        baseline_params=baseline,
        refined_params=refined,
        fps=30.0,
        preset="moderate",
        pin_enabled=False,
        out_path=out_path,
    )
    assert payload["verdict"] == "neutral"
    assert abs(payload["foot_skating_delta_m"]) < 1e-6
