"""Tests for camera-stabilized world grounding.

Covers smoothing helpers, cam_angvel computation, round-trip consistency,
and graceful fallback when data is missing.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_c2w(n: int) -> np.ndarray:
    """Return n identical identity C2W matrices."""
    c2w = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    return c2w


def _noisy_c2w(n: int, seed: int = 42) -> np.ndarray:
    """Return n C2W matrices with mild random rotation + translation noise."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    c2w = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    # Small random rotations
    rotvecs = rng.normal(scale=0.02, size=(n, 3)).astype(np.float32)
    c2w[:, :3, :3] = Rotation.from_rotvec(rotvecs).as_matrix().astype(np.float32)
    # Small random translations
    c2w[:, :3, 3] = rng.normal(scale=0.1, size=(n, 3)).astype(np.float32)
    return c2w


# ---------------------------------------------------------------------------
# _smooth_c2w
# ---------------------------------------------------------------------------


def test_smooth_c2w_identity_unchanged():
    """Smoothing identity matrices should return (near-)identity."""
    from workers.spring_refine.camera_stabilize import _smooth_c2w

    c2w = _identity_c2w(20)
    smoothed = _smooth_c2w(c2w, fps=30.0, preset="moderate")
    # Rotation part stays close to identity
    assert np.allclose(smoothed[:, :3, :3], np.eye(3), atol=1e-4)
    # Translation stays near zero
    assert np.allclose(smoothed[:, :3, 3], 0.0, atol=1e-4)


def test_smooth_c2w_reduces_noise():
    """Smoothing noisy C2W should reduce frame-to-frame variation."""
    from workers.spring_refine.camera_stabilize import _smooth_c2w

    c2w = _noisy_c2w(50)
    smoothed = _smooth_c2w(c2w, fps=30.0, preset="heavy")

    # Frame-to-frame translation jitter
    raw_jitter = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=-1)
    smooth_jitter = np.linalg.norm(np.diff(smoothed[:, :3, 3], axis=0), axis=-1)
    assert smooth_jitter.mean() < raw_jitter.mean()


def test_smooth_c2w_short_sequence():
    """Sequences shorter than 3 frames should return unchanged."""
    from workers.spring_refine.camera_stabilize import _smooth_c2w

    c2w = _noisy_c2w(2)
    smoothed = _smooth_c2w(c2w, fps=30.0, preset="moderate")
    np.testing.assert_allclose(smoothed, c2w)


# ---------------------------------------------------------------------------
# _compute_cam_angvel
# ---------------------------------------------------------------------------


def test_cam_angvel_identity_rotations():
    """Identity W2C rotations should produce (near-)identity 6D cam_angvel."""
    from workers.spring_refine.camera_stabilize import _compute_cam_angvel
    from pytorch3d.transforms import rotation_6d_to_matrix

    R = torch.eye(3).unsqueeze(0).expand(20, -1, -1)
    angvel = _compute_cam_angvel(R)
    assert angvel.shape == (20, 6)
    # Decoded back to matrices should be near identity
    mats = rotation_6d_to_matrix(angvel)
    assert torch.allclose(mats, torch.eye(3).unsqueeze(0).expand(20, -1, -1), atol=1e-5)


def test_cam_angvel_shape_and_padding():
    """cam_angvel should have same frame count as input (last frame padded)."""
    from workers.spring_refine.camera_stabilize import _compute_cam_angvel
    from scipy.spatial.transform import Rotation

    N = 30
    rotvecs = np.random.default_rng(0).normal(scale=0.01, size=(N, 3))
    R_np = Rotation.from_rotvec(rotvecs).as_matrix().astype(np.float32)
    R = torch.from_numpy(R_np)

    angvel = _compute_cam_angvel(R)
    assert angvel.shape == (N, 6)
    # Last frame should equal second-to-last (padding)
    torch.testing.assert_close(angvel[-1], angvel[-2])


# ---------------------------------------------------------------------------
# _normalize_slam_w2c
# ---------------------------------------------------------------------------


def test_normalize_slam_w2c_4x4_passthrough():
    """4x4 matrices should pass through unchanged."""
    from workers.spring_refine.camera_stabilize import _normalize_slam_w2c

    w2c = np.tile(np.eye(4, dtype=np.float32), (10, 1, 1))
    result = _normalize_slam_w2c(w2c)
    np.testing.assert_allclose(result, w2c)


def test_normalize_slam_w2c_dpvo_format():
    """DPVO 7-column rows should be converted to W2C 4x4."""
    from workers.spring_refine.camera_stabilize import _normalize_slam_w2c
    from scipy.spatial.transform import Rotation

    N = 5
    # Identity C2W: zero translation, identity quaternion (xyzw = 0,0,0,1)
    rows = np.zeros((N, 7), dtype=np.float32)
    rows[:, 6] = 1.0  # qw = 1
    result = _normalize_slam_w2c(rows)
    assert result.shape == (N, 4, 4)
    # Should be identity W2C
    for i in range(N):
        np.testing.assert_allclose(result[i, :3, :3], np.eye(3), atol=1e-5)
        np.testing.assert_allclose(result[i, :3, 3], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# stabilize_world_params (integration)
# ---------------------------------------------------------------------------


def test_stabilize_returns_none_when_no_net_outputs(tmp_path):
    """Should return None gracefully for .pt files without net_outputs."""
    from workers.spring_refine.camera_stabilize import stabilize_world_params

    pt = tmp_path / "hmr4d_results.pt"
    torch.save({"smpl_params_global": {}}, str(pt))
    result = stabilize_world_params(pt)
    assert result is None


def test_stabilize_returns_none_when_no_slam(tmp_path):
    """Should return None when SLAM data cannot be found."""
    from workers.spring_refine.camera_stabilize import stabilize_world_params

    L = 20
    decode_dict = {
        "global_orient_gv": torch.zeros(1, L, 3),
        "local_transl_vel": torch.zeros(1, L, 3),
        "global_orient": torch.zeros(1, L, 3),
    }
    pt = tmp_path / "hmr4d_results.pt"
    torch.save({"net_outputs": {"decode_dict": decode_dict}}, str(pt))
    result = stabilize_world_params(pt)
    assert result is None


def test_stabilize_returns_none_for_missing_file(tmp_path):
    """Should return None for a non-existent .pt file."""
    from workers.spring_refine.camera_stabilize import stabilize_world_params

    result = stabilize_world_params(tmp_path / "nonexistent.pt")
    assert result is None


def test_stabilize_round_trip_identity(tmp_path):
    """With identity SLAM (no camera motion), stabilized params should
    closely match the original global params."""
    try:
        from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import get_smpl_params_w_Rt_v2  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        pytest.skip("GVHMR pipeline deps (hydra_zen etc.) not available")

    from workers.spring_refine.camera_stabilize import stabilize_world_params

    L = 30
    decode_dict = {
        "global_orient_gv": torch.zeros(1, L, 3),
        "local_transl_vel": torch.zeros(1, L, 3),
        "global_orient": torch.zeros(1, L, 3),
    }
    pt = tmp_path / "hmr4d_results.pt"
    slam = np.tile(np.eye(4, dtype=np.float32), (L, 1, 1))
    slam_path = tmp_path / "shared_slam.pt"
    torch.save(torch.from_numpy(slam), str(slam_path))
    torch.save({"net_outputs": {"decode_dict": decode_dict}}, str(pt))

    result = stabilize_world_params(pt, slam_path=slam_path)
    assert result is not None
    assert result["global_orient"].shape == (L, 3)
    assert result["transl"].shape == (L, 3)
    # With identity everything, output should be near zero
    assert np.linalg.norm(result["transl"]) < 1.0
