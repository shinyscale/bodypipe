"""Tests for GEM-X worker and SOMA BVH export."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workers.gemx_worker import load_gemx_soma_output, GEMXWorker
from workers.soma_bvh_export import convert_soma_to_bvh, _axis_angle_to_euler_zxy
from models.skeleton import SOMA_SKELETON


def _make_soma_params(n_frames=30, n_joints=77):
    """Create synthetic SOMA params for testing."""
    return {
        "poses": np.random.randn(n_frames, n_joints, 3).astype(np.float32) * 0.1,
        "transl": np.random.randn(n_frames, 3).astype(np.float32) * 0.5,
        "global_orient": np.random.randn(n_frames, 3).astype(np.float32) * 0.1,
        "identity_coeffs": np.zeros((1, 45), dtype=np.float32),
        "scale_params": np.zeros((1, 68), dtype=np.float32),
        "identity_model_type": "mhr",
    }


class TestAxisAngleToEulerZxy:
    """Euler conversion for BVH export."""

    def test_zero_input(self):
        euler = _axis_angle_to_euler_zxy(np.zeros(3))
        np.testing.assert_allclose(euler, 0.0, atol=1e-5)

    def test_output_shape(self):
        euler = _axis_angle_to_euler_zxy(np.array([0.1, 0.2, 0.3]))
        assert euler.shape == (3,)
        assert euler.dtype == np.float32

    def test_nonzero_rotation(self):
        aa = np.array([np.pi / 4, 0, 0])  # 45° around X
        euler = _axis_angle_to_euler_zxy(aa)
        assert not np.allclose(euler, 0.0)


class TestConvertSomaToBvh:
    """SOMA BVH writer."""

    def test_writes_file(self, tmp_path):
        params = _make_soma_params(n_frames=10)
        bvh_path = tmp_path / "test.bvh"
        result = convert_soma_to_bvh(params, bvh_path, fps=30.0)
        assert result == bvh_path
        assert bvh_path.is_file()

    def test_bvh_header(self, tmp_path):
        params = _make_soma_params(n_frames=5)
        bvh_path = tmp_path / "test.bvh"
        convert_soma_to_bvh(params, bvh_path, fps=30.0)
        content = bvh_path.read_text()
        assert content.startswith("HIERARCHY")
        assert "MOTION" in content
        assert "Frames: 5" in content
        assert "Frame Time: 0.033333" in content

    def test_root_joint_name(self, tmp_path):
        params = _make_soma_params(n_frames=3)
        bvh_path = tmp_path / "test.bvh"
        convert_soma_to_bvh(params, bvh_path)
        content = bvh_path.read_text()
        assert "ROOT Hips" in content

    def test_root_has_6_channels(self, tmp_path):
        params = _make_soma_params(n_frames=3)
        bvh_path = tmp_path / "test.bvh"
        convert_soma_to_bvh(params, bvh_path)
        content = bvh_path.read_text()
        assert "CHANNELS 6 Xposition Yposition Zposition" in content

    def test_frame_count_matches(self, tmp_path):
        params = _make_soma_params(n_frames=20)
        bvh_path = tmp_path / "test.bvh"
        convert_soma_to_bvh(params, bvh_path)
        content = bvh_path.read_text()
        motion_section = content.split("MOTION\n")[1]
        frame_lines = [l for l in motion_section.strip().split("\n") if not l.startswith("Frames:") and not l.startswith("Frame Time:")]
        assert len(frame_lines) == 20

    def test_custom_joint_names(self, tmp_path):
        """Custom joint names/parents override defaults."""
        names = ("Root", "Child1", "Child2")
        parents = (-1, 0, 0)
        params = {
            "poses": np.zeros((5, 3, 3), dtype=np.float32),
            "transl": np.zeros((5, 3), dtype=np.float32),
        }
        bvh_path = tmp_path / "custom.bvh"
        convert_soma_to_bvh(params, bvh_path, joint_names=names, joint_parents=parents)
        content = bvh_path.read_text()
        assert "ROOT Root" in content
        assert "JOINT Child1" in content


class TestLoadGemxSomaOutput:
    """Loading SOMA params from GEM-X output."""

    def test_loads_npz(self, tmp_path):
        params = _make_soma_params(n_frames=10)
        np.savez(tmp_path / "soma_results.npz", **params)
        loaded = load_gemx_soma_output(tmp_path)
        assert loaded is not None
        assert loaded["poses"].shape == (10, 77, 3)
        assert loaded["transl"].shape == (10, 3)

    def test_returns_none_for_empty_dir(self, tmp_path):
        assert load_gemx_soma_output(tmp_path) is None

    def test_returns_none_for_invalid_npz(self, tmp_path):
        np.savez(tmp_path / "bad.npz", data=np.zeros(10))
        assert load_gemx_soma_output(tmp_path) is None

    def test_float32_output(self, tmp_path):
        params = _make_soma_params(n_frames=5)
        # Save as float64 to test conversion
        params["poses"] = params["poses"].astype(np.float64)
        np.savez(tmp_path / "soma_results.npz", **params)
        loaded = load_gemx_soma_output(tmp_path)
        assert loaded["poses"].dtype == np.float32


class TestGEMXWorker:
    """GEMXWorker construction and interface."""

    def test_constructor(self, tmp_path):
        from models.pipeline_config import PipelineConfig

        worker = GEMXWorker(
            video_path=Path("/tmp/test.mp4"),
            config=PipelineConfig(estimation_backend="gemx", body_model="soma"),
            gemx_root=Path("/opt/gemx"),
            output_dir=tmp_path,
            fps=30.0,
        )
        assert hasattr(worker, "progress")
        assert hasattr(worker, "log_line")
        assert hasattr(worker, "finished")
        assert hasattr(worker, "error")

    def test_has_cancel(self, tmp_path):
        from models.pipeline_config import PipelineConfig

        worker = GEMXWorker(
            video_path=Path("/tmp/test.mp4"),
            config=PipelineConfig(),
            gemx_root=Path("/opt/gemx"),
            output_dir=tmp_path,
        )
        worker.cancel()
        assert worker._cancelled is True
