"""Tests for pipeline workers."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.pipeline_config import PipelineConfig
from workers.gvhmr_worker import (
    STAGE_PATTERNS,
    find_output_dir,
    find_file,
    match_stage_progress,
    GVHMRWorker,
)
from workers.smplestx_worker import SMPLestXWorker
from workers.pipeline_orchestrator import _FULL_STAGES, FullPipelineWorker, MultiPersonWorker


# ---------------------------------------------------------------------------
# STAGE_PATTERNS regex matching
# ---------------------------------------------------------------------------


class TestStagePatterns:
    @pytest.mark.parametrize(
        "line,expected",
        [
            # Preprocessing (0.05)
            ("Loading video frames...", (0.05, "Preprocessing")),
            ("Preprocessing input video", (0.05, "Preprocessing")),
            ("Reading video file", (0.05, "Preprocessing")),
            # YOLO Tracking (0.15)
            ("Running YOLO detector", (0.15, "YOLO Tracking")),
            ("Tracking objects in frame", (0.15, "YOLO Tracking")),
            ("Person detection complete", (0.15, "YOLO Tracking")),
            # ViTPose (0.30)
            ("ViTPose estimation running", (0.30, "ViTPose")),
            ("2D pose extraction", (0.30, "ViTPose")),
            ("Pose estimation step 3/5", (0.30, "ViTPose")),
            # HMR2 Features (0.45)
            ("HMR2 feature computation", (0.45, "HMR2 Features")),
            ("hmr4d_feature extraction", (0.45, "HMR2 Features")),
            ("Feature extraction done", (0.45, "HMR2 Features")),
            # Camera Estimation (0.60)
            ("DPVO processing frame 100", (0.60, "Camera Estimation")),
            ("simple_vo running", (0.60, "Camera Estimation")),
            ("Camera estimation step", (0.60, "Camera Estimation")),
            ("SLAM initialization", (0.60, "Camera Estimation")),
            # GVHMR Prediction (0.80)
            ("GVHMR prediction step", (0.80, "GVHMR Prediction")),
            ("Predicting body motion", (0.80, "GVHMR Prediction")),
            ("Diffusion sampling...", (0.80, "GVHMR Prediction")),
            # Rendering (0.95)
            ("Rendering output videos", (0.95, "Rendering")),
            ("Saving results to disk", (0.95, "Rendering")),
            ("Visualization complete", (0.95, "Rendering")),
        ],
    )
    def test_pattern_match(self, line, expected):
        assert match_stage_progress(line) == expected

    @pytest.mark.parametrize(
        "line",
        [
            "epoch 42 loss 0.001",
            "",
            "Some random log line",
            "Downloading model weights...",
        ],
    )
    def test_no_match(self, line):
        assert match_stage_progress(line) is None

    def test_first_match_wins(self):
        """'detection' matches YOLO (0.15), not anything later."""
        assert match_stage_progress("detection phase") == (0.15, "YOLO Tracking")

    def test_case_insensitive(self):
        assert match_stage_progress("VITPOSE running") == (0.30, "ViTPose")
        assert match_stage_progress("gvhmr PREDICTION") == (0.80, "GVHMR Prediction")


# ---------------------------------------------------------------------------
# Output directory discovery
# ---------------------------------------------------------------------------


class TestFindOutputDir:
    def test_demo_stem(self, tmp_path):
        gvhmr_root = tmp_path / "GVHMR"
        video = tmp_path / "test_video.mp4"
        expected = gvhmr_root / "outputs" / "demo" / "test_video"
        expected.mkdir(parents=True)

        assert find_output_dir(video, gvhmr_root) == expected

    def test_stem_fallback(self, tmp_path):
        gvhmr_root = tmp_path / "GVHMR"
        video = tmp_path / "test_video.mp4"
        expected = gvhmr_root / "outputs" / "test_video"
        expected.mkdir(parents=True)

        assert find_output_dir(video, gvhmr_root) == expected

    def test_demo_stem_takes_priority(self, tmp_path):
        gvhmr_root = tmp_path / "GVHMR"
        video = tmp_path / "test_video.mp4"
        # Both exist — demo/stem should win
        demo = gvhmr_root / "outputs" / "demo" / "test_video"
        plain = gvhmr_root / "outputs" / "test_video"
        demo.mkdir(parents=True)
        plain.mkdir(parents=True)

        assert find_output_dir(video, gvhmr_root) == demo

    def test_most_recent_dir(self, tmp_path):
        gvhmr_root = tmp_path / "GVHMR"
        video = tmp_path / "other_video.mp4"
        demo_dir = gvhmr_root / "outputs" / "demo"
        old = demo_dir / "old_run"
        new = demo_dir / "new_run"
        old.mkdir(parents=True)
        time.sleep(0.05)
        new.mkdir(parents=True)

        assert find_output_dir(video, gvhmr_root) == new

    def test_no_output(self, tmp_path):
        gvhmr_root = tmp_path / "GVHMR"
        video = tmp_path / "test.mp4"

        assert find_output_dir(video, gvhmr_root) is None

    def test_no_outputs_dir(self, tmp_path):
        gvhmr_root = tmp_path / "GVHMR"
        gvhmr_root.mkdir()
        video = tmp_path / "test.mp4"

        assert find_output_dir(video, gvhmr_root) is None


# ---------------------------------------------------------------------------
# find_file
# ---------------------------------------------------------------------------


class TestFindFile:
    def test_exact(self, tmp_path):
        (tmp_path / "hmr4d_results.pt").touch()
        assert find_file(tmp_path, "hmr4d_results.pt") == tmp_path / "hmr4d_results.pt"

    def test_glob(self, tmp_path):
        (tmp_path / "video_side_by_side.mp4").touch()
        assert find_file(tmp_path, "*side_by_side*.mp4") == tmp_path / "video_side_by_side.mp4"

    def test_nested(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "result.pt").touch()
        assert find_file(tmp_path, "result.pt") == sub / "result.pt"

    def test_no_match(self, tmp_path):
        assert find_file(tmp_path, "nonexistent.xyz") is None


# ---------------------------------------------------------------------------
# GVHMRWorker command building
# ---------------------------------------------------------------------------


class TestGVHMRWorkerCommand:
    def test_default(self, qapp):
        config = PipelineConfig(static_cam=True, use_dpvo=False, focal_mm=24.0)
        w = GVHMRWorker(Path("/tmp/v.mp4"), config, Path("/tmp/GVHMR"))
        cmd = w._build_command()

        assert cmd[0] == sys.executable
        assert cmd[1] == "tools/demo/demo.py"
        assert f"--video={Path('/tmp/v.mp4')}" in cmd
        assert "--static_cam" in cmd
        assert "--use_dpvo" not in cmd
        assert not any(c.startswith("--f_mm") for c in cmd)

    def test_all_options(self, qapp):
        config = PipelineConfig(static_cam=False, use_dpvo=True, focal_mm=50.0)
        w = GVHMRWorker(
            Path("/tmp/v.mp4"),
            config,
            Path("/tmp/GVHMR"),
            extra_args=["--output_root=/tmp/out"],
        )
        cmd = w._build_command()

        assert "--static_cam" not in cmd
        assert "--use_dpvo" in cmd
        assert "--f_mm=50.0" in cmd
        assert "--output_root=/tmp/out" in cmd

    def test_cancel(self, qapp):
        config = PipelineConfig()
        w = GVHMRWorker(Path("/tmp/v.mp4"), config, Path("/tmp/GVHMR"))
        assert not w._cancelled
        w.cancel()
        assert w._cancelled

    def test_parse_progress_flag(self, qapp):
        config = PipelineConfig()
        w = GVHMRWorker(
            Path("/tmp/v.mp4"), config, Path("/tmp/GVHMR"), parse_progress=False
        )
        assert not w._parse_progress


# ---------------------------------------------------------------------------
# SMPLestXWorker command building
# ---------------------------------------------------------------------------


class TestSMPLestXWorkerCommand:
    def test_command(self, qapp):
        w = SMPLestXWorker(
            video_path=Path("/tmp/v.mp4"),
            fps=24.0,
            output_dir=Path("/tmp/out"),
            smplestx_python="/env/bin/python",
            smplestx_dir="/opt/smplestx",
        )
        cmd = w._build_command()

        assert cmd[0] == "/env/bin/python"
        assert "smplestx_inference.py" in cmd
        assert f"--video={Path('/tmp/v.mp4')}" in cmd
        assert "--fps=24.0" in cmd
        assert f"--output_dir={Path('/tmp/out')}" in cmd
        assert "--no_render" in cmd

    def test_cancel(self, qapp):
        w = SMPLestXWorker(
            Path("/tmp/v.mp4"), 30.0, Path("/tmp/out"), "python", "."
        )
        assert not w._cancelled
        w.cancel()
        assert w._cancelled


# ---------------------------------------------------------------------------
# SMPLestXWorker result discovery
# ---------------------------------------------------------------------------


class TestSMPLestXFindResult:
    def test_finds_pt(self, qapp, tmp_path):
        (tmp_path / "result.pt").touch()
        w = SMPLestXWorker(Path("v.mp4"), 30.0, tmp_path, "python", ".")
        assert w._find_result() == tmp_path / "result.pt"

    def test_finds_npz(self, qapp, tmp_path):
        (tmp_path / "result.npz").touch()
        w = SMPLestXWorker(Path("v.mp4"), 30.0, tmp_path, "python", ".")
        assert w._find_result() == tmp_path / "result.npz"

    def test_prefers_recent(self, qapp, tmp_path):
        old = tmp_path / "old.pt"
        old.touch()
        time.sleep(0.05)
        new = tmp_path / "new.npz"
        new.touch()
        w = SMPLestXWorker(Path("v.mp4"), 30.0, tmp_path, "python", ".")
        assert w._find_result() == new

    def test_none_when_empty(self, qapp, tmp_path):
        w = SMPLestXWorker(Path("v.mp4"), 30.0, tmp_path, "python", ".")
        assert w._find_result() is None


# ---------------------------------------------------------------------------
# FullPipelineWorker stage ranges
# ---------------------------------------------------------------------------


class TestFullPipelineStages:
    def test_contiguous(self):
        """Stage ranges must be contiguous and span [0, 1]."""
        assert _FULL_STAGES[0][0] == 0.0
        assert _FULL_STAGES[-1][1] == 1.0
        for i in range(len(_FULL_STAGES) - 1):
            assert abs(_FULL_STAGES[i][1] - _FULL_STAGES[i + 1][0]) < 1e-9, (
                f"Gap between stage {i} and {i + 1}"
            )

    def test_monotonic(self):
        """Each stage range must be ascending."""
        for start, end, _label in _FULL_STAGES:
            assert start < end

    def test_all_labeled(self):
        for _start, _end, label in _FULL_STAGES:
            assert len(label) > 0

    def test_cancel(self, qapp):
        config = PipelineConfig()
        w = FullPipelineWorker(
            Path("/tmp/v.mp4"), config, Path("/tmp/GVHMR"), Path("/tmp/out")
        )
        assert not w._cancelled
        w.cancel()
        assert w._cancelled


# ---------------------------------------------------------------------------
# MultiPersonWorker
# ---------------------------------------------------------------------------


class TestMultiPersonWorker:
    def test_cancel(self, qapp):
        config = PipelineConfig()
        w = MultiPersonWorker(
            Path("/tmp/v.mp4"), config, Path("/tmp/GVHMR"), Path("/tmp/out")
        )
        assert not w._cancelled
        w.cancel()
        assert w._cancelled

    def test_has_signals(self, qapp):
        config = PipelineConfig()
        w = MultiPersonWorker(
            Path("/tmp/v.mp4"), config, Path("/tmp/GVHMR"), Path("/tmp/out")
        )
        # Verify signal attributes exist
        assert hasattr(w, "progress")
        assert hasattr(w, "log_line")
        assert hasattr(w, "finished")
        assert hasattr(w, "error")
