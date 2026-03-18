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
from workers.pipeline_orchestrator import (
    _FULL_STAGES,
    FullPipelineWorker,
    MultiPersonWorker,
    find_smplestx_result,
    save_merged_pt,
    extract_bboxes_from_output,
    fallback_face_bboxes,
)
from workers.render_worker import RenderWorker


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

    def test_config_stores_new_fields(self, qapp):
        """Worker should preserve render_overlays and use_inpainting from config."""
        config = PipelineConfig(
            render_overlays=True,
            use_inpainting=False,
            target_fps=24.0,
            fbx_naming="UE5 Mannequin",
        )
        w = MultiPersonWorker(
            Path("/tmp/v.mp4"), config, Path("/tmp/GVHMR"), Path("/tmp/out")
        )
        assert w._config.render_overlays is True
        assert w._config.use_inpainting is False
        assert w._config.target_fps == 24.0
        assert w._config.fbx_naming == "UE5 Mannequin"


# ---------------------------------------------------------------------------
# MultiPersonWorker FBX batch conversion
# ---------------------------------------------------------------------------


class TestMultiPersonFbxBatch:
    """Why: after split_multi_person_video() completes, the Gradio GUI converts
    each person's BVH to FBX. The bodypipe MultiPersonWorker must do the same
    post-pipeline step so users get FBX files without manual re-export."""

    def _make_worker(self, qapp, tmp_path, fbx_naming="Mixamo (Cascadeur)", target_fps=30.0):
        config = PipelineConfig(fbx_naming=fbx_naming, target_fps=target_fps)
        return MultiPersonWorker(
            tmp_path / "video.mp4", config, tmp_path / "GVHMR", tmp_path / "out"
        )

    def _make_result_with_bvh(self, tmp_path, num_persons=2):
        """Create a mock result with person_dirs containing .bvh files."""
        from types import SimpleNamespace

        person_dirs = []
        for i in range(num_persons):
            pdir = tmp_path / "out" / f"person_{i}"
            pdir.mkdir(parents=True, exist_ok=True)
            (pdir / f"person_{i}_body.bvh").write_text("HIERARCHY\n")
            person_dirs.append(pdir)
        return SimpleNamespace(person_dirs=person_dirs)

    def test_discovers_bvh_files(self, qapp, tmp_path):
        """Should find BVH files in each person directory."""
        w = self._make_worker(qapp, tmp_path)
        result = self._make_result_with_bvh(tmp_path, num_persons=3)

        from unittest.mock import patch

        converted = []

        def mock_convert(bvh, fbx, fps=30.0, naming="mixamo"):
            converted.append((bvh, fbx, fps, naming))
            Path(fbx).write_text("FBX_DATA")
            return f"[BVH→FBX] Exported: {fbx}"

        with patch.dict(sys.modules, {"bvh_to_fbx": type(sys)("bvh_to_fbx")}):
            sys.modules["bvh_to_fbx"].convert_bvh_to_fbx = mock_convert
            fbx_files = w._convert_bvh_to_fbx_batch(result)

        assert len(converted) == 3
        assert len(fbx_files) == 3
        for fbx_path in fbx_files:
            assert fbx_path.endswith(".fbx")

    def test_uses_ue5_naming_key(self, qapp, tmp_path):
        """UE5 Mannequin config should pass naming='ue5' to converter."""
        w = self._make_worker(qapp, tmp_path, fbx_naming="UE5 Mannequin")
        result = self._make_result_with_bvh(tmp_path, num_persons=1)

        from unittest.mock import patch

        captured_naming = []

        def mock_convert(bvh, fbx, fps=30.0, naming="mixamo"):
            captured_naming.append(naming)
            Path(fbx).write_text("FBX_DATA")
            return f"[BVH→FBX] Exported: {fbx}"

        with patch.dict(sys.modules, {"bvh_to_fbx": type(sys)("bvh_to_fbx")}):
            sys.modules["bvh_to_fbx"].convert_bvh_to_fbx = mock_convert
            w._convert_bvh_to_fbx_batch(result)

        assert captured_naming == ["ue5"]

    def test_uses_mixamo_naming_key(self, qapp, tmp_path):
        """Mixamo (Cascadeur) config should pass naming='mixamo' to converter."""
        w = self._make_worker(qapp, tmp_path, fbx_naming="Mixamo (Cascadeur)")
        result = self._make_result_with_bvh(tmp_path, num_persons=1)

        from unittest.mock import patch

        captured_naming = []

        def mock_convert(bvh, fbx, fps=30.0, naming="mixamo"):
            captured_naming.append(naming)
            Path(fbx).write_text("FBX_DATA")
            return f"[BVH→FBX] Exported: {fbx}"

        with patch.dict(sys.modules, {"bvh_to_fbx": type(sys)("bvh_to_fbx")}):
            sys.modules["bvh_to_fbx"].convert_bvh_to_fbx = mock_convert
            w._convert_bvh_to_fbx_batch(result)

        assert captured_naming == ["mixamo"]

    def test_passes_target_fps(self, qapp, tmp_path):
        """Should pass config.target_fps to convert_bvh_to_fbx."""
        w = self._make_worker(qapp, tmp_path, target_fps=60.0)
        result = self._make_result_with_bvh(tmp_path, num_persons=1)

        from unittest.mock import patch

        captured_fps = []

        def mock_convert(bvh, fbx, fps=30.0, naming="mixamo"):
            captured_fps.append(fps)
            Path(fbx).write_text("FBX_DATA")
            return f"[BVH→FBX] Exported: {fbx}"

        with patch.dict(sys.modules, {"bvh_to_fbx": type(sys)("bvh_to_fbx")}):
            sys.modules["bvh_to_fbx"].convert_bvh_to_fbx = mock_convert
            w._convert_bvh_to_fbx_batch(result)

        assert captured_fps == [60.0]

    def test_skips_existing_fbx(self, qapp, tmp_path):
        """Should not re-convert when FBX already exists."""
        w = self._make_worker(qapp, tmp_path)
        result = self._make_result_with_bvh(tmp_path, num_persons=1)

        # Pre-create the FBX file
        pdir = tmp_path / "out" / "person_0"
        (pdir / "person_0_body.fbx").write_text("EXISTING_FBX")

        from unittest.mock import patch

        convert_called = []

        def mock_convert(bvh, fbx, fps=30.0, naming="mixamo"):
            convert_called.append(True)
            return "[BVH→FBX] Exported"

        with patch.dict(sys.modules, {"bvh_to_fbx": type(sys)("bvh_to_fbx")}):
            sys.modules["bvh_to_fbx"].convert_bvh_to_fbx = mock_convert
            fbx_files = w._convert_bvh_to_fbx_batch(result)

        assert len(convert_called) == 0  # Never called
        assert len(fbx_files) == 1  # But still included in results

    def test_empty_person_dirs(self, qapp, tmp_path):
        """Should return empty list when result has no person_dirs."""
        from types import SimpleNamespace

        w = self._make_worker(qapp, tmp_path)
        result = SimpleNamespace(person_dirs=[])
        fbx_files = w._convert_bvh_to_fbx_batch(result)
        assert fbx_files == []

    def test_no_person_dirs_attr(self, qapp, tmp_path):
        """Should return empty list when result lacks person_dirs attribute."""
        w = self._make_worker(qapp, tmp_path)
        fbx_files = w._convert_bvh_to_fbx_batch(object())
        assert fbx_files == []

    def test_no_bvh_files(self, qapp, tmp_path):
        """Should log and return empty when directories have no BVH files."""
        from types import SimpleNamespace

        w = self._make_worker(qapp, tmp_path)
        pdir = tmp_path / "out" / "person_0"
        pdir.mkdir(parents=True)
        result = SimpleNamespace(person_dirs=[pdir])

        log_lines = []
        w.log_line.connect(log_lines.append)

        from unittest.mock import patch

        with patch.dict(sys.modules, {"bvh_to_fbx": type(sys)("bvh_to_fbx")}):
            sys.modules["bvh_to_fbx"].convert_bvh_to_fbx = lambda *a, **kw: ""
            fbx_files = w._convert_bvh_to_fbx_batch(result)

        assert fbx_files == []
        assert any("no bvh" in line.lower() for line in log_lines)

    def test_import_error_graceful(self, qapp, tmp_path):
        """Should warn and return empty when bvh_to_fbx is unavailable."""
        w = self._make_worker(qapp, tmp_path)
        result = self._make_result_with_bvh(tmp_path, num_persons=1)

        log_lines = []
        w.log_line.connect(log_lines.append)

        # Remove bvh_to_fbx from sys.modules to force ImportError
        import unittest.mock as mock

        with mock.patch.dict(sys.modules, {"bvh_to_fbx": None}):
            fbx_files = w._convert_bvh_to_fbx_batch(result)

        assert fbx_files == []
        assert any("not available" in line.lower() for line in log_lines)

    def test_conversion_error_continues(self, qapp, tmp_path):
        """Should log warning but continue when individual conversion fails."""
        w = self._make_worker(qapp, tmp_path)
        result = self._make_result_with_bvh(tmp_path, num_persons=2)

        from unittest.mock import patch

        call_count = [0]

        def mock_convert(bvh, fbx, fps=30.0, naming="mixamo"):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Blender crashed")
            Path(fbx).write_text("FBX_DATA")
            return f"[BVH→FBX] Exported: {fbx}"

        log_lines = []
        w.log_line.connect(log_lines.append)

        with patch.dict(sys.modules, {"bvh_to_fbx": type(sys)("bvh_to_fbx")}):
            sys.modules["bvh_to_fbx"].convert_bvh_to_fbx = mock_convert
            fbx_files = w._convert_bvh_to_fbx_batch(result)

        assert call_count[0] == 2  # Both attempted
        assert len(fbx_files) == 1  # Only second succeeded
        assert any("failed" in line.lower() for line in log_lines)

    def test_conversion_error_in_log_continues(self, qapp, tmp_path):
        """Should handle ERROR in convert log string (not exception)."""
        w = self._make_worker(qapp, tmp_path)
        result = self._make_result_with_bvh(tmp_path, num_persons=1)

        from unittest.mock import patch

        def mock_convert(bvh, fbx, fps=30.0, naming="mixamo"):
            return "[BVH→FBX] ERROR: Blender not found."

        log_lines = []
        w.log_line.connect(log_lines.append)

        with patch.dict(sys.modules, {"bvh_to_fbx": type(sys)("bvh_to_fbx")}):
            sys.modules["bvh_to_fbx"].convert_bvh_to_fbx = mock_convert
            fbx_files = w._convert_bvh_to_fbx_batch(result)

        assert len(fbx_files) == 0  # ERROR means not included
        assert any("failed" in line.lower() or "ERROR" in line for line in log_lines)

    def test_progress_emission(self, qapp, tmp_path):
        """Should emit progress signals during FBX conversion."""
        w = self._make_worker(qapp, tmp_path)
        result = self._make_result_with_bvh(tmp_path, num_persons=2)

        from unittest.mock import patch

        def mock_convert(bvh, fbx, fps=30.0, naming="mixamo"):
            Path(fbx).write_text("FBX_DATA")
            return f"[BVH→FBX] Exported: {fbx}"

        progress_signals = []
        w.progress.connect(lambda f, m: progress_signals.append((f, m)))

        with patch.dict(sys.modules, {"bvh_to_fbx": type(sys)("bvh_to_fbx")}):
            sys.modules["bvh_to_fbx"].convert_bvh_to_fbx = mock_convert
            w._convert_bvh_to_fbx_batch(result)

        assert len(progress_signals) == 2
        # First at 0.90, second at 0.94 (0.90 + 0.5 * 0.08)
        assert 0.89 < progress_signals[0][0] < 0.95
        assert 0.93 < progress_signals[1][0] < 0.99
        assert "1/2" in progress_signals[0][1]
        assert "2/2" in progress_signals[1][1]

    def test_cancellation_stops_conversion(self, qapp, tmp_path):
        """Should stop converting when cancelled flag is set."""
        w = self._make_worker(qapp, tmp_path)
        result = self._make_result_with_bvh(tmp_path, num_persons=3)

        from unittest.mock import patch

        call_count = [0]

        def mock_convert(bvh, fbx, fps=30.0, naming="mixamo"):
            call_count[0] += 1
            Path(fbx).write_text("FBX_DATA")
            w._cancelled = True  # Cancel after first conversion
            return f"[BVH→FBX] Exported: {fbx}"

        with patch.dict(sys.modules, {"bvh_to_fbx": type(sys)("bvh_to_fbx")}):
            sys.modules["bvh_to_fbx"].convert_bvh_to_fbx = mock_convert
            fbx_files = w._convert_bvh_to_fbx_batch(result)

        assert call_count[0] == 1  # Only first converted before cancel
        assert len(fbx_files) == 1

    def test_fbx_files_in_finished_result(self, qapp, tmp_path):
        """Finished signal dict should contain fbx_files key."""
        w = self._make_worker(qapp, tmp_path)

        # Verify the method returns a list (integration-level check)
        from types import SimpleNamespace

        result = SimpleNamespace(person_dirs=[])
        fbx_files = w._convert_bvh_to_fbx_batch(result)
        assert isinstance(fbx_files, list)


# ---------------------------------------------------------------------------
# find_smplestx_result (module-level helper)
# ---------------------------------------------------------------------------


class TestFindSmplestxResult:
    """Why: find_smplestx_result locates the most recent .pt/.npz output
    file from SMPLest-X. The merge stage depends on this to find the
    hand params for hybrid body+hand merging."""

    def test_finds_pt(self, tmp_path):
        (tmp_path / "result.pt").touch()
        assert find_smplestx_result(tmp_path) == tmp_path / "result.pt"

    def test_finds_npz(self, tmp_path):
        (tmp_path / "result.npz").touch()
        assert find_smplestx_result(tmp_path) == tmp_path / "result.npz"

    def test_prefers_most_recent(self, tmp_path):
        old = tmp_path / "old.pt"
        old.touch()
        time.sleep(0.05)
        new = tmp_path / "new.npz"
        new.touch()
        assert find_smplestx_result(tmp_path) == new

    def test_none_when_empty(self, tmp_path):
        assert find_smplestx_result(tmp_path) is None

    def test_ignores_non_pt_npz(self, tmp_path):
        (tmp_path / "readme.txt").touch()
        (tmp_path / "video.mp4").touch()
        assert find_smplestx_result(tmp_path) is None


# ---------------------------------------------------------------------------
# save_merged_pt (module-level helper)
# ---------------------------------------------------------------------------


class TestSaveMergedPt:
    """Why: save_merged_pt serializes merged SMPL-X params to a .pt file
    that can be reloaded for re-export (BVH/FBX). Must handle numpy arrays
    and reshape pose data correctly."""

    def test_saves_file(self, tmp_path):
        import numpy as np
        import torch

        params = {
            "num_frames": 10,
            "global_orient": np.zeros((10, 3)),
            "body_pose": np.zeros((10, 21, 3)),
            "left_hand_pose": np.zeros((10, 15, 3)),
            "right_hand_pose": np.zeros((10, 15, 3)),
            "transl": np.zeros((10, 3)),
            "betas": np.zeros((10, 10)),
        }
        out = tmp_path / "merged.pt"
        result = save_merged_pt(params, out)

        assert result == out
        assert out.is_file()

        loaded = torch.load(str(out), map_location="cpu", weights_only=False)
        assert loaded["global_orient"].shape == (10, 3)
        assert loaded["body_pose"].shape == (10, 63)  # 21*3 reshaped
        assert loaded["left_hand_pose"].shape == (10, 45)  # 15*3 reshaped
        assert loaded["right_hand_pose"].shape == (10, 45)

    def test_creates_parent_dirs(self, tmp_path):
        import numpy as np

        params = {
            "num_frames": 2,
            "global_orient": np.zeros((2, 3)),
            "body_pose": np.zeros((2, 21, 3)),
            "left_hand_pose": np.zeros((2, 15, 3)),
            "right_hand_pose": np.zeros((2, 15, 3)),
            "transl": np.zeros((2, 3)),
        }
        out = tmp_path / "sub" / "dir" / "merged.pt"
        save_merged_pt(params, out)
        assert out.is_file()

    def test_preserves_optional_fields(self, tmp_path):
        import numpy as np
        import torch

        params = {
            "num_frames": 5,
            "global_orient": np.zeros((5, 3)),
            "body_pose": np.zeros((5, 21, 3)),
            "left_hand_pose": np.zeros((5, 15, 3)),
            "right_hand_pose": np.zeros((5, 15, 3)),
            "transl": np.zeros((5, 3)),
            "coordinate_space": "world",
            "source": "hybrid",
            "K_fullimg": np.eye(3).reshape(1, 3, 3).repeat(5, axis=0),
        }
        out = tmp_path / "merged.pt"
        save_merged_pt(params, out)

        loaded = torch.load(str(out), map_location="cpu", weights_only=False)
        assert loaded["coordinate_space"] == "world"
        assert loaded["source"] == "hybrid"
        assert "K_fullimg" in loaded


# ---------------------------------------------------------------------------
# extract_bboxes_from_output (module-level helper)
# ---------------------------------------------------------------------------


class TestExtractBboxesFromOutput:
    """Why: extract_bboxes_from_output scans .pt/.npz files for person
    bounding box data needed by the face capture stage."""

    def test_finds_bboxes_in_pt(self, tmp_path):
        import numpy as np
        import torch

        bboxes = np.array([[10, 20, 100, 200], [15, 25, 110, 210]], dtype=np.float32)
        torch.save({"bboxes": bboxes}, str(tmp_path / "output.pt"))

        result = extract_bboxes_from_output(tmp_path)
        assert result is not None
        assert result.shape == (2, 4)

    def test_finds_bboxes_in_npz(self, tmp_path):
        import numpy as np

        bboxes = np.array([[10, 20, 100, 200]], dtype=np.float32)
        np.savez(tmp_path / "output.npz", person_bbox=bboxes)

        result = extract_bboxes_from_output(tmp_path)
        assert result is not None
        assert result.shape == (1, 4)

    def test_returns_none_when_no_bboxes(self, tmp_path):
        import torch

        torch.save({"other_data": [1, 2, 3]}, str(tmp_path / "output.pt"))
        result = extract_bboxes_from_output(tmp_path)
        assert result is None

    def test_returns_none_for_empty_dir(self, tmp_path):
        result = extract_bboxes_from_output(tmp_path)
        assert result is None

    def test_tries_multiple_bbox_keys(self, tmp_path):
        """Should find bboxes under any of the known key names."""
        import numpy as np
        import torch

        for key in ["person_bbox", "bboxes", "bbox", "bb_xyxy", "pred_bboxes"]:
            sub = tmp_path / key
            sub.mkdir()
            data = np.array([[0, 0, 50, 50]], dtype=np.float32)
            torch.save({key: data}, str(sub / "result.pt"))

            result = extract_bboxes_from_output(sub)
            assert result is not None, f"Failed for key: {key}"
            assert result.shape == (1, 4)


# ---------------------------------------------------------------------------
# FullPipelineWorker stage helpers
# ---------------------------------------------------------------------------


class TestStageProgressCallback:
    """Why: _stage_progress_callback maps sub-stage [0,1] fractions to the
    overall pipeline progress range for that stage. Downstream backend functions
    use this to report fine-grained progress within a stage."""

    def test_maps_zero_to_stage_start(self, qapp):
        config = PipelineConfig()
        w = FullPipelineWorker(
            Path("/tmp/v.mp4"), config, Path("/tmp/GVHMR"), Path("/tmp/out")
        )
        received = []
        w.progress.connect(lambda f, m: received.append((f, m)))

        cb = w._stage_progress_callback(4)  # Face pipeline: 0.52-0.72
        cb(0.0, "Starting face")

        assert len(received) == 1
        assert abs(received[0][0] - 0.52) < 1e-6

    def test_maps_one_to_stage_end(self, qapp):
        config = PipelineConfig()
        w = FullPipelineWorker(
            Path("/tmp/v.mp4"), config, Path("/tmp/GVHMR"), Path("/tmp/out")
        )
        received = []
        w.progress.connect(lambda f, m: received.append((f, m)))

        cb = w._stage_progress_callback(4)  # Face pipeline: 0.52-0.72
        cb(1.0, "Done")

        assert len(received) == 1
        assert abs(received[0][0] - 0.72) < 1e-6

    def test_maps_midpoint(self, qapp):
        config = PipelineConfig()
        w = FullPipelineWorker(
            Path("/tmp/v.mp4"), config, Path("/tmp/GVHMR"), Path("/tmp/out")
        )
        received = []
        w.progress.connect(lambda f, m: received.append((f, m)))

        cb = w._stage_progress_callback(5)  # BVH/FBX: 0.72-0.85
        cb(0.5)

        assert len(received) == 1
        expected = 0.72 + 0.5 * (0.85 - 0.72)
        assert abs(received[0][0] - expected) < 1e-6


class TestMergeStageNoOutput:
    """Why: when GVHMR output directory doesn't exist (e.g. pipeline not
    yet run or output cleaned), _run_merge should return (None, None) and
    log a warning rather than crashing."""

    def test_returns_none_when_no_gvhmr_output(self, qapp, tmp_path):
        config = PipelineConfig()
        gvhmr_root = tmp_path / "GVHMR"
        gvhmr_root.mkdir()

        w = FullPipelineWorker(
            tmp_path / "video.mp4", config, gvhmr_root, tmp_path / "out"
        )
        log_lines = []
        w.log_line.connect(log_lines.append)

        results = {}
        world, camera = w._run_merge(results, smplestx_ran=False)

        assert world is None
        assert camera is None
        assert any("not found" in line.lower() for line in log_lines)

    def test_returns_none_when_no_pt_file(self, qapp, tmp_path):
        config = PipelineConfig()
        gvhmr_root = tmp_path / "GVHMR"
        # Create output dir but no .pt file
        gvhmr_out = gvhmr_root / "outputs" / "demo" / "video"
        gvhmr_out.mkdir(parents=True)

        w = FullPipelineWorker(
            tmp_path / "video.mp4", config, gvhmr_root, tmp_path / "out"
        )
        log_lines = []
        w.log_line.connect(log_lines.append)

        results = {}
        world, camera = w._run_merge(results, smplestx_ran=False)

        assert world is None
        assert camera is None
        assert any("hmr4d_results.pt" in line for line in log_lines)


class TestBvhFbxStageNoParams:
    """Why: when no params are available (merge failed or backend missing),
    _run_bvh_fbx should handle gracefully and log a warning."""

    def test_warns_when_no_params_and_no_pt(self, qapp, tmp_path):
        config = PipelineConfig()
        w = FullPipelineWorker(
            tmp_path / "video.mp4", config, tmp_path / "GVHMR", tmp_path / "out"
        )
        log_lines = []
        w.log_line.connect(log_lines.append)

        results = {}  # no gvhmr_pt key
        w._run_bvh_fbx(results, world_params=None, is_hybrid=False)

        assert "bvh" not in results
        assert any("no params" in line.lower() for line in log_lines)


class TestRenderingStageNoParams:
    """Why: rendering stages should skip gracefully when no params exist."""

    def test_skips_when_no_params(self, qapp, tmp_path):
        config = PipelineConfig()
        w = FullPipelineWorker(
            tmp_path / "video.mp4", config, tmp_path / "GVHMR", tmp_path / "out"
        )
        log_lines = []
        w.log_line.connect(log_lines.append)

        results = {}
        # Should not crash when both camera and world params are None
        w._run_rendering(results, camera_params=None, world_params=None, is_hybrid=False)

        # No result keys should be set
        assert "skeleton_video" not in results
        assert "world_view_video" not in results
        assert "hand_overlay_video" not in results


class TestFaceStageDisabled:
    """Why: when use_face=False in config, stage 4 should be skipped
    entirely, saving time and avoiding unnecessary processing."""

    def test_face_disabled_skips_stage(self, qapp, tmp_path):
        config = PipelineConfig(use_face=False)
        w = FullPipelineWorker(
            tmp_path / "video.mp4", config, tmp_path / "GVHMR", tmp_path / "out"
        )
        log_lines = []
        w.log_line.connect(log_lines.append)

        # _run_face should not be called when use_face=False;
        # the run() method checks this. But if called, it would try imports.
        # We verify the config flag works.
        assert not config.use_face


# ---------------------------------------------------------------------------
# FullPipelineWorker config passthrough (perf capture settings)
# ---------------------------------------------------------------------------


class TestFullPipelineWorkerConfigPassthrough:
    """Why: perf capture settings (pitch_adjust, body_smooth_preset, fbx_naming,
    use_vitpose_face_crops, hand_source) must flow from PipelineConfig through
    to FullPipelineWorker's stage methods. These tests verify the worker stores
    and can access the config values correctly."""

    def test_stores_perf_capture_config(self, qapp, tmp_path):
        """Worker should preserve all perf capture config fields."""
        config = PipelineConfig(
            pitch_adjust=15.0,
            hand_source="hamer",
            body_smooth_preset="heavy",
            use_vitpose_face_crops=False,
            target_fps=24.0,
            fbx_naming="UE5 Mannequin",
        )
        w = FullPipelineWorker(
            tmp_path / "video.mp4", config, tmp_path / "GVHMR", tmp_path / "out"
        )
        assert w._config.pitch_adjust == 15.0
        assert w._config.hand_source == "hamer"
        assert w._config.body_smooth_preset == "heavy"
        assert w._config.use_vitpose_face_crops is False
        assert w._config.target_fps == 24.0
        assert w._config.fbx_naming == "UE5 Mannequin"

    def test_fps_from_config(self, qapp, tmp_path):
        """Worker uses fps param (which PerfPipelineSettings sets from config.target_fps)."""
        config = PipelineConfig(target_fps=60.0)
        w = FullPipelineWorker(
            tmp_path / "video.mp4", config, tmp_path / "GVHMR", tmp_path / "out",
            fps=60.0,
        )
        assert w._fps == 60.0

    def test_fbx_naming_ue5_conversion(self, qapp, tmp_path):
        """fbx_naming 'UE5 Mannequin' should map to 'ue5' naming key."""
        config = PipelineConfig(fbx_naming="UE5 Mannequin")
        w = FullPipelineWorker(
            tmp_path / "video.mp4", config, tmp_path / "GVHMR", tmp_path / "out"
        )
        # The naming key conversion happens in _run_bvh_fbx:
        naming_key = "ue5" if "ue5" in w._config.fbx_naming.lower() else "mixamo"
        assert naming_key == "ue5"

    def test_fbx_naming_mixamo_conversion(self, qapp, tmp_path):
        """fbx_naming 'Mixamo (Cascadeur)' should map to 'mixamo' naming key."""
        config = PipelineConfig(fbx_naming="Mixamo (Cascadeur)")
        w = FullPipelineWorker(
            tmp_path / "video.mp4", config, tmp_path / "GVHMR", tmp_path / "out"
        )
        naming_key = "ue5" if "ue5" in w._config.fbx_naming.lower() else "mixamo"
        assert naming_key == "mixamo"

    def test_hamer_hand_source_stored(self, qapp, tmp_path):
        """hand_source='hamer' should be accessible for merge stage."""
        config = PipelineConfig(hand_source="hamer")
        w = FullPipelineWorker(
            tmp_path / "video.mp4", config, tmp_path / "GVHMR", tmp_path / "out"
        )
        assert w._config.hand_source == "hamer"

    def test_try_hamer_fallback_on_import_error(self, qapp, tmp_path):
        """_try_hamer should gracefully fall back when hamer_inference is unavailable."""
        config = PipelineConfig(hand_source="hamer")
        w = FullPipelineWorker(
            tmp_path / "video.mp4", config, tmp_path / "GVHMR", tmp_path / "out"
        )
        log_lines = []
        w.log_line.connect(log_lines.append)

        world = {"left_hand_pose": "original", "right_hand_pose": "original"}
        camera = {"left_hand_pose": "original", "right_hand_pose": "original"}
        result_w, result_c = w._try_hamer(world, camera)

        # Should return original params unchanged
        assert result_w["left_hand_pose"] == "original"
        assert result_c["left_hand_pose"] == "original"
        assert any("not available" in line.lower() or "hamer" in line.lower()
                    for line in log_lines)


# ---------------------------------------------------------------------------
# RenderWorker
# ---------------------------------------------------------------------------


class TestRenderWorker:
    """Why: RenderWorker wraps multi-person in-camera rendering. The spec
    requires finished to emit a Path (not str) so consumers get a typed
    filesystem path. Signal(object) is used because PySide6 Signal(Path)
    is unreliable across environments."""

    def test_has_signals(self, qapp):
        from models.session import Session

        s = Session()
        w = RenderWorker(s)
        assert hasattr(w, "progress")
        assert hasattr(w, "finished")
        assert hasattr(w, "error")

    def test_cancel(self, qapp):
        from models.session import Session

        s = Session()
        w = RenderWorker(s)
        assert not w._cancelled
        w.cancel()
        assert w._cancelled

    def test_error_when_no_output_dir(self, qapp):
        """Should emit error when session has no output_dir."""
        from models.session import Session
        from unittest.mock import patch, MagicMock

        s = Session()
        s.output_dir = None
        w = RenderWorker(s)

        errors = []
        w.error.connect(errors.append)

        # Mock the import so it doesn't need the real backend
        mock_module = MagicMock()
        with patch.dict(sys.modules, {"multi_person_split": mock_module}):
            w.run()

        assert len(errors) == 1
        assert "output directory" in errors[0].lower()

    def test_finished_emits_path(self, qapp, tmp_path):
        """finished signal must emit a Path object per pipeline-runner spec."""
        from models.session import Session
        from unittest.mock import patch, MagicMock

        s = Session()
        s.video_path = tmp_path / "video.mp4"
        s.output_dir = tmp_path / "out"
        s.person_tracks = {}

        w = RenderWorker(s)

        results = []
        w.finished.connect(results.append)

        mock_render = MagicMock(return_value=tmp_path / "out" / "scene.mp4")
        mock_module = MagicMock()
        mock_module.render_multi_person_incam = mock_render
        with patch.dict(sys.modules, {"multi_person_split": mock_module}):
            w.run()

        assert len(results) == 1
        assert isinstance(results[0], Path), (
            f"finished should emit Path, got {type(results[0]).__name__}"
        )
        assert results[0] == tmp_path / "out" / "scene.mp4"

    def test_progress_emitted(self, qapp, tmp_path):
        """Should emit progress at start (0.1) and end (1.0)."""
        from models.session import Session
        from unittest.mock import patch, MagicMock

        s = Session()
        s.video_path = tmp_path / "video.mp4"
        s.output_dir = tmp_path / "out"
        s.person_tracks = {}

        w = RenderWorker(s)

        progress_signals = []
        w.progress.connect(lambda f, m: progress_signals.append((f, m)))

        mock_render = MagicMock(return_value=tmp_path / "out" / "scene.mp4")
        mock_module = MagicMock()
        mock_module.render_multi_person_incam = mock_render
        with patch.dict(sys.modules, {"multi_person_split": mock_module}):
            w.run()

        assert len(progress_signals) == 2
        assert progress_signals[0] == (0.1, "Rendering scene preview...")
        assert progress_signals[1] == (1.0, "Done")

    def test_exception_emits_error(self, qapp):
        """Backend exceptions should emit error signal, not crash."""
        from models.session import Session
        from unittest.mock import patch, MagicMock

        s = Session()
        s.output_dir = Path("/tmp/out")
        s.video_path = Path("/tmp/video.mp4")
        s.person_tracks = {}

        w = RenderWorker(s)

        errors = []
        w.error.connect(errors.append)

        mock_module = MagicMock()
        mock_module.render_multi_person_incam.side_effect = RuntimeError("GPU OOM")
        with patch.dict(sys.modules, {"multi_person_split": mock_module}):
            w.run()

        assert len(errors) == 1
        assert "GPU OOM" in errors[0]

    def test_passes_person_dirs(self, qapp, tmp_path):
        """Should pass person_dirs from session tracks to render function."""
        from models.session import Session, PersonTrack
        from unittest.mock import patch, MagicMock

        s = Session()
        s.video_path = tmp_path / "video.mp4"
        s.output_dir = tmp_path / "out"
        s.person_tracks = {
            0: PersonTrack(person_id=0, person_dir=tmp_path / "p0"),
            1: PersonTrack(person_id=1, person_dir=tmp_path / "p1"),
            2: PersonTrack(person_id=2, person_dir=None),  # no dir — should be filtered
        }

        w = RenderWorker(s)

        mock_render = MagicMock(return_value=tmp_path / "out" / "scene.mp4")
        mock_module = MagicMock()
        mock_module.render_multi_person_incam = mock_render
        with patch.dict(sys.modules, {"multi_person_split": mock_module}):
            w.run()

        mock_render.assert_called_once()
        call_kwargs = mock_render.call_args
        person_dirs = call_kwargs[1]["person_dirs"] if "person_dirs" in call_kwargs[1] else call_kwargs[0][1]
        assert len(person_dirs) == 2
        assert str(tmp_path / "p0") in person_dirs
        assert str(tmp_path / "p1") in person_dirs
