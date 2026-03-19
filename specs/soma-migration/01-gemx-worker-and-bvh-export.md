# Handoff A: GEM-X Worker + SOMA BVH Export (Phases 2 + 5)

**Prerequisite**: Branch `soma/skeleton-registry` with Phase 0-1 complete.
**Run after**: `git pull origin soma/skeleton-registry`

---

## Context

GEM-X replaces the 3-stage GVHMR+SMPLest-X+merge pipeline with a single-pass estimator that outputs native SOMA-77 params. The SOMA BVH exporter converts those params to standard BVH format for Blender/FBX import. These two are paired because the BVH exporter is the first consumer of GEM-X output.

---

## Step 1: Create `workers/gemx_worker.py`

This worker follows the same `SubprocessWorkerBase` pattern as `workers/gvhmr_worker.py`. It runs GEM-X's `demo_soma.py` as a subprocess and parses its output.

```python
"""GEM-X estimation worker — single-pass video → SOMA-77 params.

Why: GEM-X replaces the multi-stage GVHMR+SMPLest-X+merge pipeline with a
single model that estimates body, hands, and face simultaneously in SOMA
format. This eliminates the merge step and produces higher-quality results
for hand and face tracking.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal

from models.pipeline_config import PipelineConfig
from workers._base import SubprocessWorkerBase

logger = logging.getLogger(__name__)

# Stage fraction ranges for GEM-X pipeline (simpler than GVHMR — single pass)
_GEMX_STAGES: list[tuple[float, float, str]] = [
    (0.00, 0.05, "Preprocessing"),
    (0.05, 0.80, "GEM-X estimation"),
    (0.80, 0.90, "BVH/FBX conversion"),
    (0.90, 1.00, "Rendering"),
]


def load_gemx_soma_output(output_dir: Path) -> dict | None:
    """Load SOMA params from GEM-X output directory.

    GEM-X saves results as .npz with keys:
    - poses: (N, 77, 3) axis-angle per joint
    - transl: (N, 3)
    - global_orient: (N, 3)
    - identity_coeffs: (1, 45)
    - scale_params: (1, 68)
    - identity_model_type: str

    Returns dict matching PersonTrack.soma_params format, or None.
    """
    # GEM-X outputs to {output_dir}/soma_results.npz
    for pattern in ["soma_results.npz", "*.npz"]:
        for f in sorted(output_dir.glob(pattern)):
            try:
                data = dict(np.load(str(f), allow_pickle=True))
                if "poses" in data and data["poses"].ndim == 3:
                    result = {
                        "poses": data["poses"].astype(np.float32),
                        "transl": data.get("transl", np.zeros((data["poses"].shape[0], 3))).astype(np.float32),
                        "global_orient": data.get("global_orient", data["poses"][:, 0]).astype(np.float32),
                    }
                    if "identity_coeffs" in data:
                        result["identity_coeffs"] = data["identity_coeffs"].astype(np.float32)
                    if "scale_params" in data:
                        result["scale_params"] = data["scale_params"].astype(np.float32)
                    result["identity_model_type"] = str(data.get("identity_model_type", "mhr"))
                    return result
            except Exception:
                continue
    return None


class GEMXWorker(SubprocessWorkerBase):
    """Run GEM-X estimation as a subprocess, output SOMA-77 params."""

    def __init__(
        self,
        video_path: Path,
        config: PipelineConfig,
        gemx_root: Path,
        output_dir: Path,
        fps: float = 30.0,
        parent=None,
    ):
        super().__init__(parent)
        self._video_path = video_path
        self._config = config
        self._gemx_root = gemx_root
        self._output_dir = output_dir
        self._fps = fps

    def run(self):
        try:
            results: dict = {"video_path": str(self._video_path)}
            self._output_dir.mkdir(parents=True, exist_ok=True)

            # Stage 0: Preprocessing
            self._emit_stage(0)
            if self._cancelled:
                return

            # Stage 1: GEM-X estimation
            self._emit_stage(1)
            cmd = self._gemx_command()
            self.log_line.emit(f"$ {' '.join(cmd)}")
            rc, lines = self._run_subprocess(cmd, self._gemx_root)
            if self._cancelled:
                return
            if rc != 0:
                self.error.emit(f"GEM-X estimation failed (exit {rc})")
                return
            results["gemx_log"] = "\n".join(lines)

            # Load SOMA output
            soma_params = load_gemx_soma_output(self._output_dir)
            if soma_params is None:
                self.error.emit("GEM-X produced no SOMA output")
                return
            n_frames = soma_params["poses"].shape[0]
            self.log_line.emit(f"GEM-X output: {n_frames} frames, 77 joints (SOMA)")
            results["soma_params"] = soma_params
            results["n_frames"] = n_frames

            # Stage 2: BVH/FBX conversion
            self._emit_stage(2)
            self._run_bvh_fbx(results, soma_params)
            if self._cancelled:
                return

            # Stage 3: Rendering
            self._emit_stage(3)
            self.log_line.emit("Rendering skipped (SOMA renderer not yet integrated).")

            self.progress.emit(1.0, "GEM-X pipeline complete")
            results["output_dir"] = str(self._output_dir)
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))

    def _run_bvh_fbx(self, results: dict, soma_params: dict) -> None:
        """Convert SOMA params to BVH, then BVH to FBX."""
        stem = self._video_path.stem
        bvh_path = str(self._output_dir / f"{stem}_soma.bvh")

        try:
            from workers.soma_bvh_export import convert_soma_to_bvh

            convert_soma_to_bvh(
                soma_params=soma_params,
                output_path=bvh_path,
                fps=self._fps,
            )
            results["bvh"] = bvh_path
            self.log_line.emit(f"SOMA BVH written: {bvh_path}")
        except Exception as exc:
            self.log_line.emit(f"WARNING: SOMA BVH conversion failed: {exc}")
            return

        # FBX via Blender bridge (same as SMPL-X path)
        fbx_path = str(self._output_dir / f"{stem}_soma.fbx")
        try:
            from bvh_to_fbx import convert_bvh_to_fbx

            naming_key = "ue5" if "ue5" in self._config.fbx_naming.lower() else "mixamo"
            fbx_log = convert_bvh_to_fbx(bvh_path, fbx_path, fps=self._fps, naming=naming_key)
            self.log_line.emit(fbx_log)
            if "ERROR" not in fbx_log:
                results["fbx"] = fbx_path
        except ImportError:
            self.log_line.emit("WARNING: bvh_to_fbx not available, skipping FBX.")
        except Exception as exc:
            self.log_line.emit(f"WARNING: FBX conversion failed: {exc}")

    def _emit_stage(self, idx: int):
        start, _end, label = _GEMX_STAGES[idx]
        self.progress.emit(start, label)

    def _gemx_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "demo_soma.py",
            f"--video={self._video_path}",
            f"--output_dir={self._output_dir}",
            f"--fps={self._fps}",
        ]
        if self._config.static_cam:
            cmd.append("--static_cam")
        return cmd
```

---

## Step 2: Create `workers/soma_bvh_export.py`

BVH writer for SOMA-77 skeleton. Takes `soma_params` dict, writes standard BVH.

```python
"""SOMA-77 → BVH export.

Why: SOMA uses a unified 77-joint poses tensor (axis-angle) while BVH
requires Euler angles in a specific hierarchy. This module handles the
conversion: build the SOMA-77 skeleton hierarchy, convert axis-angle
rotations to ZXY Euler (BVH standard), and write the BVH file.

The resulting BVH works with the existing bvh_to_fbx.py Blender bridge
for FBX conversion.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from models.skeleton import SOMA_SKELETON

logger = logging.getLogger(__name__)


def _axis_angle_to_euler_zxy(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle (3,) to ZXY Euler degrees for BVH.

    BVH convention uses ZXY rotation order (Zrotation Xrotation Yrotation).
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(aa.astype(np.float64)).as_euler("ZXY", degrees=True).astype(np.float32)


def convert_soma_to_bvh(
    soma_params: dict,
    output_path: str | Path,
    fps: float = 30.0,
    joint_names: tuple[str, ...] | None = None,
    joint_parents: tuple[int, ...] | None = None,
) -> Path:
    """Convert SOMA params to BVH file.

    Parameters
    ----------
    soma_params : dict
        Must contain 'poses' (N, 77, 3) and 'transl' (N, 3).
    output_path : path to write BVH file
    fps : frame rate for BVH timing
    joint_names : override joint names (default: SOMA_SKELETON.joint_names)
    joint_parents : override parents (default: SOMA_SKELETON.joint_parents)

    Returns
    -------
    Path to written BVH file.
    """
    output_path = Path(output_path)
    poses = np.asarray(soma_params["poses"], dtype=np.float32)  # (N, 77, 3)
    transl = np.asarray(soma_params["transl"], dtype=np.float32)  # (N, 3)

    n_frames, n_joints, _ = poses.shape
    names = joint_names or SOMA_SKELETON.joint_names
    parents = joint_parents or SOMA_SKELETON.joint_parents

    assert len(names) == n_joints, f"Joint count mismatch: {len(names)} names vs {n_joints} in poses"
    assert len(parents) == n_joints

    # Get rest-pose offsets
    offsets = np.zeros((n_joints, 3), dtype=np.float32)
    for i, name in enumerate(names):
        off = SOMA_SKELETON.default_offsets.get(name, [0.0, 0.0, 0.0])
        offsets[i] = off

    # Build BVH hierarchy string
    frame_time = 1.0 / fps

    def _write_joint(f, joint_idx, indent):
        """Recursively write HIERARCHY section."""
        name = names[joint_idx]
        off = offsets[joint_idx]
        children = [j for j in range(n_joints) if parents[j] == joint_idx]

        is_root = parents[joint_idx] == -1
        prefix = "ROOT" if is_root else "JOINT"

        if not children and not is_root:
            # End site for leaf joints
            f.write(f"{'  ' * indent}{prefix} {name}\n")
            f.write(f"{'  ' * indent}{{\n")
            f.write(f"{'  ' * (indent + 1)}OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
            if is_root:
                f.write(f"{'  ' * (indent + 1)}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
            else:
                f.write(f"{'  ' * (indent + 1)}CHANNELS 3 Zrotation Xrotation Yrotation\n")
            f.write(f"{'  ' * (indent + 1)}End Site\n")
            f.write(f"{'  ' * (indent + 1)}{{\n")
            f.write(f"{'  ' * (indent + 2)}OFFSET 0.000000 0.010000 0.000000\n")
            f.write(f"{'  ' * (indent + 1)}}}\n")
            f.write(f"{'  ' * indent}}}\n")
            return

        f.write(f"{'  ' * indent}{prefix} {name}\n")
        f.write(f"{'  ' * indent}{{\n")
        f.write(f"{'  ' * (indent + 1)}OFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}\n")
        if is_root:
            f.write(f"{'  ' * (indent + 1)}CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation\n")
        else:
            f.write(f"{'  ' * (indent + 1)}CHANNELS 3 Zrotation Xrotation Yrotation\n")

        for child in children:
            _write_joint(f, child, indent + 1)

        f.write(f"{'  ' * indent}}}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("HIERARCHY\n")

        # Find root (parent == -1)
        root_idx = next(i for i, p in enumerate(parents) if p == -1)
        _write_joint(f, root_idx, 0)

        # MOTION section
        f.write("MOTION\n")
        f.write(f"Frames: {n_frames}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")

        for frame in range(n_frames):
            values = []

            # Write joints in hierarchy order (BFS from root)
            def _write_frame_joint(joint_idx):
                children = [j for j in range(n_joints) if parents[j] == joint_idx]

                euler = _axis_angle_to_euler_zxy(poses[frame, joint_idx])
                is_root = parents[joint_idx] == -1

                if is_root:
                    # Root: position + rotation
                    t = transl[frame]
                    values.extend([t[0], t[1], t[2]])
                values.extend([euler[0], euler[1], euler[2]])

                for child in children:
                    _write_frame_joint(child)

            _write_frame_joint(root_idx)
            f.write(" ".join(f"{v:.6f}" for v in values) + "\n")

    logger.info("SOMA BVH written: %s (%d frames, %d joints)", output_path, n_frames, n_joints)
    return output_path
```

---

## Step 3: Update `workers/pipeline_orchestrator.py`

### 3a. Add GEM-X stage definitions

After the existing `_FULL_STAGES` list (line 26), add:

```python
# Stage fraction ranges for GEM-X pipeline (single-pass, fewer stages).
_GEMX_STAGES: list[tuple[float, float, str]] = [
    (0.00, 0.05, "Preprocessing"),
    (0.05, 0.80, "GEM-X estimation"),
    (0.80, 0.90, "BVH/FBX conversion"),
    (0.90, 1.00, "Rendering"),
]
```

### 3b. Update `FullPipelineWorker._run_bvh_fbx()`

In the `_run_bvh_fbx` method, add a SOMA path at the top before the existing SMPL-X path. Find this line:

```python
    def _run_bvh_fbx(
        self, results: dict, world_params: dict | None, is_hybrid: bool
    ) -> None:
        """Convert SMPL-X params to BVH, then BVH to FBX via Blender."""
```

Replace with:

```python
    def _run_bvh_fbx(
        self, results: dict, world_params: dict | None, is_hybrid: bool
    ) -> None:
        """Convert body params to BVH, then BVH to FBX via Blender.

        Dispatches to SOMA BVH exporter when body_model is 'soma',
        otherwise uses the existing SMPL-X smplx_to_bvh path.
        """
```

Then at the start of the method body, before `stem = ...`, add:

```python
        # SOMA path — use soma_bvh_export when params contain SOMA data
        if world_params is not None and "poses" in world_params:
            self._run_soma_bvh_fbx(results, world_params)
            return
```

### 3c. Add `_run_soma_bvh_fbx` method

Add this method to `FullPipelineWorker` (after `_run_bvh_fbx`):

```python
    def _run_soma_bvh_fbx(self, results: dict, soma_params: dict) -> None:
        """Convert SOMA params to BVH, then BVH to FBX."""
        stem = self._video_path.stem
        bvh_path = str(self._output_dir / f"{stem}_soma.bvh")

        try:
            from workers.soma_bvh_export import convert_soma_to_bvh

            convert_soma_to_bvh(soma_params, bvh_path, fps=self._fps)
            results["bvh"] = bvh_path
            self.log_line.emit(f"SOMA BVH written: {bvh_path}")
        except Exception as exc:
            self.log_line.emit(f"WARNING: SOMA BVH conversion failed: {exc}")
            return

        if self._cancelled:
            return

        # FBX via Blender bridge
        fbx_path = str(self._output_dir / f"{stem}_soma.fbx")
        try:
            from bvh_to_fbx import convert_bvh_to_fbx

            naming_key = "ue5" if "ue5" in self._config.fbx_naming.lower() else "mixamo"
            fbx_log = convert_bvh_to_fbx(bvh_path, fbx_path, fps=self._fps, naming=naming_key)
            self.log_line.emit(fbx_log)
            if "ERROR" not in fbx_log:
                results["fbx"] = fbx_path
                self.log_line.emit(f"FBX written: {fbx_path}")
        except ImportError:
            self.log_line.emit("WARNING: bvh_to_fbx not available, skipping FBX.")
        except Exception as exc:
            self.log_line.emit(f"WARNING: FBX conversion failed: {exc}")
```

---

## Step 4: Add tests

Add a new test file `tests/test_gemx_worker.py`:

```python
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
        assert "ROOT Pelvis" in content

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
```

---

## Step 5: Run tests

```bash
cd ~/bodypipe
python -m pytest tests/ -x -q
```

Expected: All previous tests pass + ~16 new tests.

---

## Step 6: Commit

```bash
git add workers/gemx_worker.py workers/soma_bvh_export.py workers/pipeline_orchestrator.py tests/test_gemx_worker.py
git commit -m "feat: add GEM-X worker and SOMA BVH export (Phases 2 + 5)

New workers/gemx_worker.py: subprocess wrapper for GEM-X single-pass
estimation, outputs native SOMA-77 params. Follows SubprocessWorkerBase
pattern with 4-stage progress.

New workers/soma_bvh_export.py: converts SOMA-77 poses (axis-angle)
to standard BVH format with ZXY Euler rotation order. Uses SOMA_SKELETON
hierarchy from models/skeleton.py.

Updated pipeline_orchestrator.py: FullPipelineWorker dispatches to SOMA
BVH exporter when params contain SOMA 'poses' key.

16 new tests covering BVH output format, SOMA param loading, euler
conversion, and GEMXWorker construction."
```
