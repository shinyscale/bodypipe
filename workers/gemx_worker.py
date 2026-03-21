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

GVHMR_ROOT = Path(__file__).resolve().parents[2] / "GVHMR"
GVHMR_PYTHON = Path("/home/shinyscale/miniconda3/envs/gvhmr/bin/python")

# Stage fraction ranges for GEM-X pipeline (simpler than GVHMR — single pass)
_GEMX_STAGES: list[tuple[float, float, str]] = [
    (0.00, 0.05, "Preprocessing"),
    (0.05, 0.80, "GEM-X estimation"),
    (0.80, 0.90, "BVH/FBX conversion"),
    (0.90, 1.00, "Rendering"),
]


def _load_gvhmr_world_params(hpe_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract world-space orient/transl from a co-located GVHMR result.

    GVHMR's world-grounding (get_body_params_w_Rt_v2 + SimpleVO) is well-
    calibrated with its model.  When GVHMR results exist alongside GEM-X
    output, we can borrow GVHMR's body_params_global for orbit-mode world
    grounding while using GEM-X's superior body/hand/face estimation.

    Search order for hmr4d_results.pt:
    1. Sibling ``demo/`` directory (multi-person layout: person_N/demo/…)
    2. Sibling ``gvhmr_wg/`` directory (single-person GEM-X layout)
    3. Same directory tree as hpe_results.pt
    """
    import torch

    search_dirs = []
    # Multi-person: person_N/gemx_demo/…/hpe_results.pt → person_N/demo/
    for parent in hpe_path.parents:
        demo_dir = parent / "demo"
        if demo_dir.is_dir():
            search_dirs.append(demo_dir)
        gvhmr_wg = parent / "gvhmr_wg"
        if gvhmr_wg.is_dir():
            search_dirs.append(gvhmr_wg)
        if parent.name.startswith("person_"):
            break
    search_dirs.append(hpe_path.parent)

    for d in search_dirs:
        for pt in d.rglob("hmr4d_results.pt"):
            try:
                data = torch.load(str(pt), map_location="cpu", weights_only=False)
                bp_global = data.get("body_params_global")
                if bp_global is None:
                    continue
                go = np.array(bp_global["global_orient"]).astype(np.float32)
                tr = np.array(bp_global["transl"]).astype(np.float32)

                # GVHMR body_params_global is in CV convention (Y-down).
                # The viewport's orbit mode with _data_is_global=True expects
                # GL convention (Y-up).  Conjugate the rotation by the CV→GL
                # flip: rotvec (rx,ry,rz) → (rx,-ry,-rz), transl → (x,-y,-z).
                go[:, 1] *= -1
                go[:, 2] *= -1
                tr[:, 1] *= -1
                tr[:, 2] *= -1

                logger.info("Loaded GVHMR world params from %s", pt)
                return go, tr
            except Exception:
                continue
    return None


def load_gemx_soma_output(output_dir: Path) -> dict | None:
    """Load SOMA params from GEM-X output directory.

    GEM-X outputs ``hpe_results.pt`` (a torch dict) containing:
    - ``body_params_global``: dict with ``body_pose`` (L,63), ``global_orient`` (L,3),
      ``transl`` (L,3), ``identity_coeffs`` (1,45), ``scale_params`` (1,68)
    - ``body_params_incam``: same structure, camera space
    - ``K_fullimg``: camera intrinsics

    Falls back to legacy ``.npz`` format for backward compatibility.

    Returns dict matching PersonTrack.soma_params format, or None.
    """
    import torch

    # Primary: look for hpe_results.pt (the real GEM-X output)
    for pattern in ["**/hpe_results.pt", "**/preprocess/hpe_results.pt", "hpe_results.pt"]:
        for f in sorted(output_dir.glob(pattern)):
            try:
                data = torch.load(str(f), map_location="cpu", weights_only=False)
                # Use incam (camera-space) params — matches GVHMR convention
                # and the viewport's CV→GL coordinate pipeline.
                bp = data.get("body_params_incam") or data.get("body_params_global")
                if bp is None:
                    continue

                body_pose = np.array(bp["body_pose"])    # (L, J*3) — 63 or 228
                global_orient = np.array(bp["global_orient"])  # (L, 3)
                transl = np.array(bp["transl"])            # (L, 3)
                n_frames = body_pose.shape[0]
                n_pose_joints = body_pose.shape[1] // 3   # 21 or 76

                # Extract per-frame confidence from GEM-X logits
                confidences = None
                net_out = data.get("net_outputs", {})
                conf_logits = net_out.get("static_conf_logits")
                if conf_logits is not None:
                    conf_logits = np.array(conf_logits)
                    if conf_logits.ndim == 3:
                        conf_logits = conf_logits[0]  # (N, 6)
                    conf_probs = 1.0 / (1.0 + np.exp(-conf_logits.astype(np.float64)))
                    confidences = conf_probs.mean(axis=1).astype(np.float32)

                # World-space orient/transl for orbit mode.
                # Prefer GVHMR's body_params_global (well-calibrated world-grounding)
                # over GEM-X's (insufficient for large orbits).
                bp_global = data.get("body_params_global", {})
                go_world = np.array(bp_global.get("global_orient", global_orient)).astype(np.float32)
                tr_world = np.array(bp_global.get("transl", transl)).astype(np.float32)

                gvhmr_world = _load_gvhmr_world_params(f)
                if gvhmr_world is not None:
                    gvhmr_go, gvhmr_tr = gvhmr_world
                    # Length must match GEM-X output
                    if len(gvhmr_go) == n_frames:
                        go_world, tr_world = gvhmr_go, gvhmr_tr
                    else:
                        logger.warning(
                            "GVHMR frames %d != GEM-X %d — using GEM-X global",
                            len(gvhmr_go), n_frames,
                        )

                if n_pose_joints <= 21:
                    # SMPL-X format (21 body joints) — use smplx_params path
                    result = {
                        "body_pose": body_pose.reshape(n_frames, n_pose_joints, 3).astype(np.float32),
                        "global_orient": global_orient.astype(np.float32),
                        "transl": transl.astype(np.float32),
                        "global_orient_world": go_world,
                        "transl_world": tr_world,
                        "body_model_type": "smplx",
                    }
                else:
                    # True SOMA format (76 body joints) — build poses (L, 77, 3)
                    body_pose_3 = body_pose.reshape(n_frames, n_pose_joints, 3)
                    go_3 = global_orient.reshape(n_frames, 1, 3)
                    poses = np.concatenate(
                        [go_3, body_pose_3[:, :76]], axis=1
                    ).astype(np.float32)
                    result = {
                        "poses": poses,
                        "transl": transl.astype(np.float32),
                        "global_orient": global_orient.astype(np.float32),
                        "global_orient_world": go_world,
                        "transl_world": tr_world,
                        "body_model_type": "soma",
                    }

                if "identity_coeffs" in bp:
                    result["identity_coeffs"] = np.array(bp["identity_coeffs"]).astype(np.float32)
                if "scale_params" in bp:
                    result["scale_params"] = np.array(bp["scale_params"]).astype(np.float32)
                if "K_fullimg" in data:
                    result["K_fullimg"] = np.array(data["K_fullimg"]).astype(np.float32)
                if confidences is not None:
                    result["confidences"] = confidences

                result["identity_model_type"] = "gemx"
                result["source_file"] = str(f)
                logger.info("Loaded GEM-X output from %s: %d frames", f, n_frames)
                return result
            except Exception as exc:
                logger.debug("Failed to load %s: %s", f, exc)
                continue

    # Fallback: legacy .npz format
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

            # Confirm GEM-X backend is actually being used (not a GVHMR fallback)
            demo_script = self._gemx_root / "scripts" / "demo" / "demo_soma.py"
            self.log_line.emit(f"[GEM-X] Root: {self._gemx_root}")
            self.log_line.emit(f"[GEM-X] Root exists: {self._gemx_root.is_dir()}")
            self.log_line.emit(f"[GEM-X] demo_soma.py exists: {demo_script.is_file()}")
            if not self._gemx_root.is_dir():
                self.error.emit(
                    f"GEM-X root not found: {self._gemx_root} — "
                    "check that GEM-X is cloned as a sibling of GVHMR"
                )
                return
            if not demo_script.is_file():
                self.error.emit(
                    f"GEM-X demo script not found: {demo_script}"
                )
                return

            # Stage 0: Preprocessing — run GVHMR for world-grounding
            self._emit_stage(0)
            if self._cancelled:
                return

            if not self._config.static_cam:
                camera_pt = self._compute_simple_vo()
                if camera_pt is None:
                    self.log_line.emit("[GEM-X] SimpleVO unavailable — static camera fallback.")
                self._run_gvhmr_for_world_grounding()

            # Stage 1: GEM-X estimation
            self._emit_stage(1)
            cmd = self._gemx_command()
            self.log_line.emit(f"[GEM-X] Running: {' '.join(cmd)}")
            self.log_line.emit(f"[GEM-X] cwd: {self._gemx_root}")
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
            n_joints = soma_params["poses"].shape[1] if soma_params["poses"].ndim == 3 else "?"
            self.log_line.emit(
                f"[GEM-X] CONFIRMED: {n_frames} frames, {n_joints} joints (SOMA) — "
                f"this is GEM-X output, NOT GVHMR"
            )
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

    def _compute_simple_vo(self) -> Path | None:
        """Run SimpleVO via GVHMR env subprocess, stage camera.pt for GEM-X.

        Returns the camera.pt path on success, None on failure (graceful fallback).
        """
        # GEM-X expects camera.pt at {output_root}/{video_stem}/preprocess/camera.pt
        video_stem = self._video_path.stem
        camera_pt = self._output_dir / video_stem / "preprocess" / "camera.pt"
        if camera_pt.exists():
            self.log_line.emit(f"[GEM-X] SimpleVO: reusing existing {camera_pt}")
            return camera_pt

        vo_script = GVHMR_ROOT / "tools" / "compute_vo.py"
        if not vo_script.exists():
            self.log_line.emit(f"[GEM-X] SimpleVO: script not found: {vo_script}")
            return None

        gvhmr_python = str(GVHMR_PYTHON) if GVHMR_PYTHON.exists() else sys.executable
        camera_pt.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            gvhmr_python,
            str(vo_script),
            f"--video={self._video_path}",
            f"--output={camera_pt}",
        ]
        self.log_line.emit(f"[GEM-X] SimpleVO: {' '.join(cmd)}")
        rc, lines = self._run_subprocess(cmd, GVHMR_ROOT)
        if self._cancelled:
            return None
        if rc != 0 or not camera_pt.exists():
            self.log_line.emit(f"[GEM-X] SimpleVO failed (exit {rc})")
            return None

        self.log_line.emit(f"[GEM-X] SimpleVO: camera.pt staged at {camera_pt}")
        return camera_pt

    def _run_gvhmr_for_world_grounding(self) -> None:
        """Run GVHMR (lightweight, no render) to produce hmr4d_results.pt.

        GEM-X's internal world-grounding is insufficient for large camera
        orbits.  GVHMR's body_params_global provides world-space root
        orient/transl that the viewport's orbit mode needs.  The GVHMR
        result is loaded at result-loading time by _load_gvhmr_world_params().

        Uses a separate output_root (gvhmr_wg/) to avoid preprocess format
        conflicts — GVHMR and GEM-X have incompatible bbx.pt layouts.
        """
        # Use separate dir to avoid sharing preprocess/ with GEM-X
        gvhmr_output = self._output_dir / "gvhmr_wg"

        # Check if GVHMR already ran
        existing = list(gvhmr_output.rglob("hmr4d_results.pt"))
        if existing:
            self.log_line.emit(f"[GEM-X] GVHMR world-grounding: reusing {existing[0]}")
            return

        demo_script = GVHMR_ROOT / "tools" / "demo" / "demo.py"
        if not demo_script.is_file():
            self.log_line.emit("[GEM-X] GVHMR demo.py not found — skipping world-grounding")
            return

        gvhmr_python = str(GVHMR_PYTHON) if GVHMR_PYTHON.exists() else sys.executable
        cmd = [
            gvhmr_python,
            str(demo_script),
            f"--video={self._video_path}",
            f"--output_root={gvhmr_output}",
            "--skip_render",
        ]
        if self._config.static_cam:
            cmd.append("--static_cam")

        self.log_line.emit(f"[GEM-X] Running GVHMR for world-grounding: {' '.join(cmd)}")
        rc, lines = self._run_subprocess(cmd, GVHMR_ROOT)
        if self._cancelled:
            return
        if rc != 0:
            self.log_line.emit(f"[GEM-X] GVHMR world-grounding failed (exit {rc}) — orbit mode may lazy-susan")

    def _gemx_command(self) -> list[str]:
        cmd = [
            sys.executable,
            "scripts/demo/demo_soma.py",
            f"--video={self._video_path}",
            f"--output_root={self._output_dir}",
        ]
        if self._config.static_cam:
            cmd.append("--static_cam")
        return cmd
