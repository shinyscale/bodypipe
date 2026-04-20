"""GEM-X estimation worker — single-pass video → SOMA-77 params.

Why: GEM-X replaces the multi-stage GVHMR+SMPLest-X+merge pipeline with a
single model that estimates body, hands, and face simultaneously in SOMA
format. This eliminates the merge step and produces higher-quality results
for hand and face tracking.
"""

from __future__ import annotations

import time

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


def _load_gvhmr_world_params(hpe_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract world-space orient/transl from a co-located GVHMR result.

    GVHMR's world-grounding (get_body_params_w_Rt_v2 + SimpleVO) is well-
    calibrated with its model.  When GVHMR results exist alongside GEM-X
    output, we can borrow GVHMR's smpl_params_global for orbit-mode world
    grounding while using GEM-X's superior body/hand/face estimation.

    Returns (global_orient, transl) both as Y-up np.float32, or None.

    Search order for hmr4d_results.pt:
    1. Sibling ``demo/`` directory (multi-person layout: person_N/demo/…)
    2. Sibling ``gvhmr_wg/`` directory (single-person GEM-X layout)
    3. Same directory tree as hpe_results.pt
    """
    import torch

    search_dirs = []
    # Multi-person: person_N/gemx_demo/…/hpe_results.pt → person_N/demo/
    for parent in hpe_path.parents:
        for subdir in ("demo", "demo.bak", "gvhmr_wg"):
            d = parent / subdir
            if d.is_dir():
                search_dirs.append(d)
        if parent.name.startswith("person_"):
            break
    search_dirs.append(hpe_path.parent)

    for d in search_dirs:
        for pt in d.rglob("hmr4d_results.pt"):
            try:
                data = torch.load(str(pt), map_location="cpu", weights_only=False)
                # GVHMR uses "smpl_params_global", not "body_params_global"
                bp_global = data.get("smpl_params_global") or data.get("body_params_global")
                if bp_global is None:
                    continue
                go_global = np.array(bp_global["global_orient"]).astype(np.float32)
                tr_global = np.array(bp_global["transl"]).astype(np.float32)
                # GVHMR global params are already Y-up — no flip needed.
                logger.info("Loaded GVHMR world params from %s", pt)
                return go_global, tr_global
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
                gemx_global_present = bool(
                    bp_global
                    and "transl" in bp_global
                    and "global_orient" in bp_global
                )
                go_world = np.array(bp_global.get("global_orient", global_orient)).astype(np.float32)
                tr_world = np.array(bp_global.get("transl", transl)).astype(np.float32)
                world_source = "gemx_global" if gemx_global_present else "camera_fallback"

                gvhmr_world = _load_gvhmr_world_params(f)
                if gvhmr_world is not None:
                    gvhmr_go_global, gvhmr_tr_global = gvhmr_world
                    if len(gvhmr_go_global) == n_frames:
                        # Both orient and transl from GVHMR — already Y-up,
                        # matching the orbit camera's world_up = [0, 1, 0].
                        go_world = gvhmr_go_global
                        tr_world = gvhmr_tr_global
                        world_source = "gvhmr"
                    else:
                        logger.warning(
                            "GVHMR frames %d != GEM-X %d — using GEM-X global",
                            len(gvhmr_go_global), n_frames,
                        )

                # Ground-normalize: shift transl_world Y so feet land at Y=0.
                # FK uses mean-shape bone offsets; leg length differs by format.
                _LEG_LENGTH = 0.908 if n_pose_joints > 21 else 0.933
                if tr_world.shape[0] > 0:
                    floor_y = float(tr_world[0, 1]) - _LEG_LENGTH
                    tr_world = tr_world.copy()
                    tr_world[:, 1] -= floor_y

                has_world_grounding = world_source != "camera_fallback"

                # Diagnostic: incam vs world path lengths so world-grounding
                # regressions surface in the worker log instead of silently
                # producing camera-space data labelled as world.
                if tr_world.shape[0] > 1 and transl.shape[0] > 1:
                    incam_path = float(np.linalg.norm(np.diff(transl, axis=0), axis=1).sum())
                    world_path = float(np.linalg.norm(np.diff(tr_world, axis=0), axis=1).sum())
                    logger.info(
                        "GEM-X translation: source=%s, incam path=%.2fm, world path=%.2fm",
                        world_source, incam_path, world_path,
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
                result["has_world_grounding"] = has_world_grounding
                result["world_source"] = world_source
                logger.info(
                    "Loaded GEM-X output from %s: %d frames, world_source=%s",
                    f, n_frames, world_source,
                )
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

            # Stage 0: Preprocessing
            self._emit_stage(0)
            if self._cancelled:
                return

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

            results["stage_timings"] = self._emit_timing_summary()
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
        now = time.monotonic()
        if hasattr(self, "_stage_start_time") and self._current_stage_idx >= 0:
            elapsed = now - self._stage_start_time
            prev_label = _GEMX_STAGES[self._current_stage_idx][2]
            self._stage_timings.append((prev_label, elapsed))
        else:
            self._stage_timings: list[tuple[str, float]] = []
            self._pipeline_start_time = now
        self._current_stage_idx = idx
        self._stage_start_time = now
        start, _end, label = _GEMX_STAGES[idx]
        self.progress.emit(start, label)

    def _emit_timing_summary(self):
        """Close the last stage and emit a timing summary to the log."""
        now = time.monotonic()
        if hasattr(self, "_stage_start_time") and self._current_stage_idx >= 0:
            elapsed = now - self._stage_start_time
            label = _GEMX_STAGES[self._current_stage_idx][2]
            self._stage_timings.append((label, elapsed))
        total = now - getattr(self, "_pipeline_start_time", now)
        self.log_line.emit("")
        self.log_line.emit("── Stage Timings ──")
        for label, secs in self._stage_timings:
            self.log_line.emit(f"  {label:<28s} {secs:6.1f}s")
        self.log_line.emit(f"  {'TOTAL':<28s} {total:6.1f}s")
        self.log_line.emit("")
        return {label: round(secs, 2) for label, secs in self._stage_timings}

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
