"""Orchestration workers for multi-stage pipelines."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from models.pipeline_config import PipelineConfig
from workers._base import SubprocessWorkerBase

logger = logging.getLogger(__name__)

# Stage fraction ranges for FullPipelineWorker — (start, end, label).
# Ranges are contiguous and span [0, 1].
_FULL_STAGES: list[tuple[float, float, str]] = [
    (0.00, 0.02, "Preprocessing"),
    (0.02, 0.35, "GVHMR body solve"),
    (0.35, 0.50, "SMPLest-X hand solve"),
    (0.50, 0.52, "Merging body + hands"),
    (0.52, 0.60, "Motion refinement"),
    (0.60, 0.75, "Face pipeline"),
    (0.75, 0.88, "BVH/FBX conversion"),
    (0.88, 1.00, "Rendering"),
]

# Stage fraction ranges for GEM-X pipeline (single-pass, fewer stages).
_GEMX_STAGES: list[tuple[float, float, str]] = [
    (0.00, 0.05, "Preprocessing"),
    (0.05, 0.80, "GEM-X estimation"),
    (0.80, 0.90, "BVH/FBX conversion"),
    (0.90, 1.00, "Rendering"),
]


def find_sandpipe_physics(output_dir: Path, video_path: Path) -> Path | None:
    """Discover sandpipe-physics.json for a video.

    Search order:
    1. ``output_dir / "sandpipe-physics.json"``
    2. Walk up from output_dir looking for
       ``research/videos/<video_stem>/sandpipe-physics.json``
    """
    # Direct: alongside outputs
    direct = output_dir / "sandpipe-physics.json"
    if direct.is_file():
        return direct

    # Research tree: look for research/videos/<stem>/sandpipe-physics.json
    stem = video_path.stem
    cur = output_dir
    for _ in range(8):  # don't walk up forever
        candidate = cur / "research" / "videos" / stem / "sandpipe-physics.json"
        if candidate.is_file():
            return candidate
        parent = cur.parent
        if parent == cur:
            break
        cur = parent

    return None


def find_smplestx_result(output_dir: Path) -> Path | None:
    """Return the most recently modified .pt or .npz in *output_dir*."""
    candidates = list(output_dir.glob("*.pt")) + list(output_dir.glob("*.npz"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def save_merged_pt(params: dict, output_path: Path) -> Path:
    """Save merged SMPL-X params dict as a .pt file for re-export.

    Mirrors ``gvhmr_gui.py:_save_merged_pt()`` — converts numpy arrays to
    torch tensors and reshapes pose arrays to ``(F, -1)`` layout.
    """
    import torch

    n = params["num_frames"]
    save_dict = {
        "global_orient": torch.tensor(params["global_orient"], dtype=torch.float32),
        "body_pose": torch.tensor(
            params["body_pose"].reshape(n, -1), dtype=torch.float32
        ),
        "left_hand_pose": torch.tensor(
            params["left_hand_pose"].reshape(n, -1), dtype=torch.float32
        ),
        "right_hand_pose": torch.tensor(
            params["right_hand_pose"].reshape(n, -1), dtype=torch.float32
        ),
        "transl": torch.tensor(params["transl"], dtype=torch.float32),
        "coordinate_space": params.get("coordinate_space", "world"),
        "camera_model": params.get("camera_model", "world_space"),
        "translation_origin": params.get("translation_origin", "model_origin"),
        "source": params.get("source", "hybrid"),
    }
    if "betas" in params:
        save_dict["betas"] = torch.tensor(params["betas"], dtype=torch.float32)
    if "bbox" in params:
        save_dict["bbox"] = torch.tensor(params["bbox"], dtype=torch.float32)
    for wrist_key in ["left_wrist_orient", "right_wrist_orient"]:
        if wrist_key in params:
            save_dict[wrist_key] = torch.tensor(params[wrist_key], dtype=torch.float32)
    for key in [
        "transl_cam", "transl_world",
        "global_orient_cam", "global_orient_world",
        "body_pose_cam", "body_pose_world",
        "transl_world_baseline", "transl_world_physics", "transl_world_spring",
        "transl_world_spring_unpin",
        "global_orient_world_baseline", "global_orient_world_physics",
        "global_orient_world_spring",
        "body_pose_world_baseline", "body_pose_world_physics",
        "body_pose_world_spring",
        "K_fullimg",
    ]:
        if key in params:
            value = params[key]
            if key.startswith("body_pose"):
                value = value.reshape(n, -1)
            save_dict[key] = torch.tensor(value, dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, str(output_path))
    return output_path


def extract_bboxes_from_output(output_dir: Path):
    """Extract person bounding boxes from SMPLest-X output files.

    Returns a numpy array of shape ``(N, 4)`` or ``None``.
    Mirrors ``gvhmr_gui.py:_extract_bboxes_from_smplestx()``.
    """
    try:
        import numpy as np
        import torch
    except ImportError:
        return None

    for pattern in ["*.pt", "*.npz"]:
        for f in sorted(output_dir.rglob(pattern)):
            try:
                if f.suffix == ".pt":
                    data = torch.load(str(f), map_location="cpu", weights_only=False)
                else:
                    data = dict(np.load(str(f), allow_pickle=True))
                for key in ["person_bbox", "bboxes", "bbox", "bb_xyxy", "pred_bboxes"]:
                    if key in data:
                        return np.array(data[key]).reshape(-1, 4)
            except Exception:
                continue
    return None


def fallback_face_bboxes(video_path: str | Path):
    """Generate approximate face bboxes using upper portion of frame.

    Fallback when SMPLest-X bbox data isn't available.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    bboxes = np.zeros((n_frames, 4))
    bboxes[:, 0] = w * 0.2   # x1
    bboxes[:, 1] = 0          # y1
    bboxes[:, 2] = w * 0.8   # x2
    bboxes[:, 3] = h * 0.6   # y2
    return bboxes


class FullPipelineWorker(SubprocessWorkerBase):
    """Orchestrates GVHMR + SMPLest-X + merge + face + BVH/FBX + render."""

    def __init__(
        self,
        video_path: Path,
        config: PipelineConfig,
        gvhmr_root: Path,
        output_dir: Path,
        fps: float = 30.0,
        smplestx_python: str | Path | None = None,
        smplestx_dir: str | Path | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._video_path = video_path
        self._config = config
        self._gvhmr_root = gvhmr_root
        self._output_dir = output_dir
        self._fps = fps
        self._smplestx_python = str(smplestx_python) if smplestx_python else None
        self._smplestx_dir = str(smplestx_dir) if smplestx_dir else None

    def run(self):
        try:
            results: dict = {"video_path": str(self._video_path)}
            self._output_dir.mkdir(parents=True, exist_ok=True)

            # Stage 0: Preprocessing
            self._emit_stage(0)
            if self._cancelled:
                return

            # Stage 1: GVHMR body solve
            self._emit_stage(1)
            gvhmr_cmd = self._gvhmr_command()
            self.log_line.emit(f"$ {' '.join(gvhmr_cmd)}")
            rc, lines = self._run_subprocess(gvhmr_cmd, self._gvhmr_root)
            if self._cancelled:
                return
            if rc != 0:
                self.error.emit(f"GVHMR body solve failed (exit {rc})")
                return
            results["gvhmr_log"] = "\n".join(lines)

            # Stage 2: SMPLest-X hand solve (if enabled and configured)
            smplestx_ran = False
            if self._config.use_hands and self._smplestx_python and self._smplestx_dir:
                self._emit_stage(2)
                smplx_cmd = self._smplestx_command()
                self.log_line.emit(f"$ {' '.join(smplx_cmd)}")
                rc, lines = self._run_subprocess(smplx_cmd, self._smplestx_dir)
                if self._cancelled:
                    return
                if rc != 0:
                    self.error.emit(f"SMPLest-X hand solve failed (exit {rc})")
                    return
                results["smplestx_log"] = "\n".join(lines)
                smplestx_ran = True

            # Stage 3: Merge body + hands
            self._emit_stage(3)
            world_params, camera_params = self._run_merge(results, smplestx_ran)
            if self._cancelled:
                return

            # Stage 4: Motion refinement (spring filter or dormant physics)
            self._emit_stage(4)
            # Camera stabilization: re-derive world params from smoothed SLAM
            if (
                self._config.use_camera_stabilize
                and world_params is not None
                and results.get("gvhmr_pt")
            ):
                try:
                    from workers.spring_refine.camera_stabilize import stabilize_world_params

                    stabilized = stabilize_world_params(
                        Path(results["gvhmr_pt"]),
                        cam_smooth_preset=self._config.cam_smooth_preset,
                        fps=float(self._fps),
                    )
                    if stabilized is not None:
                        world_params["global_orient"] = stabilized["global_orient"]
                        world_params["transl"] = stabilized["transl"]
                        self.log_line.emit("Camera stabilization applied.")
                except Exception as exc:
                    self.log_line.emit(f"WARNING: Camera stabilization failed: {exc}")

            if (
                (self._config.use_spring_refine or self._config.use_foot_pin)
                and world_params is not None
            ):
                world_params, spring_ok = self._run_spring_refine(world_params)
                if spring_ok:
                    snapshot_source = (
                        "spring_refined_pinned"
                        if world_params.get("source") == "spring_refined_pinned"
                        else "spring_refined"
                    )
                    self._save_viewport_params_snapshot(
                        results, world_params, camera_params, source=snapshot_source
                    )
            elif self._config.use_physics_refine and world_params is not None:
                world_params, physics_refined_ok = self._run_physics_refine(world_params)
                # Only write the refined viewport snapshot when PHC actually
                # ran — otherwise _run_physics_refine silently returns the
                # original baseline params and we would mislabel the snapshot
                # as `source=phc_refined` over unrefined data.
                if physics_refined_ok:
                    self._save_viewport_params_snapshot(
                        results, world_params, camera_params, source="phc_refined"
                    )
            elif self._config.use_physics_refine or self._config.use_spring_refine:
                self.log_line.emit("Motion refinement enabled but no params available, skipping.")
            else:
                self.log_line.emit("Motion refinement disabled, skipping.")
            if self._cancelled:
                return

            # Stage 5: Face pipeline
            self._emit_stage(5)
            if self._config.use_face:
                self._run_face(results)
            else:
                self.log_line.emit("Face capture disabled, skipping.")
            if self._cancelled:
                return

            # Stage 6: BVH/FBX conversion
            self._emit_stage(6)
            self._run_bvh_fbx(results, world_params, smplestx_ran)
            if self._cancelled:
                return

            # Stage 7: Rendering
            self._emit_stage(7)
            self._run_rendering(results, camera_params, world_params, smplestx_ran)
            if self._cancelled:
                return

            results["stage_timings"] = self._emit_timing_summary()
            self.progress.emit(1.0, "Pipeline complete")
            results["output_dir"] = str(self._output_dir)
            self.finished.emit(results)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Stage 3: Merge GVHMR body + SMPLest-X hands
    # ------------------------------------------------------------------

    def _run_merge(
        self, results: dict, smplestx_ran: bool
    ) -> tuple[dict | None, dict | None]:
        """Merge GVHMR body params with SMPLest-X hand params.

        Returns ``(world_params, camera_params)`` — both may be ``None``
        if extraction fails (backend unavailable or outputs missing).
        """
        from workers.gvhmr_worker import find_output_dir, find_file

        self.log_line.emit("Merging GVHMR and SMPLest-X parameters...")

        # Find GVHMR output
        gvhmr_out_dir = find_output_dir(self._video_path, self._gvhmr_root)
        if gvhmr_out_dir is None:
            self.log_line.emit("WARNING: GVHMR output directory not found.")
            return None, None

        gvhmr_pt = find_file(gvhmr_out_dir, "hmr4d_results.pt")
        if gvhmr_pt is None:
            self.log_line.emit("WARNING: hmr4d_results.pt not found in GVHMR output.")
            return None, None

        self.log_line.emit(f"GVHMR output: {gvhmr_pt}")
        results["gvhmr_pt"] = str(gvhmr_pt)

        try:
            from smplx_to_bvh import (
                extract_gvhmr_params,
                extract_smplx_params,
                merge_gvhmr_smplestx_params,
            )

            gvhmr_params = extract_gvhmr_params(str(gvhmr_pt))
            self.log_line.emit(
                f"GVHMR params: {gvhmr_params['num_frames']} frames"
            )

            if smplestx_ran:
                smplestx_pt = find_smplestx_result(self._output_dir)
                if smplestx_pt is not None:
                    self.log_line.emit(f"SMPLest-X output: {smplestx_pt}")
                    smplestx_params = extract_smplx_params(str(smplestx_pt))
                    self.log_line.emit(
                        f"SMPLest-X params: {smplestx_params['num_frames']} frames"
                    )

                    world_params = merge_gvhmr_smplestx_params(
                        gvhmr_params, smplestx_params, coordinate_space="world"
                    )
                    camera_params = merge_gvhmr_smplestx_params(
                        gvhmr_params, smplestx_params, coordinate_space="camera"
                    )
                    self.log_line.emit(
                        f"Merged hybrid: {world_params['num_frames']} frames"
                    )

                    # HaMeR hand replacement (if selected)
                    if self._config.hand_source == "hamer":
                        world_params, camera_params = self._try_hamer(
                            world_params, camera_params
                        )

                    # Save merged .pt for re-export
                    stem = self._video_path.stem
                    merged_path = self._output_dir / f"{stem}_hybrid_smplx.pt"
                    save_merged_pt(world_params, merged_path)
                    results["merged_pt"] = str(merged_path)
                    self.log_line.emit(f"Saved merged: {merged_path}")
                    return world_params, camera_params

                self.log_line.emit(
                    "WARNING: SMPLest-X output not found, using GVHMR only."
                )

            # GVHMR-only — use its params for both world and camera
            self.log_line.emit("Using GVHMR params (no hand merge).")
            return gvhmr_params, gvhmr_params

        except ImportError:
            self.log_line.emit(
                "WARNING: smplx_to_bvh not available, skipping merge."
            )
        except Exception as exc:
            self.log_line.emit(f"WARNING: Merge failed: {exc}")
        return None, None

    def _try_hamer(
        self, world_params: dict, camera_params: dict
    ) -> tuple[dict, dict]:
        """Run HaMeR hand reconstruction and merge into existing params.

        Falls back to original params if HaMeR is unavailable or fails.
        Mirrors Gradio's HaMeR integration in ``run_full_pipeline()``.
        """
        try:
            from hamer_inference import run_hamer
        except ImportError:
            self.log_line.emit("WARNING: hamer_inference not available, using SMPLest-X hands.")
            return world_params, camera_params

        try:
            self.log_line.emit("Running HaMeR hand reconstruction...")
            # Find ViTPose from GVHMR preprocessing
            from workers.gvhmr_worker import find_output_dir, find_file
            vitpose_pt = None
            gvhmr_out = find_output_dir(self._video_path, self._gvhmr_root)
            if gvhmr_out:
                vp = find_file(gvhmr_out, "vitpose.pt")
                if vp:
                    vitpose_pt = str(vp)
                    self.log_line.emit(f"Using ViTPose for HaMeR: {vp}")
            hamer_result = run_hamer(
                video_path=str(self._video_path),
                vitpose_path=vitpose_pt,
            )
            if hamer_result is not None:
                # Merge HaMeR hand poses + wrist orient into world and camera params
                for params in (world_params, camera_params):
                    if params is None:
                        continue
                    for key in ["left_hand_pose", "right_hand_pose",
                                "left_wrist_orient", "right_wrist_orient"]:
                        if key in hamer_result:
                            params[key] = hamer_result[key]
                self.log_line.emit("HaMeR hands merged successfully.")
            else:
                self.log_line.emit("WARNING: HaMeR returned no results, keeping SMPLest-X hands.")
        except Exception as exc:
            self.log_line.emit(f"WARNING: HaMeR failed: {exc}, keeping SMPLest-X hands.")

        return world_params, camera_params

    # ------------------------------------------------------------------
    # Stage 4: Physics refinement
    # ------------------------------------------------------------------

    def _run_physics_refine(self, world_params: dict) -> tuple[dict, bool]:
        """Run PHC physics refinement on world-space params.

        Runs synchronously (not as nested QThread) since we're already on a
        worker thread. Returns ``(params, refined_ok)`` where ``refined_ok``
        is ``True`` only when PHC actually produced refined data.  On any
        failure the original ``world_params`` is returned with
        ``refined_ok=False`` so downstream stages still have valid data, but
        the caller can tell whether to label a viewport snapshot as refined.
        """
        try:
            from workers.physics.gvhmr_to_amass import params_to_amass_npz
            from workers.physics.phc_runner import run_phc_local
            from workers.physics.phc_to_smpl import phc_output_to_params
            from workers.physics.evaluate import (
                compute_verdict,
                evaluate_refinement,
                write_metrics_json,
            )
        except ImportError as exc:
            self.log_line.emit(f"WARNING: Physics modules not available: {exc}")
            return world_params, False

        progress_cb = self._stage_progress_callback(4)

        try:
            # Convert to AMASS
            phc_dir = self._output_dir / "physics"
            phc_dir.mkdir(parents=True, exist_ok=True)
            input_npz = phc_dir / "phc_input.npz"
            params_to_amass_npz(world_params, input_npz, fps=int(self._fps))
            self.log_line.emit(f"Physics: AMASS input saved -> {input_npz}")
            progress_cb(0.1, "Running PHC...")

            # Run PHC
            phc_output_dir = phc_dir / "phc_output"
            result = run_phc_local(input_npz, phc_output_dir)

            if not result.success:
                self.log_line.emit(
                    f"WARNING: PHC failed, using original params. "
                    f"Log: {result.log[-300:]}"
                )
                return world_params, False

            progress_cb(0.8, "Converting PHC output...")
            refined = phc_output_to_params(result.output_path, world_params)

            # Evaluate
            progress_cb(0.9, "Evaluating refinement...")
            try:
                metrics = evaluate_refinement(world_params, refined, fps=self._fps)
                for key, val in metrics.items():
                    if isinstance(val, dict):
                        for sk, sv in val.items():
                            if isinstance(sv, (int, float)):
                                self.log_line.emit(f"  Physics {key}.{sk}: {sv:.4f}")
                    elif isinstance(val, float):
                        self.log_line.emit(f"  Physics {key}: {val:.4f}")
                metrics_path = phc_dir / "metrics.json"
                write_metrics_json(metrics, metrics_path)
                status, reason = compute_verdict(metrics)
                verdict_msg = f"Physics verdict: {status.upper()}"
                if reason:
                    verdict_msg += f" — {reason}"
                self.log_line.emit(verdict_msg)
                self.log_line.emit(f"Physics metrics saved -> {metrics_path}")
            except Exception as exc:
                self.log_line.emit(f"WARNING: Physics metrics failed: {exc}")

            progress_cb(1.0, "Physics refinement complete")
            self.log_line.emit(
                f"Physics refinement: {refined['num_frames']} frames refined"
            )
            return refined, True

        except Exception as exc:
            self.log_line.emit(f"WARNING: Physics refinement failed: {exc}")
            return world_params, False

    def _run_spring_refine(self, world_params: dict) -> tuple[dict, bool]:
        """Run spring-based motion refinement on world-space params.

        In-process (no subprocess), pure numpy. Returns
        ``(refined_params, ok)`` where ``ok`` is True only when the filter
        produced refined data. On failure the original ``world_params`` is
        returned with ``ok=False`` so downstream stages still have valid
        data, but the caller can tell whether to label a viewport snapshot
        as refined.

        v2: honours ``use_foot_pin`` by passing the flag through to
        ``run_spring_refine`` and writes a ``spring_refine/metrics.json``
        with before/after foot-skating in meters.
        """
        import numpy as np

        try:
            from workers.spring_refine import run_spring_refine
        except ImportError as exc:
            self.log_line.emit(f"WARNING: Spring refine module not available: {exc}")
            return world_params, False

        # Preserve the pre-refine baseline so we can compute metrics and
        # persist a pre-pin transl alongside the pinned output.
        baseline_params = dict(world_params)
        baseline_transl = np.asarray(world_params.get("transl")).copy() if world_params.get("transl") is not None else None

        progress_cb = self._stage_progress_callback(4)
        try:
            progress_cb(0.1, "Running spring filter...")
            sp_path = None
            if self._config.use_sandpipe:
                sp_path = find_sandpipe_physics(self._output_dir, self._video_path)
                if sp_path:
                    self.log_line.emit(f"Sandpipe physics found: {sp_path}")

            refined, ok = run_spring_refine(
                world_params,
                preset=self._config.spring_refine_preset,
                fps=float(self._fps),
                progress_cb=lambda f: progress_cb(0.1 + 0.8 * f, "Running spring filter..."),
                pin_feet=bool(self._config.use_foot_pin),
                filter_rotations=bool(self._config.use_spring_refine),
                foot_pin_sensitivity=self._config.foot_pin_sensitivity,
                foot_pin_strength=float(self._config.foot_pin_strength),
                sandpipe_physics_path=str(sp_path) if sp_path else None,
            )
            if not ok:
                self.log_line.emit(
                    "WARNING: Spring refine returned no data, using original params."
                )
                return world_params, False

            # Persist the pre-pin transl alongside the pinned output so
            # future A/B diffs don't require a re-run. Only meaningful
            # when the pin ran — otherwise the two would be identical.
            if (
                self._config.use_foot_pin
                and refined.get("source") == "spring_refined_pinned"
                and baseline_transl is not None
            ):
                refined["transl_world_spring_unpin"] = baseline_transl

            # Write metrics JSON (before/after foot skating + verdict).
            try:
                from workers.spring_refine import write_spring_metrics_json

                metrics_dir = self._output_dir / "spring_refine"
                metrics_path = metrics_dir / "metrics.json"
                payload = write_spring_metrics_json(
                    baseline_params=baseline_params,
                    refined_params=refined,
                    fps=float(self._fps),
                    preset=self._config.spring_refine_preset,
                    pin_enabled=bool(self._config.use_foot_pin),
                    out_path=metrics_path,
                    pin_stats=refined.get("foot_pin_stats"),
                )
                verdict = payload.get("verdict", "unknown")
                delta = payload.get("foot_skating_delta_m")
                delta_str = f"{delta * 100:.2f} cm" if isinstance(delta, (int, float)) and not np.isnan(delta) else "n/a"
                self.log_line.emit(
                    f"Spring verdict: {verdict.upper()} "
                    f"(foot-skating delta={delta_str}) -> {metrics_path}"
                )
            except Exception as metrics_exc:
                self.log_line.emit(
                    f"WARNING: Spring metrics failed: {metrics_exc}"
                )

            progress_cb(1.0, "Spring refinement complete")
            self.log_line.emit(
                f"Spring refinement: {refined['num_frames']} frames, "
                f"preset={self._config.spring_refine_preset}, "
                f"pin={'on' if self._config.use_foot_pin else 'off'}"
            )
            return refined, True
        except Exception as exc:
            self.log_line.emit(f"WARNING: Spring refinement failed: {exc}")
            return world_params, False

    def _save_viewport_params_snapshot(
        self,
        results: dict,
        world_params: dict,
        camera_params: dict | None,
        source: str = "phc_refined",
    ) -> None:
        """Persist a viewport-readable SMPL-X snapshot after refinement.

        Why: the desktop viewport reloads GVHMR results from disk. When
        refinement only updates in-memory ``world_params``, exports reflect
        the refined motion but a later UI reload falls back to stale body
        data. The ``source`` tag is written so downstream viewport code can
        distinguish PHC physics (``phc_refined``) from spring-filter
        (``spring_refined``) output.
        """
        try:
            snapshot = dict(world_params)
            snapshot["source"] = source
            if camera_params is not None:
                for src_key, dst_key in [
                    ("global_orient", "global_orient_cam"),
                    ("body_pose", "body_pose_cam"),
                    ("transl", "transl_cam"),
                ]:
                    if dst_key not in snapshot and src_key in camera_params:
                        snapshot[dst_key] = camera_params[src_key]
                if "K_fullimg" not in snapshot and "K_fullimg" in camera_params:
                    snapshot["K_fullimg"] = camera_params["K_fullimg"]
                for key in [
                    "left_hand_pose",
                    "right_hand_pose",
                    "left_wrist_orient",
                    "right_wrist_orient",
                ]:
                    if key not in snapshot and key in camera_params:
                        snapshot[key] = camera_params[key]
                for src_key, dst_key in [
                    ("global_orient_world", "global_orient_world_baseline"),
                    ("body_pose_world", "body_pose_world_baseline"),
                    ("transl_world", "transl_world_baseline"),
                ]:
                    if dst_key not in snapshot and src_key in camera_params:
                        snapshot[dst_key] = camera_params[src_key]

            snapshot.setdefault("global_orient_world", snapshot["global_orient"])
            snapshot.setdefault("body_pose_world", snapshot["body_pose"])
            snapshot.setdefault("transl_world", snapshot["transl"])
            # Always populate the physics triad so legacy viewport code that
            # looks for "*_world_physics" keys still finds the refined data,
            # regardless of whether the refinement came from PHC or spring.
            snapshot.setdefault("global_orient_world_physics", snapshot["global_orient_world"])
            snapshot.setdefault("body_pose_world_physics", snapshot["body_pose_world"])
            snapshot.setdefault("transl_world_physics", snapshot["transl_world"])
            if source in ("spring_refined", "spring_refined_pinned"):
                snapshot.setdefault("global_orient_world_spring", snapshot["global_orient_world"])
                snapshot.setdefault("body_pose_world_spring", snapshot["body_pose_world"])
                snapshot.setdefault("transl_world_spring", snapshot["transl_world"])
            snapshot.setdefault("global_orient_world_baseline", snapshot["global_orient_world"])
            snapshot.setdefault("body_pose_world_baseline", snapshot["body_pose_world"])
            snapshot.setdefault("transl_world_baseline", snapshot["transl_world"])

            out_path = results.get("merged_pt")
            if out_path:
                out_path = Path(out_path)
            else:
                stem = self._video_path.stem
                out_path = self._output_dir / f"{stem}_hybrid_smplx.pt"
            save_merged_pt(snapshot, out_path)
            results["merged_pt"] = str(out_path)
            self.log_line.emit(f"Saved viewport params snapshot: {out_path}")
        except Exception as exc:
            self.log_line.emit(f"WARNING: Failed to save viewport params snapshot: {exc}")

    # ------------------------------------------------------------------
    # Stage 5: Face pipeline
    # ------------------------------------------------------------------

    def _run_face(self, results: dict) -> None:
        """Run face capture (MediaPipe ARKit blendshapes) and face mesh render."""
        stem = self._video_path.stem
        video_str = str(self._video_path)

        try:
            from face_capture import run_face_pipeline, render_face_mesh_video
        except ImportError:
            self.log_line.emit("WARNING: face_capture module not available, skipping.")
            return

        # Get bboxes for face crop regions
        bboxes = extract_bboxes_from_output(self._output_dir)
        if bboxes is None:
            self.log_line.emit(
                "No person bboxes found, using heuristic fallback for face crops."
            )
            try:
                bboxes = fallback_face_bboxes(video_str)
            except Exception as exc:
                self.log_line.emit(f"WARNING: Fallback face bboxes failed: {exc}")
                return

        # Find ViTPose data for improved face crops
        from workers.gvhmr_worker import find_output_dir, find_file

        vitpose_path = None
        gvhmr_out_dir = find_output_dir(self._video_path, self._gvhmr_root)
        if gvhmr_out_dir:
            vp = find_file(gvhmr_out_dir, "vitpose.pt")
            if vp:
                vitpose_path = str(vp)
                self.log_line.emit(f"Found ViTPose: {vitpose_path}")

        use_vitpose = self._config.use_vitpose_face_crops and vitpose_path is not None

        # Face blendshape extraction
        face_csv_path = str(self._output_dir / f"{stem}_arkit_blendshapes.csv")
        progress_cb = self._stage_progress_callback(5)
        try:
            run_face_pipeline(
                video_str,
                bboxes,
                face_csv_path,
                fps=self._fps,
                progress_callback=progress_cb,
                vitpose_path=vitpose_path,
                use_vitpose_crops=use_vitpose,
            )
            results["face_csv"] = face_csv_path
            self.log_line.emit(f"Face CSV: {face_csv_path}")
        except Exception as exc:
            self.log_line.emit(f"WARNING: Face pipeline failed: {exc}")

        if self._cancelled:
            return

        # Face mesh visualization
        face_mesh_path = str(self._output_dir / f"{stem}_face_mesh.mp4")
        try:
            render_face_mesh_video(
                video_str,
                bboxes,
                face_mesh_path,
                fps=self._fps,
                progress_callback=progress_cb,
                vitpose_path=vitpose_path,
                use_vitpose_crops=use_vitpose,
            )
            results["face_mesh_video"] = face_mesh_path
            self.log_line.emit(f"Face mesh video: {face_mesh_path}")
        except Exception as exc:
            self.log_line.emit(f"WARNING: Face mesh render failed: {exc}")

    # ------------------------------------------------------------------
    # Stage 6: BVH/FBX conversion
    # ------------------------------------------------------------------

    def _run_bvh_fbx(
        self, results: dict, world_params: dict | None, is_hybrid: bool
    ) -> None:
        """Convert body params to BVH, then BVH to FBX via Blender.

        Dispatches to SOMA BVH exporter when body_model is 'soma',
        otherwise uses the existing SMPL-X smplx_to_bvh path.
        """
        # SOMA path — use soma_bvh_export when params contain SOMA data
        if world_params is not None and "poses" in world_params:
            self._run_soma_bvh_fbx(results, world_params)
            return

        stem = self._video_path.stem
        bvh_path = str(self._output_dir / f"{stem}_body_hands.bvh")

        # BVH conversion
        try:
            if world_params is not None:
                from smplx_to_bvh import convert_params_to_bvh

                bvh_kwargs = dict(
                    fps=self._fps,
                    skip_world_grounding=is_hybrid,
                    smooth_body=not is_hybrid,
                    smooth_hands=True,
                )
                if not is_hybrid:
                    bvh_kwargs["pitch_adjust_deg"] = self._config.pitch_adjust
                if self._config.body_smooth_preset != "moderate":
                    bvh_kwargs["body_smooth_preset"] = self._config.body_smooth_preset
                convert_params_to_bvh(world_params, bvh_path, **bvh_kwargs)
                results["bvh"] = bvh_path
                self.log_line.emit(f"BVH written: {bvh_path}")
            else:
                # Fall back to .pt file path if we have one
                gvhmr_pt = results.get("gvhmr_pt")
                if gvhmr_pt:
                    from smplx_to_bvh import convert_smplx_to_bvh

                    convert_smplx_to_bvh(gvhmr_pt, bvh_path, fps=self._fps)
                    results["bvh"] = bvh_path
                    self.log_line.emit(f"BVH written: {bvh_path}")
                else:
                    self.log_line.emit("WARNING: No params available for BVH conversion.")
                    return
        except ImportError:
            self.log_line.emit("WARNING: smplx_to_bvh not available, skipping BVH.")
            return
        except Exception as exc:
            self.log_line.emit(f"WARNING: BVH conversion failed: {exc}")
            return

        if self._cancelled:
            return

        # FBX conversion via Blender
        fbx_path = str(self._output_dir / f"{stem}_body_hands.fbx")
        try:
            from bvh_to_fbx import convert_bvh_to_fbx

            naming_key = "ue5" if "ue5" in self._config.fbx_naming.lower() else "mixamo"
            fbx_log = convert_bvh_to_fbx(
                bvh_path, fbx_path, fps=self._fps, naming=naming_key
            )
            self.log_line.emit(fbx_log)
            if "ERROR" not in fbx_log:
                results["fbx"] = fbx_path
                self.log_line.emit(f"FBX written: {fbx_path}")
            else:
                self.log_line.emit("WARNING: FBX conversion reported errors.")
        except ImportError:
            self.log_line.emit("WARNING: bvh_to_fbx not available, skipping FBX.")
        except Exception as exc:
            self.log_line.emit(f"WARNING: FBX conversion failed: {exc}")

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

    # ------------------------------------------------------------------
    # Stage 7: Rendering
    # ------------------------------------------------------------------

    def _run_rendering(
        self,
        results: dict,
        camera_params: dict | None,
        world_params: dict | None,
        is_hybrid: bool,
    ) -> None:
        """Render skeleton overlay, hand overlay, and world views."""
        stem = self._video_path.stem
        video_str = str(self._video_path)

        try:
            from visualize_skeleton import (
                render_skeleton_video,
                render_world_views,
                render_hand_overlay_video,
            )
        except ImportError:
            self.log_line.emit(
                "WARNING: visualize_skeleton not available, skipping rendering."
            )
            return

        progress_cb = self._stage_progress_callback(7)
        render_params = camera_params if camera_params is not None else world_params

        # Skeleton overlay
        if render_params is not None:
            skeleton_path = str(self._output_dir / f"{stem}_skeleton.mp4")
            try:
                render_skeleton_video(
                    video_path=video_str,
                    output_path=skeleton_path,
                    fps=self._fps,
                    progress_callback=progress_cb,
                    params=render_params,
                    coordinate_space="camera",
                    render_label="Hybrid-camera" if is_hybrid else "GVHMR",
                )
                results["skeleton_video"] = skeleton_path
                self.log_line.emit(f"Skeleton video: {skeleton_path}")
            except Exception as exc:
                self.log_line.emit(f"WARNING: Skeleton render failed: {exc}")

        if self._cancelled:
            return

        # World views
        if world_params is not None:
            world_path = str(self._output_dir / f"{stem}_world_view.mp4")
            try:
                render_world_views(
                    output_path=world_path,
                    fps=self._fps,
                    params=world_params,
                    skip_world_grounding=is_hybrid,
                    progress_callback=progress_cb,
                )
                results["world_view_video"] = world_path
                self.log_line.emit(f"World view video: {world_path}")
            except Exception as exc:
                self.log_line.emit(f"WARNING: World view render failed: {exc}")

        if self._cancelled:
            return

        # Hand overlay
        if render_params is not None:
            hand_path = str(self._output_dir / f"{stem}_hand_overlay.mp4")
            try:
                render_hand_overlay_video(
                    video_path=video_str,
                    output_path=hand_path,
                    fps=self._fps,
                    progress_callback=progress_cb,
                    params=render_params,
                    coordinate_space="camera",
                    render_label="Hands (Hybrid)" if is_hybrid else "Hands (GVHMR)",
                )
                results["hand_overlay_video"] = hand_path
                self.log_line.emit(f"Hand overlay video: {hand_path}")
            except Exception as exc:
                self.log_line.emit(f"WARNING: Hand overlay render failed: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _emit_stage(self, idx: int):
        """Emit progress for the start of stage *idx* and record timing."""
        now = time.monotonic()
        # Close previous stage timing
        if hasattr(self, "_stage_start_time") and self._current_stage_idx >= 0:
            elapsed = now - self._stage_start_time
            prev_label = _FULL_STAGES[self._current_stage_idx][2]
            self._stage_timings.append((prev_label, elapsed))
        else:
            self._stage_timings: list[tuple[str, float]] = []
            self._pipeline_start_time = now
        self._current_stage_idx = idx
        self._stage_start_time = now
        start, _end, label = _FULL_STAGES[idx]
        self.progress.emit(start, label)

    def _emit_timing_summary(self):
        """Close the last stage and emit a timing summary to the log."""
        now = time.monotonic()
        if hasattr(self, "_stage_start_time") and self._current_stage_idx >= 0:
            elapsed = now - self._stage_start_time
            label = _FULL_STAGES[self._current_stage_idx][2]
            self._stage_timings.append((label, elapsed))
        total = now - getattr(self, "_pipeline_start_time", now)
        self.log_line.emit("")
        self.log_line.emit("── Stage Timings ──")
        for label, secs in self._stage_timings:
            self.log_line.emit(f"  {label:<28s} {secs:6.1f}s")
        self.log_line.emit(f"  {'TOTAL':<28s} {total:6.1f}s")
        self.log_line.emit("")
        return {label: round(secs, 2) for label, secs in self._stage_timings}

    def _stage_progress_callback(self, stage_idx: int):
        """Return a callback mapping ``[0, 1]`` sub-stage fraction to overall."""
        start, end, label = _FULL_STAGES[stage_idx]
        span = end - start

        def callback(frac: float, msg: str = ""):
            overall = start + frac * span
            self.progress.emit(overall, msg or label)
            if msg:
                self.log_line.emit(msg)

        return callback

    def _gvhmr_command(self) -> list[str]:
        args = [f"--video={self._video_path}"]
        if self._config.static_cam:
            args.append("--static_cam")
        if self._config.use_dpvo:
            args.append("--use_dpvo")
        if self._config.focal_mm != 24.0:
            args.append(f"--f_mm={self._config.focal_mm}")

        # On native Windows, delegate to WSL2's gvhmr conda env
        if sys.platform == "win32":
            from platform_info import wsl_solve_command
            return wsl_solve_command(
                conda_env="gvhmr",
                script=str(self._gvhmr_root / "tools" / "demo" / "demo.py"),
                args=args,
                cwd=str(self._gvhmr_root),
            )
        return [sys.executable, "tools/demo/demo.py"] + args

    def _smplestx_command(self) -> list[str]:
        return [
            self._smplestx_python,
            "smplestx_inference.py",
            f"--video={self._video_path}",
            f"--fps={self._fps}",
            f"--output_dir={self._output_dir}",
            "--no_render",
        ]


class MultiPersonWorker(QThread):
    """Wraps ``split_multi_person_video()`` with progress mapped to signals."""

    progress = Signal(float, str)
    log_line = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(
        self,
        video_path: Path,
        config: PipelineConfig,
        gvhmr_root: Path,
        output_dir: Path,
        parent=None,
    ):
        super().__init__(parent)
        self._video_path = video_path
        self._config = config
        self._gvhmr_root = gvhmr_root
        self._output_dir = output_dir
        self._cancelled = False
        # The multi-person pipeline assumes 30fps throughout (PHC input NPZ
        # is tagged fps=30, evaluate_refinement and BVH conversion take fps
        # as a parameter).  Keep this as a plain attribute so physics
        # refinement and the metrics verdict block can reference it without
        # crashing on AttributeError.
        self._fps: float = 30.0

    def run(self):
        try:
            from multi_person_split import split_multi_person_video

            timings: list[tuple[str, float]] = []
            pipeline_t0 = time.monotonic()

            def progress_callback(frac: float, msg: str):
                if frac < 0:
                    # Log-only: stream subprocess output without updating progress bar
                    self.log_line.emit(msg)
                else:
                    overall = 0.02 + frac * 0.83
                    self.progress.emit(overall, msg)
                    self.log_line.emit(f"[MultiPerson] {msg}")

            self.progress.emit(0.02, "Starting multi-person pipeline...")

            t0 = time.monotonic()
            result = split_multi_person_video(
                video_path=str(self._video_path),
                output_dir=str(self._output_dir),
                static_cam=self._config.static_cam,
                use_dpvo=self._config.use_dpvo,
                max_persons=self._config.max_persons,
                render_overlays=self._config.render_overlays,
                use_inpainting=self._config.use_inpainting,
                progress_callback=progress_callback,
                estimation_backend=self._config.estimation_backend,
                use_hands=self._config.use_hands and self._config.hand_source != "hamer",
            )
            timings.append(("Detection + Tracking + Estimation", time.monotonic() - t0))

            if self._cancelled:
                return

            # ── Post-pipeline HaMeR hand replacement (per-person) ──
            if self._config.hand_source == "hamer":
                t0 = time.monotonic()
                self._try_hamer_multi(result)
                timings.append(("HaMeR hands", time.monotonic() - t0))

            # ── Post-pipeline motion refinement (per-person) ──
            if (
                self._config.use_spring_refine
                or self._config.use_foot_pin
                or self._config.use_camera_stabilize
            ):
                t0 = time.monotonic()
                self._run_spring_multi(result)
                timings.append(("Spring refinement", time.monotonic() - t0))
            elif self._config.use_physics_refine:
                t0 = time.monotonic()
                self._run_physics_multi(result)
                timings.append(("Physics refinement", time.monotonic() - t0))

            # ── Post-pipeline FBX batch conversion ──
            t0 = time.monotonic()
            fbx_files = self._convert_bvh_to_fbx_batch(result)
            timings.append(("BVH/FBX conversion", time.monotonic() - t0))

            total = time.monotonic() - pipeline_t0
            self.log_line.emit("")
            self.log_line.emit("── Stage Timings ──")
            for label, secs in timings:
                self.log_line.emit(f"  {label:<36s} {secs:6.1f}s")
            self.log_line.emit(f"  {'TOTAL':<36s} {total:6.1f}s")
            self.log_line.emit("")

            stage_timings = {label: round(secs, 2) for label, secs in timings}

            self.progress.emit(1.0, "Multi-person pipeline complete")
            self.finished.emit({
                "output_dir": str(self._output_dir),
                "result": result,
                "fbx_files": fbx_files,
                "stage_timings": stage_timings,
            })

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _try_hamer_multi(self, result) -> None:
        """Run HaMeR hand reconstruction per-person with confidence merging.

        For each person directory, finds ViTPose + isolated video, runs
        ``run_hamer()``, merges hand poses using per-frame confidence
        thresholds, and re-exports the BVH so downstream FBX conversion
        picks it up.
        """
        try:
            from hamer_inference import run_hamer, _load_hamer_model, merge_gvhmr_hamer_params
            from smplx_to_bvh import extract_gvhmr_params
        except ImportError:
            self.log_line.emit(
                "WARNING: hamer_inference not available, skipping HaMeR hands."
            )
            return

        import torch

        person_dirs = getattr(result, "person_dirs", None) or []
        person_videos = getattr(result, "person_video_paths", None) or []
        if not person_dirs:
            return

        total_persons = len(person_dirs)
        self.progress.emit(0.85, f"Running HaMeR hand reconstruction ({total_persons} persons)...")

        # Pre-scan: skip model load entirely if all persons already have hand data
        needs_hamer = []
        for i, person_dir in enumerate(person_dirs):
            person_dir = Path(person_dir)
            hybrid_pts = sorted(person_dir.glob("*_hybrid_smplx.pt"))
            gvhmr_pts = list(person_dir.rglob("hmr4d_results.pt"))
            pt_path = (hybrid_pts[-1] if hybrid_pts
                       else gvhmr_pts[0] if gvhmr_pts else None)
            if pt_path is not None:
                try:
                    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                    lh = data.get("left_hand_pose")
                    if (lh is not None and hasattr(lh, 'numel')
                            and lh.numel() > 0 and lh.abs().sum() > 0):
                        continue
                except Exception:
                    pass
            needs_hamer.append(i)

        if not needs_hamer:
            self.log_line.emit("[HaMeR] All persons already have hand data, skipping model load.")
            return

        # Load the HaMeR model once and share across all persons
        try:
            shared_model, shared_model_cfg = _load_hamer_model("cuda")
        except Exception as exc:
            self.log_line.emit(f"WARNING: HaMeR model load failed: {exc}")
            return

        for i, person_dir in enumerate(person_dirs):
            if self._cancelled:
                return
            person_dir = Path(person_dir)

            base_frac = 0.85 + (i / total_persons) * 0.07
            self.progress.emit(base_frac, f"HaMeR hands: person {i + 1}/{total_persons}")

            # Resolve target .pt BEFORE inference
            hybrid_pts = sorted(person_dir.glob("*_hybrid_smplx.pt"))
            gvhmr_pts = list(person_dir.rglob("hmr4d_results.pt"))
            pt_path = (hybrid_pts[-1] if hybrid_pts
                       else gvhmr_pts[0] if gvhmr_pts else None)
            if pt_path is None:
                self.log_line.emit(
                    f"[HaMeR] Person {i}: no params .pt found, skipping."
                )
                continue

            # Skip if hand data already present
            try:
                existing = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                lh = existing.get("left_hand_pose")
                if (lh is not None and hasattr(lh, 'numel')
                        and lh.numel() > 0 and lh.abs().sum() > 0):
                    self.log_line.emit(
                        f"[HaMeR] Person {i}: hands already in {pt_path.name}, skipping."
                    )
                    continue
            except Exception:
                pass

            # Find isolated video for this person
            video_path = person_videos[i] if i < len(person_videos) else None
            if not video_path or not Path(video_path).exists():
                self.log_line.emit(
                    f"[HaMeR] Person {i}: no isolated video, skipping."
                )
                continue

            # Find ViTPose from GVHMR preprocessing
            vitpose_files = list(person_dir.rglob("vitpose.pt"))
            vitpose_pt = str(vitpose_files[0]) if vitpose_files else None
            if vitpose_pt:
                self.log_line.emit(
                    f"[HaMeR] Person {i}: using ViTPose {vitpose_pt}"
                )

            viz_path = person_dir / "hamer_hands.mp4"
            try:
                hamer_result = run_hamer(
                    video_path=str(video_path),
                    vitpose_path=vitpose_pt,
                    model=shared_model,
                    model_cfg=shared_model_cfg,
                    viz_output_path=str(viz_path),
                    hand_size_frac=0.12,
                )
            except Exception as exc:
                self.log_line.emit(
                    f"[HaMeR] Person {i}: failed — {exc}"
                )
                continue

            if hamer_result is None:
                self.log_line.emit(
                    f"[HaMeR] Person {i}: no results returned."
                )
                continue

            # Confidence-based merge: only replace hand poses on frames
            # where HaMeR confidence exceeds the threshold
            try:
                gvhmr_params = extract_gvhmr_params(str(pt_path))
                merged = merge_gvhmr_hamer_params(gvhmr_params, hamer_result)

                # Update existing file in-place to preserve format for BVH export
                data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                n = merged["num_frames"]
                data["left_hand_pose"] = torch.tensor(
                    merged["left_hand_pose"].reshape(n, -1),
                    dtype=torch.float32,
                )
                data["right_hand_pose"] = torch.tensor(
                    merged["right_hand_pose"].reshape(n, -1),
                    dtype=torch.float32,
                )
                for wrist_key in ["left_wrist_orient", "right_wrist_orient"]:
                    if wrist_key in merged:
                        data[wrist_key] = torch.tensor(
                            merged[wrist_key], dtype=torch.float32
                        )
                torch.save(data, str(pt_path))
                self.log_line.emit(
                    f"[HaMeR] Person {i}: confidence-merged hands → {pt_path.name}"
                )
            except Exception as exc:
                self.log_line.emit(
                    f"[HaMeR] Person {i}: merge failed — {exc}"
                )
                continue

            # Re-export BVH so FBX conversion uses updated hands
            try:
                from multi_person_split import _export_person_bvh
                bvh_path = _export_person_bvh(person_dir)
                if bvh_path:
                    self.log_line.emit(
                        f"[HaMeR] Person {i}: re-exported BVH → {bvh_path.name}"
                    )
            except Exception as exc:
                self.log_line.emit(
                    f"[HaMeR] Person {i}: BVH re-export failed — {exc}"
                )

            done_frac = 0.85 + ((i + 1) / total_persons) * 0.07
            self.progress.emit(done_frac, f"HaMeR hands: person {i + 1}/{total_persons} complete")

        # Free shared model
        del shared_model, shared_model_cfg
        torch.cuda.empty_cache()

    def _run_physics_multi(self, result) -> None:
        """Run PHC physics refinement per-person.

        For each person directory, loads params, runs physics refinement,
        saves refined params, and re-exports BVH.
        """
        try:
            from workers.physics.gvhmr_to_amass import params_to_amass_npz
            from workers.physics.phc_runner import run_phc_local
            from workers.physics.phc_to_smpl import phc_output_to_params
            from workers.physics.evaluate import (
                compute_verdict,
                evaluate_refinement,
                write_metrics_json,
            )
            from smplx_to_bvh import extract_gvhmr_params
        except ImportError as exc:
            self.log_line.emit(f"WARNING: Physics modules not available: {exc}")
            return

        import torch

        person_dirs = getattr(result, "person_dirs", None) or []
        if not person_dirs:
            return

        total = len(person_dirs)
        self.progress.emit(0.85, f"Physics refinement ({total} persons)...")

        for i, person_dir in enumerate(person_dirs):
            if self._cancelled:
                return
            person_dir = Path(person_dir)

            base_frac = 0.85 + (i / total) * 0.07
            self.progress.emit(base_frac, f"Physics: person {i + 1}/{total}")

            pt_path = self._candidate_gvhmr_pt(person_dir)
            if pt_path is None:
                self.log_line.emit(f"[Physics] Person {i}: no params .pt, skipping.")
                continue

            try:
                params = extract_gvhmr_params(str(pt_path))

                phc_dir = person_dir / "physics"
                phc_dir.mkdir(parents=True, exist_ok=True)
                input_npz = phc_dir / "phc_input.npz"
                params_to_amass_npz(params, input_npz)

                phc_output_dir = phc_dir / "phc_output"
                phc_result = run_phc_local(input_npz, phc_output_dir)

                if not phc_result.success:
                    self.log_line.emit(
                        f"[Physics] Person {i}: PHC failed, skipping. "
                        f"{phc_result.log[-200:]}"
                    )
                    continue

                refined = phc_output_to_params(phc_result.output_path, params)

                # Evaluate + persist metrics + log verdict
                try:
                    metrics = evaluate_refinement(params, refined, fps=self._fps)
                    metrics_path = phc_dir / "metrics.json"
                    write_metrics_json(metrics, metrics_path)
                    status, reason = compute_verdict(metrics)
                    msg = f"[Physics] Person {i}: verdict {status.upper()}"
                    if reason:
                        msg += f" — {reason}"
                    self.log_line.emit(msg)
                except Exception as exc:
                    self.log_line.emit(
                        f"[Physics] Person {i}: metrics failed — {exc}"
                    )

                # Update the .pt file with refined body params
                data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                n = refined["num_frames"]
                # Capture pre-refinement body params as the "baseline" so the
                # viewport toggle can distinguish World physics from World
                # baseline.  The fallback chain has to end in
                # ``smpl_params_global`` because a fresh-from-GVHMR
                # hmr4d_results.pt does NOT yet have top-level body_pose /
                # body_pose_world keys — those are created by the refinement
                # overwrite below.  Without the smpl_params_global fallback,
                # the first refinement run on any clip would leave the
                # ``*_world_baseline`` triad missing and the UI toggle would
                # silently collapse to "World physics" on both sides.
                # (Explicit ``is not None`` checks — Python's ``or`` calls
                # ``bool()`` which raises on multi-element tensors.)
                _spg = data.get("smpl_params_global")
                if not isinstance(_spg, dict):
                    _spg = {}

                def _first_present(*keys_and_dicts):
                    for entry in keys_and_dicts:
                        src, key = entry
                        v = src.get(key)
                        if v is not None:
                            return v
                    return None

                baseline_go = _first_present(
                    (data, "global_orient_world_baseline"),
                    (data, "global_orient_world"),
                    (data, "global_orient"),
                    (_spg, "global_orient"),
                )
                baseline_bp = _first_present(
                    (data, "body_pose_world_baseline"),
                    (data, "body_pose_world"),
                    (data, "body_pose"),
                    (_spg, "body_pose"),
                )
                baseline_tr = _first_present(
                    (data, "transl_world_baseline"),
                    (data, "transl_world"),
                    (data, "transl"),
                    (_spg, "transl"),
                )
                data["global_orient"] = torch.tensor(
                    refined["global_orient"].reshape(n, -1), dtype=torch.float32
                )
                data["body_pose"] = torch.tensor(
                    refined["body_pose"].reshape(n, -1), dtype=torch.float32
                )
                data["transl"] = torch.tensor(
                    refined["transl"].reshape(n, -1), dtype=torch.float32
                )
                data["global_orient_world"] = data["global_orient"]
                data["body_pose_world"] = data["body_pose"]
                data["transl_world"] = data["transl"]
                data["global_orient_world_physics"] = data["global_orient"]
                data["body_pose_world_physics"] = data["body_pose"]
                data["transl_world_physics"] = data["transl"]
                if baseline_go is not None:
                    data["global_orient_world_baseline"] = baseline_go
                if baseline_bp is not None:
                    data["body_pose_world_baseline"] = baseline_bp
                if baseline_tr is not None:
                    data["transl_world_baseline"] = baseline_tr
                data["source"] = "phc_refined"
                torch.save(data, str(pt_path))
                # Snapshot write is best-effort: the refined .pt is already
                # persisted above, so a snapshot failure must not mask the
                # refinement success or skip the BVH re-export below.
                snapshot_path: Path | None = None
                try:
                    snapshot_path = self._save_person_physics_hybrid_snapshot(
                        person_dir,
                        data,
                    )
                except Exception as snap_exc:
                    self.log_line.emit(
                        f"[Physics] Person {i}: snapshot write failed — {snap_exc}"
                    )
                self.log_line.emit(
                    f"[Physics] Person {i}: refined {n} frames -> {pt_path.name}"
                )
                if snapshot_path is not None:
                    self.log_line.emit(
                        f"[Physics] Person {i}: wrote viewport snapshot -> {snapshot_path.name}"
                    )

                # Re-export BVH
                try:
                    from multi_person_split import _export_person_bvh
                    bvh_path = _export_person_bvh(person_dir)
                    if bvh_path:
                        self.log_line.emit(
                            f"[Physics] Person {i}: re-exported BVH -> {bvh_path.name}"
                        )
                except Exception as exc:
                    self.log_line.emit(
                        f"[Physics] Person {i}: BVH re-export failed — {exc}"
                    )

            except Exception as exc:
                self.log_line.emit(f"[Physics] Person {i}: failed — {exc}")
                continue

            done_frac = 0.85 + ((i + 1) / total) * 0.07
            self.progress.emit(done_frac, f"Physics: person {i + 1}/{total} complete")

    @staticmethod
    def _candidate_gvhmr_pt(person_dir: Path) -> Path | None:
        """Return the .pt file to use as motion-refinement input for a person.

        Refinement (PHC physics or spring filter) must always start from
        the authoritative GVHMR output (``hmr4d_results.pt``), never from a
        ``*_hybrid_smplx.pt``. Two reasons:

        1. Circular hazard: the refinement snapshot writers write
           ``phc_refined_hybrid_smplx.pt`` / ``spring_refined_hybrid_smplx.pt``
           after a successful refinement. Those files match the hybrid
           glob but are flat viewport snapshots — they do NOT carry
           ``smpl_params_global``, so ``extract_gvhmr_params`` (which does
           bracket access ``data["smpl_params_global"]``) would raise
           KeyError on every subsequent refinement attempt.
        2. Even a valid HaMeR-merged ``*_hybrid_smplx.pt`` would be the
           wrong input: refinement should operate on the *original* GVHMR
           trajectory, not re-refine already-refined output.

        The refined hmr4d_results.pt preserves the pristine
        ``smpl_params_global`` across refinement runs (the overwrite at
        the bottom of the refinement block only touches flat top-level
        keys), so ``extract_gvhmr_params`` always sees clean input.
        """
        gvhmr_pts = list(person_dir.rglob("hmr4d_results.pt"))
        if gvhmr_pts:
            return gvhmr_pts[0]
        # Legacy fallback: directories that only have a merged hybrid
        # (pre-multi-person layout). Skip refinement-snapshot files
        # explicitly — they cannot be parsed as GVHMR-format input.
        excluded = {
            "phc_refined_hybrid_smplx.pt",
            "spring_refined_hybrid_smplx.pt",
        }
        hybrid_pts = sorted(
            p for p in person_dir.glob("*_hybrid_smplx.pt") if p.name not in excluded
        )
        return hybrid_pts[-1] if hybrid_pts else None

    def _run_spring_multi(self, result) -> None:
        """Run spring-based motion refinement per-person.

        Mirrors ``_run_physics_multi`` in structure: for each person,
        reads pristine GVHMR params, runs the spring filter, writes the
        refined triad back to ``hmr4d_results.pt`` under ``*_world_spring``
        keys (alongside the ``*_world_physics`` triad for viewport
        compatibility), persists a viewport snapshot, and re-exports BVH.
        """
        try:
            from workers.spring_refine import (
                run_spring_refine,
                write_spring_metrics_json,
            )
            from smplx_to_bvh import extract_gvhmr_params
        except ImportError as exc:
            self.log_line.emit(f"WARNING: Spring refine modules not available: {exc}")
            return

        import numpy as np
        import torch

        person_dirs = getattr(result, "person_dirs", None) or []
        if not person_dirs:
            return

        total = len(person_dirs)
        preset = self._config.spring_refine_preset
        pin_enabled = bool(self._config.use_foot_pin)
        filter_enabled = bool(self._config.use_spring_refine)

        # Sandpipe discovery (per-video, shared across persons)
        sp_path = None
        if self._config.use_sandpipe:
            sp_path = find_sandpipe_physics(self._output_dir, self._video_path)
            if sp_path:
                self.log_line.emit(f"Sandpipe physics found: {sp_path}")

        self.progress.emit(
            0.85,
            f"Spring refinement ({total} persons, preset={preset}, pin={'on' if pin_enabled else 'off'})...",
        )

        for i, person_dir in enumerate(person_dirs):
            if self._cancelled:
                return
            person_dir = Path(person_dir)

            base_frac = 0.85 + (i / total) * 0.07
            self.progress.emit(base_frac, f"Spring: person {i + 1}/{total}")

            pt_path = self._candidate_gvhmr_pt(person_dir)
            if pt_path is None:
                self.log_line.emit(f"[Spring] Person {i}: no params .pt, skipping.")
                continue

            try:
                params = extract_gvhmr_params(str(pt_path))

                # IK death detection: GVHMR's process_ik uses a recursive
                # exponential average of static confidence to blend joint
                # targets.  Even low confidence (0.05) accumulates over
                # hundreds of frames, killing the pose signal.  Detect
                # spans where the post-processed body_pose is frozen but
                # the raw network output is alive, and substitute the raw
                # poses for those frames.
                #
                # Gated on use_spring_refine: this is a destructive data
                # substitution and belongs under the "Enable spring-based
                # refinement" toggle. When that toggle is off, we leave
                # GVHMR's body_pose untouched.
                if self._config.use_spring_refine:
                    try:
                        ik_data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                        raw_bp = ik_data.get("raw_body_pose")
                        if raw_bp is not None:
                            raw_bp = np.asarray(raw_bp, dtype=np.float32).reshape(-1, 63)
                            final_bp = np.asarray(params["body_pose"], dtype=np.float32).reshape(-1, 63)
                            n = min(len(raw_bp), len(final_bp))
                            # Per-frame delta ratio: raw motion / post-processed motion
                            raw_delta = np.linalg.norm(np.diff(raw_bp[:n], axis=0), axis=1)
                            fin_delta = np.linalg.norm(np.diff(final_bp[:n], axis=0), axis=1)
                            dead = (raw_delta > 0.08) & (fin_delta < 0.02)
                            n_dead = int(dead.sum())
                            if n_dead > 0:
                                # Build contiguous spans for logging
                                dead_padded = np.concatenate([[False], dead, [False]])
                                edges = np.diff(dead_padded.astype(int))
                                starts = np.where(edges == 1)[0]
                                ends = np.where(edges == -1)[0] - 1
                                spans = list(zip(starts.tolist(), ends.tolist()))
                                # Substitute raw body_pose for dead frames.
                                # +1 offset because dead is computed on diff (frame pairs)
                                for s, e in spans:
                                    # dead[s] means frames s and s+1 pair is dead
                                    fs, fe = s + 1, min(e + 2, n)
                                    final_bp[fs:fe] = raw_bp[fs:fe]
                                params["body_pose"] = final_bp
                                # Also fix global_orient if similarly frozen
                                spg = ik_data.get("smpl_params_global", {})
                                net_out = ik_data.get("net_outputs", {})
                                raw_go = None
                                if isinstance(net_out, dict):
                                    psg = net_out.get("pred_smpl_params_global", {})
                                    if isinstance(psg, dict) and "global_orient" in psg:
                                        raw_go = np.asarray(psg["global_orient"], dtype=np.float32)
                                        if raw_go.ndim == 3:
                                            raw_go = raw_go[0]  # remove batch dim
                                        raw_go = raw_go.reshape(-1, 3)
                                if raw_go is not None:
                                    final_go = np.asarray(params["global_orient"], dtype=np.float32).reshape(-1, 3)
                                    for s, e in spans:
                                        fs, fe = s + 1, min(e + 2, n)
                                        final_go[fs:fe] = raw_go[fs:fe]
                                    params["global_orient"] = final_go
                                # Save IK death spans to disk for UI markers
                                ik_spans_path = person_dir / "ik_death_spans.json"
                                import json as _json
                                ik_spans_path.write_text(_json.dumps({
                                    "spans": [[int(s + 1), int(min(e + 2, n) - 1)] for s, e in spans],
                                    "total_dead_frames": n_dead,
                                    "total_frames": n,
                                }, indent=2))
                                span_desc = ", ".join(f"{s+1}-{min(e+2,n)-1}" for s, e in spans[:5])
                                if len(spans) > 5:
                                    span_desc += f" (+{len(spans)-5} more)"
                                self.log_line.emit(
                                    f"[Spring] Person {i}: IK death bypass — {n_dead} frames "
                                    f"substituted with raw network poses [{span_desc}]"
                                )
                        del ik_data
                    except Exception as ik_exc:
                        self.log_line.emit(
                            f"[Spring] Person {i}: IK death detection skipped: {ik_exc}"
                        )

                # Camera stabilization (per-person)
                if self._config.use_camera_stabilize:
                    try:
                        from workers.spring_refine.camera_stabilize import stabilize_world_params

                        stabilized = stabilize_world_params(
                            pt_path,
                            cam_smooth_preset=self._config.cam_smooth_preset,
                            fps=float(self._fps),
                        )
                        if stabilized is not None:
                            params["global_orient"] = stabilized["global_orient"]
                            params["transl"] = stabilized["transl"]
                            self.log_line.emit(
                                f"[Spring] Person {i}: camera stabilization applied."
                            )
                    except Exception as stab_exc:
                        self.log_line.emit(
                            f"[Spring] Person {i}: camera stabilization failed: {stab_exc}"
                        )

                # Drift analysis (after cam stabilize, before spring refine)
                drift_diag: dict = {}
                effective_pin_strength = float(self._config.foot_pin_strength)
                if self._config.use_drift_analysis:
                    try:
                        from workers.spring_refine.drift_analysis import (
                            analyze_and_correct_drift,
                            write_drift_diagnostics,
                        )

                        corrected, drift_diag = analyze_and_correct_drift(
                            pt_path,
                            params,
                            fps=float(self._fps),
                            cam_smooth_preset=self._config.cam_smooth_preset,
                        )
                        if corrected is not None:
                            params["global_orient"] = corrected["global_orient"]
                            params["transl"] = corrected["transl"]
                            self.log_line.emit(
                                f"[Spring] Person {i}: drift analysis corrected "
                                f"({drift_diag.get('pre_correction_drift_xz_cm', '?')} → "
                                f"{drift_diag.get('post_correction_drift_xz_cm', '?')} cm XZ)"
                            )
                        effective_pin_strength = min(
                            effective_pin_strength,
                            drift_diag.get("recommended_pin_strength", 1.0),
                        )
                        # Write diagnostics JSON
                        diag_path = person_dir / "spring_refine" / "drift_analysis.json"
                        write_drift_diagnostics(drift_diag, diag_path)
                    except Exception as drift_exc:
                        self.log_line.emit(
                            f"[Spring] Person {i}: drift analysis failed: {drift_exc}"
                        )

                baseline_params = dict(params)
                baseline_transl = (
                    np.asarray(params.get("transl")).copy()
                    if params.get("transl") is not None
                    else None
                )
                refined, ok = run_spring_refine(
                    params,
                    preset=preset,
                    fps=float(self._fps),
                    pin_feet=pin_enabled,
                    filter_rotations=filter_enabled,
                    foot_pin_sensitivity=self._config.foot_pin_sensitivity,
                    foot_pin_strength=effective_pin_strength,
                    sandpipe_physics_path=str(sp_path) if sp_path else None,
                    sandpipe_person_idx=i,
                )
                drift_corrected = drift_diag.get("correction_applied", False)
                if not ok and not drift_corrected:
                    self.log_line.emit(
                        f"[Spring] Person {i}: no refinement applied, skipping."
                    )
                    continue
                if not ok:
                    # Spring filter/pin were no-ops but drift analysis
                    # corrected params — use the drift-corrected params
                    # as the "refined" output so they get persisted.
                    refined = dict(params)
                    refined["num_frames"] = int(
                        np.asarray(params["body_pose"]).reshape(-1, 63).shape[0]
                    )
                    refined["source"] = "drift_corrected"

                # Capture baseline with the same fall-through chain as
                # _run_physics_multi. Freshly-solved hmr4d_results.pt stores
                # body_pose / global_orient / transl inside smpl_params_global,
                # so without that fallback the baseline triad would be missing
                # on the first refinement run and the viewport toggle would
                # silently collapse to a single motion source.
                data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                n = refined["num_frames"]
                _spg = data.get("smpl_params_global")
                if not isinstance(_spg, dict):
                    _spg = {}

                def _first_present(*entries):
                    for src, key in entries:
                        v = src.get(key)
                        if v is not None:
                            return v
                    return None

                baseline_go = _first_present(
                    (data, "global_orient_world_baseline"),
                    (data, "global_orient_world"),
                    (data, "global_orient"),
                    (_spg, "global_orient"),
                )
                baseline_bp = _first_present(
                    (data, "body_pose_world_baseline"),
                    (data, "body_pose_world"),
                    (data, "body_pose"),
                    (_spg, "body_pose"),
                )
                baseline_tr = _first_present(
                    (data, "transl_world_baseline"),
                    (data, "transl_world"),
                    (data, "transl"),
                    (_spg, "transl"),
                )

                go_t = torch.tensor(
                    np.asarray(refined["global_orient"]).reshape(n, -1),
                    dtype=torch.float32,
                )
                bp_t = torch.tensor(
                    np.asarray(refined["body_pose"]).reshape(n, -1),
                    dtype=torch.float32,
                )
                tr_t = torch.tensor(
                    np.asarray(refined["transl"]).reshape(n, -1),
                    dtype=torch.float32,
                )
                data["global_orient"] = go_t
                data["body_pose"] = bp_t
                data["transl"] = tr_t
                data["global_orient_world"] = go_t
                data["body_pose_world"] = bp_t
                data["transl_world"] = tr_t
                data["global_orient_world_spring"] = go_t
                data["body_pose_world_spring"] = bp_t
                data["transl_world_spring"] = tr_t
                # Dual-write the *_world_physics triad so existing viewport
                # source recognition (which keys off *_world_physics) still
                # picks up the refined data on reload.
                data["global_orient_world_physics"] = go_t
                data["body_pose_world_physics"] = bp_t
                data["transl_world_physics"] = tr_t
                if baseline_go is not None:
                    data["global_orient_world_baseline"] = baseline_go
                if baseline_bp is not None:
                    data["body_pose_world_baseline"] = baseline_bp
                if baseline_tr is not None:
                    data["transl_world_baseline"] = baseline_tr
                pinned = refined.get("source") == "spring_refined_pinned"
                if pinned and baseline_transl is not None:
                    data["transl_world_spring_unpin"] = torch.tensor(
                        np.asarray(baseline_transl).reshape(n, -1),
                        dtype=torch.float32,
                    )
                if pinned:
                    data["source"] = "spring_refined_pinned"
                elif refined.get("source") == "drift_corrected":
                    data["source"] = "drift_corrected"
                else:
                    data["source"] = "spring_refined"
                torch.save(data, str(pt_path))

                snapshot_path: Path | None = None
                try:
                    snapshot_path = self._save_person_spring_hybrid_snapshot(
                        person_dir, data
                    )
                except Exception as snap_exc:
                    self.log_line.emit(
                        f"[Spring] Person {i}: snapshot write failed — {snap_exc}"
                    )

                # Per-person metrics JSON (baseline vs refined foot skating).
                try:
                    metrics_path = person_dir / "spring_refine" / "metrics.json"
                    payload = write_spring_metrics_json(
                        baseline_params=baseline_params,
                        refined_params=refined,
                        fps=float(self._fps),
                        preset=preset,
                        pin_enabled=pin_enabled,
                        out_path=metrics_path,
                        pin_stats=refined.get("foot_pin_stats"),
                    )
                    verdict = payload.get("verdict", "unknown")
                    delta = payload.get("foot_skating_delta_m")
                    delta_str = (
                        f"{delta * 100:.2f} cm"
                        if isinstance(delta, (int, float)) and not np.isnan(delta)
                        else "n/a"
                    )
                    self.log_line.emit(
                        f"[Spring] Person {i}: verdict={verdict.upper()} "
                        f"(delta={delta_str}) -> {metrics_path.name}"
                    )
                except Exception as metrics_exc:
                    self.log_line.emit(
                        f"[Spring] Person {i}: metrics write failed — {metrics_exc}"
                    )

                self.log_line.emit(
                    f"[Spring] Person {i}: refined {n} frames "
                    f"(preset={preset}, pin={'on' if pin_enabled else 'off'}) -> {pt_path.name}"
                )
                if snapshot_path is not None:
                    self.log_line.emit(
                        f"[Spring] Person {i}: wrote viewport snapshot -> {snapshot_path.name}"
                    )

                # Re-export BVH so FBX conversion picks up refined motion
                try:
                    from multi_person_split import _export_person_bvh
                    bvh_path = _export_person_bvh(person_dir)
                    if bvh_path:
                        self.log_line.emit(
                            f"[Spring] Person {i}: re-exported BVH -> {bvh_path.name}"
                        )
                except Exception as exc:
                    self.log_line.emit(
                        f"[Spring] Person {i}: BVH re-export failed — {exc}"
                    )

            except Exception as exc:
                self.log_line.emit(f"[Spring] Person {i}: failed — {exc}")
                continue

            done_frac = 0.85 + ((i + 1) / total) * 0.07
            self.progress.emit(done_frac, f"Spring: person {i + 1}/{total} complete")

    def _save_person_spring_hybrid_snapshot(
        self, person_dir: Path, data: dict
    ) -> Path | None:
        import torch

        snapshot: dict = {}
        for key in [
            "global_orient",
            "body_pose",
            "transl",
            "global_orient_world",
            "body_pose_world",
            "transl_world",
            "global_orient_world_baseline",
            "body_pose_world_baseline",
            "transl_world_baseline",
            "global_orient_world_physics",
            "body_pose_world_physics",
            "transl_world_physics",
            "global_orient_world_spring",
            "body_pose_world_spring",
            "transl_world_spring",
            "transl_world_spring_unpin",
            "left_hand_pose",
            "right_hand_pose",
            "left_wrist_orient",
            "right_wrist_orient",
            "betas",
            "K_fullimg",
            "source",
        ]:
            if key in data:
                snapshot[key] = data[key]

        incam = data.get("smpl_params_incam")
        if isinstance(incam, dict):
            for src_key, dst_key in [
                ("global_orient", "global_orient_cam"),
                ("body_pose", "body_pose_cam"),
                ("transl", "transl_cam"),
            ]:
                if dst_key not in snapshot and src_key in incam:
                    snapshot[dst_key] = incam[src_key]

        if not snapshot:
            return None

        output_path = person_dir / "spring_refined_hybrid_smplx.pt"
        torch.save(snapshot, str(output_path))
        return output_path

    def _save_person_physics_hybrid_snapshot(self, person_dir: Path, data: dict) -> Path | None:
        import torch

        snapshot: dict = {}
        for key in [
            "global_orient",
            "body_pose",
            "transl",
            "global_orient_world",
            "body_pose_world",
            "transl_world",
            "global_orient_world_baseline",
            "body_pose_world_baseline",
            "transl_world_baseline",
            "global_orient_world_physics",
            "body_pose_world_physics",
            "transl_world_physics",
            "left_hand_pose",
            "right_hand_pose",
            "left_wrist_orient",
            "right_wrist_orient",
            "betas",
            "K_fullimg",
            "source",
        ]:
            if key in data:
                snapshot[key] = data[key]

        incam = data.get("smpl_params_incam")
        if isinstance(incam, dict):
            for src_key, dst_key in [
                ("global_orient", "global_orient_cam"),
                ("body_pose", "body_pose_cam"),
                ("transl", "transl_cam"),
            ]:
                if dst_key not in snapshot and src_key in incam:
                    snapshot[dst_key] = incam[src_key]

        if not snapshot:
            return None

        output_path = person_dir / "phc_refined_hybrid_smplx.pt"
        torch.save(snapshot, str(output_path))
        return output_path

    def _convert_bvh_to_fbx_batch(self, result) -> list[str]:
        """Convert per-person BVH files to FBX after split pipeline completes.

        Mirrors Gradio's ``run_multi_person_pipeline()`` post-pipeline step:
        scan each person directory for BVH files and convert using Blender
        with the user-chosen naming convention (Mixamo vs UE5).
        """
        fbx_files: list[str] = []
        person_dirs = getattr(result, "person_dirs", None) or []
        if not person_dirs:
            return fbx_files

        try:
            from bvh_to_fbx import convert_bvh_to_fbx
        except ImportError:
            self.log_line.emit(
                "WARNING: bvh_to_fbx not available, skipping FBX conversion."
            )
            return fbx_files

        naming_key = (
            "ue5" if "ue5" in self._config.fbx_naming.lower() else "mixamo"
        )
        fps = self._config.target_fps

        # Collect all BVH files across person directories
        bvh_paths: list[Path] = []
        for person_dir in person_dirs:
            person_dir = Path(person_dir)
            bvh_paths.extend(sorted(person_dir.rglob("*.bvh")))

        if not bvh_paths:
            self.log_line.emit("[FBX] No BVH files found, skipping FBX conversion.")
            return fbx_files

        # Separate cached vs needing conversion
        to_convert: list[tuple[int, Path, str]] = []
        for i, bvh_path in enumerate(bvh_paths):
            fbx_path = str(bvh_path.with_suffix(".fbx"))
            if Path(fbx_path).exists():
                fbx_files.append(fbx_path)
                self.log_line.emit(f"[FBX] Already exists: {fbx_path}")
            else:
                to_convert.append((i, bvh_path, fbx_path))

        if not to_convert:
            return fbx_files

        self.progress.emit(0.92, f"Converting {len(to_convert)} BVH to FBX (parallel)...")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _convert_one(bvh_path: Path, fbx_path: str):
            fbx_log = convert_bvh_to_fbx(
                str(bvh_path), fbx_path, fps=fps, naming=naming_key,
            )
            return fbx_path, fbx_log

        with ThreadPoolExecutor(max_workers=max(len(to_convert), 1)) as pool:
            futures = {}
            for idx, bvh_path, fbx_path in to_convert:
                if self._cancelled:
                    break
                futures[pool.submit(_convert_one, bvh_path, fbx_path)] = (idx, bvh_path)

            done_count = 0
            for future in as_completed(futures):
                idx, bvh_path = futures[future]
                done_count += 1
                try:
                    fbx_path, fbx_log = future.result()
                    self.log_line.emit(fbx_log)
                    if "ERROR" not in fbx_log:
                        fbx_files.append(fbx_path)
                    else:
                        self.log_line.emit(
                            f"WARNING: FBX conversion failed for {bvh_path.name}"
                        )
                except Exception as exc:
                    self.log_line.emit(
                        f"WARNING: FBX conversion failed for {bvh_path.name}: {exc}"
                    )
                frac = 0.90 + (done_count / max(len(to_convert), 1)) * 0.08
                self.progress.emit(frac, f"FBX done ({done_count}/{len(to_convert)})")

        return fbx_files

    def cancel(self):
        self._cancelled = True
