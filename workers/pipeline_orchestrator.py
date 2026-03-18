"""Orchestration workers for multi-stage pipelines."""

from __future__ import annotations

import logging
import sys
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
    (0.52, 0.72, "Face pipeline"),
    (0.72, 0.85, "BVH/FBX conversion"),
    (0.85, 1.00, "Rendering"),
]


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
    for key in [
        "transl_cam", "transl_world",
        "global_orient_cam", "global_orient_world",
        "body_pose_cam", "body_pose_world",
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

            # Stage 4: Face pipeline
            self._emit_stage(4)
            if self._config.use_face:
                self._run_face(results)
            else:
                self.log_line.emit("Face capture disabled, skipping.")
            if self._cancelled:
                return

            # Stage 5: BVH/FBX conversion
            self._emit_stage(5)
            self._run_bvh_fbx(results, world_params, smplestx_ran)
            if self._cancelled:
                return

            # Stage 6: Rendering
            self._emit_stage(6)
            self._run_rendering(results, camera_params, world_params, smplestx_ran)
            if self._cancelled:
                return

            self.progress.emit(1.0, "Pipeline complete")
            results["output_dir"] = str(self._output_dir)
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))

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
            hamer_result = run_hamer(
                video_path=str(self._video_path),
                output_dir=str(self._output_dir),
            )
            if hamer_result is not None:
                # Merge HaMeR hand poses into world and camera params
                for params in (world_params, camera_params):
                    if params is not None and "left_hand_pose" in hamer_result:
                        params["left_hand_pose"] = hamer_result["left_hand_pose"]
                    if params is not None and "right_hand_pose" in hamer_result:
                        params["right_hand_pose"] = hamer_result["right_hand_pose"]
                self.log_line.emit("HaMeR hands merged successfully.")
            else:
                self.log_line.emit("WARNING: HaMeR returned no results, keeping SMPLest-X hands.")
        except Exception as exc:
            self.log_line.emit(f"WARNING: HaMeR failed: {exc}, keeping SMPLest-X hands.")

        return world_params, camera_params

    # ------------------------------------------------------------------
    # Stage 4: Face pipeline
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
        progress_cb = self._stage_progress_callback(4)
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
    # Stage 5: BVH/FBX conversion
    # ------------------------------------------------------------------

    def _run_bvh_fbx(
        self, results: dict, world_params: dict | None, is_hybrid: bool
    ) -> None:
        """Convert SMPL-X params to BVH, then BVH to FBX via Blender."""
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

    # ------------------------------------------------------------------
    # Stage 6: Rendering
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

        progress_cb = self._stage_progress_callback(6)
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
        """Emit progress for the start of stage *idx*."""
        start, _end, label = _FULL_STAGES[idx]
        self.progress.emit(start, label)

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
        cmd = [
            sys.executable,
            "tools/demo/demo.py",
            f"--video={self._video_path}",
        ]
        if self._config.static_cam:
            cmd.append("--static_cam")
        if self._config.use_dpvo:
            cmd.append("--use_dpvo")
        if self._config.focal_mm != 24.0:
            cmd.append(f"--f_mm={self._config.focal_mm}")
        return cmd

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

    def run(self):
        try:
            from multi_person_split import split_multi_person_video

            def progress_callback(frac: float, msg: str):
                overall = 0.05 + frac * 0.85
                self.progress.emit(overall, msg)
                self.log_line.emit(f"[MultiPerson] {msg}")

            self.progress.emit(0.02, "Starting multi-person pipeline...")

            result = split_multi_person_video(
                video_path=str(self._video_path),
                output_dir=str(self._output_dir),
                static_cam=self._config.static_cam,
                use_dpvo=self._config.use_dpvo,
                max_persons=self._config.max_persons,
                render_overlays=self._config.render_overlays,
                use_inpainting=self._config.use_inpainting,
                progress_callback=progress_callback,
            )

            if self._cancelled:
                return

            self.progress.emit(1.0, "Multi-person pipeline complete")
            self.finished.emit({
                "output_dir": str(self._output_dir),
                "result": result,
            })

        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True
