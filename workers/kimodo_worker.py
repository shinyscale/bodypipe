"""Kimodo text-conditioned motion correction worker.

Why: Bad motion spans (detected by pose corrector's auto-scan) can be
fixed by describing what should happen. Kimodo's diffusion model generates
motion conditioned on boundary keyframes and a text prompt, filling the gap
with plausible motion that blends smoothly at the edges.

Requires: kimodo package (pip install from NVIDIA repo).
VRAM: ~25GB (17GB diffusion + 8GB LLM2Vec encoder).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


@dataclass
class KimodoRequest:
    """Parameters for a Kimodo motion generation request."""

    soma_params: dict          # full SOMA params for the person
    start_frame: int           # first frame of the span to fill
    end_frame: int             # last frame of the span to fill (inclusive)
    text_prompt: str           # what should happen in this span
    model_name: str = "kimodo-soma-rp"
    blend_frames: int = 5     # cosine crossfade frames at boundaries


@dataclass
class KimodoResult:
    """Output from Kimodo generation."""

    poses: np.ndarray          # (span_length, 77, 3) generated poses
    start_frame: int
    end_frame: int
    blended_poses: np.ndarray  # (span_length + 2*blend, 77, 3) with crossfade applied
    blend_start: int           # first frame of blended region
    blend_end: int             # last frame of blended region


def _cosine_crossfade(
    original: np.ndarray,
    generated: np.ndarray,
    blend_frames: int,
) -> np.ndarray:
    """Apply cosine crossfade between original and generated motion.

    Parameters
    ----------
    original : (N, J, 3) original poses for the full blended region
    generated : (M, J, 3) generated poses (may be shorter than N)
    blend_frames : number of frames for crossfade at each boundary

    Returns
    -------
    (N, J, 3) blended result
    """
    N = original.shape[0]
    result = original.copy()

    # Place generated poses in the center
    gen_start = blend_frames
    gen_end = gen_start + generated.shape[0]
    if gen_end > N:
        gen_end = N
        generated = generated[:N - gen_start]

    result[gen_start:gen_end] = generated

    # Cosine blend at entry
    for i in range(min(blend_frames, gen_start)):
        t = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / (blend_frames + 1)))
        result[gen_start - blend_frames + i] = (
            (1.0 - t) * original[gen_start - blend_frames + i] + t * generated[0]
        )

    # Cosine blend at exit
    for i in range(min(blend_frames, N - gen_end)):
        t = 0.5 * (1.0 + np.cos(np.pi * (i + 1) / (blend_frames + 1)))
        result[gen_end + i] = (
            t * generated[-1] + (1.0 - t) * original[gen_end + i]
        )

    return result


def _try_import_kimodo():
    """Lazily import kimodo. Returns (load_model, FullBodyConstraintSet, SOMASkeleton30, SOMASkeleton77, axis_angle_to_matrix, matrix_to_axis_angle) or Nones."""
    try:
        from kimodo import load_model
        from kimodo.constraints import FullBodyConstraintSet
        from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77
        from kimodo.geometry import axis_angle_to_matrix, matrix_to_axis_angle
        import torch  # noqa: F401 — needed at call site
        return load_model, FullBodyConstraintSet, SOMASkeleton30, SOMASkeleton77, axis_angle_to_matrix, matrix_to_axis_angle
    except ImportError:
        return None, None, None, None, None, None


class KimodoWorker(QThread):
    """Run Kimodo diffusion to generate text-conditioned motion for a span."""

    progress = Signal(float, str)
    log_line = Signal(str)
    finished = Signal(object)  # KimodoResult
    error = Signal(str)

    def __init__(self, request: KimodoRequest, parent=None):
        super().__init__(parent)
        self._request = request
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            import torch

            req = self._request
            self.progress.emit(0.0, "Loading Kimodo model...")

            load_model, FullBodyConstraintSet, Skel30, Skel77, axis_angle_to_matrix, matrix_to_axis_angle = _try_import_kimodo()
            if load_model is None:
                self.error.emit(
                    "Kimodo not installed. Install from NVIDIA repo: pip install kimodo"
                )
                return

            if self._cancelled:
                return

            # Load model
            self.progress.emit(0.1, f"Loading {req.model_name}...")
            model = load_model(req.model_name, device="cuda")
            self.log_line.emit(f"Kimodo model loaded: {req.model_name}")

            if self._cancelled:
                return

            # Extract boundary keyframes
            poses = np.asarray(req.soma_params["poses"], dtype=np.float32)
            transl = np.asarray(req.soma_params["transl"], dtype=np.float32)
            n_frames = poses.shape[0]
            start = max(0, req.start_frame)
            end = min(n_frames - 1, req.end_frame)
            span_length = end - start + 1

            self.progress.emit(0.2, f"Building constraints for frames {start}-{end}...")

            # Boundary frames axis-angle → rotation matrices for FK
            boundary_aa = torch.tensor(
                np.stack([poses[start], poses[end]]),  # (2, 77, 3)
                dtype=torch.float32, device="cuda",
            )
            boundary_transl = torch.tensor(
                np.stack([transl[start], transl[end]]),  # (2, 3)
                dtype=torch.float32, device="cuda",
            )
            boundary_rotmats = axis_angle_to_matrix(boundary_aa)  # (2, 77, 3, 3)

            # FK for global positions/rotations using full 77-joint skeleton
            skel77 = Skel77()
            global_rots, global_pos, _ = skel77.fk(boundary_rotmats, boundary_transl)  # (2, 77, 3)

            # Build constraint from boundary global poses
            constraint = FullBodyConstraintSet(
                skeleton=skel77,
                frame_indices=torch.tensor([0, span_length - 1], device="cuda"),
                global_joints_positions=global_pos,
                global_joints_rots=global_rots,
            )

            if self._cancelled:
                return

            # Run diffusion
            self.progress.emit(0.4, "Running Kimodo diffusion...")
            self.log_line.emit(f"Generating {span_length} frames: \"{req.text_prompt}\"")

            output = model(
                prompts=req.text_prompt,
                num_frames=span_length,
                num_denoising_steps=20,
                constraint_lst=[constraint],
                cfg_weight=[2.0, 2.0],
                return_numpy=False,
                post_processing=True,
            )

            if self._cancelled:
                return

            self.progress.emit(0.8, "Upconverting 30→77 joints...")

            # Output local_rot_mats is (1, T, 30, 3, 3) — upconvert to 77 joints
            local_rot_mats = output.local_rot_mats  # (1, T, 30, 3, 3)
            rot77 = Skel30().to_SOMASkeleton77(local_rot_mats)  # (1, T, 77, 3, 3)
            generated_aa = matrix_to_axis_angle(rot77)[0].cpu().numpy()  # (T, 77, 3)
            self.log_line.emit(f"Generated: {generated_aa.shape[0]} frames @ 77 joints")

            # Build blended result
            self.progress.emit(0.9, "Blending...")
            blend = req.blend_frames
            blend_start = max(0, start - blend)
            blend_end = min(n_frames - 1, end + blend)

            original_region = poses[blend_start:blend_end + 1].copy()
            blended = _cosine_crossfade(original_region, generated_aa, blend)

            result = KimodoResult(
                poses=generated_aa,
                start_frame=start,
                end_frame=end,
                blended_poses=blended,
                blend_start=blend_start,
                blend_end=blend_end,
            )

            self.progress.emit(1.0, "Kimodo generation complete")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"Kimodo failed: {e}")
