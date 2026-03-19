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
    """Lazily import kimodo. Returns (load_model, FullBodyConstraintSet, SOMASkeleton30, SOMASkeleton77) or Nones."""
    try:
        from kimodo import load_model
        from kimodo.constraints import FullBodyConstraintSet
        from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77
        return load_model, FullBodyConstraintSet, SOMASkeleton30, SOMASkeleton77
    except ImportError:
        return None, None, None, None


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
            req = self._request
            self.progress.emit(0.0, "Loading Kimodo model...")

            load_model, FullBodyConstraintSet, Skel30, Skel77 = _try_import_kimodo()
            if load_model is None:
                self.error.emit(
                    "Kimodo not installed. Install from NVIDIA repo: pip install kimodo"
                )
                return

            if self._cancelled:
                return

            # Load model
            self.progress.emit(0.1, f"Loading {req.model_name}...")
            model = load_model(req.model_name)
            self.log_line.emit(f"Kimodo model loaded: {req.model_name}")

            if self._cancelled:
                return

            # Extract boundary keyframes
            poses = np.asarray(req.soma_params["poses"], dtype=np.float32)
            n_frames = poses.shape[0]
            start = max(0, req.start_frame)
            end = min(n_frames - 1, req.end_frame)
            span_length = end - start + 1

            self.progress.emit(0.2, f"Building constraints for frames {start}-{end}...")

            # Downconvert 77 → 30 for Kimodo's internal representation
            start_pose_77 = poses[start]  # (77, 3)
            end_pose_77 = poses[end]      # (77, 3)

            # Build constraint set from boundary keyframes
            constraints = FullBodyConstraintSet()
            constraints.add_keyframe(0, start_pose_77)
            constraints.add_keyframe(span_length - 1, end_pose_77)
            constraints.set_text(req.text_prompt)

            if self._cancelled:
                return

            # Run diffusion
            self.progress.emit(0.4, "Running Kimodo diffusion...")
            self.log_line.emit(f"Generating {span_length} frames: \"{req.text_prompt}\"")

            generated_30 = model.generate(
                constraints=constraints,
                n_frames=span_length,
                guidance_scale=7.5,
            )

            if self._cancelled:
                return

            self.progress.emit(0.8, "Upconverting 30→77 joints...")

            # Upconvert 30 → 77
            generated_77 = Skel30.to_SOMASkeleton77(generated_30)
            self.log_line.emit(f"Generated: {generated_77.shape[0]} frames @ 77 joints")

            # Build blended result
            self.progress.emit(0.9, "Blending...")
            blend = req.blend_frames
            blend_start = max(0, start - blend)
            blend_end = min(n_frames - 1, end + blend)

            original_region = poses[blend_start:blend_end + 1].copy()
            blended = _cosine_crossfade(original_region, generated_77, blend)

            result = KimodoResult(
                poses=generated_77,
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
