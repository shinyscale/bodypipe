"""Pre-compute camera rotations via GVHMR's SimpleVO for GEM-X world grounding.

GEM-X's body_params_global is broken without visual odometry input — it falls
back to identity camera trajectory, causing global_orient to track camera
panning (~347° yaw on typical orbiting shots).

This script runs SimpleVO (SIFT-based, CPU, no GPU needed) on the input video
and saves the result as ``camera.pt`` in the GEM-X preprocess directory. GEM-X
then uses these rotations in ``get_body_params_w_Rt_v2()`` to produce correct
world-grounded output.

Usage (standalone):
    python workers/simplevo_preprocess.py --video input.mp4 --output_dir outputs/demo_soma/video_name/preprocess/

Usage (from bodypipe GEM-X worker):
    Called automatically before GEM-X estimation when estimation_backend="gemx".
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_simplevo(
    video_path: str | Path,
    output_path: str | Path,
    gvhmr_root: str | Path | None = None,
    scale: float = 0.5,
    step: int = 8,
    f_mm: float = 24.0,
) -> Path:
    """Run SimpleVO on a video and save camera.pt for GEM-X.

    Parameters
    ----------
    video_path : path to input video
    output_path : where to save camera.pt (e.g. ``preprocess/camera.pt``)
    gvhmr_root : GVHMR repo root (for imports). Defaults to ``../GVHMR``.
    scale : downscale factor for VO (0.5 = half resolution, faster)
    step : frame step for feature matching (8 = every 8th frame)
    f_mm : assumed focal length in mm (24mm = standard fullframe)

    Returns
    -------
    Path to the saved camera.pt file.
    """
    import numpy as np
    import torch

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add GVHMR to path for SimpleVO import
    if gvhmr_root is None:
        gvhmr_root = Path(__file__).resolve().parent.parent.parent / "GVHMR"
    gvhmr_root = Path(gvhmr_root)
    if str(gvhmr_root) not in sys.path:
        sys.path.insert(0, str(gvhmr_root))

    from hmr4d.utils.preproc.relpose.simple_vo import SimpleVO

    logger.info(f"[SimpleVO] Running on {video_path.name} (scale={scale}, step={step})")
    vo = SimpleVO(str(video_path), scale=scale, step=step, f_mm=f_mm)
    T_w2c = vo.compute()  # (L, 4, 4) numpy — world-to-camera transforms

    # SimpleVO returns world-to-camera transforms normalized so frame 0 = identity.
    # GEM-X expects this exact format at paths.slam (camera.pt).
    if isinstance(T_w2c, torch.Tensor):
        T_w2c = T_w2c.numpy()

    torch.save(T_w2c, str(output_path))
    logger.info(f"[SimpleVO] Saved {T_w2c.shape[0]} frames → {output_path}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Pre-compute SimpleVO camera.pt for GEM-X")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output_dir", required=True, help="GEM-X preprocess directory")
    parser.add_argument("--gvhmr_root", default=None, help="GVHMR repo root")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--f_mm", type=float, default=24.0)
    args = parser.parse_args()

    output = Path(args.output_dir) / "camera.pt"
    run_simplevo(args.video, output, args.gvhmr_root, args.scale, args.step, args.f_mm)
