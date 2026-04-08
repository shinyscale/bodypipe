"""Diagnostic CLI: compare SLAM vs body-derived vs smoothed camera trajectories.

Usage:
    python -m tools.diagnose_camera /path/to/output_dir [--person 0] [--plot]

Prints angular velocity (deg/frame) and translation velocity (m/frame)
for each camera variant.  With --plot, saves camera_diagnostic.png.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_slam(output_dir: Path, person_dirs: list[Path]):
    """Load SLAM W2C matrices from shared_slam.pt or per-person slam.pt."""
    candidates = [output_dir / "shared_slam.pt"]
    for pd in person_dirs:
        candidates.append(pd / "demo" / "preprocess" / "slam.pt")

    for path in candidates:
        if not path.is_file():
            continue
        import torch
        slam = torch.load(str(path), map_location="cpu", weights_only=False)
        arr = np.array(slam, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 7:
            from scipy.spatial.transform import Rotation
            quats_xyzw = arr[:, 3:7]
            R_w2c = Rotation.from_quat(quats_xyzw).as_matrix().transpose(0, 2, 1)
            T = np.tile(np.eye(4, dtype=np.float32), (len(arr), 1, 1))
            T[:, :3, :3] = R_w2c
            T[:, :3, 3] = arr[:, :3]
            arr = T
        print(f"Loaded SLAM from {path} — shape {arr.shape}")
        return arr
    return None


def _load_body_params(person_dir: Path):
    """Load incam + world body params from hpe_results.pt."""
    results_path = person_dir / "demo" / "hpe_results.pt"
    if not results_path.is_file():
        return None, None, None, None
    import torch
    results = torch.load(str(results_path), map_location="cpu", weights_only=False)
    incam = results.get("smpl_params_incam", {})
    world = results.get("smpl_params_global", {})
    go_i = np.array(incam.get("global_orient", []), dtype=np.float32)
    tr_i = np.array(incam.get("transl", []), dtype=np.float32)
    go_w = np.array(world.get("global_orient", []), dtype=np.float32)
    tr_w = np.array(world.get("transl", []), dtype=np.float32)
    if go_i.shape[0] == 0 or go_w.shape[0] == 0:
        return None, None, None, None
    return go_i, tr_i, go_w, tr_w


def _compute_camera_c2w(go_incam, tr_incam, go_world, tr_world):
    """Derive per-frame C2W (same as app_window._compute_camera_c2w)."""
    from scipy.spatial.transform import Rotation
    N = go_world.shape[0]
    c2w = np.zeros((N, 4, 4), dtype=np.float32)
    R_cams = Rotation.from_rotvec(go_incam).as_matrix()
    R_worlds = Rotation.from_rotvec(go_world).as_matrix()
    R_c2w = R_worlds @ np.swapaxes(R_cams, -1, -2)
    t_cam = tr_world - np.einsum("nij,nj->ni", R_c2w, tr_incam)
    c2w[:, :3, :3] = R_c2w
    c2w[:, :3, 3] = t_cam
    c2w[:, 3, 3] = 1.0
    return c2w


def _umeyama_align(src, dst):
    """Umeyama similarity alignment: find s, R, t such that dst ≈ s*R@src + t."""
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = np.sum(src_c ** 2) / n
    if var_s < 1e-12:
        return 1.0, np.eye(3, dtype=np.float32), (mu_d - mu_s).astype(np.float32)
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    return s, R.astype(np.float32), t.astype(np.float32)


def _align_slam_to_world(slam_w2c, body_c2w, n_refs=10):
    """Align SLAM W2C to GVHMR world frame via Umeyama similarity transform."""
    N = slam_w2c.shape[0]
    body_pos = body_c2w[:, :3, 3]
    slam_c2w_raw = np.linalg.inv(slam_w2c)
    slam_pos = slam_c2w_raw[:, :3, 3]

    s, R_align, t_align = _umeyama_align(slam_pos, body_pos)
    print(f"  SLAM alignment scale factor: {s:.4f}")

    aligned = np.zeros((N, 4, 4), dtype=np.float32)
    for i in range(N):
        aligned[i, :3, 3] = s * R_align @ slam_pos[i] + t_align
        aligned[i, :3, :3] = R_align @ slam_c2w_raw[i, :3, :3]
        aligned[i, 3, 3] = 1.0
    return aligned


# Import smooth function from app_window for consistency
def _get_smooth_c2w():
    """Import _smooth_c2w from app_window."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app_window import _smooth_c2w, _CAM_SMOOTH_PRESETS
    return _smooth_c2w, _CAM_SMOOTH_PRESETS


def _angular_velocity(c2w):
    """Angular velocity in degrees/frame between consecutive frames."""
    from scipy.spatial.transform import Rotation
    R = c2w[:, :3, :3]
    dR = np.swapaxes(R[:-1], -1, -2) @ R[1:]
    angles = np.abs(Rotation.from_matrix(dR).as_rotvec())
    return np.linalg.norm(angles, axis=-1) * (180.0 / np.pi)


def _translation_velocity(c2w):
    """Translation velocity in m/frame between consecutive frames."""
    t = c2w[:, :3, 3]
    return np.linalg.norm(np.diff(t, axis=0), axis=-1)


def _print_stats(name, c2w):
    av = _angular_velocity(c2w)
    tv = _translation_velocity(c2w)
    print(f"  {name:30s}  ang (deg/f): med={np.median(av):6.3f}  P95={np.percentile(av, 95):6.3f}  max={np.max(av):6.3f}  |  "
          f"trans (m/f): med={np.median(tv):6.4f}  P95={np.percentile(tv, 95):6.4f}  max={np.max(tv):6.4f}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose camera trajectories")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--person", type=int, default=0, help="Person index to use for body-derived c2w")
    parser.add_argument("--plot", action="store_true", help="Save camera_diagnostic.png")
    args = parser.parse_args()

    # Find person dirs
    person_dirs = sorted(args.output_dir.glob("person_*"))
    if not person_dirs:
        # Single person — output_dir IS the person dir
        person_dirs = [args.output_dir]

    idx = min(args.person, len(person_dirs) - 1)
    pd = person_dirs[idx]
    print(f"Using person dir: {pd}")

    # Load body params
    go_i, tr_i, go_w, tr_w = _load_body_params(pd)
    if go_i is None:
        print("ERROR: Could not load body params")
        sys.exit(1)

    c2w_body = _compute_camera_c2w(go_i, tr_i, go_w, tr_w)
    print(f"Body-derived c2w: {c2w_body.shape[0]} frames\n")

    # Collect variants
    variants = {"Body-derived (raw)": c2w_body}

    # Smoothed variants
    _smooth_c2w, _CAM_SMOOTH_PRESETS = _get_smooth_c2w()
    for preset in _CAM_SMOOTH_PRESETS:
        variants[f"One Euro ({preset})"] = _smooth_c2w(c2w_body, fps=30.0, preset=preset)

    # SLAM
    slam_w2c = _load_slam(args.output_dir, person_dirs)
    if slam_w2c is not None:
        N = min(slam_w2c.shape[0], c2w_body.shape[0])
        aligned = _align_slam_to_world(slam_w2c[:N], c2w_body[:N])
        variants["SLAM (aligned)"] = aligned
        # Trim all to same length for fair comparison
        for k in list(variants):
            variants[k] = variants[k][:N]

    print("Camera trajectory metrics:")
    for name, c2w in variants.items():
        _print_stats(name, c2w)

    if args.plot:
        _plot(variants, args.output_dir)


def _plot(variants, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for name, c2w in variants.items():
        av = _angular_velocity(c2w)
        tv = _translation_velocity(c2w)
        axes[0].plot(av, label=name, alpha=0.7)
        axes[1].plot(tv, label=name, alpha=0.7)

    axes[0].set_ylabel("Angular velocity (deg/frame)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Camera Trajectory Diagnostic")
    axes[1].set_ylabel("Translation velocity (m/frame)")
    axes[1].set_xlabel("Frame")
    axes[1].legend(fontsize=8)

    out_path = output_dir / "camera_diagnostic.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
