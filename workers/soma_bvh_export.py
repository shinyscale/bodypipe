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


def _compute_soma_shape_offsets(soma_params: dict) -> dict[str, list[float]] | None:
    """Compute shape-dependent bone offsets from SOMA body model.

    Returns dict mapping joint name → [x, y, z] parent-relative offset,
    or None if SOMA body model or identity data is unavailable.
    """
    ic = soma_params.get("identity_coeffs")
    sc = soma_params.get("scale_params")

    try:
        import torch
        from gem.utils.soma_utils.soma_layer import SomaLayer
        from soma.assets import get_assets_dir

        data_root = str(get_assets_dir())
        soma = SomaLayer(
            data_root=data_root,
            low_lod=True,
            device="cpu",
            identity_model_type="soma",
            mode="dense",
        )
        soma.eval()

        n_coeffs = 128
        if ic is not None:
            ic_t = torch.tensor(np.asarray(ic), dtype=torch.float32)
            if ic_t.ndim == 1:
                ic_t = ic_t.unsqueeze(0)
            if ic_t.shape[-1] != n_coeffs:
                ic_t = torch.zeros(1, n_coeffs)
        else:
            ic_t = torch.zeros(1, n_coeffs)

        sc_t = None
        if sc is not None:
            sc_t = torch.tensor(np.asarray(sc), dtype=torch.float32)
            if sc_t.ndim == 1:
                sc_t = sc_t.unsqueeze(0)
        if sc_t is None:
            sc_t = torch.ones(1, 1)

        with torch.no_grad():
            rest_joints = soma.get_skeleton(ic_t, sc_t)  # (1, 77, 3)
        joints_np = rest_joints[0].cpu().numpy()

        parents = SOMA_SKELETON.joint_parents
        result: dict[str, list[float]] = {}
        for i, name in enumerate(SOMA_SKELETON.joint_names):
            if i == 0:
                result[name] = [0.0, 0.0, 0.0]
            else:
                off = joints_np[i] - joints_np[parents[i]]
                result[name] = off.tolist()
        return result

    except Exception as e:
        logger.debug("SOMA shape offsets unavailable: %s", e)
        return None


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

    # Get rest-pose offsets — use shape-dependent offsets from SOMA body model
    # when identity data is available, otherwise fall back to skeleton defaults.
    offsets = np.zeros((n_joints, 3), dtype=np.float32)
    shape_offsets = _compute_soma_shape_offsets(soma_params)
    if shape_offsets is not None:
        for i, name in enumerate(names):
            offsets[i] = shape_offsets.get(name, [0.0, 0.0, 0.0])
        logger.info("BVH using shape-dependent offsets from SOMA body model")
    else:
        for i, name in enumerate(names):
            offsets[i] = SOMA_SKELETON.default_offsets.get(name, [0.0, 0.0, 0.0])

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
