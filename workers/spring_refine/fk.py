"""Headless forward kinematics for body-only SMPL-X skeletons.

Computes world-space positions for the first 22 joints (body) using
rest-pose offsets from ``models.skeleton.SMPLX_SKELETON``. Betas are
ignored on purpose — foot-pinning wants *consistent* foot positions
frame-to-frame (so the pin offset is meaningful), not shape-accurate
ones. Shape-aware FK is deferred to v3.

This is a port (not reuse) of ``views.mesh_viewport.forward_kinematics``
so the spring-refine worker does not pull PySide6 into its import graph.
"""

from __future__ import annotations

import numpy as np

from models.skeleton import SMPLX_SKELETON

_N_BODY = SMPLX_SKELETON.n_body_joints  # 22
_JOINT_NAMES = SMPLX_SKELETON.joint_names
_JOINT_PARENTS = SMPLX_SKELETON.joint_parents

# Precompute the (22, 3) parent-relative offset table once.
_BODY_OFFSETS = np.array(
    [SMPLX_SKELETON.default_offsets[_JOINT_NAMES[j]] for j in range(_N_BODY)],
    dtype=np.float64,
)
_BODY_PARENTS = np.array(_JOINT_PARENTS[:_N_BODY], dtype=np.int64)


def _rotvec_to_matrix_batch(rotvecs: np.ndarray) -> np.ndarray:
    """Convert ``(M, 3)`` axis-angle vectors to ``(M, 3, 3)`` rotation matrices.

    Vectorised Rodrigues — matches ``mesh_viewport._rotvec_to_matrix_batch``
    so the two FK implementations stay bit-for-bit compatible.
    """
    rotvecs = np.asarray(rotvecs, dtype=np.float64)
    angles = np.linalg.norm(rotvecs, axis=1, keepdims=True)
    safe = np.where(angles > 1e-8, angles, np.ones_like(angles))
    k = rotvecs / safe
    K = np.zeros((len(k), 3, 3), dtype=np.float64)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    sin_a = np.sin(angles)[..., np.newaxis]
    cos_a = np.cos(angles)[..., np.newaxis]
    I = np.eye(3, dtype=np.float64)[np.newaxis]
    R = I + sin_a * K + (1 - cos_a) * (K @ K)
    near_zero = (angles.ravel() < 1e-8)
    R[near_zero] = np.eye(3, dtype=np.float64)
    return R


def forward_kinematics_body(
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    transl: np.ndarray,
) -> np.ndarray:
    """Compute body-joint world positions for an entire sequence.

    Parameters
    ----------
    body_pose : ``(N, 63)`` or ``(N, 21, 3)`` axis-angle body pose.
    global_orient : ``(N, 3)`` or ``(N, 1, 3)`` axis-angle root orientation.
    transl : ``(N, 3)`` root translation.

    Returns
    -------
    positions : ``(N, 22, 3)`` float64 joint positions in world space.
    """
    bp = np.asarray(body_pose)
    if bp.ndim == 2 and bp.shape[-1] == 63:
        bp = bp.reshape(-1, 21, 3)
    elif bp.ndim == 3 and bp.shape[1:] == (21, 3):
        pass
    else:
        raise ValueError(
            f"body_pose must be (N, 63) or (N, 21, 3); got {bp.shape}"
        )
    bp = bp.astype(np.float64)

    go = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
    tr = np.asarray(transl, dtype=np.float64).reshape(-1, 3)

    N = bp.shape[0]
    if go.shape[0] != N or tr.shape[0] != N:
        raise ValueError(
            f"frame count mismatch: body_pose={N}, global_orient={go.shape[0]}, transl={tr.shape[0]}"
        )

    # Gather all (N, 22, 3) axis-angles into one flat (N*22, 3) batch.
    all_aa = np.zeros((N, _N_BODY, 3), dtype=np.float64)
    all_aa[:, 0, :] = go
    all_aa[:, 1:, :] = bp  # body_pose indexes joints 1..21

    R_all = _rotvec_to_matrix_batch(all_aa.reshape(-1, 3)).reshape(N, _N_BODY, 3, 3)

    positions = np.zeros((N, _N_BODY, 3), dtype=np.float64)
    world_R = np.zeros((N, _N_BODY, 3, 3), dtype=np.float64)

    world_R[:, 0] = R_all[:, 0]
    positions[:, 0] = tr

    for j in range(1, _N_BODY):
        parent = int(_BODY_PARENTS[j])
        # Accumulated rotation from world frame down through this joint's parent.
        world_R[:, j] = world_R[:, parent] @ R_all[:, j]
        # Child position = parent position + parent_world_R * offset.
        offset = _BODY_OFFSETS[j]
        positions[:, j] = positions[:, parent] + (world_R[:, parent] @ offset)

    return positions
