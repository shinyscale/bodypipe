"""Vendored rotation conversions from pytorch3d.transforms.

Only the functions bodypipe actually uses are included here so we can
drop the pytorch3d build dependency (CUDA kernel compilation is painful
on Windows + Blackwell).  These are pure-torch operations — no custom
CUDA code involved.

Original source: pytorch3d.transforms.rotation_conversions
License: BSD-3-Clause (Facebook Research)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrix.

    Uses Gram-Schmidt orthogonalization per Zhou et al. (CVPR 2019).

    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        Batch of rotation matrices of size (*, 3, 3)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to 6D rotation representation.

    Drops the last row of the rotation matrix per Zhou et al. (CVPR 2019).

    Args:
        matrix: Batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))
