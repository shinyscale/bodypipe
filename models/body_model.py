"""Body model adapters — protocol + implementations for SMPL-X and SOMA.

Why: The viewport, pose corrector, and export pipeline all need to compute
forward kinematics and access skeleton metadata. This protocol decouples
those consumers from the specific body model format, enabling SOMA support
alongside the existing SMPL-X path without touching consumer code.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from models.skeleton import SkeletonDef, SMPLX_SKELETON, SOMA_SKELETON


@runtime_checkable
class BodyModelAdapter(Protocol):
    """Interface for body model forward passes."""

    @property
    def skeleton(self) -> SkeletonDef:
        """The skeleton definition for this body model."""
        ...

    def forward_kinematics(self, params: dict, frame_idx: int) -> np.ndarray:
        """Compute joint positions for one frame.

        Returns (n_joints, 3) float64 positions in camera/world space.
        """
        ...


class SmplxAdapter:
    """SMPL-X body model adapter wrapping the existing FK implementation."""

    @property
    def skeleton(self) -> SkeletonDef:
        return SMPLX_SKELETON

    def forward_kinematics(self, params: dict, frame_idx: int) -> np.ndarray:
        # Import here to avoid circular dependency (mesh_viewport imports skeleton)
        from views.mesh_viewport import forward_kinematics
        return forward_kinematics(params, frame_idx)


class SomaAdapter:
    """SOMA body model adapter — placeholder until py-soma-x is available."""

    @property
    def skeleton(self) -> SkeletonDef:
        return SOMA_SKELETON

    def forward_kinematics(self, params: dict, frame_idx: int) -> np.ndarray:
        raise NotImplementedError(
            "SOMA forward kinematics requires py-soma-x (Phase 3)"
        )
