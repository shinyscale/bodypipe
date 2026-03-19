"""Parameter format conversion between SMPL-X and SOMA representations.

Why: The session model supports both smplx_params and soma_params. When
loading legacy sessions (SMPL-X) into a SOMA-native pipeline, or exporting
SOMA results to SMPL-X-based tools, these converters bridge the gap.

Actual conversion requires py-soma-x for analytical inversion (<1° error).
These stubs establish the interface for Phase 3 implementation.
"""

from __future__ import annotations


def smplx_to_soma(smplx_params: dict) -> dict:
    """Convert SMPL-X parameters to SOMA format.

    Parameters
    ----------
    smplx_params : dict
        SMPL-X format with keys: global_orient (N, 3), body_pose (N, 21, 3),
        left_hand_pose (N, 15, 3), right_hand_pose (N, 15, 3), transl (N, 3),
        betas (1, 10).

    Returns
    -------
    dict
        SOMA format with keys: poses (N, 77, 3), transl (N, 3),
        global_orient (N, 3), identity_coeffs (1, 45), scale_params (1, 68),
        identity_model_type str.

    Raises
    ------
    NotImplementedError
        Until py-soma-x is available (Phase 3).
    """
    raise NotImplementedError(
        "SMPL-X → SOMA conversion requires py-soma-x. Install with: pip install py-soma-x"
    )


def soma_to_smplx(soma_params: dict) -> dict:
    """Convert SOMA parameters to SMPL-X format.

    Parameters
    ----------
    soma_params : dict
        SOMA format with keys: poses (N, 77, 3), transl (N, 3),
        identity_coeffs (1, 45), scale_params (1, 68).

    Returns
    -------
    dict
        SMPL-X format with keys: global_orient (N, 3), body_pose (N, 21, 3),
        left_hand_pose (N, 15, 3), right_hand_pose (N, 15, 3), transl (N, 3).

    Raises
    ------
    NotImplementedError
        Until py-soma-x is available (Phase 3).
    """
    raise NotImplementedError(
        "SOMA → SMPL-X conversion requires py-soma-x. Install with: pip install py-soma-x"
    )
