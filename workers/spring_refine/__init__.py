"""Spring-based motion refinement.

Per-joint critically-damped second-order IIR filter applied in quaternion
log space. Linear IIR implementation of a PD controller with zero target
velocity — pure numpy + scipy, CPU-only, deterministic. Replaces the
dormant PHC physics refinement stage on hardware where Isaac Sim + PhysX
are non-viable (WSL2 + Blackwell sm_120).
"""

from __future__ import annotations

from .runner import refine_body_sequence, run_spring_refine

__all__ = ["refine_body_sequence", "run_spring_refine"]
