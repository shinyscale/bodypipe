"""Spring-based motion refinement.

Per-joint critically-damped second-order IIR filter applied in quaternion
log space. Linear IIR implementation of a PD controller with zero target
velocity — pure numpy + scipy, CPU-only, deterministic. Replaces the
dormant PHC physics refinement stage on hardware where Isaac Sim + PhysX
are non-viable (WSL2 + Blackwell sm_120).
"""

from __future__ import annotations

from .footpin import apply_foot_pin, compute_pin_offset
from .metrics import (
    classify_verdict,
    compute_foot_skating_meters,
    write_spring_metrics_json,
)
from .runner import refine_body_sequence, run_spring_refine

__all__ = [
    "apply_foot_pin",
    "classify_verdict",
    "compute_foot_skating_meters",
    "compute_pin_offset",
    "refine_body_sequence",
    "run_spring_refine",
    "write_spring_metrics_json",
]
