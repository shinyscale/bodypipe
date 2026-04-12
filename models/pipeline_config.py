"""Pipeline configuration — persisted solve settings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    """Persisted solve settings (replaces save_solve_config/load_solve_config)."""

    mode: str = "single"  # "single" | "perf" | "multi"
    static_cam: bool = True
    use_dpvo: bool = False
    focal_mm: float = 24.0
    use_hands: bool = True
    use_face: bool = False
    hand_mode: str = "hybrid"  # "hybrid" | "smplestx_only"
    max_persons: int = 8
    confidence_threshold: float = 0.5
    # Pipeline settings (match Gradio GUI options)
    target_fps: float = 30.0
    fbx_naming: str = "Mixamo (Cascadeur)"  # "Mixamo (Cascadeur)" | "UE5 Mannequin"
    render_overlays: bool = False
    use_inpainting: bool = True
    # Perf capture settings (Gradio Tab 2 parity)
    pitch_adjust: float = 0.0  # -30.0 to +30.0 degrees
    hand_source: str = "smplestx"  # "smplestx" | "hamer"
    body_smooth_preset: str = "moderate"  # "light" | "moderate" | "heavy"
    use_vitpose_face_crops: bool = True
    # Body model and estimation backend (SOMA migration)
    body_model: str = "smplx"            # "smplx" | "soma"
    estimation_backend: str = "gvhmr"    # "gvhmr" | "gemx"
    kimodo_model: str = "kimodo-soma-rp"  # Kimodo model for AI correction
    use_physics_refine: bool = False
    use_spring_refine: bool = False
    spring_refine_preset: str = "moderate"  # "light" | "moderate" | "heavy"
    use_foot_pin: bool = False
    cam_smooth_preset: str = "moderate"  # "light" | "moderate" | "heavy"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "static_cam": self.static_cam,
            "use_dpvo": self.use_dpvo,
            "focal_mm": self.focal_mm,
            "use_hands": self.use_hands,
            "use_face": self.use_face,
            "hand_mode": self.hand_mode,
            "max_persons": self.max_persons,
            "confidence_threshold": self.confidence_threshold,
            "target_fps": self.target_fps,
            "fbx_naming": self.fbx_naming,
            "render_overlays": self.render_overlays,
            "use_inpainting": self.use_inpainting,
            "pitch_adjust": self.pitch_adjust,
            "hand_source": self.hand_source,
            "body_smooth_preset": self.body_smooth_preset,
            "use_vitpose_face_crops": self.use_vitpose_face_crops,
            "body_model": self.body_model,
            "estimation_backend": self.estimation_backend,
            "kimodo_model": self.kimodo_model,
            "use_physics_refine": self.use_physics_refine,
            "use_spring_refine": self.use_spring_refine,
            "spring_refine_preset": self.spring_refine_preset,
            "use_foot_pin": self.use_foot_pin,
            "cam_smooth_preset": self.cam_smooth_preset,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        obj = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        # PHC physics refinement is currently non-viable on this hardware (Isaac Sim
        # fails on WSL2 + Blackwell sm_120, MuJoCo fallback produces unusable motion).
        # Force the flag off on load so a previously persisted True does not revive it.
        obj.use_physics_refine = False
        return obj

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> PipelineConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))
