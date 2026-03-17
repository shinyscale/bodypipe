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
        }

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> PipelineConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))
