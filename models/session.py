"""Session state model — single source of truth for all application state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PersonTrack:
    """Per-person state."""

    person_id: int = -1
    person_dir: Path | None = None
    # These hold references to backend objects (IdentityTrack, etc.)
    # Typed as Any to avoid importing GVHMR backend at module level
    identity_track: Any = None
    smplx_params: dict | None = None
    confidences: list | None = None
    bboxes: np.ndarray | None = None
    original_bboxes: np.ndarray | None = None
    bbox_corrections: np.ndarray | None = None

    def to_dict(self) -> dict:
        d = {
            "person_id": self.person_id,
            "person_dir": str(self.person_dir) if self.person_dir else None,
        }
        if self.identity_track is not None and hasattr(self.identity_track, "to_dict"):
            d["identity_track"] = self.identity_track.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PersonTrack:
        return cls(
            person_id=data.get("person_id", -1),
            person_dir=Path(data["person_dir"]) if data.get("person_dir") else None,
        )


@dataclass
class Session:
    """Root state object for the entire application."""

    # Video source
    video_path: Path | None = None
    num_frames: int = 0
    fps: float = 30.0
    img_width: int = 0
    img_height: int = 0

    # Output
    output_dir: Path | None = None

    # Pipeline config
    pipeline_mode: str = "single"  # "single" | "perf" | "multi"
    static_cam: bool = True
    use_dpvo: bool = False
    focal_mm: float = 24.0

    # Multi-person tracking
    person_tracks: dict[int, PersonTrack] = field(default_factory=dict)
    inactive_tracks: set[int] = field(default_factory=set)
    crossing_spans: list[tuple[int, int, int, int]] = field(default_factory=list)

    # Pose correction (CorrectionTrack objects, keyed by person_id)
    correction_tracks: dict[int, Any] = field(default_factory=dict)

    # Camera intrinsics
    camera_K: np.ndarray | None = None

    # UI state (transient, not serialized)
    current_frame: int = 0
    selected_person: int = -1
    dirty_persons: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict (skips transient/heavy fields)."""
        d = {
            "video_path": str(self.video_path) if self.video_path else None,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "img_width": self.img_width,
            "img_height": self.img_height,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "pipeline_mode": self.pipeline_mode,
            "static_cam": self.static_cam,
            "use_dpvo": self.use_dpvo,
            "focal_mm": self.focal_mm,
            "person_tracks": {
                str(k): v.to_dict() for k, v in self.person_tracks.items()
            },
            "inactive_tracks": list(self.inactive_tracks),
            "crossing_spans": self.crossing_spans,
        }
        if self.camera_K is not None:
            d["camera_K"] = self.camera_K.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Session:
        """Deserialize from JSON dict."""
        session = cls(
            video_path=Path(data["video_path"]) if data.get("video_path") else None,
            num_frames=data.get("num_frames", 0),
            fps=data.get("fps", 30.0),
            img_width=data.get("img_width", 0),
            img_height=data.get("img_height", 0),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
            pipeline_mode=data.get("pipeline_mode", "single"),
            static_cam=data.get("static_cam", True),
            use_dpvo=data.get("use_dpvo", False),
            focal_mm=data.get("focal_mm", 24.0),
            inactive_tracks=set(data.get("inactive_tracks", [])),
            crossing_spans=[tuple(s) for s in data.get("crossing_spans", [])],
        )
        for k, v in data.get("person_tracks", {}).items():
            session.person_tracks[int(k)] = PersonTrack.from_dict(v)
        if data.get("camera_K"):
            session.camera_K = np.array(data["camera_K"])
        return session

    def save(self, path: Path):
        """Save session to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Session:
        """Load session from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def reset(self):
        """Clear all state for a fresh session."""
        self.video_path = None
        self.num_frames = 0
        self.fps = 30.0
        self.img_width = 0
        self.img_height = 0
        self.output_dir = None
        self.person_tracks.clear()
        self.inactive_tracks.clear()
        self.crossing_spans.clear()
        self.correction_tracks.clear()
        self.camera_K = None
        self.current_frame = 0
        self.selected_person = -1
        self.dirty_persons.clear()
