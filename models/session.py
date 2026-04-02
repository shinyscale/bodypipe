"""Session state model — single source of truth for all application state.

Why: A single Session dataclass tree eliminates scattered module-level dicts
and gives all widgets a shared, serializable state object. The UndoStack
enables non-destructive editing — users can freely experiment with bbox edits,
track operations, and pose corrections knowing they can revert any mistake.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Undo / Redo
# ---------------------------------------------------------------------------

@dataclass
class UndoEntry:
    """A single undoable operation with forward and reverse callables."""

    description: str
    undo_fn: Callable[[], None]
    redo_fn: Callable[[], None]


class UndoStack:
    """Bounded undo/redo stack using the command pattern.

    Why: Interactive editing sessions (bbox corrections, track swaps, pose
    adjustments) are error-prone. An undo stack lets users experiment freely —
    max_depth keeps memory bounded, and the on_changed callback lets the UI
    (Edit menu) stay in sync without polling.
    """

    def __init__(self, max_depth: int = 50):
        self._undo: list[UndoEntry] = []
        self._redo: list[UndoEntry] = []
        self._max_depth = max_depth
        self.on_changed: Callable[[], None] | None = None

    def push(self, entry: UndoEntry) -> None:
        """Record a new undoable operation (clears redo history)."""
        self._undo.append(entry)
        if len(self._undo) > self._max_depth:
            self._undo.pop(0)
        self._redo.clear()
        self._notify()

    def undo(self) -> str | None:
        """Undo the most recent operation. Returns its description, or None."""
        if not self._undo:
            return None
        entry = self._undo.pop()
        try:
            entry.undo_fn()
        except Exception:
            log.exception("Undo failed for '%s'", entry.description)
        self._redo.append(entry)
        self._notify()
        return entry.description

    def redo(self) -> str | None:
        """Redo the most recently undone operation. Returns its description, or None."""
        if not self._redo:
            return None
        entry = self._redo.pop()
        try:
            entry.redo_fn()
        except Exception:
            log.exception("Redo failed for '%s'", entry.description)
        self._undo.append(entry)
        self._notify()
        return entry.description

    def can_undo(self) -> bool:
        return len(self._undo) > 0

    def can_redo(self) -> bool:
        return len(self._redo) > 0

    def peek_undo(self) -> str | None:
        """Description of the next operation that would be undone."""
        return self._undo[-1].description if self._undo else None

    def peek_redo(self) -> str | None:
        """Description of the next operation that would be redone."""
        return self._redo[-1].description if self._redo else None

    def clear(self) -> None:
        """Discard all undo/redo history."""
        self._undo.clear()
        self._redo.clear()
        self._notify()

    def _notify(self) -> None:
        if self.on_changed is not None:
            self.on_changed()


@dataclass
class PersonTrack:
    """Per-person state."""

    person_id: int = -1
    person_dir: Path | None = None
    # These hold references to backend objects (IdentityTrack, etc.)
    # Typed as Any to avoid importing GVHMR backend at module level
    identity_track: Any = None
    smplx_params: dict | None = None
    soma_params: dict | None = None
    body_model_type: str = "smplx"  # "smplx" | "soma"
    confidences: list | None = None
    bboxes: np.ndarray | None = None
    original_bboxes: np.ndarray | None = None
    bbox_corrections: np.ndarray | None = None
    # User-created keyframes for identity verification
    keyframes: list[dict] = field(default_factory=list)
    # Per-component confidence breakdown (computed from backend, not serialized)
    # Keys: "detection", "visibility", "overlap", "shape", "motion", "overall"
    confidence_breakdown: dict[str, list[float]] | None = None
    # Crop metadata for world-grounding (loaded from person_meta.json / hpe_results.pt)
    crop_bbox: list[int] | None = None  # [x1, y1, x2, y2] in original video coords
    K_crop: np.ndarray | None = None  # (3,3) intrinsics for the crop camera

    def to_dict(self) -> dict:
        # Ensure keyframe confidence values are JSON-serializable
        # (TrackConfidence objects → dicts or floats)
        safe_kfs = []
        for kf in self.keyframes:
            kf_copy = dict(kf)
            conf = kf_copy.get("confidence")
            if conf is not None and hasattr(conf, "to_dict"):
                kf_copy["confidence"] = conf.to_dict()
            elif conf is not None and not isinstance(conf, (int, float, dict, type(None))):
                kf_copy["confidence"] = float(conf) if hasattr(conf, "__float__") else None
            safe_kfs.append(kf_copy)

        d = {
            "person_id": self.person_id,
            "person_dir": str(self.person_dir) if self.person_dir else None,
            "body_model_type": self.body_model_type,
            "keyframes": safe_kfs,
        }
        if self.crop_bbox is not None:
            d["crop_bbox"] = self.crop_bbox
        if self.identity_track is not None and hasattr(self.identity_track, "to_dict"):
            d["identity_track"] = self.identity_track.to_dict()
        if self.bbox_corrections is not None:
            d["bbox_corrections"] = self.bbox_corrections.tolist()
        if self.original_bboxes is not None:
            d["original_bboxes"] = self.original_bboxes.tolist()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> PersonTrack:
        pt = cls(
            person_id=data.get("person_id", -1),
            person_dir=Path(data["person_dir"]) if data.get("person_dir") else None,
            body_model_type=data.get("body_model_type", "smplx"),
            keyframes=data.get("keyframes", []),
            crop_bbox=data.get("crop_bbox"),
        )
        if "bbox_corrections" in data:
            pt.bbox_corrections = np.array(data["bbox_corrections"], dtype=float)
        if "original_bboxes" in data:
            pt.original_bboxes = np.array(data["original_bboxes"], dtype=float)
        return pt


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
    crossing_spans: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    crossing_threshold: float = 0.15  # bbox overlap IoU threshold for auto-crossing detection

    # Pose correction (CorrectionTrack objects, keyed by person_id)
    correction_tracks: dict[int, Any] = field(default_factory=dict)

    # Camera intrinsics
    camera_K: np.ndarray | None = None
    K_orig: np.ndarray | None = None  # (3,3) original video intrinsics (GEM-X convention)
    slam_c2w: np.ndarray | None = None  # (N, 4, 4) per-frame camera-to-world

    # Session metadata (serialized)
    notes: str = ""
    tags: list[str] = field(default_factory=list)
    version: int = 1

    # UI state (transient, not serialized)
    current_frame: int = 0
    selected_person: int = -1
    dirty_persons: set[int] = field(default_factory=set)

    # Undo/redo (transient, not serialized)
    undo_stack: UndoStack = field(default_factory=UndoStack)

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
            "crossing_spans": {
                str(k): v for k, v in self.crossing_spans.items()
            },
            "crossing_threshold": self.crossing_threshold,
        }
        if self.camera_K is not None:
            d["camera_K"] = self.camera_K.tolist()
        d["notes"] = self.notes
        d["tags"] = list(self.tags)
        d["version"] = self.version
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
            crossing_spans=(
                {
                    int(k): [tuple(s) for s in v]
                    for k, v in data.get("crossing_spans", {}).items()
                }
                if isinstance(data.get("crossing_spans", {}), dict)
                else {}
            ),
        )
        session.crossing_threshold = data.get("crossing_threshold", 0.15)
        session.notes = data.get("notes", "")
        session.tags = list(data.get("tags", []))
        session.version = data.get("version", 1)
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
        """Clear all state for a fresh session.

        Why: Every field must return to its dataclass default so that loading
        a new video or session starts from a clean slate.  Pipeline config
        fields (pipeline_mode, static_cam, use_dpvo, focal_mm) were previously
        missed, which could leave stale settings from a prior session.
        """
        self.video_path = None
        self.num_frames = 0
        self.fps = 30.0
        self.img_width = 0
        self.img_height = 0
        self.output_dir = None
        # Pipeline config — reset to dataclass defaults
        self.pipeline_mode = "single"
        self.static_cam = True
        self.use_dpvo = False
        self.focal_mm = 24.0
        # Multi-person tracking
        self.person_tracks.clear()
        self.inactive_tracks.clear()
        self.crossing_spans.clear()
        self.correction_tracks.clear()
        self.camera_K = None
        self.K_orig = None
        self.slam_c2w = None
        self.notes = ""
        self.tags.clear()
        self.version = 1
        self.current_frame = 0
        self.selected_person = -1
        self.dirty_persons.clear()
        self.undo_stack.clear()
