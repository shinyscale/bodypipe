# Spec: State Management

## Overview

Replace the scattered module-level `_SESSION_DATA` and `_POSE_SESSION` dicts with a single `Session` dataclass tree. The Session is the single source of truth — widgets read from it and mutate it through defined methods, never through direct dict access.

## Data Model

### Session (`models/session.py`)

```python
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

    # Pose correction
    correction_tracks: dict[int, Any] = field(default_factory=dict)  # CorrectionTrack objects

    # Camera
    camera_K: np.ndarray | None = None

    # UI state (not serialized)
    current_frame: int = 0
    selected_person: int = -1
    dirty_persons: set[int] = field(default_factory=set)


@dataclass
class PersonTrack:
    """Per-person state."""
    person_id: int
    person_dir: Path | None = None
    identity_track: Any = None       # IdentityTrack object
    smplx_params: dict | None = None
    confidences: list | None = None
    bboxes: np.ndarray | None = None
    original_bboxes: np.ndarray | None = None
    bbox_corrections: np.ndarray | None = None
```

### Serialization

- `Session.to_dict()` / `Session.from_dict()` for JSON persistence
- Skip transient fields: frame_cache, smplx_params tensors (loaded on demand from .pt files)
- Save to `{output_dir}/bodypipe_session.json`
- Load existing Gradio session data: parse `_SESSION_DATA` layout into `Session`

### Undo/Redo Stack

```python
@dataclass
class UndoEntry:
    description: str
    undo_fn: Callable[[], None]
    redo_fn: Callable[[], None]

class UndoStack:
    def __init__(self, max_depth: int = 50):
        self._undo: list[UndoEntry] = []
        self._redo: list[UndoEntry] = []

    def push(self, entry: UndoEntry): ...
    def undo(self) -> str | None: ...
    def redo(self) -> str | None: ...
    def can_undo(self) -> bool: ...
    def can_redo(self) -> bool: ...
    def clear(self): ...
```

Undo/redo is optional for Phase 1 — implement the stack interface but wire it up later.

## Pipeline Config Persistence (`models/pipeline_config.py`)

```python
@dataclass
class PipelineConfig:
    """Persisted solve settings (replaces save_solve_config/load_solve_config)."""
    mode: str = "single"
    static_cam: bool = True
    use_dpvo: bool = False
    focal_mm: float = 24.0
    use_hands: bool = True
    use_face: bool = False
    hand_mode: str = "hybrid"  # "hybrid" | "smplestx_only"

    def save(self, path: Path): ...

    @classmethod
    def load(cls, path: Path) -> 'PipelineConfig': ...
```

## Session Lifecycle

1. **New session**: User opens a video → `Session(video_path=..., num_frames=..., fps=...)`
2. **Pipeline complete**: Worker emits result → populate `person_tracks`, `output_dir`
3. **Load existing**: Open session JSON → `Session.load(path)` → reload .pt params on demand
4. **Save**: `Session.save()` → writes JSON (lightweight, params stay as .pt files)
5. **Close**: Clear session, reset all widgets

## Integration with Widgets

Widgets receive the Session object (not a copy). They read from it directly and call mutation methods that can hook into undo/redo. The AppWindow owns the Session and passes it to child widgets.

```python
# In AppWindow
self._session = Session()

# Pass to widgets
self.identity_panel.set_session(self._session)
self.pose_corrector.set_session(self._session)
```

## Source Reference

- `identity_panel.py:_SESSION_DATA` — keys and structure
- `pose_correction_panel.py:_POSE_SESSION` — keys and structure
- `gvhmr_gui.py:save_solve_config/load_solve_config` — config persistence

## Acceptance Criteria

- Session dataclass round-trips through JSON (save/load)
- PersonTrack correctly maps to existing person_dir structure
- PipelineConfig replaces solve_config.json format
- No module-level mutable state in any bodypipe module
