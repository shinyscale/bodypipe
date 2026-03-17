# bodypipe — Build & Test Reference

## Environment

```bash
# Activate the GVHMR conda env (has PyTorch, SMPL-X, etc.)
conda activate gvhmr

# Install Qt dependencies (one-time)
pip install PySide6 PyOpenGL PyOpenGL-accelerate

# GVHMR root (backend modules live here)
export GVHMR_ROOT=/mnt/f/GVHMR/GVHMR
```

## Build & Test Commands

```bash
# Run all tests (headless — requires QT_QPA_PLATFORM=offscreen on WSL)
QT_QPA_PLATFORM=offscreen python -m pytest tests/ -x -q

# Type check
python -m mypy bodypipe/ --ignore-missing-imports

# Smoke test (opens window briefly, loads fixture, exits)
QT_QPA_PLATFORM=offscreen python main.py --smoke-test

# Run the app
python main.py
```

## Codebase Patterns

### Widget Pattern
```python
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Signal

class MyWidget(QWidget):
    # Declare signals as class attributes
    value_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Create child widgets and layout."""
        ...

    def _connect_signals(self):
        """Wire internal signals/slots."""
        ...
```

### Worker Pattern
```python
from PySide6.QtCore import QThread, Signal

class MyWorker(QThread):
    progress = Signal(float, str)   # (fraction, message)
    finished = Signal(dict)          # result payload
    error = Signal(str)              # error message

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self._cancelled = False

    def run(self):
        try:
            # Do work, emit progress
            self.progress.emit(0.5, "Processing...")
            result = self._do_work()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

    def cancel(self):
        self._cancelled = True
```

### Session Model Pattern
```python
from dataclasses import dataclass, field
from pathlib import Path
import json

@dataclass
class Session:
    """Single source of truth for all app state."""
    video_path: Path | None = None
    output_dir: Path | None = None
    num_frames: int = 0
    fps: float = 30.0
    # ... all state here, no module-level dicts

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'Session':
        with open(path) as f:
            return cls.from_dict(json.load(f))
```

### Import Pattern for GVHMR Backend
```python
import sys
from pathlib import Path

# Add GVHMR root to path (done once in main.py)
GVHMR_ROOT = Path(__file__).resolve().parent.parent / "GVHMR"
if str(GVHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GVHMR_ROOT))

# Then import backend modules
from identity_tracking import IdentityTrack, IdentityKeyframe
from pose_correction import CorrectionTrack, PoseCorrection
```

### Frame Cache Pattern
```python
from collections import OrderedDict
import cv2
import numpy as np

class FrameCache:
    """LRU cache for decoded video frames."""

    def __init__(self, maxsize: int = 200):
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._maxsize = maxsize

    def get(self, idx: int) -> np.ndarray | None:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        return None

    def put(self, idx: int, frame: np.ndarray):
        if idx in self._cache:
            self._cache.move_to_end(idx)
        else:
            self._cache[idx] = frame
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()
```

## File Organization

```
bodypipe/
├── main.py              # Entry point, sys.path setup, QApplication
├── app_window.py        # QMainWindow shell
├── models/              # Pure data, no Qt imports
│   ├── session.py       # Session dataclass
│   └── pipeline_config.py
├── views/               # QWidget subclasses
│   ├── video_player.py
│   ├── confidence_timeline.py
│   ├── bbox_editor.py
│   ├── mesh_viewport.py
│   ├── identity_panel.py
│   ├── pose_corrector.py
│   └── pipeline_tabs.py
├── workers/             # QThread subclasses
│   ├── gvhmr_worker.py
│   ├── reprocess_worker.py
│   └── render_worker.py
├── shaders/
│   ├── mesh.vert
│   └── mesh.frag
└── tests/
    └── conftest.py
```

## GVHMR Source Reference

The Gradio app being ported lives at `/mnt/f/GVHMR/GVHMR/`. **Read these files before implementing any task.** They are your ground truth.

### GUI layer (what we're replacing with Qt)

| File | Lines | What's inside |
|------|-------|---------------|
| `gvhmr_gui.py` | 1448 | Main Gradio app: 3 tabs, `run_gvhmr()`, `run_full_pipeline()`, `run_multi_person_pipeline()`, `_run_gvhmr_subprocess()`, `_run_smplestx_subprocess()`, `save/load_solve_config()`, `find_output_dir()` |
| `identity_panel.py` | 2502 | Identity inspector: `_SESSION_DATA` dict, `_LRUFrameCache`, `_extract_frame()`, `_render_frame_with_bboxes()`, `_render_confidence_timeline()`, `build_identity_panel()`, `init_panel_state()`, all identity callbacks (`on_verify`, `on_add_keyframe`, `on_frame_click`, `on_swap_ids`, `on_split_track`, `on_merge_track`, `on_reprocess_all_dirty`, `on_scan_issues`, etc.) |
| `pose_correction_panel.py` | 1179 | Pose corrector: `_POSE_SESSION` dict, `_render_skeleton_preview()`, `_camera_space_params()`, `build_pose_correction_panel()`, `init_pose_session()`, all pose callbacks (`on_skeleton_click`, `on_euler_change`, `on_apply_correction`, `on_flip_whole_body`, `on_mirror_lr`, `on_copy_from_frame`, `on_reexport_bvh`, `on_add_space_override`, etc.) |

### Backend layer (imported read-only, never modified)

| File | Lines | What's inside |
|------|-------|---------------|
| `identity_tracking.py` | 286 | `IdentityKeyframe`, `IdentityTrack` (keyframe CRUD, nearest/surrounding queries, serialization), `auto_generate_keyframes()` |
| `identity_confidence.py` | 295 | `TrackConfidence` (detection, visibility, overlap, shape, motion → `.overall`), `compute_all_confidences()`, `confidence_to_array()` |
| `identity_bridge.py` | 440 | `OcclusionBridge` (SLERP/linear interpolation through crossings), `crossing_spans_from_signal()`, `crossing_spans_from_overlap()` |
| `pose_correction.py` | 485 | `FrameSpaceOverride`, `PoseCorrection`, `CorrectionTrack`, `apply_corrections()`, `compute_skeleton_frame()`, `find_nearest_joint()`, `flip_global_orient()`, `mirror_lr_pose()`, `copy_pose_from_frame()`, `axis_angle_to_euler_deg()`, `euler_deg_to_axis_angle()` |
| `world_assembly.py` | 433 | `compute_person_offsets()`, `apply_offsets_to_smplx()`, `apply_identity_position_constraints()`, `assemble_scene()` |
| `multi_person_split.py` | 1792 | `split_multi_person_video()` (detection→segmentation→isolation→SLAM→solve→assembly), `reprocess_person()`, `render_multi_person_incam()`, `interpolate_bbox_corrections()` |
| `smplx_to_bvh.py` | 1882 | `extract_gvhmr_params()`, `extract_smplx_params()`, `merge_gvhmr_smplestx_params()`, `convert_smplx_to_bvh()`, `activate_coordinate_space()`, smoothing filters |
| `visualize_skeleton.py` | 1005 | `forward_kinematics()`, `project_to_2d()`, `render_skeleton_frame()`, `render_skeleton_video()`, `BONE_CONNECTIONS`, `JOINT_NAMES`, `JOINT_PARENTS`, `DEFAULT_OFFSETS` |

## Style Guide

- Type hints on all public methods
- Docstrings on classes (one-liner is fine)
- No module-level mutable state — all state lives in Session or widgets
- Snake_case for methods and variables, PascalCase for classes
- Signals declared as class attributes, never instance attributes
- Color constants use the app theme (see app-shell.md spec)
