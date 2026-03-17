# Spec: Multi-Person Tab

## Overview

Third tab — multi-person pipeline with identity inspector and pose corrector embedded. This is the most complex tab, hosting the identity panel and 3D viewport.

## Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│ Multi-Person Capture                                                 │
├─────────────────────┬───────────────────────────────────────────────┤
│                     │                                               │
│  Video Input        │  ┌─────────────────────────────────────────┐  │
│  [same as Tab 1]    │  │         Main Viewport                   │  │
│                     │  │  (VideoPlayer + bbox overlay             │  │
│  Pipeline Settings  │  │   OR 3D mesh viewport)                  │  │
│  [same as Tab 1]    │  │                                         │  │
│  + Multi-person:    │  └─────────────────────────────────────────┘  │
│  Max persons: 8     │                                               │
│  Confidence thresh   │  ┌──────────────────┬──────────────────────┐  │
│                     │  │ Identity Inspector│  Pose Corrector      │  │
│  [▶ Run Pipeline]   │  │ (left sub-panel)  │  (right sub-panel)   │  │
│  ████░░░ 30%        │  │                  │                      │  │
│                     │  │  Person selector  │  Joint selector      │  │
│  Track Overview     │  │  Keyframes table  │  Euler sliders       │  │
│  • Person 1: ██████ │  │  Confidence viz   │  Quick-fix buttons   │  │
│  • Person 2: ██████ │  │  Track operations │  Corrections table   │  │
│  • Person 3: ██░░░░ │  │  Reprocess button │  BVH/FBX export      │  │
│                     │  │                  │                      │  │
└─────────────────────┴──┴──────────────────┴──────────────────────┘
```

## Components

### Left Sidebar — Pipeline Control

- Video input + settings (shared with other tabs)
- Multi-person specific settings:
  - Max persons spinbox
  - Confidence threshold slider
- Run button + progress

### Track Overview

After pipeline completes, shows a timeline bar per person:
- Green = high confidence, Yellow = medium, Red = low
- Click to select person + seek to frame

### Main Viewport

Switches between:
1. **Video + BBox overlay** — default, shows current frame with tracked bboxes
2. **3D Mesh viewport** — when pose corrector is active, shows SMPL mesh

Toggle via toolbar buttons above viewport.

### Bottom Panels

`QSplitter` horizontal split between Identity Inspector (left) and Pose Corrector (right). Either can be collapsed.

- **Identity Inspector**: see `identity-inspector.md` spec
- **Pose Corrector**: see `pose-corrector.md` spec

## Signal Wiring

The multi-person tab is the signal hub:

```python
# Frame sync: all panels share frame position
self.video_player.frame_changed.connect(self.identity_panel.on_frame_changed)
self.video_player.frame_changed.connect(self.pose_corrector.on_frame_changed)
self.video_player.frame_changed.connect(self.mesh_viewport.on_frame_changed)

# Person selection: identity panel drives
self.identity_panel.person_changed.connect(self.pose_corrector.set_person)
self.identity_panel.person_changed.connect(self.mesh_viewport.set_person)

# Viewport clicks: route to active editor
self.video_player.frame_clicked.connect(self._route_click)
self.mesh_viewport.joint_clicked.connect(self.pose_corrector.set_joint)

# Dirty state: identity edits mark persons dirty
self.identity_panel.person_dirty.connect(self._mark_dirty)

# Reprocess: after identity edits
self.identity_panel.reprocess_requested.connect(self._run_reprocess)
```

## Source Reference

- `gvhmr_gui.py:1400-1448` — Tab 3 Gradio layout
- `gvhmr_gui.py:run_multi_person_pipeline()` — pipeline
- `identity_panel.py:build_identity_panel()` — all identity UI
- `pose_correction_panel.py:build_pose_correction_panel()` — pose UI

## Acceptance Criteria

- Pipeline runs and shows per-person track timeline
- Frame navigation syncs across all sub-panels
- Person selection propagates to pose corrector and viewport
- Identity inspector and pose corrector functional (see their specs)
- Splitter layout persists across sessions
