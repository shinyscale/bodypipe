# Spec: Pose Corrector

## Overview

Interactive pose correction panel with embedded 3D viewport, joint selection, euler angle sliders, quick-fix actions, and export. Replaces `pose_correction_panel.py` (1180 lines).

## Layout

```
┌─────────────────────────────────────────────────┐
│ Pose Corrector                                   │
├──────────────────────┬──────────────────────────┤
│                      │                          │
│  3D Viewport         │  Joint Controls          │
│  ┌────────────────┐  │                          │
│  │                │  │  Person: [▼ Person 1]    │
│  │  SMPL mesh     │  │  Joint:  [▼ Left Knee]   │
│  │  + skeleton    │  │                          │
│  │  click joint   │  │  Rotation (degrees)      │
│  │                │  │  X: [═══●═══] -15.3°     │
│  └────────────────┘  │  Y: [═══●═══]   5.1°    │
│                      │  Z: [═══●═══]  -2.7°    │
│  Camera: [In-cam▼]   │                          │
│  Color:  [Solid ▼]   │  [Reset Joint] [Reset All]│
│                      │                          │
├──────────────────────┤  Quick Fix                │
│                      │  [Flip Body]  [Invert]   │
│  Corrections Table   │  [Mirror L/R] [Copy From]│
│  ┌──────────────┐    │                          │
│  │Frame│Joint│..│    │  Space Overrides          │
│  ├──────────────┤    │  ┌────┬────┬──────┐      │
│  │ 42  │GO  │..│    │  │From│ To │Space │      │
│  │ 180 │GO  │..│    │  ├────┼────┼──────┤      │
│  └──────────────┘    │  │100 │200 │camera│      │
│                      │  └────┴────┴──────┘      │
│  [Delete Selected]   │  [Add Override] [Delete]  │
│                      │                          │
│  Export              │  Auto-Detect              │
│  [Re-export BVH]     │  [Detect Bad Spans]       │
│  [Re-export FBX]     │                          │
└──────────────────────┴──────────────────────────┘
```

## Components

### 3D Viewport (embedded)

Uses the `MeshViewport` widget from `3d-viewport.md`. The pose corrector hosts its own instance. When a joint is clicked in the viewport, the joint dropdown updates and euler sliders populate.

Camera mode dropdown:
- In-camera: match video perspective
- Free orbit: detached camera for inspection

Color mode dropdown:
- Solid: uniform skin tone
- Joint influence: per-joint coloring
- Confidence: confidence heatmap

### Person Selector

`QComboBox` — same person list as identity inspector. Changing person loads that person's params and correction track.

### Joint Selector

`QComboBox` with all SMPL-X joints:
- Body: 22 joints (Pelvis, L_Hip, R_Hip, ... Head)
- Hands: 30 joints (15 per hand) — collapsed group in dropdown

When selected (via dropdown or viewport click):
- Euler sliders update to show current rotation
- Joint highlighted in viewport
- Joint info label shows: name, parent, current axis-angle

### Euler Sliders

Three `QSlider` + `QDoubleSpinBox` combos:
- Range: -180° to +180°
- Step: 0.1° (spinbox), 1° (slider)
- Real-time preview: changing a slider immediately updates the viewport
- Under the hood: euler → axis-angle → update params → recompute vertices

```python
def _on_euler_changed(self):
    euler = [self._euler_x.value(), self._euler_y.value(), self._euler_z.value()]
    axis_angle = euler_deg_to_axis_angle(euler)
    # Apply to current frame's params (temporary, not committed)
    self._preview_correction(axis_angle)
    self._viewport.update()  # re-render

def _on_apply(self):
    # Commit the correction as a PoseCorrection keyframe
    correction = PoseCorrection(
        frame_index=self._current_frame,
        person_id=self._current_person,
        correction_type="joint",
        body_pose={self._current_joint: axis_angle},
    )
    self._session.correction_tracks[self._current_person].add_correction(correction)
    self._save_corrections()
```

### Quick-Fix Buttons

- **Flip Body**: 180° rotation on selected axis (yaw/pitch/roll dialog)
- **Invert Upright**: Flip global_orient about X or Z (for upside-down poses)
- **Mirror L/R**: Swap left/right joint pairs
- **Copy From Frame**: Open dialog to select source frame, create copy correction

Each wraps the corresponding function from `pose_correction.py`.

### Corrections Table

`QTableWidget`:
- Columns: Frame, Type, Joint, Values, Actions
- "Go" button seeks to frame
- "Delete" removes correction keyframe
- Sorted by frame index

### Space Overrides

`QTableWidget`:
- Columns: Start Frame, End Frame, Space, Reference Person
- Add: opens dialog for frame range + space selection
- Delete: removes selected override

Space options: "world", "camera", "carried" (carried by reference person)

### Export

- **Re-export BVH**: Apply all corrections, export to BVH
- **Re-export FBX**: Apply corrections, export to FBX via Blender subprocess

Both use `smplx_to_bvh.py:convert_smplx_to_bvh()` with correction track.

## Signals

```python
class PoseCorrectorPanel(QWidget):
    joint_selected = Signal(int)        # joint index
    correction_applied = Signal(int, int)  # person_id, frame_index
    export_requested = Signal(str)       # "bvh" or "fbx"
```

## Source Reference

- `pose_correction_panel.py` — entire file (1179 lines)
- `pose_correction.py` — CorrectionTrack, PoseCorrection, apply_corrections
- `pose_correction.py:compute_skeleton_frame()` — FK + projection
- `pose_correction.py:flip_global_orient()`, `mirror_lr_pose()`, `copy_pose_from_frame()`
- `smplx_to_bvh.py:convert_smplx_to_bvh()` — BVH export with corrections

## Acceptance Criteria

- Joint selection via viewport click or dropdown
- Euler sliders show correct values and preview changes in real-time
- Apply button commits correction to CorrectionTrack
- Quick-fix buttons work (flip, invert, mirror, copy)
- Corrections table shows all keyframes, navigable
- Space overrides can be added/removed
- BVH/FBX export applies all corrections
- All state flows through Session
