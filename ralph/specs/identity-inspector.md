# Spec: Identity Inspector

## Overview

The most complex panel — interactive per-person identity verification, bbox editing, keyframe management, confidence visualization, and track operations. This replaces the 2500-line `identity_panel.py` Gradio implementation.

## Layout

```
┌─────────────────────────────────────────────────┐
│ Identity Inspector                               │
├─────────────────────────────────────────────────┤
│ Person: [▼ Person 1 - "Alice"  ]  [Swap▼] [Swap]│
│ ☑ Show all tracks   [Split] [Merge▼] [Merge]    │
├─────────────────────────────────────────────────┤
│ Confidence Timeline                              │
│ ┌───────────────────────────────────────────┐    │
│ │ ▂▃▅▇▇▇▆▅▃▂▁▂▃▅▇▇▇▆▅▃▂▁▂▃▅▇  (graph)   │    │
│ │ ▲kf    ▲kf         ▲kf       (keyframes) │    │
│ └───────────────────────────────────────────┘    │
│ Detection: 0.92  Visibility: 0.85  Overlap: 0.12│
│ Shape: 0.95      Motion: 0.88      Overall: 0.87│
├─────────────────────────────────────────────────┤
│ Keyframes                                        │
│ ┌─────┬────────┬──────┬────────┬───────────┐    │
│ │Frame│Verified│ Conf │  BBox  │  Actions   │    │
│ ├─────┼────────┼──────┼────────┼───────────┤    │
│ │  42 │   ✓    │ 0.95 │256x384 │ [Go][Del] │    │
│ │ 180 │   ✗    │ 0.78 │248x380 │ [Go][Del] │    │
│ │ 360 │   ✓    │ 0.91 │260x390 │ [Go][Del] │    │
│ └─────┴────────┴──────┴────────┴───────────┘    │
│ [✓ Verify] [+ Add KF] [- Remove KF]             │
│ [◄ Prev KF] [► Next KF]                         │
├─────────────────────────────────────────────────┤
│ BBox Editing                                     │
│ Status: "Click top-left corner, then bottom-right│
│ [Cancel Edit] [Interpolate] [Apply All]          │
├─────────────────────────────────────────────────┤
│ Crossing Spans                                   │
│ ┌──────┬──────┬───────┬───────┐                  │
│ │Start │ End  │Person1│Person2│                  │
│ ├──────┼──────┼───────┼───────┤                  │
│ │  120 │  180 │   1   │   2   │                  │
│ └──────┴──────┴───────┴───────┘                  │
│ [Mark Crossing Start] [Mark Crossing End]        │
├─────────────────────────────────────────────────┤
│ Review Scanner                                   │
│ Issues found: 3   [Scan] [◄ Prev] [► Next]      │
├─────────────────────────────────────────────────┤
│ [Reprocess (2 dirty)]                            │
└─────────────────────────────────────────────────┘
```

## Components

### Person Selector

- `QComboBox` with person ID + name
- Swap controls: target dropdown + swap button
- "Show all tracks" checkbox — when checked, overlay all bboxes on frame

### Confidence Timeline (`views/confidence_timeline.py`)

Custom `QWidget` using `QPainter`:
- X-axis: frames, Y-axis: overall confidence (0-1)
- Colored band: green (>0.8), yellow (0.5-0.8), red (<0.5)
- Keyframe markers as vertical lines with triangles
- Click to seek to frame
- Current frame indicator (vertical red line)

Replaces the matplotlib-based `_render_confidence_timeline()` — QPainter is much faster and interactive.

### Confidence Breakdown

Six `QLabel` widgets showing per-component scores:
- Detection, Visibility, Overlap, Shape, Motion, Overall
- Color-coded (green/yellow/red) based on value

### Keyframe Table

`QTableWidget`:
- Columns: Frame, Verified, Confidence, BBox, Actions
- "Go" button seeks to frame
- "Del" button removes keyframe
- Rows sorted by frame index
- Double-click row to navigate

### BBox Editing

Two-click workflow (same as Gradio):
1. Click frame → sets top-left corner
2. Click again → sets bottom-right corner
3. BBox shown as dashed rectangle overlay

Status label shows current edit state. Buttons:
- Cancel: abort edit
- Interpolate: linear interpolate between keyframe bboxes
- Apply All: commit bbox corrections to disk

### Track Operations

- **Split**: Split track at current frame into two tracks
- **Merge**: Merge selected track into another (dropdown target)
- **Crossing spans**: Mark start/end of occlusion spans
  - Table showing span start, end, person IDs involved

### Review Scanner

- Scan button: auto-detect issues (low confidence, missing keyframes, large gaps)
- Issues table with prev/next navigation
- Auto-seeks to problem frames

### Reprocess Button

- Shows count of dirty persons
- Triggers `ReprocessWorker` for all dirty persons
- Disabled when no persons are dirty

## Signals

```python
class IdentityPanel(QWidget):
    person_changed = Signal(int)           # person_id
    frame_requested = Signal(int)          # seek to frame
    person_dirty = Signal(int)             # person_id needs reprocess
    reprocess_requested = Signal(list)     # list of dirty person_ids
    bbox_overlay_changed = Signal(object)  # updated overlay data for video player
    keyframe_changed = Signal(int, int)    # person_id, frame_index
```

## BBox Overlay Rendering

When the frame changes, the identity panel computes an overlay:

```python
def _compute_frame_overlay(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """Draw bboxes, keyframe markers, confidence indicators on frame."""
    overlay = frame.copy()
    for pid, track in self._session.person_tracks.items():
        bbox = self._get_bbox_at_frame(pid, frame_idx)
        if bbox is None:
            continue
        color = self._person_color(pid)
        # Draw bbox rectangle (dashed if corrected)
        cv2.rectangle(overlay, ...)
        # Draw person ID label
        cv2.putText(overlay, f"P{pid}", ...)
        # Draw confidence indicator dot
        ...
    return overlay
```

## Source Reference

- `identity_panel.py` — entire file (2502 lines)
- `identity_tracking.py` — IdentityTrack, IdentityKeyframe
- `identity_confidence.py` — TrackConfidence, compute_all_confidences
- `identity_bridge.py` — OcclusionBridge, crossing_spans_from_signal

## Acceptance Criteria

- Person selector shows all tracked persons
- Confidence timeline renders and is clickable
- Keyframe table CRUD (add, remove, verify, navigate)
- BBox editing via two-click on video frame
- Track split/merge operations work
- Crossing span marking
- Review scanner finds and navigates to issues
- Reprocess button triggers worker for dirty persons
- All state flows through Session, not module-level dicts
