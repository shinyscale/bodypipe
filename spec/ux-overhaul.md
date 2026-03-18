# bodypipe UX Overhaul Spec

> Synthesized from strixhalo research tasks #721 (mocap tool UX), #722 (VFX tool UX), #723 (Qt/PySide6 best practices), plus Deck Nine ShootDay color extraction. Current bodypipe: 49 commits, 24.7K LOC, 1160 tests, all passing.

---

## Current State (what exists)

| Area | Status |
|------|--------|
| 3D Viewport | QOpenGLWidget, Phong shading, orbit + incam camera, skeleton overlay, FBO joint picking, 3 color modes (solid/joint/confidence), vertex cache (LRU 200) |
| Video Player | Transport bar (7 buttons), slider, LRU frame cache (200 frames, 15 read-ahead), keyboard seek, play/pause |
| Identity Inspector | Person dropdown, confidence table (6 metrics), confidence timeline (QPainter), keyframe management, bbox editing (2-click), review scanner (angular jumps, jitter, low confidence) |
| Pose Corrector | Euler XYZ sliders + spinboxes, quick-fix buttons (flip 180, invert, mirror L/R, copy from frame), corrections table, space overrides, BVH/FBX export |
| Layout | Fixed QTabWidget (3 tabs), log dock (bottom), status bar (frame + FPS), menu bar |
| Theme | Deck Nine palette (`theme.py`), Fusion style + QPalette + QSS |
| Session | Dataclass with JSON save/load, UndoStack (50 depth), QSettings geometry restore |
| Bbox Overlay | OpenCV rendering, per-person color palette, dashed corrections, confidence dots |

---

## Phase 1: Dockable Layout + Workspace Presets

**Why first**: Every pattern in the research assumes panels can be rearranged. Fixed tabs are the single biggest gap between bodypipe and professional tools. This unblocks everything else.

### 1.1 Replace QTabWidget with QDockWidget panels

Convert all major panels to dock widgets:

| Panel | Current Location | Dock Area |
|-------|-----------------|-----------|
| Video Player | Embedded in each tab | Center (primary) |
| 3D Viewport | Embedded in multi-person tab | Center (tabbed with video, or side-by-side) |
| Identity Inspector | Embedded in multi-person tab | Right |
| Pose Corrector | Embedded in multi-person tab | Right (tabbed with Identity) |
| Pipeline Settings | Left sidebar in each tab | Left |
| Confidence Timeline | Inline in Identity Inspector | Bottom (tabbed with Log) |
| Log Panel | Bottom dock (already a dock) | Bottom |

**Implementation**: Use [Qt Advanced Docking System](https://github.com/githubuser0xFFFF/Qt-Advanced-Docking-System) (PySide6 compatible). It supports docking, tabbing, floating, and serialization. Natron, DaVinci Resolve, and multiple Deck Nine tools use this pattern.

If the ADS dependency is too heavy, Qt's built-in `QDockWidget` with `setAllowedAreas()` + `tabifyDockWidget()` is sufficient for v1.

### 1.2 Workspace presets

Save/restore named layouts via `QMainWindow.saveState()` / `restoreState()`:

| Preset | Layout |
|--------|--------|
| **Review** | Video large center, Inspector right, Timeline bottom |
| **Correction** | Video + 3D Viewport side-by-side center, Pose Corrector right, Timeline bottom |
| **Tracking** | Video large center, Identity Inspector right (expanded), Track Overview bottom |
| **Pipeline** | Video center, Pipeline Settings left, Log bottom (expanded) |

Add `View > Workspace` submenu with preset selection + "Save Current Layout..." + "Reset to Default".

### 1.3 Pipeline mode switcher

Replace the 3 tabs with a mode selector (toolbar dropdown or top-bar radio buttons) that swaps which pipeline settings are shown in the left dock. The viewport, inspector, and corrector panels stay visible across all modes.

Modes: **Single Person** | **Performance Capture** | **Multi-Person**

---

## Phase 2: Vertical Track Timeline

**Why second**: The confidence timeline is currently a flat QPainter bar per-person. A vertical track-based timeline (like DAWs and Motive/MVN) scales to multiple people and is the backbone for scrubbing, marker display, and issue flagging.

### 2.1 Track layout

```
┌─────────────────────────────────────────────────┐
│ Person 0 ● ▼  ═══█████████████████████════════  │  ← collapsible
│ Person 1 ● ▼  ════════█████████████════════════  │
│ Person 2 ● ▼  ══════════════████████████════════ │
├─────────────────────────────────────────────────┤
│ ◄  1    ▶  ▶▶  │  0:12.4 / 0:45.0  │ 1x ▼     │  ← transport
└─────────────────────────────────────────────────┘
```

Each person track is a horizontal lane, height ~24px, showing:
- **Confidence heatmap**: Color fill per frame (green >0.8, yellow 0.5–0.8, red <0.5)
- **Keyframe markers**: Small triangles at bottom of lane (green = verified)
- **Issue flags**: Red diamonds for auto-detected problems (angular jumps, jitter)
- **Correction markers**: Amber dots where user has applied pose corrections
- **Overlap regions**: Blue tint where SAM2 detected person overlap (inpaint zones)

Clicking a track lane selects that person (updates Inspector + Corrector). Clicking a position seeks the video.

### 2.2 Color coding per person

Assign distinct track colors from the existing 8-color palette in `bbox_overlay.py`. Track header shows: colored dot + "Person N" label + collapse arrow.

### 2.3 Implementation

Start with `QGraphicsView` + custom `QGraphicsItem` subclasses for tracks, markers, and the playhead. `QGraphicsView` handles zoom (scroll-wheel scales X axis, Y locked), pan (middle-drag), and hit testing natively.

Embed as a dock widget docked bottom. Share the playhead position with `VideoPlayer` bidirectionally via signal/slot.

---

## Phase 3: Transport & Scrubbing Polish

### 3.1 Viewport-embedded transport

Overlay transport controls semi-transparently on the Video Player widget (like Mocha Pro). Auto-hide after 2s of mouse inactivity. Show on mouse enter.

Contents: play/pause, frame-back/forward, current frame / total, timecode, speed selector.

### 3.2 Speed presets

Replace freeform speed control with toggle chips: **0.25x** | **0.5x** | **1x** | **2x** | **4x**. Display in transport bar and track timeline footer.

### 3.3 Scrub performance

- Auto-switch 3D viewport to wireframe skeleton during active scrubbing (restore on pause)
- Frame cache read-ahead already exists (15 frames) — increase to 30 during play
- Decode in background thread, push frames to main via signal (already the pattern in VideoPlayer)

---

## Phase 4: Joint Selection & Chain Highlighting

### 4.1 Chain select on click

When user clicks a joint in the 3D viewport (FBO pick), highlight the entire kinematic chain from root to that joint as a wireframe path. Already have `JOINT_PARENTS` hierarchy — walk up the chain and draw highlighted bones.

Colors: selected joint = accent amber, chain bones = dimmer amber, other bones = default white/gray.

### 4.2 Selection helpers

Right-click joint in viewport → context menu:
- **Select Chain** (root → this joint)
- **Select Siblings** (same parent's children)
- **Select Opposite** (L_Hip ↔ R_Hip, L_Shoulder ↔ R_Shoulder, etc.)
- **Select Body / Left Hand / Right Hand** (region select)

### 4.3 Auto-focus Inspector

When a joint is clicked in the viewport, auto-scroll the Pose Corrector panel to that joint's Euler sliders. Update the joint dropdown to match.

Already partially implemented — `joint_clicked(int)` signal exists on MeshViewport.

---

## Phase 5: Pose Correction UX Upgrades

### 5.1 Preview range slider

Add a range slider below the Euler spinboxes: **±N frames** (default ±15). When adjusting a correction, the viewport auto-plays the range so the artist can see temporal impact. Prevents "pop" artifacts from single-frame edits.

### 5.2 Smoothing controls

Add high-level sliders alongside the raw Euler controls:
- **Drift Reduction** (0–100%) — low-pass filter on selected joint over N frames
- **Temporal Smoothing** (window: 3/5/7/9 frames) — moving average on corrections

These are more intuitive for artists than editing individual rotation channels. Keep raw Euler as an "Advanced" toggle.

### 5.3 Apply to similar

"Apply to All Occurrences" button: find all frames matching the current joint angle + velocity thresholds, apply the same correction. Useful for repeated errors (e.g., "all right knees flex past 150°").

### 5.4 Correction propagation

When a correction is applied at frame N, offer propagation modes:
- **This frame only** (current behavior)
- **Smooth falloff** (cosine blend over ±K frames, like OcclusionBridge)
- **Until next keyframe** (constant correction until next user-annotated keyframe)

---

## Phase 6: Error Heatmap on Skeleton

### 6.1 Joint confidence heatmap

Add a "Heatmap" toggle to the viewport toolbar. When enabled, each joint is colored by its per-frame confidence/error:
- Red = high error (>5cm positional drift or confidence <0.4)
- Yellow = moderate
- Green = clean

Already have `confidence_to_color()` in mesh_viewport.py and the `"confidence"` vertex color mode. Extend to per-joint coloring on the skeleton overlay (not just the mesh surface).

### 6.2 Customizable thresholds

Settings panel: let user define per-joint thresholds (e.g., hip = 3cm, foot = 1cm). Store in session or QSettings.

---

## Phase 7: Viewport Quality Toggle

### 7.1 Preview quality levels

Add a quality button to the viewport toolbar:

| Level | Rendering |
|-------|-----------|
| **Wireframe** | Skeleton only, no mesh, no shading. For fast pose alignment. |
| **Fast** | Simplified mesh (decimated), no shadows. For scrubbing. |
| **Full** | Current Phong-shaded mesh with all features. Default. |

### 7.2 Auto-quality switching

During playback or active scrubbing, auto-switch to Wireframe. Restore to previous quality when paused. This ensures 60fps scrubbing even on complex scenes.

---

## Phase 8: Property Panel Refinement

### 8.1 Tabbed parameter groups

Reorganize the Pose Corrector into collapsible/tabbed sections:
- **Pose** — Euler sliders, quick-fix buttons, smoothing controls
- **Corrections** — corrections table, propagation mode
- **Export** — BVH/FBX export, bone naming convention
- **Space** — coordinate space overrides (world/camera/carried)

### 8.2 Consistent grid layout

All parameter labels left-aligned to a 120px column. All spinboxes same width. All sliders same height. Monospace font for numeric fields. Right-click any parameter → "Edit Value" popup for direct numeric entry.

---

## Phase 9: Session Library

### 9.1 Library panel

New dock widget: a media-pool-style panel (like DaVinci Resolve or Shogun):
- Thumbnail per session (first frame render, 80px)
- Metadata: video name, duration, person count, correction count, date
- Tags: user-assignable labels ("walking", "crossing", "fight scene")
- Version indicator: "v2" if re-exported
- Double-click to load session
- Drag from library to timeline to compare

### 9.2 Session annotations

Free-text notes per session. Stored in session JSON. Displayed in library panel tooltip.

---

## Phase 10: Keyboard & Interaction Polish

### 10.1 Mode stack

Context-aware shortcuts that change by active tool:

| Mode | Key | Action |
|------|-----|--------|
| **Navigate** | W/A/S/D | Orbit camera |
| **Navigate** | G | Go to frame (opens input) |
| **Select** | Click | Pick joint |
| **Select** | Shift+Click | Add to selection |
| **Select** | G | Grab (start correction drag) |
| **Correct** | G | Open Euler sliders for selected joint |
| **Correct** | R | Reset correction on current frame |
| **Track** | G | Go to next unreviewed frame |
| **Track** | Tab | Next person |

Mode indicator in status bar and as a subtle badge on viewport HUD.

### 10.2 On-screen HUD

Semi-transparent overlay on viewport (bottom-right, like Unity Editor):
- Current frame / total
- Playback speed
- Active mode ("Correcting Right Knee")
- Selected person ("Person 0")
- FPS counter

Auto-hide after 2s mouse idle. Toggle via View menu.

---

## Color Palette Reference (already in theme.py)

| Role | Hex | Usage |
|------|-----|-------|
| bg_primary | `#31363b` | Main window, panels |
| bg_input | `#232629` | Inputs, tables, trees |
| bg_active_tab | `#54575B` | Active surfaces |
| text_primary | `#eff0f1` | All text |
| accent | `#ca952e` | Focus, hover, selected tabs, progress bars |
| accent_pressed | `#ab7e28` | Pressed buttons, selection |
| accent_active | `#ba8826` | Active selection |
| hover | `#685126` | Item hover |
| border | `#76797C` | Frame borders, separators |
| success | `#4ecca3` | Good confidence, verified keyframes |
| warning | `#ffd93d` | Moderate confidence, flagged issues |
| error | `#ff6b6b` | Low confidence, failures |

Fonts: LCD-style fonts (LCD-N, LCDMB) available from ShootDay for timecode display if desired. Labels bold by default per Deck Nine convention.

---

## Implementation Priority

| Phase | Impact | Effort | Dependencies |
|-------|--------|--------|-------------|
| 1. Dockable Layout | **Critical** | Medium | None — unblocks everything |
| 2. Track Timeline | **High** | Medium | Phase 1 (needs its own dock) |
| 3. Transport Polish | Medium | Low | Phase 2 |
| 4. Chain Highlighting | Medium | Low | None |
| 5. Correction UX | **High** | Medium | Phase 4 |
| 6. Error Heatmap | Medium | Low | None |
| 7. Quality Toggle | Medium | Low | None |
| 8. Property Panels | Medium | Low | Phase 1 |
| 9. Session Library | Low | Medium | Phase 1 |
| 10. Keyboard Polish | Medium | Low | Phase 1 |

**Recommended order**: 1 → 2 → 4 → 5 → 3 → 6 → 7 → 10 → 8 → 9

Phases 4, 6, and 7 are independent and can be done anytime.

---

## Reference Tools Studied

| Tool | Key Takeaway for bodypipe |
|------|--------------------------|
| Vicon Shogun | Joint error heatmaps, workspace presets, session management |
| MotionBuilder | Chain highlighting on joint select, context-aware shortcuts |
| OptiTrack Motive | Vertical track timeline with color-coded person lanes |
| Xsens MVN | Preview range + auto-smooth on corrections |
| Rokoko Studio | One-click identity reassignment via bbox drag |
| Nuke | Embedded non-modal panels, keyboard-centric transport |
| Mocha Pro | Transport overlay on viewport, real-time smoothing sliders |
| After Effects | Layer panel as navigation hub, speed preset chips |
| Houdini APEX | Hierarchical bone tree, channel editor with numeric entry |
| Fusion | Tabbed inspectors per node, per-viewer quality settings |
| Natron | Open-source PySide6 reference (docks, OpenGL, undo, dark theme) |
| Kdenlive | QGraphicsView-based timeline |
