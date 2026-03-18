# bodypipe — Implementation Plan

## Completed Phases (summary)

All 7 original phases fully implemented. 1214 tests passing after Commit 1A.

- **Phase 1**: Pipeline Tabs — SinglePersonTab, PerfCaptureTab, MultiPersonTab
- **Phase 2**: Identity Inspector — person selector, confidence, keyframes, bbox overlay, two-click bbox editing, track operations (swap/split/merge), review scanner, reprocess
- **Phase 3**: 3D Viewport + Pose Corrector — QOpenGLWidget mesh, camera modes (in-camera/orbit), skeleton overlay + joint picking (screen-space + FBO), pose corrector (euler sliders, quick-fix, corrections table, space overrides, BVH/FBX export)
- **Phase 4**: App Shell Polish — session save/load, undo/redo (50-depth), keyboard shortcuts dialog
- **Phase 5**: Spec Compliance — VideoPlayer context menu, status bar FPS wiring, viewport switching, splitter persistence, multi-person settings parity, 3D viewport color modes/grid/joint labels/FBO picking, FullPipelineWorker stages 3-6, pose corrector auto-detect
- **Phase 6**: Gradio Parity — perf capture settings parity, multi-person FBX batch conversion
- **Phase 7**: Settings Persistence — pipeline config QSettings round-trip, perf capture multi-stage progress, video frame composite, FBO joint picking

---

# UX Overhaul — Dockable Layout + Viewport Improvements

Full spec: `spec/ux-overhaul.md`

## Architecture Decisions

**Built-in QDockWidget** (not ADS): No ADS pip package for PySide6/WSL2. Built-in supports tabbing (`tabifyDockWidget`), save/restore (`saveState`), all dock areas. Log panel already uses this pattern (`app_window.py:366-376`). Can migrate to ADS later if needed.

**Empty central widget**: All real content in docks. `setDockNestingEnabled(True)`. Maximum workspace flexibility — users can arrange Video and Mesh side-by-side, tabbed, or in different areas. `saveState()`/`restoreState()` handles geometry.

**Mode selector replaces tabs**: 3 standalone settings widgets in a stacked PipelineSettingsDock. Mode combo switches settings + shows/hides multi-only docks. Signal hub moves from MultiPersonTab to AppWindow.

---

## Current Task Queue

### Commit 1A: Extract Pipeline Settings Widgets *(DONE)*

- [x] Created `views/pipeline_settings.py` with 3 classes:
  - `SinglePipelineSettings` — video input, body settings, run/cancel/progress, GVHMRWorker lifecycle
  - `PerfPipelineSettings` — extends Single with hand/face/pipeline settings, FullPipelineWorker, multi-stage progress
  - `MultiPipelineSettings` — pipeline + multi-person settings, MultiPersonWorker lifecycle
- [x] Each class has: `get_config()`, `set_config()`, signals: `status_message`, `log_message`, `pipeline_finished`, `pipeline_error`, `video_loaded`
- [x] Each keeps its run/cancel/progress UI and worker lifecycle
- [x] Tab classes delegate via composition with `__getattr__`/`__setattr__` proxy for backward compat
  - `SinglePersonTab._create_settings()` returns `SinglePipelineSettings`
  - `PerfCaptureTab._create_settings()` returns `PerfPipelineSettings`
  - `MultiPersonTab` composes `MultiPipelineSettings` in sidebar
- [x] 54 new tests in `tests/test_pipeline_settings.py`
- [x] All 1214 tests pass (1160 original + 54 new)

**Architecture note:** `PerfCaptureTab` reduced from 367 LOC to 15 LOC — just a `_create_settings()` override. The `__getattr__` proxy ensures all existing tests pass without modification to assertions (only 2 mock patch targets updated).

### Commit 1B: Create Dock Widget Wrappers *(low risk)*

- [ ] Create `views/dock_widgets.py` with thin QDockWidget subclasses:
  - `VideoDock` wrapping VideoPlayer
  - `MeshViewportDock` wrapping MeshViewport
  - `IdentityDock` wrapping IdentityInspector
  - `PoseCorrectorDock` wrapping PoseCorrectorPanel
  - `TrackOverviewDock` wrapping _TrackOverview
  - `PipelineSettingsDock` wrapping QStackedWidget of 3 settings panels + mode QComboBox
- [ ] Each has `setObjectName()` for `saveState()`/`restoreState()` serialization
- [ ] Each exposes inner widget via `@property`
- [ ] Add ~20 new tests in `tests/test_dock_widgets.py`
- [ ] All existing tests still pass

**Files:** new `views/dock_widgets.py`

### Commit 1C: Rewire AppWindow from Tabs to Docks *(medium risk)*

- [ ] Remove QTabWidget from `_setup_ui()`
- [ ] Create empty central widget (hidden), `setDockNestingEnabled(True)`
- [ ] Instantiate all docks, add to dock areas:
  - Pipeline: left
  - Video: center area
  - Mesh: center area (tabbed with Video)
  - Identity: right
  - PoseCorrector: right (tabbed with Identity)
  - TrackOverview: bottom
  - Log: bottom (tabbed with TrackOverview)
- [ ] Mode selector in PipelineSettingsDock switches stacked widget + shows/hides multi-only docks (Identity, PoseCorrector, TrackOverview)
- [ ] Move signal hub from MultiPersonTab to AppWindow:
  - `VideoPlayer.frame_changed` → IdentityInspector, PoseCorrector, MeshViewport, TrackOverview
  - `IdentityInspector.person_changed` → MeshViewport, PoseCorrector
  - `MeshViewport.joint_clicked` → PoseCorrector.set_joint
  - `IdentityInspector.frame_requested` → VideoPlayer.seek
  - `PoseCorrector.frame_requested` → VideoPlayer.seek
  - `IdentityInspector.bbox_overlay_changed` → video frame composite
  - `IdentityInspector.track_modified` → TrackOverview refresh
- [ ] Update `closeEvent` worker cleanup to iterate dock panels
- [ ] Update `_on_open_video` to load into VideoPlayer directly
- [ ] Add backward-compat properties: `_tab_single`, `_tab_perf`, `_tab_multi` (point to settings widgets)
- [ ] Update View menu: toggle action per dock panel
- [ ] Rewrite ~25 tests in `test_app_window.py` (tab count → mode selector, tab text → dock titles, currentWidget → mode combo)

**Files:** `app_window.py` (major rewrite), `views/dock_widgets.py` (minor)

### Commit 1D: Workspace Presets *(low risk)*

- [ ] Add `View > Workspace` submenu to `app_window.py`:
  - Review: Video large center, Inspector right, Timeline bottom
  - Correction: Video + 3D side-by-side, PoseCorrector right, Timeline bottom
  - Tracking: Video center, Inspector right (expanded), TrackOverview bottom
  - Pipeline: Video center, PipelineSettings left, Log bottom (expanded)
- [ ] "Save Current Layout..." → named QByteArray in QSettings
- [ ] "Reset to Default" → restore hardcoded initial layout
- [ ] Add ~15 new tests

**Files:** `app_window.py` (~80 LOC addition)

### Commit 1E: Remove Old Tab Classes *(cleanup)*

- [ ] Move video loading logic (drag-drop, browse, path handling) to AppWindow or shared mixin
- [ ] Move worker lifecycle management fully into AppWindow
- [ ] Delete or gut `single_person_tab.py`, `perf_capture_tab.py`, `multi_person_tab.py`
- [ ] Relocate ~100 tests to `test_pipeline_settings.py` and `test_app_window.py`
- [ ] Net test count increases

**Files:** delete/gut 3 tab files, update 4 test files

### Commit 1F: Remove backward-compat shims *(cleanup)*

- [ ] Drop `_tab_single`, `_tab_perf`, `_tab_multi` properties
- [ ] Final test cleanup

---

## Independent Viewport Improvements (can interleave with Phase 1)

### Phase 4: Joint Chain Highlighting *(1 commit, ~80 LOC)*

- [ ] On `joint_clicked`, walk `JOINT_PARENTS` from selected joint to root
- [ ] Highlight chain bones in accent amber, selected joint brighter
- [ ] Right-click context menu: Select Chain, Select Siblings, Select Opposite, Select Region

**File:** `views/mesh_viewport.py`

### Phase 6: Error Heatmap on Skeleton *(1 commit, ~50 LOC)*

- [ ] Add `set_skeleton_heatmap(bool)` to MeshViewport public API
- [ ] Per-joint confidence coloring on skeleton overlay (not just mesh surface)
- [ ] Reuse existing `confidence_to_color()` function

**File:** `views/mesh_viewport.py`

### Phase 7: Viewport Quality Toggle *(1 commit, ~60 LOC)*

- [ ] Add render mode enum: Wireframe / Fast / Full
- [ ] Wireframe = skeleton only, no mesh, no shading
- [ ] Auto-switch to Wireframe during active scrubbing, restore on pause

**File:** `views/mesh_viewport.py`

---

## Execution Order

1A → 1B → (Phases 4/6/7 in parallel) → 1C → 1D → 1E → 1F

## Verification (after each commit)

1. `python -m pytest tests/ -x -q` — all tests pass
2. `python main.py` — app launches, dark theme, panels visible
3. Load a video → pipeline settings populate, run button works
4. After 1C: dock panels can be dragged, floated, tabbed, closed/reopened via View menu
5. After 1D: workspace presets restore correct layouts
6. After Phase 4/6/7: 3D viewport shows chain highlights, heatmap toggle, quality modes
