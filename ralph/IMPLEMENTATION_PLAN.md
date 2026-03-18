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

### Commit 1B: Create Dock Widget Wrappers *(DONE)*

- [x] Created `views/dock_widgets.py` with 6 thin QDockWidget subclasses:
  - `VideoDock` wrapping VideoPlayer
  - `MeshViewportDock` wrapping MeshViewport
  - `IdentityDock` wrapping IdentityInspector
  - `PoseCorrectorDock` wrapping PoseCorrectorPanel
  - `TrackOverviewDock` wrapping _TrackOverview (with QScrollArea)
  - `PipelineSettingsDock` wrapping QStackedWidget of 3 settings panels + mode QComboBox
- [x] Each has `setObjectName()` for `saveState()`/`restoreState()` serialization
- [x] Each exposes inner widget via `@property`
- [x] PipelineSettingsDock: mode_changed signal, set_mode(), current_mode/current_settings properties, settings_widget() accessor
- [x] 28 new tests in `tests/test_dock_widgets.py`
- [x] All 1242 tests pass (1214 original + 28 new)

**Files:** new `views/dock_widgets.py`, new `tests/test_dock_widgets.py`

### Commit 1C: Rewire AppWindow from Tabs to Docks *(DONE)*

- [x] Removed QTabWidget from `_setup_ui()` — replaced with dock-based layout
- [x] Created empty central widget (hidden, max size 0×0), `setDockNestingEnabled(True)`
- [x] Instantiated all docks, added to dock areas:
  - Pipeline: left
  - Video: right area (acts as center since central is hidden)
  - Mesh: right area (tabbed with Video)
  - Identity: right (split from Video)
  - PoseCorrector: right (tabbed with Identity)
  - TrackOverview: bottom
  - Log: bottom (tabbed with TrackOverview)
- [x] Mode selector in PipelineSettingsDock switches stacked widget + shows/hides multi-only docks (Identity, PoseCorrector, TrackOverview)
- [x] Moved signal hub from MultiPersonTab to AppWindow (`_setup_signal_hub`):
  - `VideoPlayer.frame_changed` → `_on_video_frame_changed` → broadcasts to IdentityInspector, PoseCorrector, MeshViewport, TrackOverview (multi mode only)
  - `IdentityInspector.person_changed` → MeshViewport, PoseCorrector
  - `MeshViewport.joint_clicked` → PoseCorrector.set_joint
  - `IdentityInspector.frame_requested` → VideoPlayer.seek
  - `PoseCorrector.frame_requested` → VideoPlayer.seek
  - `IdentityInspector.bbox_overlay_changed` → video frame composite
  - `IdentityInspector.track_modified` → TrackOverview refresh
- [x] Moved reprocess logic from MultiPersonTab to AppWindow
- [x] Moved person track loading (`_load_person_tracks_from_result`, `_load_smplx_params`, `_load_confidences_csv`, `_populate_tracks`) from MultiPersonTab to AppWindow
- [x] Updated `closeEvent` worker cleanup to iterate settings panels + reprocess worker
- [x] Updated `_on_open_video` to load into active settings panel (video_loaded signal loads into shared VideoPlayer)
- [x] Added backward-compat properties: `_tab_single`, `_tab_perf`, `_tab_multi` (point to settings widgets)
- [x] Updated View menu: toggle action per dock panel via `toggleViewAction()`
- [x] Rewrote 8 tests in `test_app_window.py`, 6 tests across `test_single_person_tab.py`, `test_perf_capture_tab.py`, `test_multi_person_tab.py`
- [x] Added `_DOCK_VERSION = 1` for `saveState()`/`restoreState()` versioning (old tab-based state ignored)
- [x] Added `mode_changed` Signal(str) on AppWindow; `tab_changed` Signal(int) emitted for backward compat
- [x] All 1282 tests pass (1242 original, 5 new mode/dock tests, some rewritten)

**Files:** `app_window.py` (major rewrite), `tests/test_app_window.py`, `tests/test_single_person_tab.py`, `tests/test_perf_capture_tab.py`, `tests/test_multi_person_tab.py`

### Commit 1D: Workspace Presets *(DONE)*

- [x] Added `View > Workspace` submenu to `app_window.py`:
  - Review: Video + Inspector + Timeline (hides Mesh, PoseCorrector, Log)
  - Correction: Video + 3D + Pose Corrector + Timeline (hides Identity, Log)
  - Tracking: Video + Inspector + Track Overview (hides Mesh, PoseCorrector, Log)
  - Pipeline: Video + Settings + Log (hides Mesh, Identity, PoseCorrector, Track)
- [x] "Save Current Layout..." → prompts for name, saves QByteArray in QSettings (`workspace/state/{name}`)
- [x] "Reset to Default" → restores `_default_state` captured at init
- [x] Custom saved layouts appear in Workspace submenu and can be restored
- [x] `_default_state` QByteArray captured after initial dock setup, before `_restore_geometry()`
- [x] 18 new tests in `tests/test_app_window.py`
- [x] All 1300 tests pass (1282 original + 18 new)

**Files:** `app_window.py` (~85 LOC addition), `tests/test_app_window.py`

### Commit 1E: Remove Old Tab Classes *(DONE)*

- [x] Extracted `TrackOverview` + `PERSON_COLORS` from `multi_person_tab.py` → new `views/track_overview.py`
- [x] Updated imports in `app_window.py` and `views/dock_widgets.py` (`_TrackOverview` → `TrackOverview`)
- [x] Updated `_TAB_CONFIG_MAP` to reference `_single_settings`/`_perf_settings`/`_multi_settings` directly
- [x] Deleted `single_person_tab.py`, `perf_capture_tab.py`, `multi_person_tab.py`
- [x] Rewrote `test_single_person_tab.py` → tests `SinglePipelineSettings` directly
- [x] Rewrote `test_perf_capture_tab.py` → tests `PerfPipelineSettings` directly
- [x] Rewrote `test_multi_person_tab.py` → tests `MultiPipelineSettings` + `TrackOverview` + `AppWindow`
- [x] Removed tab composition tests from `test_pipeline_settings.py`
- [x] Migrated 7 integration test files from `MultiPersonTab` → `AppWindow` (bbox_editing, bbox_overlay, mesh_viewport, pose_corrector, review_scanner, track_operations, dock_widgets)
- [x] Updated `test_app_window.py` — `_tab_single`/`_tab_perf`/`_tab_multi` → `_single_settings`/`_perf_settings`/`_multi_settings`
- [x] All 1227 tests pass (removed ~73 duplicate/tab-specific tests, all unique behavior preserved)

**Files:** deleted 3 tab files, new `views/track_overview.py`, updated `app_window.py`, `views/dock_widgets.py`, 12 test files

### Commit 1F: Remove backward-compat shims *(DONE)*

- [x] Dropped `_tab_single`, `_tab_perf`, `_tab_multi` properties from `app_window.py`
- [x] No test references remained (already cleaned in Commit 1E)
- [x] All 1227 tests pass

---

## Independent Viewport Improvements (can interleave with Phase 1)

### Phase 4: Joint Chain Highlighting *(DONE)*

- [x] On `joint_clicked`, walk `JOINT_PARENTS` from selected joint to root
- [x] Highlight chain bones in accent amber (thicker lines), selected joint brighter
- [x] Right-click context menu: Select Chain, Select Siblings, Select Opposite, Select Region
- [x] Pure helpers: `get_joint_chain`, `get_joint_chain_bones`, `get_joint_siblings`, `get_opposite_joint`, `get_joint_region`
- [x] Constants: `_SELECTED_ACCENT_COLOR`, `_CHAIN_BONE_COLOR`, `_CHAIN_JOINT_COLOR`, `_CHAIN_JOINT_POINT_SIZE`, `_CHAIN_BONE_LINE_WIDTH`
- [x] Data: `_LR_PAIRS`, `_JOINT_REGIONS`, `_JOINT_TO_REGION`
- [x] 39 new tests in `tests/test_mesh_viewport.py`
- [x] All 1281 tests pass (1242 original + 39 new)

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
