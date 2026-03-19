# bodypipe — Implementation Plan

## Completed Phases (summary)

All 7 original phases + UX overhaul (Commits 1A-1F) + viewport improvements (Phases 2-7) + dock data flow fix + Phase 5 Pose Correction UX + Phase 10 Keyboard & Interaction Polish fully implemented. 1508 tests passing.

- **Phases 1-7 (original)**: Pipeline tabs, identity inspector, 3D viewport + pose corrector, app shell polish, spec compliance, Gradio parity, settings persistence
- **Commits 1A-1F**: Tab-to-dock migration — pipeline settings extraction, dock wrappers, AppWindow rewire, workspace presets, tab class removal, shim cleanup
- **Viewport Phases 2-7**: Joint chain highlighting, skeleton heatmap, quality toggle, DAW-style track timeline, transport overlay + speed chips + adaptive cache
- **Dock data flow fix**: Startup restore, panel hydration, _refresh_all_panels()
- **Phase 5 (Pose Correction UX)**: Preview range, smoothing, apply-to-similar, correction propagation — all with undo support
- **Phase 10 (Keyboard & Interaction Polish)**: InteractionMode enum (Navigate/Select/Correct/Track), context-aware shortcuts, mode indicator, viewport HUD overlay

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

### Phase 6: Error Heatmap on Skeleton *(DONE)*

- [x] Added `set_skeleton_heatmap(bool)` to MeshViewport public API
- [x] Per-joint confidence coloring on skeleton overlay (not just mesh surface)
- [x] Reused existing `confidence_to_color()` function
- [x] Added `_get_frame_confidence()` helper (reads confidence_breakdown["overall"] → raw confidences → 0.5 default)
- [x] Chain highlighting and selected-joint accent take priority over heatmap
- [x] 19 new tests in `tests/test_mesh_viewport.py` (TestSkeletonHeatmapAPI, TestGetFrameConfidence, TestSkeletonHeatmapRendering)
- [x] All 1246 tests pass (1227 original + 19 new)

**File:** `views/mesh_viewport.py`, `tests/test_mesh_viewport.py`

### Phase 7: Viewport Quality Toggle *(DONE)*

- [x] Added `RenderMode` enum (WIREFRAME / FAST / FULL) in `views/mesh_viewport.py`
- [x] `set_render_mode(mode)` — validates, sets mode, triggers refresh
- [x] Wireframe = skeleton only, no mesh, no SMPL-X forward pass — fastest for scrubbing
- [x] Fast = mesh rendered with ambient-only shading (no Phong diffuse)
- [x] Full = default, full Phong shading
- [x] `set_scrubbing(active)` — auto-switch to wireframe, saves/restores previous mode
- [x] `_refresh_mesh()` skips `_compute_vertices()` in wireframe mode (FK joints still computed)
- [x] `paintGL()` skips mesh triangle draw in wireframe, uses ambient-only in fast mode
- [x] Added `scrub_started`/`scrub_ended` signals to `VideoPlayer` (from QSlider pressed/released)
- [x] AppWindow wires `scrub_started`, `scrub_ended`, `playback_toggled` → `set_scrubbing()`
- [x] 29 new tests: `TestRenderModeEnum` (6), `TestSetRenderMode` (10), `TestSetScrubbing` (9), `TestScrubAutoSwitch` (4 in test_app_window.py)
- [x] All 1275 tests pass (1246 original + 29 new)

**Files:** `views/mesh_viewport.py`, `views/video_player.py`, `app_window.py`, `tests/test_mesh_viewport.py`, `tests/test_video_player.py`, `tests/test_app_window.py`

### Phase 2: Vertical Track Timeline *(DONE)*

- [x] Rewrote `views/track_overview.py` — QGraphicsView-based DAW-style timeline replacing simple ConfidenceTimeline stack
- [x] `_TrackLaneItem` (QGraphicsItem): per-person horizontal heatmap lane with:
  - Confidence heatmap (green >0.8, yellow 0.5-0.8, red <0.5) with binning optimization for zoomed-out views
  - Keyframe markers (triangles at lane bottom, green=verified, yellow=unverified)
  - Issue flags (red diamonds near lane top, from review scanner)
  - Correction markers (amber dots at mid-lane)
  - Crossing-span overlays (blue tint for SAM2-detected overlaps)
  - Collapsible lanes via header toggle button
- [x] `_PlayheadItem` (QGraphicsItem): vertical playhead line, ItemIgnoresTransformations for crisp 2px rendering at any zoom
- [x] `_TimelineView` (QGraphicsView): horizontal zoom (scroll wheel, X-axis only), pan (middle-drag), click-to-select person + seek frame
- [x] `_TrackHeader` (QWidget): colored dot + "Person N" label + collapse arrow, fixed-width left column
- [x] `TrackOverview.set_track_markers()` — new API for keyframes, issues, corrections, crossing spans
- [x] Updated `TrackOverviewDock` — removed QScrollArea wrapper (QGraphicsView handles own scrolling)
- [x] Updated `app_window._populate_tracks()` → calls new `_populate_track_markers()` for keyframes, corrections, crossing spans, and review scanner issues
- [x] Marker X-dimensions use inverse device scale for constant screen-pixel size at any zoom level
- [x] 56 new tests in `tests/test_track_overview.py` (TestConfColor, TestTrackLaneItem, TestPlayheadItem, TestTimelineView, TestTrackHeader, TestTrackOverview, TestPersonColors)
- [x] Updated 6 existing tests in `test_multi_person_tab.py` (_timelines→_lanes, _labels→_headers, frame_clicked signal chain)
- [x] Updated 1 test in `test_dock_widgets.py` (QScrollArea→direct widget)
- [x] All 1331 tests pass (1275 original + 56 new)

**Files:** `views/track_overview.py` (rewrite), `views/dock_widgets.py`, `app_window.py`, `tests/test_track_overview.py` (new), `tests/test_multi_person_tab.py`, `tests/test_dock_widgets.py`

### Phase 3: Transport & Scrubbing Polish *(DONE)*

- [x] `_TransportOverlay` widget — semi-transparent floating bar at bottom of video display with rounded rect background (QColor(0,0,0,180))
- [x] Auto-hide: QTimer single-shot 2000ms, pauses on overlay enterEvent, resumes on leaveEvent
- [x] Show on mouse activity: FrameDisplay mouse tracking + eventFilter on VideoPlayer for MouseMove/Enter events
- [x] Overlay contents: back1/play/fwd1 buttons, frame counter (synced with slider row), timecode (MM:SS:FF via frame_to_timecode()), speed chips
- [x] Speed toggle chips: 0.25x | 0.5x | 1x | 2x | 4x as exclusive QButtonGroup in both overlay and track timeline footer
- [x] Bidirectional speed sync: VideoPlayer.speed_changed ↔ TrackOverview.speed_changed, wired in AppWindow._setup_signal_hub(), loop-safe via blockSignals()
- [x] FrameCache adaptive read-ahead: DEFAULT_READAHEAD=15 at rest, PLAYBACK_READAHEAD=30 during playback, switched in _toggle_play()
- [x] Hidden transport buttons: first/back10/fwd10/last remain as keyboard-only QToolButtons for backward compat
- [x] TrackOverview footer: QVBoxLayout wrapping _TimelineView + footer QWidget with speed chips
- [x] 44 new tests: TestTimecode (6), TestTransportOverlay (4), TestSpeedChips (10), TestFrameCacheReadahead (5), TestOverlayLabels (4), TestTrackOverviewSpeedChips (8), TestSpeedSync (3), plus 4 inline tests in TestVideoPlayer
- [x] All 1399 tests pass (1355 original + 44 new)

**Files:** `views/video_player.py` (major rewrite — _TransportOverlay, speed chips, timecode, adaptive cache), `views/track_overview.py` (footer speed chips), `app_window.py` (speed sync wiring), `tests/test_video_player.py`, `tests/test_track_overview.py`, `tests/test_app_window.py`

---

## Execution Order

1A → 1B → 1C → 1D → 1E → 1F → Phase 4 → Phase 6 → Phase 7 → Phase 2 → Phase 3 → Phase 5 → Phase 10

## Current Task Queue

### CRITICAL: Fix dock data flow regressions

The tab→dock migration (commits 1A-1F) broke the runtime data flow. The dock layout works and tests pass, but the live app has these issues:

- [x] **Video Player dock is empty** — Fixed by saving last video path in QSettings and restoring on startup via `_restore_last_video()`. `_on_video_loaded` now also triggers `_try_restore_results()` to load cached pipeline output.
- [x] **3D Mesh dock is empty** — Fixed by `_refresh_all_panels()` which calls `set_person()` and `on_frame_changed()` on MeshViewport after results are loaded.
- [x] **Identity Inspector dock not populated** — Fixed by `_refresh_all_panels()` calling `refresh()`, `set_person()`, `set_frame()` on the inspector.
- [x] **Pose Corrector dock not populated** — Fixed by `_refresh_all_panels()` calling `set_person()` and `on_frame_changed()` on PoseCorrectorPanel.
- [x] **Track Overview not synced** — Fixed by `_refresh_all_panels()` calling `_populate_tracks()` and `set_current_frame()`. Track click signal hub was already wired correctly.
- [x] **Pipeline results not loading on startup** — Fixed by `_try_restore_results()` which checks for existing output dirs and loads person tracks from disk via `_load_results_from_output_dir()`, or hydrates existing session tracks via `_hydrate_person_tracks()`.
- [x] **Workspace presets should show correct panels** — Fixed as a side effect — the panels are now populated before workspace presets are applied.
- [x] **Missing stub methods** — `_on_smooth`, `_on_apply_to_similar`, `_on_propagate` in PoseCorrectorPanel were fully implemented in Phase 5 (preview range, smoothing, apply-to-similar, correction propagation).

**Test verification:** After fixes, `python main.py` with a previously-processed video should show: video frames in Video dock with working transport controls, 3D mesh in Mesh dock, populated Identity Inspector, populated Pose Corrector, and synced Track Timeline.

**Architecture:** Added 5 new methods to AppWindow:
- `_restore_last_video()` — startup video path restore from QSettings
- `_try_restore_results(video_path)` — detects cached output and routes to loader
- `_load_results_from_output_dir(output_dir)` — scans person_N dirs, creates PersonTrack objects
- `_hydrate_person_tracks()` — loads smplx_params + confidences from disk for existing tracks
- `_refresh_all_panels()` — centralised "populate everything" (replaces ad-hoc calls in _on_multi_pipeline_finished, etc.)
24 new tests in tests/test_app_window.py. All 1355 tests pass.

### Phase 5: Pose Correction UX Upgrades *(DONE)*

- [x] **Preview range slider** — QSpinBox ±N frames (default ±15) with Preview button + auto-play QTimer. "Auto-preview after Apply" checkbox triggers preview after Apply and Quick Fix operations.
- [x] **Smoothing controls** — `_on_smooth()` implemented: uses `smooth_joint_rotations()` (gaussian/moving average) over frame range. Supports "Current Joint" or "All Body Joints" scope. Undo support via raw body_pose snapshot/restore.
- [x] **Apply to Similar** — `_on_apply_to_similar()` implemented: uses `find_similar_frames()` with configurable angular threshold. Applies current euler slider correction to all matching frames. Status label shows match count. Undo support.
- [x] **Correction propagation** — `_on_propagate()` implemented: uses `propagate_corrections()` for SLERP interpolation between start/end frames in the Frame Range. Works for both body joints and global_orient. Undo support.

**Files:** `views/pose_corrector_panel.py` (replaced 3 stub methods with full implementations, added preview range UI), `tests/test_pose_corrector.py` (56 new tests)

### Phase 10: Keyboard & Interaction Polish *(DONE)*

- [x] **InteractionMode enum** — 4 modes (Navigate, Select, Correct, Track) with `InteractionMode` enum in `app_window.py`
- [x] **Context-aware shortcuts** that change by active mode:
  - Mode switching: 1=Navigate, 2=Select, 3=Correct, 4=Track
  - Navigate: WASD orbit camera nudge (auto-switches to orbit mode), G go-to-frame dialog
  - Select: Click pick joint, G go-to-frame, Escape deselect
  - Correct: G opens/raises Pose Corrector dock, R resets current joint rotation, Escape deselect
  - Track: G next unreviewed keyframe (via review scanner), Tab/Shift+Tab cycle persons
- [x] **Mode indicator in status bar** — Color-coded pill label showing active mode name and number (amber=Navigate, blue=Select, red=Correct, green=Track)
- [x] **On-screen HUD overlay on viewport** (bottom-right): `_ViewportHUD` widget showing current frame, playback speed, active mode, selected person. Auto-hide after 2s idle. Mouse movement over viewport triggers show. Toggle via View > Toggle HUD Overlay (Ctrl+H).
- [x] **Public API additions**: `PoseCorrectorPanel.reset_current_joint()`, `IdentityInspector.go_to_next_unreviewed()`
- [x] **Keyboard shortcuts dialog updated** — Added Mode, Navigate, Select, Correct, Track categories + Ctrl+H for HUD toggle
- [x] 53 new tests: TestInteractionModeEnum (2), TestInteractionModeManager (7), TestModeKeyboardShortcuts (4), TestNavigateModeKeys (2), TestCorrectModeKeys (2), TestTrackModeKeys (4), TestViewportHUD (11), TestKeyboardShortcutsDialog (5), TestGoToFrameDialog (2), TestViewportHUDWidget (14 in test_mesh_viewport.py)
- [x] All 1508 tests pass (1455 original + 53 new)

**Files:** `app_window.py` (InteractionMode enum, mode manager, keyboard routing, mode indicator, HUD toggle), `views/mesh_viewport.py` (_ViewportHUD overlay, HUD API methods, mouse tracking), `views/keyboard_shortcuts_dialog.py` (mode-specific shortcuts), `views/pose_corrector_panel.py` (reset_current_joint), `views/identity_inspector.py` (go_to_next_unreviewed), `tests/test_app_window.py`, `tests/test_mesh_viewport.py`

### Phase 8: Property Panel Refinement

- [ ] Reorganize PoseCorrectorPanel into collapsible/tabbed sections: Pose, Corrections, Export, Space.
- [ ] Consistent grid layout: labels 120px column, same-width spinboxes, same-height sliders, monospace numeric fields.

### Phase 9: Session Library

- [ ] New dock widget: media-pool-style panel. Thumbnail per session (first frame, 80px), metadata (video name, duration, person count, correction count, date), user-assignable tags, version indicator, double-click to load.
- [ ] Session annotations: free-text notes stored in session JSON, displayed in tooltip.

## Verification

1. `QT_QPA_PLATFORM=offscreen python -m pytest tests/ -x -q` — all 1508 tests pass
2. `python main.py` with previously-processed video → video frames visible, transport works, 3D mesh renders, inspector populated, timeline synced
3. Dock panels can be dragged, floated, tabbed, closed/reopened via View menu
4. Workspace presets restore correct layouts with populated panels
5. 3D viewport: chain highlights, heatmap toggle, quality modes (wireframe/fast/full)
6. Full end-to-end: load video → run pipeline → results populate all panels → corrections workflow functional
7. Track timeline: zoom (scroll wheel), pan (middle-drag), click to select person + seek, collapsible lanes, marker overlays
8. Transport overlay: auto-hide after 2s, speed chips sync between video player and track timeline, adaptive read-ahead during playback
9. Keyboard interaction: mode switching (1-4), context-aware shortcuts per mode, status bar mode indicator, viewport HUD overlay (Ctrl+H toggle)
