# bodypipe — Implementation Plan

## Completed Phases (summary)

All 7 phases fully implemented with 1129 tests passing.

- **Phase 1**: Pipeline Tabs — SinglePersonTab, PerfCaptureTab, MultiPersonTab
- **Phase 2**: Identity Inspector — person selector, confidence, keyframes, bbox overlay, two-click bbox editing, track operations (swap/split/merge), review scanner, reprocess
- **Phase 3**: 3D Viewport + Pose Corrector — QOpenGLWidget mesh, camera modes (in-camera/orbit), skeleton overlay + joint picking (screen-space + FBO), pose corrector (euler sliders, quick-fix, corrections table, space overrides, BVH/FBX export)
- **Phase 4**: App Shell Polish — session save/load, undo/redo (50-depth), keyboard shortcuts dialog
- **Phase 5**: Spec Compliance — VideoPlayer context menu, status bar FPS wiring, viewport switching, splitter persistence, multi-person settings parity, 3D viewport color modes/grid/joint labels/FBO picking, FullPipelineWorker stages 3-6, pose corrector auto-detect
- **Phase 6**: Gradio Parity — perf capture settings parity, multi-person FBX batch conversion
- **Phase 7**: Settings Persistence — pipeline config QSettings round-trip, perf capture multi-stage progress, video frame composite, FBO joint picking

## Spec Compliance Fixes

- [x] Session.reset() pipeline config fields: `models/session.py` — reset() now resets pipeline_mode, static_cam, use_dpvo, focal_mm to dataclass defaults. Previously these were missed, allowing stale settings to persist across session resets. Test in `tests/test_models.py` (test_reset_clears_pipeline_config_fields). (1129 total pass).

## Known Minor Gaps (deferred)

- [ ] RenderWorker.finished signal emits `str` instead of `Path` per spec — minimal impact (no consumers use Path), PySide6 Signal(Path) can be unreliable across environments.
