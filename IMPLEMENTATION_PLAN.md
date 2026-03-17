# bodypipe — Implementation Plan

## Phase 1: Foundation (get a window running)

### Task 1: App shell + dark theme
- **Phase**: 1
- **Files**: `main.py`, `app_window.py`
- **Depends on**: —
- **Acceptance**: `QT_QPA_PLATFORM=offscreen python main.py --smoke-test` exits cleanly
- **Notes**: Already scaffolded. Verify theme, menus, log panel, geometry persistence.
- **Status**: [x] done

### Task 2: Session + PipelineConfig data models
- **Phase**: 1
- **Files**: `models/session.py`, `models/pipeline_config.py`
- **Depends on**: —
- **Acceptance**: `pytest tests/test_models.py -x -q` passes
- **Notes**: Already scaffolded. JSON round-trip, reset, from_dict edge cases.
- **Status**: [x] done

### Task 3: Video player widget
- **Phase**: 1
- **Files**: `views/video_player.py`
- **Depends on**: Task 1
- **Acceptance**: Unit test: construct widget, set_video, seek, verify frame_changed signal fires
- **Notes**: Already scaffolded. Add test for FrameCache and VideoPlayer signal emission.
- **Status**: [x] done

### Task 4: Confidence timeline widget
- **Phase**: 1
- **Files**: `views/confidence_timeline.py`
- **Depends on**: Task 1
- **Acceptance**: Unit test: construct, set_data, verify paintEvent doesn't crash
- **Notes**: Already scaffolded. QPainter-based, no matplotlib dependency.
- **Status**: [x] done

### Task 5: Smoke test end-to-end
- **Phase**: 1
- **Files**: `tests/test_smoke.py`
- **Depends on**: Tasks 1-4
- **Acceptance**: `QT_QPA_PLATFORM=offscreen python main.py --smoke-test` and `pytest tests/ -x -q` both pass
- **Notes**: Write test_smoke.py that creates AppWindow, verifies tabs, log panel.
- **Status**: [ ] todo

---

## Phase 2: Pipeline Execution

### Task 6: GVHMR worker
- **Phase**: 2
- **Files**: `workers/gvhmr_worker.py`
- **Depends on**: Task 2
- **Acceptance**: Unit test: mock subprocess, verify progress/log_line/finished signals
- **Notes**: Already scaffolded. Needs tests with mocked subprocess.
- **Status**: [ ] todo

### Task 7: Reprocess + render workers
- **Phase**: 2
- **Files**: `workers/reprocess_worker.py`, `workers/render_worker.py`
- **Depends on**: Task 2
- **Acceptance**: Unit test: construct workers, verify signal declarations
- **Notes**: Already scaffolded. Full integration test requires GVHMR backend.
- **Status**: [ ] todo

### Task 8: Single-person tab
- **Phase**: 2
- **Files**: `views/pipeline_tabs.py` (SinglePersonTab class)
- **Depends on**: Tasks 3, 6
- **Acceptance**: Tab renders, video input works, run button wires to GVHMRWorker
- **Notes**: Replace placeholder in app_window.py tab 1.
- **Status**: [ ] todo

### Task 9: Performance capture tab
- **Phase**: 2
- **Files**: `views/pipeline_tabs.py` (PerfCaptureTab class)
- **Depends on**: Task 8
- **Acceptance**: Tab renders with hand/face toggle groups, multi-stage progress display
- **Notes**: Replace placeholder in app_window.py tab 2.
- **Status**: [ ] todo

### Task 10: Multi-person tab shell
- **Phase**: 2
- **Files**: `views/pipeline_tabs.py` (MultiPersonTab class)
- **Depends on**: Tasks 3, 7
- **Acceptance**: Tab renders with splitter layout, pipeline run works, track overview shows
- **Notes**: Placeholder panels for identity inspector + pose corrector. Signal hub wiring.
- **Status**: [ ] todo

---

## Phase 3: HITL Editing (the hard part)

### Task 11: Identity inspector — person selector + keyframe table
- **Phase**: 3
- **Files**: `views/identity_panel.py`
- **Depends on**: Tasks 3, 4, 10
- **Acceptance**: Person dropdown populates from session, keyframe table CRUD works
- **Notes**: First slice of identity inspector. No bbox editing yet.
- **Status**: [ ] todo

### Task 12: Identity inspector — bbox overlay + editing
- **Phase**: 3
- **Files**: `views/identity_panel.py`, `views/bbox_editor.py`
- **Depends on**: Task 11
- **Acceptance**: Two-click bbox editing on video frame, interpolation between keyframes
- **Notes**: Frame click → set corners → dashed overlay. Needs coordinate mapping.
- **Status**: [ ] todo

### Task 13: Identity inspector — track operations + review scanner
- **Phase**: 3
- **Files**: `views/identity_panel.py`
- **Depends on**: Task 11
- **Acceptance**: Split/merge tracks, mark crossing spans, scan finds issues and navigates
- **Notes**: Crossing span table, issue scanner with prev/next navigation.
- **Status**: [ ] todo

### Task 14: Identity inspector — reprocess integration
- **Phase**: 3
- **Files**: `views/identity_panel.py`
- **Depends on**: Tasks 7, 13
- **Acceptance**: Dirty persons tracked, reprocess button triggers ReprocessWorker
- **Notes**: Wire dirty_persons set in Session, enable/disable reprocess button.
- **Status**: [ ] todo

### Task 15: 3D viewport — basic mesh rendering
- **Phase**: 3
- **Files**: `views/mesh_viewport.py`, `shaders/mesh.vert`, `shaders/mesh.frag`
- **Depends on**: Task 1
- **Acceptance**: QOpenGLWidget renders a static SMPL mesh with Phong shading
- **Notes**: Load SMPL-X body model, compute vertices, upload to VBO, render with shaders.
- **Status**: [ ] todo

### Task 16: 3D viewport — camera modes + frame scrubbing
- **Phase**: 3
- **Files**: `views/mesh_viewport.py`
- **Depends on**: Task 15
- **Acceptance**: In-camera mode matches video K, orbit mode has mouse controls, scrubbing updates mesh
- **Notes**: Per-frame vertex update from params. Cache vertices if params unchanged.
- **Status**: [ ] todo

### Task 17: 3D viewport — skeleton overlay + joint picking
- **Phase**: 3
- **Files**: `views/mesh_viewport.py`
- **Depends on**: Task 16
- **Acceptance**: Skeleton drawn as GL_LINES, joints clickable, joint_clicked signal fires
- **Notes**: Screen-space distance picking first, color-pass picking as optimization later.
- **Status**: [ ] todo

### Task 18: Pose corrector — joint controls + euler sliders
- **Phase**: 3
- **Files**: `views/pose_corrector.py`
- **Depends on**: Tasks 15-17
- **Acceptance**: Joint selection updates sliders, slider changes preview in real-time
- **Notes**: Euler↔axis-angle conversion. Preview = temporary param modification + viewport update.
- **Status**: [ ] todo

### Task 19: Pose corrector — quick-fix buttons + corrections table
- **Phase**: 3
- **Files**: `views/pose_corrector.py`
- **Depends on**: Task 18
- **Acceptance**: Flip/invert/mirror/copy work, corrections table shows all keyframes
- **Notes**: Wrap pose_correction.py functions. Table with Go/Delete actions.
- **Status**: [ ] todo

### Task 20: Pose corrector — space overrides + BVH/FBX export
- **Phase**: 3
- **Files**: `views/pose_corrector.py`
- **Depends on**: Task 19
- **Acceptance**: Space override table CRUD, re-export BVH/FBX applies all corrections
- **Notes**: Wire to smplx_to_bvh.convert_smplx_to_bvh() with correction track.
- **Status**: [ ] todo

### Task 21: Full integration test
- **Phase**: 3
- **Files**: `tests/test_integration.py`
- **Depends on**: Tasks 14, 20
- **Acceptance**: Load fixture session, verify all panels wire together, export works
- **Notes**: May need fixture data (small .pt file, identity tracks).
- **Status**: [ ] todo
