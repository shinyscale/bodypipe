# bodypipe — Implementation Plan

## Phase 1: Foundation (get a window running)

### Task 1: App shell + dark theme
- **Phase**: 1
- **Files**: `main.py`, `app_window.py`
- **Read first**: `gvhmr_gui.py:1300-1448` (tab structure, theme)
- **Depends on**: —
- **Acceptance**: `QT_QPA_PLATFORM=offscreen python main.py --smoke-test` exits cleanly
- **Notes**: Already scaffolded. Verify theme, menus, log panel, geometry persistence.
- **Status**: [x] done

### Task 2: Session + PipelineConfig data models
- **Phase**: 1
- **Files**: `models/session.py`, `models/pipeline_config.py`
- **Read first**: `identity_panel.py` (`_SESSION_DATA`), `pose_correction_panel.py` (`_POSE_SESSION`), `gvhmr_gui.py` (`save_solve_config`)
- **Depends on**: —
- **Acceptance**: `pytest tests/test_models.py -x -q` passes
- **Notes**: Already scaffolded. JSON round-trip, reset, from_dict edge cases.
- **Status**: [x] done

### Task 3: Video player widget
- **Phase**: 1
- **Files**: `views/video_player.py`
- **Read first**: `identity_panel.py` (`_LRUFrameCache`, `_extract_frame`, transport button callbacks)
- **Depends on**: Task 1
- **Acceptance**: Unit test: construct widget, set_video, seek, verify frame_changed signal fires
- **Notes**: Already scaffolded. Add test for FrameCache and VideoPlayer signal emission.
- **Status**: [x] done

### Task 4: Confidence timeline widget
- **Phase**: 1
- **Files**: `views/confidence_timeline.py`
- **Read first**: `identity_panel.py` (`_render_confidence_timeline`), `identity_confidence.py` (`TrackConfidence`)
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
- **Read first**: `gvhmr_gui.py` (`_run_gvhmr_subprocess`, `run_gvhmr`, progress regex patterns)
- **Depends on**: Task 2
- **Acceptance**: Unit test: mock subprocess, verify progress/log_line/finished signals
- **Notes**: Already scaffolded. Needs tests with mocked subprocess.
- **Status**: [ ] todo

### Task 7: Reprocess + render workers
- **Phase**: 2
- **Files**: `workers/reprocess_worker.py`, `workers/render_worker.py`
- **Read first**: `multi_person_split.py` (`reprocess_person`, `render_multi_person_incam`)
- **Depends on**: Task 2
- **Acceptance**: Unit test: construct workers, verify signal declarations
- **Notes**: Already scaffolded. Full integration test requires GVHMR backend.
- **Status**: [ ] todo

### Task 8: Single-person tab
- **Phase**: 2
- **Files**: `views/pipeline_tabs.py` (SinglePersonTab class), `app_window.py`
- **Read first**: `gvhmr_gui.py:1300-1360` (Tab 1 layout, checkboxes, run button), `gvhmr_gui.py:run_gvhmr()`, `gvhmr_gui.py:find_output_dir()`
- **Depends on**: Tasks 3, 6
- **Acceptance**: Tab renders, video input works, run button wires to GVHMRWorker, smoke test still passes
- **Notes**: Replace placeholder in app_window.py tab 1.
- **Status**: [ ] todo

### Task 9: Performance capture tab
- **Phase**: 2
- **Files**: `views/pipeline_tabs.py` (PerfCaptureTab class), `app_window.py`
- **Read first**: `gvhmr_gui.py:1360-1400` (Tab 2 layout), `gvhmr_gui.py:run_full_pipeline()`, `gvhmr_gui.py:_run_smplestx_subprocess()`
- **Depends on**: Task 8
- **Acceptance**: Tab renders with hand/face toggle groups, multi-stage progress display, smoke test passes
- **Notes**: Replace placeholder in app_window.py tab 2.
- **Status**: [ ] todo

### Task 10: Multi-person tab shell
- **Phase**: 2
- **Files**: `views/pipeline_tabs.py` (MultiPersonTab class), `app_window.py`
- **Read first**: `gvhmr_gui.py:1400-1448` (Tab 3 layout), `gvhmr_gui.py:run_multi_person_pipeline()`, `identity_panel.py:build_identity_panel()` (component list)
- **Depends on**: Tasks 3, 7
- **Acceptance**: Tab renders with splitter layout, pipeline run wires up, placeholder panels for identity/pose, smoke test passes
- **Notes**: Placeholder panels for identity inspector + pose corrector. Signal hub wiring.
- **Status**: [ ] todo

---

## Phase 3: HITL Editing (the hard part)

### Task 11: Identity inspector — person selector + keyframe table
- **Phase**: 3
- **Files**: `views/identity_panel.py`
- **Read first**: `identity_panel.py` (`_person_choices`, `_keyframe_dataframe`, `on_person_change`, `on_add_keyframe`, `on_remove_keyframe`, `on_verify`, `on_prev_keyframe`, `on_next_keyframe`), `identity_tracking.py` (`IdentityTrack`, `IdentityKeyframe`)
- **Depends on**: Tasks 3, 4, 10
- **Acceptance**: Person dropdown populates from session, keyframe table CRUD works, confidence breakdown displays
- **Notes**: First slice of identity inspector. No bbox editing yet.
- **Status**: [ ] todo

### Task 12: Identity inspector — bbox overlay + editing
- **Phase**: 3
- **Files**: `views/identity_panel.py`, `views/bbox_editor.py`
- **Read first**: `identity_panel.py` (`_render_frame_with_bboxes`, `on_frame_click`, `_save_bbox_corrections`, `_load_bbox_corrections`, `on_interpolate_bboxes`, `on_apply_keyframes`)
- **Depends on**: Task 11
- **Acceptance**: Two-click bbox editing on video frame, dashed overlay renders, interpolation between keyframes
- **Notes**: Frame click → set corners → dashed overlay. Needs coordinate mapping from normalized VideoPlayer coords to pixel coords.
- **Status**: [ ] todo

### Task 13: Identity inspector — track operations + review scanner
- **Phase**: 3
- **Files**: `views/identity_panel.py`
- **Read first**: `identity_panel.py` (`on_split_track`, `on_merge_track`, `on_swap_ids`, `on_crossing_start`, `on_crossing_end`, `on_scan_issues`, `on_next_issue`, `on_prev_issue`), `identity_bridge.py` (`crossing_spans_from_signal`)
- **Depends on**: Task 11
- **Acceptance**: Split/merge tracks, mark crossing spans, scan finds issues and navigates
- **Notes**: Crossing span table, issue scanner with prev/next navigation.
- **Status**: [ ] todo

### Task 14: Identity inspector — reprocess integration
- **Phase**: 3
- **Files**: `views/identity_panel.py`
- **Read first**: `identity_panel.py` (`on_reprocess_all_dirty`), `multi_person_split.py` (`reprocess_person`, `interpolate_bbox_corrections`)
- **Depends on**: Tasks 7, 13
- **Acceptance**: Dirty persons tracked, reprocess button triggers ReprocessWorker, results reload into session
- **Notes**: Wire dirty_persons set in Session, enable/disable reprocess button.
- **Status**: [ ] todo

### Task 15: 3D viewport — basic mesh rendering
- **Phase**: 3
- **Files**: `views/mesh_viewport.py`, `shaders/mesh.vert`, `shaders/mesh.frag`
- **Read first**: `multi_person_split.py:render_multi_person_incam()` (SMPL-X vertex computation, camera transforms), `visualize_skeleton.py` (`forward_kinematics`, `JOINT_PARENTS`, `DEFAULT_OFFSETS`)
- **Depends on**: Task 1
- **Acceptance**: QOpenGLWidget renders a static SMPL mesh with Phong shading
- **Notes**: Load SMPL-X body model, compute vertices, upload to VBO, render with shaders. May need PyOpenGL install.
- **Status**: [ ] todo

### Task 16: 3D viewport — camera modes + frame scrubbing
- **Phase**: 3
- **Files**: `views/mesh_viewport.py`
- **Read first**: `visualize_skeleton.py` (`_get_projection_intrinsics`, `project_to_2d`), `pose_correction_panel.py` (`_camera_space_params`)
- **Depends on**: Task 15
- **Acceptance**: In-camera mode matches video K, orbit mode has mouse controls, scrubbing updates mesh
- **Notes**: Per-frame vertex update from params. Cache vertices if params unchanged.
- **Status**: [ ] todo

### Task 17: 3D viewport — skeleton overlay + joint picking
- **Phase**: 3
- **Files**: `views/mesh_viewport.py`
- **Read first**: `visualize_skeleton.py` (`forward_kinematics`, `BONE_CONNECTIONS`, `JOINT_NAMES`), `pose_correction.py` (`compute_skeleton_frame`, `find_nearest_joint`)
- **Depends on**: Task 16
- **Acceptance**: Skeleton drawn as GL_LINES, joints clickable, joint_clicked signal fires with correct index
- **Notes**: Screen-space distance picking first, color-pass picking as optimization later.
- **Status**: [ ] todo

### Task 18: Pose corrector — joint controls + euler sliders
- **Phase**: 3
- **Files**: `views/pose_corrector.py`
- **Read first**: `pose_correction_panel.py` (`on_joint_dropdown_change`, `on_euler_change`, `on_apply_correction`, `_get_joint_euler`), `pose_correction.py` (`axis_angle_to_euler_deg`, `euler_deg_to_axis_angle`)
- **Depends on**: Tasks 15-17
- **Acceptance**: Joint selection updates sliders, slider changes preview mesh in real-time, apply commits to CorrectionTrack
- **Notes**: Euler↔axis-angle conversion. Preview = temporary param modification + viewport update.
- **Status**: [ ] todo

### Task 19: Pose corrector — quick-fix buttons + corrections table
- **Phase**: 3
- **Files**: `views/pose_corrector.py`
- **Read first**: `pose_correction_panel.py` (`on_flip_whole_body`, `on_invert_upright`, `on_mirror_lr`, `on_copy_from_frame`, `_corrections_dataframe`, `on_delete_correction`), `pose_correction.py` (`flip_global_orient`, `mirror_lr_pose`, `copy_pose_from_frame`)
- **Depends on**: Task 18
- **Acceptance**: Flip/invert/mirror/copy work, corrections table shows all keyframes with Go/Delete actions
- **Notes**: Wrap pose_correction.py functions. Table with Go/Delete actions.
- **Status**: [ ] todo

### Task 20: Pose corrector — space overrides + BVH/FBX export
- **Phase**: 3
- **Files**: `views/pose_corrector.py`
- **Read first**: `pose_correction_panel.py` (`on_add_space_override`, `on_delete_space_override`, `on_reexport_bvh`, `on_reexport_fbx`), `pose_correction.py` (`FrameSpaceOverride`, `CorrectionTrack`), `smplx_to_bvh.py` (`convert_smplx_to_bvh`)
- **Depends on**: Task 19
- **Acceptance**: Space override table CRUD, re-export BVH/FBX applies all corrections
- **Notes**: Wire to smplx_to_bvh.convert_smplx_to_bvh() with correction track.
- **Status**: [ ] todo

### Task 21: Full integration test
- **Phase**: 3
- **Files**: `tests/test_integration.py`
- **Read first**: All GUI files — verify behavior matches
- **Depends on**: Tasks 14, 20
- **Acceptance**: Load fixture session, verify all panels wire together, export works
- **Notes**: May need fixture data (small .pt file, identity tracks).
- **Status**: [ ] todo
