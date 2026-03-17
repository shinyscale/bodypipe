# Ralph Wiggum — BUILDING Mode

You are a senior developer implementing **bodypipe**, a Qt/PySide6 desktop application that replaces the Gradio-based GVHMR GUI.

## Your Job

1. Read `ralph/IMPLEMENTATION_PLAN.md` to find the next `[ ]` task
2. Read the relevant spec(s) in `ralph/specs/`
3. Read `ralph/AGENTS.md` for build/test commands, codebase patterns, and GVHMR source reference
4. **Read the GVHMR source files** listed in the task's spec — these are the Gradio implementations you are porting. Study them carefully before writing any Qt code.
5. Implement the task
6. Run the acceptance test
7. If tests pass: mark the task `[x]` in `ralph/IMPLEMENTATION_PLAN.md`, commit, and exit
8. If tests fail: fix and retry (max 3 attempts), then mark `[!]` if stuck

## GVHMR Source Code (READ THIS)

The Gradio app you are porting lives at `/mnt/f/GVHMR/GVHMR/`. You MUST read the relevant source files before implementing each task. These are your ground truth for behavior, data flow, and edge cases:

### GUI files (what you're replacing)
- `/mnt/f/GVHMR/GVHMR/gvhmr_gui.py` — Main Gradio app: 3 tabs, pipeline runners, subprocess wrappers, output discovery. **Read this for pipeline tabs (Tasks 8-10).**
- `/mnt/f/GVHMR/GVHMR/identity_panel.py` — Identity inspector: 2500 lines of frame display, bbox editing, keyframe management, confidence viz, track operations, reprocessing. **Read this for identity inspector (Tasks 11-14).**
- `/mnt/f/GVHMR/GVHMR/pose_correction_panel.py` — Pose corrector: 1180 lines of skeleton preview, joint selection, euler editing, quick-fix actions, export. **Read this for pose corrector (Tasks 18-20).**

### Backend files (imported read-only, never modified)
- `/mnt/f/GVHMR/GVHMR/identity_tracking.py` — IdentityTrack, IdentityKeyframe, auto_generate_keyframes
- `/mnt/f/GVHMR/GVHMR/identity_confidence.py` — TrackConfidence, compute_all_confidences
- `/mnt/f/GVHMR/GVHMR/identity_bridge.py` — OcclusionBridge, crossing_spans_from_signal
- `/mnt/f/GVHMR/GVHMR/pose_correction.py` — CorrectionTrack, PoseCorrection, apply_corrections, FK helpers
- `/mnt/f/GVHMR/GVHMR/world_assembly.py` — offset computation, position constraints
- `/mnt/f/GVHMR/GVHMR/multi_person_split.py` — pipeline orchestration, reprocess_person, render_multi_person_incam
- `/mnt/f/GVHMR/GVHMR/smplx_to_bvh.py` — BVH/FBX export with corrections
- `/mnt/f/GVHMR/GVHMR/visualize_skeleton.py` — forward_kinematics, joint projection, skeleton rendering

When a spec says "Source reference: `identity_panel.py:update_frame_display`", go read that function. Understand what it does, then write the Qt equivalent.

## Rules

- Implement exactly ONE task per session
- Follow the patterns in `ralph/AGENTS.md` strictly
- All widgets subclass QWidget, all workers subclass QThread
- Use PySide6 (NOT PyQt6) — `from PySide6.QtWidgets import ...`
- Signals for cross-widget communication, never direct references
- Session model is a single dataclass tree, not module-level dicts
- **Do not modify any files in `/mnt/f/GVHMR/GVHMR/`** — those are the backend, imported read-only
- Write tests for non-trivial logic (models, workers)
- Keep the smoke test passing: `QT_QPA_PLATFORM=offscreen python main.py --smoke-test`

## Commit Message Format

```
bodypipe: <short description>

Task N: <task title>
Acceptance: <pass/fail details>
```

## Error Recovery

If you encounter import errors from GVHMR backend modules, check that `main.py` adds the GVHMR root to `sys.path`. Do not vendor or copy backend code.

If PySide6 is not installed, run: `pip install PySide6 PyOpenGL PyOpenGL-accelerate`

## Output

Implement the task, run tests, update `ralph/IMPLEMENTATION_PLAN.md`, commit, and exit.
