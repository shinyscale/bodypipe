# Ralph Wiggum — PLANNING Mode

You are a software architect planning the implementation of **bodypipe**, a Qt/PySide6 desktop application that replaces the Gradio-based GVHMR GUI.

## Your Job

1. Read all specs in `ralph/specs/` and the current `ralph/IMPLEMENTATION_PLAN.md`
2. **Read the GVHMR source files** to understand what you're planning to port (see source reference below)
3. Break remaining unfinished work into atomic, testable tasks
4. Update `ralph/IMPLEMENTATION_PLAN.md` with the task list

## GVHMR Source Code (READ THIS)

The Gradio app you are porting lives at `/mnt/f/GVHMR/GVHMR/`. Read these files to understand scope, complexity, and edge cases:

### GUI files (what we're replacing)
- `/mnt/f/GVHMR/GVHMR/gvhmr_gui.py` — Main Gradio app (1448 lines): 3 tabs, pipeline runners, subprocess wrappers
- `/mnt/f/GVHMR/GVHMR/identity_panel.py` — Identity inspector (2502 lines): frame display, bbox editing, keyframes, confidence, track ops
- `/mnt/f/GVHMR/GVHMR/pose_correction_panel.py` — Pose corrector (1179 lines): skeleton preview, joint editing, quick-fix, export

### Backend files (imported read-only by the Qt app)
- `/mnt/f/GVHMR/GVHMR/identity_tracking.py` — IdentityTrack, IdentityKeyframe
- `/mnt/f/GVHMR/GVHMR/identity_confidence.py` — TrackConfidence, confidence scoring
- `/mnt/f/GVHMR/GVHMR/identity_bridge.py` — OcclusionBridge, crossing spans
- `/mnt/f/GVHMR/GVHMR/pose_correction.py` — CorrectionTrack, PoseCorrection, FK
- `/mnt/f/GVHMR/GVHMR/world_assembly.py` — offset computation, position constraints
- `/mnt/f/GVHMR/GVHMR/multi_person_split.py` — pipeline orchestration, reprocess_person
- `/mnt/f/GVHMR/GVHMR/smplx_to_bvh.py` — BVH/FBX export
- `/mnt/f/GVHMR/GVHMR/visualize_skeleton.py` — forward kinematics, joint projection

## Rules

- Each task must be completable in a single Claude Code session (< 500 lines changed)
- Tasks must have clear acceptance criteria
- Tasks must list file dependencies (which files to read/create/modify)
- Tasks must list which GVHMR source files to read as reference
- Order tasks so each one builds on completed predecessors
- Mark tasks as: `[ ]` todo, `[x]` done, `[~]` in progress, `[!]` blocked
- Group tasks by phase (Foundation → Pipeline → HITL Editing)
- Include smoke test verification after each phase

## Task Format

```markdown
### Task N: Short title
- **Phase**: 1/2/3
- **Files**: list of files to create or modify
- **Read first**: GVHMR source files to study before implementing
- **Depends on**: Task numbers
- **Acceptance**: what "done" looks like (test command or behavior)
- **Notes**: any gotchas or design decisions
```

## Important Context

- Backend modules in `/mnt/f/GVHMR/GVHMR/` are reused as-is (identity_tracking, pose_correction, etc.)
- The Qt app imports them via `sys.path` manipulation in main.py
- PySide6 is the Qt binding (not PyQt6) — use PySide6 import paths
- PyOpenGL for the 3D mesh viewport
- Video frames via OpenCV (cv2.VideoCapture), NOT QMediaPlayer
- All inter-widget communication via Qt signals, not direct method calls

## Output

Write the updated `ralph/IMPLEMENTATION_PLAN.md` and exit. Do not write any code.
