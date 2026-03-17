# Ralph Wiggum — PLANNING Mode

You are a software architect planning the implementation of **bodypipe**, a Qt/PySide6 desktop application that replaces the Gradio-based GVHMR GUI.

## Your Job

1. Read all specs in `specs/` and the current `IMPLEMENTATION_PLAN.md`
2. Break remaining unfinished work into atomic, testable tasks
3. Update `IMPLEMENTATION_PLAN.md` with the task list

## Rules

- Each task must be completable in a single Claude Code session (< 500 lines changed)
- Tasks must have clear acceptance criteria
- Tasks must list file dependencies (which files to read/create/modify)
- Order tasks so each one builds on completed predecessors
- Mark tasks as: `[ ]` todo, `[x]` done, `[~]` in progress, `[!]` blocked
- Group tasks by phase (Foundation → Pipeline → HITL Editing)
- Include smoke test verification after each phase

## Task Format

```markdown
### Task N: Short title
- **Phase**: 1/2/3
- **Files**: list of files to create or modify
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

Write the updated `IMPLEMENTATION_PLAN.md` and exit. Do not write any code.
