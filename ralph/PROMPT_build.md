# Ralph Wiggum — BUILD mode (MEOW)

You are implementing ONE feature for **bodypipe**, a Qt/PySide6 app replacing the GVHMR Gradio GUI.

## Your Job

1. Read `ralph/FEATURES.md` — find the next `[P]` feature (planned, ready to build)
2. Read its trace document in `ralph/traces/` — this is your specification
3. Read `ralph/AGENTS.md` — patterns, build commands, source reference
4. Read the existing bodypipe code to understand what's already built
5. Implement the feature
6. Run tests: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ -x -q`
7. Run smoke test: `QT_QPA_PLATFORM=offscreen python main.py --smoke-test`
8. If tests pass: mark the feature `[x]` in `ralph/FEATURES.md`, commit, exit
9. If tests fail: fix and retry (max 3 attempts), then mark `[!]` if stuck

## Rules

- Implement exactly ONE feature per session
- The trace document is your spec — follow it faithfully
- If the trace doc says a callback reads `_SESSION_DATA["confidences"]`, your Qt code reads `session.person_tracks[pid].confidences`
- All widgets subclass QWidget, all workers subclass QThread
- Use PySide6 (NOT PyQt6)
- Signals for cross-widget communication, never direct references
- No module-level mutable state — all state in Session or widgets
- Do not modify any GVHMR source files — import them read-only
- Write tests for non-trivial logic
- Keep the smoke test passing

## Commit Message Format

```
bodypipe: <short description>

Feature NN: <feature name>
```

## GVHMR Source Location

The backend modules live at: GVHMR_ROOT (passed in prompt below).
Import them via the sys.path setup in main.py.
