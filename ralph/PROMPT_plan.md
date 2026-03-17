# Ralph Wiggum — PLAN mode (MEOW)

You are tracing ONE feature through the GVHMR Gradio codebase to document how it works, so a future build session can implement the Qt/PySide6 equivalent.

## Your Job

1. Read `ralph/FEATURES.md` — find the next `[ ]` feature
2. Read `ralph/AGENTS.md` — find which GVHMR source files are relevant
3. **Read those actual source files.** Trace the feature end-to-end:
   - What Gradio components are involved?
   - What callbacks fire and in what order?
   - What state gets read and mutated?
   - What backend functions get called?
   - What are the inputs and outputs at each step?
4. Write a trace document to `ralph/traces/NN-feature-slug.md`
5. Mark the feature `[P]` in `ralph/FEATURES.md`
6. Commit and exit

## Trace Document Format

```markdown
# Feature: <name>

## What it does
<1-2 sentence description of the user-facing behavior>

## Gradio Implementation

### Components
<List every Gradio component involved: type, variable name, file:line>

### Signal Flow
<Step-by-step: user action → callback → state change → UI update>
<Include function signatures and the exact state dict keys read/written>

### Backend Calls
<Which backend functions are called, with what args, returning what>

## State
<What keys in _SESSION_DATA / _POSE_SESSION are touched>
<What gets persisted to disk and where>

## Edge Cases
<Anything tricky: error handling, missing data, race conditions>

## Qt Implementation Notes
<How this maps to Qt: which widget, which signals, what changes>
<Call out anything that works differently in Qt vs Gradio>
```

## Rules

- Trace exactly ONE feature per session
- Read the ACTUAL source code — do not guess or summarize from memory
- Include real function signatures, real variable names, real line numbers
- If a callback chain crosses files, follow it across files
- Do not write any Python code — only the trace document
- The GVHMR source is read-only — do not modify it

## GVHMR Source Location

The Gradio app lives at: GVHMR_ROOT (passed in prompt below).
Backend modules are in the same directory.
