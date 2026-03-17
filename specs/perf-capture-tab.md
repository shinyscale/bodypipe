# Spec: Performance Capture Tab

## Overview

Second tab — full body+hands+face capture. Extends the single-person tab with hand mode selection and face capture toggle.

## Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Full Performance Capture                                     │
├──────────────────────┬──────────────────────────────────────┤
│                      │                                      │
│  Video Input         │  Output Preview                      │
│  [same as Tab 1]     │  ┌──────────────────────────────┐    │
│                      │  │  Skeleton overlay video       │    │
│  Body Settings       │  │  (body + hands + face)        │    │
│  [same as Tab 1]     │  └──────────────────────────────┘    │
│                      │                                      │
│  Hand Capture        │  Output Files                        │
│  ☑ Enable hands      │  • skeleton.bvh                      │
│  ○ Hybrid (GVHMR+SX) │  • skeleton.fbx                      │
│  ○ SMPLest-X only    │  • merged_params.pt                  │
│                      │  • hand_overlay.mp4                  │
│  Face Capture        │  • face_mesh.mp4                     │
│  ☐ Enable face       │                                      │
│                      │  [Download All]                       │
│  [▶ Run Pipeline]    │                                      │
│  ████████░░░ 60%     │                                      │
│  Stage 2/4: Hands    │                                      │
└──────────────────────┴──────────────────────────────────────┘
```

## Components

### Additional Settings (beyond Tab 1)

- **Hand capture group** (`QGroupBox`):
  - Enable checkbox
  - Radio buttons: "Hybrid (GVHMR body + SMPLest-X hands)" vs "SMPLest-X only"

- **Face capture group**:
  - Enable checkbox (runs face_capture.py)

### Multi-Stage Progress

The full pipeline has multiple stages. Show a stage indicator:

```
Stage 1/4: Body capture (GVHMR)     ████████████░░░░ 75%
Stage 2/4: Hand capture (SMPLest-X)  ░░░░░░░░░░░░░░░░  0%
Stage 3/4: Face capture              ░░░░░░░░░░░░░░░░  0%
Stage 4/4: Merge & Export            ░░░░░░░░░░░░░░░░  0%
```

Use a `QProgressBar` per stage or a single bar with stage label.

## Worker

Reuse `GVHMRWorker` for body, add a `FullPipelineWorker` that orchestrates:
1. GVHMRWorker (body)
2. SMPLestXWorker (hands — runs in separate conda env)
3. FaceCaptureWorker (face)
4. Merge step (in-process, fast)

Or: a single `FullPipelineWorker` that runs the full sequence internally, emitting per-stage progress.

The simpler approach (single worker) is preferred — matches the Gradio `run_full_pipeline()` which does everything sequentially.

## Source Reference

- `gvhmr_gui.py:1360-1400` — Tab 2 Gradio layout
- `gvhmr_gui.py:run_full_pipeline()` — full pipeline orchestration
- `gvhmr_gui.py:_run_smplestx_subprocess()` — SMPLest-X subprocess
- `smplx_to_bvh.py:merge_gvhmr_smplestx_params()` — merge step

## Acceptance Criteria

- All settings from Tab 1 plus hand/face toggles
- Pipeline runs all enabled stages sequentially
- Per-stage progress visible
- Output includes merged BVH/FBX when hands enabled
- SMPLest-X runs in its own conda env (subprocess activation)
