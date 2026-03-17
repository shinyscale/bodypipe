# Spec: Single-Person Tab (GVHMR Body)

## Overview

First tab — simple single-person body capture. Video in, motion out. Mirrors Gradio Tab 1.

## Layout

```
┌─────────────────────────────────────────────────────────────┐
│ GVHMR Body Capture                                          │
├──────────────────────┬──────────────────────────────────────┤
│                      │                                      │
│  Video Input         │  Output Preview                      │
│  ┌──────────────┐    │  ┌──────────────────────────────┐    │
│  │ Drop video   │    │  │                              │    │
│  │ or Browse... │    │  │    Side-by-side video         │    │
│  └──────────────┘    │  │    (after pipeline)           │    │
│                      │  │                              │    │
│  Settings            │  └──────────────────────────────┘    │
│  ☑ Static camera     │                                      │
│  ☐ Use DPVO          │  Output Files                        │
│  Focal length: 24mm  │  • incam.mp4                         │
│                      │  • global.mp4                         │
│  [▶ Run GVHMR]       │  • hmr4d_results.pt                  │
│  ████████░░░ 60%     │  • skeleton.bvh                      │
│  "Running GVHMR..."  │                                      │
│                      │  [Download All]                       │
└──────────────────────┴──────────────────────────────────────┘
```

## Components

### Left Panel — Input & Controls

- **Video input**: Drag-and-drop area (`QLabel` with drop event) + Browse button (`QFileDialog`)
  - Show thumbnail of first frame after loading
  - Show video info: resolution, frame count, FPS, duration

- **Settings group** (`QGroupBox`):
  - `static_cam` checkbox (default: checked)
  - `use_dpvo` checkbox (default: unchecked)
  - `focal_mm` spinbox (QDoubleSpinBox, range 10-200, default 24)

- **Run button**: QPushButton, disabled until video loaded
  - While running: shows progress bar + status text
  - Cancel button appears during run

### Right Panel — Output

- **Video preview**: VideoPlayer widget showing side-by-side result
- **Output files**: QListWidget with clickable items
  - Double-click opens file in system viewer
  - Right-click: Copy Path, Open Containing Folder
- **Download All**: Copies output dir (or opens it in file manager)

## State

Uses `PipelineConfig` for settings persistence. On tab switch or app close, settings are saved.

## Worker Integration

```python
def _on_run(self):
    config = PipelineConfig(
        mode="single",
        static_cam=self._static_cam.isChecked(),
        use_dpvo=self._use_dpvo.isChecked(),
        focal_mm=self._focal_mm.value(),
    )
    self._worker = GVHMRWorker(self._video_path, config)
    self._worker.progress.connect(self._update_progress)
    self._worker.finished.connect(self._on_finished)
    self._worker.error.connect(self._on_error)
    self._worker.start()
    self._set_running(True)
```

## Source Reference

- `gvhmr_gui.py:1300-1360` — Tab 1 Gradio layout
- `gvhmr_gui.py:run_gvhmr()` — solo pipeline function
- `gvhmr_gui.py:find_output_dir()` — output discovery

## Acceptance Criteria

- Load video via browse or drag-and-drop
- Settings persist across app launches
- Run button triggers GVHMRWorker
- Progress bar + status update during run
- Output files listed and openable after completion
- Cancel stops the pipeline
