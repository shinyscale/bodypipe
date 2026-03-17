# Spec: Pipeline Runner

## Overview

QThread-based workers for running GVHMR and related subprocess pipelines with progress reporting, log capture, and cancellation. Replaces the Gradio `gr.Progress` pattern.

## Workers

### GVHMRWorker (`workers/gvhmr_worker.py`)

Runs the GVHMR demo.py subprocess for single-person body capture.

```python
class GVHMRWorker(QThread):
    progress = Signal(float, str)   # (0.0-1.0, stage description)
    log_line = Signal(str)          # raw stdout/stderr line
    finished = Signal(dict)         # {"output_dir": Path, "params_path": Path, ...}
    error = Signal(str)

    def __init__(self, video_path: Path, config: PipelineConfig, parent=None): ...
    def run(self): ...
    def cancel(self): ...
```

**Progress Detection**: Parse stdout line-by-line with regex to detect stages:
- "Preprocessing" → 0.1
- "Running ViTPose" → 0.2
- "Running DPVO/SLAM" → 0.4
- "Running GVHMR" → 0.6
- "Rendering" → 0.8
- "Done" → 1.0

(Same patterns as `_run_gvhmr_subprocess` in gvhmr_gui.py)

### ReprocessWorker (`workers/reprocess_worker.py`)

Runs incremental per-person reprocess after bbox corrections.

```python
class ReprocessWorker(QThread):
    progress = Signal(float, str)
    person_done = Signal(int)        # person_id completed
    finished = Signal(dict)          # {"reprocessed": list[int]}
    error = Signal(str)

    def __init__(self, session: Session, person_ids: list[int], parent=None): ...
    def run(self): ...
    def cancel(self): ...
```

Wraps `multi_person_split.reprocess_person()` — runs it for each dirty person sequentially, emitting `person_done` after each.

### RenderWorker (`workers/render_worker.py`)

Runs scene preview rendering (multi-person in-camera composite).

```python
class RenderWorker(QThread):
    progress = Signal(float, str)
    finished = Signal(Path)          # path to rendered video
    error = Signal(str)

    def __init__(self, session: Session, parent=None): ...
    def run(self): ...
    def cancel(self): ...
```

Wraps `multi_person_split.render_multi_person_incam()`.

## Subprocess Execution Pattern

All subprocess workers follow this pattern:

```python
def run(self):
    cmd = self._build_command()
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    for line in process.stdout:
        if self._cancelled:
            process.terminate()
            process.wait(timeout=5)
            return

        self.log_line.emit(line.rstrip())
        stage, fraction = self._parse_progress(line)
        if fraction is not None:
            self.progress.emit(fraction, stage)

    returncode = process.wait()
    if returncode != 0:
        self.error.emit(f"Process exited with code {returncode}")
        return

    result = self._collect_results()
    self.finished.emit(result)
```

## Cancellation

- Set `self._cancelled = True` from the main thread
- Worker checks flag each stdout line
- On cancel: `SIGTERM` → wait 5s → `SIGKILL` if still alive
- Emit no finished/error signal on cancel

## Integration with UI

Pipeline tabs connect workers to UI elements:

```python
# In pipeline tab
self._worker = GVHMRWorker(video_path, config)
self._worker.progress.connect(self._on_progress)
self._worker.log_line.connect(self._log_panel.append_line)
self._worker.finished.connect(self._on_finished)
self._worker.error.connect(self._on_error)
self._worker.start()

# Progress bar
def _on_progress(self, fraction: float, stage: str):
    self._progress_bar.setValue(int(fraction * 100))
    self._status_label.setText(stage)
```

## Source Reference

- `gvhmr_gui.py:run_gvhmr()` — solo pipeline with subprocess
- `gvhmr_gui.py:_run_gvhmr_subprocess()` — subprocess wrapper with progress parsing
- `gvhmr_gui.py:run_full_pipeline()` — full capture orchestration
- `gvhmr_gui.py:run_multi_person_pipeline()` — multi-person orchestration
- `multi_person_split.py:reprocess_person()` — incremental reprocess

## Acceptance Criteria

- GVHMRWorker runs demo.py and reports progress
- Log lines appear in log panel in real-time
- Cancel stops subprocess within 5 seconds
- ReprocessWorker handles multiple persons sequentially
- Errors propagate to UI without crashing
