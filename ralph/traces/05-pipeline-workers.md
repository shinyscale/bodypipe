# Feature: Pipeline Workers

## What it does
Runs GVHMR and SMPLest-X inference as child processes, streams stdout line-by-line to parse progress stages, and reports progress to the Gradio UI. There is **no cancellation mechanism** in the Gradio implementation — once a pipeline starts, it blocks the callback thread until the subprocess exits.

## Gradio Implementation

### Components
| Type | Variable | File:Line | Role |
|------|----------|-----------|------|
| `gr.Button` | `gvhmr_run_btn` | `gvhmr_gui.py:1083` | Starts Tab 1 (GVHMR Body) |
| `gr.Button` | `perf_run_btn` | `gvhmr_gui.py:1197` | Starts Tab 2 (Full Performance Capture) |
| `gr.Button` | `mp_run_btn` | `gvhmr_gui.py:1342` | Starts Tab 3 (Multi-Person) |
| `gr.Textbox` | `gvhmr_log` | `gvhmr_gui.py:1094` | Shows Tab 1 log output |
| `gr.Textbox` | `perf_log` | `gvhmr_gui.py:1218` | Shows Tab 2 log output |
| `gr.Textbox` | `mp_log` | `gvhmr_gui.py:1375` | Shows Tab 3 log output |
| `gr.Progress` | (implicit) | passed as `progress` param | Gradio progress bar updates |

### Signal Flow

#### Tab 1: GVHMR Body — `run_gvhmr()`

1. **User clicks** `gvhmr_run_btn` → `gvhmr_run_btn.click(fn=run_gvhmr, ...)` (`gvhmr_gui.py:1099`)
2. **`run_gvhmr(video_upload, video_path_text, static_cam, use_dpvo, focal_length, progress)`** (`gvhmr_gui.py:159`)
   - Resolves video path via `resolve_video_path()`
   - Saves solve config via `save_solve_config()` for session restore
   - Calls `preprocess_video()` for portrait rotation fix
   - Builds shell command: `python tools/demo/demo.py --video=... [--static_cam] [--use_dpvo] [--f_mm=N]`
   - Emits `progress(0.02, "Starting GVHMR pipeline...")`
3. **Subprocess launch** (`gvhmr_gui.py:201`):
   ```python
   proc = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, cwd=GVHMR_DIR, bufsize=1)
   ```
4. **Line-by-line stdout parsing** (`gvhmr_gui.py:210-216`):
   ```python
   for line in proc.stdout:
       log_lines.append(line.rstrip())
       for pattern, stage_name, frac in STAGE_PATTERNS:
           if pattern.search(line):
               progress(frac, desc=stage_name)
               break
   ```
5. **`proc.wait()`** — blocks until process exits (`gvhmr_gui.py:218`)
6. **On success**: `find_output_dir()` locates results, returns `(side_by_side, incam, global_view, pt_file, log)`
7. **On failure** (`returncode != 0`): returns `(None, None, None, None, full_log)`

#### Tab 2: Full Performance Capture — `run_full_pipeline()`

1. **User clicks** `perf_run_btn` → `run_full_pipeline(...)` (`gvhmr_gui.py:426`)
2. Orchestrates multiple stages sequentially in a single callback:
   - **Stage 1**: `preprocess_video()` — portrait fix + fps resample
   - **Stage 2a** (Hybrid): `_run_gvhmr_subprocess()` → body solve
   - **Stage 2b** (Hybrid): `_run_smplestx_subprocess()` → hand solve
   - **Stage 2c**: `merge_gvhmr_smplestx_params()` — merge body+hands
   - **Stage 2d** (optional): `run_hamer()` — HaMeR hand capture
   - **Stage 3**: Face pipeline (multiple sub-stages with nested progress callbacks)
   - **Stage 4**: BVH/FBX conversion
   - **Stage 5**: Rendering (skeleton, world view, hand overlay)
3. Progress is reported via fractional ranges: each stage owns a fraction of [0.0, 1.0].
   Nested callbacks remap sub-progress into the parent range:
   ```python
   def face_progress(frac, msg):
       overall = 0.50 + frac * 0.22
       progress(overall, desc=msg)
   ```

#### Tab 3: Multi-Person — `run_multi_person_pipeline()`

1. **User clicks** `mp_run_btn` → `run_multi_person_pipeline(...)` (`gvhmr_gui.py:888`)
2. Calls `split_multi_person_video()` which internally:
   - Detects & tracks people (Step 1)
   - Computes shared SLAM (Step 2)
   - Isolates each person's video via SAM2/inpainting or bbox crop (Step 3)
   - Calls `run_person_pipeline()` per person (Step 4) — each spawns `subprocess.Popen` against `demo.py`
   - Assembles world-space results (Step 5)
3. Progress flows via `progress_callback=mp_progress` where:
   ```python
   def mp_progress(frac, msg):
       progress(0.05 + frac * 0.85, desc=msg)
       log_lines.append(f"[MultiPerson] {msg}")
   ```
4. After `split_multi_person_video` returns, the callback does FBX conversion and identity panel init.

### Backend Calls

#### `_run_gvhmr_subprocess(video_path, static_cam, use_dpvo)` → `(pt_path | None, log_lines)`
- **File**: `gvhmr_gui.py:346-388`
- **Command**: `python tools/demo/demo.py --video=... [--static_cam] [--use_dpvo]`
- **CWD**: `GVHMR_DIR`
- **Stdout**: collected line-by-line into `log_lines`, **no progress parsing** (unlike `run_gvhmr()`)
- **Returns**: path to `hmr4d_results.pt` or `None` on failure

#### `_run_smplestx_subprocess(video_path, fps, output_dir)` → `(pt_path | None, log_lines)`
- **File**: `gvhmr_gui.py:249-295`
- **Command**: `{SMPLESTX_PYTHON} smplestx_inference.py --video=... --fps=... --output_dir=... --no_render`
- **CWD**: `SMPLESTX_DIR` (different conda env!)
- **Python**: Uses a hardcoded path to the `smplestx` conda env's Python: `/home/shinyscale/miniconda3/envs/smplestx/bin/python` (`gvhmr_gui.py:34`)
- **Stdout**: collected but **no stage pattern parsing**
- **Returns**: most recently modified `.pt` or `.npz` file from `output_dir`, or `None`

#### `run_person_pipeline(video_path, person_id, output_dir, ...)` → `(pt_path | None, log_lines)`
- **File**: `multi_person_split.py:457-537`
- **Command**: `{sys.executable} tools/demo/demo.py --video=... --output_root=... [--static_cam] [--use_dpvo] [--f_mm=N] [--slam_override=...] [--bbx_override=...] [--skip_render] [--render_incam_only]`
- **CWD**: `GVHMR_DIR`
- **Extra args** vs single-person: `--output_root`, `--slam_override`, `--bbx_override`, `--skip_render`/`--render_incam_only`
- **Stdout**: collected line-by-line, **no progress parsing**
- **Returns**: path to `hmr4d_results.pt` or `None` on failure

#### `reprocess_person(video_path, person_index, person_dir, updated_bboxes, ...)` → `dict`
- **File**: `multi_person_split.py:1250-1561`
- Re-isolates person (using cached SAM2 masks), then calls `run_person_pipeline()` again
- Has its own `progress_callback(frac, msg)` for reprocess stages
- On failure: restores backed-up isolation artifacts (`.bak` files)

### Progress Parsing — STAGE_PATTERNS

Only used by `run_gvhmr()` (Tab 1). Defined at `gvhmr_gui.py:79-87`:

| Pattern (regex, case-insensitive) | Stage Name | Fraction |
|-----------------------------------|------------|----------|
| `preprocess\|loading video\|reading video` | Preprocessing | 0.05 |
| `yolo\|tracking\|detection` | YOLO Tracking | 0.15 |
| `vitpose\|pose estimation\|2d pose` | ViTPose | 0.30 |
| `hmr2\|hmr4d_feature\|feature extraction` | HMR2 Features | 0.45 |
| `dpvo\|simple_vo\|camera estimation\|slam` | Camera Estimation | 0.60 |
| `gvhmr\|predicting\|prediction\|diffusion` | GVHMR Prediction | 0.80 |
| `render\|saving\|visualization` | Rendering | 0.95 |

Each line of stdout is checked against all patterns; the first match updates the progress bar.

## State

### No persistent worker state
- Subprocess runs are fire-and-forget — no worker objects, no process handles stored
- `log_lines` is a local list built up during the callback, returned as the final log string
- `progress` is Gradio's implicit progress tracker (server-push to the client)

### Config persistence
- `save_solve_config(output_dir, tab, **params)` → writes `solve_config.json` alongside results (`gvhmr_gui.py:129-136`)
- `load_solve_config(video_path, tab)` → reads it back to restore UI params when a video is re-loaded (`gvhmr_gui.py:139-154`)
- Config keys vary by tab:
  - `gvhmr`: `static_cam`, `use_dpvo`, `focal_length`
  - `perfcap`: `target_fps`, `fbx_naming`, `pitch_adjust`, `pipeline_mode`, `static_cam`, `use_dpvo`, `use_vitpose_face`, `hand_source`, `body_smooth_preset`
  - `multi_person`: `target_fps`, `fbx_naming`, `static_cam`, `use_dpvo`, `max_persons`, `use_inpainting`

### Output discovery
- `find_output_dir(video_path)` → searches `outputs/demo/{stem}` then `outputs/{stem}` then most-recent dir in `outputs/demo/` (`gvhmr_gui.py:105-120`)
- `find_file(output_dir, pattern)` → `rglob` for first match (`gvhmr_gui.py:123-126`)

## Edge Cases

1. **No cancellation**: Once a subprocess starts, there is no way to abort it from the UI. Closing the browser tab leaves the process running on the server. The Qt version must add `proc.terminate()` / `proc.kill()` support.

2. **Blocking callback**: All three `run_*` functions block the Gradio server thread. Gradio handles this by running callbacks in a thread pool, but the user cannot interact with the tab while it runs. Qt must use `QThread` to avoid blocking the GUI event loop.

3. **SMPLest-X uses a different Python**: The command uses a hardcoded path to a separate conda env (`SMPLESTX_PYTHON = Path("/home/shinyscale/miniconda3/envs/smplestx/bin/python")`). The Qt version should make this configurable.

4. **Nested progress remapping**: `run_full_pipeline` has 6+ sub-stages, each with its own progress fraction range. The sub-stage callbacks do math like `0.50 + frac * 0.22` to map into the parent range. Qt should use a similar approach with `QThread.progress.emit(overall_frac, msg)`.

5. **No stdout progress parsing for helper subprocesses**: `_run_gvhmr_subprocess()` and `_run_smplestx_subprocess()` (the helper versions called by `run_full_pipeline`) do **not** parse `STAGE_PATTERNS` — only `run_gvhmr()` (Tab 1's direct callback) does. This means Tab 2/3 show coarser progress.

6. **`bufsize=1`**: Line-buffered I/O is used, but Python subprocess stdout may still buffer if the child process doesn't flush. `demo.py` output may arrive in bursts.

7. **Return value cardinality**: Each tab returns a different number of outputs:
   - Tab 1 `run_gvhmr`: 5 outputs (3 videos + pt + log)
   - Tab 2 `run_full_pipeline`: 9 outputs (4 videos + bvh + fbx + face_csv + pt + log)
   - Tab 3 `run_multi_person_pipeline`: 9 outputs (track_viz + scene + count + bvh_list + fbx_list + manifest + panel_state + confidence_csvs + log)

## Qt Implementation Notes

### Widget → QThread mapping
Each pipeline tab should have a dedicated `QThread` worker:
- `GVHMRWorker(QThread)` — wraps `subprocess.Popen` for `demo.py`
- `SMPLestXWorker(QThread)` — wraps `subprocess.Popen` for `smplestx_inference.py`
- `FullPipelineWorker(QThread)` — orchestrates GVHMR + SMPLest-X + merge + BVH/FBX sequentially, emitting progress at each stage
- `MultiPersonWorker(QThread)` — wraps `split_multi_person_video()` with `progress_callback` mapped to `Signal`

### Signals
```python
class PipelineWorker(QThread):
    progress = Signal(float, str)   # (fraction 0..1, stage description)
    log_line = Signal(str)          # individual log line for live log panel
    finished = Signal(dict)         # result payload (paths, etc.)
    error = Signal(str)             # error message
```

### Cancellation (NEW — not in Gradio)
The Qt version **must** add cancellation:
```python
def cancel(self):
    self._cancelled = True
    if self._proc and self._proc.poll() is None:
        self._proc.terminate()  # SIGTERM
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()  # SIGKILL
```
Check `self._cancelled` between stages in `FullPipelineWorker` to bail early.

### Progress parsing
Port `STAGE_PATTERNS` as-is for GVHMR subprocess progress. For the orchestration workers, use fractional ranges just like the Gradio version.

### Key differences from Gradio
1. **Threading**: Gradio runs callbacks in a thread pool. Qt uses `QThread` — never run subprocesses on the main thread.
2. **Progress**: Gradio's `progress()` is a server-push mechanism. Qt uses `Signal/Slot` — emit `progress.emit(frac, msg)` and connect to a `QProgressBar` or status label.
3. **Log streaming**: Gradio accumulates log and returns it all at the end. Qt should emit `log_line` signals in real time for a live log panel (`QPlainTextEdit.appendPlainText()`).
4. **Cancellation**: Gradio has none. Qt must support `cancel()` with `proc.terminate()`/`proc.kill()`.
5. **Config paths**: The hardcoded `SMPLESTX_PYTHON` and `SMPLESTX_DIR` should be settings in the Qt app, not constants.
