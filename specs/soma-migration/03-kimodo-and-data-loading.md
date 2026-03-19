# Handoff C: Kimodo Text Correction + Data Loading (Phases 6 + 7)

**Prerequisite**: Handoff B complete (Viewport + Pose Corrector SOMA support).

---

## Context

Kimodo is NVIDIA's text-conditioned motion generation model. It takes boundary keyframes + a text prompt ("person walks forward") and fills in a motion span via diffusion. This is the "AI Correction" feature — when the pose corrector finds a bad span, the user types what should happen and Kimodo generates the correction.

Phase 7 updates data loading in `app_window.py` to detect GEM-X SOMA output alongside existing GVHMR SMPL-X output.

**VRAM**: Kimodo needs ~25GB (17GB diffusion + 8GB LLM2Vec). It runs on Powerhouse only. The worker should fail gracefully when Kimodo isn't installed.

---

## Step 1: Create `workers/kimodo_worker.py`

```python
"""Kimodo text-conditioned motion correction worker.

Why: Bad motion spans (detected by pose corrector's auto-scan) can be
fixed by describing what should happen. Kimodo's diffusion model generates
motion conditioned on boundary keyframes and a text prompt, filling the gap
with plausible motion that blends smoothly at the edges.

Requires: kimodo package (pip install from NVIDIA repo).
VRAM: ~25GB (17GB diffusion + 8GB LLM2Vec encoder).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


@dataclass
class KimodoRequest:
    """Parameters for a Kimodo motion generation request."""

    soma_params: dict          # full SOMA params for the person
    start_frame: int           # first frame of the span to fill
    end_frame: int             # last frame of the span to fill (inclusive)
    text_prompt: str           # what should happen in this span
    model_name: str = "kimodo-soma-rp"
    blend_frames: int = 5     # cosine crossfade frames at boundaries


@dataclass
class KimodoResult:
    """Output from Kimodo generation."""

    poses: np.ndarray          # (span_length, 77, 3) generated poses
    start_frame: int
    end_frame: int
    blended_poses: np.ndarray  # (span_length + 2*blend, 77, 3) with crossfade applied
    blend_start: int           # first frame of blended region
    blend_end: int             # last frame of blended region


def _cosine_crossfade(
    original: np.ndarray,
    generated: np.ndarray,
    blend_frames: int,
) -> np.ndarray:
    """Apply cosine crossfade between original and generated motion.

    Parameters
    ----------
    original : (N, J, 3) original poses for the full blended region
    generated : (M, J, 3) generated poses (may be shorter than N)
    blend_frames : number of frames for crossfade at each boundary

    Returns
    -------
    (N, J, 3) blended result
    """
    N = original.shape[0]
    result = original.copy()

    # Place generated poses in the center
    gen_start = blend_frames
    gen_end = gen_start + generated.shape[0]
    if gen_end > N:
        gen_end = N
        generated = generated[:N - gen_start]

    result[gen_start:gen_end] = generated

    # Cosine blend at entry
    for i in range(min(blend_frames, gen_start)):
        t = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / (blend_frames + 1)))
        result[gen_start - blend_frames + i] = (
            (1.0 - t) * original[gen_start - blend_frames + i] + t * generated[0]
        )

    # Cosine blend at exit
    for i in range(min(blend_frames, N - gen_end)):
        t = 0.5 * (1.0 + np.cos(np.pi * (i + 1) / (blend_frames + 1)))
        result[gen_end + i] = (
            t * generated[-1] + (1.0 - t) * original[gen_end + i]
        )

    return result


def _try_import_kimodo():
    """Lazily import kimodo. Returns (load_model, FullBodyConstraintSet, SOMASkeleton30) or Nones."""
    try:
        from kimodo import load_model
        from kimodo.constraints import FullBodyConstraintSet
        from kimodo.skeleton import SOMASkeleton30, SOMASkeleton77
        return load_model, FullBodyConstraintSet, SOMASkeleton30, SOMASkeleton77
    except ImportError:
        return None, None, None, None


class KimodoWorker(QThread):
    """Run Kimodo diffusion to generate text-conditioned motion for a span."""

    progress = Signal(float, str)
    log_line = Signal(str)
    finished = Signal(object)  # KimodoResult
    error = Signal(str)

    def __init__(self, request: KimodoRequest, parent=None):
        super().__init__(parent)
        self._request = request
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            req = self._request
            self.progress.emit(0.0, "Loading Kimodo model...")

            load_model, FullBodyConstraintSet, Skel30, Skel77 = _try_import_kimodo()
            if load_model is None:
                self.error.emit(
                    "Kimodo not installed. Install from NVIDIA repo: pip install kimodo"
                )
                return

            if self._cancelled:
                return

            # Load model
            self.progress.emit(0.1, f"Loading {req.model_name}...")
            model = load_model(req.model_name)
            self.log_line.emit(f"Kimodo model loaded: {req.model_name}")

            if self._cancelled:
                return

            # Extract boundary keyframes
            poses = np.asarray(req.soma_params["poses"], dtype=np.float32)
            n_frames = poses.shape[0]
            start = max(0, req.start_frame)
            end = min(n_frames - 1, req.end_frame)
            span_length = end - start + 1

            self.progress.emit(0.2, f"Building constraints for frames {start}-{end}...")

            # Downconvert 77 → 30 for Kimodo's internal representation
            start_pose_77 = poses[start]  # (77, 3)
            end_pose_77 = poses[end]      # (77, 3)

            # Build constraint set from boundary keyframes
            constraints = FullBodyConstraintSet()
            constraints.add_keyframe(0, start_pose_77)
            constraints.add_keyframe(span_length - 1, end_pose_77)
            constraints.set_text(req.text_prompt)

            if self._cancelled:
                return

            # Run diffusion
            self.progress.emit(0.4, "Running Kimodo diffusion...")
            self.log_line.emit(f"Generating {span_length} frames: \"{req.text_prompt}\"")

            generated_30 = model.generate(
                constraints=constraints,
                n_frames=span_length,
                guidance_scale=7.5,
            )

            if self._cancelled:
                return

            self.progress.emit(0.8, "Upconverting 30→77 joints...")

            # Upconvert 30 → 77
            generated_77 = Skel30.to_SOMASkeleton77(generated_30)
            self.log_line.emit(f"Generated: {generated_77.shape[0]} frames @ 77 joints")

            # Build blended result
            self.progress.emit(0.9, "Blending...")
            blend = req.blend_frames
            blend_start = max(0, start - blend)
            blend_end = min(n_frames - 1, end + blend)

            original_region = poses[blend_start:blend_end + 1].copy()
            blended = _cosine_crossfade(original_region, generated_77, blend)

            result = KimodoResult(
                poses=generated_77,
                start_frame=start,
                end_frame=end,
                blended_poses=blended,
                blend_start=blend_start,
                blend_end=blend_end,
            )

            self.progress.emit(1.0, "Kimodo generation complete")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"Kimodo failed: {e}")
```

---

## Step 2: Update `views/pose_corrector_panel.py` — AI Correction UI

Add a new collapsible section to `PoseCorrectorPanel._setup_ui()`. Find where the existing sections are created and add after the last one (likely after the BVH/FBX export section):

### 2a. Add import at the top of the file

After the existing imports from models/workers:

```python
from workers.kimodo_worker import KimodoWorker, KimodoRequest, KimodoResult
```

Wrap in a try/except to handle when the worker file exists but kimodo doesn't:

```python
try:
    from workers.kimodo_worker import KimodoWorker, KimodoRequest, KimodoResult
    _HAS_KIMODO_WORKER = True
except ImportError:
    _HAS_KIMODO_WORKER = False
```

### 2b. Add AI Correction section in `_setup_ui()`

Add a new `_CollapsibleSection` titled "AI Correction" (collapsed by default):

```python
        # AI Correction (Kimodo)
        if _HAS_KIMODO_WORKER:
            self._ai_section = _CollapsibleSection("AI Correction", collapsed=True)

            ai_layout = self._ai_section.content_layout

            # Text prompt
            prompt_row = QHBoxLayout()
            prompt_row.addWidget(QLabel("Prompt:"))
            self._ai_prompt = QLineEdit()
            self._ai_prompt.setPlaceholderText("e.g., person walks forward naturally")
            prompt_row.addWidget(self._ai_prompt)
            ai_layout.addLayout(prompt_row)

            # Frame range
            range_row = QHBoxLayout()
            range_row.addWidget(QLabel("Start:"))
            self._ai_start_frame = QSpinBox()
            self._ai_start_frame.setRange(0, 999999)
            range_row.addWidget(self._ai_start_frame)
            range_row.addWidget(QLabel("End:"))
            self._ai_end_frame = QSpinBox()
            self._ai_end_frame.setRange(0, 999999)
            range_row.addWidget(self._ai_end_frame)
            ai_layout.addLayout(range_row)

            # Buttons
            btn_row = QHBoxLayout()
            self._ai_generate_btn = QPushButton("Generate")
            self._ai_accept_btn = QPushButton("Accept")
            self._ai_reject_btn = QPushButton("Reject")
            self._ai_accept_btn.setEnabled(False)
            self._ai_reject_btn.setEnabled(False)
            btn_row.addWidget(self._ai_generate_btn)
            btn_row.addWidget(self._ai_accept_btn)
            btn_row.addWidget(self._ai_reject_btn)
            ai_layout.addLayout(btn_row)

            # Status label
            self._ai_status = QLabel("")
            ai_layout.addWidget(self._ai_status)

            main_layout.addWidget(self._ai_section)
```

Note: You'll need to add `QLineEdit` to the imports from PySide6.QtWidgets at the top.

### 2c. Connect signals in `_connect_signals()`

```python
        if _HAS_KIMODO_WORKER:
            self._ai_generate_btn.clicked.connect(self._on_ai_generate)
            self._ai_accept_btn.clicked.connect(self._on_ai_accept)
            self._ai_reject_btn.clicked.connect(self._on_ai_reject)
```

### 2d. Add handler methods

```python
    # ------------------------------------------------------------------
    # AI Correction (Kimodo)
    # ------------------------------------------------------------------

    def _on_ai_generate(self):
        """Start Kimodo generation for the selected frame range."""
        if not _HAS_KIMODO_WORKER:
            return

        track = self._session.person_tracks.get(self._current_person)
        if track is None or track.soma_params is None:
            self._ai_status.setText("No SOMA params — AI correction requires SOMA body model")
            return

        prompt = self._ai_prompt.text().strip()
        if not prompt:
            self._ai_status.setText("Enter a text prompt describing the desired motion")
            return

        request = KimodoRequest(
            soma_params=track.soma_params,
            start_frame=self._ai_start_frame.value(),
            end_frame=self._ai_end_frame.value(),
            text_prompt=prompt,
        )

        self._ai_worker = KimodoWorker(request)
        self._ai_worker.progress.connect(
            lambda frac, msg: self._ai_status.setText(msg)
        )
        self._ai_worker.finished.connect(self._on_ai_finished)
        self._ai_worker.error.connect(
            lambda err: self._ai_status.setText(f"Error: {err}")
        )
        self._ai_generate_btn.setEnabled(False)
        self._ai_status.setText("Generating...")
        self._ai_worker.start()

    def _on_ai_finished(self, result):
        """Kimodo generation complete — show preview, enable accept/reject."""
        self._ai_result = result
        self._ai_generate_btn.setEnabled(True)
        self._ai_accept_btn.setEnabled(True)
        self._ai_reject_btn.setEnabled(True)
        n = result.end_frame - result.start_frame + 1
        self._ai_status.setText(
            f"Generated {n} frames. Preview in viewport, then Accept or Reject."
        )
        # TODO: Preview blended poses in viewport without committing

    def _on_ai_accept(self):
        """Commit the Kimodo-generated poses to the session."""
        if not hasattr(self, "_ai_result") or self._ai_result is None:
            return

        result = self._ai_result
        track = self._session.person_tracks.get(self._current_person)
        if track is None or track.soma_params is None:
            return

        # Apply blended poses to the session
        poses = np.asarray(track.soma_params["poses"])
        old_region = poses[result.blend_start:result.blend_end + 1].copy()

        poses[result.blend_start:result.blend_end + 1] = result.blended_poses

        # Push undo
        from models.session import UndoEntry
        blend_start = result.blend_start
        blend_end = result.blend_end

        self._session.undo_stack.push(UndoEntry(
            description=f"AI correction frames {result.start_frame}-{result.end_frame}",
            undo_fn=lambda: poses.__setitem__(
                slice(blend_start, blend_end + 1), old_region
            ),
            redo_fn=lambda: poses.__setitem__(
                slice(blend_start, blend_end + 1), result.blended_poses
            ),
        ))

        self._ai_accept_btn.setEnabled(False)
        self._ai_reject_btn.setEnabled(False)
        self._ai_status.setText("Correction applied.")
        self._ai_result = None
        self.correction_applied.emit(self._current_person, result.start_frame)

    def _on_ai_reject(self):
        """Discard the Kimodo-generated poses."""
        self._ai_result = None
        self._ai_accept_btn.setEnabled(False)
        self._ai_reject_btn.setEnabled(False)
        self._ai_status.setText("Correction rejected.")
```

### 2e. Auto-populate frame range from issue scanner

In the method that handles navigating to a pose issue (look for the "Go" button handler in the issues table), add:

```python
        # Auto-fill AI correction frame range when navigating to an issue
        if _HAS_KIMODO_WORKER and hasattr(self, '_ai_start_frame'):
            issue = self._pose_issues[self._pose_issue_idx]
            self._ai_start_frame.setValue(issue.span[0])
            self._ai_end_frame.setValue(issue.span[1])
```

---

## Step 3: Update `models/pipeline_config.py`

Add Kimodo model config field. Find the end of the existing fields (after `estimation_backend`):

```python
    estimation_backend: str = "gvhmr"    # "gvhmr" | "gemx"
    kimodo_model: str = "kimodo-soma-rp"  # Kimodo model for AI correction
```

Add to `to_dict()`:

```python
            "kimodo_model": self.kimodo_model,
```

`from_dict()` auto-handles it via the `**{k: v ...}` pattern.

---

## Step 4: Update `app_window.py` — Data Loading (Phase 7)

### 4a. Update `_load_smplx_params()` to also check for SOMA output

Find `_load_smplx_params` (line ~1447):

```python
    def _load_smplx_params(self, person_dir: Path) -> dict | None:
        """Load SMPL-X parameters from hmr4d_results.pt for mesh rendering."""
```

Rename to `_load_motion_params` and add SOMA detection:

```python
    def _load_motion_params(self, person_dir: Path) -> tuple[dict | None, dict | None, str]:
        """Load motion parameters — tries GEM-X (SOMA) first, falls back to GVHMR (SMPL-X).

        Returns (smplx_params, soma_params, body_model_type).
        """
        # Check for GEM-X SOMA output first
        soma_npz = person_dir / "soma_results.npz"
        if soma_npz.is_file():
            try:
                from workers.gemx_worker import load_gemx_soma_output

                soma_params = load_gemx_soma_output(person_dir)
                if soma_params is not None:
                    return None, soma_params, "soma"
            except Exception:
                pass

        # Fall back to GVHMR SMPL-X
        smplx_params = self._load_smplx_params_legacy(person_dir)
        if smplx_params is not None:
            return smplx_params, None, "smplx"

        return None, None, "smplx"

    def _load_smplx_params_legacy(self, person_dir: Path) -> dict | None:
        """Load SMPL-X parameters from hmr4d_results.pt."""
```

Move the existing `_load_smplx_params` body into `_load_smplx_params_legacy`.

### 4b. Update callers

Find all calls to `self._load_smplx_params(...)` in app_window.py and update them:

**In `_load_results_from_output_dir`** (line ~1137):

Replace:
```python
            smplx_params = self._load_smplx_params(pdir)
```

With:
```python
            smplx_params, soma_params, body_model_type = self._load_motion_params(pdir)
```

And update the PersonTrack construction:
```python
            pt = PersonTrack(
                person_id=pid,
                person_dir=pdir,
                confidences=confidences,
                confidence_breakdown=confidence_breakdown,
                smplx_params=smplx_params,
                soma_params=soma_params,
                body_model_type=body_model_type,
            )
```

**In `_hydrate_person_tracks`** (line ~1175):

Replace:
```python
            if track.smplx_params is None:
                track.smplx_params = self._load_smplx_params(track.person_dir)
```

With:
```python
            if track.smplx_params is None and track.soma_params is None:
                smplx, soma, bmt = self._load_motion_params(track.person_dir)
                track.smplx_params = smplx
                track.soma_params = soma
                track.body_model_type = bmt
```

**Any other callers** — search for `_load_smplx_params` and update similarly.

---

## Step 5: Add tests

Add to `tests/test_gemx_worker.py` (or create `tests/test_kimodo_worker.py`):

```python
class TestCossineCrossfade:
    """Cosine crossfade blending for Kimodo output."""

    def test_output_shape_matches_input(self):
        from workers.kimodo_worker import _cosine_crossfade
        original = np.zeros((20, 77, 3), dtype=np.float32)
        generated = np.ones((10, 77, 3), dtype=np.float32)
        result = _cosine_crossfade(original, generated, blend_frames=3)
        assert result.shape == original.shape

    def test_center_is_generated(self):
        from workers.kimodo_worker import _cosine_crossfade
        original = np.zeros((20, 77, 3), dtype=np.float32)
        generated = np.ones((10, 77, 3), dtype=np.float32)
        result = _cosine_crossfade(original, generated, blend_frames=3)
        # Center of the blended region should be close to generated (1.0)
        np.testing.assert_allclose(result[8, 0, 0], 1.0, atol=1e-5)

    def test_edges_blend(self):
        from workers.kimodo_worker import _cosine_crossfade
        original = np.zeros((20, 77, 3), dtype=np.float32)
        generated = np.ones((10, 77, 3), dtype=np.float32)
        result = _cosine_crossfade(original, generated, blend_frames=3)
        # Edges should be between 0 and 1
        assert 0.0 < result[2, 0, 0] < 1.0


class TestKimodoRequest:
    """KimodoRequest dataclass."""

    def test_defaults(self):
        from workers.kimodo_worker import KimodoRequest
        req = KimodoRequest(
            soma_params={"poses": np.zeros((10, 77, 3))},
            start_frame=5,
            end_frame=15,
            text_prompt="walk forward",
        )
        assert req.model_name == "kimodo-soma-rp"
        assert req.blend_frames == 5


class TestKimodoWorker:
    """KimodoWorker construction."""

    def test_constructor(self):
        from workers.kimodo_worker import KimodoWorker, KimodoRequest
        req = KimodoRequest(
            soma_params={"poses": np.zeros((10, 77, 3))},
            start_frame=0,
            end_frame=9,
            text_prompt="stand still",
        )
        worker = KimodoWorker(req)
        assert hasattr(worker, "progress")
        assert hasattr(worker, "finished")
        assert hasattr(worker, "error")

    def test_cancel(self):
        from workers.kimodo_worker import KimodoWorker, KimodoRequest
        req = KimodoRequest(
            soma_params={"poses": np.zeros((10, 77, 3))},
            start_frame=0,
            end_frame=9,
            text_prompt="walk",
        )
        worker = KimodoWorker(req)
        worker.cancel()
        assert worker._cancelled is True
```

---

## Step 6: Run tests

```bash
python -m pytest tests/ -x -q
```

Expected: All previous tests pass + ~10 new tests.

---

## Step 7: Commit

```bash
git add workers/kimodo_worker.py views/pose_corrector_panel.py models/pipeline_config.py app_window.py tests/
git commit -m "feat: add Kimodo AI correction and SOMA data loading (Phases 6 + 7)

New workers/kimodo_worker.py: text-conditioned motion generation via
Kimodo diffusion. Takes boundary keyframes + text prompt, generates
SOMA-77 poses with cosine crossfade blending at boundaries. Gracefully
fails when kimodo package not installed (~25GB VRAM required).

Pose corrector: new 'AI Correction' collapsible section with text prompt,
frame range, Generate/Accept/Reject workflow. Auto-populates frame range
from issue scanner. Undoable via session undo stack.

Pipeline config: add kimodo_model field.

App window: _load_motion_params() checks for GEM-X SOMA output first
(soma_results.npz), falls back to GVHMR SMPL-X (hmr4d_results.pt).
PersonTrack populated with correct body_model_type on load."
```

---

## Step 8: Final merge

After all three handoffs are complete:

```bash
# Verify everything
python -m pytest tests/ -x -q

# Push final state
git push origin soma/skeleton-registry

# Create PR for review
gh pr create --title "SOMA migration: skeleton registry + GEM-X + viewport + Kimodo" --body "$(cat <<'EOF'
## Summary
- Phase 0: Skeleton registry (`models/skeleton.py`) — single source of truth for joint constants
- Phase 1: Session model SOMA support — `soma_params`, `body_model_type` on PersonTrack
- Phase 2: GEM-X worker — single-pass video→SOMA estimation
- Phase 3: Viewport SOMA rendering — dual-path FK, 77-joint support
- Phase 4: Pose corrector rewrite — flat SOMA joint indexing, SOMA issue detection
- Phase 5: SOMA BVH export — 77-joint hierarchy, ZXY Euler, Blender-compatible
- Phase 6: Kimodo AI correction — text-conditioned motion fill with crossfade blending
- Phase 7: Data loading — auto-detect SOMA vs SMPL-X on disk

## Test plan
- [ ] All existing 1522 tests pass (SMPL-X path unchanged)
- [ ] ~50 new SOMA-specific tests pass
- [ ] Manual: Load SMPL-X session → viewport renders, pose corrector works
- [ ] Manual: Load SOMA session → viewport renders 77-joint skeleton
- [ ] Manual: Export SOMA BVH → opens in Blender
- [ ] Manual: Kimodo AI correction (requires GPU + kimodo installed)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
