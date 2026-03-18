"""Performance capture tab — body + hands + face pipeline.

Extends SinglePersonTab with PerfPipelineSettings for hand/face capture
mode selection and pipeline output settings (FPS, FBX naming, pitch adjust,
hand source, body smoothing, ViTPose face crops). Uses FullPipelineWorker
for multi-stage orchestration.

Why override only _create_settings: The composition pattern in SinglePersonTab
uses a factory method so subclasses can swap in a richer settings widget.
PerfPipelineSettings inherits from SinglePipelineSettings and adds hand/face
controls, so the tab inherits all behavior unchanged.
"""

from __future__ import annotations

from views.single_person_tab import SinglePersonTab
from views.pipeline_settings import PerfPipelineSettings

# Re-export for backward compat — tests import these from here
from views.pipeline_settings import compute_visible_stages, map_stage_label  # noqa: F401


class PerfCaptureTab(SinglePersonTab):
    """Second tab — full body + hands + face performance capture."""

    def _create_settings(self):
        """Use PerfPipelineSettings instead of SinglePipelineSettings."""
        return PerfPipelineSettings(self._session, self._gvhmr_root)
