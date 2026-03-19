"""Tests for Kimodo worker: crossfade, request dataclass, worker construction."""

import numpy as np
import pytest


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
