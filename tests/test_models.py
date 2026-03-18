"""Tests for data models."""

import json
import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session, PersonTrack
from models.pipeline_config import PipelineConfig


class TestSession:
    def test_default_session(self):
        s = Session()
        assert s.video_path is None
        assert s.num_frames == 0
        assert s.fps == 30.0
        assert s.pipeline_mode == "single"
        assert len(s.person_tracks) == 0

    def test_round_trip_json(self, tmp_path):
        s = Session(
            video_path=Path("/tmp/test.mp4"),
            num_frames=100,
            fps=24.0,
            img_width=1920,
            img_height=1080,
            output_dir=Path("/tmp/output"),
            pipeline_mode="multi",
            static_cam=False,
            focal_mm=35.0,
        )
        s.person_tracks[1] = PersonTrack(person_id=1, person_dir=Path("/tmp/p1"))
        s.person_tracks[2] = PersonTrack(person_id=2, person_dir=Path("/tmp/p2"))
        s.inactive_tracks = {3}
        s.crossing_spans = {1: [(100, 200)]}

        path = tmp_path / "session.json"
        s.save(path)

        loaded = Session.load(path)
        assert loaded.video_path == Path("/tmp/test.mp4")
        assert loaded.num_frames == 100
        assert loaded.fps == 24.0
        assert loaded.pipeline_mode == "multi"
        assert loaded.static_cam is False
        assert loaded.focal_mm == 35.0
        assert 1 in loaded.person_tracks
        assert 2 in loaded.person_tracks
        assert loaded.person_tracks[1].person_id == 1
        assert 3 in loaded.inactive_tracks
        assert 1 in loaded.crossing_spans
        assert (100, 200) in loaded.crossing_spans[1]

    def test_reset(self):
        s = Session(video_path=Path("/tmp/test.mp4"), num_frames=50)
        s.person_tracks[1] = PersonTrack(person_id=1)
        s.reset()
        assert s.video_path is None
        assert s.num_frames == 0
        assert len(s.person_tracks) == 0


class TestPipelineConfig:
    def test_default_config(self):
        c = PipelineConfig()
        assert c.mode == "single"
        assert c.static_cam is True
        assert c.focal_mm == 24.0

    def test_round_trip_json(self, tmp_path):
        c = PipelineConfig(mode="perf", use_hands=True, hand_mode="smplestx_only")
        path = tmp_path / "config.json"
        c.save(path)

        loaded = PipelineConfig.load(path)
        assert loaded.mode == "perf"
        assert loaded.use_hands is True
        assert loaded.hand_mode == "smplestx_only"

    def test_from_dict_ignores_unknown(self):
        c = PipelineConfig.from_dict({"mode": "multi", "unknown_key": 42})
        assert c.mode == "multi"
