"""Tests for data models."""

import json
import sys
from pathlib import Path

# Ensure imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.session import Session, PersonTrack, UndoEntry, UndoStack
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


class TestUndoStack:
    """Tests for UndoStack — the command-pattern undo/redo engine.

    Why: Undo/redo is critical for interactive editing sessions. These tests
    verify the stack correctly manages entries, enforces depth limits, clears
    redo on new push, and fires the on_changed callback so the UI stays in sync.
    """

    def test_empty_stack(self):
        """Fresh stack has nothing to undo or redo."""
        stack = UndoStack()
        assert not stack.can_undo()
        assert not stack.can_redo()
        assert stack.undo() is None
        assert stack.redo() is None
        assert stack.peek_undo() is None
        assert stack.peek_redo() is None

    def test_push_and_undo(self):
        """Push an entry, then undo should call undo_fn and return description."""
        called = []
        entry = UndoEntry(
            description="test op",
            undo_fn=lambda: called.append("undo"),
            redo_fn=lambda: called.append("redo"),
        )
        stack = UndoStack()
        stack.push(entry)

        assert stack.can_undo()
        assert not stack.can_redo()
        assert stack.peek_undo() == "test op"

        desc = stack.undo()
        assert desc == "test op"
        assert called == ["undo"]
        assert not stack.can_undo()
        assert stack.can_redo()
        assert stack.peek_redo() == "test op"

    def test_undo_then_redo(self):
        """Undo followed by redo should call both callbacks in order."""
        value = [0]
        entry = UndoEntry(
            description="set to 1",
            undo_fn=lambda: value.__setitem__(0, 0),
            redo_fn=lambda: value.__setitem__(0, 1),
        )
        stack = UndoStack()
        stack.push(entry)

        stack.undo()
        assert value[0] == 0

        desc = stack.redo()
        assert desc == "set to 1"
        assert value[0] == 1
        assert stack.can_undo()
        assert not stack.can_redo()

    def test_push_clears_redo(self):
        """A new push after undo should wipe the redo stack."""
        stack = UndoStack()
        stack.push(UndoEntry("a", lambda: None, lambda: None))
        stack.undo()
        assert stack.can_redo()

        stack.push(UndoEntry("b", lambda: None, lambda: None))
        assert not stack.can_redo()
        assert stack.peek_undo() == "b"

    def test_max_depth(self):
        """Stack should evict oldest entries when max_depth is exceeded."""
        stack = UndoStack(max_depth=3)
        for i in range(5):
            stack.push(UndoEntry(f"op{i}", lambda: None, lambda: None))

        assert stack.can_undo()
        descs = []
        while stack.can_undo():
            descs.append(stack.undo())
        # Only the last 3 pushes should remain
        assert descs == ["op4", "op3", "op2"]

    def test_clear(self):
        """Clear should empty both undo and redo stacks."""
        stack = UndoStack()
        stack.push(UndoEntry("a", lambda: None, lambda: None))
        stack.push(UndoEntry("b", lambda: None, lambda: None))
        stack.undo()
        assert stack.can_undo()
        assert stack.can_redo()

        stack.clear()
        assert not stack.can_undo()
        assert not stack.can_redo()

    def test_on_changed_callback(self):
        """on_changed should fire on push, undo, redo, and clear."""
        calls = []
        stack = UndoStack()
        stack.on_changed = lambda: calls.append("changed")

        stack.push(UndoEntry("a", lambda: None, lambda: None))
        assert len(calls) == 1

        stack.undo()
        assert len(calls) == 2

        stack.redo()
        assert len(calls) == 3

        stack.clear()
        assert len(calls) == 4

    def test_multiple_entries(self):
        """Multiple entries should undo in LIFO order."""
        log = []
        for i in range(3):
            val = i
            entry = UndoEntry(
                f"op{val}",
                undo_fn=lambda v=val: log.append(f"undo{v}"),
                redo_fn=lambda v=val: log.append(f"redo{v}"),
            )
            stack = UndoStack() if i == 0 else stack
            stack.push(entry)

        stack.undo()
        stack.undo()
        assert log == ["undo2", "undo1"]

        stack.redo()
        assert log[-1] == "redo1"

    def test_undo_exception_does_not_corrupt(self):
        """If undo_fn raises, the entry still moves to redo stack."""
        stack = UndoStack()
        stack.push(UndoEntry(
            "bad",
            undo_fn=lambda: (_ for _ in ()).throw(ValueError("oops")),
            redo_fn=lambda: None,
        ))
        desc = stack.undo()
        assert desc == "bad"
        # Entry should be on redo stack despite the exception
        assert stack.can_redo()
        assert not stack.can_undo()

    def test_session_has_undo_stack(self):
        """Session should include a default UndoStack."""
        s = Session()
        assert isinstance(s.undo_stack, UndoStack)
        assert not s.undo_stack.can_undo()

    def test_session_reset_clears_undo(self):
        """Session.reset() should clear the undo stack."""
        s = Session()
        s.undo_stack.push(UndoEntry("x", lambda: None, lambda: None))
        assert s.undo_stack.can_undo()
        s.reset()
        assert not s.undo_stack.can_undo()

    def test_session_save_load_preserves_undo_stack(self, tmp_path):
        """Saving and loading should not crash due to undo_stack field."""
        s = Session(video_path=Path("/tmp/test.mp4"), num_frames=10)
        s.undo_stack.push(UndoEntry("x", lambda: None, lambda: None))
        path = tmp_path / "session.json"
        s.save(path)
        loaded = Session.load(path)
        # Loaded session gets a fresh (empty) undo stack
        assert isinstance(loaded.undo_stack, UndoStack)
        assert not loaded.undo_stack.can_undo()

    def test_undo_redo_data_round_trip(self):
        """End-to-end: push a state mutation, undo it, redo it, verify state."""
        session = Session(num_frames=100)
        session.person_tracks[1] = PersonTrack(
            person_id=1, keyframes=[{"frame": 10, "verified": False}]
        )

        track = session.person_tracks[1]
        old_kf = [dict(kf) for kf in track.keyframes]

        # Simulate adding a keyframe
        track.keyframes.append({"frame": 50, "verified": True})
        new_kf = [dict(kf) for kf in track.keyframes]

        session.undo_stack.push(UndoEntry(
            "Add keyframe",
            undo_fn=lambda: setattr(track, "keyframes", [dict(kf) for kf in old_kf]),
            redo_fn=lambda: setattr(track, "keyframes", [dict(kf) for kf in new_kf]),
        ))

        assert len(track.keyframes) == 2

        session.undo_stack.undo()
        assert len(track.keyframes) == 1
        assert track.keyframes[0]["frame"] == 10

        session.undo_stack.redo()
        assert len(track.keyframes) == 2
        assert track.keyframes[1]["frame"] == 50
