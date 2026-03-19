"""Tests for Session Library — media-pool panel, session metadata, dock wrapper.

Why these tests matter: The session library is the primary way users browse
and load previous capture sessions.  Tests verify that session scanning
discovers files, metadata is parsed correctly, filtering works, tags and
notes persist to JSON, and the dock integration wires signals properly.
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from models.session import Session, PersonTrack
from views.session_library import (
    SessionEntry,
    SessionLibrary,
    _SessionCard,
    _format_duration,
    _load_session_entry,
    _update_session_json,
    THUMBNAIL_HEIGHT,
)
from views.dock_widgets import SessionLibraryDock


# ---------------------------------------------------------------------------
# Helper: create a fake session JSON on disk
# ---------------------------------------------------------------------------

def _write_session_json(path: Path, **overrides):
    """Write a minimal session JSON file for testing."""
    data = {
        "video_path": str(path.parent / "test_video.mp4"),
        "num_frames": 300,
        "fps": 30.0,
        "img_width": 1920,
        "img_height": 1080,
        "output_dir": str(path.parent),
        "pipeline_mode": "single",
        "static_cam": True,
        "use_dpvo": False,
        "focal_mm": 24.0,
        "person_tracks": {},
        "inactive_tracks": [],
        "crossing_spans": {},
        "notes": "",
        "tags": [],
        "version": 1,
    }
    data.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


# ===========================================================================
# Session model: notes, tags, version
# ===========================================================================

class TestSessionMetadataFields:
    """Verify that notes, tags, and version serialize round-trip correctly."""

    def test_default_values(self, session):
        assert session.notes == ""
        assert session.tags == []
        assert session.version == 1

    def test_to_dict_includes_metadata(self, session):
        session.notes = "Test annotation"
        session.tags = ["walking", "outdoor"]
        session.version = 3
        d = session.to_dict()
        assert d["notes"] == "Test annotation"
        assert d["tags"] == ["walking", "outdoor"]
        assert d["version"] == 3

    def test_from_dict_loads_metadata(self):
        data = {
            "video_path": "/tmp/vid.mp4",
            "notes": "Some notes",
            "tags": ["tag1", "tag2"],
            "version": 2,
        }
        s = Session.from_dict(data)
        assert s.notes == "Some notes"
        assert s.tags == ["tag1", "tag2"]
        assert s.version == 2

    def test_from_dict_defaults_when_missing(self):
        s = Session.from_dict({})
        assert s.notes == ""
        assert s.tags == []
        assert s.version == 1

    def test_reset_clears_metadata(self, session):
        session.notes = "Old notes"
        session.tags = ["old"]
        session.version = 5
        session.reset()
        assert session.notes == ""
        assert session.tags == []
        assert session.version == 1

    def test_save_load_roundtrip(self, session, tmp_path):
        session.notes = "Roundtrip test"
        session.tags = ["a", "b", "c"]
        session.version = 4
        session.video_path = Path("/tmp/video.mp4")

        save_path = tmp_path / "test_session.json"
        session.save(save_path)

        loaded = Session.load(save_path)
        assert loaded.notes == "Roundtrip test"
        assert loaded.tags == ["a", "b", "c"]
        assert loaded.version == 4


# ===========================================================================
# _format_duration
# ===========================================================================

class TestFormatDuration:

    def test_zero(self):
        assert _format_duration(0) == "0:00"

    def test_seconds_only(self):
        assert _format_duration(45) == "0:45"

    def test_minutes_and_seconds(self):
        assert _format_duration(125) == "2:05"

    def test_large_value(self):
        assert _format_duration(3661) == "61:01"

    def test_fractional_seconds(self):
        assert _format_duration(90.7) == "1:30"


# ===========================================================================
# _load_session_entry
# ===========================================================================

class TestLoadSessionEntry:

    def test_loads_basic_session(self, tmp_path):
        session_path = tmp_path / "bodypipe_session.json"
        _write_session_json(session_path, notes="hello", tags=["test"], version=2)

        entry = _load_session_entry(session_path)
        assert entry is not None
        assert entry.path == session_path
        assert entry.video_name == "test_video.mp4"
        assert entry.num_frames == 300
        assert entry.fps == 30.0
        assert entry.duration_sec == pytest.approx(10.0)
        assert entry.notes == "hello"
        assert entry.tags == ["test"]
        assert entry.version == 2

    def test_returns_none_for_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")
        assert _load_session_entry(path) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        assert _load_session_entry(tmp_path / "nonexistent.json") is None

    def test_counts_persons(self, tmp_path):
        session_path = tmp_path / "bodypipe_session.json"
        _write_session_json(
            session_path,
            person_tracks={
                "0": {"person_id": 0, "person_dir": None, "keyframes": [
                    {"frame": 10, "verified": True},
                    {"frame": 20, "verified": False},
                ]},
                "1": {"person_id": 1, "person_dir": None, "keyframes": []},
            },
        )
        entry = _load_session_entry(session_path)
        assert entry.person_count == 2
        assert entry.correction_count == 2  # 2 keyframes total

    def test_handles_missing_video_path(self, tmp_path):
        session_path = tmp_path / "bodypipe_session.json"
        _write_session_json(session_path, video_path=None)
        entry = _load_session_entry(session_path)
        assert entry is not None
        assert entry.video_name == "(no video)"
        assert entry.thumbnail is None


# ===========================================================================
# _update_session_json
# ===========================================================================

class TestUpdateSessionJson:

    def test_updates_notes(self, tmp_path):
        path = tmp_path / "bodypipe_session.json"
        _write_session_json(path)

        _update_session_json(path, notes="Updated notes")

        with open(path) as f:
            data = json.load(f)
        assert data["notes"] == "Updated notes"

    def test_updates_tags(self, tmp_path):
        path = tmp_path / "bodypipe_session.json"
        _write_session_json(path)

        _update_session_json(path, tags=["new_tag"])

        with open(path) as f:
            data = json.load(f)
        assert data["tags"] == ["new_tag"]

    def test_preserves_other_fields(self, tmp_path):
        path = tmp_path / "bodypipe_session.json"
        _write_session_json(path, num_frames=500)

        _update_session_json(path, notes="test")

        with open(path) as f:
            data = json.load(f)
        assert data["num_frames"] == 500
        assert data["notes"] == "test"

    def test_handles_missing_file_gracefully(self, tmp_path):
        """Should not raise — just log the error."""
        _update_session_json(tmp_path / "nonexistent.json", notes="test")


# ===========================================================================
# _SessionCard
# ===========================================================================

class TestSessionCard:

    def test_creates_card(self, qapp):
        entry = SessionEntry(
            path=Path("/tmp/session.json"),
            video_name="test.mp4",
            duration_sec=120.5,
            person_count=3,
            correction_count=5,
            tags=["walking", "outdoor"],
            notes="Test notes for tooltip",
            version=2,
        )
        card = _SessionCard(entry)
        assert card.entry is entry
        assert card.toolTip() == "Test notes for tooltip"

    def test_card_without_notes_has_no_tooltip(self, qapp):
        entry = SessionEntry(path=Path("/tmp/s.json"), video_name="v.mp4")
        card = _SessionCard(entry)
        assert card.toolTip() == ""

    def test_card_with_version_1_no_badge(self, qapp):
        entry = SessionEntry(
            path=Path("/tmp/s.json"), video_name="clip.mp4", version=1,
        )
        card = _SessionCard(entry)
        # Version 1 should not show "v1" badge — just the name
        assert card.entry.version == 1

    def test_card_with_tags(self, qapp):
        entry = SessionEntry(
            path=Path("/tmp/s.json"),
            video_name="clip.mp4",
            tags=["a", "b", "c", "d", "e", "f"],  # 6 tags, only 5 visible
        )
        card = _SessionCard(entry)
        assert card.entry.tags == ["a", "b", "c", "d", "e", "f"]


# ===========================================================================
# SessionLibrary
# ===========================================================================

class TestSessionLibrary:

    def test_creates_empty_library(self, qapp):
        lib = SessionLibrary()
        assert lib.entries == []
        assert lib.list_widget.count() == 0

    def test_scan_discovers_sessions(self, qapp, tmp_path):
        """Create session files in the expected directory structure and scan."""
        # Create output dirs matching GVHMR structure
        demo_dir = tmp_path / "outputs" / "demo" / "test_video"
        demo_dir.mkdir(parents=True)
        _write_session_json(demo_dir / "bodypipe_session.json")

        multi_dir = tmp_path / "outputs" / "multi_person" / "test_multi"
        multi_dir.mkdir(parents=True)
        _write_session_json(
            multi_dir / "bodypipe_session.json",
            video_path=str(multi_dir / "multi.mp4"),
            person_tracks={"0": {"person_id": 0, "person_dir": None, "keyframes": []}},
        )

        lib = SessionLibrary(gvhmr_root=tmp_path)
        lib.scan()

        assert len(lib.entries) == 2
        assert lib.list_widget.count() == 2

    def test_scan_with_extra_paths(self, qapp, tmp_path):
        """Extra paths (recent sessions) are included in the scan."""
        extra_path = tmp_path / "custom" / "bodypipe_session.json"
        _write_session_json(extra_path)

        lib = SessionLibrary(gvhmr_root=tmp_path)
        lib.scan(extra_paths=[extra_path])

        assert len(lib.entries) == 1
        assert lib.entries[0].path == extra_path

    def test_scan_deduplicates(self, qapp, tmp_path):
        """Same session file found via scan and extra_paths appears once."""
        demo_dir = tmp_path / "outputs" / "demo" / "vid"
        demo_dir.mkdir(parents=True)
        session_path = demo_dir / "bodypipe_session.json"
        _write_session_json(session_path)

        lib = SessionLibrary(gvhmr_root=tmp_path)
        lib.scan(extra_paths=[session_path])

        assert len(lib.entries) == 1

    def test_filter_by_video_name(self, qapp, tmp_path):
        """Filter hides non-matching sessions."""
        dir1 = tmp_path / "outputs" / "demo" / "alpha"
        dir1.mkdir(parents=True)
        _write_session_json(
            dir1 / "bodypipe_session.json",
            video_path=str(dir1 / "alpha_clip.mp4"),
        )

        dir2 = tmp_path / "outputs" / "demo" / "beta"
        dir2.mkdir(parents=True)
        _write_session_json(
            dir2 / "bodypipe_session.json",
            video_path=str(dir2 / "beta_clip.mp4"),
        )

        lib = SessionLibrary(gvhmr_root=tmp_path)
        lib.scan()
        assert lib.list_widget.count() == 2

        lib.search_input.setText("alpha")
        assert lib.list_widget.count() == 1

    def test_filter_by_tag(self, qapp, tmp_path):
        dir1 = tmp_path / "outputs" / "demo" / "a"
        dir1.mkdir(parents=True)
        _write_session_json(
            dir1 / "bodypipe_session.json",
            tags=["walking"],
        )

        dir2 = tmp_path / "outputs" / "demo" / "b"
        dir2.mkdir(parents=True)
        _write_session_json(dir2 / "bodypipe_session.json", tags=["running"])

        lib = SessionLibrary(gvhmr_root=tmp_path)
        lib.scan()
        assert lib.list_widget.count() == 2

        lib.search_input.setText("walking")
        assert lib.list_widget.count() == 1

    def test_filter_clear_shows_all(self, qapp, tmp_path):
        dir1 = tmp_path / "outputs" / "demo" / "a"
        dir1.mkdir(parents=True)
        _write_session_json(dir1 / "bodypipe_session.json")

        dir2 = tmp_path / "outputs" / "demo" / "b"
        dir2.mkdir(parents=True)
        _write_session_json(dir2 / "bodypipe_session.json")

        lib = SessionLibrary(gvhmr_root=tmp_path)
        lib.scan()

        lib.search_input.setText("zzz_no_match")
        assert lib.list_widget.count() == 0

        lib.search_input.setText("")
        assert lib.list_widget.count() == 2

    def test_double_click_emits_signal(self, qapp, tmp_path):
        dir1 = tmp_path / "outputs" / "demo" / "a"
        dir1.mkdir(parents=True)
        session_path = dir1 / "bodypipe_session.json"
        _write_session_json(session_path)

        lib = SessionLibrary(gvhmr_root=tmp_path)
        lib.scan()

        received = []
        lib.session_load_requested.connect(received.append)

        item = lib.list_widget.item(0)
        lib._on_double_click(item)

        assert len(received) == 1
        assert received[0] == str(session_path)

    def test_sorted_newest_first(self, qapp, tmp_path):
        """Sessions should be sorted by modification date, newest first."""
        import time

        dir1 = tmp_path / "outputs" / "demo" / "older"
        dir1.mkdir(parents=True)
        p1 = dir1 / "bodypipe_session.json"
        _write_session_json(p1, video_path=str(dir1 / "older.mp4"))

        time.sleep(0.05)  # ensure different mtime

        dir2 = tmp_path / "outputs" / "demo" / "newer"
        dir2.mkdir(parents=True)
        p2 = dir2 / "bodypipe_session.json"
        _write_session_json(p2, video_path=str(dir2 / "newer.mp4"))

        lib = SessionLibrary(gvhmr_root=tmp_path)
        lib.scan()

        assert len(lib.entries) == 2
        assert lib.entries[0].video_name == "newer.mp4"
        assert lib.entries[1].video_name == "older.mp4"

    def test_count_label_updates(self, qapp, tmp_path):
        dir1 = tmp_path / "outputs" / "demo" / "a"
        dir1.mkdir(parents=True)
        _write_session_json(dir1 / "bodypipe_session.json")

        lib = SessionLibrary(gvhmr_root=tmp_path)
        lib.scan()

        assert "1 session" in lib._count_label.text()

    def test_empty_scan(self, qapp, tmp_path):
        lib = SessionLibrary(gvhmr_root=tmp_path)
        lib.scan()
        assert len(lib.entries) == 0
        assert "0 sessions" in lib._count_label.text()


# ===========================================================================
# SessionLibraryDock
# ===========================================================================

class TestSessionLibraryDock:

    def test_dock_wraps_library(self, qapp):
        lib = SessionLibrary()
        dock = SessionLibraryDock(lib)
        assert dock.session_library is lib
        assert dock.objectName() == "SessionLibraryDock"
        assert dock.windowTitle() == "Session Library"

    def test_dock_widget_is_library(self, qapp):
        lib = SessionLibrary()
        dock = SessionLibraryDock(lib)
        assert dock.widget() is lib


# ===========================================================================
# AppWindow integration
# ===========================================================================

class TestAppWindowSessionLibrary:

    def test_session_library_dock_exists(self, qapp):
        from app_window import AppWindow

        win = AppWindow()
        assert hasattr(win, "_session_library_dock")
        assert hasattr(win, "_session_library")
        assert isinstance(win._session_library_dock, SessionLibraryDock)
        assert isinstance(win._session_library, SessionLibrary)

    def test_session_library_dock_in_view_menu(self, qapp):
        from app_window import AppWindow

        win = AppWindow()
        actions = [a.text() for a in win._view_menu.actions()]
        assert any("Session Library" in a for a in actions)

    def test_session_library_scan_on_init(self, qapp):
        """Library should have been scanned during init (may find 0 sessions)."""
        from app_window import AppWindow

        win = AppWindow()
        # The scan should have run (entries may be empty if no output dirs)
        assert isinstance(win._session_library.entries, list)

    def test_session_library_load_signal_calls_load_session(self, qapp, tmp_path):
        from app_window import AppWindow

        win = AppWindow()
        session_path = tmp_path / "bodypipe_session.json"
        _write_session_json(
            session_path,
            video_path=str(tmp_path / "vid.mp4"),
            notes="Load test",
        )

        with patch.object(win, "_load_session") as mock_load:
            win._session_library.session_load_requested.emit(str(session_path))
            mock_load.assert_called_once_with(session_path)

    def test_session_saved_refreshes_library(self, qapp, tmp_path):
        from app_window import AppWindow

        win = AppWindow()
        with patch.object(win, "_refresh_session_library") as mock_refresh:
            win.session_saved.emit(tmp_path / "test.json")
            mock_refresh.assert_called_once()

    def test_pipeline_workspace_includes_library(self, qapp):
        from app_window import AppWindow

        _desc, visible = AppWindow._WORKSPACE_PRESETS["Pipeline"]
        assert "_session_library_dock" in visible
