"""Tests for AppWindow session I/O, recent sessions, and menu wiring."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app_window import AppWindow, LogPanel
from models.pipeline_config import PipelineConfig
from models.session import Session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app_window(qapp):
    """Create an AppWindow instance for testing."""
    # Clear persisted settings before construction so each test
    # starts with clean defaults (prevents test ordering pollution).
    from PySide6.QtCore import QSettings

    settings = QSettings("GVHMR", "bodypipe")
    settings.remove("recent_sessions")
    settings.remove("pipeline_config/single")
    settings.remove("pipeline_config/perf")
    settings.remove("pipeline_config/multi")
    settings.remove("workspace")
    settings.remove("last_video_path")
    settings.sync()

    window = AppWindow()
    yield window
    # Force immediate destruction to prevent resource accumulation.
    # deleteLater() alone doesn't reclaim resources fast enough,
    # causing exponential slowdown after ~20 AppWindow instances.
    import shiboken6

    window.close()
    if shiboken6.isValid(window):
        shiboken6.delete(window)


@pytest.fixture
def tmp_session_file(tmp_path):
    """Create a temp session JSON file with known data."""
    session = Session(
        video_path=None,
        num_frames=100,
        fps=25.0,
        img_width=1920,
        img_height=1080,
        pipeline_mode="multi",
        static_cam=False,
        focal_mm=35.0,
    )
    path = tmp_path / "test_session.json"
    session.save(path)
    return path


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestAppWindowConstruction:
    """Verify AppWindow initializes all components correctly."""

    def test_window_title(self, app_window):
        assert "bodypipe" in app_window.windowTitle()

    def test_has_pipeline_dock_with_three_modes(self, app_window):
        assert app_window._pipeline_dock.mode_combo.count() == 3

    def test_has_session(self, app_window):
        assert isinstance(app_window.session, Session)

    def test_has_session_path_initially_none(self, app_window):
        assert app_window._session_path is None

    def test_has_log_panel(self, app_window):
        assert isinstance(app_window.log_panel, LogPanel)

    def test_has_status_bar(self, app_window):
        assert app_window._status_bar is not None

    def test_has_recent_menu(self, app_window):
        assert app_window._recent_menu is not None


# ---------------------------------------------------------------------------
# Menu Wiring
# ---------------------------------------------------------------------------


class TestMenuWiring:
    """Verify menu actions are connected to handlers."""

    def _find_action(self, window, menu_title, action_text):
        """Find a menu action by menu title and action text."""
        for action in window.menuBar().actions():
            if menu_title.replace("&", "") in action.text().replace("&", ""):
                menu = action.menu()
                if menu:
                    for a in menu.actions():
                        if action_text.replace("&", "") in a.text().replace("&", ""):
                            return a
        return None

    def test_open_session_connected(self, app_window):
        action = self._find_action(app_window, "File", "Open Session")
        assert action is not None
        assert action.shortcut().toString() == "Ctrl+Shift+O"

    def test_save_session_connected(self, app_window):
        action = self._find_action(app_window, "File", "Save Session")
        assert action is not None
        assert action.shortcut().toString() == "Ctrl+S"

    def test_about_connected(self, app_window):
        action = self._find_action(app_window, "Help", "About")
        assert action is not None

    def test_toggle_status_bar_exists(self, app_window):
        assert app_window._toggle_statusbar_action is not None
        assert app_window._toggle_statusbar_action.isCheckable()
        assert app_window._toggle_statusbar_action.isChecked()

    def test_toggle_log_panel_exists(self, app_window):
        assert app_window._toggle_log_action is not None
        assert app_window._toggle_log_action.isCheckable()

    def test_recent_sessions_menu_exists(self, app_window):
        """Recent Sessions appears as a submenu in File menu."""
        assert app_window._recent_menu is not None
        assert app_window._recent_menu.title() == "Recent Sessions"


# ---------------------------------------------------------------------------
# Session Save
# ---------------------------------------------------------------------------


class TestSessionSave:
    """Test File > Save Session functionality."""

    def test_save_with_session_path(self, app_window, tmp_path):
        """When _session_path is set, save directly without dialog."""
        save_path = tmp_path / "saved.json"
        app_window._session_path = save_path
        app_window._session.num_frames = 42
        app_window._session.fps = 24.0

        app_window._on_save_session()

        assert save_path.is_file()
        data = json.loads(save_path.read_text())
        assert data["num_frames"] == 42
        assert data["fps"] == 24.0

    def test_save_with_output_dir(self, app_window, tmp_path):
        """When output_dir is set but no session_path, save to output_dir."""
        app_window._session.output_dir = tmp_path
        app_window._session.num_frames = 99

        app_window._on_save_session()

        expected = tmp_path / "bodypipe_session.json"
        assert expected.is_file()
        data = json.loads(expected.read_text())
        assert data["num_frames"] == 99
        assert app_window._session_path == expected

    def test_save_emits_signal(self, app_window, tmp_path):
        """session_saved signal is emitted with the save path."""
        save_path = tmp_path / "sig.json"
        app_window._session_path = save_path
        received = []
        app_window.session_saved.connect(lambda p: received.append(p))

        app_window._on_save_session()

        assert len(received) == 1
        assert received[0] == save_path

    def test_save_updates_status(self, app_window, tmp_path):
        """Status bar shows save confirmation."""
        save_path = tmp_path / "status.json"
        app_window._session_path = save_path

        app_window._on_save_session()

        assert "status.json" in app_window._status_label.text()

    def test_save_adds_to_recent(self, app_window, tmp_path):
        """Saving adds the path to recent sessions."""
        save_path = tmp_path / "recent.json"
        app_window._session_path = save_path

        app_window._on_save_session()

        recent = app_window._get_recent()
        assert str(save_path) in recent

    @patch("app_window.QFileDialog.getSaveFileName", return_value=("", ""))
    def test_save_no_path_cancelled(self, mock_dialog, app_window):
        """When no path and dialog cancelled, nothing happens."""
        app_window._session_path = None
        app_window._session.output_dir = None

        app_window._on_save_session()

        assert app_window._session_path is None

    @patch("app_window.QFileDialog.getSaveFileName")
    def test_save_prompts_when_no_path(self, mock_dialog, app_window, tmp_path):
        """When no session_path or output_dir, prompt with file dialog."""
        save_path = tmp_path / "prompted.json"
        mock_dialog.return_value = (str(save_path), "")
        app_window._session_path = None
        app_window._session.output_dir = None
        app_window._session.num_frames = 77

        app_window._on_save_session()

        assert save_path.is_file()
        assert app_window._session_path == save_path


# ---------------------------------------------------------------------------
# Session Load
# ---------------------------------------------------------------------------


class TestSessionLoad:
    """Test File > Open Session functionality."""

    def test_load_session_updates_fields(self, app_window, tmp_session_file):
        """Loading a session updates the shared session object."""
        app_window._load_session(tmp_session_file)

        assert app_window._session.num_frames == 100
        assert app_window._session.fps == 25.0
        assert app_window._session.img_width == 1920
        assert app_window._session.pipeline_mode == "multi"
        assert app_window._session.static_cam is False
        assert app_window._session.focal_mm == 35.0

    def test_load_session_sets_session_path(self, app_window, tmp_session_file):
        """Loading sets _session_path for subsequent saves."""
        app_window._load_session(tmp_session_file)
        assert app_window._session_path == tmp_session_file

    def test_load_session_emits_signal(self, app_window, tmp_session_file):
        """session_loaded signal is emitted after load."""
        received = []
        app_window.session_loaded.connect(lambda s: received.append(s))

        app_window._load_session(tmp_session_file)

        assert len(received) == 1

    def test_load_session_updates_status(self, app_window, tmp_session_file):
        """Status bar shows load confirmation."""
        app_window._load_session(tmp_session_file)
        assert "test_session.json" in app_window._status_label.text()

    def test_load_session_adds_to_recent(self, app_window, tmp_session_file):
        """Loading adds the path to recent sessions."""
        app_window._load_session(tmp_session_file)
        recent = app_window._get_recent()
        assert str(tmp_session_file) in recent

    def test_load_invalid_json_shows_warning(self, app_window, tmp_path):
        """Loading invalid JSON shows warning without crashing."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json{{{")

        with patch("app_window.QMessageBox.warning") as mock_warn:
            app_window._load_session(bad_file)
            mock_warn.assert_called_once()

        # Session should be unchanged
        assert app_window._session_path is None

    def test_load_preserves_shared_reference(self, app_window, tmp_session_file):
        """Settings widgets should still see updates via shared session reference."""
        settings_session_ref = app_window._single_settings._session
        app_window._load_session(tmp_session_file)

        # The settings widget's reference should be the same object
        assert settings_session_ref is app_window._session
        # And it should have the loaded values
        assert settings_session_ref.num_frames == 100

    def test_round_trip_save_load(self, app_window, tmp_path):
        """Save then load produces equivalent session state."""
        app_window._session.num_frames = 256
        app_window._session.fps = 60.0
        app_window._session.pipeline_mode = "perf"
        app_window._session.focal_mm = 50.0

        save_path = tmp_path / "roundtrip.json"
        app_window._session_path = save_path
        app_window._on_save_session()

        # Reset session
        app_window._session.num_frames = 0
        app_window._session.fps = 30.0

        # Reload
        app_window._load_session(save_path)

        assert app_window._session.num_frames == 256
        assert app_window._session.fps == 60.0
        assert app_window._session.pipeline_mode == "perf"
        assert app_window._session.focal_mm == 50.0


# ---------------------------------------------------------------------------
# Recent Sessions
# ---------------------------------------------------------------------------


class TestRecentSessions:
    """Test Recent Sessions submenu management."""

    def test_empty_recent_shows_placeholder(self, app_window):
        """Empty recent list shows disabled placeholder action."""
        actions = app_window._recent_menu.actions()
        assert len(actions) == 1
        assert not actions[0].isEnabled()
        assert "No recent" in actions[0].text()

    def test_add_recent_populates_menu(self, app_window, tmp_path):
        """Adding a recent session populates the submenu."""
        path = tmp_path / "a.json"
        path.write_text("{}")
        app_window._add_recent(path)

        actions = app_window._recent_menu.actions()
        assert len(actions) == 1
        assert actions[0].isEnabled()
        assert "a.json" in actions[0].text()

    def test_recent_max_limit(self, app_window, tmp_path):
        """Recent list is capped at MAX_RECENT entries."""
        for i in range(10):
            app_window._add_recent(tmp_path / f"session_{i}.json")

        recent = app_window._get_recent()
        assert len(recent) == AppWindow.MAX_RECENT

    def test_recent_most_recent_first(self, app_window, tmp_path):
        """Most recently added session appears first."""
        app_window._add_recent(tmp_path / "old.json")
        app_window._add_recent(tmp_path / "new.json")

        recent = app_window._get_recent()
        assert "new.json" in recent[0]
        assert "old.json" in recent[1]

    def test_recent_deduplicates(self, app_window, tmp_path):
        """Adding same path twice moves it to front, no duplicate."""
        path = tmp_path / "dup.json"
        app_window._add_recent(tmp_path / "other.json")
        app_window._add_recent(path)
        app_window._add_recent(tmp_path / "other.json")

        recent = app_window._get_recent()
        assert recent.count(str(tmp_path / "other.json")) == 1
        assert recent.count(str(path)) == 1
        assert "other.json" in recent[0]  # most recent

    def test_open_recent_missing_file(self, app_window, tmp_path):
        """Opening a missing recent file shows warning and removes it."""
        missing = tmp_path / "missing.json"
        app_window._add_recent(missing)
        assert len(app_window._get_recent()) == 1

        with patch("app_window.QMessageBox.warning"):
            app_window._open_recent(str(missing))

        # Should be removed from recent
        assert len(app_window._get_recent()) == 0

    def test_open_recent_loads_session(self, app_window, tmp_session_file):
        """Opening a valid recent file loads the session."""
        app_window._add_recent(tmp_session_file)

        app_window._open_recent(str(tmp_session_file))

        assert app_window._session.num_frames == 100
        assert app_window._session_path == tmp_session_file


# ---------------------------------------------------------------------------
# Toggle Status Bar
# ---------------------------------------------------------------------------


class TestToggleStatusBar:
    """Test View > Toggle Status Bar."""

    def test_status_bar_toggle_hides(self, app_window):
        """Unchecking toggle hides the status bar."""
        app_window._toggle_statusbar_action.setChecked(False)
        assert app_window._status_bar.isHidden()

    def test_status_bar_toggle_shows(self, app_window):
        """Re-checking toggle shows the status bar."""
        app_window._toggle_statusbar_action.setChecked(False)
        app_window._toggle_statusbar_action.setChecked(True)
        # In offscreen mode, isHidden() is the reliable check
        assert not app_window._status_bar.isHidden()


# ---------------------------------------------------------------------------
# About Dialog
# ---------------------------------------------------------------------------


class TestAboutDialog:
    """Test Help > About dialog."""

    @patch("app_window.QMessageBox.about")
    def test_about_shows_dialog(self, mock_about, app_window):
        """About action opens a dialog with app name."""
        app_window._on_about()
        mock_about.assert_called_once()
        args = mock_about.call_args
        assert "bodypipe" in args[0][1]  # title
        assert "Motion Capture" in args[0][2]  # text


# ---------------------------------------------------------------------------
# Status and Logging
# ---------------------------------------------------------------------------


class TestStatusAndLogging:
    """Test status bar updates and log panel."""

    def test_set_status(self, app_window):
        app_window.set_status("Running GVHMR...")
        assert app_window._status_label.text() == "Running GVHMR..."

    def test_set_frame_info(self, app_window):
        app_window.set_frame_info(42, 100)
        assert "42" in app_window._frame_label.text()
        assert "100" in app_window._frame_label.text()

    def test_set_fps_info(self, app_window):
        app_window.set_fps_info(29.97)
        assert "29.97" in app_window._fps_label.text() or "30.0" in app_window._fps_label.text()

    def test_log_panel_append(self, app_window):
        app_window.log_panel.append_line("test message", "info")
        assert "test message" in app_window.log_panel.toPlainText()

    def test_log_panel_auto_detect_error(self, app_window):
        app_window.log_panel.append_stdout("RuntimeError: something broke")
        text = app_window.log_panel.toPlainText()
        assert "RuntimeError" in text


# ---------------------------------------------------------------------------
# Shared VideoPlayer
# ---------------------------------------------------------------------------


class TestSharedVideoPlayer:
    """AppWindow owns a single shared VideoPlayer in the VideoDock."""

    def test_has_shared_video_player(self, app_window):
        from views.video_player import VideoPlayer
        assert isinstance(app_window._video_player, VideoPlayer)

    def test_video_dock_wraps_shared_player(self, app_window):
        assert app_window._video_dock.video_player is app_window._video_player

    def test_has_mesh_viewport(self, app_window):
        from views.mesh_viewport import MeshViewport
        assert isinstance(app_window._mesh_viewport, MeshViewport)


# ---------------------------------------------------------------------------
# Status bar frame/FPS wiring
# ---------------------------------------------------------------------------


class TestStatusBarWiring:
    """Status bar updates from the shared video player frame changes."""

    def test_frame_change_updates_status_bar(self, app_window):
        """When video player emits frame_changed, status bar updates."""
        app_window._video_player._num_frames = 200
        app_window._video_player._fps = 24.0

        app_window._on_video_frame_changed(42)

        assert "42" in app_window._frame_label.text()
        assert "200" in app_window._frame_label.text()
        assert "24.0" in app_window._fps_label.text()

    def test_mode_switch_shows_multi_docks(self, app_window):
        """Switching to multi mode shows identity/pose/track docks."""
        app_window._pipeline_dock.set_mode("multi")

        assert not app_window._identity_dock.isHidden()
        assert not app_window._pose_corrector_dock.isHidden()
        assert not app_window._track_overview_dock.isHidden()

    def test_mode_switch_hides_multi_docks(self, app_window):
        """Switching back to single mode hides multi-only docks."""
        app_window._pipeline_dock.set_mode("multi")
        app_window._pipeline_dock.set_mode("single")

        assert app_window._identity_dock.isHidden()
        assert app_window._pose_corrector_dock.isHidden()
        assert app_window._track_overview_dock.isHidden()

    def test_mode_change_emits_tab_changed(self, app_window):
        """mode_changed emits backward-compat tab_changed signal."""
        received = []
        app_window.tab_changed.connect(received.append)
        app_window._pipeline_dock.set_mode("multi")
        assert received == [2]

    def test_mode_change_emits_mode_changed(self, app_window):
        """mode_changed signal carries the mode string."""
        received = []
        app_window.mode_changed.connect(received.append)
        app_window._pipeline_dock.set_mode("perf")
        assert received == ["perf"]


# ---------------------------------------------------------------------------
# Pipeline Config Persistence
# ---------------------------------------------------------------------------


class TestPipelineConfigPersistence:
    """Test pipeline settings save/restore across app close/reopen."""

    def test_save_writes_configs_to_settings(self, app_window):
        """_save_pipeline_configs writes JSON to QSettings for each tab."""
        # Modify single tab settings
        app_window._single_settings._static_cam.setChecked(False)
        app_window._single_settings._focal_mm.setValue(50.0)

        app_window._save_pipeline_configs()

        raw = app_window._settings.value("pipeline_config/single")
        assert raw is not None
        data = json.loads(raw)
        assert data["static_cam"] is False
        assert data["focal_mm"] == 50.0

    def test_save_writes_perf_config(self, app_window):
        """Perf capture tab settings are saved including hand/face options."""
        app_window._perf_settings._use_hands.setChecked(False)
        app_window._perf_settings._use_face.setChecked(True)

        app_window._save_pipeline_configs()

        raw = app_window._settings.value("pipeline_config/perf")
        data = json.loads(raw)
        assert data["use_hands"] is False
        assert data["use_face"] is True

    def test_save_writes_multi_config(self, app_window):
        """Multi-person tab settings are saved."""
        app_window._multi_settings._max_persons.setValue(4)
        app_window._multi_settings._confidence_threshold.setValue(0.7)

        app_window._save_pipeline_configs()

        raw = app_window._settings.value("pipeline_config/multi")
        data = json.loads(raw)
        assert data["max_persons"] == 4
        assert data["confidence_threshold"] == 0.7

    def test_restore_applies_saved_configs(self, app_window):
        """_restore_pipeline_configs reads QSettings and applies to tabs."""
        config = PipelineConfig(static_cam=False, focal_mm=85.0, use_dpvo=True)
        app_window._settings.setValue(
            "pipeline_config/single", json.dumps(config.to_dict())
        )

        app_window._restore_pipeline_configs()

        assert app_window._single_settings._static_cam.isChecked() is False
        assert app_window._single_settings._focal_mm.value() == 85.0
        assert app_window._single_settings._use_dpvo.isChecked() is True

    def test_restore_perf_config(self, app_window):
        """Perf tab config is restored from QSettings."""
        config = PipelineConfig(
            use_hands=False, use_face=True, hand_mode="smplestx_only",
            pitch_adjust=5.0, body_smooth_preset="heavy",
        )
        app_window._settings.setValue(
            "pipeline_config/perf", json.dumps(config.to_dict())
        )

        app_window._restore_pipeline_configs()

        assert app_window._perf_settings._use_hands.isChecked() is False
        assert app_window._perf_settings._use_face.isChecked() is True
        assert app_window._perf_settings._hand_smplestx.isChecked() is True
        assert app_window._perf_settings._pitch_adjust.value() == 5.0

    def test_restore_multi_config(self, app_window):
        """Multi-person tab config is restored from QSettings."""
        config = PipelineConfig(max_persons=3, confidence_threshold=0.8)
        app_window._settings.setValue(
            "pipeline_config/multi", json.dumps(config.to_dict())
        )

        app_window._restore_pipeline_configs()

        assert app_window._multi_settings._max_persons.value() == 3
        assert app_window._multi_settings._confidence_threshold.value() == 0.8

    def test_round_trip_single_tab(self, app_window):
        """Save then restore produces same settings on single tab."""
        app_window._single_settings._static_cam.setChecked(False)
        app_window._single_settings._use_dpvo.setChecked(True)
        app_window._single_settings._focal_mm.setValue(35.0)

        app_window._save_pipeline_configs()

        # Reset to defaults
        app_window._single_settings._static_cam.setChecked(True)
        app_window._single_settings._use_dpvo.setChecked(False)
        app_window._single_settings._focal_mm.setValue(24.0)

        app_window._restore_pipeline_configs()

        assert app_window._single_settings._static_cam.isChecked() is False
        assert app_window._single_settings._use_dpvo.isChecked() is True
        assert app_window._single_settings._focal_mm.value() == 35.0

    def test_round_trip_perf_tab(self, app_window):
        """Save then restore produces same settings on perf tab."""
        app_window._perf_settings._use_hands.setChecked(False)
        app_window._perf_settings._use_face.setChecked(True)
        app_window._perf_settings._target_fps.setValue(60.0)
        app_window._perf_settings._pitch_adjust.setValue(-10.0)

        app_window._save_pipeline_configs()

        # Reset
        app_window._perf_settings._use_hands.setChecked(True)
        app_window._perf_settings._use_face.setChecked(False)
        app_window._perf_settings._target_fps.setValue(30.0)
        app_window._perf_settings._pitch_adjust.setValue(0.0)

        app_window._restore_pipeline_configs()

        assert app_window._perf_settings._use_hands.isChecked() is False
        assert app_window._perf_settings._use_face.isChecked() is True
        assert app_window._perf_settings._target_fps.value() == 60.0
        assert app_window._perf_settings._pitch_adjust.value() == -10.0

    def test_restore_ignores_missing_settings(self, app_window):
        """No error when QSettings has no saved configs."""
        app_window._settings.remove("pipeline_config/single")
        app_window._settings.remove("pipeline_config/perf")
        app_window._settings.remove("pipeline_config/multi")

        # Set known state before restore
        app_window._single_settings._focal_mm.setValue(42.0)

        # Should not raise, and should not change widget state
        app_window._restore_pipeline_configs()

        # Widgets unchanged — restore is a no-op when settings are missing
        assert app_window._single_settings._focal_mm.value() == 42.0

    def test_restore_ignores_corrupt_json(self, app_window):
        """Corrupt JSON in QSettings is silently ignored."""
        app_window._settings.setValue("pipeline_config/single", "not valid json{{{")

        # Should not raise
        app_window._restore_pipeline_configs()

        # Defaults intact
        assert app_window._single_settings._static_cam.isChecked() is True

    def test_close_event_saves_configs(self, app_window):
        """closeEvent calls _save_pipeline_configs."""
        app_window._single_settings._focal_mm.setValue(100.0)

        # Simulate close
        from PySide6.QtGui import QCloseEvent
        event = QCloseEvent()
        app_window.closeEvent(event)

        raw = app_window._settings.value("pipeline_config/single")
        assert raw is not None
        data = json.loads(raw)
        assert data["focal_mm"] == 100.0

    def test_configs_restored_on_construction(self, qapp):
        """New AppWindow instance restores previously saved configs."""
        # First, save config via an existing window
        w1 = AppWindow()
        w1._single_settings._focal_mm.setValue(77.0)
        w1._single_settings._static_cam.setChecked(False)
        w1._save_pipeline_configs()

        # Create a new window — should restore
        w2 = AppWindow()

        assert w2._single_settings._focal_mm.value() == 77.0
        assert w2._single_settings._static_cam.isChecked() is False

        # Cleanup: remove the saved settings to not pollute other tests
        w2._settings.remove("pipeline_config/single")
        w2._settings.remove("pipeline_config/perf")
        w2._settings.remove("pipeline_config/multi")

        # Force immediate destruction to avoid resource accumulation
        import shiboken6

        w1.close()
        if shiboken6.isValid(w1):
            shiboken6.delete(w1)
        w2.close()
        if shiboken6.isValid(w2):
            shiboken6.delete(w2)


# ---------------------------------------------------------------------------
# Workspace Presets
# ---------------------------------------------------------------------------


class TestWorkspacePresets:
    """Test View > Workspace submenu and preset/custom layout management."""

    def test_workspace_submenu_exists(self, app_window):
        """Workspace submenu is present under View menu."""
        assert app_window._workspace_menu is not None
        assert app_window._workspace_menu.title() == "&Workspace"

    def test_workspace_menu_has_four_presets(self, app_window):
        """Four built-in presets appear as actions."""
        actions = [a for a in app_window._workspace_menu.actions()
                   if not a.isSeparator() and a.text() not in (
                       "Save Current Layout...", "Reset to Default")]
        assert len(actions) == 4
        names = [a.text().split("  —")[0] for a in actions]
        assert "Review" in names
        assert "Correction" in names
        assert "Tracking" in names
        assert "Pipeline" in names

    def test_workspace_menu_has_save_action(self, app_window):
        """'Save Current Layout...' action exists."""
        texts = [a.text() for a in app_window._workspace_menu.actions()]
        assert "Save Current Layout..." in texts

    def test_workspace_menu_has_reset_action(self, app_window):
        """'Reset to Default' action exists."""
        texts = [a.text() for a in app_window._workspace_menu.actions()]
        assert "Reset to Default" in texts

    def test_default_state_captured(self, app_window):
        """_default_state QByteArray is captured at init."""
        from PySide6.QtCore import QByteArray
        assert isinstance(app_window._default_state, QByteArray)
        assert len(app_window._default_state) > 0

    def test_apply_review_preset(self, app_window):
        """Review preset shows video, identity, track; hides mesh, pose."""
        app_window._apply_preset("Review")

        assert not app_window._video_dock.isHidden()
        assert not app_window._identity_dock.isHidden()
        assert not app_window._track_overview_dock.isHidden()
        assert app_window._mesh_dock.isHidden()
        assert app_window._pose_corrector_dock.isHidden()

    def test_apply_correction_preset(self, app_window):
        """Correction preset shows video, mesh, pose corrector; hides identity."""
        app_window._apply_preset("Correction")

        assert not app_window._video_dock.isHidden()
        assert not app_window._mesh_dock.isHidden()
        assert not app_window._pose_corrector_dock.isHidden()
        assert not app_window._track_overview_dock.isHidden()
        assert app_window._identity_dock.isHidden()

    def test_apply_tracking_preset(self, app_window):
        """Tracking preset shows video, identity, track; hides mesh, pose."""
        app_window._apply_preset("Tracking")

        assert not app_window._video_dock.isHidden()
        assert not app_window._identity_dock.isHidden()
        assert not app_window._track_overview_dock.isHidden()
        assert app_window._mesh_dock.isHidden()
        assert app_window._pose_corrector_dock.isHidden()

    def test_apply_pipeline_preset(self, app_window):
        """Pipeline preset shows video + pipeline + log; hides inspector docks."""
        app_window._apply_preset("Pipeline")

        assert not app_window._video_dock.isHidden()
        assert not app_window._pipeline_dock.isHidden()
        assert not app_window._log_dock.isHidden()
        assert app_window._mesh_dock.isHidden()
        assert app_window._identity_dock.isHidden()
        assert app_window._pose_corrector_dock.isHidden()
        assert app_window._track_overview_dock.isHidden()

    def test_pipeline_preset_hides_log_toggle_synced(self, app_window):
        """Pipeline preset sets the log toggle action to checked."""
        app_window._apply_preset("Pipeline")
        assert app_window._toggle_log_action.isChecked() is True

    def test_non_pipeline_preset_hides_log(self, app_window):
        """Non-Pipeline presets hide the log dock."""
        app_window._apply_preset("Review")
        assert app_window._log_dock.isHidden()

    def test_apply_preset_updates_status(self, app_window):
        """Applying a preset updates the status bar."""
        app_window._apply_preset("Review")
        assert "Review" in app_window._status_label.text()

    def test_reset_workspace_restores_default(self, app_window):
        """Reset to Default restores the initial dock layout."""
        # Apply a preset that hides some docks
        app_window._apply_preset("Pipeline")
        assert app_window._mesh_dock.isHidden()

        # Reset
        app_window._on_reset_workspace()

        # After reset, default layout is restored — video dock should be visible
        assert not app_window._video_dock.isHidden()
        assert "reset" in app_window._status_label.text().lower()

    @patch("app_window.QInputDialog.getText", return_value=("My Layout", True))
    def test_save_custom_workspace(self, mock_input, app_window):
        """Saving a custom workspace stores state in QSettings."""
        app_window._on_save_workspace()

        names = app_window._get_custom_workspace_names()
        assert "My Layout" in names
        # Verify the state bytes are stored
        state = app_window._settings.value("workspace/state/My Layout")
        assert state is not None
        assert "My Layout" in app_window._status_label.text()

    @patch("app_window.QInputDialog.getText", return_value=("", False))
    def test_save_custom_workspace_cancelled(self, mock_input, app_window):
        """Cancelling save dialog does not create a custom workspace."""
        app_window._on_save_workspace()
        names = app_window._get_custom_workspace_names()
        assert len(names) == 0

    @patch("app_window.QInputDialog.getText", return_value=("Test WS", True))
    def test_custom_workspace_appears_in_menu(self, mock_input, app_window):
        """After saving, custom workspace appears in the submenu."""
        app_window._on_save_workspace()

        texts = [a.text() for a in app_window._workspace_menu.actions()]
        assert "Test WS" in texts

    @patch("app_window.QInputDialog.getText", return_value=("Saved", True))
    def test_apply_custom_workspace(self, mock_input, app_window):
        """Restoring a custom workspace applies the saved state."""
        # Save current state as "Saved"
        app_window._on_save_workspace()

        # Apply a different preset
        app_window._apply_preset("Pipeline")
        assert app_window._mesh_dock.isHidden()

        # Restore the custom workspace
        app_window._apply_custom_workspace("Saved")

        assert "Saved" in app_window._status_label.text()

    def test_get_custom_workspace_names_empty(self, app_window):
        """No custom workspaces initially."""
        names = app_window._get_custom_workspace_names()
        assert names == []


class TestSpeedSync:
    """Phase 3: speed chips synced between video player and track overview.

    Why sync: the user may change playback speed in either the video overlay
    or the track timeline footer — both should stay in agreement.
    """

    def test_video_speed_updates_track_overview(self, app_window):
        """Changing speed in video player updates track overview chips."""
        app_window._video_player.speed_changed.emit(2.0)
        assert app_window._track_overview._speed_chips[2.0].isChecked()

    def test_track_overview_speed_updates_video_player(self, app_window):
        """Changing speed in track overview updates video player."""
        app_window._track_overview.speed_changed.emit(0.5)
        assert app_window._video_player._playback_speed == 0.5
        assert app_window._video_player._speed_chips[0.5].isChecked()

    def test_no_infinite_loop(self, app_window):
        """Bidirectional sync doesn't cause infinite recursion."""
        # Simulate actual chip click (not just signal emit) to exercise full path
        app_window._video_player._speed_chips[4.0].click()
        assert app_window._track_overview._speed_chips[4.0].isChecked()
        assert app_window._video_player._speed_chips[4.0].isChecked()
        assert app_window._video_player._playback_speed == 4.0


class TestScrubAutoSwitch:
    """Verify that scrubbing/playback auto-switches the viewport to wireframe.

    Why: The SMPL-X forward pass is expensive (~50ms per frame). During
    slider scrubbing or playback, this latency makes the viewport choppy.
    Auto-switching to wireframe (skeleton-only, FK-only) ensures smooth
    60fps frame updates, restoring full quality when the user stops.
    """

    def test_scrub_started_switches_viewport(self, qapp):
        """scrub_started signal should set viewport to wireframe."""
        from app_window import AppWindow
        from views.mesh_viewport import RenderMode

        window = AppWindow()
        assert window._mesh_viewport._render_mode is RenderMode.FULL
        window._video_player.scrub_started.emit()
        assert window._mesh_viewport._render_mode is RenderMode.WIREFRAME

    def test_scrub_ended_restores_viewport(self, qapp):
        """scrub_ended signal should restore viewport to previous mode."""
        from app_window import AppWindow
        from views.mesh_viewport import RenderMode

        window = AppWindow()
        window._video_player.scrub_started.emit()
        window._video_player.scrub_ended.emit()
        assert window._mesh_viewport._render_mode is RenderMode.FULL

    def test_playback_start_switches_viewport(self, qapp):
        """playback_toggled(True) should switch to wireframe."""
        from app_window import AppWindow
        from views.mesh_viewport import RenderMode

        window = AppWindow()
        window._video_player.playback_toggled.emit(True)
        assert window._mesh_viewport._render_mode is RenderMode.WIREFRAME

    def test_playback_stop_restores_viewport(self, qapp):
        """playback_toggled(False) should restore mode."""
        from app_window import AppWindow
        from views.mesh_viewport import RenderMode

        window = AppWindow()
        window._video_player.playback_toggled.emit(True)
        window._video_player.playback_toggled.emit(False)
        assert window._mesh_viewport._render_mode is RenderMode.FULL


# ---------------------------------------------------------------------------
# Dock data flow — startup restore and result loading
# ---------------------------------------------------------------------------


class TestVideoLoadSavesPath:
    """_on_video_loaded persists the path for startup restore."""

    def test_video_loaded_saves_path_to_qsettings(self, app_window, tmp_path):
        """When video_loaded signal fires, the path is saved in QSettings."""
        video = tmp_path / "test.mp4"
        video.touch()

        # Simulate settings panel metadata (normally set by _load_video)
        app_window._session.num_frames = 100
        app_window._session.fps = 30.0

        with patch.object(app_window._video_player, "set_video"):
            app_window._on_video_loaded(video)

        saved = app_window._settings.value("last_video_path")
        assert saved == str(video)

    def test_video_loaded_calls_try_restore(self, app_window, tmp_path):
        """_on_video_loaded triggers _try_restore_results."""
        video = tmp_path / "test.mp4"
        video.touch()
        app_window._session.num_frames = 100
        app_window._session.fps = 30.0

        with patch.object(app_window._video_player, "set_video"), \
             patch.object(app_window, "_try_restore_results") as mock_restore:
            app_window._on_video_loaded(video)

        mock_restore.assert_called_once_with(video)


class TestRestoreLastVideo:
    """_restore_last_video loads the previously-opened video on startup."""

    def test_no_saved_path_is_noop(self, app_window):
        """When no last_video_path in QSettings, nothing happens."""
        app_window._settings.remove("last_video_path")
        # Should not raise
        app_window._restore_last_video()

    def test_missing_file_is_noop(self, app_window, tmp_path):
        """When saved path points to a missing file, nothing happens."""
        app_window._settings.setValue("last_video_path", str(tmp_path / "gone.mp4"))
        with patch.object(app_window._pipeline_dock.current_settings,
                          "_load_video") as mock_load:
            app_window._restore_last_video()
        mock_load.assert_not_called()

    def test_valid_path_loads_video(self, app_window, tmp_path):
        """When saved path exists, it is loaded into the settings panel."""
        video = tmp_path / "test_restore.mp4"
        video.touch()
        app_window._settings.setValue("last_video_path", str(video))

        with patch.object(app_window._pipeline_dock.current_settings,
                          "_load_video") as mock_load:
            app_window._restore_last_video()
        mock_load.assert_called_once_with(str(video))


class TestTryRestoreResults:
    """_try_restore_results loads cached pipeline output when available."""

    def test_existing_session_tracks_triggers_hydrate(self, app_window, tmp_path):
        """When session already has tracks, hydrate + refresh (no disk scan)."""
        from models.session import PersonTrack

        pdir = tmp_path / "person_0"
        pdir.mkdir()
        app_window._session.person_tracks[0] = PersonTrack(
            person_id=0, person_dir=pdir,
        )
        with patch.object(app_window, "_hydrate_person_tracks") as mock_h, \
             patch.object(app_window, "_refresh_all_panels") as mock_r:
            app_window._try_restore_results(tmp_path / "video.mp4")
        mock_h.assert_called_once()
        mock_r.assert_called_once()

    def test_no_output_dir_is_noop(self, app_window, tmp_path):
        """When no output directory exists, nothing happens."""
        app_window._pipeline_dock.set_mode("multi")
        with patch.object(app_window, "_load_results_from_output_dir") as mock_load:
            app_window._try_restore_results(tmp_path / "nonexistent.mp4")
        mock_load.assert_not_called()

    def test_multi_mode_loads_from_output_dir(self, app_window, tmp_path):
        """In multi mode, scans output dir for person directories."""
        app_window._pipeline_dock.set_mode("multi")
        # Use tmp_path as gvhmr_root to avoid touching real output dirs
        app_window._gvhmr_root = tmp_path
        output_dir = tmp_path / "outputs" / "multi_person" / "testvid"
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch.object(app_window, "_load_results_from_output_dir") as mock_load:
            app_window._try_restore_results(Path("testvid.mp4"))
        mock_load.assert_called_once_with(output_dir)

    def test_single_mode_loads_preview_video(self, app_window, tmp_path):
        """In single mode, loads output preview video if available."""
        app_window._pipeline_dock.set_mode("single")
        # Use tmp_path as gvhmr_root to avoid touching real output dirs
        app_window._gvhmr_root = tmp_path
        output_dir = tmp_path / "outputs" / "demo" / "testvid2"
        output_dir.mkdir(parents=True, exist_ok=True)
        incam = output_dir / "incam.mp4"
        incam.touch()

        with patch.object(app_window, "_load_output_preview") as mock_preview:
            app_window._try_restore_results(Path("testvid2.mp4"))
        mock_preview.assert_called_once_with(incam)

    def test_uses_session_output_dir_if_set(self, app_window, tmp_path):
        """Prefers session.output_dir over mode-derived path."""
        out = tmp_path / "custom_output"
        out.mkdir()
        app_window._session.output_dir = out
        app_window._pipeline_dock.set_mode("multi")

        with patch.object(app_window, "_load_results_from_output_dir") as mock_load:
            app_window._try_restore_results(tmp_path / "any.mp4")
        mock_load.assert_called_once_with(out)


class TestHydratePersonTracks:
    """_hydrate_person_tracks fills heavy data from disk."""

    def test_loads_smplx_and_confidences(self, app_window, tmp_path):
        """Fills smplx_params and confidences from person_dir on disk."""
        from models.session import PersonTrack
        import csv

        pdir = tmp_path / "person_0"
        pdir.mkdir()

        # Write a minimal confidence.csv
        csv_path = pdir / "confidence.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "detection", "visible_kp", "bbox_overlap",
                "shape_dist", "motion_dist", "overall",
            ])
            writer.writeheader()
            writer.writerow({
                "detection": 0.9, "visible_kp": 0.8, "bbox_overlap": 0.1,
                "shape_dist": 0.05, "motion_dist": 0.02, "overall": 0.85,
            })

        pt = PersonTrack(person_id=0, person_dir=pdir)
        app_window._session.person_tracks[0] = pt

        app_window._hydrate_person_tracks()

        assert pt.confidences is not None
        assert len(pt.confidences) == 1
        assert abs(pt.confidences[0] - 0.85) < 0.01
        assert pt.confidence_breakdown is not None
        assert "overall" in pt.confidence_breakdown

    def test_skips_missing_person_dir(self, app_window):
        """Tracks with no person_dir or non-existent dir are skipped."""
        from models.session import PersonTrack

        pt = PersonTrack(person_id=0, person_dir=None)
        app_window._session.person_tracks[0] = pt
        # Should not raise
        app_window._hydrate_person_tracks()
        assert pt.smplx_params is None

    def test_does_not_overwrite_existing_data(self, app_window, tmp_path):
        """If smplx_params is already loaded, don't reload."""
        from models.session import PersonTrack

        pdir = tmp_path / "person_0"
        pdir.mkdir()

        existing_params = {"body_pose": "already_loaded"}
        pt = PersonTrack(
            person_id=0, person_dir=pdir,
            smplx_params=existing_params,
            confidences=[0.9],
        )
        app_window._session.person_tracks[0] = pt

        app_window._hydrate_person_tracks()

        # Should not have been overwritten
        assert pt.smplx_params is existing_params
        assert pt.confidences == [0.9]


class TestRefreshAllPanels:
    """_refresh_all_panels syncs all panels from session state."""

    def test_no_tracks_is_noop(self, app_window):
        """With no person_tracks, nothing crashes."""
        app_window._session.person_tracks.clear()
        app_window._refresh_all_panels()  # Should not raise

    def test_auto_selects_first_person(self, app_window):
        """When selected_person is -1, auto-selects the first person."""
        import numpy as np
        from models.session import PersonTrack

        app_window._session.person_tracks[3] = PersonTrack(
            person_id=3, confidences=[0.9, 0.8],
        )
        app_window._session.person_tracks[7] = PersonTrack(
            person_id=7, confidences=[0.7, 0.6],
        )
        app_window._session.selected_person = -1
        app_window._session.num_frames = 2

        app_window._pipeline_dock.set_mode("multi")
        app_window._refresh_all_panels()

        assert app_window._session.selected_person == 3

    def test_populates_track_overview(self, app_window):
        """Track overview gets lanes after refresh."""
        import numpy as np
        from models.session import PersonTrack

        app_window._session.person_tracks[0] = PersonTrack(
            person_id=0, confidences=[0.9, 0.8, 0.7],
        )
        app_window._session.num_frames = 3

        app_window._pipeline_dock.set_mode("multi")
        app_window._refresh_all_panels()

        assert len(app_window._track_overview._lanes) == 1

    def test_calls_identity_inspector_refresh(self, app_window):
        """Identity inspector is refreshed after panel refresh."""
        from models.session import PersonTrack

        app_window._session.person_tracks[0] = PersonTrack(
            person_id=0, confidences=[0.9],
        )
        app_window._session.num_frames = 1

        with patch.object(app_window._identity_inspector, "refresh") as mock_r:
            app_window._refresh_all_panels()
        mock_r.assert_called_once()

    def test_sets_person_on_all_panels(self, app_window):
        """Mesh viewport and pose corrector receive set_person."""
        from models.session import PersonTrack

        app_window._session.person_tracks[5] = PersonTrack(
            person_id=5, confidences=[0.9],
        )
        app_window._session.selected_person = 5
        app_window._session.num_frames = 1

        with patch.object(app_window._mesh_viewport, "set_person") as mock_mv, \
             patch.object(app_window._pose_corrector, "set_person") as mock_pc:
            app_window._refresh_all_panels()
        mock_mv.assert_called_with(5)
        mock_pc.assert_called_with(5)


class TestLoadResultsFromOutputDir:
    """_load_results_from_output_dir scans person directories on disk."""

    def test_creates_tracks_from_person_dirs(self, app_window, tmp_path):
        """Person directories are discovered and loaded into session."""
        import csv

        for i in range(2):
            pdir = tmp_path / f"person_{i}"
            pdir.mkdir()
            csv_path = pdir / "confidence.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "detection", "visible_kp", "bbox_overlap",
                    "shape_dist", "motion_dist", "overall",
                ])
                writer.writeheader()
                writer.writerow({
                    "detection": 0.9, "visible_kp": 0.8, "bbox_overlap": 0.1,
                    "shape_dist": 0.05, "motion_dist": 0.02, "overall": 0.85,
                })

        app_window._session.num_frames = 1
        app_window._pipeline_dock.set_mode("multi")
        app_window._load_results_from_output_dir(tmp_path)

        assert 0 in app_window._session.person_tracks
        assert 1 in app_window._session.person_tracks
        assert app_window._session.person_tracks[0].person_dir == tmp_path / "person_0"

    def test_skips_non_person_dirs(self, app_window, tmp_path):
        """Directories not matching person_N pattern are ignored."""
        (tmp_path / "logs").mkdir()
        (tmp_path / "person_0").mkdir()
        (tmp_path / "config.json").touch()

        app_window._session.num_frames = 1
        app_window._pipeline_dock.set_mode("multi")
        app_window._load_results_from_output_dir(tmp_path)

        assert len(app_window._session.person_tracks) == 1

    def test_empty_output_dir_is_noop(self, app_window, tmp_path):
        """No person directories means no tracks loaded."""
        app_window._session.num_frames = 1
        app_window._pipeline_dock.set_mode("multi")
        app_window._load_results_from_output_dir(tmp_path)

        assert len(app_window._session.person_tracks) == 0

    def test_loads_crossing_spans(self, app_window, tmp_path):
        """Crossing spans JSON files are loaded from person dirs."""
        pdir = tmp_path / "person_0"
        pdir.mkdir()

        spans = [[10, 20], [50, 60]]
        (pdir / "crossing_spans.json").write_text(json.dumps(spans))

        app_window._session.num_frames = 100
        app_window._pipeline_dock.set_mode("multi")
        app_window._load_results_from_output_dir(tmp_path)

        assert 0 in app_window._session.crossing_spans
        assert app_window._session.crossing_spans[0] == [(10, 20), (50, 60)]


class TestMultiPipelineFinishedUsesRefresh:
    """_on_multi_pipeline_finished uses _refresh_all_panels."""

    def test_refresh_called_after_pipeline(self, app_window):
        """After multi pipeline finishes, _refresh_all_panels is called."""
        with patch.object(app_window, "_refresh_all_panels") as mock_r, \
             patch.object(app_window, "_load_person_tracks_from_result"):
            app_window._on_multi_pipeline_finished({"output_dir": "/tmp/out"})
        mock_r.assert_called_once()


class TestSessionLoadRefreshesData:
    """Loading a session triggers hydration and panel refresh."""

    def test_session_with_tracks_triggers_hydrate_via_video_load(
        self, app_window, tmp_path,
    ):
        """When a session has person tracks and a video, hydration occurs."""
        from models.session import PersonTrack

        # Create a session with a person track
        pdir = tmp_path / "person_0"
        pdir.mkdir()
        session = Session(
            video_path=tmp_path / "video.mp4",
            num_frames=50,
            fps=30.0,
            output_dir=tmp_path,
        )
        session.person_tracks[0] = PersonTrack(
            person_id=0, person_dir=pdir,
        )
        session_path = tmp_path / "session.json"
        session.save(session_path)

        # Create the video file so _load_video gets called
        (tmp_path / "video.mp4").touch()

        with patch.object(app_window._pipeline_dock.current_settings,
                          "_load_video") as mock_load:
            app_window._load_session(session_path)

        # Video should have been loaded
        mock_load.assert_called_once()
        # Session path should be set
        assert app_window._session_path == session_path


# ---------------------------------------------------------------------------
# Phase 10: Interaction modes, HUD, keyboard shortcuts
# ---------------------------------------------------------------------------


class TestInteractionModeEnum:
    """Verify InteractionMode enum structure."""

    def test_four_modes(self):
        from app_window import InteractionMode
        assert len(InteractionMode) == 4

    def test_mode_values(self):
        from app_window import InteractionMode
        assert InteractionMode.NAVIGATE.value == "Navigate"
        assert InteractionMode.SELECT.value == "Select"
        assert InteractionMode.CORRECT.value == "Correct"
        assert InteractionMode.TRACK.value == "Track"


class TestInteractionModeManager:
    """Verify mode switching and signal emission."""

    def test_default_mode_is_navigate(self, app_window):
        from app_window import InteractionMode
        assert app_window._interaction_mode == InteractionMode.NAVIGATE

    def test_set_mode_changes_state(self, app_window):
        from app_window import InteractionMode
        app_window.set_interaction_mode(InteractionMode.SELECT)
        assert app_window._interaction_mode == InteractionMode.SELECT

    def test_set_mode_emits_signal(self, app_window):
        from app_window import InteractionMode
        received = []
        app_window.interaction_mode_changed.connect(received.append)
        app_window.set_interaction_mode(InteractionMode.CORRECT)
        assert received == ["Correct"]

    def test_set_same_mode_no_signal(self, app_window):
        from app_window import InteractionMode
        received = []
        app_window.interaction_mode_changed.connect(received.append)
        # Already in NAVIGATE, setting again should not emit
        app_window.set_interaction_mode(InteractionMode.NAVIGATE)
        assert received == []

    def test_mode_label_exists(self, app_window):
        assert hasattr(app_window, "_mode_label")
        assert app_window._mode_label.text() != ""

    def test_mode_label_updates_on_switch(self, app_window):
        from app_window import InteractionMode
        app_window.set_interaction_mode(InteractionMode.TRACK)
        assert "Track" in app_window._mode_label.text()

    def test_mode_label_contains_number(self, app_window):
        """Mode pill shows the mode number for quick reference."""
        from app_window import InteractionMode
        app_window.set_interaction_mode(InteractionMode.CORRECT)
        # Correct is mode #3
        assert "#3" in app_window._mode_label.text()


class TestModeKeyboardShortcuts:
    """Verify mode switching via 1-4 number keys."""

    def test_key_1_navigate(self, app_window):
        from app_window import InteractionMode
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        # Start in a different mode
        app_window.set_interaction_mode(InteractionMode.SELECT)
        event = QKeyEvent(QEvent.KeyPress, Qt.Key_1, Qt.NoModifier)
        app_window.keyPressEvent(event)
        assert app_window._interaction_mode == InteractionMode.NAVIGATE

    def test_key_2_select(self, app_window):
        from app_window import InteractionMode
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        event = QKeyEvent(QEvent.KeyPress, Qt.Key_2, Qt.NoModifier)
        app_window.keyPressEvent(event)
        assert app_window._interaction_mode == InteractionMode.SELECT

    def test_key_3_correct(self, app_window):
        from app_window import InteractionMode
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        event = QKeyEvent(QEvent.KeyPress, Qt.Key_3, Qt.NoModifier)
        app_window.keyPressEvent(event)
        assert app_window._interaction_mode == InteractionMode.CORRECT

    def test_key_4_track(self, app_window):
        from app_window import InteractionMode
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        event = QKeyEvent(QEvent.KeyPress, Qt.Key_4, Qt.NoModifier)
        app_window.keyPressEvent(event)
        assert app_window._interaction_mode == InteractionMode.TRACK


class TestNavigateModeKeys:
    """Verify Navigate mode keyboard shortcuts."""

    def test_wasd_switches_to_orbit(self, app_window):
        """WASD keys switch viewport to orbit mode if not already."""
        from app_window import InteractionMode
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        app_window.set_interaction_mode(InteractionMode.NAVIGATE)
        app_window._mesh_viewport._camera_mode = "incam"

        event = QKeyEvent(QEvent.KeyPress, Qt.Key_W, Qt.NoModifier)
        app_window.keyPressEvent(event)
        assert app_window._mesh_viewport._camera_mode == "orbit"

    def test_g_key_navigate_mode(self, app_window):
        """G key in navigate mode should open go-to-frame dialog."""
        from app_window import InteractionMode
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        app_window.set_interaction_mode(InteractionMode.NAVIGATE)
        # Set up video player so seek works
        app_window._video_player._num_frames = 200

        with patch("app_window.QInputDialog.getInt", return_value=(42, True)):
            event = QKeyEvent(QEvent.KeyPress, Qt.Key_G, Qt.NoModifier)
            app_window.keyPressEvent(event)
        # Should have sought to frame 42
        assert app_window._video_player._current_frame == 42


class TestCorrectModeKeys:
    """Verify Correct mode keyboard shortcuts."""

    def test_g_key_shows_pose_corrector(self, app_window):
        """G key in correct mode calls setVisible(True) and raise_() on the dock."""
        from app_window import InteractionMode
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        app_window.set_interaction_mode(InteractionMode.CORRECT)

        with patch.object(app_window._pose_corrector_dock, "setVisible") as mock_vis, \
             patch.object(app_window._pose_corrector_dock, "raise_") as mock_raise:
            event = QKeyEvent(QEvent.KeyPress, Qt.Key_G, Qt.NoModifier)
            app_window.keyPressEvent(event)
        mock_vis.assert_called_once_with(True)
        mock_raise.assert_called_once()

    def test_r_key_resets_joint(self, app_window):
        """R key in correct mode calls reset_current_joint."""
        from app_window import InteractionMode
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        app_window.set_interaction_mode(InteractionMode.CORRECT)

        with patch.object(app_window._pose_corrector, "reset_current_joint") as mock_reset:
            event = QKeyEvent(QEvent.KeyPress, Qt.Key_R, Qt.NoModifier)
            app_window.keyPressEvent(event)
        mock_reset.assert_called_once()


class TestTrackModeKeys:
    """Verify Track mode keyboard shortcuts."""

    def test_g_key_next_unreviewed(self, app_window):
        """G key in track mode calls go_to_next_unreviewed."""
        from app_window import InteractionMode
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        app_window.set_interaction_mode(InteractionMode.TRACK)

        with patch.object(
            app_window._identity_inspector, "go_to_next_unreviewed"
        ) as mock_go:
            event = QKeyEvent(QEvent.KeyPress, Qt.Key_G, Qt.NoModifier)
            app_window.keyPressEvent(event)
        mock_go.assert_called_once()

    def test_tab_cycles_person(self, app_window):
        """Tab key in track mode cycles to next person."""
        from app_window import InteractionMode
        from models.session import PersonTrack
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        app_window.set_interaction_mode(InteractionMode.TRACK)

        # Set up two person tracks
        app_window._session.person_tracks[0] = PersonTrack(person_id=0)
        app_window._session.person_tracks[1] = PersonTrack(person_id=1)
        app_window._session.selected_person = 0

        event = QKeyEvent(QEvent.KeyPress, Qt.Key_Tab, Qt.NoModifier)
        app_window.keyPressEvent(event)
        assert app_window._session.selected_person == 1

    def test_shift_tab_cycles_person_backwards(self, app_window):
        """Shift+Tab in track mode cycles to previous person."""
        from app_window import InteractionMode
        from models.session import PersonTrack
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        app_window.set_interaction_mode(InteractionMode.TRACK)

        app_window._session.person_tracks[0] = PersonTrack(person_id=0)
        app_window._session.person_tracks[1] = PersonTrack(person_id=1)
        app_window._session.selected_person = 0

        event = QKeyEvent(QEvent.KeyPress, Qt.Key_Tab, Qt.ShiftModifier)
        app_window.keyPressEvent(event)
        # Should wrap to last person
        assert app_window._session.selected_person == 1

    def test_cycle_person_no_tracks(self, app_window):
        """Cycling persons with empty tracks is a no-op."""
        from app_window import InteractionMode
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtCore import QEvent

        app_window.set_interaction_mode(InteractionMode.TRACK)
        app_window._session.person_tracks.clear()
        app_window._session.selected_person = -1

        event = QKeyEvent(QEvent.KeyPress, Qt.Key_Tab, Qt.NoModifier)
        app_window.keyPressEvent(event)
        # Should not crash
        assert app_window._session.selected_person == -1


class TestViewportHUD:
    """Verify HUD overlay on mesh viewport."""

    def test_hud_exists(self, app_window):
        assert hasattr(app_window._mesh_viewport, "_hud")

    def test_hud_enabled_by_default(self, app_window):
        assert app_window._mesh_viewport._hud_enabled is True

    def test_set_hud_visible_enables(self, app_window):
        app_window._mesh_viewport.set_hud_visible(True)
        assert app_window._mesh_viewport._hud_enabled is True

    def test_set_hud_visible_false_disables(self, app_window):
        app_window._mesh_viewport.set_hud_visible(True)
        app_window._mesh_viewport.set_hud_visible(False)
        assert app_window._mesh_viewport._hud_enabled is False

    def test_set_hud_mode_updates_text(self, app_window):
        app_window._mesh_viewport.set_hud_mode("Correct")
        assert app_window._mesh_viewport._hud._mode_text == "Correct"

    def test_update_hud_frame(self, app_window):
        app_window._mesh_viewport.update_hud(frame=42, total_frames=100)
        assert "42" in app_window._mesh_viewport._hud._frame_text
        assert "100" in app_window._mesh_viewport._hud._frame_text

    def test_update_hud_speed(self, app_window):
        app_window._mesh_viewport.update_hud(speed=2.0)
        assert "2x" in app_window._mesh_viewport._hud._speed_text

    def test_update_hud_person(self, app_window):
        app_window._mesh_viewport.update_hud(person=3)
        assert "3" in app_window._mesh_viewport._hud._person_text

    def test_hud_toggle_in_view_menu(self, app_window):
        """View menu has Toggle HUD Overlay action."""
        assert hasattr(app_window, "_toggle_hud_action")
        assert app_window._toggle_hud_action.isCheckable()

    def test_hud_toggle_disables_hud(self, app_window):
        """Unchecking HUD toggle disables the HUD overlay."""
        app_window._toggle_hud_action.setChecked(False)
        assert app_window._mesh_viewport._hud_enabled is False

    def test_hud_toggle_enables_hud(self, app_window):
        """Checking HUD toggle enables the HUD overlay."""
        app_window._toggle_hud_action.setChecked(False)
        app_window._toggle_hud_action.setChecked(True)
        assert app_window._mesh_viewport._hud_enabled is True


class TestKeyboardShortcutsDialog:
    """Verify the shortcuts dialog includes Phase 10 entries."""

    def test_mode_shortcuts_present(self):
        from views.keyboard_shortcuts_dialog import SHORTCUTS
        categories = {s[0] for s in SHORTCUTS}
        assert "Mode" in categories
        assert "Navigate" in categories
        assert "Select" in categories
        assert "Correct" in categories
        assert "Track" in categories

    def test_hud_toggle_shortcut(self):
        from views.keyboard_shortcuts_dialog import SHORTCUTS
        hud = [s for s in SHORTCUTS if "HUD" in s[2]]
        assert len(hud) == 1
        assert hud[0][1] == "Ctrl+H"

    def test_mode_switch_keys_listed(self):
        from views.keyboard_shortcuts_dialog import SHORTCUTS
        mode_entries = [s for s in SHORTCUTS if s[0] == "Mode"]
        keys = {s[1] for s in mode_entries}
        assert keys == {"1", "2", "3", "4"}

    def test_navigate_wasd_listed(self):
        from views.keyboard_shortcuts_dialog import SHORTCUTS
        nav_entries = [s for s in SHORTCUTS if s[0] == "Navigate"]
        keys = {s[1] for s in nav_entries}
        assert {"W", "A", "S", "D", "G"} <= keys

    def test_track_tab_listed(self):
        from views.keyboard_shortcuts_dialog import SHORTCUTS
        track_entries = [s for s in SHORTCUTS if s[0] == "Track"]
        keys = {s[1] for s in track_entries}
        assert "Tab" in keys
        assert "Shift+Tab" in keys


class TestGoToFrameDialog:
    """Verify go-to-frame dialog integration."""

    def test_go_to_frame_seeks(self, app_window):
        """Go to frame dialog seeks when user enters a frame number."""
        app_window._video_player._num_frames = 200
        with patch("app_window.QInputDialog.getInt", return_value=(99, True)):
            app_window._go_to_frame_dialog()
        assert app_window._video_player._current_frame == 99

    def test_go_to_frame_cancelled(self, app_window):
        """Go to frame dialog does nothing when cancelled."""
        app_window._video_player._num_frames = 200
        app_window._video_player._current_frame = 50
        with patch("app_window.QInputDialog.getInt", return_value=(0, False)):
            app_window._go_to_frame_dialog()
        assert app_window._video_player._current_frame == 50
