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
