"""Tests for KeyboardShortcutsDialog and Help menu wiring."""

from unittest.mock import patch

import pytest

from views.keyboard_shortcuts_dialog import KeyboardShortcutsDialog, SHORTCUTS
from app_window import AppWindow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dialog(qapp):
    """Create a KeyboardShortcutsDialog instance."""
    dlg = KeyboardShortcutsDialog()
    yield dlg


@pytest.fixture
def app_window(qapp):
    """Create an AppWindow instance."""
    window = AppWindow()
    yield window


# ---------------------------------------------------------------------------
# Dialog Construction
# ---------------------------------------------------------------------------


class TestDialogConstruction:
    """Verify dialog initializes correctly."""

    def test_window_title(self, dialog):
        assert dialog.windowTitle() == "Keyboard Shortcuts"

    def test_table_row_count_matches_shortcuts(self, dialog):
        assert dialog.table.rowCount() == len(SHORTCUTS)

    def test_table_has_three_columns(self, dialog):
        assert dialog.table.columnCount() == 3

    def test_table_headers(self, dialog):
        headers = [
            dialog.table.horizontalHeaderItem(i).text()
            for i in range(3)
        ]
        assert headers == ["Category", "Shortcut", "Action"]

    def test_table_not_editable(self, dialog):
        from PySide6.QtWidgets import QAbstractItemView

        assert dialog.table.editTriggers() == QAbstractItemView.NoEditTriggers

    def test_table_vertical_header_hidden(self, dialog):
        assert dialog.table.verticalHeader().isHidden()


# ---------------------------------------------------------------------------
# Shortcut Content
# ---------------------------------------------------------------------------


class TestShortcutContent:
    """Verify the SHORTCUTS data and table population."""

    def test_shortcuts_not_empty(self):
        assert len(SHORTCUTS) > 0

    def test_all_shortcuts_have_three_fields(self):
        for entry in SHORTCUTS:
            assert len(entry) == 3, f"Bad entry: {entry}"

    def test_categories_present(self):
        categories = {s[0] for s in SHORTCUTS}
        assert "File" in categories
        assert "Edit" in categories
        assert "View" in categories
        assert "Video" in categories

    def test_table_content_matches_shortcuts(self, dialog):
        for row, (cat, key, desc) in enumerate(SHORTCUTS):
            assert dialog.table.item(row, 0).text() == cat
            assert dialog.table.item(row, 1).text() == key
            assert dialog.table.item(row, 2).text() == desc

    def test_file_shortcuts(self):
        file_shortcuts = [(k, d) for c, k, d in SHORTCUTS if c == "File"]
        keys = [k for k, _ in file_shortcuts]
        assert "Ctrl+O" in keys
        assert "Ctrl+Shift+O" in keys
        assert "Ctrl+S" in keys
        assert "Ctrl+Q" in keys

    def test_edit_shortcuts(self):
        edit_shortcuts = [(k, d) for c, k, d in SHORTCUTS if c == "Edit"]
        keys = [k for k, _ in edit_shortcuts]
        assert "Ctrl+Z" in keys
        assert "Ctrl+Shift+Z" in keys

    def test_video_shortcuts(self):
        video_shortcuts = [(k, d) for c, k, d in SHORTCUTS if c == "Video"]
        keys = [k for k, _ in video_shortcuts]
        assert "Space" in keys
        assert "Left" in keys
        assert "Right" in keys
        assert "Ctrl+Left" in keys
        assert "Ctrl+Right" in keys
        assert "Home" in keys
        assert "End" in keys


# ---------------------------------------------------------------------------
# Menu Wiring
# ---------------------------------------------------------------------------


class TestMenuWiring:
    """Verify Help > Keyboard Shortcuts action exists and works."""

    def _find_action(self, window, menu_title, action_text):
        for action in window.menuBar().actions():
            if menu_title.replace("&", "") in action.text().replace("&", ""):
                menu = action.menu()
                if menu:
                    for a in menu.actions():
                        if action_text.replace("&", "") in a.text().replace("&", ""):
                            return a
        return None

    def test_keyboard_shortcuts_action_exists(self, app_window):
        action = self._find_action(app_window, "Help", "Keyboard Shortcuts")
        assert action is not None

    def test_keyboard_shortcuts_action_triggers(self, app_window):
        """Triggering the action creates the dialog (mocked exec)."""
        with patch.object(KeyboardShortcutsDialog, "exec") as mock_exec:
            app_window._on_keyboard_shortcuts()
            mock_exec.assert_called_once()
