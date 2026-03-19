"""Keyboard Shortcuts dialog — lists all app shortcuts in a searchable table."""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt


# Authoritative shortcut registry: (category, shortcut_key, description)
SHORTCUTS = [
    # File menu
    ("File", "Ctrl+O", "Open Video"),
    ("File", "Ctrl+Shift+O", "Open Session"),
    ("File", "Ctrl+S", "Save Session"),
    ("File", "Ctrl+Q", "Exit"),
    # Edit menu
    ("Edit", "Ctrl+Z", "Undo"),
    ("Edit", "Ctrl+Shift+Z", "Redo"),
    # View menu
    ("View", "Ctrl+L", "Toggle Log Panel"),
    ("View", "Ctrl+H", "Toggle HUD Overlay"),
    # Video playback
    ("Video", "Space", "Play / Pause"),
    ("Video", "Left", "Back 1 frame"),
    ("Video", "Right", "Forward 1 frame"),
    ("Video", "Ctrl+Left", "Back 10 frames"),
    ("Video", "Ctrl+Right", "Forward 10 frames"),
    ("Video", "Home", "First frame"),
    ("Video", "End", "Last frame"),
    # Interaction modes
    ("Mode", "1", "Navigate mode"),
    ("Mode", "2", "Select mode"),
    ("Mode", "3", "Correct mode"),
    ("Mode", "4", "Track mode"),
    # Navigate mode shortcuts
    ("Navigate", "W", "Orbit camera up"),
    ("Navigate", "A", "Orbit camera left"),
    ("Navigate", "S", "Orbit camera down"),
    ("Navigate", "D", "Orbit camera right"),
    ("Navigate", "G", "Go to frame"),
    # Select mode shortcuts
    ("Select", "Click", "Pick joint"),
    ("Select", "Shift+Click", "Add joint to selection"),
    ("Select", "G", "Go to frame"),
    ("Select", "Escape", "Deselect joint"),
    # Correct mode shortcuts
    ("Correct", "G", "Focus Pose Corrector"),
    ("Correct", "R", "Reset current joint"),
    ("Correct", "Escape", "Deselect joint"),
    # Track mode shortcuts
    ("Track", "G", "Next unreviewed keyframe"),
    ("Track", "Tab", "Next person"),
    ("Track", "Shift+Tab", "Previous person"),
]


class KeyboardShortcutsDialog(QDialog):
    """Modal dialog displaying all application keyboard shortcuts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(480, 400)
        self.resize(520, 450)

        layout = QVBoxLayout(self)

        # Table
        self._table = QTableWidget(len(SHORTCUTS), 3)
        self._table.setHorizontalHeaderLabels(["Category", "Shortcut", "Action"])
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)

        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        for row, (category, key, description) in enumerate(SHORTCUTS):
            cat_item = QTableWidgetItem(category)
            cat_item.setFlags(cat_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 0, cat_item)

            key_item = QTableWidgetItem(key)
            key_item.setFlags(key_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 1, key_item)

            desc_item = QTableWidgetItem(description)
            desc_item.setFlags(desc_item.flags() & ~Qt.ItemIsEditable)
            self._table.setItem(row, 2, desc_item)

        layout.addWidget(self._table)

        # Close button
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @property
    def table(self) -> QTableWidget:
        """Expose table for testing."""
        return self._table
