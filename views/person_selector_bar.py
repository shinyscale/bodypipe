"""Person selector bar — always-visible person switcher above Identity/Pose tabs.

Why: The person selector is the most fundamental control in multi-person mode,
determining which person's mesh, pose, identity, and corrections are viewed/edited.
Previously it was buried inside two tabified dock panels, easily missed.  This bar
sits above the tab widget so it's always visible regardless of which tab is active.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QWidget,
)
from PySide6.QtCore import Signal

from theme import COLORS, PERSON_COLORS


class PersonSelectorBar(QWidget):
    """Compact horizontal bar with color swatch, person combo, and show-all toggle."""

    person_changed = Signal(int)       # person_id
    show_all_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._session = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(6)

        # Color swatch
        self._swatch = QFrame()
        self._swatch.setFixedSize(14, 14)
        self._swatch.setStyleSheet(
            "border-radius: 3px; background: #666;"
        )
        layout.addWidget(self._swatch)

        # Person combo
        self._combo = QComboBox()
        self._combo.setMinimumWidth(120)
        self._combo.setStyleSheet("font-size: 12px;")
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        layout.addWidget(self._combo, 1)

        # Show all tracks checkbox
        self._show_all = QCheckBox("Show all tracks")
        self._show_all.toggled.connect(self.show_all_toggled)
        layout.addWidget(self._show_all)

        # Bar styling — accent bottom border
        self.setStyleSheet(
            f"PersonSelectorBar {{"
            f"  background: {COLORS['bg_active_tab']};"
            f"  border-bottom: 1px solid {COLORS['accent']};"
            f"}}"
        )

    def set_session(self, session):
        """Store session reference for person_tracks access."""
        self._session = session

    def refresh(self):
        """Rebuild combo from session.person_tracks (skip inactive)."""
        self._combo.blockSignals(True)
        self._combo.clear()
        if self._session:
            for pid in sorted(self._session.person_tracks.keys()):
                if pid not in self._session.inactive_tracks:
                    self._combo.addItem(f"Person {pid}", userData=pid)
        self._combo.blockSignals(False)
        self._update_swatch()

    def set_person(self, person_id: int):
        """Programmatic selection — blocks signals, updates swatch."""
        idx = self._combo.findData(person_id)
        if idx < 0:
            return
        self._combo.blockSignals(True)
        self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)
        self._update_swatch()

    def current_person(self) -> int:
        """Return the currently selected person_id, or -1."""
        data = self._combo.currentData()
        return data if data is not None else -1

    def _on_combo_changed(self, index: int):
        if index < 0:
            return
        person_id = self._combo.itemData(index)
        if person_id is None:
            return
        self._update_swatch()
        self.person_changed.emit(person_id)

    def _update_swatch(self):
        pid = self._combo.currentData()
        if pid is not None:
            color = PERSON_COLORS[pid % len(PERSON_COLORS)]
        else:
            color = "#666"
        self._swatch.setStyleSheet(
            f"border-radius: 3px; background: {color};"
        )
