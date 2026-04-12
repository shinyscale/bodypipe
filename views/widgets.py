"""Shared Qt widget helpers used by multiple panels.

``CollapsibleSection`` is promoted here from ``pose_corrector_panel.py`` so
the pipeline settings dock can reuse the same header-toggle section as the
pose corrector. Kept as a single module so the widget stays lightweight
and easy to import from anywhere in ``views``.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton, QVBoxLayout, QWidget

from theme import COLORS


class CollapsibleSection(QWidget):
    """Collapsible section with arrow toggle.

    Why: Settings panels accumulate many groups and scroll quickly fills
    the dock. A header-plus-arrow toggle lets the user hide groups they
    are not actively using while keeping them one click away. Uses
    ``QToolButton`` with an arrow-type indicator for a clean, consistent
    toggle mechanism shared across the pose corrector and pipeline
    settings panels.
    """

    def __init__(self, title: str, parent=None, collapsed: bool = False):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 2)
        main_layout.setSpacing(0)

        self._header = QToolButton()
        self._header.setArrowType(Qt.RightArrow if collapsed else Qt.DownArrow)
        self._header.setText(title)
        self._header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._header.setCheckable(True)
        self._header.setChecked(not collapsed)
        self._header.setStyleSheet(
            f"QToolButton {{ border: none; font-weight: bold; "
            f"color: {COLORS['text_primary']}; padding: 4px 2px; }}"
            f"QToolButton:hover {{ color: {COLORS['accent']}; }}"
        )
        main_layout.addWidget(self._header)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 2, 2, 2)
        self._content.setVisible(not collapsed)
        main_layout.addWidget(self._content)

        self._header.toggled.connect(self._on_toggled)

    def _on_toggled(self, checked: bool):
        self._content.setVisible(checked)
        self._header.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    @property
    def content_layout(self) -> QVBoxLayout:
        """Layout to add child widgets into."""
        return self._content_layout
