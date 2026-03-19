"""Session Library — media-pool-style panel for browsing saved sessions.

Why: Users accumulate multiple capture sessions over time.  Without a library,
they must remember file paths or use File > Recent.  The session library provides
a visual overview with thumbnails, metadata, and tags for quick identification
and loading of any previous session.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QPixmap, QImage

from theme import COLORS

log = logging.getLogger(__name__)

THUMBNAIL_HEIGHT = 80
THUMBNAIL_CACHE_NAME = ".bodypipe_thumb.png"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class SessionEntry:
    """Metadata for a single saved session."""

    path: Path
    video_name: str = ""
    duration_sec: float = 0.0
    fps: float = 30.0
    num_frames: int = 0
    person_count: int = 0
    correction_count: int = 0
    date: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    version: int = 1
    thumbnail: QPixmap | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_thumbnail(video_path: Path, session_dir: Path) -> QPixmap | None:
    """Extract first frame from video and cache as a small PNG.

    Returns a QPixmap scaled to THUMBNAIL_HEIGHT, or None on failure.
    Caches the result alongside the session JSON so subsequent scans
    skip the video read.
    """
    cache_path = session_dir / THUMBNAIL_CACHE_NAME

    # Try cached thumbnail first
    if cache_path.is_file():
        pix = QPixmap(str(cache_path))
        if not pix.isNull():
            return pix

    if not video_path.is_file():
        return None

    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None

        import numpy as np  # noqa: F811

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = THUMBNAIL_HEIGHT / max(h, 1)
        new_w = max(1, int(w * scale))
        new_h = THUMBNAIL_HEIGHT
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        qimg = QImage(rgb.data, new_w, new_h, new_w * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())  # detach from numpy buffer

        # Persist to disk
        try:
            pix.save(str(cache_path), "PNG")
        except Exception:
            pass

        return pix
    except Exception:
        log.debug("Thumbnail generation failed for %s", video_path, exc_info=True)
        return None


def _load_session_entry(session_path: Path) -> SessionEntry | None:
    """Load lightweight session metadata from a JSON file.

    Why lightweight: We only need the summary fields (video name, person count,
    tags, etc.) for the library card — not the heavy SMPL-X params or per-frame
    confidences that live on disk.
    """
    try:
        with open(session_path) as f:
            data = json.load(f)
    except Exception:
        return None

    video_path = Path(data["video_path"]) if data.get("video_path") else None
    video_name = video_path.name if video_path else "(no video)"

    num_frames = data.get("num_frames", 0)
    fps = data.get("fps", 30.0)
    duration_sec = num_frames / fps if fps > 0 else 0.0

    person_tracks = data.get("person_tracks", {})
    person_count = len(person_tracks)

    # Count keyframes across all person tracks as a proxy for corrections
    correction_count = 0
    for pt in person_tracks.values():
        correction_count += len(pt.get("keyframes", []))

    # File modification date
    try:
        mtime = os.path.getmtime(session_path)
        date = datetime.fromtimestamp(mtime)
    except Exception:
        date = datetime.now()

    tags = data.get("tags", [])
    notes = data.get("notes", "")
    version = data.get("version", 1)

    # Generate thumbnail (gracefully fails in headless/test environments)
    thumbnail = None
    if video_path:
        thumbnail = _generate_thumbnail(video_path, session_path.parent)

    return SessionEntry(
        path=session_path,
        video_name=video_name,
        duration_sec=duration_sec,
        fps=fps,
        num_frames=num_frames,
        person_count=person_count,
        correction_count=correction_count,
        date=date,
        tags=list(tags),
        notes=notes,
        version=version,
        thumbnail=thumbnail,
    )


def _format_duration(seconds: float) -> str:
    """Format duration as MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


# ---------------------------------------------------------------------------
# Session card widget
# ---------------------------------------------------------------------------

class _SessionCard(QWidget):
    """A single session entry card with thumbnail and metadata.

    Why custom widget: QListWidgetItem only supports text/icon.  A card
    with thumbnail, multi-line metadata, and tag chips gives users the
    at-a-glance overview needed to quickly find the right session.
    """

    TAG_COLORS = [
        "#3d85c6", "#6aa84f", "#cc4125", "#e69138",
        "#8e7cc3", "#c27ba0", "#76a5af", "#f1c232",
    ]

    def __init__(self, entry: SessionEntry, parent=None):
        super().__init__(parent)
        self._entry = entry
        self._setup_ui()

    @property
    def entry(self) -> SessionEntry:
        return self._entry

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        # --- Thumbnail ---
        thumb_w = THUMBNAIL_HEIGHT * 16 // 9  # ~142px for 16:9 aspect
        thumb_label = QLabel()
        thumb_label.setFixedSize(thumb_w, THUMBNAIL_HEIGHT)
        thumb_label.setStyleSheet(
            f"QLabel {{ background: {COLORS['bg_input']}; "
            f"border: 1px solid {COLORS['border']}; }}"
        )
        thumb_label.setAlignment(Qt.AlignCenter)
        if self._entry.thumbnail and not self._entry.thumbnail.isNull():
            scaled = self._entry.thumbnail.scaled(
                thumb_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
            thumb_label.setPixmap(scaled)
        else:
            thumb_label.setText("No\nPreview")
            thumb_label.setStyleSheet(
                thumb_label.styleSheet()
                + f" color: {COLORS['text_disabled']}; font-size: 10px;"
            )
        layout.addWidget(thumb_label)

        # --- Metadata column ---
        meta_layout = QVBoxLayout()
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.setSpacing(2)

        # Video name + version badge
        name_text = self._entry.video_name
        if self._entry.version > 1:
            name_text += f"  v{self._entry.version}"
        name_label = QLabel(name_text)
        name_label.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-weight: bold; font-size: 12px;"
        )
        meta_layout.addWidget(name_label)

        # Duration | person count | keyframe count
        detail_parts = [_format_duration(self._entry.duration_sec)]
        if self._entry.person_count > 0:
            p = self._entry.person_count
            detail_parts.append(f"{p} person{'s' if p != 1 else ''}")
        if self._entry.correction_count > 0:
            detail_parts.append(f"{self._entry.correction_count} keyframes")
        detail_label = QLabel(" | ".join(detail_parts))
        detail_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        meta_layout.addWidget(detail_label)

        # Date
        date_str = self._entry.date.strftime("%Y-%m-%d %H:%M")
        date_label = QLabel(date_str)
        date_label.setStyleSheet(
            f"color: {COLORS['text_disabled']}; font-size: 10px;"
        )
        meta_layout.addWidget(date_label)

        # Tag chips (max 5 visible)
        if self._entry.tags:
            tag_layout = QHBoxLayout()
            tag_layout.setContentsMargins(0, 0, 0, 0)
            tag_layout.setSpacing(4)
            for i, tag in enumerate(self._entry.tags[:5]):
                color = self.TAG_COLORS[i % len(self.TAG_COLORS)]
                tag_label = QLabel(tag)
                tag_label.setStyleSheet(
                    f"QLabel {{ background: {color}; color: #ffffff; "
                    f"padding: 1px 6px; border-radius: 3px; font-size: 9px; }}"
                )
                tag_layout.addWidget(tag_label)
            tag_layout.addStretch()
            meta_layout.addLayout(tag_layout)

        meta_layout.addStretch()
        layout.addLayout(meta_layout, 1)

        # Tooltip shows session notes
        if self._entry.notes:
            self.setToolTip(self._entry.notes)


# ---------------------------------------------------------------------------
# Session Library widget
# ---------------------------------------------------------------------------

class SessionLibrary(QWidget):
    """Media-pool-style session browser with thumbnails and metadata.

    Why: As users accumulate capture sessions, navigating them via file paths
    becomes impractical.  The library panel provides visual browsing with
    thumbnails, sortable metadata, tags for categorization, and notes for
    context — all without leaving the application.
    """

    session_load_requested = Signal(str)  # path to session JSON

    def __init__(self, gvhmr_root: Path | None = None, parent=None):
        super().__init__(parent)
        self._gvhmr_root = gvhmr_root
        self._entries: list[SessionEntry] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Top bar: search + refresh ---
        top_bar = QHBoxLayout()
        top_bar.setSpacing(4)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Filter sessions...")
        self._search_input.textChanged.connect(self._apply_filter)
        top_bar.addWidget(self._search_input, 1)

        self._refresh_btn = QToolButton()
        self._refresh_btn.setText("Refresh")
        self._refresh_btn.setToolTip("Rescan for sessions")
        self._refresh_btn.clicked.connect(self.scan)
        top_bar.addWidget(self._refresh_btn)

        layout.addLayout(top_bar)

        # --- Session list ---
        self._list = QListWidget()
        self._list.setSpacing(2)
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._list, 1)

        # --- Count label ---
        self._count_label = QLabel("0 sessions")
        self._count_label.setStyleSheet(
            f"color: {COLORS['text_disabled']}; font-size: 10px;"
        )
        layout.addWidget(self._count_label)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan(self, extra_paths: list[Path] | None = None):
        """Scan for session files and populate the list.

        Looks in GVHMR output directories for bodypipe_session.json files
        and merges with any extra paths (e.g., from the recent sessions list).
        """
        self._entries.clear()
        seen_paths: set[str] = set()

        # Scan GVHMR output directories
        if self._gvhmr_root:
            for outputs_subdir in ("outputs/demo", "outputs/multi_person"):
                scan_root = self._gvhmr_root / outputs_subdir
                if scan_root.is_dir():
                    for session_file in scan_root.rglob("bodypipe_session.json"):
                        key = str(session_file.resolve())
                        if key not in seen_paths:
                            seen_paths.add(key)
                            entry = _load_session_entry(session_file)
                            if entry:
                                self._entries.append(entry)

        # Add extra paths (e.g., recent sessions)
        if extra_paths:
            for p in extra_paths:
                key = str(p.resolve()) if p.exists() else str(p)
                if key not in seen_paths and p.is_file():
                    seen_paths.add(key)
                    entry = _load_session_entry(p)
                    if entry:
                        self._entries.append(entry)

        # Sort by date (newest first)
        self._entries.sort(key=lambda e: e.date, reverse=True)

        self._populate_list()

    # ------------------------------------------------------------------
    # List display
    # ------------------------------------------------------------------

    def _populate_list(self):
        """Rebuild list widget from entries, respecting the current filter."""
        self._list.clear()
        filter_text = self._search_input.text().lower()

        visible_count = 0
        for entry in self._entries:
            if filter_text:
                searchable = (
                    entry.video_name.lower()
                    + " " + " ".join(entry.tags).lower()
                    + " " + entry.notes.lower()
                )
                if filter_text not in searchable:
                    continue

            card = _SessionCard(entry)
            item = QListWidgetItem()
            item.setSizeHint(QSize(0, THUMBNAIL_HEIGHT + 12))
            item.setData(Qt.UserRole, str(entry.path))
            self._list.addItem(item)
            self._list.setItemWidget(item, card)
            visible_count += 1

        total = len(self._entries)
        if filter_text:
            self._count_label.setText(f"{visible_count} / {total} sessions")
        else:
            self._count_label.setText(
                f"{total} session{'s' if total != 1 else ''}"
            )

    def _apply_filter(self, _text: str):
        """Re-filter displayed sessions when search text changes."""
        self._populate_list()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_double_click(self, item: QListWidgetItem):
        """Load session on double-click."""
        path = item.data(Qt.UserRole)
        if path:
            self.session_load_requested.emit(path)

    def _on_context_menu(self, pos):
        """Right-click context menu for session operations."""
        item = self._list.itemAt(pos)
        if not item:
            return

        path_str = item.data(Qt.UserRole)
        if not path_str:
            return

        menu = QMenu(self)

        open_action = menu.addAction("Open Session")
        open_action.triggered.connect(
            lambda: self.session_load_requested.emit(path_str)
        )

        menu.addSeparator()

        edit_notes_action = menu.addAction("Edit Notes...")
        edit_notes_action.triggered.connect(lambda: self._edit_notes(path_str))

        edit_tags_action = menu.addAction("Edit Tags...")
        edit_tags_action.triggered.connect(lambda: self._edit_tags(path_str))

        menu.exec(self._list.mapToGlobal(pos))

    # ------------------------------------------------------------------
    # Metadata editing
    # ------------------------------------------------------------------

    def _find_entry(self, path_str: str) -> SessionEntry | None:
        """Find a SessionEntry by path string."""
        for e in self._entries:
            if str(e.path) == path_str:
                return e
        return None

    def _edit_notes(self, path_str: str):
        """Edit session notes via input dialog and persist to JSON."""
        entry = self._find_entry(path_str)
        if not entry:
            return

        text, ok = QInputDialog.getMultiLineText(
            self, "Edit Notes", "Session notes:", entry.notes,
        )
        if not ok:
            return

        entry.notes = text
        _update_session_json(entry.path, notes=text)
        self._populate_list()

    def _edit_tags(self, path_str: str):
        """Edit session tags via input dialog and persist to JSON."""
        entry = self._find_entry(path_str)
        if not entry:
            return

        current = ", ".join(entry.tags)
        text, ok = QInputDialog.getText(
            self, "Edit Tags", "Tags (comma-separated):", text=current,
        )
        if not ok:
            return

        new_tags = [t.strip() for t in text.split(",") if t.strip()]
        entry.tags = new_tags
        _update_session_json(entry.path, tags=new_tags)
        self._populate_list()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def entries(self) -> list[SessionEntry]:
        return self._entries

    @property
    def list_widget(self) -> QListWidget:
        """Expose list widget for testing."""
        return self._list

    @property
    def search_input(self) -> QLineEdit:
        """Expose search input for testing."""
        return self._search_input


# ---------------------------------------------------------------------------
# JSON update helper
# ---------------------------------------------------------------------------

def _update_session_json(path: Path, **kwargs):
    """Update specific fields in a session JSON file.

    Why partial update: Loading a full Session (with person_tracks, etc.)
    just to change notes or tags would be wasteful.  We read the raw dict,
    merge the changed fields, and write back.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        data.update(kwargs)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        log.exception("Failed to update session JSON at %s", path)
