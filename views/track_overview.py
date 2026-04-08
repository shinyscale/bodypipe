"""DAW-style vertical track timeline with confidence heatmaps.

Why: Phase 2 of the UX overhaul spec.  The previous TrackOverview was a simple
vertical stack of ConfidenceTimeline widgets.  This version uses QGraphicsView
for horizontal zoom/pan, per-frame confidence heatmaps, keyframe markers,
issue flags, correction markers, and crossing-span highlighting — matching
the track timeline patterns in OptiTrack Motive and Kdenlive.

Key design decisions:
- QGraphicsView for the timeline area (handles zoom via X-only transform, pan
  via middle-drag, and hit testing natively).
- Separate fixed-width header panel for track labels so headers don't scroll
  horizontally with the timeline.
- Each person gets a _TrackLaneItem (QGraphicsItem) that paints its own
  heatmap, markers, and overlays in a single paint() call for performance.
- Marker X-dimensions use inverse device scale so they stay a constant screen
  size regardless of zoom level (Y is never scaled so Y sizes are fixed).
- _PlayheadItem uses ItemIgnoresTransformations for crisp 2px line at any zoom.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QMouseEvent,
    QPainter,
    QPen,
    QPolygonF,
    QTransform,
    QWheelEvent,
)

from theme import COLORS, PERSON_COLORS

LANE_HEIGHT = 24
LANE_SPACING = 4
HEADER_WIDTH = 100
_MIN_ZOOM = 0.01
_MAX_ZOOM = 50.0
_SPEED_PRESETS = (0.25, 0.5, 1.0, 2.0, 4.0)

# Pre-built QColor objects for hot-path painting
_COLOR_SUCCESS = QColor(COLORS["success"])
_COLOR_WARNING = QColor(COLORS["warning"])
_COLOR_ERROR = QColor(COLORS["error"])
_COLOR_ACCENT = QColor(COLORS["accent"])
_COLOR_BORDER = QColor(COLORS.get("border", "#76797C"))
_COLOR_NEUTRAL = QColor(COLORS.get("bg_active_tab", "#54575B"))
_COLOR_CROSSING = QColor("#0984e3")
_COLOR_CROSSING.setAlpha(80)
_COLOR_INTERP = QColor("#ca952e")  # amber, matches correction dots
_COLOR_INTERP.setAlpha(100)
_COLOR_FOOT_SLIDE = QColor("#00b894")  # teal, foot-slide correction spans
_COLOR_FOOT_SLIDE.setAlpha(80)
_COLOR_DRIFT = QColor("#6c5ce7")  # purple, drift correction spans
_COLOR_DRIFT.setAlpha(80)
_COLOR_POSITION = QColor("#e17055")  # coral, position correction markers


def _conf_color(conf: float) -> QColor:
    """Map confidence 0-1 → green (>0.8) / yellow (0.5-0.8) / red (<0.5)."""
    if conf > 0.8:
        return _COLOR_SUCCESS
    if conf > 0.5:
        return _COLOR_WARNING
    return _COLOR_ERROR


# ---------------------------------------------------------------------------
# QGraphicsItem subclasses
# ---------------------------------------------------------------------------


class _TrackLaneItem(QGraphicsItem):
    """One person's horizontal heatmap lane drawn inside the QGraphicsScene.

    Scene coordinates: X = frame index (0 .. num_frames), Y = pixel height.
    The view transform scales X only, so marker X-sizes are computed using
    the inverse device scale to stay a constant screen size.
    """

    def __init__(self, person_id: int, num_frames: int, y_pos: float):
        super().__init__()
        self._person_id = person_id
        self._num_frames = max(1, num_frames)
        self._confidences: np.ndarray | None = None
        self._keyframe_frames: list[int] = []
        self._verified_frames: set[int] = set()
        self._issue_frames: list[int] = []
        self._correction_frames: list[int] = []
        self._crossing_spans: list[tuple[int, int]] = []
        self._interpolation_spans: list[tuple[int, int]] = []
        self._foot_slide_spans: list[tuple[int, int]] = []
        self._drift_correction_spans: list[tuple[int, int]] = []
        self._foot_pin_frames: list[int] = []
        self._position_correction_frames: list[int] = []
        self._collapsed = False
        self.setPos(0, y_pos)

    # -- QGraphicsItem interface ------------------------------------------

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(0, 0, self._num_frames, LANE_HEIGHT)

    def paint(self, painter: QPainter, option, widget=None):  # noqa: ARG002
        if self._collapsed:
            pen = QPen(_COLOR_BORDER, 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(
                QPointF(0, LANE_HEIGHT / 2),
                QPointF(self._num_frames, LANE_HEIGHT / 2),
            )
            return

        exposed = option.exposedRect
        f_start = max(0, int(exposed.left()))
        f_end = min(self._num_frames, int(exposed.right()) + 2)

        # -- confidence heatmap -------------------------------------------
        if self._confidences is not None and len(self._confidences) > 0:
            sx = max(0.001, abs(painter.deviceTransform().m11()))
            if sx >= 1.0:
                for i in range(f_start, f_end):
                    if i < len(self._confidences):
                        painter.fillRect(
                            QRectF(i, 0, 1.0, LANE_HEIGHT),
                            _conf_color(float(self._confidences[i])),
                        )
            else:
                bin_sz = max(1, int(1.0 / sx))
                for bs in range(f_start, f_end, bin_sz):
                    be = min(bs + bin_sz, f_end, len(self._confidences))
                    if bs < len(self._confidences):
                        avg = float(np.mean(self._confidences[bs:be]))
                        painter.fillRect(
                            QRectF(bs, 0, be - bs, LANE_HEIGHT),
                            _conf_color(avg),
                        )
        else:
            painter.fillRect(self.boundingRect(), _COLOR_NEUTRAL)

        # -- crossing spans (blue tint) -----------------------------------
        for span_s, span_e in self._crossing_spans:
            s = max(span_s, f_start)
            e = min(span_e, f_end)
            if s < e:
                painter.fillRect(QRectF(s, 0, e - s, LANE_HEIGHT), _COLOR_CROSSING)

        # -- foot-slide correction spans (teal tint) ----------------------
        for span_s, span_e in self._foot_slide_spans:
            s = max(span_s, f_start)
            e = min(span_e, f_end)
            if s < e:
                painter.fillRect(QRectF(s, 0, e - s, LANE_HEIGHT), _COLOR_FOOT_SLIDE)

        # -- drift correction spans (purple tint) -------------------------
        for span_s, span_e in self._drift_correction_spans:
            s = max(span_s, f_start)
            e = min(span_e, f_end)
            if s < e:
                painter.fillRect(QRectF(s, 0, e - s, LANE_HEIGHT), _COLOR_DRIFT)

        # Inverse X scale for fixed-screen-size markers
        sx = max(0.001, abs(painter.deviceTransform().m11()))
        inv = 1.0 / sx

        # -- interpolation spans (amber band connecting correction dots) ----
        if self._interpolation_spans:
            cy = LANE_HEIGHT / 2.0
            pen = QPen(_COLOR_INTERP, 2.0)
            pen.setCosmetic(True)
            painter.setPen(pen)
            for span_s, span_e in self._interpolation_spans:
                s = max(span_s, f_start)
                e = min(span_e, f_end)
                if s < e:
                    painter.drawLine(QPointF(s + 0.5, cy), QPointF(e + 0.5, cy))

        # -- correction markers (amber dots, mid-lane) --------------------
        if self._correction_frames:
            painter.setPen(Qt.NoPen)
            painter.setBrush(_COLOR_ACCENT)
            cy = LANE_HEIGHT / 2.0
            rx = 2.5 * inv
            for f in self._correction_frames:
                if f_start <= f < f_end:
                    painter.drawEllipse(QPointF(f + 0.5, cy), rx, 2.5)

        # -- issue flags (red diamonds, near top) -------------------------
        if self._issue_frames:
            painter.setPen(Qt.NoPen)
            painter.setBrush(_COLOR_ERROR)
            dx = 3.0 * inv
            for f in self._issue_frames:
                if f_start <= f < f_end:
                    cx = f + 0.5
                    painter.drawPolygon(QPolygonF([
                        QPointF(cx, 2.0),
                        QPointF(cx + dx, 5.0),
                        QPointF(cx, 8.0),
                        QPointF(cx - dx, 5.0),
                    ]))

        # -- keyframe markers (triangles at bottom) -----------------------
        tw = 3.0 * inv
        for f in self._keyframe_frames:
            if f_start <= f < f_end:
                painter.setPen(Qt.NoPen)
                painter.setBrush(
                    _COLOR_SUCCESS if f in self._verified_frames else _COLOR_WARNING,
                )
                cx = f + 0.5
                painter.drawPolygon(QPolygonF([
                    QPointF(cx - tw, LANE_HEIGHT),
                    QPointF(cx + tw, LANE_HEIGHT),
                    QPointF(cx, LANE_HEIGHT - 5),
                ]))

        # -- foot pin markers (teal dots at top of lane) -----------------
        if self._foot_pin_frames:
            _pin_color = QColor("#00b894")  # solid teal
            painter.setPen(Qt.NoPen)
            painter.setBrush(_pin_color)
            pin_rx = 2.5 * inv
            pin_cy = 4.0  # near top of lane
            for f in self._foot_pin_frames:
                if f_start <= f < f_end:
                    painter.drawEllipse(QPointF(f + 0.5, pin_cy), pin_rx, 2.5)

        # -- position correction markers (coral squares, mid-lane) ---------
        if self._position_correction_frames:
            painter.setPen(Qt.NoPen)
            painter.setBrush(_COLOR_POSITION)
            cy = LANE_HEIGHT / 2.0
            hw = 2.5 * inv
            for f in self._position_correction_frames:
                if f_start <= f < f_end:
                    painter.drawRect(QRectF(f + 0.5 - hw, cy - 2.5, 2 * hw, 5.0))

        # -- lane border --------------------------------------------------
        pen = QPen(_COLOR_BORDER, 1)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(0, 0, self._num_frames, LANE_HEIGHT))

    # -- public API -------------------------------------------------------

    @property
    def person_id(self) -> int:
        return self._person_id

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    def set_confidences(self, conf: np.ndarray):
        self._confidences = conf
        self.update()

    def set_keyframes(self, frames: list[int], verified: set[int] | None = None):
        self._keyframe_frames = list(frames)
        self._verified_frames = verified or set()
        self.update()

    def set_issues(self, frames: list[int]):
        self._issue_frames = list(frames)
        self.update()

    def set_corrections(self, frames: list[int]):
        self._correction_frames = list(frames)
        self.update()

    def set_crossing_spans(self, spans: list[tuple[int, int]]):
        self._crossing_spans = list(spans)
        self.update()

    def set_interpolation_spans(self, spans: list[tuple[int, int]]):
        self._interpolation_spans = list(spans)
        self.update()

    def set_foot_slide_spans(self, spans: list[tuple[int, int]]):
        self._foot_slide_spans = list(spans)
        self.update()

    def set_drift_correction_spans(self, spans: list[tuple[int, int]]):
        self._drift_correction_spans = list(spans)
        self.update()

    def set_foot_pin_frames(self, frames: list[int]):
        self._foot_pin_frames = list(frames)
        self.update()

    def set_position_correction_frames(self, frames: list[int]):
        self._position_correction_frames = list(frames)
        self.update()

    def set_collapsed(self, collapsed: bool):
        self._collapsed = collapsed
        self.update()


class _PlayheadItem(QGraphicsItem):
    """Vertical playhead line spanning all track lanes.

    Uses ItemIgnoresTransformations so the line is always 2 device-pixels
    wide, while still positioning correctly via the view transform.
    """

    def __init__(self, total_height: float):
        super().__init__()
        self._total_height = max(1.0, total_height)
        self.setZValue(100)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

    def boundingRect(self) -> QRectF:  # noqa: N802
        return QRectF(-2, -1, 4, self._total_height + 2)

    def paint(self, painter: QPainter, option, widget=None):  # noqa: ARG002
        pen = QPen(_COLOR_ACCENT, 2)
        painter.setPen(pen)
        painter.drawLine(QPointF(0, 0), QPointF(0, self._total_height))

    def set_frame(self, frame: int):
        self.setPos(frame + 0.5, 0)

    def set_height(self, h: float):
        self.prepareGeometryChange()
        self._total_height = max(1.0, h)

    @property
    def frame(self) -> int:
        return int(self.pos().x())


# ---------------------------------------------------------------------------
# QGraphicsView subclass (zoom / pan / click)
# ---------------------------------------------------------------------------


class _TimelineView(QGraphicsView):
    """Timeline viewport with scroll-wheel horizontal zoom and middle-drag pan."""

    frame_clicked = Signal(int, int)  # (person_id, frame_index)

    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None):
        super().__init__(scene, parent)
        self._zoom_x = 1.0
        self._num_frames = 1
        self._lanes: dict[int, _TrackLaneItem] = {}
        self._panning = False

        self.setRenderHint(QPainter.Antialiasing, False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)

    # -- public -----------------------------------------------------------

    def set_lanes(self, lanes: dict[int, _TrackLaneItem]):
        self._lanes = lanes

    def set_num_frames(self, n: int):
        self._num_frames = max(1, n)
        self._fit_zoom()

    @property
    def zoom_x(self) -> float:
        return self._zoom_x

    # -- internal ---------------------------------------------------------

    def _apply_transform(self):
        t = QTransform()
        t.scale(self._zoom_x, 1.0)
        self.setTransform(t)

    def _fit_zoom(self):
        vw = self.viewport().width()
        if vw > 0 and self._num_frames > 0:
            self._zoom_x = max(_MIN_ZOOM, vw / self._num_frames)
        self._apply_transform()

    # -- events -----------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent):  # noqa: N802
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1.0 / 1.15
        self._zoom_x = max(_MIN_ZOOM, min(self._zoom_x * factor, _MAX_ZOOM))
        self._apply_transform()
        event.accept()

    def mousePressEvent(self, event: QMouseEvent):  # noqa: N802
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            fake = QMouseEvent(
                event.type(),
                event.position(),
                event.globalPosition(),
                Qt.LeftButton,
                Qt.LeftButton,
                event.modifiers(),
            )
            super().mousePressEvent(fake)
            return

        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            frame = max(0, min(int(pos.x()), self._num_frames - 1))
            for pid, lane in self._lanes.items():
                y_top = lane.pos().y()
                if y_top <= pos.y() < y_top + LANE_HEIGHT:
                    self.frame_clicked.emit(pid, frame)
                    event.accept()
                    return

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):  # noqa: N802
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self._fit_zoom()


# ---------------------------------------------------------------------------
# Track header (fixed-width left column)
# ---------------------------------------------------------------------------


class _TrackHeader(QWidget):
    """Header row: colored dot + 'Person N' label + collapse arrow."""

    collapse_toggled = Signal(int, bool)  # (person_id, collapsed)

    def __init__(self, person_id: int, parent: QWidget | None = None):
        super().__init__(parent)
        self._person_id = person_id
        self._collapsed = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(4)

        color = PERSON_COLORS[person_id % len(PERSON_COLORS)]

        dot = QLabel("\u25cf")  # ●
        dot.setStyleSheet(f"color: {color}; font-size: 12px;")
        dot.setFixedWidth(14)
        layout.addWidget(dot)

        self._label = QLabel(f"Person {person_id}")
        self._label.setStyleSheet(
            f"color: {color}; font-weight: bold; font-size: 11px;",
        )
        layout.addWidget(self._label, 1)

        self._btn = QToolButton()
        self._btn.setText("\u25bc")  # ▼
        self._btn.setFixedSize(16, 16)
        self._btn.setStyleSheet("border: none; color: #eff0f1; font-size: 8px;")
        self._btn.clicked.connect(self._toggle)
        layout.addWidget(self._btn)

        self.setFixedHeight(LANE_HEIGHT + LANE_SPACING)

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._btn.setText("\u25ba" if self._collapsed else "\u25bc")  # ► / ▼
        self.collapse_toggled.emit(self._person_id, self._collapsed)

    @property
    def collapsed(self) -> bool:
        return self._collapsed

    @property
    def person_id(self) -> int:
        return self._person_id


# ---------------------------------------------------------------------------
# Main public widget
# ---------------------------------------------------------------------------


class TrackOverview(QWidget):
    """DAW-style vertical track timeline with confidence heatmaps,
    keyframe markers, issue flags, correction markers, and crossing spans.

    Public API (backward-compatible with the old ConfidenceTimeline stack):
        set_tracks(dict[int, np.ndarray])   — confidence arrays per person
        set_current_frame(int)              — move playhead
        person_clicked  Signal(int, int)    — (person_id, frame_index)

    New API:
        set_track_markers(person_id, ...)   — keyframes / issues / corrections / spans
        set_num_frames(int)                 — update total frame count
    """

    person_clicked = Signal(int, int)  # (person_id, frame_index)
    speed_changed = Signal(float)  # emitted on user speed chip click
    play_toggled = Signal()        # play/pause clicked
    go_to_start = Signal()         # go-to-start clicked
    loop_toggled = Signal(bool)    # loop toggled

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._num_frames = 1
        self._lanes: dict[int, _TrackLaneItem] = {}
        self._headers: dict[int, _TrackHeader] = {}

        main = QHBoxLayout(self)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        # Fixed-width header panel (track labels, collapse arrows)
        self._header_panel = QWidget()
        self._header_panel.setFixedWidth(HEADER_WIDTH)
        self._header_layout = QVBoxLayout(self._header_panel)
        self._header_layout.setContentsMargins(0, 0, 0, 0)
        self._header_layout.setSpacing(0)
        self._header_layout.addStretch()
        main.addWidget(self._header_panel)

        # Vertical separator
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {COLORS.get('border', '#76797C')};")
        main.addWidget(sep)

        # Right side: timeline view + speed chip footer
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)

        # QGraphicsView timeline
        self._scene = QGraphicsScene(self)
        self._view = _TimelineView(self._scene, self)
        self._view.frame_clicked.connect(self.person_clicked)
        right.addWidget(self._view, 1)

        # Footer — transport buttons + speed chips
        self._footer = QWidget()
        self._footer.setFixedHeight(24)
        footer_layout = QHBoxLayout(self._footer)
        footer_layout.setContentsMargins(4, 2, 4, 2)
        footer_layout.setSpacing(4)

        _transport_style = (
            "QToolButton { background: transparent; "
            "color: #eff0f1; border: none; border-radius: 2px; font-size: 11px; }"
            "QToolButton:hover { background: rgba(202,149,46,80); }"
            "QToolButton:pressed { background: rgba(202,149,46,160); }"
            "QToolButton:checked { background: rgba(202,149,46,200); color: #eff0f1; }"
        )

        self._footer_first = QToolButton()
        self._footer_first.setText("\u23ee")
        self._footer_first.setToolTip("Go to start")
        self._footer_first.setFixedSize(20, 18)
        self._footer_first.setStyleSheet(_transport_style)
        self._footer_first.setFocusPolicy(Qt.NoFocus)
        self._footer_first.clicked.connect(self.go_to_start.emit)
        footer_layout.addWidget(self._footer_first)

        self._footer_play = QToolButton()
        self._footer_play.setText("\u25b6")
        self._footer_play.setToolTip("Play/Pause")
        self._footer_play.setFixedSize(22, 18)
        self._footer_play.setStyleSheet(_transport_style)
        self._footer_play.setFocusPolicy(Qt.NoFocus)
        self._footer_play.clicked.connect(self.play_toggled.emit)
        footer_layout.addWidget(self._footer_play)

        self._footer_loop = QToolButton()
        self._footer_loop.setText("\u21bb")
        self._footer_loop.setToolTip("Loop")
        self._footer_loop.setCheckable(True)
        self._footer_loop.setFixedSize(20, 18)
        self._footer_loop.setStyleSheet(_transport_style)
        self._footer_loop.setFocusPolicy(Qt.NoFocus)
        self._footer_loop.clicked.connect(
            lambda checked: self.loop_toggled.emit(checked)
        )
        footer_layout.addWidget(self._footer_loop)

        footer_layout.addSpacing(8)
        footer_layout.addStretch()

        self._speed_group = QButtonGroup(self)
        self._speed_group.setExclusive(True)
        self._speed_chips: dict[float, QToolButton] = {}

        _chip_style = (
            "QToolButton { background: transparent; color: #76797C; "
            "border: 1px solid rgba(118,121,124,60); border-radius: 2px; "
            "padding: 0px 3px; font-size: 9px; }"
            "QToolButton:checked { background: rgba(202,149,46,180); "
            "color: #eff0f1; border-color: #ca952e; }"
            "QToolButton:hover { border-color: #ca952e; }"
        )
        for speed in _SPEED_PRESETS:
            label = f"{int(speed)}x" if speed == int(speed) else f"{speed}x"
            chip = QToolButton()
            chip.setText(label)
            chip.setCheckable(True)
            chip.setFixedHeight(18)
            chip.setStyleSheet(_chip_style)
            chip.setFocusPolicy(Qt.NoFocus)
            self._speed_group.addButton(chip)
            self._speed_chips[speed] = chip
            footer_layout.addWidget(chip)
        self._speed_chips[1.0].setChecked(True)
        self._speed_group.buttonClicked.connect(self._on_speed_chip_clicked)

        right.addWidget(self._footer)
        main.addLayout(right, 1)

        # Playhead (always on top)
        self._playhead = _PlayheadItem(1)
        self._scene.addItem(self._playhead)

        self.setMinimumHeight(40)

    # -- backward-compatible API ------------------------------------------

    def set_tracks(self, tracks: dict[int, np.ndarray]):
        """Set confidence arrays per person_id.  Clears previous state."""
        self._clear()
        if not tracks:
            return

        self._num_frames = max((len(c) for c in tracks.values()), default=1)
        self._num_frames = max(1, self._num_frames)

        y = 0.0
        for pid in sorted(tracks.keys()):
            # Header widget
            header = _TrackHeader(pid)
            header.collapse_toggled.connect(self._on_collapse)
            self._headers[pid] = header
            idx = self._header_layout.count() - 1  # insert before stretch
            self._header_layout.insertWidget(idx, header)

            # Lane item in the scene
            lane = _TrackLaneItem(pid, self._num_frames, y)
            lane.set_confidences(np.asarray(tracks[pid], dtype=np.float32))
            self._scene.addItem(lane)
            self._lanes[pid] = lane

            y += LANE_HEIGHT + LANE_SPACING

        total_h = max(1.0, y - LANE_SPACING) if y > 0 else 1.0
        self._scene.setSceneRect(0, 0, self._num_frames, total_h)
        self._playhead.set_height(total_h)
        self._view.set_lanes(self._lanes)
        self._view.set_num_frames(self._num_frames)

        # Resize widget to fit all tracks + scrollbar
        sb_h = self._view.horizontalScrollBar().height() if self._view.horizontalScrollBar() else 0
        self.setMinimumHeight(max(40, int(y) + sb_h + 4))

    def set_current_frame(self, frame: int):
        """Move the playhead to *frame*."""
        self._playhead.set_frame(frame)

    # -- new API ----------------------------------------------------------

    def set_track_markers(
        self,
        person_id: int,
        *,
        keyframes: list[int] | None = None,
        verified_frames: set[int] | None = None,
        issue_frames: list[int] | None = None,
        correction_frames: list[int] | None = None,
        crossing_spans: list[tuple[int, int]] | None = None,
        interpolation_spans: list[tuple[int, int]] | None = None,
        foot_slide_spans: list[tuple[int, int]] | None = None,
        foot_pin_frames: list[int] | None = None,
        drift_correction_spans: list[tuple[int, int]] | None = None,
        position_correction_frames: list[int] | None = None,
    ):
        """Set additional markers for a specific person's track lane."""
        lane = self._lanes.get(person_id)
        if lane is None:
            return
        if keyframes is not None:
            lane.set_keyframes(keyframes, verified_frames)
        if issue_frames is not None:
            lane.set_issues(issue_frames)
        if correction_frames is not None:
            lane.set_corrections(correction_frames)
        if crossing_spans is not None:
            lane.set_crossing_spans(crossing_spans)
        if interpolation_spans is not None:
            lane.set_interpolation_spans(interpolation_spans)
        if foot_slide_spans is not None:
            lane.set_foot_slide_spans(foot_slide_spans)
        if foot_pin_frames is not None:
            lane.set_foot_pin_frames(foot_pin_frames)
        if drift_correction_spans is not None:
            lane.set_drift_correction_spans(drift_correction_spans)
        if position_correction_frames is not None:
            lane.set_position_correction_frames(position_correction_frames)

    def set_num_frames(self, num_frames: int):
        """Update total frame count for timeline scaling."""
        self._num_frames = max(1, num_frames)
        self._scene.setSceneRect(
            0, 0, self._num_frames, self._scene.sceneRect().height(),
        )
        self._view.set_num_frames(self._num_frames)

    # -- speed chips ------------------------------------------------------

    def set_speed(self, speed: float):
        """Set the selected speed chip (called externally to sync).

        Why blockSignals: prevents signal loop when synced with VideoPlayer.
        """
        chip = self._speed_chips.get(speed)
        if chip:
            self._speed_group.blockSignals(True)
            chip.setChecked(True)
            self._speed_group.blockSignals(False)

    def _on_speed_chip_clicked(self, button):
        """User clicked a speed chip — emit signal for sync."""
        for speed, chip in self._speed_chips.items():
            if chip is button:
                self.speed_changed.emit(speed)
                break

    # -- transport sync ---------------------------------------------------

    def set_playing(self, playing: bool):
        """Update play button icon (visual only, no signal emitted)."""
        self._footer_play.setText("\u275a\u275a" if playing else "\u25b6")

    def set_loop(self, looping: bool):
        """Update loop button checked state (no signal emitted)."""
        self._footer_loop.blockSignals(True)
        self._footer_loop.setChecked(looping)
        self._footer_loop.blockSignals(False)

    # -- internal ---------------------------------------------------------

    def _on_collapse(self, person_id: int, collapsed: bool):
        lane = self._lanes.get(person_id)
        if lane:
            lane.set_collapsed(collapsed)

    def _clear(self):
        for lane in list(self._lanes.values()):
            self._scene.removeItem(lane)
        self._lanes.clear()

        for header in list(self._headers.values()):
            self._header_layout.removeWidget(header)
            header.deleteLater()
        self._headers.clear()

        self._playhead.set_frame(0)
