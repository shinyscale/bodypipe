# Spec: App Shell

## Overview

QMainWindow with QTabWidget (3 tabs), dark theme matching the Gradio orange accent, menu bar, and status bar. This is the outermost container — all other widgets nest inside it.

## Components

### MainWindow (`app_window.py`)

- Subclass `QMainWindow`
- Window title: "bodypipe — Motion Capture Studio"
- Default size: 1600x900, resizable
- Remember window geometry on close (QSettings)

### Tab Widget

Three tabs matching the Gradio app:

1. **GVHMR Body** — single-person body capture
2. **Performance Capture** — full body+hands+face pipeline
3. **Multi-Person** — multi-person pipeline + identity inspector + pose corrector

Tab bar at the top, icons optional (can add later).

### Menu Bar

- **File**: Open Video, Open Session, Save Session, Recent Sessions (QSettings-backed), Exit
- **Edit**: Undo, Redo (wired to Session undo/redo stack)
- **View**: Toggle Status Bar, Toggle Log Panel
- **Help**: About, Keyboard Shortcuts

### Status Bar

- Left: current operation status ("Ready", "Running GVHMR...", etc.)
- Center: frame counter ("Frame 142 / 3600")
- Right: FPS display, GPU memory usage (optional, polled)

### Log Panel

- Collapsible QDockWidget at the bottom
- QPlainTextEdit in read-only mode
- Captures subprocess stdout/stderr from pipeline workers
- Color-coded: info=white, warning=yellow, error=red
- Max 10000 lines with ring buffer behavior

## Theme

Dark palette matching current Gradio dark mode:

```python
COLORS = {
    "bg_primary": "#1a1a2e",       # main background
    "bg_secondary": "#16213e",     # panel backgrounds
    "bg_tertiary": "#0f3460",      # hover/active states
    "accent": "#e94560",           # orange-red accent (matches Gradio)
    "accent_hover": "#ff6b6b",
    "text_primary": "#eaeaea",
    "text_secondary": "#a0a0a0",
    "border": "#2a2a4a",
    "success": "#4ecca3",
    "warning": "#ffd93d",
    "error": "#ff6b6b",
}
```

Apply via `QApplication.setPalette()` + stylesheet for fine-grained control. Use fusion style as base (`QApplication.setStyle("Fusion")`).

## Signals

```python
class AppWindow(QMainWindow):
    session_loaded = Signal(object)    # Session object
    session_saved = Signal(Path)
    tab_changed = Signal(int)          # tab index
```

## Smoke Test Support

```python
if __name__ == "__main__" or "--smoke-test" in sys.argv:
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    if "--smoke-test" in sys.argv:
        QTimer.singleShot(500, app.quit)
    sys.exit(app.exec())
```

## Source Reference

- `gvhmr_gui.py:1300-1448` — Gradio tab structure, title, theme
- `gvhmr_gui.py:1-50` — imports and module setup

## Acceptance Criteria

- Window opens with 3 tabs, dark theme, menu bar, status bar
- `--smoke-test` flag opens and exits cleanly
- Window geometry persists across launches
- Log panel toggles visibility
