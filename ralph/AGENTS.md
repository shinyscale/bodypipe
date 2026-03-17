## Build & Run

- PySide6 Qt app, entry point: `python main.py`
- Smoke test: `QT_QPA_PLATFORM=offscreen python main.py --smoke-test`
- GVHMR backend imported via sys.path from `../GVHMR/` (set in `main.py`)

## Validation

- Tests: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ -x -q`
- Smoke: `QT_QPA_PLATFORM=offscreen python main.py --smoke-test`

## Codebase Patterns

- All widgets subclass QWidget, signals for inter-widget communication
- All workers subclass QThread, emit progress(float, str) and finished(dict)
- Session model is a single dataclass tree (`models/session.py`), no module-level dicts
- Video frames via OpenCV cv2.VideoCapture (same as Gradio app)
- PySide6 imports (NOT PyQt6): `from PySide6.QtWidgets import ...`
- Dark theme with accent color #e94560 (see app_window.py COLORS dict)
