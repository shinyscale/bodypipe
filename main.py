"""bodypipe — Qt/PySide6 Motion Capture Studio.

Entry point for the application. Sets up sys.path for GVHMR backend imports,
creates the QApplication, and launches the main window.
"""

import sys
from pathlib import Path

# Add GVHMR root to sys.path so backend modules are importable
GVHMR_ROOT = Path(__file__).resolve().parent.parent / "GVHMR"
if str(GVHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GVHMR_ROOT))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from app_window import AppWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("bodypipe")
    app.setOrganizationName("GVHMR")

    window = AppWindow()
    window.show()

    if "--smoke-test" in sys.argv:
        # Open window, verify it renders, then exit
        QTimer.singleShot(500, app.quit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
