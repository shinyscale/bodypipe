"""bodypipe — Qt/PySide6 Motion Capture Studio.

Entry point for the application. Sets up sys.path for GVHMR backend imports,
creates the QApplication, and launches the main window.
"""

import os
import platform
import sys
from pathlib import Path

# Add GVHMR root to sys.path so backend modules are importable
GVHMR_ROOT = Path(__file__).resolve().parent.parent / "GVHMR"
if str(GVHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GVHMR_ROOT))

# On WSL2, two OpenGL platform issues must be fixed before any GL library
# is loaded (i.e. before QApplication is created):
#
# 1. MESA's ZINK (Vulkan→GL) and DRI2 backends often fail, leaving only
#    legacy swrast (GL ≤2.1).  LIBGL_ALWAYS_SOFTWARE=1 forces llvmpipe
#    (software renderer, GL 4.5) which supports modern GL features.
#
# 2. WSLg uses Wayland/EGL, so Qt creates an EGL context.  PyOpenGL
#    defaults to the GLX platform, which can't see EGL contexts and
#    reports all GL functions as undefined.  PYOPENGL_PLATFORM=egl
#    tells PyOpenGL to query the EGL context instead.
try:
    if "microsoft" in platform.uname().release.lower():
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
except Exception:
    pass

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
