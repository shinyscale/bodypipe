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

# On WSL2, force llvmpipe (GL 4.5) since MESA's hardware backends often
# fail under WSL2, leaving only swrast (GL ≤2.1).  PYOPENGL_PLATFORM=egl
# tells PyOpenGL to query the EGL context (Wayland uses EGL, not GLX).
try:
    if "microsoft" in platform.uname().release.lower():
        os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
except Exception:
    pass

from PySide6.QtWidgets import QApplication, QComboBox
from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat

from app_window import AppWindow

# Force all OpenGL surfaces to have no alpha buffer — prevents the Wayland
# compositor on WSL2 from treating the QOpenGLWidget as transparent.
# Must be set before QApplication is created.
_fmt = QSurfaceFormat()
_fmt.setAlphaBufferSize(0)
_fmt.setDepthBufferSize(24)
_fmt.setVersion(3, 3)
_fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
QSurfaceFormat.setDefaultFormat(_fmt)


def _patch_combobox_for_wayland():
    """WSL2/Wayland fix: combo box popups don't dismiss after selection.

    The Wayland compositor in WSLg doesn't properly close popup windows.
    This patches QComboBox.hidePopup to also close the underlying popup
    container widget, ensuring it disappears on Wayland.
    """
    _original = QComboBox.hidePopup

    def _patched_hidePopup(self):
        _original(self)
        # The popup is the combo's view's parent (a QFrame container)
        view = self.view()
        if view and view.parent():
            view.parent().close()

    QComboBox.hidePopup = _patched_hidePopup


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("bodypipe")
    app.setOrganizationName("GVHMR")

    # Fix WSL2/Wayland combo box popup persistence
    try:
        if "microsoft" in platform.uname().release.lower():
            _patch_combobox_for_wayland()
    except Exception:
        pass

    window = AppWindow()
    window.show()

    if "--smoke-test" in sys.argv:
        # Open window, verify it renders, then exit
        QTimer.singleShot(500, app.quit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
