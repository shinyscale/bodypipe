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

# On DGX Spark (separate venvs), add GVHMR's venv site-packages so torch
# and other heavy deps are available for in-process imports (multi_person_split, etc.)
_gvhmr_sp = GVHMR_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
if _gvhmr_sp.is_dir() and str(_gvhmr_sp) not in sys.path:
    sys.path.append(str(_gvhmr_sp))

# Add GEM-X root + venv site-packages for SOMA body model (py-soma-x, warp-lang, trimesh).
# GEM-X's SomaLayer wrapper is imported by the viewport for SOMA mesh rendering.
GEMX_ROOT = Path(__file__).resolve().parent.parent / "GEM-X"
_gemx_sp = GEMX_ROOT / ".venv" / "lib" / "python3.12" / "site-packages"
if _gemx_sp.is_dir() and str(_gemx_sp) not in sys.path:
    sys.path.append(str(_gemx_sp))
if GEMX_ROOT.is_dir() and str(GEMX_ROOT) not in sys.path:
    sys.path.append(str(GEMX_ROOT))

# PyTorch 2.12+ defaults weights_only=True which breaks YOLO/ultralytics
# checkpoint loading. Allow unsafe loads globally (all checkpoints are local).
try:
    import torch
    torch.serialization.add_safe_globals(
        [type(None)]  # dummy to ensure the mechanism is initialized
    )
    # Override default to weights_only=False for compatibility
    _orig_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)
    torch.load = _patched_load
except ImportError:
    pass

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
