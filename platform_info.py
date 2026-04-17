"""Platform detection — WSL2 vs native Windows vs native Linux.

Computed once at import time.  Every module that needs platform-specific
branching should ``from platform_info import IS_WSL, IS_WINDOWS`` rather
than rolling its own ``"microsoft" in platform.uname()`` check.
"""

from __future__ import annotations

import os
import platform
import sys

# Native Windows (not WSL)
IS_WINDOWS: bool = sys.platform == "win32"

# WSL2 Linux kernel (has "microsoft" in the release string)
IS_WSL: bool = False
if not IS_WINDOWS:
    try:
        IS_WSL = "microsoft" in platform.uname().release.lower()
    except Exception:
        pass

# True on any Linux — native or WSL
IS_LINUX: bool = sys.platform.startswith("linux")


def conda_python(env_name: str) -> list[str]:
    """Return candidate paths for a conda environment's Python binary.

    On Windows: ``envs/<name>/python.exe``  (also ``Scripts/python.exe``)
    On Unix:    ``envs/<name>/bin/python``
    """
    roots = [
        "miniconda3",
        "anaconda3",
        ".conda",
    ]
    home = __import__("pathlib").Path.home()
    candidates: list[str] = []
    for root in roots:
        env_dir = home / root / "envs" / env_name
        if IS_WINDOWS:
            # conda on Windows puts python.exe directly in the env root
            candidates.append(str(env_dir / "python.exe"))
            candidates.append(str(env_dir / "Scripts" / "python.exe"))
        else:
            candidates.append(str(env_dir / "bin" / "python"))
    return candidates
