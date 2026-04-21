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


def win_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path (``F:\\foo\\bar``) to WSL (``/mnt/f/foo/bar``)."""
    p = win_path.replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        p = f"/mnt/{drive}{p[2:]}"
    return p


def wsl_solve_command(
    conda_env: str,
    script: str,
    args: list[str],
    cwd: str | None = None,
) -> list[str]:
    """Build a command list that runs a Python script inside WSL2's conda env.

    On native Windows, this wraps the command with ``wsl.exe`` and translates
    paths.  On Linux/WSL2, returns a direct conda-run command.

    *script* and *cwd* should be Windows paths when called on Windows —
    they are auto-converted to WSL paths.
    """
    if IS_WINDOWS:
        wsl_script = win_to_wsl_path(script)
        def _maybe_convert(a: str) -> str:
            """Convert arg if it looks like it contains a Windows path."""
            # Check for --flag=C:\... or bare C:\... or C:/...
            for prefix in ("", "--video=", "--output_root=", "--slam_override=",
                           "--bbx_override=", "--f_mm="):
                if not a.startswith(prefix):
                    continue
                val = a[len(prefix):]
                if len(val) >= 2 and val[1] == ":" and val[0].isalpha():
                    return prefix + win_to_wsl_path(val)
                if "\\" in val:
                    return prefix + win_to_wsl_path(val)
            return a
        wsl_args = [_maybe_convert(a) for a in args]
        # Build a bash command that activates the conda env and runs the script.
        # Shell-quote each arg to handle spaces in paths.
        import shlex
        quoted_args = " ".join(shlex.quote(a) for a in wsl_args)
        bash_cmd = (
            f"source ~/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate {conda_env} && "
            f"python {shlex.quote(wsl_script)} {quoted_args}"
        )
        cmd = ["wsl.exe", "bash", "-c", bash_cmd]
        return cmd
    else:
        # Direct execution on Linux/WSL2
        python_candidates = conda_python(conda_env)
        python_exe = "python"
        for p in python_candidates:
            if __import__("os").path.isfile(p):
                python_exe = p
                break
        return [python_exe, script] + args


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
