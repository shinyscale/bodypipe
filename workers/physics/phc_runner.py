"""PHC invocation — local subprocess or SSH to Powerhouse."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PhcResult:
    """Result of a PHC invocation."""

    success: bool
    output_path: Path | None = None
    log: str = ""


def _find_phc_python(phc_root: Path) -> str:
    """Find the Python binary for the PHC conda environment."""
    from platform_info import conda_python

    for c in conda_python("phc"):
        if Path(c).is_file():
            return c
    return "python"  # Fall back to system python


def run_phc_local(
    input_npz: Path,
    output_dir: Path,
    phc_root: Path | str = "~/mocap-filter/PHC",
    checkpoint: str = "phc_3",
    progress_cb=None,
) -> PhcResult:
    """Run PHC physics refinement locally via MuJoCo subprocess.

    Uses ``scripts/refine_mujoco.py`` which runs the pretrained PHC policy
    through MuJoCo simulation (no Isaac Sim/Lab dependency).

    Parameters
    ----------
    input_npz : Path to AMASS-format NPZ input.
    output_dir : Directory for PHC output.
    phc_root : Root directory of PHC installation.
    checkpoint : PHC checkpoint name (e.g. "phc_3").
    progress_cb : Optional callback(fraction, message).
    """
    phc_root = Path(phc_root).expanduser()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_npz = output_dir / "phc_refined.npz"
    # PHC MotionLib expects PKL format, not NPZ
    input_pkl = input_npz.with_suffix(".pkl")
    motion_file = str(input_pkl) if input_pkl.is_file() else str(input_npz)

    policy_path = phc_root / "output" / "HumanoidIm" / checkpoint / "Humanoid.pth"

    if not policy_path.is_file():
        return PhcResult(
            success=False,
            log=f"PHC checkpoint not found: {policy_path}",
        )

    phc_python = _find_phc_python(phc_root)

    cmd = [
        phc_python,
        "scripts/refine_mujoco.py",
        "--motion_file", motion_file,
        "--output_file", str(output_npz),
        "--policy_path", str(policy_path),
    ]

    if progress_cb:
        progress_cb(0.05, "Starting PHC MuJoCo simulation...")

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(phc_root),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        log = proc.stdout + "\n" + proc.stderr

        if proc.returncode != 0:
            logger.error(
                "PHC failed (exit %d): %s", proc.returncode, proc.stderr[-500:]
            )
            return PhcResult(success=False, log=log)

        if not output_npz.is_file():
            return PhcResult(success=False, log=log + "\nNo output file found.")

        if progress_cb:
            progress_cb(0.95, "PHC simulation complete.")

        return PhcResult(success=True, output_path=output_npz, log=log)

    except subprocess.TimeoutExpired:
        return PhcResult(success=False, log="PHC timed out after 600 seconds.")
    except FileNotFoundError:
        return PhcResult(
            success=False,
            log=f"PHC not found at {phc_root}. Install PHC first (see spec).",
        )
    except Exception as exc:
        return PhcResult(success=False, log=f"PHC error: {exc}")


def run_phc_ssh(
    input_npz: Path,
    output_dir: Path,
    remote_host: str = "100.125.206.12",
    remote_user: str = "shinyscale",
    remote_phc_root: str = "~/mocap-filter/PHC",
    checkpoint: str = "phc_prim",
    progress_cb=None,
) -> PhcResult:
    """Run PHC on a remote machine via SSH (Tailscale).

    Workflow: SCP input -> SSH run PHC -> SCP output back.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    remote_addr = f"{remote_user}@{remote_host}"
    remote_tmp = f"/tmp/phc_job_{input_npz.stem}"
    remote_input = f"{remote_tmp}/input.npz"
    remote_output = f"{remote_tmp}/output"

    log_lines: list[str] = []

    def _log(msg):
        log_lines.append(msg)
        logger.info(msg)
        if progress_cb:
            progress_cb(-1, msg)  # Log-only (negative fraction)

    try:
        # Create remote directory
        _log(f"Creating remote directory {remote_tmp}...")
        subprocess.run(
            ["ssh", remote_addr, f"mkdir -p {remote_tmp} {remote_output}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # SCP input
        if progress_cb:
            progress_cb(0.05, "Uploading input to remote...")
        _log(f"Uploading {input_npz.name}...")
        subprocess.run(
            ["scp", str(input_npz), f"{remote_addr}:{remote_input}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Run PHC
        if progress_cb:
            progress_cb(0.1, "Running PHC on remote...")
        phc_cmd = (
            f"cd {remote_phc_root} && "
            f"python phc/run.py --cfg {checkpoint} --epoch -1 --test "
            f"--motion_file {remote_input} --output_dir {remote_output}"
        )
        _log(f"Running: {phc_cmd}")
        proc = subprocess.run(
            ["ssh", remote_addr, phc_cmd],
            capture_output=True,
            text=True,
            timeout=600,
        )
        log_lines.append(proc.stdout)
        log_lines.append(proc.stderr)

        if proc.returncode != 0:
            return PhcResult(success=False, log="\n".join(log_lines))

        # SCP output back
        if progress_cb:
            progress_cb(0.9, "Downloading results...")
        _log("Downloading results...")
        subprocess.run(
            ["scp", "-r", f"{remote_addr}:{remote_output}/*", str(output_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Cleanup remote
        subprocess.run(
            ["ssh", remote_addr, f"rm -rf {remote_tmp}"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Find output
        output_candidates = list(output_dir.glob("*.npz")) + list(
            output_dir.glob("*.pkl")
        )
        if not output_candidates:
            return PhcResult(
                success=False,
                log="\n".join(log_lines) + "\nNo output file found.",
            )

        output_path = max(output_candidates, key=lambda p: p.stat().st_mtime)

        if progress_cb:
            progress_cb(0.95, "PHC remote complete.")

        return PhcResult(
            success=True, output_path=output_path, log="\n".join(log_lines)
        )

    except subprocess.TimeoutExpired:
        return PhcResult(
            success=False, log="\n".join(log_lines) + "\nSSH/SCP timed out."
        )
    except subprocess.CalledProcessError as exc:
        return PhcResult(
            success=False,
            log="\n".join(log_lines) + f"\nSubprocess failed: {exc.stderr}",
        )
    except Exception as exc:
        return PhcResult(
            success=False, log="\n".join(log_lines) + f"\nError: {exc}"
        )
