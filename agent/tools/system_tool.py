import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

from langchain_core.tools import tool

LOG_DIR = Path("/tmp/agent_bg_jobs")


@tool
def check_gpu_status() -> str:
    """Check GPU/VRAM usage via nvidia-smi. Run before launching a training
    job to avoid stacking jobs on an already-busy GPU."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"ERROR: {result.stderr.strip()}"
        return result.stdout.strip()
    except FileNotFoundError:
        return "No NVIDIA GPU / nvidia-smi found on this machine."
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


@tool
def launch_background_process(command: str, job_name: str = None, cwd: str = None) -> str:
    """Launch a long-running command in the background (detached, doesn't
    block). Output goes to a log file; check with tail_log.

    Args:
        command: Shell command to run in background.
        job_name: Job label/log filename. Random id if omitted.
        cwd: Working directory.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    job_id = job_name or uuid.uuid4().hex[:8]
    log_path = LOG_DIR / f"{job_id}.log"
    # Wrap in `sh -c` so compound commands (cd x && ..., loops, pipes) are
    # interpreted by a shell instead of nohup trying to exec a shell
    # keyword/builtin (e.g. "cd") directly as a program.
    full_cmd = f"nohup sh -c {shlex.quote(command)} > {shlex.quote(str(log_path))} 2>&1 &"
    try:
        subprocess.Popen(full_cmd, shell=True, cwd=cwd)
        return f"OK: launched job '{job_id}', logging to {log_path}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


@tool
def tail_log(job_name: str, n_lines: int = 50) -> str:
    """Read the last N lines of a background job's log.

    Args:
        job_name: Job name used in launch_background_process.
        n_lines: Lines to read from the end. Default 50.
    """
    log_path = LOG_DIR / f"{job_name}.log"
    if not log_path.exists():
        available = ", ".join(p.stem for p in LOG_DIR.glob("*.log")) if LOG_DIR.exists() else ""
        return f"ERROR: no log found for job '{job_name}'. Available logs: {available or '(none)'}"
    try:
        result = subprocess.run(["tail", "-n", str(n_lines), str(log_path)], capture_output=True, text=True)
        return result.stdout
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


@tool
def delete_path(path: str, confirm: bool = False) -> str:
    """Delete a file/directory. REQUIRES confirm=True (only after the user
    explicitly agrees) — destructive.

    Args:
        path: File/directory path to delete.
        confirm: Must be True to actually delete. Default False.
    """
    if not confirm:
        return f"BLOCKED: delete_path requires explicit confirm=True to delete '{path}'. Confirm with the user first."
    p = Path(path)
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return f"OK: deleted {path}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"
        