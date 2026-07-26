import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

from langchain_core.tools import tool

LOG_DIR = Path("/tmp/agent_bg_jobs")


@tool
def check_gpu_status() -> str:
    """Check GPU availability, VRAM usage, and utilization via nvidia-smi.
    Run this before launching a training job to avoid OOM crashes caused by
    stacking jobs on an already-busy GPU.
    """
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
    """Launch a long-running command (e.g. a model training run) in the
    background and detach immediately, instead of blocking the tool call.
    Output is redirected to a log file; use tail_log to check progress later.

    Args:
        command: Shell command to run in the background.
        job_name: Label for the job, used to name its log file. A random id
            is generated if not given — remember it to check logs later.
        cwd: Working directory to launch the process from.
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
    """Read the last N lines of a background job's log file to check progress.

    Args:
        job_name: The job_name used when launching it with launch_background_process.
        n_lines: Number of lines to read from the end of the log. Defaults to 50.
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
    """Delete a file or directory. This is destructive and is on the
    confirm-before-run list: it REQUIRES confirm=True. Never set
    confirm=True on your own initiative — only after the user has
    explicitly agreed to the deletion.

    Args:
        path: File or directory path to delete.
        confirm: Must be explicitly True to actually delete. Defaults to False.
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
        