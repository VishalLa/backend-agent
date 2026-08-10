from __future__ import annotations

import subprocess

from langchain_core.tools import tool

MAX_OUTPUT_CHARS = 4000


@tool
def run_shell_command(
    command: str,
    cwd: str = None,
    timeout: int = 60,
) -> str:
    """Run a shell command; returns stdout/stderr/exit code. For one-shot
    CLI ops (tests, installs, linters). NOT for long-running commands — use
    launch_background_process instead.

    Args:
        command: Shell command to run.
        cwd: Working directory. Default current dir.
        timeout: Max seconds before killing it. Default 60.
    """

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"

    parts = [f"EXIT_CODE: {result.returncode}"]
    if result.stdout:
        parts.append(f"STDOUT:\n{result.stdout[-MAX_OUTPUT_CHARS:]}")
    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr[-MAX_OUTPUT_CHARS:]}")
    return "\n".join(parts)
