import subprocess

from langchain_core.tools import tool

MAX_OUTPUT_CHARS = 4000


@tool
def run_shell_command(
    command: str,
    cwd: str = None,
    timeout: int = 60,
) -> str:
    """Execute a shell command and return its stdout, stderr, and exit code.

    Use this for running tests, installing packages, running linters/type
    checkers, or any other one-shot CLI operation. Do NOT use this for
    long-running commands like model training — it blocks until the command
    finishes or times out. Use launch_background_process for those instead.

    Args:
        command: The shell command to execute.
        cwd: Working directory to run the command in. Defaults to the
            current directory.
        timeout: Max seconds to wait before killing the process and
            returning an error. Defaults to 60.
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
