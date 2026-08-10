from __future__ import annotations

from typing import Literal, Optional

from langchain_core.tools import tool

from sandbox import SandboxError, run_code_once


@tool
def execute_in_sandbox(
    code: str,
    language: Literal["python", "shell"] = "python",
    timeout: int = 30,
    filename: Optional[str] = None,
) -> str:
    """Execute Python or shell code in a temporary, network-isolated Docker sandbox.

    The sandbox has no access to the host project directory, network, or host
    credentials. Use this for self-contained experiments and verification.
    For code that depends on repository files, create the needed small fixture
    directly in the submitted code.

    Args:
        code: Python source or shell script to execute.
        language: Either ``python`` or ``shell``.
        timeout: Maximum execution time in seconds (1-120).
        filename: Optional filename under the sandbox's temporary workspace.
    """
    try:
        result = run_code_once(code, language=language, timeout=timeout)

    except SandboxError as exc:
        return f"ERROR: sandbox unavailable: {exc}"

    except Exception as exc:  # noqa: BLE001 - tool failures are returned to the agent.
        return f"ERROR: sandbox execution failed: {exc}"

    status = "timed out" if result.timed_out else f"exit code {result.exit_code}"
    parts = [f"SANDBOX: {status} ({result.duration_ms} ms)"]

    if result.stdout:
        parts.append(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr}")
    return "\n".join(parts)
