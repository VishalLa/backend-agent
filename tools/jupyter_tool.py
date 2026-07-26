import atexit
import base64
import queue
import time
import uuid
from pathlib import Path

from jupyter_client import KernelManager
from langchain_core.tools import tool

_kernels: dict[str, KernelManager] = {}
IMAGE_DIR = Path("/tmp/kernel_outputs")


def _get_kernel(project_id: str) -> KernelManager:
    if project_id not in _kernels:
        km = KernelManager()
        km.start_kernel()
        _kernels[project_id] = km
    return _kernels[project_id]


@atexit.register
def _shutdown_all_kernels() -> None:
    """Best-effort cleanup so kernel subprocesses don't linger as orphans
    after the agent process exits. Registered once at import time."""
    for km in list(_kernels.values()):
        try:
            km.shutdown_kernel(now=True)
        except Exception:  # noqa: BLE001 - cleanup path, never raise on the way out
            pass


def _collect_outputs(kc, timeout: int) -> dict:
    stdout, stderr, result, error = "", "", "", ""
    images: list[str] = []
    timed_out = True
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=timeout)
        except queue.Empty:
            break
        msg_type = msg["header"]["msg_type"]
        content = msg["content"]
        if msg_type == "stream":
            if content["name"] == "stdout":
                stdout += content["text"]
            else:
                stderr += content["text"]
        elif msg_type in ("execute_result", "display_data"):
            data = content.get("data", {})
            if "text/plain" in data:
                result += data["text/plain"]
            if "image/png" in data:
                images.append(data["image/png"])
        elif msg_type == "error":
            error = "\n".join(content.get("traceback", []))
        elif msg_type == "status" and content.get("execution_state") == "idle":
            timed_out = False
            break
    return {
        "stdout": stdout,
        "stderr": stderr,
        "result": result,
        "error": error,
        "images": images,
        "timed_out": timed_out,
    }


@tool
def execute_code(code: str, project_id: str = "default", timeout: int = 120) -> str:
    """Execute Python in a persistent Jupyter kernel; state persists across
    calls sharing the same project_id (don't re-import/reload). Use
    launch_background_process for long-running training instead.

    Args:
        code: Python code to run.
        project_id: Kernel identifier; keep consistent per project.
        timeout: Max seconds to wait.
    """
    km = _get_kernel(project_id)
    kc = km.client()
    kc.start_channels()
    try:
        try:
            kc.wait_for_ready(timeout=30)
        except RuntimeError as e:
            return f"ERROR: kernel '{project_id}' did not become ready in time ({e}). Try restart_kernel and retry."
        kc.execute(code)
        outputs = _collect_outputs(kc, timeout=timeout)
        if outputs["timed_out"]:
            # The kernel is still running this cell in the background — if we
            # don't interrupt it, its output will leak into the next call.
            km.interrupt_kernel()
    finally:
        kc.stop_channels()

    parts = []
    if outputs["timed_out"]:
        parts.append(
            f"ERROR: execution exceeded {timeout}s timeout; kernel was interrupted. "
            "Any partial output below completed before the interrupt."
        )
    if outputs["stdout"]:
        parts.append(f"STDOUT:\n{outputs['stdout']}")
    if outputs["result"]:
        parts.append(f"RESULT:\n{outputs['result']}")
    if outputs["stderr"]:
        parts.append(f"STDERR:\n{outputs['stderr']}")
    if outputs["error"]:
        parts.append(f"ERROR:\n{outputs['error']}")
    if outputs["images"]:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for img_b64 in outputs["images"]:
            unique = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
            img_path = IMAGE_DIR / f"{project_id}_{unique}_{len(saved)}.png"
            img_path.write_bytes(base64.b64decode(img_b64))
            saved.append(str(img_path))
        parts.append(f"IMAGES SAVED: {', '.join(saved)}")

    return "\n\n".join(parts) if parts else "(no output)"


@tool
def restart_kernel(project_id: str = "default") -> str:
    """Restart a project's kernel, clearing all state. Use if it's in a
    bad state (GPU OOM, hung) or you need a clean environment.

    Args:
        project_id: Kernel identifier to restart.
    """
    if project_id in _kernels:
        _kernels[project_id].shutdown_kernel(now=True)
        del _kernels[project_id]
    _get_kernel(project_id)
    return f"OK: kernel '{project_id}' restarted"
