from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

WORKSPACE = Path("/workspace")
WORKSPACE.mkdir(parents=True, exist_ok=True)

MAX_OUTPUT_CHARS = 200_000
DEFAULT_TIMEOUT_S = 30
HARD_TIMEOUT_CEILING_S = 120

app = FastAPI(title="agent-sandbox", version="1.0")


class ExecRequest(BaseModel):
    code: str = Field(..., description="Python source to execute, or a shell command if language='shell'")
    language: str = Field("python", pattern="^(python|shell)$")
    timeout: int = Field(DEFAULT_TIMEOUT_S, ge=1, le=HARD_TIMEOUT_CEILING_S)
    filename: Optional[str] = Field(
        None, description="Optional file name (under /workspace) to write the code to before running it"
    )


class ExecResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: int


class UploadRequest(BaseModel):
    filename: str
    content: str


def _safe_workspace_path(filename: str) -> Path:
    """Resolve `filename` under WORKSPACE, rejecting any attempt to escape it
    (../, absolute paths, symlink tricks)."""
    candidate = (WORKSPACE / filename).resolve()
    try:
        candidate.relative_to(WORKSPACE.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"path escapes workspace: {filename!r}")
    return candidate


def _truncate(s: str) -> str:
    if len(s) > MAX_OUTPUT_CHARS:
        return s[:MAX_OUTPUT_CHARS] + f"\n...[truncated, {len(s) - MAX_OUTPUT_CHARS} more chars]"
    return s


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    """Wipe the workspace between unrelated tasks without restarting the container."""
    for entry in WORKSPACE.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
    return {"status": "reset"}


@app.post("/upload")
def upload(req: UploadRequest):
    path = _safe_workspace_path(req.filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(req.content)
    return {"status": "written", "path": str(path.relative_to(WORKSPACE))}


@app.post("/exec", response_model=ExecResponse)
def exec_code(req: ExecRequest):
    start = time.monotonic()

    if req.filename:
        target = _safe_workspace_path(req.filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(req.code)
        argv = ["python3", str(target)] if req.language == "python" else ["sh", str(target)]
        cwd = str(WORKSPACE)
    else:
        suffix = ".py" if req.language == "python" else ".sh"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=str(WORKSPACE))
        with os.fdopen(fd, "w") as f:
            f.write(req.code)
        argv = ["python3", tmp_path] if req.language == "python" else ["sh", tmp_path]
        cwd = str(WORKSPACE)

    timed_out = False
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=req.timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(WORKSPACE)},
        )
        stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + f"\n[sandbox] killed after {req.timeout}s timeout"
        exit_code = -1
    finally:
        if not req.filename:
            # Ephemeral run - don't let the temp file linger in /workspace.
            Path(tmp_path).unlink(missing_ok=True)

    duration_ms = int((time.monotonic() - start) * 1000)
    return ExecResponse(
        stdout=_truncate(stdout if isinstance(stdout, str) else stdout.decode(errors="replace")),
        stderr=_truncate(stderr if isinstance(stderr, str) else stderr.decode(errors="replace")),
        exit_code=exit_code,
        timed_out=timed_out,
        duration_ms=duration_ms,
    )
