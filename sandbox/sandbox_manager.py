from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import docker
import requests
from docker.errors import BuildError, DockerException, ImageNotFound, NotFound
from docker.models.containers import Container

from .sandbox_config import SandboxConfig

logger = logging.getLogger("sandbox")

IMAGE_DIR = Path(__file__).parent


class SandboxError(RuntimeError):
    """Base class for sandbox failures (build, startup, exec, etc.)."""


class SandboxStartupError(SandboxError):
    pass


class SandboxExecError(SandboxError):
    pass


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    duration_ms: int

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


def _free_port() -> int:
    """Ask the OS for an unused localhost port, to map the container's fixed
    internal port (8787) onto something free on the host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ensure_image_built(client: "docker.DockerClient", config: SandboxConfig, force: bool = False) -> None:
    """Build the sandbox image from ./image if it doesn't already exist (or
    if force=True). Safe to call on every startup - it's a no-op when the
    image is already present and force=False."""
    if not force:
        try:
            client.images.get(config.image_tag)
            return
        except ImageNotFound:
            pass

    logger.info("Building sandbox image %s from %s ...", config.image_tag, IMAGE_DIR)
    try:
        client.images.build(path=str(IMAGE_DIR), tag=config.image_tag, rm=True)
    except BuildError as e:
        raise SandboxError(f"failed to build sandbox image: {e}") from e


class DockerSandbox:
    """
    Manages the lifecycle of one hardened sandbox container.

    Use as a context manager so the container is guaranteed to be cleaned up
    even if the agent loop raises:

        with DockerSandbox(config) as sandbox:
            sandbox.exec(code)
    """

    def __init__(self, config: Optional[SandboxConfig] = None, client: Optional["docker.DockerClient"] = None):
        self.config = config or SandboxConfig()
        try:
            self.client = client or docker.from_env()

        except DockerException as e:
            raise SandboxStartupError(
                "could not connect to the Docker daemon - is Docker running, "
                "and does this user have permission to talk to it?"
            ) from e

        self._container: Optional[Container] = None
        self._host_port: Optional[int] = None

    # lifecycle

    def start(self) -> "DockerSandbox":
        ensure_image_built(self.client, self.config)

        self._host_port = _free_port()
        kwargs = self.config.container_kwargs()
        kwargs["ports"] = {"8787/tcp": self._host_port}

        logger.info("Starting sandbox container on host port %s ...", self._host_port)
        self._container = self.client.containers.run(self.config.image_tag, **kwargs)

        self._wait_healthy()
        return self


    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_s
        url = f"{self._base_url}/health"
        last_error: Optional[Exception] = None

        while time.monotonic() < deadline:
            try:
                resp = requests.get(url, timeout=1.0)
                if resp.status_code == 200:
                    return
            except requests.RequestException as e:
                last_error = e
            time.sleep(0.25)

        # Grab logs before tearing down so failures are debuggable.
        logs = self._safe_logs()
        self.stop()
        raise SandboxStartupError(
            f"sandbox did not become healthy within {self.config.startup_timeout_s}s "
            f"(last error: {last_error}). Container logs:\n{logs}"
        )


    def stop(self) -> None:
        if self._container is None:
            return
        try:
            self._container.stop(timeout=5)
        except (DockerException, NotFound):
            pass
        try:
            self._container.remove(force=True)
        except (DockerException, NotFound):
            pass
        logger.info("Sandbox container stopped and removed.")
        self._container = None
        self._host_port = None


    def _safe_logs(self) -> str:
        if self._container is None:
            return ""
        try:
            return self._container.logs(tail=200).decode(errors="replace")
        except DockerException:
            return "<could not retrieve logs>"


    def __enter__(self) -> "DockerSandbox":
        return self.start()


    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


    # exec API
    @property
    def _base_url(self) -> str:
        if self._host_port is None:
            raise SandboxError("sandbox is not running - call start() or use as a context manager")
        return f"http://127.0.0.1:{self._host_port}"


    def exec(
        self,
        code: str,
        language: str = "python",
        timeout: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> ExecResult:
        """Run `code` inside the sandbox and return its output.

        `timeout` bounds both the in-container subprocess AND this HTTP
        call (with a small grace period), so a hung container can't hang
        the agent loop.
        """
        timeout = timeout or self.config.default_exec_timeout_s
        try:
            resp = requests.post(
                f"{self._base_url}/exec",
                json={"code": code, "language": language, "timeout": timeout, "filename": filename},
                timeout=timeout + 5,  # grace period beyond the in-container timeout
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise SandboxExecError(f"exec request failed: {e}") from e

        data = resp.json()
        return ExecResult(**data)


    def upload(self, filename: str, content: str) -> None:
        """Write a file into the sandbox's /workspace (e.g. a multi-file
        project the agent is testing). Path traversal is rejected server-side."""
        try:
            resp = requests.post(
                f"{self._base_url}/upload",
                json={"filename": filename, "content": content},
                timeout=10,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise SandboxExecError(f"upload failed: {e}") from e


    def reset(self) -> None:
        """Wipe /workspace clean without paying container-startup cost again -
        use this between unrelated tasks in the same session."""
        try:
            resp = requests.post(f"{self._base_url}/reset", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise SandboxExecError(f"reset failed: {e}") from e


def run_code_once(
    code: str,
    language: str = "python",
    config: Optional[SandboxConfig] = None,
    timeout: Optional[int] = None,
) -> ExecResult:
    """Convenience one-shot helper: spins up a fresh container, runs one
    snippet, tears the container down. Matches the 'ephemeral container per
    task' behavior from the workflow diagram - use DockerSandbox directly
    instead if you want to run several snippets without paying container
    startup cost each time."""
    with DockerSandbox(config) as sandbox:
        return sandbox.exec(code, language=language, timeout=timeout)
