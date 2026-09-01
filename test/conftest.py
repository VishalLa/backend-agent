"""
Fixtures for running REAL integration tests against an already-running
sandbox container (as opposed to test_sandbox_manager_mocked.py, which
mocks the docker daemon entirely).

Points at your currently running container by default:
    ddeb335ed1af / hungry_leakey

Override via env vars if you rebuild/restart it:
    SANDBOX_CONTAINER=<id-or-name>   # which container to test against
    SANDBOX_BASE_URL=http://...      # skip discovery entirely, use this URL
"""
from __future__ import annotations

import os

import docker
import pytest
import requests

DEFAULT_CONTAINER_REF = "agent-sandbox"
SANDBOX_PORT = 8787


@pytest.fixture(scope="session")
def docker_client():
    try:
        return docker.from_env()
    except docker.errors.DockerException as e:
        pytest.skip(f"cannot reach the Docker daemon from this machine: {e}")


@pytest.fixture(scope="session")
def container(docker_client):
    ref = os.environ.get("SANDBOX_CONTAINER", DEFAULT_CONTAINER_REF)
    try:
        c = docker_client.containers.get(ref)
    except docker.errors.NotFound:
        pytest.fail(
            f"no container found matching {ref!r}. Is it still running? "
            f"Check `docker ps` and set SANDBOX_CONTAINER if the ID/name changed."
        )
    c.reload()
    if c.status != "running":
        pytest.fail(f"container {ref} exists but is not running (status={c.status}).")
    return c


@pytest.fixture(scope="session")
def base_url(container):
    explicit = os.environ.get("SANDBOX_BASE_URL")
    if explicit:
        url = explicit.rstrip("/")
    else:
        # Prefer a published host port if one exists...
        port_bindings = container.attrs["NetworkSettings"]["Ports"].get(f"{SANDBOX_PORT}/tcp")
        if port_bindings:
            host_port = port_bindings[0]["HostPort"]
            url = f"http://127.0.0.1:{host_port}"
        else:
            # ...otherwise fall back to the container's internal bridge IP.
            # Works when this test runner is on the same Docker host (the
            # default bridge network routes container IPs like 172.17.0.x).
            networks = container.attrs["NetworkSettings"]["Networks"]
            ip = next((n["IPAddress"] for n in networks.values() if n.get("IPAddress")), None)
            if not ip:
                pytest.fail(
                    f"container {container.short_id} has no published port and no "
                    f"reachable network IP. Re-run it with `-p {SANDBOX_PORT}:{SANDBOX_PORT}`, "
                    f"or set SANDBOX_BASE_URL manually."
                )
            url = f"http://{ip}:{SANDBOX_PORT}"

    # Fail fast with a clear message rather than every test timing out individually.
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as e:
        pytest.fail(f"could not reach sandbox at {url}/health: {e}")

    return url


@pytest.fixture
def clean_workspace(base_url):
    """Ensure each test starts from an empty /workspace, and clean up after."""
    requests.post(f"{base_url}/reset", timeout=10)
    yield
    requests.post(f"{base_url}/reset", timeout=10)