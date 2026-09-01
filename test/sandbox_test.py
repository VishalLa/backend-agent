"""
Real integration tests against the live sandbox container.

Run:
    pip install pytest requests docker
    pytest -v tests_live/

Point at a different container:
    SANDBOX_CONTAINER=<id-or-name> pytest -v tests_live/

Skip past discovery entirely (e.g. you published a port yourself):
    SANDBOX_BASE_URL=http://127.0.0.1:54321 pytest -v tests_live/
"""
from __future__ import annotations

import time

import pytest
import requests


# ---------------------------------------------------------------------------
# Functional correctness
# ---------------------------------------------------------------------------

def test_health(base_url):
    r = requests.get(f"{base_url}/health", timeout=5)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_exec_basic_arithmetic(base_url, clean_workspace):
    r = requests.post(
        f"{base_url}/exec",
        json={"code": "print(6 * 7)", "language": "python", "timeout": 10},
        timeout=15,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["stdout"] == "42\n"
    assert body["exit_code"] == 0
    assert body["timed_out"] is False


def test_exec_captures_stderr_and_nonzero_exit(base_url, clean_workspace):
    r = requests.post(
        f"{base_url}/exec",
        json={"code": "raise ValueError('boom')", "language": "python", "timeout": 10},
        timeout=15,
    )
    body = r.json()
    assert body["exit_code"] != 0
    assert "ValueError: boom" in body["stderr"]
    assert body["timed_out"] is False


def test_exec_timeout_is_enforced(base_url, clean_workspace):
    start = time.monotonic()
    r = requests.post(
        f"{base_url}/exec",
        json={"code": "import time; time.sleep(30)", "language": "python", "timeout": 2},
        timeout=15,
    )
    elapsed = time.monotonic() - start
    body = r.json()
    assert body["timed_out"] is True
    assert body["exit_code"] != 0
    # Should be killed close to the 2s timeout, not run the full 30s sleep.
    assert elapsed < 10, f"took {elapsed:.1f}s - timeout enforcement looks broken"


def test_upload_then_import_module(base_url, clean_workspace):
    r = requests.post(
        f"{base_url}/upload",
        json={"filename": "helper_mod.py", "content": "def add(a, b):\n    return a + b\n"},
        timeout=10,
    )
    assert r.status_code == 200

    r = requests.post(
        f"{base_url}/exec",
        json={"code": "from helper_mod import add\nprint(add(2, 3))", "language": "python", "timeout": 10},
        timeout=15,
    )
    body = r.json()
    assert body["exit_code"] == 0, f"stderr: {body['stderr']}"
    assert body["stdout"] == "5\n"


def test_upload_path_traversal_is_rejected(base_url, clean_workspace):
    r = requests.post(
        f"{base_url}/upload",
        json={"filename": "../../etc/evil.py", "content": "x = 1"},
        timeout=10,
    )
    assert r.status_code == 400
    assert "escapes workspace" in r.json()["detail"]


def test_exec_filename_path_traversal_is_rejected(base_url, clean_workspace):
    r = requests.post(
        f"{base_url}/exec",
        json={"code": "x = 1", "language": "python", "timeout": 10, "filename": "../../tmp/evil.py"},
        timeout=15,
    )
    assert r.status_code == 400


def test_reset_actually_clears_workspace(base_url, clean_workspace):
    requests.post(f"{base_url}/upload", json={"filename": "leftover.py", "content": "x=1"}, timeout=10)
    r = requests.post(f"{base_url}/reset", timeout=10)
    assert r.status_code == 200

    # File should be gone - importing it should now fail.
    r = requests.post(
        f"{base_url}/exec",
        json={"code": "import leftover", "language": "python", "timeout": 10},
        timeout=15,
    )
    body = r.json()
    assert body["exit_code"] != 0
    assert "ModuleNotFoundError" in body["stderr"]


def test_output_is_truncated_for_huge_prints(base_url, clean_workspace):
    r = requests.post(
        f"{base_url}/exec",
        json={
            "code": "print('x' * 300_000)",
            "language": "python",
            "timeout": 10,
        },
        timeout=15,
    )
    body = r.json()
    assert len(body["stdout"]) < 300_000, "server should truncate huge stdout, not return it raw"
    assert "truncated" in body["stdout"]


# ---------------------------------------------------------------------------
# Isolation / hardening
#
# These only pass if the container was started with the hardening flags from
# SandboxConfig.container_kwargs() (network_disabled, read_only rootfs,
# cap_drop=ALL, mem/pids limits). A plain `docker run -d agent-sandbox` will
# fail some of these - that's the tests correctly telling you the container
# isn't locked down the way sandbox_manager.py intends.
# ---------------------------------------------------------------------------

def test_runs_as_non_root(base_url, clean_workspace):
    """Baked into the image (Dockerfile USER sandbox) - should pass
    regardless of how the container was launched."""
    r = requests.post(
        f"{base_url}/exec",
        json={"code": "import os; print(os.getuid())", "language": "python", "timeout": 10},
        timeout=15,
    )
    body = r.json()
    uid = int(body["stdout"].strip())
    assert uid != 0, "sandbox is running as root - USER directive in Dockerfile isn't taking effect"


def test_network_is_disabled(base_url, clean_workspace):
    """Requires the container to have been started with network_disabled=True
    (or --network none). If this fails, the running container has network
    access and generated code could exfiltrate data or fetch a payload."""
    r = requests.post(
        f"{base_url}/exec",
        json={
            "code": (
                "import urllib.request\n"
                "try:\n"
                "    urllib.request.urlopen('http://example.com', timeout=5)\n"
                "    print('NETWORK_REACHABLE')\n"
                "except Exception as e:\n"
                "    print(f'NETWORK_BLOCKED: {type(e).__name__}')\n"
            ),
            "language": "python",
            "timeout": 15,
        },
        timeout=20,
    )
    body = r.json()
    assert "NETWORK_BLOCKED" in body["stdout"], (
        "sandbox can reach the internet - if this container wasn't started with "
        "network_disabled=True / --network none, this failure is expected. "
        f"stdout={body['stdout']!r} stderr={body['stderr']!r}"
    )


def test_rootfs_outside_workspace_is_read_only(base_url, clean_workspace):
    """Requires the container to have been started with read_only=True.
    If this fails, generated code can write anywhere in the container's
    filesystem (still not your host, but a bigger blast radius than intended)."""
    r = requests.post(
        f"{base_url}/exec",
        json={
            "code": (
                "try:\n"
                "    open('/app/should_not_be_writable.txt', 'w').write('x')\n"
                "    print('WRITE_SUCCEEDED')\n"
                "except OSError as e:\n"
                "    print(f'WRITE_BLOCKED: {e}')\n"
            ),
            "language": "python",
            "timeout": 10,
        },
        timeout=15,
    )
    body = r.json()
    assert "WRITE_BLOCKED" in body["stdout"], (
        "sandbox rootfs is writable outside /workspace - container likely wasn't "
        f"started with read_only=True. stdout={body['stdout']!r}"
    )


def test_workspace_itself_is_writable(base_url, clean_workspace):
    """Sanity check that hardening didn't over-lock things - /workspace must
    stay writable even with a read-only rootfs (it's a tmpfs mount)."""
    r = requests.post(
        f"{base_url}/exec",
        json={"code": "open('/workspace/ok.txt', 'w').write('fine')\nprint('OK')", "language": "python", "timeout": 10},
        timeout=15,
    )
    body = r.json()
    assert body["exit_code"] == 0, f"workspace should be writable: {body['stderr']}"


@pytest.mark.slow
def test_memory_limit_kills_runaway_allocation(base_url, clean_workspace):
    """Requires mem_limit to actually be applied by the container runtime.
    Allocates well beyond a typical 256-512m limit; expects the process to be
    killed (nonzero/negative exit code), not to succeed."""
    r = requests.post(
        f"{base_url}/exec",
        json={
            "code": "data = bytearray(2 * 1024 * 1024 * 1024)  # 2GiB\nprint('ALLOCATED')",
            "language": "python",
            "timeout": 20,
        },
        timeout=25,
    )
    body = r.json()
    assert "ALLOCATED" not in body["stdout"], (
        "sandbox allowed a 2GiB allocation to succeed - mem_limit likely isn't "
        f"applied on this container. stdout={body['stdout']!r} exit_code={body['exit_code']}"
    )


@pytest.mark.slow
def test_pids_limit_blocks_fork_bomb(base_url, clean_workspace):
    """Requires pids_limit to actually be applied. Tries to spawn far more
    processes than the configured limit (default 128) and expects failure."""
    r = requests.post(
        f"{base_url}/exec",
        json={
            "code": (
                "import subprocess\n"
                "procs = []\n"
                "try:\n"
                "    for _ in range(500):\n"
                "        procs.append(subprocess.Popen(['sleep', '5']))\n"
                "    print('SPAWNED_ALL')\n"
                "except OSError as e:\n"
                "    print(f'BLOCKED: {e}')\n"
                "finally:\n"
                "    for p in procs:\n"
                "        p.kill()\n"
            ),
            "language": "python",
            "timeout": 20,
        },
        timeout=25,
    )
    body = r.json()
    assert "SPAWNED_ALL" not in body["stdout"], (
        "sandbox allowed spawning 500 processes - pids_limit likely isn't "
        f"applied on this container. stdout={body['stdout']!r}"
    )
