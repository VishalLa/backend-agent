from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# sys.path.insert(0, str(Path(__file__).parent.parent))

from sandbox.sandbox_config import SandboxConfig
from sandbox.sandbox_manager import DockerSandbox, SandboxStartupError, ensure_image_built
from docker.errors import ImageNotFound


def test_ensure_image_built_skips_when_image_exists():
    client = MagicMock()
    client.images.get.return_value = object()
    ensure_image_built(client, SandboxConfig())
    client.images.build.assert_not_called()
    print("OK: ensure_image_built skips build when image already exists")


def test_ensure_image_built_builds_when_missing():
    client = MagicMock()
    client.images.get.side_effect = ImageNotFound("nope")
    ensure_image_built(client, SandboxConfig())
    client.images.build.assert_called_once()
    print("OK: ensure_image_built builds when image is missing")


def test_container_kwargs_hardening():
    cfg = SandboxConfig()
    kwargs = cfg.container_kwargs()
    assert kwargs["network_disabled"] is True, "network must be disabled by default"
    assert kwargs["read_only"] is True, "rootfs must be read-only by default"
    assert kwargs["cap_drop"] == ["ALL"], "all capabilities must be dropped"
    assert "no-new-privileges:true" in kwargs["security_opt"]
    assert kwargs["user"] == "sandbox", "must not run as root"
    assert kwargs["pids_limit"] == 128
    assert kwargs["memswap_limit"] == kwargs["mem_limit"], "swap must not exceed mem_limit (no swap escape)"
    assert "/workspace" in kwargs["tmpfs"], "workspace must be a size-capped tmpfs when rootfs is read-only"
    print("OK: container_kwargs applies expected hardening defaults")


def test_start_waits_for_health_then_succeeds():
    client = MagicMock()
    client.images.get.return_value = object()  # image already built
    fake_container = MagicMock()
    client.containers.run.return_value = fake_container

    with patch("sandbox.sandbox_manager.requests.get") as mock_get, \
         patch("sandbox.sandbox_manager._free_port", return_value=54321):
        mock_get.return_value.status_code = 200
        sandbox = DockerSandbox(config=SandboxConfig(startup_timeout_s=2), client=client)
        sandbox.start()

        client.containers.run.assert_called_once()
        _, run_kwargs = client.containers.run.call_args
        assert run_kwargs["ports"] == {"8787/tcp": 54321}
        assert sandbox._base_url == "http://127.0.0.1:54321"
        print("OK: start() maps a free host port and waits for /health")

    sandbox.stop()
    fake_container.stop.assert_called_once()
    fake_container.remove.assert_called_once()
    print("OK: stop() stops and removes the container")


def test_start_raises_and_cleans_up_if_never_healthy():
    client = MagicMock()
    client.images.get.return_value = object()
    fake_container = MagicMock()
    fake_container.logs.return_value = b"container never came up"
    client.containers.run.return_value = fake_container

    import requests as requests_module
    with patch("sandbox.sandbox_manager.requests.get",
               side_effect=requests_module.exceptions.ConnectionError("connection refused")), \
         patch("sandbox.sandbox_manager._free_port", return_value=54322), \
         patch("sandbox.sandbox_manager.time.sleep"):  # skip real waiting
        sandbox = DockerSandbox(config=SandboxConfig(startup_timeout_s=0.01), client=client)
        try:
            sandbox.start()
            assert False, "expected SandboxStartupError"
        except SandboxStartupError as e:
            assert "container logs" in str(e).lower() or "logs" in str(e).lower()
            print("OK: start() raises SandboxStartupError with logs when health check never succeeds")

    fake_container.stop.assert_called_once()
    fake_container.remove.assert_called_once()
    print("OK: failed startup still cleans up the half-started container")


def test_exec_posts_expected_payload():
    client = MagicMock()
    client.images.get.return_value = object()
    fake_container = MagicMock()
    client.containers.run.return_value = fake_container

    with patch("sandbox.sandbox_manager.requests.get") as mock_get, \
         patch("sandbox.sandbox_manager.requests.post") as mock_post, \
         patch("sandbox.sandbox_manager._free_port", return_value=54323):
        mock_get.return_value.status_code = 200
        mock_post.return_value.raise_for_status = lambda: None
        mock_post.return_value.json.return_value = {
            "stdout": "4\n", "stderr": "", "exit_code": 0, "timed_out": False, "duration_ms": 12
        }

        with DockerSandbox(config=SandboxConfig(startup_timeout_s=2), client=client) as sandbox:
            result = sandbox.exec("print(2+2)", timeout=10)
            assert result.success
            assert result.stdout == "4\n"

            _, post_kwargs = mock_post.call_args
            assert post_kwargs["json"]["code"] == "print(2+2)"
            assert post_kwargs["json"]["timeout"] == 10
            assert post_kwargs["timeout"] == 15, "HTTP-level timeout must exceed in-container timeout"
            print("OK: exec() posts correct payload and enforces a larger HTTP-level timeout")


if __name__ == "__main__":
    test_ensure_image_built_skips_when_image_exists()
    test_ensure_image_built_builds_when_missing()
    test_container_kwargs_hardening()
    test_start_waits_for_health_then_succeeds()
    test_start_raises_and_cleans_up_if_never_healthy()
    test_exec_posts_expected_payload()
    print("\nALL MOCKED TESTS PASSED")