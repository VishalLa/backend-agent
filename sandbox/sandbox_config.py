from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SandboxConfig:

    image_tag: str = "agent-sandbox:latest"

    # resource limits
    mem_limit: str = "512m"
    memswap_limit: Optional[str] = None
    nano_cpus: int = 1_000_000_000
    pids_limit: int = 128
    workspace_tmpfs_size_mb: int = 256

    # network / filesystem isolation
    network_disabled: bool = True
    internal_network_name: str = "agent-sandbox-net"
    read_only_rootfs: bool = True

    # lifecycle
    startup_timeout_s: float = 15.0
    default_exec_timeout_s: int = 30
    idle_container_ttl_s: float = 600.0

    # security hardening passed straight to the docker daemon
    cap_drop: list[str] = field(default_factory=lambda: ["ALL"])
    security_opt: list[str] = field(default_factory=lambda: ["no-new-privileges:true"])
    user: str = "sandbox"

    def container_kwargs(self) -> dict:
        """Translate this config into docker-py `containers.run(**kwargs)` args."""
        tmpfs = {
            "/tmp": f"size={self.workspace_tmpfs_size_mb}m,mode=1777",
        }
        if self.read_only_rootfs:
            # /workspace also needs to be writable even with a read-only rootfs.
            tmpfs["/workspace"] = f"size={self.workspace_tmpfs_size_mb}m,uid=1000,gid=1000,mode=0755"

        return dict(
            mem_limit=self.mem_limit,
            memswap_limit=self.memswap_limit or self.mem_limit,
            nano_cpus=self.nano_cpus,
            pids_limit=self.pids_limit,
            network_disabled=self.network_disabled,
            read_only=self.read_only_rootfs,
            tmpfs=tmpfs,
            cap_drop=self.cap_drop,
            security_opt=self.security_opt,
            user=self.user,
            detach=True,
            auto_remove=False,
        )
