from __future__ import annotations

from typing import Any, Optional

from config import Config

from .base_graph import BaseAgent


class BackendAgent(BaseAgent):
    """Flask/FastAPI and general backend-development workflow."""

    TASK_MODE = "backend"


def build_backend_graph(
    config: Config,
    tools: list[Any],
    checkpointer: Optional[Any] = None,
) -> Any:
    """Create a compiled backend-agent graph."""
    return BackendAgent(config, tools, checkpointer).graph
