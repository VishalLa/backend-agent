from __future__ import annotations

from typing import Any, Optional

from config import Config

from .base_graph import BaseAgent


class BackendAgent(BaseAgent):
    """Flask/FastAPI and general backend-development workflow."""

    TASK_MODE = "backend"
    MD_FILENAME = "backend_agent.md"
    FALLBACK_SYSTEM_PROMPT = (
        "You are a backend coding agent. Inspect before editing, use only the provided "
        "tools, and verify changes with tests or an HTTP request when appropriate. "
        "Never claim a change succeeded unless its tool result confirms it."
    )


def build_backend_graph(
    config: Config,
    tools: list[Any],
    checkpointer: Optional[Any] = None,
) -> Any:
    """Create a compiled backend-agent graph."""
    return BackendAgent(config, tools, checkpointer).graph
