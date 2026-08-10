from __future__ import annotations

from typing import Any, Optional

from config import Config

from .base_graph import BaseAgent


class AlgoAgent(BaseAgent):
    """Correctness- and complexity-sensitive algorithm-development workflow."""

    TASK_MODE = "algorithms"
    MD_FILENAME = "algo_agent.md"
    FALLBACK_SYSTEM_PROMPT = (
        "You are an algorithms agent. State complexity targets before implementing, test "
        "edge cases, run the code to verify correctness, and state the final complexity "
        "explicitly."
    )


def build_algo_graph(
    config: Config,
    tools: list[Any],
    checkpointer: Optional[Any] = None,
) -> Any:
    """Create a compiled algorithms-agent graph."""
    return AlgoAgent(config, tools, checkpointer).graph
