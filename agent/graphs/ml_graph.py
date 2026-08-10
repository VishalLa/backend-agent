from __future__ import annotations

from typing import Any, Optional

from config import Config

from .base_graph import BaseAgent


class MLAgent(BaseAgent):
    """Training, evaluation, data-pipeline, and experiment workflow."""

    TASK_MODE = "ml"
    MD_FILENAME = "ml_agent.md"
    FALLBACK_SYSTEM_PROMPT = (
        "You are an ML/AI engineering agent. Check GPU status before launching training, "
        "use the persistent kernel for iteration and background jobs for long runs, and "
        "confirm results from actual tool output before reporting them."
    )


def build_ml_graph(
    config: Config,
    tools: list[Any],
    checkpointer: Optional[Any] = None,
) -> Any:
    """Create a compiled ML-agent graph."""
    return MLAgent(config, tools, checkpointer).graph
