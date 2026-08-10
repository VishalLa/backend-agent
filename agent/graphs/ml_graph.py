from __future__ import annotations

from typing import Any, Optional

from config import Config

from .base_graph import BaseAgent


class MLAgent(BaseAgent):
    """Training, evaluation, data-pipeline, and experiment workflow."""

    TASK_MODE = "ml"


def build_ml_graph(
    config: Config,
    tools: list[Any],
    checkpointer: Optional[Any] = None,
) -> Any:
    """Create a compiled ML-agent graph."""
    return MLAgent(config, tools, checkpointer).graph
