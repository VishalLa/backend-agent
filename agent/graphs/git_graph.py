from __future__ import annotations

from typing import Any, Optional

from config import Config

from .base_graph import BaseAgent


class GitAgent(BaseAgent):
    """Inspect, branch, commit, and push through the git-only tool profile."""

    TASK_MODE = "git"
    MD_FILENAME = "git_agent.md"
    FALLBACK_SYSTEM_PROMPT = (
        "You are a git workflow agent. Check status and diff before committing, write "
        "commit messages that explain why, and never push without explicit human approval."
    )


def build_git_graph(
    config: Config,
    tools: list[Any],
    checkpointer: Optional[Any] = None,
) -> Any:
    """Create a compiled git-agent graph."""
    return GitAgent(config, tools, checkpointer).graph
