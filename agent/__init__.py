"""Local coding agent: binds tools/ALL_TOOLS to a shared Groq LLM via a
LangGraph graph, with pydantic-validated config/state/results and a human
confirmation gate on shell commands and other destructive actions.

Usage:
    from agent import run_agent
    result = run_agent("list the files in this repo")
    print(result.output)
"""

from .config import AgentConfig
from .confirmation import ALWAYS_CONFIRM_TOOLS, default_cli_confirmation_handler
from .runner import run_agent
from .schemas import AgentResult, AgentState, ConfirmationDecision, ConfirmationRequest, ToolCallLog

__all__ = [
    "AgentConfig",
    "AgentResult",
    "AgentState",
    "ConfirmationDecision",
    "ConfirmationRequest",
    "ToolCallLog",
    "ALWAYS_CONFIRM_TOOLS",
    "default_cli_confirmation_handler",
    "run_agent",
]
