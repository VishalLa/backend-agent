import uuid
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from .config import AgentConfig
from .confirmation import default_cli_confirmation_handler
from .graph import build_graph
from .schemas import AgentResult, ConfirmationDecision, ConfirmationRequest, ToolCallLog

ConfirmHandler = Callable[[ConfirmationRequest], ConfirmationDecision]

_graph_cache: dict[tuple, Any] = {}


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field off either a dict or a pydantic/object state
    representation — LangGraph's exact return shape for get_state().values
    can vary by version, so don't assume one or the other."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _get_or_build_graph(config: AgentConfig, tools: list):
    tool_names = tuple(sorted(t.name for t in tools))
    cache_key = (config.model_name, config.temperature, config.max_tokens, tool_names)
    if cache_key not in _graph_cache:
        _graph_cache[cache_key] = build_graph(config, tools)
    return _graph_cache[cache_key]


def run_agent(
    prompt: str,
    config: Optional[AgentConfig] = None,
    tools: Optional[list] = None,
    confirm_handler: Optional[ConfirmHandler] = None,
    thread_id: Optional[str] = None,
) -> AgentResult:
    """Run the agent to completion on a single prompt.

    Any tool call to a confirmation-required tool (see
    agent.confirmation.needs_confirmation) pauses execution and calls
    confirm_handler(request) to get a decision; the default handler prompts
    on stdin. Pass your own handler to wire this into a web UI, Slack, etc.
    — its only contract is (ConfirmationRequest) -> ConfirmationDecision.

    thread_id lets you resume/continue a specific conversation later; a
    fresh one is generated if not given.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    thread_id = thread_id or str(uuid.uuid4())

    try:
        config = config or AgentConfig.from_env()
    except Exception as e:  # noqa: BLE001 - bad/missing env config
        return AgentResult(output="", status="error", iterations=0, thread_id=thread_id, error=f"config error: {e}")

    if tools is None:
        try:
            from tools import ALL_TOOLS  # sibling package from the tools deliverable
        except ImportError as e:
            return AgentResult(
                output="",
                status="error",
                iterations=0,
                thread_id=thread_id,
                error=f"couldn't import tools.ALL_TOOLS ({e}); pass tools= explicitly or fix PYTHONPATH",
            )
        tools = ALL_TOOLS

    confirm_handler = confirm_handler or default_cli_confirmation_handler

    try:
        graph = _get_or_build_graph(config, tools)
    except Exception as e:  # noqa: BLE001 - bad toolset (dupes, empty, etc.)
        return AgentResult(output="", status="error", iterations=0, thread_id=thread_id, error=f"failed to build graph: {e}")

    thread_cfg = {"configurable": {"thread_id": thread_id}}
    step_input: Any = {"messages": [HumanMessage(content=prompt)]}

    try:
        while True:
            graph.invoke(step_input, config=thread_cfg)
            snapshot = graph.get_state(thread_cfg)

            if not snapshot.next:
                break  # no pending node — the run finished

            if not snapshot.tasks or not snapshot.tasks[0].interrupts:
                # Defensive: paused with nothing to resume. Shouldn't happen,
                # but bail cleanly instead of looping forever.
                break

            request = ConfirmationRequest.model_validate(snapshot.tasks[0].interrupts[0].value)
            try:
                decision = confirm_handler(request)
            except (KeyboardInterrupt, EOFError):
                decision = ConfirmationDecision(approved=False, reason="cancelled — no confirmation given")
            except Exception as e:  # noqa: BLE001 - a broken custom confirm_handler shouldn't crash the run
                decision = ConfirmationDecision(approved=False, reason=f"confirmation handler failed: {e}")

            step_input = Command(resume=decision.model_dump())
    except Exception as e:  # noqa: BLE001 - last-resort net around the whole graph run
        return AgentResult(output="", status="error", iterations=0, thread_id=thread_id, error=f"agent run failed: {e}")

    final = graph.get_state(thread_cfg).values
    messages = _field(final, "messages", []) or []
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    output = last_ai.content if last_ai is not None else ""

    status = _field(final, "status", "completed") or "completed"
    if status == "running":  # graph ended without anyone marking a terminal status
        status = "completed"

    raw_tool_log = _field(final, "tool_log", []) or []
    tool_calls = [d if isinstance(d, ToolCallLog) else ToolCallLog.model_validate(d) for d in raw_tool_log]

    return AgentResult(
        output=output if isinstance(output, str) else str(output),
        status=status,
        iterations=_field(final, "iterations", 0) or 0,
        tool_calls=tool_calls,
        thread_id=thread_id,
        error=_field(final, "error", None),
    )
