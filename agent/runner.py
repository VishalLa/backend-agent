import uuid
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from .config import AgentConfig
from .confirmation import default_cli_confirmation_handler
from .graph import build_graph
from .logging_utils import DEFAULT_LOG_FILE, log_event
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
    cache_key = (
        config.provider_mode, config.model_name, config.fallback_model_name, config.ollama_model,
        config.enable_ollama_fallback, config.ollama_num_ctx, config.ollama_num_predict,
        config.ollama_keep_alive, config.ollama_num_thread,
        config.temperature, config.max_tokens, tool_names,
    )
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

    Every meaningful step of this run (start, each LLM call and which
    provider tier answered, each tool call, confirmation decisions, and the
    final outcome) is appended to config.log_file as JSON lines — see
    agent.logging_utils.log_event for the schema.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    thread_id = thread_id or str(uuid.uuid4())

    try:
        config = config or AgentConfig.from_env()
    except Exception as e:  # noqa: BLE001 - bad/missing env config
        log_event(DEFAULT_LOG_FILE, "run_failed", thread_id=thread_id, reason="config_error", error=str(e))
        return AgentResult(output="", status="error", iterations=0, thread_id=thread_id, error=f"config error: {e}")

    log_event(config.log_file, "run_start", thread_id=thread_id, prompt=prompt[:500], provider_mode=config.provider_mode)

    if tools is None:
        try:
            from tools import ALL_TOOLS  # sibling package from the tools deliverable
        except ImportError as e:
            log_event(config.log_file, "run_failed", thread_id=thread_id, reason="tools_import_error", error=str(e))
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
        log_event(config.log_file, "run_failed", thread_id=thread_id, reason="graph_build_error", error=str(e))
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

            # Logged here (once per real decision) rather than inside
            # graph.py's confirm_node, which can re-execute several times
            # per decision due to LangGraph's interrupt()-replay semantics.
            log_event(
                config.log_file, "confirmation_decision", thread_id=thread_id,
                tool_name=request.tool_name, approved=decision.approved, reason=decision.reason,
            )
            step_input = Command(resume=decision.model_dump())

    except Exception as e:  # noqa: BLE001 - last-resort net around the whole graph run
        log_event(config.log_file, "run_failed", thread_id=thread_id, reason="graph_execution_crashed", error=str(e))
        return AgentResult(
            output="Agent crashed during execution. Check the log file for details.",
            status="error",
            iterations=0,
            thread_id=thread_id,
            error=f"Graph execution failed. Details: {str(e)}",
        )

    final = graph.get_state(thread_cfg).values
    messages = _field(final, "messages", []) or []
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    output = last_ai.content if last_ai is not None else ""

    status = _field(final, "status", "completed") or "completed"
    if status == "running":  # graph ended without anyone marking a terminal status
        status = "completed"

    raw_tool_log = _field(final, "tool_log", []) or []
    tool_calls = [d if isinstance(d, ToolCallLog) else ToolCallLog.model_validate(d) for d in raw_tool_log]

    result = AgentResult(
        output=output if isinstance(output, str) else str(output),
        status=status,
        iterations=_field(final, "iterations", 0) or 0,
        tool_calls=tool_calls,
        thread_id=thread_id,
        error=_field(final, "error", None),
    )
    log_event(
        config.log_file, "run_end", thread_id=thread_id, status=result.status,
        iterations=result.iterations, tool_call_count=len(result.tool_calls), error=result.error,
    )
    return result
    