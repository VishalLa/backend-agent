import re
import time
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from .config import AgentConfig
from .confirmation import needs_confirmation
from .llm import get_llm_with_tools
from .logging_utils import log_event, safe_args
from .schemas import AgentState, ConfirmationRequest, ToolCallLog

MAX_TOOL_OUTPUT_CHARS = 3000

RETRYABLE_ERROR_MARKERS = (
    "rate limit",
    "rate_limit",
    "429",
    "timeout",
    "timed out",
    "connection",
    "502",
    "503",
    "overloaded",
)
CONTEXT_LENGTH_MARKERS = (
    "context length",
    "context_length",
    "too many tokens",
    "maximum context",
    "request too large",
    "reduce your message size",
    "413",
)

_RETRY_AFTER_RE = re.compile(r"try again in\s+([\d.]+)\s*s", re.IGNORECASE)
MAX_RATE_LIMIT_WAIT_SECONDS = 90.0


def _parse_retry_after(msg: str) -> Optional[float]:
    match = _RETRY_AFTER_RE.search(msg)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _thread_id(run_config: Optional[dict]) -> str:
    """Pull thread_id out of the RunnableConfig LangGraph injects into any
    node function whose signature accepts a second argument."""
    try:
        return (run_config or {}).get("configurable", {}).get("thread_id", "unknown")
    except Exception:  # noqa: BLE001
        return "unknown"


def _identify_provider(response: Any, config: AgentConfig) -> str:
    """Best-effort guess at which tier of the Groq -> OpenRouter -> Ollama
    fallback chain actually answered — with_fallbacks() doesn't tag this
    directly, so it's inferred from response metadata. Logging visibility
    only; never used for control flow."""
    meta = getattr(response, "response_metadata", {}) or {}
    model = meta.get("model_name") or meta.get("model") or ""
    if model == config.model_name:
        return f"groq:{model}"
    if model == config.fallback_model_name:
        return f"openrouter:{model}"
    if model == config.ollama_model or "done_reason" in meta:  # ollama-specific metadata key
        return f"ollama:{model or config.ollama_model}"
    return model or "unknown"


SYSTEM_PROMPT = (
    "You are a local coding agent for Python backend (Flask/FastAPI) and ML/data work. "
    "Use the available tools to inspect, edit, and verify code. Prefer targeted edit_file "
    "patches over rewriting whole files. Run tests/lint/type-checks after edits when those "
    "tools are available. Shell commands, background job launches, deletions, git pushes, "
    "and overwriting existing files require human confirmation. If one is declined, do not "
    "immediately retry the same call — explain the block to the user and propose a safer "
    "alternative or ask how to proceed.\n\n"
    "IMPORTANT — large files: the model backing this agent has a limited tokens-per-request "
    "budget. Never generate a large file (roughly 150+ lines) in a single write_file call. "
    "Instead: call write_file with the first chunk (e.g. imports, first class/function or "
    "section, ~100-150 lines), then call append_file one or more times to add the rest in "
    "similarly sized chunks. Plan the file's structure first, then write it section by "
    "section across multiple tool calls rather than one large completion.\n\n"
    "IMPORTANT — tools: only ever call a tool using its exact name from the tools provided "
    "for this request. Never invent, guess, or reuse a tool name from a different framework "
    "or convention that wasn't given to you.\n\n"
    "IMPORTANT — reading many files (e.g. summarizing/mapping a whole directory or building a "
    "diagram across many models): do NOT read every file in full up front. Use ripgrep_search or "
    "list_dir first to scope down to what's relevant, then read_file with an explicit start_line/"
    "end_line range (a few hundred lines at a time) rather than the whole file. Process files one "
    "or a few at a time and summarize what you learned in your own words before moving on, instead "
    "of keeping every full file's content in play — this avoids hitting token-per-minute/context "
    "limits on long exploratory tasks."
)


def _truncate(text: object, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    text = "" if text is None else str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


MAX_TOOL_CALL_ARG_CHARS = 300

RECENT_TOOL_RESULTS_KEPT_FULL = 4
COLLAPSED_TOOL_RESULT_CHARS = 300


def _collapse_old_tool_results(messages: list[BaseMessage]) -> list[BaseMessage]:
    tool_msg_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    collapse_indices = set(tool_msg_indices[:-RECENT_TOOL_RESULTS_KEPT_FULL]) if len(
        tool_msg_indices
    ) > RECENT_TOOL_RESULTS_KEPT_FULL else set()
    if not collapse_indices:
        return messages

    prepared = []
    for i, m in enumerate(messages):
        if i in collapse_indices:
            content = str(m.content)
            short = content[:COLLAPSED_TOOL_RESULT_CHARS]
            m = m.model_copy(
                update={
                    "content": (
                        f"{short}\n... [collapsed: {len(content) - len(short)} older chars omitted — "
                        "this was an earlier tool result no longer kept in full; re-run the tool "
                        "call if you need the complete output again]"
                    )
                    if len(content) > len(short)
                    else content
                }
            )
        prepared.append(m)
    return prepared


def _slim_tool_calls(tool_calls: list) -> list:
    slimmed = []
    changed = False
    for tc in tool_calls:
        args = dict(tc.get("args") or {})
        for k, v in list(args.items()):
            if isinstance(v, str) and len(v) > MAX_TOOL_CALL_ARG_CHARS:
                args[k] = v[:MAX_TOOL_CALL_ARG_CHARS] + f"...[{len(v) - MAX_TOOL_CALL_ARG_CHARS} chars omitted — already executed, see tool result]"
                changed = True
        slimmed.append({**tc, "args": args})
    return slimmed if changed else tool_calls


def _prepare_messages_for_model(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Return a copy of the history to send to the LLM with large tool-call
    arguments shortened and older tool RESULTS collapsed. Only affects what's
    sent on the wire for this request — state.messages (used for tool
    execution, logging, etc.) is untouched."""
    messages = _collapse_old_tool_results(messages)
    prepared = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            new_tool_calls = _slim_tool_calls(m.tool_calls)
            if new_tool_calls is not m.tool_calls:
                m = m.model_copy(update={"tool_calls": new_tool_calls})
        prepared.append(m)
    return prepared


def _last_ai_message_with_tool_calls(messages: list[BaseMessage]):
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            return m
    return None


def build_graph(agent_config: AgentConfig, tools: list):
    """Compile the agent graph for a given config + toolset. Call once per
    process per (config, toolset) pair — the graph object itself is cheap to
    reuse across many run_agent() calls with different thread_ids."""
    names = [t.name for t in tools]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate tool names in toolset: {duplicates}")
    if not tools:
        raise ValueError("tools list must not be empty")

    llm_with_tools = get_llm_with_tools(agent_config, tools)
    tools_by_name = {t.name: t for t in tools}

    # ---- nodes -------------------------------------------------------

    def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        thread_id = _thread_id(config)

        # Loop guard: stop calling the model once we've hit the cap.
        if state.iterations >= agent_config.max_iterations:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"Stopping: reached the max_iterations limit ({agent_config.max_iterations}) "
                            "without finishing. Try narrowing the request, or re-run with a higher limit."
                        )
                    )
                ],
                "status": "max_iterations_reached",
            }

        messages = list(state.messages)
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        model_input = _prepare_messages_for_model(messages)

        last_error: Optional[Exception] = None
        rate_limit_retries_left = 3
        attempt = 0
        while True:
            try:
                response = llm_with_tools.invoke(model_input)
                log_event(
                    agent_config.log_file, "llm_call", thread_id=thread_id,
                    provider=_identify_provider(response, agent_config),
                )
                return {"messages": [response], "iterations": state.iterations + 1}
            except Exception as e:  # noqa: BLE001 - provider errors are heterogeneous by design
                last_error = e
                msg = str(e).lower()
                if any(marker in msg for marker in CONTEXT_LENGTH_MARKERS):
                    log_event(
                        agent_config.log_file, "llm_call_failed", thread_id=thread_id,
                        reason="context_too_large", error=str(e),
                    )
                    return {
                        "messages": [
                            AIMessage(
                                content=(
                                    "This conversation (or a single tool result in it) is too large for "
                                    "the model's per-request/context limit. Start a new thread (/new), or "
                                    "summarize progress so far and continue there."
                                )
                            )
                        ],
                        "iterations": state.iterations + 1,
                        "status": "error",
                        "error": str(e),
                    }

                retry_after = _parse_retry_after(msg)
                is_rate_limit = retry_after is not None or "rate_limit" in msg or "rate limit" in msg
                is_retryable = is_rate_limit or any(marker in msg for marker in RETRYABLE_ERROR_MARKERS)

                if is_rate_limit and rate_limit_retries_left > 0:
                    wait = min((retry_after or agent_config.retry_backoff_seconds) + 0.5, MAX_RATE_LIMIT_WAIT_SECONDS)
                    print(f"[rate limited — waiting {wait:.1f}s before retrying the model call]")
                    log_event(agent_config.log_file, "llm_rate_limited_wait", thread_id=thread_id, wait_seconds=wait)
                    time.sleep(wait)
                    rate_limit_retries_left -= 1
                    continue

                if not is_retryable or attempt >= agent_config.max_retries:
                    break
                time.sleep(agent_config.retry_backoff_seconds * (2**attempt))
                attempt += 1

        log_event(
            agent_config.log_file, "llm_call_failed", thread_id=thread_id,
            reason="all_providers_exhausted", error=str(last_error),
        )
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I hit an error calling the model across every configured provider "
                        f"(Groq, OpenRouter, and local Ollama — whichever are set up) and "
                        f"couldn't recover: {last_error}"
                    )
                )
            ],
            "iterations": state.iterations + 1,
            "status": "error",
            "error": str(last_error),
        }

    def confirm_node(state: AgentState) -> dict:
        last_ai = _last_ai_message_with_tool_calls(list(state.messages))
        tool_calls = last_ai.tool_calls if last_ai else []

        new_messages: list[BaseMessage] = []
        for tc in tool_calls:
            args = tc.get("args", {}) or {}
            if not needs_confirmation(tc["name"], args):
                continue
            request = ConfirmationRequest(
                tool_name=tc["name"],
                tool_args=args,
                call_id=tc["id"],
            )

            raw = interrupt(request.model_dump())
            approved = bool(isinstance(raw, dict) and raw.get("approved"))
            if not approved:
                reason = (raw.get("reason") if isinstance(raw, dict) else None) or "declined by the user"
                new_messages.append(
                    ToolMessage(
                        content=(
                            f"BLOCKED: {reason}. Do not retry this exact command — "
                            "explain the block to the user or propose a safer alternative."
                        ),
                        tool_call_id=tc["id"],
                        name=tc["name"],
                    )
                )
        return {"messages": new_messages} if new_messages else {}

    def tools_node(state: AgentState, config: RunnableConfig) -> dict:
        thread_id = _thread_id(config)
        messages = list(state.messages)
        last_ai = _last_ai_message_with_tool_calls(messages)
        tool_calls = last_ai.tool_calls if last_ai else []
        answered = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}

        new_messages: list[BaseMessage] = []
        logs: list[dict] = []
        for tc in tool_calls:
            call_id = tc["id"]
            name = tc["name"]
            args = tc.get("args", {}) or {}

            if call_id in answered:

                blocked_msg = answered[call_id]
                log_event(
                    agent_config.log_file, "tool_call_blocked", thread_id=thread_id,
                    tool_name=name, args=safe_args(args), result=str(blocked_msg.content)[:500],
                )
                continue

            tool = tools_by_name.get(name)

            if tool is None:
                output = f"ERROR: unknown tool '{name}'. Available tools: {', '.join(sorted(tools_by_name))}"
                success = False
            else:
                try:
                    raw_output = tool.invoke(args)
                    output = _truncate(raw_output)
                    success = not (isinstance(raw_output, str) and raw_output.startswith(("ERROR", "BLOCKED")))
                except Exception as e:  # noqa: BLE001 - bad args, tool bugs, etc. must not crash the graph
                    output = f"ERROR: tool '{name}' raised an exception: {e}"
                    success = False

            new_messages.append(ToolMessage(content=output, tool_call_id=call_id, name=name))
            logs.append(
                ToolCallLog(
                    call_id=call_id, tool_name=name, args=args, result=output, confirmed=True, success=success
                ).model_dump()
            )
            log_event(
                agent_config.log_file, "tool_call", thread_id=thread_id, tool_name=name,
                args=safe_args(args), success=success, result=output[:500],
            )

        return {"messages": new_messages, "tool_log": logs}

    # ---- routing -------------------------------------------------------

    def route_after_agent(state: AgentState) -> Literal["confirm", "tools", "__end__"]:
        if state.status in ("max_iterations_reached", "error"):
            return "__end__"
        last = state.messages[-1] if state.messages else None
        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            return "__end__"
        if any(needs_confirmation(tc["name"], tc.get("args", {}) or {}) for tc in tool_calls):
            return "confirm"
        return "tools"

    # ---- wiring -------------------------------------------------------

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("confirm", confirm_node)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"confirm": "confirm", "tools": "tools", "__end__": END})
    builder.add_edge("confirm", "tools")
    builder.add_edge("tools", "agent")

    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)
