import re
import time
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from .config import AgentConfig
from .confirmation import needs_confirmation
from .llm import get_llm_with_tools
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


def _parse_retry_after(msg: str) -> float | None:
    match = _RETRY_AFTER_RE.search(msg)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


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
    "section across multiple tool calls rather than one large completion."
)


def _truncate(text: object, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    text = "" if text is None else str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


MAX_TOOL_CALL_ARG_CHARS = 300


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
    arguments shortened (see MAX_TOOL_CALL_ARG_CHARS). Only affects what's
    sent on the wire for this request — state.messages (used for tool
    execution, logging, etc.) is untouched."""
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


def build_graph(config: AgentConfig, tools: list):
    """Compile the agent graph for a given config + toolset. Call once per
    process per (config, toolset) pair — the graph object itself is cheap to
    reuse across many run_agent() calls with different thread_ids."""
    names = [t.name for t in tools]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate tool names in toolset: {duplicates}")
    if not tools:
        raise ValueError("tools list must not be empty")

    llm_with_tools = get_llm_with_tools(config, tools)
    tools_by_name = {t.name: t for t in tools}

    # ---- nodes -------------------------------------------------------

    def agent_node(state: AgentState) -> dict:
        # Loop guard: stop calling the model once we've hit the cap
        if state.iterations >= config.max_iterations:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"Stopping: reached the max_iterations limit ({config.max_iterations}) "
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

        last_error: Exception | None = None
        rate_limit_retries_left = 3
        attempt = 0
        while True:
            try:
                response = llm_with_tools.invoke(model_input)
                return {"messages": [response], "iterations": state.iterations + 1}
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                if any(marker in msg for marker in CONTEXT_LENGTH_MARKERS):
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
                    wait = min((retry_after or config.retry_backoff_seconds) + 0.5, MAX_RATE_LIMIT_WAIT_SECONDS)
                    print(f"[rate limited — waiting {wait:.1f}s before retrying the model call]")
                    time.sleep(wait)
                    rate_limit_retries_left -= 1
                    continue

                if not is_retryable or attempt >= config.max_retries:
                    break
                time.sleep(config.retry_backoff_seconds * (2**attempt))
                attempt += 1

        return {
            "messages": [AIMessage(content=f"Critical API Error: Both primary (Groq) and fallback (OpenRouter) models failed to respond or hit unrecoverable limits. Details: {last_error}")],
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

    def tools_node(state: AgentState) -> dict:
        messages = list(state.messages)
        last_ai = _last_ai_message_with_tool_calls(messages)
        tool_calls = last_ai.tool_calls if last_ai else []
        already_answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}

        new_messages: list[BaseMessage] = []
        logs: list[dict] = []
        for tc in tool_calls:
            call_id = tc["id"]
            if call_id in already_answered:
                continue
            name = tc["name"]
            args = tc.get("args", {}) or {}
            tool = tools_by_name.get(name)

            if tool is None:
                # The model hallucinated a tool name — don't crash, tell it what's actually available.
                output = f"ERROR: unknown tool '{name}'. Available tools: {', '.join(sorted(tools_by_name))}"
                success = False
            else:
                try:
                    raw_output = tool.invoke(args)
                    output = _truncate(raw_output)
                    success = not (isinstance(raw_output, str) and raw_output.startswith(("ERROR", "BLOCKED")))
                except Exception as e:
                    output = f"ERROR: tool '{name}' raised an exception: {e}"
                    success = False

            new_messages.append(ToolMessage(content=output, tool_call_id=call_id, name=name))
            logs.append(
                ToolCallLog(
                    call_id=call_id, tool_name=name, args=args, result=output, confirmed=True, success=success
                ).model_dump()
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
