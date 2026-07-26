import json
import re
import time
import uuid
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from .config import AgentConfig
from .confirmation import needs_confirmation
from .llm import get_llm_with_tools
from .logging_utils import log_event, safe_args
from .schemas import AgentState, ConfirmationRequest, ToolCallLog

MAX_TOOL_OUTPUT_CHARS = 1200

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

TOOL_NAME_ERROR_MARKERS = (
    "not in request.tools",
    "tool_use_failed",
    "which was not in request",
)
MAX_TOOL_HALLUCINATION_RETRIES = 2
MAX_EMPTY_RESPONSE_RETRIES = 2

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


def _find_json_object(text: str) -> Optional[str]:
    """Find the first balanced {...} substring in text (brace-counting, so
    it handles nested braces correctly, unlike a naive regex)."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_leaked_tool_call(text: str, valid_tool_names: set) -> Optional[dict]:
    """Some models (especially smaller local ones via Ollama) occasionally
    write out a tool call as plain text content — e.g.
    '{"name": "read_file", "arguments": {"path": "x.py"}}' — instead of
    using the model's real structured tool-calling mechanism. Left alone,
    this looks like empty tool_calls + a final text answer, so the graph
    ends the run with that JSON string as the "answer" and nothing actually
    happens. Detect that shape and repair it into a real tool call so
    tools_node can execute it normally.

    Returns a langchain-style tool_call dict ({"name", "args", "id"}) if a
    valid one was found, else None (in which case the text is left as a
    normal — if odd-looking — final answer)."""
    if "{" not in text or '"name"' not in text:
        return None
    blob = _find_json_object(text)
    if not blob:
        return None
    try:
        parsed = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    if not isinstance(name, str) or name not in valid_tool_names:
        return None
    args = parsed.get("arguments", parsed.get("args", parsed.get("parameters", {})))
    if not isinstance(args, dict):
        args = {}
    return {"name": name, "args": args, "id": f"repaired-{uuid.uuid4().hex[:8]}"}


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
    "You are a local coding agent for Python backend/ML work. Use tools to inspect, edit, "
    "verify code. Prefer targeted edit_file patches over rewriting whole files. Run tests/"
    "lint after edits when available. Shell, background jobs, deletions, git push, and "
    "file overwrite need human confirmation — if declined, don't retry; explain and propose "
    "an alternative.\n"
    "Large files: never write 150+ lines in one write_file call — write_file for the first "
    "chunk, then append_file for the rest in similar-sized chunks.\n"
    "Only call tools by their exact given names, never invented ones.\n"
    "Scanning many files (e.g. mapping a whole directory): use ripgrep_search/list_dir to "
    "scope down first, then read_file in ranges, one file at a time — don't read everything "
    "in full, it will exceed the per-request token budget."
)


def _truncate(text: object, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    text = "" if text is None else str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


MAX_TOOL_CALL_ARG_CHARS = 300

RECENT_TOOL_RESULTS_KEPT_FULL = 2
COLLAPSED_TOOL_RESULT_CHARS = 150


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
        tool_hallucination_retries_left = MAX_TOOL_HALLUCINATION_RETRIES
        empty_response_retries_left = MAX_EMPTY_RESPONSE_RETRIES
        attempt = 0
        while True:
            try:
                response = llm_with_tools.invoke(model_input)
                has_tool_calls = bool(getattr(response, "tool_calls", None))
                content_text = response.content if isinstance(response.content, str) else str(response.content or "")

                if not has_tool_calls and content_text.strip():
                    leaked = _extract_leaked_tool_call(content_text, set(tools_by_name))
                    if leaked:
                        log_event(
                            agent_config.log_file, "llm_tool_call_leaked_as_text",
                            thread_id=thread_id, tool_name=leaked["name"],
                        )
                        response = AIMessage(
                            content="",
                            tool_calls=[leaked],
                            response_metadata=getattr(response, "response_metadata", {}) or {},
                        )
                        has_tool_calls = True

                if not has_tool_calls and not content_text.strip():
                    log_event(agent_config.log_file, "llm_empty_response", thread_id=thread_id)
                    if empty_response_retries_left > 0:
                        empty_response_retries_left -= 1
                        model_input = model_input + [
                            HumanMessage(
                                content=(
                                    "SYSTEM: your last response was empty — no tool call and no answer "
                                    "text. Continue the task now: call the next tool you need (e.g. "
                                    "write_file if you were asked to save output to a file), or if the "
                                    "task is genuinely finished, write your final answer as plain text."
                                )
                            )
                        ]
                        continue
                    # retries exhausted — return the empty response rather than looping forever;
                    # run_end will show tool_call_count/iterations so this is at least diagnosable.

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

                if any(marker in msg for marker in TOOL_NAME_ERROR_MARKERS):
                    log_event(
                        agent_config.log_file, "llm_tool_hallucination", thread_id=thread_id, error=str(e),
                    )
                    if tool_hallucination_retries_left > 0:
                        tool_hallucination_retries_left -= 1
                        valid_names = ", ".join(sorted(tools_by_name))
                        model_input = model_input + [
                            HumanMessage(
                                content=(
                                    "SYSTEM: your last tool call used a tool name that doesn't exist "
                                    f"in this toolset. The ONLY valid tool names are exactly: "
                                    f"{valid_names}. Call one of these exactly (spelled and cased as "
                                    "given, no prefixes/namespaces), or reply with plain text if you "
                                    "don't actually need a tool."
                                )
                            )
                        ]
                        continue
                    # retries exhausted — fall through to the generic failure path below

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
