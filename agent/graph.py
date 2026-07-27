import json
import re
import time
import uuid
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
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
    "disconnected",
    "connection refused",
    "broken pipe",
    "remote protocol error",
    "502",
    "503",
    "overloaded",
)

LOCAL_SERVER_STARTUP_MARKERS = ("disconnected", "connection refused", "broken pipe", "remote protocol error")
LOCAL_SERVER_COLD_START_WAIT_SECONDS = 15.0
CONTEXT_LENGTH_MARKERS = (
    "context length",
    "context_length",
    "too many tokens",
    "maximum context",
    "request too large",
    "reduce your message size",
    "413",
)

TOKEN_RATE_LIMIT_MARKERS = (
    "tokens per minute",
    "requests per minute",
    "tpm",
    "rpm)",
    "rate_limit_exceeded",
)

TOOL_NAME_ERROR_MARKERS = (
    "not in request.tools",
    "tool_use_failed",
    "which was not in request",
)
MAX_TOOL_HALLUCINATION_RETRIES = 2
MAX_EMPTY_RESPONSE_RETRIES = 2


READ_ONLY_TOOL_NAMES = {
    "read_file", "list_dir", "ripgrep_search", "git_status", "git_diff",
    "git_log", "git_branch", "web_search", "check_gpu_status", "tail_log",
    "fetch_openapi_schema", "http_request",
}
STAGNATION_REPEAT_THRESHOLD = 3

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


try:
    from langgraph.config import get_config as _lg_get_config
except ImportError:
    def _lg_get_config() -> dict:
        return {}


def _thread_id() -> str:
    """Current run's thread_id, read from LangGraph's own contextvar-based
    config accessor rather than an injected node-function parameter.

    Node functions can optionally accept a second parameter to receive the
    RunnableConfig, but LangGraph versions differ on whether it's passed
    positionally or as a keyword, and don't guarantee a specific parameter
    name is required — a mismatch there is a real "missing required
    argument" / "unexpected keyword argument" crash risk. get_config()
    reads directly from the contextvar LangGraph sets around every node
    call, independent of the node's own signature, so it works the same way
    regardless of that calling convention."""
    try:
        cfg = _lg_get_config()
        return (cfg or {}).get("configurable", {}).get("thread_id", "unknown")
    except Exception:  # noqa: BLE001 - logging/id lookup must never break the run
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
    "in full, it will exceed the per-request token budget.\n"
    "NEVER construct edit_file's old_str from memory of an earlier write_file/edit_file call or "
    "an older turn in this conversation — the arguments of older tool calls get automatically "
    "shortened as the conversation grows and are no longer reliable. Always call read_file on the "
    "exact path immediately before edit_file (even a file you just wrote) and copy old_str verbatim, "
    "including whitespace and newlines, from that fresh read_file result. If edit_file still reports "
    "old_str not found after doing this, re-read the file again rather than guessing at a fix.\n"
    "Never tell the user a file was created/updated/saved unless the corresponding write_file/"
    "append_file/edit_file call actually returned success — if every attempt at a requested file "
    "change failed, say so plainly and explain what went wrong instead of describing the change as done."
)


def _truncate(text: object, limit: int = MAX_TOOL_OUTPUT_CHARS, continuation_hint_tool: Optional[str] = None) -> str:
    """Truncate long tool output before it goes back to the model.

    For read_file specifically, a flat character cutoff with no pointer back
    into the file causes two distinct real failure modes seen in practice:
    a model that can't tell it only saw a partial file may either (a) keep
    re-reading with guessed-larger end_line values forever, since every read
    from line 1 returns the identical truncated head regardless of the range
    requested, or worse (b) treat the truncated slice as the whole file,
    make a correct-looking but tiny edit confined to what it could see, and
    report the (much larger) task as done. read_file's own output is
    formatted as "<line_number>\\t<content>" per line (seen consistently in
    practice), so we can parse the last fully-visible line number out of the
    truncated text and tell the model exactly where to resume — turning a
    silent, ambiguous cutoff into an explicit, actionable pointer.
    """
    text = "" if text is None else str(text)
    if len(text) <= limit:
        return text
    truncated = text[:limit]
    suffix = f"\n... [truncated {len(text) - limit} chars]"
    if continuation_hint_tool == "read_file":
        last_newline = truncated.rfind("\n")
        last_line = truncated[last_newline + 1 :] if last_newline != -1 else truncated
        m = re.match(r"(\d+)\t", last_line)
        if m:
            next_line = int(m.group(1)) + 1
            suffix += (
                f" — this file continues past line {m.group(1)}. It is NOT fully shown above; "
                f"call read_file again with start_line={next_line} to keep reading before editing "
                "or claiming the file is fully updated."
            )
        else:
            suffix += (
                " — this output was cut off and may not represent the full file/result; "
                "re-read with a later start_line if you need the rest before editing."
            )
    return truncated + suffix


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


def _detect_read_only_stagnation(tool_log: list, threshold: int = STAGNATION_REPEAT_THRESHOLD) -> Optional[str]:
    """If the last `threshold` tool calls were all the same read-only tool
    against the same target (its `path` arg, when present) with no mutating
    call in between, the model is very likely stuck probing rather than
    making progress — e.g. re-reading a file with different line ranges
    instead of ever calling edit_file/write_file. Returns a corrective nudge
    to append to model_input, or None if nothing looks stuck.

    Deliberately narrow (exact same tool + same target, back-to-back) to
    avoid false positives on legitimate multi-file exploration — reading
    several different files in a row is normal and untouched by this check.
    """
    if len(tool_log) < threshold:
        return None
    recent = tool_log[-threshold:]
    names = {d.get("tool_name") for d in recent}
    if len(names) != 1:
        return None
    (name,) = names
    if name not in READ_ONLY_TOOL_NAMES:
        return None
    targets = {(d.get("args") or {}).get("path") for d in recent}
    if len(targets) != 1:
        return None
    (target,) = targets
    where = f" on `{target}`" if target else ""
    return (
        f"SYSTEM: you've called '{name}'{where} {threshold} times in a row without making "
        "any change. You already have enough context from these reads — make the edit now "
        "with edit_file or write_file, or if you're genuinely blocked (e.g. the file looks "
        "inconsistent/corrupted), say so plainly instead of reading it again."
    )


_SHARED_CHECKPOINTER = InMemorySaver()


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

    def agent_node(state: AgentState) -> dict:
        thread_id = _thread_id()

        if state.status in ("cancelled", "error", "max_iterations_reached"):
            return {}

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

        stagnation_nudge = _detect_read_only_stagnation(state.tool_log)
        if stagnation_nudge:
            log_event(
                agent_config.log_file, "llm_read_only_stagnation_detected", thread_id=thread_id,
                tool_name=(state.tool_log[-1] or {}).get("tool_name") if state.tool_log else None,
            )
            model_input = model_input + [HumanMessage(content=stagnation_nudge)]

        last_error: Optional[Exception] = None
        retries_left = agent_config.max_retries
        retries_used = 0
        tool_hallucination_retries_left = MAX_TOOL_HALLUCINATION_RETRIES
        empty_response_retries_left = MAX_EMPTY_RESPONSE_RETRIES
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
            except KeyboardInterrupt:
                log_event(agent_config.log_file, "llm_call_cancelled", thread_id=thread_id)
                return {
                    "messages": [AIMessage(content="Cancelled (Ctrl-C) while waiting on the model.")],
                    "iterations": state.iterations + 1,
                    "status": "cancelled",
                    "error": "cancelled by user (Ctrl-C) during LLM call",
                }
            except Exception as e:  # noqa: BLE001 - provider errors are heterogeneous by design
                last_error = e
                msg = str(e).lower()
                is_token_rate_limit = any(marker in msg for marker in TOKEN_RATE_LIMIT_MARKERS)

                if any(marker in msg for marker in CONTEXT_LENGTH_MARKERS) and not is_token_rate_limit:
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
                is_rate_limit = (
                    retry_after is not None or "rate_limit" in msg or "rate limit" in msg or is_token_rate_limit
                )
                is_local_server_hiccup = agent_config.provider_mode == "local" and any(
                    marker in msg for marker in LOCAL_SERVER_STARTUP_MARKERS
                )
                is_retryable = (
                    is_rate_limit or is_local_server_hiccup or any(marker in msg for marker in RETRYABLE_ERROR_MARKERS)
                )

                if not is_retryable or retries_left <= 0:
                    break

                if is_rate_limit:
                    wait = min((retry_after or agent_config.retry_backoff_seconds) + 0.5, MAX_RATE_LIMIT_WAIT_SECONDS)
                    print(f"[rate limited — waiting {wait:.1f}s before retrying the model call]")
                    log_event(agent_config.log_file, "llm_rate_limited_wait", thread_id=thread_id, wait_seconds=wait)
                    time.sleep(wait)
                elif is_local_server_hiccup:
                    wait = LOCAL_SERVER_COLD_START_WAIT_SECONDS * (retries_used + 1)
                    print(f"[local model server not responding yet (still loading?) — waiting {wait:.1f}s before retrying]")
                    log_event(
                        agent_config.log_file, "llm_local_server_cold_start_wait", thread_id=thread_id, wait_seconds=wait,
                    )
                    time.sleep(wait)
                else:
                    time.sleep(agent_config.retry_backoff_seconds * (2**retries_used))

                retries_left -= 1
                retries_used += 1

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

    def _dedup_and_validate_tool_calls(tool_calls: list, thread_id: str) -> list:
        """Filter out malformed tool calls (missing id/name — not impossible
        from a weaker fallback tier producing an oddly-shaped response) and
        drop duplicate call_ids (a model emitting the same call twice would
        otherwise execute it twice and produce two ToolMessages with the
        same tool_call_id, which some providers reject on the next turn).
        Never raises — problems here are dropped and logged, not a crash."""
        seen_ids: set = set()
        valid = []
        for tc in tool_calls:
            call_id = tc.get("id")
            name = tc.get("name")
            if not call_id or not name:
                log_event(agent_config.log_file, "tool_call_malformed", thread_id=thread_id, raw=str(tc)[:300])
                continue
            if call_id in seen_ids:
                log_event(
                    agent_config.log_file, "tool_call_duplicate_id", thread_id=thread_id,
                    tool_name=name, call_id=call_id,
                )
                continue
            seen_ids.add(call_id)
            valid.append(tc)
        return valid

    def confirm_node(state: AgentState) -> dict:
        thread_id = _thread_id()
        last_ai = _last_ai_message_with_tool_calls(list(state.messages))
        tool_calls = _dedup_and_validate_tool_calls(last_ai.tool_calls if last_ai else [], thread_id)

        pending: list[ConfirmationRequest] = []
        for tc in tool_calls:
            args = tc.get("args", {}) or {}
            if needs_confirmation(tc["name"], args, agent_config.confirm_all_tools):
                pending.append(
                    ConfirmationRequest(tool_name=tc["name"], tool_args=args, call_id=tc["id"])
                )

        if not pending:
            return {}

        raw = interrupt([r.model_dump() for r in pending])
        decisions = raw if isinstance(raw, dict) else {}

        new_messages: list[BaseMessage] = []
        for req in pending:
            decision = decisions.get(req.call_id) or {}
            approved = bool(isinstance(decision, dict) and decision.get("approved"))
            if not approved:
                reason = (decision.get("reason") if isinstance(decision, dict) else None) or (
                    "declined by the user"
                )
                new_messages.append(
                    ToolMessage(
                        content=(
                            f"BLOCKED: {reason}. Do not retry this exact command — "
                            "explain the block to the user or propose a safer alternative."
                        ),
                        tool_call_id=req.call_id,
                        name=req.tool_name,
                    )
                )
        return {"messages": new_messages} if new_messages else {}

    def tools_node(state: AgentState) -> dict:
        thread_id = _thread_id()
        messages = list(state.messages)
        last_ai = _last_ai_message_with_tool_calls(messages)
        tool_calls = _dedup_and_validate_tool_calls(last_ai.tool_calls if last_ai else [], thread_id)
        answered = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}

        new_messages: list[BaseMessage] = []
        logs: list[dict] = []
        cancelled = False
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
                    output = _truncate(raw_output, continuation_hint_tool=name)
                    success = not (isinstance(raw_output, str) and raw_output.startswith(("ERROR", "BLOCKED")))
                except KeyboardInterrupt:
                    # Ctrl-C during a slow tool call (long shell command, Jupyter
                    # cell, etc). Not an Exception subclass, so the `except
                    # Exception` below would have missed it and let it crash the
                    # REPL. Record this call as cancelled and stop the batch —
                    # don't keep firing off further tool calls after a cancel.
                    output = "CANCELLED: interrupted by user (Ctrl-C) while this tool was running."
                    success = False
                    new_messages.append(ToolMessage(content=output, tool_call_id=call_id, name=name))
                    logs.append(
                        ToolCallLog(
                            call_id=call_id, tool_name=name, args=args, result=output,
                            confirmed=True, success=success,
                        ).model_dump()
                    )
                    log_event(agent_config.log_file, "tool_call_cancelled", thread_id=thread_id, tool_name=name)
                    cancelled = True
                    break
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

        result: dict = {"messages": new_messages, "tool_log": logs}
        if cancelled:
            result["status"] = "cancelled"
            result["error"] = "cancelled by user (Ctrl-C) during a tool call"
        return result

    # ---- routing -------------------------------------------------------

    def route_after_agent(state: AgentState) -> Literal["confirm", "tools", "__end__"]:
        if state.status in ("max_iterations_reached", "error", "cancelled"):
            return "__end__"
        last = state.messages[-1] if state.messages else None
        raw_tool_calls = getattr(last, "tool_calls", None) or []
        tool_calls = _dedup_and_validate_tool_calls(raw_tool_calls, _thread_id())
        if not tool_calls:
            return "__end__"
        if any(needs_confirmation(tc["name"], tc.get("args", {}) or {}, agent_config.confirm_all_tools) for tc in tool_calls):
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

    return builder.compile(checkpointer=_SHARED_CHECKPOINTER)
