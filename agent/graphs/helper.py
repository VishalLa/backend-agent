import re
import json
import uuid
from typing import Iterable, Optional, Any
from config import Config

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage

LOCAL_SERVER_COLD_START_WAIT_SECONDS = 15.0
MAX_TOOL_HALLUCINATION_RETRIES = 2
MAX_EMPTY_RESPONSE_RETRIES = 2
STAGNATION_REPEAT_THRESHOLD = 3
MAX_RATE_LIMIT_WAIT_SECONDS = 90.0


MAX_TOOL_CALL_ARG_CHARS = 200
MAX_TOOL_OUTPUT_CHARS = 800

RECENT_TOOL_RESULTS_KEPT_FULL = 2
COLLAPSED_TOOL_RESULT_CHARS = 80

_RETRY_AFTER_RE = re.compile(r"try again in\s+([\d.]+)\s*s", re.IGNORECASE)

READ_ONLY_TOOL_NAMES = frozenset({
    "read_file",
    "list_dir",
    "ripgrep_search",
    "web_search",
    "git_status",
    "git_diff",
    "git_log",
    "check_gpu_status",
    "tail_log",
    "fetch_openapi_schema",
})

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

LOCAL_SERVER_STARTUP_MARKERS = (
    "disconnected", 
    "connection refused", 
    "broken pipe", 
    "remote protocol error"
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

TOKEN_RATE_LIMIT_MARKERS = (
    "tokens per minute",
    "requests per minute",
    "tpm",
    "rpm)",
    "rate_limit_exceeded",
)


def _parse_retry_after(msg: str) -> Optional[float]:
    match = _RETRY_AFTER_RE.search(msg)
    
    if not match:
        return None
    try: 
        return float(match.group(1))
    except ValueError:
        return None


def classify_error(error_text: str) -> dict:
    text = (error_text or "").lower()

    if any(marker in text for marker in CONTEXT_LENGTH_MARKERS):
        return {"kind": "context_length", "retryable": False, "wait_seconds": None}

    if any(marker in text for marker in TOKEN_RATE_LIMIT_MARKERS):
        wait = _parse_retry_after(text)
        return {
            "kind": "token_rate_limit",
            "retryable": True,
            "wait_seconds": min(wait, MAX_RATE_LIMIT_WAIT_SECONDS) if wait else MAX_RATE_LIMIT_WAIT_SECONDS,
        }

    if any(marker in text for marker in LOCAL_SERVER_STARTUP_MARKERS):
        return {
            "kind": "local_server_cold_start",
            "retryable": True,
            "wait_seconds": LOCAL_SERVER_COLD_START_WAIT_SECONDS,
        }

    if any(marker in text for marker in RETRYABLE_ERROR_MARKERS):
        wait = _parse_retry_after(text)
        return {"kind": "transient", "retryable": True, "wait_seconds": wait}

    return {"kind": "unknown", "retryable": False, "wait_seconds": None}


def _identify_provider(response: Any, config: Config) -> str:
    meta = getattr(response, "response_metadata", {}) or {}
    model = str(meta.get("model_name") or meta.get("model") or "").strip().lower()
    if not model:
        return "unknown"

    candidates = {
        "sambanova": config.get_model_for_task(config.agent_type),
        "groq": config.groq_model,
        "openrouter": config.openrouter_model,
        "ollama": config.ollama_model,
    }
    for provider, candidate_model in candidates.items():
        if candidate_model and candidate_model.strip().lower() == model:
            return provider
    return model or "unknown"


def _thread_id() -> str:
    try:
        from langgraph.config import get_config as _lg_get_config
    except ImportError:
        def _lg_get_config() -> dict:
            return {}
    
    try:
        cfg = _lg_get_config()
        return (cfg or {}).get("configurable", {}).get("thread_id", "unknown")
    except Exception:
        return "unknown"


def _find_json_object(text: str) -> Optional[str]:
    """Find the first balanced {...} object in text, ignoring braces that
    appear inside quoted strings (e.g. a leaked tool call whose arguments
    contain literal '{'/'}' characters, like a code snippet)."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _extract_leaked_tool_call(
    text: str,
    valid_tool_names: Iterable[str]
) -> Optional[dict]:
    """Repair a tool call the model wrote as raw JSON in its text content
    instead of an actual tool_call (a "leaked" call), so it can still be
    executed rather than silently ignored."""
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

    return {
        "name": name,
        "args": args,
        "id": f"repaired-{uuid.uuid4().hex[:8]}"
    }


def _truncate(
    text: object,
    limit: int = MAX_TOOL_OUTPUT_CHARS,
    continuation_hint_tool: Optional[str] = None
) -> str:
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
            new_line = int(m.group(1)) + 1
            suffix += (
                f" — this file continues past line {m.group(1)}. It is NOT fully shown above; "
                f"call read_file again with start_line={new_line} to keep reading before editing "
                "or claiming the file is fully updated."
            )
        else:
            suffix += (
                " — this output was cut off and may not represent the full file/result; "
                "re-read with a later start_line if you need the rest before editing."
            )
    return truncated + suffix


def _collapse_old_tool_results(
    messages: list[BaseMessage]
) -> list[BaseMessage]:
    """Shrink all but the most recent RECENT_TOOL_RESULTS_KEPT_FULL
    ToolMessages down to COLLAPSED_TOOL_RESULT_CHARS - the main lever for
    keeping a long-running session under Config.max_context_tokens."""
    tool_msg_indices = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    collapse_indices = set(
        tool_msg_indices[:-RECENT_TOOL_RESULTS_KEPT_FULL]
    ) if len(tool_msg_indices) > RECENT_TOOL_RESULTS_KEPT_FULL else set()

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


def _prepare_messages_for_model(
    messages: list[BaseMessage]
) -> list[BaseMessage]:
    collapsed = _collapse_old_tool_results(messages)
    prepared = []

    for m in collapsed:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            new_tool_calls = _slim_tool_calls(m.tool_calls)
            if new_tool_calls is not m.tool_calls:
                m = m.model_copy(update={"tool_calls": new_tool_calls})

        prepared.append(m)
    return prepared


def _last_ai_message_with_tool_calls(
    messages: list[BaseMessage]
) -> Optional[AIMessage]:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            return m
    return None


def _detect_read_only_stagnation(
    tool_log: list,
    threshold: int = STAGNATION_REPEAT_THRESHOLD
) -> Optional[str]:
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
