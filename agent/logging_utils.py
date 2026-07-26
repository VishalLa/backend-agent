import json
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_LOG_FILE = "agent_events.log"

_lock = threading.Lock()


def log_event(log_path: str, event_type: str, **fields: Any) -> None:
    """Append a single JSON-lines event to the agent's event log.

    Best-effort by design: a logging failure (bad path, disk full, etc.) is
    swallowed rather than raised, so a broken log file can never take down
    an agent run. Each line is a standalone JSON object, e.g.:

        {"timestamp": 1732650000.1, "event": "tool_call", "thread_id": "...",
         "tool_name": "write_file", "args": {...}, "success": true, "result": "..."}

    Event types currently emitted by the agent:
      run_start, run_end, run_failed        (runner.py)
      llm_call, llm_rate_limited_wait,
      llm_call_failed, llm_tool_hallucination,
      llm_tool_call_leaked_as_text,
      llm_empty_response                    (graph.py — includes which
                                              provider tier answered, when
                                              it can be determined)
      tool_call, tool_call_blocked          (graph.py)
      confirmation_decision                 (runner.py)
    """
    entry = {"timestamp": time.time(), "event": event_type, **fields}
    try:
        path = Path(log_path or DEFAULT_LOG_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, default=str)
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:  # noqa: BLE001 - logging must never break the agent
        pass


def safe_args(args: dict, limit: int = 300) -> dict:
    """Shrink large string argument values before they go into a log line
    (e.g. the full file content passed to write_file) so log lines stay
    readable and the log file doesn't balloon on large writes."""
    safe: dict[str, Any] = {}
    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > limit:
            safe[k] = v[:limit] + f"...[{len(v) - limit} chars omitted]"
        else:
            safe[k] = v
    return safe
    