import json
import threading
import time
from pathlib import Path
from typing import Any

_lock = threading.Lock()
DEFAULT_LOG_FILE = "agent_events.log"

def log_event(
    log_file: str,
    event_type: str,
    **fields: Any
) -> None:
    entry = {
        "timestamp": time.time(),
        "event": event_type,
        **fields
    }

    try:
        path = Path(log_file or DEFAULT_LOG_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, default=str)

        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    except Exception:
        pass


def safe_args(
    args: dict,
    limit: int = 300
) -> dict:
    safe: dict[str, Any] = {}

    for k, v in (args or {}).items():
        if isinstance(v, str) and len(v) > limit:
            safe[k] = v[:limit] + f"...[{len(v) - limit} chars omitted]"
        else:
            safe[k] = v
    
    return safe
