from typing import Any

from .persistence import DatabaseLogService

__all__ = ["DatabaseLogService", "enqueue_event", "persist_event"]


def __getattr__(name: str) -> Any:
    """Avoid requiring a Celery worker dependency for synchronous DB use."""
    if name in {"enqueue_event", "persist_event"}:
        from . import tasks

        return getattr(tasks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
