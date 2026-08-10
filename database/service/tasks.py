from __future__ import annotations

from typing import Any

from celery.result import AsyncResult
from sqlalchemy.exc import SQLAlchemyError

from config import Config

from .celery_app import celery_app
from .persistence import DatabaseLogService


@celery_app.task(
    bind=True,
    autoretry_for=(SQLAlchemyError,),
    retry_backoff=True,
    retry_jitter=True,
    max_retries=3,
    name="database.persist_event",
)
def persist_event(
    self: Any,
    event_type: str,
    payload: dict[str, Any],
    db_url: str,
) -> dict[str, Any]:
    """Validate and persist one event in the worker process."""
    service = DatabaseLogService(db_url)
    service.init_db()
    return service.persist_event(event_type, payload)


def enqueue_event(
    event_type: str,
    payload: dict[str, Any],
    config: Config,
) -> AsyncResult:
    """Queue a JSON-safe persistence event through Celery/Redis."""
    return persist_event.apply_async(args=(event_type, payload, config.postgres_url))
