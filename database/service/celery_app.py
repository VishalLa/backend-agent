from __future__ import annotations

import os

from celery import Celery


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_celery_app(
    broker_url: str | None = None,
    result_backend: str | None = None,
    task_always_eager: bool | None = None,
) -> Celery:
    """Create a worker app without requiring any LLM credentials."""
    app = Celery(
        "coding_agent_database",
        broker=broker_url or os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        backend=result_backend or os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
        include=["database.service.tasks"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        task_always_eager=(
            task_always_eager
            if task_always_eager is not None
            else _as_bool(os.environ.get("CELERY_TASK_ALWAYS_EAGER", "false"))
        ),
    )
    return app


celery_app = create_celery_app()
