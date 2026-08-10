from __future__ import annotations

from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.orm import Session as SASession

from database.backend_agent import (
    AgentFile,
    Confirmation,
    MemoryFact,
    Message,
    ProviderUsage,
    SandboxRun,
    Summary,
    ToolCall,
)
from database.session import DatabaseManager
from database.session_manager import Session
from schema.database_schema import (
    AgentFileCreate,
    ConfirmationDecisionDB,
    ConfirmationCreate,
    MemoryFactCreate,
    MemoryFactUpdate,
    MessageCreate,
    ProviderUsageCreate,
    SandboxRunCreate,
    SandboxRunResult,
    SessionCreate,
    SessionUpdate,
    SummaryCreate,
    ToolCallCreate,
)

SchemaT = TypeVar("SchemaT")


class DatabaseLogService:
    """Persist one validated event per short-lived SQLAlchemy transaction."""

    def __init__(
        self, 
        db_url: str,
        *, 
        echo: bool = False
    ) -> None:
        self.manager = DatabaseManager(db_url=db_url, echo=echo)


    def init_db(self) -> None:
        self.manager.init_db()


    def list_sessions(
        self,
        *,
        agent_type: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return recent persisted conversations for the dashboard sidebar."""
        if limit < 1:
            raise ValueError("limit must be at least 1")

        statement = select(Session).order_by(Session.updated_at.desc()).limit(limit)
        if agent_type:
            statement = statement.where(Session.agent_type == agent_type)

        with self.manager.session_scope() as db:
            rows = db.scalars(statement).all()
            return [
                {
                    "id": row.id,
                    "title": row.title or "Untitled conversation",
                    "project_path": row.project_path,
                    "agent_type": row.agent_type.value,
                    "status": row.status.value,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
                for row in rows
            ]


    def list_messages(self, session_id: int) -> list[dict[str, Any]]:
        """Return one persisted transcript in chronological order."""
        session_id = _required_id({"id": session_id}, "id")
        statement = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
        )
        with self.manager.session_scope() as db:
            return [
                {
                    "id": row.id,
                    "role": row.role.value,
                    "content": row.content,
                    "created_at": row.created_at,
                }
                for row in db.scalars(statement).all()
            ]


    def persist_event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate and persist a supported event; return its database identity.

        Payloads must be JSON-compatible because they cross the Celery/Redis
        boundary. Validation happens in the worker, not only at enqueue time.
        """
        handlers = {
            "session.create": self.create_session,
            "session.update": self.update_session,
            "message.create": self.create_message,
            "tool_call.create": self.create_tool_call,
            "sandbox_run.create": self.create_sandbox_run,
            "confirmation.create": self.create_confirmation,
            "confirmation.decide": self.decide_confirmation,
            "summary.create": self.create_summary,
            "memory_fact.create": self.create_memory_fact,
            "memory_fact.update": self.update_memory_fact,
            "agent_file.create": self.create_agent_file,
            "provider_usage.create": self.create_provider_usage,
        }
        try:
            handler = handlers[event_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported database event: {event_type}") from exc
        return handler(payload)


    def create_session(
        self, 
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        data = SessionCreate.model_validate(payload)

        with self.manager.session_scope() as db:
            row = Session(**data.model_dump())
            db.add(row)
            db.flush()
            return {"id": row.id}


    def update_session(
        self, 
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        session_id = _required_id(payload, "id")
        data = SessionUpdate.model_validate(
            {key: value for key, value in payload.items() if key != "id"}
        )
        with self.manager.session_scope() as db:
            row = _get_required(db, Session, session_id)

            for field, value in data.model_dump(exclude_unset=True).items():
                setattr(row, field, value)
            return {"id": row.id}


    def create_message(
        self, 
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._create(Message, MessageCreate, payload)


    def create_tool_call(
        self, 
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._create(ToolCall, ToolCallCreate, payload)


    def create_sandbox_run(
        self, 
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        data_payload = dict(payload)
        result = SandboxRunResult.model_validate(data_payload.pop("result", {}))
        data = SandboxRunCreate.model_validate(data_payload)

        with self.manager.session_scope() as db:
            row = SandboxRun(**data.model_dump(), **result.model_dump())
            db.add(row)
            db.flush()
            return {"id": row.id}


    def create_confirmation(
        self, 
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._create(Confirmation, ConfirmationCreate, payload)


    def decide_confirmation(
        self, 
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        confirmation_id = _required_id(payload, "id")
        approved = ConfirmationDecisionDB.model_validate({"approved": payload.get("approved")}).approved

        if not isinstance(approved, bool):
            raise ValueError("confirmation.decide requires an 'approved' boolean")

        with self.manager.session_scope() as db:
            row = _get_required(db, Confirmation, confirmation_id)
            row.approved = approved
            return {"id": row.id}


    def create_summary(
        self, 
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._create(Summary, SummaryCreate, payload)


    def create_memory_fact(
        self, 
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._create(MemoryFact, MemoryFactCreate, payload)


    def update_memory_fact(
        self, 
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        fact_id = _required_id(payload, "id")
        changes = {key: value for key, value in payload.items() if key != "id"}
        if not changes:
            raise ValueError("memory_fact.update requires at least one field to update")

        validated = MemoryFactUpdate.model_validate(changes)
        
        with self.manager.session_scope() as db:
            row = _get_required(db, MemoryFact, fact_id)
            for field, value in validated.model_dump(exclude_unset=True).items():
                setattr(row, field, value)

            return {"id": row.id}


    def create_agent_file(
        self, 
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        data = AgentFileCreate.model_validate(payload)

        with self.manager.session_scope() as db:
            latest = db.scalar(
                select(AgentFile.version)
                .where(AgentFile.file_type == data.file_type)
                .order_by(AgentFile.version.desc())
                .limit(1)
            )

            row = AgentFile(
                file_type=data.file_type,
                content=data.content,
                version=(latest or 0) + 1,
            )

            db.add(row)
            db.flush()

            return {"id": row.id, "version": row.version}


    def create_provider_usage(
        self, 
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._create(ProviderUsage, ProviderUsageCreate, payload)


    def _create(
        self, 
        model: type[Any], 
        schema: type[SchemaT], 
        payload: dict[str, Any]
    ) -> dict[str, Any]:
        data = schema.model_validate(payload)

        with self.manager.session_scope() as db:
            row = model(**data.model_dump())
            db.add(row)
            db.flush()
            return {"id": row.id}


def _required_id(
    payload: dict[str, Any], 
    key: str
) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"'{key}' must be a positive integer")
    return value


def _get_required(
    db: SASession, 
    model: type[Any], 
    row_id: int
) -> Any:
    row = db.get(model, row_id)
    if row is None:
        raise ValueError(f"{model.__name__} {row_id} does not exist")
    return row
