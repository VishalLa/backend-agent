from __future__ import annotations

from contextlib import contextmanager
import uuid
from typing import Iterator, Optional

from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as SASession
from sqlalchemy.orm import sessionmaker

from .base import AgentType, Base, SessionStatus, utcnow

from . import session_manager  # noqa: F401  (defines Session)
from . import backend_agent  # noqa: F401  (defines everything else)


class DatabaseManager:

    def __init__(self, db_url: str = "sqlite:///coding_agent.db", echo: bool = False):
        self.db_url = db_url
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(db_url, echo=echo, connect_args=connect_args)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def init_db(self) -> None:
        """Create the canonical schema and preserve a legacy sessions table.

        SQLAlchemy's ``create_all`` only creates missing tables; it cannot add
        the ``id`` primary key to an already-existing table. If an earlier
        version created ``sessions`` without that column, move it aside before
        creating the canonical schema and copy its compatible rows back.
        """
        inspector = inspect(self.engine)
        if inspector.has_table("sessions"):
            columns = {column["name"] for column in inspector.get_columns("sessions")}
            if "id" not in columns:
                self._migrate_legacy_sessions_table()
        Base.metadata.create_all(self.engine)

    def _migrate_legacy_sessions_table(self) -> None:
        """Preserve and copy a pre-primary-key ``sessions`` table.

        The legacy table is retained under a unique ``sessions_legacy_*`` name
        even if copying fails, so existing data is never dropped silently.
        """
        legacy_name = f"sessions_legacy_{uuid.uuid4().hex[:12]}"
        quote = self.engine.dialect.identifier_preparer.quote
        with self.engine.begin() as connection:
            connection.execute(text(f"ALTER TABLE {quote('sessions')} RENAME TO {quote(legacy_name)}"))

        Base.metadata.create_all(self.engine)
        legacy = Table(legacy_name, MetaData(), autoload_with=self.engine)

        try:
            with self.session_scope() as db:
                rows = db.execute(select(legacy)).mappings()
                for row in rows:
                    data = dict(row)
                    agent_type = _coerce_agent_type(data.get("agent_type"))
                    status = _coerce_session_status(data.get("status"))
                    db.add(
                        session_manager.Session(
                            title=data.get("title") or "Restored conversation",
                            project_path=data.get("project_path"),
                            agent_type=agent_type,
                            status=status,
                            created_at=data.get("created_at") or utcnow(),
                            updated_at=data.get("updated_at") or data.get("created_at") or utcnow(),
                        )
                    )
        except Exception as exc:
            raise RuntimeError(
                f"Created the canonical schema but could not copy legacy sessions from {legacy_name}. "
                "The legacy table was preserved unchanged."
            ) from exc

    def drop_all(self) -> None:
        """Drops every table. Destructive - intended for tests/resets only."""
        Base.metadata.drop_all(self.engine)

    def new_session(self) -> SASession:
        """Return a raw SQLAlchemy session. Caller owns commit/rollback/close.
        Prefer session_scope() for anything that isn't a long-lived REPL/script."""
        return self._session_factory()

    @contextmanager
    def session_scope(self) -> Iterator[SASession]:
        """Transactional scope: commits on success, rolls back on any
        exception, and always closes the session afterward."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def healthcheck(self) -> bool:
        """Quick connectivity check (`SELECT 1`). Useful before the agent
        loop starts, so a broken DB path fails fast instead of mid-session."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def dispose(self) -> None:
        """Close all pooled connections. Call on clean app shutdown."""
        self.engine.dispose()


_default_manager: Optional[DatabaseManager] = None


def get_default_manager(db_url: str = "sqlite:///coding_agent.db", echo: bool = False) -> DatabaseManager:
    """Lazily create a process-wide singleton DatabaseManager.

    This is a personal, single-user agent - there's no need for
    per-request DB managers, so most call sites can just import this
    function instead of threading a DatabaseManager instance everywhere.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = DatabaseManager(db_url=db_url, echo=echo)
    return _default_manager


def _coerce_agent_type(value: object) -> AgentType:
    try:
        return AgentType(str(getattr(value, "value", value)))
    except ValueError:
        return AgentType.backend


def _coerce_session_status(value: object) -> SessionStatus:
    try:
        return SessionStatus(str(getattr(value, "value", value)))
    except ValueError:
        return SessionStatus.active
