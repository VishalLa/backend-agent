from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as SASession
from sqlalchemy.orm import sessionmaker

from .base import Base

from . import session_manager  # noqa: F401  (defines Session)
from . import backend_agent  # noqa: F401  (defines everything else)


class DatabaseManager:

    def __init__(self, db_url: str = "sqlite:///coding_agent.db", echo: bool = False):
        self.db_url = db_url
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(db_url, echo=echo, connect_args=connect_args)
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def init_db(self) -> None:
        """Create all registered tables that don't already exist. Safe to
        call on every startup - it's a no-op for tables that already exist."""
        Base.metadata.create_all(self.engine)

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
