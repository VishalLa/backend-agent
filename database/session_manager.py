from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship
)

from .base import Base, SessionStatus, AgentType, utcnow


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[Optional[str]] = mapped_column(String(255))
    project_path: Mapped[Optional[str]] = mapped_column(String(1024))
    agent_type: Mapped[AgentType] = mapped_column(Enum(AgentType), nullable=False)

    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus), nullable=False, default=SessionStatus.active
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    tool_calls: Mapped[list["ToolCall"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    sandbox_runs: Mapped[list["SandboxRun"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    confirmations: Mapped[list["Confirmation"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    summaries: Mapped[list["Summary"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    provider_usage: Mapped[list["ProviderUsage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    memory_facts: Mapped[list["MemoryFact"]] = relationship(back_populates="source_session")

