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
    relationship,
)

from .base import (
	Base, 
	SessionStatus, 
	MessageRole, 
	AgentType, 
	ToolCallStatus, 
	ConfirmationActionType, 
	AgentFileType,
	utcnow,
)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tokens: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )

    session: Mapped["Session"] = relationship(back_populates="messages")
    tool_calls: Mapped[list["ToolCall"]] = relationship(back_populates="message")


class ToolCall(Base):
    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    message_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL")
    )
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    input_json: Mapped[Optional[str]] = mapped_column(Text)
    output_json: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[ToolCallStatus] = mapped_column(
        Enum(ToolCallStatus), nullable=False, default=ToolCallStatus.pending
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped["Session"] = relationship(back_populates="tool_calls")
    message: Mapped[Optional["Message"]] = relationship(back_populates="tool_calls")
    sandbox_runs: Mapped[list["SandboxRun"]] = relationship(back_populates="tool_call")


class SandboxRun(Base):
    __tablename__ = "sandbox_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tool_call_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("tool_calls.id", ondelete="SET NULL")
    )
    code: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[Optional[str]] = mapped_column(String(64))
    stdout: Mapped[Optional[str]] = mapped_column(Text)
    stderr: Mapped[Optional[str]] = mapped_column(Text)
    exit_code: Mapped[Optional[int]] = mapped_column(Integer)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped["Session"] = relationship(back_populates="sandbox_runs")
    tool_call: Mapped[Optional["ToolCall"]] = relationship(back_populates="sandbox_runs")


class Confirmation(Base):
    """Human confirmation gates A (destructive action), B (save code), C (memory update)."""

    __tablename__ = "confirmations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    action_type: Mapped[ConfirmationActionType] = mapped_column(
        Enum(ConfirmationActionType), nullable=False
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    approved: Mapped[Optional[bool]] = mapped_column(Boolean)  # NULL until answered
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped["Session"] = relationship(back_populates="confirmations")


class Summary(Base):
    """Rolling conversation summary produced by the context window handler."""

    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    covers_up_to_message_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("messages.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped["Session"] = relationship(back_populates="summaries")


class MemoryFact(Base):
    """
    Structured source of truth behind memory.md (e.g. category='tech_stack',
    key='web_framework', value='FastAPI'). memory.md is *rendered* from these
    rows rather than edited as freeform prose, so updates stay diffable.
    """

    __tablename__ = "memory_facts"
    __table_args__ = (
        CheckConstraint("length(category) > 0", name="ck_memory_facts_category_nonempty"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    source_session_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    source_session: Mapped[Optional["Session"]] = relationship(back_populates="memory_facts")


class AgentFile(Base):
    """Versioned snapshots of agent.md / memory.md, written only after Gate C approval."""

    __tablename__ = "agent_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_type: Mapped[AgentFileType] = mapped_column(Enum(AgentFileType), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ProviderUsage(Base):
    """Tracks token usage per LLM provider call, used to drive the fallback chain."""

    __tablename__ = "provider_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(64), nullable=False)  # sambanova/groq/openrouter/ollama
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    tokens_in: Mapped[Optional[int]] = mapped_column(Integer)
    tokens_out: Mapped[Optional[int]] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped["Session"] = relationship(back_populates="provider_usage")

