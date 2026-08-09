from __future__ import annotations

import operator
import time

from datetime import datetime
from typing import Annotated, Any, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from database.base import (
    AgentFileType,
    AgentType,
    ConfirmationActionType,
    MessageRole,
    SessionStatus,
    ToolCallStatus,
)
from .base import SchemaBase


# Session Schemas
class SessionBase(SchemaBase):
    title: Optional[str] = Field(default=None, max_length=255)
    project_path: Optional[str] = Field(default=None, max_length=1024)
    agent_type: AgentType


class SessionCreate(SessionBase):
    status: SessionStatus = SessionStatus.active


class SessionUpdate(SchemaBase):
    title: Optional[str] = Field(default=None, max_length=255)
    project_path: Optional[str] = Field(default=None, max_length=255)
    status: Optional[SessionStatus] = None


class SessionRead(SessionBase):
    id: int
    status: SessionStatus
    created_at: datetime
    updated_at: datetime


# Message schema
class MessageBase(SchemaBase):
    role: MessageRole
    content: str = Field(..., min_length=1)
    tokens: Optional[int] = Field(default=None, ge=0)


class MessageCreate(MessageBase):
    session_id: int = Field(..., gt=0)


class MessageRead(MessageBase):
    id: int
    session_id: int
    created_at: datetime


# Tool call schema
class ToolCallBase(SchemaBase):
    tool_name: str = Field(..., min_length=1, max_length=128)
    input_json: Optional[str] = None
    output_json: Optional[str] = None
    status: ToolCallStatus


class ToolCallCreate(ToolCallBase):
    session_id: int = Field(..., gt=0)
    message_id: Optional[int] = Field(default=None, gt=0)


class ToolCallRead(ToolCallBase):
    id: int
    session_id: int
    message_id: Optional[int]
    created_at: datetime



# Sandbox schema
class SandboxRunBase(SchemaBase):
    code: str = Field(..., min_length=1)
    language: Optional[str] = Field(default=None, max_length=64)


class SandboxRunCreate(SandboxRunBase):
    session_id: int = Field(..., gt=0)
    tool_call_id: Optional[int] = Field(default=None, gt=0)


class SandboxRunResult(SchemaBase):
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None
    success: bool
    duration_ms: Optional[int] = Field(default=None, ge=0)


class SandboxRunRead(SandboxRunBase):
    id: int
    session_id: int
    tool_call_id: Optional[int]
    stdout: Optional[str]
    stderr: Optional[str]
    exit_code: Optional[int]
    success: bool
    duration_ms: Optional[int]
    created_at: datetime


# confirmataion schemas
class ConfirmationCreate(SchemaBase):
    session_id: int = Field(..., gt=0)
    action_type: ConfirmationActionType
    description: str = Field(..., min_length=1)


class ConfirmationDecisionDB(SchemaBase):
    approved: Optional[bool] = None


class ConfirmationRead(SchemaBase):
    id: int
    session_id: int
    action_type: ConfirmationActionType
    description: str
    approved: Optional[bool]
    created_at: datetime


# Summary shcema
class SummaryCreate(SchemaBase):
    session_id: int = Field(..., gt=0)
    summary_text: str = Field(..., min_length=1)
    covers_up_to_message_id: Optional[int] = Field(default=None, gt=0)


class SummaryRead(SchemaBase):
    id: int
    session_id: int
    summary_text: str
    covers_up_to_message_id: Optional[int]
    created_at: datetime


# memory schemas

class MemoryFactBase(SchemaBase):
    category: str = Field(..., min_length=1, max_length=128)
    key: str = Field(..., min_length=1, max_length=255)
    value: str = Field(..., min_length=1)


class MemoryFactCreate(MemoryFactBase):
    source_session_id: Optional[int] = Field(default=None, gt=0)


class MemoryFactUpdate(SchemaBase):
    category: Optional[str] = Field(default=None, min_length=1, max_length=128)
    key: Optional[str] = Field(default=None, min_length=1, max_length=255)
    value: Optional[str] = Field(default=None, min_length=1)


class MemoryFactRead(MemoryFactBase):
    id: int
    source_session_id: Optional[int]
    updated_at: datetime


# agent file schemas
class AgentFileCreate(SchemaBase):
    file_type: AgentFileType
    content: str
    version: int = Field(default=1,ge=1)


class AgentFileRead(AgentFileCreate):
    id: int
    updated_at: datetime



class ProviderUsageCreate(SchemaBase):
    session_id: int = Field(..., gt=0)
    provider: str = Field(..., min_length=1, max_length=64)
    model: str = Field(..., min_length=1, max_length=128)
    tokens_in: Optional[int] = Field(default=None, ge=0)
    tokens_out: Optional[int] = Field(default=None, ge=0)


class ProviderUsageRead(ProviderUsageCreate):
    id: int
    created_at: datetime

