from __future__ import annotations

import enum 
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime, timezone

class Base(DeclarativeBase):
	pass 


class SessionStatus(str, enum.Enum):
    active = "active"
    completed = "completed"
    abandoned = "abandoned"


class AgentType(str, enum.Enum):
    backend = "backend"
    ml = "ml"


class MessageRole(str, enum.Enum):
    user = "user"
    assistant = "assistant"
    tool = "tool"
    system = "system"


class ToolCallStatus(str, enum.Enum):
    pending = "pending"
    approved = "approved"
    denied = "denied"
    success = "success"
    failed = "failed"


class ConfirmationActionType(str, enum.Enum):
    destructive_action = "destructive_action"   # Gate A
    save_code = "save_code"                     # Gate B
    memory_update = "memory_update"              # Gate C


class AgentFileType(str, enum.Enum):
    agent_md = "agent.md"
    memory_md = "memory.md"


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp, used as the default for all created_at columns."""
    return datetime.now(timezone.utc)

