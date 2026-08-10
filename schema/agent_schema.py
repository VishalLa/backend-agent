from __future__ import annotations

import operator
import time

from typing import Annotated, Any, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


from .base import SchemaBase


class ToolCallLog(SchemaBase):
	"""
    Runtime representation of a tool invocation.

    This is used by LangGraph for execution/debugging and can
    later be converted into a database ToolCall record.
    """

	call_id: str = Field(..., min_length=1, max_length=255)
	tool_name: str = Field(..., min_length=1, max_length=128)
	args: dict[str, Any] = Field(default_factory=dict)
	result: str = ""
	confirmed: bool = True
	success: bool = True
	timestamp: float = Field(default_factory=time.time)


class ConfirmationRequest(SchemaBase):
	"""
    Payload sent through LangGraph interrupt() when
    human approval is required.
    """
	
	tool_name: str = Field(..., min_length=1, max_length=128)
	tool_args: dict[str, Any] = Field(default_factory=dict)
	call_id: str = Field(..., min_length=1, max_length=255)
	reason: str = Field(
        default=(
            "This tool can make real changes to your system "
            "and requires explicit approval."
        )
    )

class ConfirmationDecision(SchemaBase):
    """
    Decision returned through Command(resume=...).
    """

    approved: bool
    reason: Optional[str] = None


class AgentState(BaseModel):
	"""
    LangGraph runtime state.

    This is intentionally separate from the SQLAlchemy models.
    """
	model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

	messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list)
	iterations: int = Field(default=0, ge=0)
	tool_logs: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)
	status: str = "running"
	error: Optional[str] = None


class AgentResult(SchemaBase):
	"""
    Final serializable result returned by the agent.
    """

	output: str
	status: str
	iterations: int = Field(default=0, ge=0)
	tool_calls: list[ToolCallLog] = Field(default_factory=list)
	thread_id: str = Field(..., min_length=1)
	error: Optional[str] = None

