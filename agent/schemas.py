import operator
import time
from typing import Annotated, Any, Literal, Optional, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

Status = Literal["running", "completed", "max_iterations_reached", "error", "cancelled"]


class ToolCallLog(BaseModel):
    """Record of a single tool invocation, kept for session/debug visibility."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    call_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    confirmed: bool = True  # True if this tool never needed confirmation, or the user approved it
    success: bool = True
    timestamp: float = Field(default_factory=time.time)


class ConfirmationRequest(BaseModel):
    """Payload sent through interrupt() when a tool call needs human approval."""

    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    call_id: str
    reason: str = "This tool can make real changes to your system and requires explicit approval."


class ConfirmationDecision(BaseModel):
    """Payload sent back via Command(resume=...) to answer a ConfirmationRequest."""

    approved: bool
    reason: Optional[str] = None


class AgentState(BaseModel):
    """LangGraph state schema. `messages` and `tool_log` accumulate across
    nodes (via reducers); everything else is overwritten by whichever node
    last set it.

    tool_log is list[dict] (ToolCallLog.model_dump() output) rather than
    list[ToolCallLog] — see the note on ToolCallLog above for why.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list)
    iterations: int = 0
    tool_log: Annotated[list[dict[str, Any]], operator.add] = Field(default_factory=list)
    status: Status = "running"
    error: Optional[str] = None


class AgentResult(BaseModel):
    """Validated, serializable summary of a completed (or failed) agent run."""

    output: str
    status: Status
    iterations: int
    tool_calls: list[ToolCallLog] = Field(default_factory=list)
    thread_id: str
    error: Optional[str] = None
