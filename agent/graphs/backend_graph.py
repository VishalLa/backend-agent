import json
import re
import time
import uuid
from typing import Any, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from config import Config
from log.log_event import log_event, safe_args
from schema.agent_schema import AgentState, ConfirmationRequest, ToolCallLog

from ..confirmation import needs_confirmation
from ..llm import ChatModel


