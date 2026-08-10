from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from config import Config

from .graphs.algo_graph import AlgoAgent
from .graphs.backend_graph import BackendAgent
from .graphs.base_graph import BaseAgent
from .graphs.git_graph import GitAgent
from .graphs.ml_graph import MLAgent
from .tools import SANDBOX_TOOLS, TOOLS_BY_TASK

if TYPE_CHECKING:
    from .storage import AgentStorage


@dataclass(frozen=True)
class AgentOption:
    """A stable selection value and the display label for a frontend."""

    key: str
    label: str
    description: str


@dataclass(frozen=True)
class AgentRunResult:
    """Graph output paired with the selected agent and conversation ID."""

    agent_key: str
    thread_id: str
    state: Mapping[str, Any]
    database_session_id: Optional[int] = None


AGENT_OPTIONS: tuple[AgentOption, ...] = (
    AgentOption("backend", "Backend", "Flask/FastAPI routes, business logic, integrations, and tests."),
    AgentOption("ml", "ML/AI", "Training, evaluation, data pipelines, and experiments."),
    AgentOption("git", "Git", "Inspect, branch, commit, and push version-control work."),
    AgentOption("algorithms", "Algorithms", "Correctness- and complexity-sensitive implementation work."),
)

_AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    "backend": BackendAgent,
    "ml": MLAgent,
    "git": GitAgent,
    "algorithms": AlgoAgent,
}


class AgentRunner:
    """Runs exactly the agent selected by the caller; it never auto-routes.

    Keep one runner in ``st.session_state`` in a Streamlit app. That preserves
    each selected graph's in-memory checkpointer so confirmation resumes and
    follow-up messages use the same ``thread_id``.
    """

    def __init__(
        self,
        config: Config,
        *,
        agent_classes: Optional[Mapping[str, type[BaseAgent]]] = None,
        tools_by_task: Optional[Mapping[str, list[Any]]] = None,
        enable_sandbox: bool = False,
        storage: Optional[AgentStorage] = None,
    ) -> None:
        self.config = config
        self._agent_classes = dict(agent_classes or _AGENT_CLASSES)
        self._tools_by_task = dict(tools_by_task or TOOLS_BY_TASK)
        self.enable_sandbox = enable_sandbox
        self.storage = storage
        if enable_sandbox:
            self._tools_by_task = {
                task_key: [*tools, *SANDBOX_TOOLS]
                if task_key in {"backend", "ml", "algorithms"}
                else list(tools)
                for task_key, tools in self._tools_by_task.items()
            }
        self._agents: dict[str, BaseAgent] = {}


    @staticmethod
    def available_agents() -> tuple[AgentOption, ...]:
        """Options to display in a frontend selection control."""
        return AGENT_OPTIONS


    def get_agent(
        self,
        agent_key: str
    ) -> BaseAgent:
        """Return the explicitly selected agent, building it on first use."""
        key = _validate_agent_key(
            agent_key=agent_key,
            agent_classes=self._agent_classes,
            tools_by_task=self._tools_by_task
        )

        if key not in self._agents:
            self._agents[key] = self._agent_classes[key](self.config, self._tools_by_task[key])

        return self._agents[key]


    def run(
        self,
        agent_key: str,
        user_message: str,
        *,
        thread_id: Optional[str] = None,
        project_path: Optional[str] = None,
    ) -> AgentRunResult:

        """Invoke the selected agent with one user message."""
        if not user_message or not user_message.strip():
            raise ValueError("user_message must not be empty")

        key = _validate_agent_key(
            agent_key=agent_key,
            agent_classes=self._agent_classes,
            tools_by_task=self._tools_by_task
        )

        conversation_id = thread_id or uuid.uuid4().hex
        if self.storage is not None:
            self.storage.ensure_session(
                thread_id=conversation_id,
                agent_key=key,
                title=user_message.strip().replace("\n", " ")[:255],
                project_path=project_path,
            )
        graph = self.get_agent(key).graph

        state = graph.invoke(
            {"messages": [HumanMessage(content=user_message.strip())]},
            {"configurable": {"thread_id": conversation_id}},
        )

        if self.storage is not None:
            self.storage.record_state(conversation_id, state)

        return AgentRunResult(
            key,
            conversation_id,
            state,
            self.storage.session_id_for(conversation_id) if self.storage else None,
        )


    def resume_confirmation(
        self,
        agent_key: str,
        thread_id: str,
        decision: Mapping[str, Any],
    ) -> AgentRunResult:
        """Resume a selected graph after the dashboard approves or denies it."""
        if not thread_id:
            raise ValueError("thread_id is required to resume a confirmation")

        key = _validate_agent_key(agent_key, self._agent_classes, self._tools_by_task)

        if self.storage is not None:
            self.storage.record_confirmation_decision(thread_id=thread_id, decision=decision)

        state = self.get_agent(key).graph.invoke(
            Command(resume=dict(decision)),
            {"configurable": {"thread_id": thread_id}},
        )

        if self.storage is not None:
            self.storage.record_state(thread_id, state)

        return AgentRunResult(
            key,
            thread_id,
            state,
            self.storage.session_id_for(thread_id) if self.storage else None,
        )


def _validate_agent_key(
    agent_key: str,
    agent_classes: Mapping[str, type[BaseAgent]],
    tools_by_task: Mapping[str, list[Any]],
) -> str:
    key = (agent_key or "").strip().lower()

    if key not in agent_classes or key not in tools_by_task:
        allowed = ", ".join(option.key for option in AGENT_OPTIONS)
        raise ValueError(f"Unknown agent '{agent_key}'. Choose one of: {allowed}")

    return key
