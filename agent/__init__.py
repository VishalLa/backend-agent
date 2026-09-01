from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from config import Config

from .eval import EvalCase, EvalHarness, EvalResult
from .graphs.algo_graph import AlgoAgent
from .graphs.backend_graph import BackendAgent
from .graphs.base_graph import BaseAgent
from .graphs.git_graph import GitAgent
from .graphs.ml_graph import MLAgent
from .tools import SANDBOX_TOOLS, TOOLS_BY_TASK
from .worktree import GitWorktree

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
    worktree_path: Optional[str] = None


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


class TaskRouter:
    """Choose the correct specialist agent for a user request."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._llm = None
        

    def _classify_with_model(self, user_message: str) -> str:
        try:
            from agent.llm import ChatModel

            self._llm = self._llm or ChatModel(self.config).get_llm()
            prompt = (
                "You are a routing classifier. Return exactly one of: backend, ml, git, algorithms.\n"
                f"User request: {user_message.strip()}"
            )
            response = self._llm.invoke([
                SystemMessage(content="Return only the agent key and nothing else."),
                HumanMessage(content=prompt),
            ])
            
            content = getattr(response, "content", str(response))
            if isinstance(content, list):
                content = " ".join(str(part.get("text", part)) for part in content if isinstance(part, dict))
                
            text = str(content).strip().lower()
            if text in {"backend", "ml", "git", "algorithms"}:
                return text
            
        except Exception:
            pass
        return self._keyword_fallback(user_message)


    @staticmethod
    def _keyword_fallback(user_message: str) -> str:
        text = (user_message or "").lower()

        git_markers = (
            "git", "branch", "commit", "merge", "pull request", "checkout",
            "push", "status", "diff", "repo", "repository", "tag"
        )
        ml_markers = (
            "train", "model", "dataset", "accuracy", "evaluate", "mlflow",
            "tensor", "pytorch", "tensorflow", "embedding", "vector", "notebook"
        )
        algorithm_markers = (
            "algorithm", "complexity", "sort", "search", "graph", "heap",
            "dynamic programming", "dp", "leetcode", "optimi", "binary search"
        )

        if any(marker in text for marker in git_markers):
            return "git"
        if any(marker in text for marker in ml_markers):
            return "ml"
        if any(marker in text for marker in algorithm_markers):
            return "algorithms"
        return "backend"

    def route(self, user_message: str) -> str:
        return self._classify_with_model(user_message)


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
        enable_worktree: bool = False,
        storage: Optional[AgentStorage] = None,
    ) -> None:
        self.config = config
        self._agent_classes = dict(agent_classes or _AGENT_CLASSES)
        self._tools_by_task = dict(tools_by_task or TOOLS_BY_TASK)
        self.dispatcher = TaskRouter(config)
        self.enable_sandbox = enable_sandbox
        self.enable_worktree = enable_worktree
        self.storage = storage
        self._worktree_managers: dict[str, GitWorktree] = {}
        self._worktree_task_ids: dict[str, tuple[str, str]] = {}
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


    def route_task(self, user_message: str) -> str:
        if not user_message or not user_message.strip():
            raise ValueError("user_message must not be empty")
        return self.dispatcher.route(user_message)


    def _project_root_for_run(
        self,
        conversation_id: str,
        project_path: Optional[str],
    ) -> Optional[str]:
        """Return the tool root, creating one isolated worktree per thread when enabled."""
        if not self.enable_worktree:
            return str(Path(project_path).expanduser().resolve()) if project_path else None

        existing = self._worktree_task_ids.get(conversation_id)
        if existing is not None:
            manager_key, task_id = existing
            return str(self._worktree_managers[manager_key].get_worktree_path(task_id))

        source_root = Path(project_path or Path.cwd()).expanduser().resolve()
        manager_key = str(source_root)
        manager = self._worktree_managers.get(manager_key)
        if manager is None:
            manager = GitWorktree(source_root)
            self._worktree_managers[manager_key] = manager
        task_id = conversation_id
        worktree_path = manager.create_worktree(task_id)
        self._worktree_task_ids[conversation_id] = (manager_key, task_id)
        return str(worktree_path)


    def get_worktree_summary(
        self,
        thread_id: str
    ) -> str:
        manager, task_id = self._worktree_for_thread(thread_id)
        return manager.get_diff_summary(task_id)


    def merge_worktree(
        self,
        thread_id: str
    ) -> str:
        manager, task_id = self._worktree_for_thread(thread_id)
        result = manager.merge_worktree(task_id)
        self._worktree_task_ids.pop(thread_id, None)
        return result


    def discard_worktree(
        self,
        thread_id: str
    ) -> str:
        manager, task_id = self._worktree_for_thread(thread_id)
        result = manager.discard_worktree(task_id)
        self._worktree_task_ids.pop(thread_id, None)
        return result


    def _worktree_for_thread(
        self,
        thread_id: str
    ) -> tuple[GitWorktree, str]:
        try:
            manager_key, task_id = self._worktree_task_ids[thread_id]
        except KeyError as exc:
            raise ValueError(f"No active worktree for thread '{thread_id}'") from exc
        return self._worktree_managers[manager_key], task_id


    def run(
        self,
        agent_key: Optional[str],
        user_message: str,
        *,
        thread_id: Optional[str] = None,
        project_path: Optional[str] = None,
    ) -> AgentRunResult:

        """Invoke the selected agent with one user message."""
        if not user_message or not user_message.strip():
            raise ValueError("user_message must not be empty")

        key = (
            self.route_task(user_message)
            if agent_key is None or agent_key.strip() == ""
            else _validate_agent_key(
                agent_key=agent_key,
                agent_classes=self._agent_classes,
                tools_by_task=self._tools_by_task,
            )
        )

        conversation_id = thread_id or uuid.uuid4().hex
        tool_project_root = self._project_root_for_run(conversation_id, project_path)
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
            {"configurable": {"thread_id": conversation_id, "project_root": tool_project_root}},
        )

        if self.storage is not None:
            self.storage.record_state(conversation_id, state)

        return AgentRunResult(
            key,
            conversation_id,
            state,
            self.storage.session_id_for(conversation_id) if self.storage else None,
            tool_project_root if self.enable_worktree else None,
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
            {"configurable": {
                "thread_id": thread_id,
                "project_root": self._project_root_for_run(thread_id, None),
            }},
        )

        if self.storage is not None:
            self.storage.record_state(thread_id, state)

        return AgentRunResult(
            key,
            thread_id,
            state,
            self.storage.session_id_for(thread_id) if self.storage else None,
            self._project_root_for_run(thread_id, None) if self.enable_worktree else None,
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
