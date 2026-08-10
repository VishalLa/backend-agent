"""Streamlit dashboard for the local coding agent.

Run with:
    streamlit run main.py
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agent import AgentRunner
from agent.storage import AgentStorage
from config import Config
from database.service.persistence import DatabaseLogService


PROVIDER_LABELS = {
    "api": "API (SambaNova → OpenRouter → Groq → Ollama fallback)",
    "local": "Local (Ollama only)",
}


@dataclass
class Conversation:
    """UI metadata for one LangGraph thread."""

    thread_id: str
    agent_key: str
    title: str
    provider: str
    sandbox_enabled: bool
    database_enabled: bool
    state: Mapping[str, Any] | None = None
    database_session_id: int | None = None
    restored: bool = False


def _initialise_session() -> None:
    st.session_state.setdefault("runners", {})
    st.session_state.setdefault("runner_errors", {})
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "active_thread_id" not in st.session_state:
        st.session_state.active_thread_id = None


def _get_runner(*, provider: str, sandbox_enabled: bool, database_enabled: bool) -> AgentRunner | None:
    """Return a checkpoint-preserving runner for the chosen integrations."""
    integration_key = (provider, sandbox_enabled, database_enabled)
    if integration_key in st.session_state.runners:
        return st.session_state.runners[integration_key]

    try:
        config = Config.from_env(provider=provider)
        storage = None
        if database_enabled:
            service = DatabaseLogService(config.postgres_url, echo=config.postgres_echo)
            service.init_db()
            storage = AgentStorage(service)
        runner = AgentRunner(config, enable_sandbox=sandbox_enabled, storage=storage)
    except Exception as exc:  # Configuration errors should be visible in the UI.
        st.session_state.runner_errors[integration_key] = str(exc)
        return None

    st.session_state.runners[integration_key] = runner
    st.session_state.runner_errors.pop(integration_key, None)
    return runner


def _postgres_url() -> str:
    return os.environ.get(
        "POSTGRES_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/data_agent",
    )


def _get_history_storage() -> AgentStorage | None:
    """Read-only storage used only to list/restore past conversations."""
    if "history_storage" in st.session_state:
        return st.session_state.history_storage
    if st.session_state.get("history_storage_unavailable"):
        return None

    try:
        service = DatabaseLogService(_postgres_url(), echo=False)
        service.init_db()
        storage = AgentStorage(service)
    except Exception as exc:
        st.session_state.history_storage_unavailable = True
        st.session_state.history_storage_error = str(exc)
        return None

    st.session_state.history_storage = storage
    st.session_state.pop("history_storage_unavailable", None)
    st.session_state.pop("history_storage_error", None)
    return storage


def _retry_history_storage() -> None:
    """Let the user retry connecting after fixing the database."""
    st.session_state.pop("history_storage", None)
    st.session_state.pop("history_storage_unavailable", None)
    st.session_state.pop("history_storage_error", None)


def _new_conversation(
    agent_key: str,
    *,
    provider: str,
    sandbox_enabled: bool,
    database_enabled: bool,
) -> Conversation:
    conversation = Conversation(
        thread_id=uuid.uuid4().hex,
        agent_key=agent_key,
        title="New conversation",
        provider=provider,
        sandbox_enabled=sandbox_enabled,
        database_enabled=database_enabled,
    )
    st.session_state.conversations[conversation.thread_id] = conversation
    st.session_state.active_thread_id = conversation.thread_id
    return conversation


def _active_conversation(
    agent_key: str,
    *,
    provider: str,
    sandbox_enabled: bool,
    database_enabled: bool,
) -> Conversation:
    thread_id = st.session_state.active_thread_id
    conversation = st.session_state.conversations.get(thread_id)
    if conversation is not None and conversation.restored:
        return conversation
    if (
        conversation is None
        or conversation.agent_key != agent_key
        or conversation.provider != provider
        or conversation.sandbox_enabled != sandbox_enabled
        or conversation.database_enabled != database_enabled
    ):
        return _new_conversation(
            agent_key,
            provider=provider,
            sandbox_enabled=sandbox_enabled,
            database_enabled=database_enabled,
        )
    return conversation


def _restore_persisted_conversations(storage: AgentStorage | None, agent_key: str) -> None:
    """Add database transcripts to session state without recreating a graph checkpoint."""
    if storage is None:
        return
    try:
        for row in storage.list_conversations(agent_key=agent_key):
            database_session_id = int(row["id"])
            thread_id = f"database-{database_session_id}"
            existing = st.session_state.conversations.get(thread_id)
            if existing is not None:
                continue
            st.session_state.conversations[thread_id] = Conversation(
                thread_id=thread_id,
                agent_key=str(row["agent_type"]),
                title=str(row["title"]),
                provider="",
                sandbox_enabled=False,
                database_enabled=True,
                state={
                    "messages": storage.load_messages(database_session_id),
                    "status": "stored transcript",
                    "iterations": 0,
                },
                database_session_id=database_session_id,
                restored=True,
            )
    except Exception as exc:
        storage.last_error = str(exc)


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return str(content)


def _render_messages(state: Mapping[str, Any] | None) -> None:
    if not state or not state.get("messages"):
        st.info("Choose an agent, then describe the task you want it to perform.")
        return

    for message in state["messages"]:
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(_message_text(message))
        elif isinstance(message, ToolMessage):
            with st.expander(f"Tool output: {message.name or 'tool'}"):
                st.code(_message_text(message), language="text")
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                text = _message_text(message)
                if text:
                    st.markdown(text)
                for call in message.tool_calls or []:
                    st.caption(f"Requested tool: `{call.get('name', 'unknown')}`")


def _interrupt_payloads(state: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """Extract JSON-safe LangGraph interrupt values across supported versions."""
    if not state:
        return []

    values = state.get("__interrupt__", [])
    if not isinstance(values, (list, tuple)):
        values = [values]

    payloads: list[Mapping[str, Any]] = []
    for value in values:
        payload = getattr(value, "value", value)
        if isinstance(payload, Mapping) and payload.get("type") == "confirmation_required":
            payloads.append(payload)
    return payloads


def _render_confirmations(runner: AgentRunner | None, conversation: Conversation) -> None:
    requests = _interrupt_payloads(conversation.state)
    if not requests:
        return

    st.subheader("Confirmation required")
    st.warning("The agent requested an action that can change your system.")

    for request in requests:
        tool_name = str(request.get("tool_name", "unknown tool"))
        call_id = str(request.get("call_id", tool_name))
        st.write(f"**{tool_name}**")
        st.caption(str(request.get("reason", "")))
        st.json(request.get("tool_args", {}), expanded=False)
        reason = st.text_input("Reason (optional)", key=f"reason-{call_id}")
        approve, deny = st.columns(2)
        approved = approve.button("Approve", key=f"approve-{call_id}", type="primary")
        denied = deny.button("Deny", key=f"deny-{call_id}")

        if (approved or denied) and runner is not None:
            with st.spinner("Resuming agent…"):
                try:
                    result = runner.resume_confirmation(
                        agent_key=conversation.agent_key,
                        thread_id=conversation.thread_id,
                        decision={"approved": approved, "reason": reason or None},
                    )
                    conversation.state = result.state
                    conversation.database_session_id = result.database_session_id
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not resume the agent: {exc}")


def _project_context(project_path: str, selected_file: str) -> str:
    parts = []
    if project_path.strip():
        parts.append(f"Project root: {project_path.strip()}")
    if selected_file.strip():
        parts.append(f"Relevant file or folder: {selected_file.strip()}")
    return "\n".join(parts)


def main() -> None:
    st.set_page_config(page_title="Local Coding Agent", page_icon="🤖", layout="wide")
    _initialise_session()

    st.title("Local Coding Agent")
    st.caption("Select a specialist, review the conversation, and approve sensitive tool actions here.")

    options = AgentRunner.available_agents()
    labels = {option.key: option.label for option in options}

    with st.sidebar:
        st.header("Workspace")
        selected_agent = st.selectbox(
            "Agent",
            options=[option.key for option in options],
            format_func=lambda key: labels[key],
            help="Each specialist has its own prompt, model selection, and tool allow-list.",
        )
        selected_option = next(option for option in options if option.key == selected_agent)
        st.caption(selected_option.description)

        # Provider Selection UI Control
        selected_provider = st.selectbox(
            "LLM Provider",
            options=list(PROVIDER_LABELS.keys()),
            format_func=lambda key: PROVIDER_LABELS[key],
            help="Choose 'api' for cloud APIs with fallback, or 'local' to run Ollama locally.",
        )

        project_path = st.text_input("Project folder", placeholder=str(Path.cwd()))
        selected_file = st.text_input("Relevant file or folder", placeholder="Optional path or filename")

        st.divider()
        st.header("Integrations")
        database_enabled = st.checkbox(
            "Store conversation in database",
            key="database_enabled",
            help="Creates sessions, messages, tool-call records, and approval decisions in the configured database.",
        )
        sandbox_enabled = st.checkbox(
            "Use Docker sandbox execution",
            key="sandbox_enabled",
            help="Adds an isolated Python/shell execution tool to backend, ML, and algorithms agents.",
        )
        if sandbox_enabled:
            st.caption("The sandbox is ephemeral, network-isolated, and has no host project files.")
        if database_enabled:
            st.caption("Uses the `postgres_url` configured in your environment.")

    runner = _get_runner(
        provider=selected_provider,
        sandbox_enabled=sandbox_enabled,
        database_enabled=database_enabled,
    )
    if database_enabled:
        _restore_persisted_conversations(runner, selected_agent)

    with st.sidebar:
        st.divider()
        st.header("Previous conversations")
        matching = [
            saved
            for saved in st.session_state.conversations.values()
            if saved.agent_key == selected_agent
            and (
                saved.restored
                or (
                    saved.provider == selected_provider
                    and saved.sandbox_enabled == sandbox_enabled
                    and saved.database_enabled == database_enabled
                )
            )
        ]
        for saved in reversed(matching):
            active = saved.thread_id == st.session_state.active_thread_id
            suffix = " · stored" if saved.restored else ""
            if st.button(
                f"{saved.title}{suffix}",
                key=f"conversation-{saved.thread_id}",
                type="primary" if active else "secondary",
                width="stretch",
            ):
                st.session_state.active_thread_id = saved.thread_id
                st.rerun()

        if st.button("New conversation", icon=":material/add:", width="stretch"):
            _new_conversation(
                selected_agent,
                provider=selected_provider,
                sandbox_enabled=sandbox_enabled,
                database_enabled=database_enabled,
            )
            st.rerun()

    conversation = _active_conversation(
        selected_agent,
        provider=selected_provider,
        sandbox_enabled=sandbox_enabled,
        database_enabled=database_enabled,
    )

    runner_error = st.session_state.runner_errors.get(
        (selected_provider, sandbox_enabled, database_enabled)
    )
    if runner_error:
        st.error(
            "The selected configuration could not start. Set `SAMBANOVA_API_KEY`, `GROQ_API_KEY`, "
            f"and `OPENROUTER_API_KEY` (and start PostgreSQL when storage is enabled). Details: {runner_error}"
        )

    transcript, details = st.columns([4, 1])
    with transcript:
        st.subheader(f"{labels[selected_agent]} output")
        _render_messages(conversation.state)
        _render_confirmations(runner, conversation)
    with details:
        st.subheader("Session")
        st.code(conversation.thread_id, language=None)
        if conversation.state:
            st.caption(f"Status: {conversation.state.get('status', 'running')}")
            st.caption(f"Iterations: {conversation.state.get('iterations', 0)}")
        if conversation.database_session_id:
            st.caption(f"Database session: {conversation.database_session_id}")
        if database_enabled and runner is not None and runner.storage and runner.storage.last_error:
            st.warning(f"Database storage issue: {runner.storage.last_error}")

    prompt = st.chat_input("Describe the task for the selected agent")
    if prompt:
        if runner is None:
            st.error("Configure the required provider keys before starting a conversation.")
            return

        if conversation.restored:
            conversation = _new_conversation(
                selected_agent,
                provider=selected_provider,
                sandbox_enabled=sandbox_enabled,
                database_enabled=database_enabled,
            )
            st.info("Started a new conversation from the stored transcript. LangGraph checkpoints are only available while the original app session is running.")

        context = _project_context(project_path, selected_file)
        full_prompt = f"{context}\n\nTask: {prompt}" if context else prompt
        if conversation.title == "New conversation":
            conversation.title = prompt.strip().replace("\n", " ")[:48] or "New conversation"

        with st.spinner("Agent is working…"):
            try:
                result = runner.run(
                    agent_key=selected_agent,
                    user_message=full_prompt,
                    thread_id=conversation.thread_id,
                    project_path=project_path or None,
                )
                conversation.state = result.state
                conversation.database_session_id = result.database_session_id
                st.rerun()
            except Exception as exc:
                st.error(f"The agent could not complete this request: {exc}")


if __name__ == "__main__":
    main()
