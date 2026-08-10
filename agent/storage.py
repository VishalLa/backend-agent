from __future__ import annotations

import json
from typing import Any, Mapping

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from database.service.persistence import DatabaseLogService


class AgentStorage:
    """Adapt graph state into the database service's event-based API.

    Persistence deliberately never changes an agent result: if the database is
    unavailable, the in-memory conversation still continues and callers can
    surface the captured error in their UI.
    """

    def __init__(self, service: DatabaseLogService) -> None:
        self.service = service
        self._session_ids: dict[str, int] = {}
        self._message_cursors: dict[str, int] = {}
        self._tool_log_cursors: dict[str, int] = {}
        self._confirmation_ids: dict[tuple[str, str], int] = {}
        self.last_error: str | None = None


    def ensure_session(
        self,
        *,
        thread_id: str,
        agent_key: str,
        title: str | None = None,
        project_path: str | None = None,
    ) -> int | None:
        if thread_id in self._session_ids:
            return self._session_ids[thread_id]

        try:
            result = self.service.persist_event(
                "session.create",
                {
                    "title": (title or "Agent conversation")[:255],
                    "project_path": project_path or None,
                    "agent_type": agent_key,
                    "status": "active",
                },
            )

            session_id = int(result["id"])
            self._session_ids[thread_id] = session_id
            self.last_error = None

            return session_id

        except Exception as exc:  # Database failures must not stop the agent.
            self.last_error = str(exc)
            return None


    def record_state(
        self, 
        thread_id: str, 
        state: Mapping[str, Any]
    ) -> None:
        session_id = self._session_ids.get(thread_id)
        if session_id is None:
            return

        try:
            messages = list(state.get("messages", []))
            start = self._message_cursors.get(thread_id, 0)

            for message in messages[start:]:
                self._record_message(session_id, message)

            self._message_cursors[thread_id] = len(messages)

            self._record_tool_calls(thread_id, session_id, state, messages)
            self._record_pending_confirmations(thread_id, session_id, state)
            self.last_error = None

        except Exception as exc:  # Keep the graph usable even if a row is invalid.
            self.last_error = str(exc)


    def record_confirmation_decision(
        self,
        *,
        thread_id: str,
        decision: Mapping[str, Any],
    ) -> None:
        for (saved_thread_id, _call_id), confirmation_id in list(self._confirmation_ids.items()):
            if saved_thread_id != thread_id:
                continue

            try:
                self.service.persist_event(
                    "confirmation.decide",
                    {
                        "id": confirmation_id, 
                        "approved": bool(decision.get("approved"))
                    },
                )

            except Exception as exc:
                self.last_error = str(exc)
            return


    def session_id_for(
        self, 
        thread_id: str
    ) -> int | None:
        return self._session_ids.get(thread_id)


    def _record_message(
        self, 
        session_id: int, 
        message: BaseMessage
    ) -> None:
        if isinstance(message, SystemMessage):
            return
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, ToolMessage):
            role = "tool"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            return

        content = _message_content(message)
        if not content:
            return
        self.service.persist_event(
            "message.create",
            {"session_id": session_id, "role": role, "content": content},
        )


    def _record_tool_calls(
        self,
        thread_id: str,
        session_id: int,
        state: Mapping[str, Any],
        messages: list[BaseMessage],
    ) -> None:
        tool_outputs = {
            message.tool_call_id: _message_content(message)
            for message in messages
            if isinstance(message, ToolMessage) and message.tool_call_id
        }

        tool_logs = list(state.get("tool_logs", []))
        start = self._tool_log_cursors.get(thread_id, 0)
        
        for log in tool_logs[start:]:
            call_id = str(log.get("call_id", ""))
            output = tool_outputs.get(call_id)

            self.service.persist_event(
                "tool_call.create",
                {
                    "session_id": session_id,
                    "tool_name": str(log.get("tool_name", "unknown")),
                    "input_json": json.dumps(log.get("args", {}), default=str),
                    "output_json": output,
                    "status": "success" if log.get("success") else "failed",
                },
            )

        self._tool_log_cursors[thread_id] = len(tool_logs)


    def _record_pending_confirmations(
        self, 
        thread_id: str, 
        session_id: int, 
        state: Mapping[str, Any]
    ) -> None:
        raw_interrupts = state.get("__interrupt__", [])
        if not isinstance(raw_interrupts, (list, tuple)):
            raw_interrupts = [raw_interrupts]

        for raw_interrupt in raw_interrupts:
            payload = getattr(raw_interrupt, "value", raw_interrupt)
            if not isinstance(payload, Mapping) or payload.get("type") != "confirmation_required":
                continue

            call_id = str(payload.get("call_id", ""))
            key = (thread_id, call_id)

            if not call_id or key in self._confirmation_ids:
                continue

            description = f"{payload.get('tool_name', 'tool')} {json.dumps(payload.get('tool_args', {}), default=str)}"
            result = self.service.persist_event(
                "confirmation.create",
                {
                    "session_id": session_id,
                    "action_type": "destructive_action",
                    "description": description,
                },
            )
            self._confirmation_ids[key] = int(result["id"])


def _message_content(message: BaseMessage) -> str:
    content = message.content
    return content if isinstance(content, str) else str(content or "")
