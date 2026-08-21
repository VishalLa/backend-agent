from __future__ import annotations

import time
from abc import ABC
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from config import Config
from log.log_event import log_event, safe_args
from schema.agent_schema import AgentState, ConfirmationRequest, ToolCallLog

from ..confirmation import (
    confirmation_request_payload,
    needs_confirmation,
    parse_confirmation_decision,
)
from ..context_window import ContextWindowHandler
from ..llm import ChatModel
from ..task_profile import filter_tools_for_task

from .helper import (
    MAX_EMPTY_RESPONSE_RETRIES,
    MAX_TOOL_HALLUCINATION_RETRIES,
    ProjectRootViolation,
    _detect_read_only_stagnation,
    _extract_leaked_tool_call,
    _identify_provider,
    _last_ai_message_with_tool_calls,
    _prepare_messages_for_model,
    _project_root,
    _thread_id,
    _truncate,
    classify_error,
    resolve_tool_path_args,
)

_GENERIC_FALLBACK_SYSTEM_PROMPT = (
    "You are a coding agent. Inspect before editing, use only the provided "
    "tools, and verify changes before claiming they succeeded."
)


class BaseAgent(ABC):
    """
    Generic LangGraph coding agent: agent_node -> confirmation_node ->
    tool_node -> agent_node, with a system prompt loaded from
    md/<TASK_MODE>_agent.md.

    To add a new agent, subclass this and set at minimum:

        class DocsAgent(BaseAgent):
            TASK_MODE = "docs"  # must match a task_profile.TASK_PROFILES key
            FALLBACK_SYSTEM_PROMPT = "You are a documentation-writing agent..."

    That's the whole thing - no graph-building code needed. Override
    agent_node / tool_node / confirmation_node / route_after_agent in a
    subclass only if that specific agent genuinely needs different node
    *behavior*, not just a different prompt, tool set, or model.

    Monitoring: every node entry/exit and lifecycle boundary (init, graph
    build, routing decisions) emits a log_event so subclasses get uniform
    observability for free without needing to add their own logging.
    """

    TASK_MODE: ClassVar[str] = ""  # e.g. "backend" / "ml" / "git" / "algorithms"

    MD_FILENAME: ClassVar[Optional[str]] = None  # defaults to f"{TASK_MODE}_agent.md"
    FALLBACK_SYSTEM_PROMPT: ClassVar[str] = _GENERIC_FALLBACK_SYSTEM_PROMPT

    MD_DIR: ClassVar[Path] = Path(__file__).resolve().parent / "md"

    def __init__(
        self,
        config: Config,
        tools: list[Any],
        checkpointer: Optional[Any] = None,
    ) -> None:
        init_started_at = time.monotonic()

        if not self.TASK_MODE:
            raise NotImplementedError(
                f"{type(self).__name__} must set a class-level TASK_MODE "
                "(matching a task_profile.TASK_PROFILES key)."
            )
        if not tools:
            raise ValueError(f"{type(self).__name__} requires at least one tool")

        self.config = config.model_copy(update={"agent_type": self.TASK_MODE})

        log_event(
            self.config.log_file,
            "agent_init_started",
            agent_class=type(self).__name__,
            task_mode=self.TASK_MODE,
            requested_tool_count=len(tools),
            has_checkpointer=checkpointer is not None,
        )

        if config.agent_type != self.TASK_MODE:
            log_event(
                self.config.log_file,
                "agent_type_overridden",
                from_task=config.agent_type,
                to_task=self.TASK_MODE,
            )

        self.tools = filter_tools_for_task(self.TASK_MODE, tools)
        self.tool_names = [tool.name for tool in self.tools]

        log_event(
            self.config.log_file,
            "agent_tools_filtered",
            task_mode=self.TASK_MODE,
            input_tool_count=len(tools),
            filtered_tool_count=len(self.tool_names),
            tool_names=sorted(self.tool_names),
        )

        duplicates = {name for name in self.tool_names if self.tool_names.count(name) > 1}
        if duplicates:
            log_event(
                self.config.log_file,
                "agent_init_failed",
                task_mode=self.TASK_MODE,
                reason="duplicate_tool_names",
                duplicates=sorted(duplicates),
            )
            raise ValueError(f"Duplicate tool names in toolset: {sorted(duplicates)}")

        self.tools_by_name = {tool.name: tool for tool in self.tools}

        self.llm_with_tools = ChatModel(self.config).get_llm_with_tools(self.tools)
        
        tool_overhead_tokens = sum(
            len(getattr(t, "description", "") or "") for t in self.tools
        ) // 4 + 50 * len(self.tools)
        self.context_window = ContextWindowHandler(
            self.config, tool_overhead_tokens=tool_overhead_tokens
        )

        self.system_prompt = self._load_system_prompt()
        if "execute_in_sandbox" in self.tools_by_name:
            self.system_prompt += (
                "\n\n## Isolated execution\n"
                "`execute_in_sandbox` is available for self-contained Python or shell verification. "
                "Prefer it over host shell execution when the check does not need project files. "
                "The sandbox has no network access and no host-project mount."
            )
        self.graph = self._build_graph(checkpointer)

        log_event(
            self.config.log_file,
            "agent_init_completed",
            agent_class=type(self).__name__,
            task_mode=self.TASK_MODE,
            tool_count=len(self.tool_names),
            system_prompt_chars=len(self.system_prompt),
            duration_ms=int((time.monotonic() - init_started_at) * 1000),
        )


    @property
    def _md_filename(self) -> str:
        return self.MD_FILENAME or f"{self.TASK_MODE}_agent.md"

    def _load_system_prompt(self) -> str:
        path = self.MD_DIR / self._md_filename
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                log_event(
                    self.config.log_file,
                    "agent_md_load_succeeded",
                    path=str(path),
                    chars=len(text),
                )
                return text
        except OSError as exc:
            log_event(
                self.config.log_file,
                "agent_md_load_failed",
                path=str(path),
                error=str(exc),
            )
        log_event(
            self.config.log_file,
            "agent_md_fallback_used",
            path=str(path),
            task_mode=self.TASK_MODE,
        )
        return self.FALLBACK_SYSTEM_PROMPT

    def agent_node(self, state: AgentState) -> dict[str, Any]:
        thread_id = _thread_id()
        node_started_at = time.monotonic()

        log_event(
            self.config.log_file,
            "node_entered",
            node="agent",
            thread_id=thread_id,
            iterations=state.iterations,
            status=state.status,
            message_count=len(state.messages),
        )

        if state.status in {"cancelled", "error", "max_iterations_reached"}:
            log_event(
                self.config.log_file,
                "node_exited",
                node="agent",
                thread_id=thread_id,
                reason="terminal_status_short_circuit",
                status=state.status,
                duration_ms=int((time.monotonic() - node_started_at) * 1000),
            )
            return {}

        if state.iterations >= self.config.max_iterations:
            log_event(
                self.config.log_file,
                "agent_max_iterations_reached",
                thread_id=thread_id,
                iterations=state.iterations,
                max_iterations=self.config.max_iterations,
            )
            log_event(
                self.config.log_file,
                "node_exited",
                node="agent",
                thread_id=thread_id,
                reason="max_iterations_reached",
                duration_ms=int((time.monotonic() - node_started_at) * 1000),
            )
            return {
                "messages": [AIMessage(content=f"Stopping: reached max_iterations ({self.config.max_iterations}).")],
                "status": "max_iterations_reached",
            }

        messages = list(state.messages)
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=self.system_prompt), *messages]

        model_input = _prepare_messages_for_model(messages)

        
        project_root = _project_root()
        if project_root:
            model_input.append(
                SystemMessage(
                    content=(
                        f"Project root for this task: {project_root}\n"
                        "All relative file/cwd tool arguments are resolved against this "
                        "root automatically - do not prefix paths with it yourself, and "
                        "do not use paths outside it."
                    )
                )
            )

        stagnation_nudge = _detect_read_only_stagnation(state.tool_logs)
        if stagnation_nudge:
            log_event(
                self.config.log_file,
                "llm_read_only_stagnation_detected",
                thread_id=thread_id,
                tool_name=(state.tool_logs[-1] or {}).get("tool_name"),
            )
            model_input.append(HumanMessage(content=stagnation_nudge))

        retries_left = self.config.max_retries
        empty_retries_left = MAX_EMPTY_RESPONSE_RETRIES
        hallucination_retries_left = MAX_TOOL_HALLUCINATION_RETRIES
        context_compression_retries_left = 1
        last_error: Optional[Exception] = None

        while True:
            try:
                response = self.llm_with_tools.invoke(model_input)
                has_tool_calls = bool(getattr(response, "tool_calls", None))
                content = response.content if isinstance(response.content, str) else str(response.content or "")

                if not has_tool_calls and content.strip():
                    leaked = _extract_leaked_tool_call(content, self.tools_by_name)

                    if leaked:
                        response = AIMessage(
                            content="",
                            tool_calls=[leaked],
                            response_metadata=getattr(response, "response_metadata", {}) or {},
                        )
                        has_tool_calls = True
                        log_event(
                            self.config.log_file,
                            "llm_tool_call_leaked_as_text",
                            thread_id=thread_id,
                            tool_name=leaked["name"]
                        )

                if not has_tool_calls and not content.strip() and empty_retries_left:
                    empty_retries_left -= 1
                    log_event(
                        self.config.log_file,
                        "llm_empty_response_retry",
                        thread_id=thread_id,
                        retries_left=empty_retries_left,
                    )
                    model_input.append(
                        HumanMessage(content="SYSTEM: Your last response was empty. Continue with a tool call or a final answer.")
                    )
                    continue

                log_event(
                    self.config.log_file, 
                    "llm_call", 
                    thread_id=thread_id, 
                    provider=_identify_provider(response, self.config)
                )
                log_event(
                    self.config.log_file,
                    "node_exited",
                    node="agent",
                    thread_id=thread_id,
                    reason="llm_response_received",
                    has_tool_calls=has_tool_calls,
                    tool_call_count=len(getattr(response, "tool_calls", None) or []),
                    duration_ms=int((time.monotonic() - node_started_at) * 1000),
                )
                return {
                    "messages": [response],
                    "iterations": state.iterations + 1
                }

            except Exception as exc:
                last_error = exc
                detail = classify_error(str(exc))

                if detail["kind"] == "context_length":
                    log_event(
                        self.config.log_file,
                        "llm_call_failed",
                        thread_id=thread_id,
                        reason="context_too_large",
                        error=str(exc)
                    )

                    if context_compression_retries_left:
                        context_compression_retries_left -= 1
                        compressed = self.context_window.prepare(messages, force_summarize=True)

                        if compressed.summarized or compressed.used_hard_truncation:
                            model_input = _prepare_messages_for_model(compressed.messages)
                            log_event(
                                self.config.log_file,
                                "context_compressed",
                                thread_id=thread_id,
                                summarized=compressed.summarized,
                                used_hard_truncation=compressed.used_hard_truncation,
                            )
                            continue

                    log_event(
                        self.config.log_file,
                        "node_exited",
                        node="agent",
                        thread_id=thread_id,
                        reason="context_too_large_unrecoverable",
                        duration_ms=int((time.monotonic() - node_started_at) * 1000),
                    )
                    return {
                        "messages": [
                            AIMessage(
                                content="The conversation is still too large after context compression. Start a new thread or reduce the supplied context."
                            )
                        ],
                        "iterations": state.iterations + 1,
                        "status": "error",
                        "error": str(exc),
                    }

                if "tool" in str(exc).lower() and "name" in str(exc).lower() and hallucination_retries_left:
                    hallucination_retries_left -= 1
                    log_event(
                        self.config.log_file,
                        "llm_tool_hallucination_retry",
                        thread_id=thread_id,
                        retries_left=hallucination_retries_left,
                        error=str(exc),
                    )
                    model_input.append(HumanMessage(content=(
                        "SYSTEM: Use only these exact tool names: " + ", ".join(sorted(self.tools_by_name))
                    )))
                    continue

                if not detail["retryable"] or retries_left <= 0:
                    break

                retries_left -= 1
                wait = detail["wait_seconds"] or self.config.retry_backoff_seconds
                log_event(
                    self.config.log_file,
                    "llm_retry",
                    thread_id=thread_id,
                    kind=detail["kind"],
                    wait_seconds=wait
                )
                time.sleep(wait)

        error_text = str(last_error) if last_error else "unknown model error"
        log_event(
            self.config.log_file,
            "llm_call_failed",
            thread_id=thread_id,
            reason="retries_exhausted",
            error=error_text
        )
        log_event(
            self.config.log_file,
            "node_exited",
            node="agent",
            thread_id=thread_id,
            reason="retries_exhausted",
            duration_ms=int((time.monotonic() - node_started_at) * 1000),
        )

        return {
            "messages": [AIMessage(content=f"I could not complete the model call: {error_text}")],
            "iterations": state.iterations + 1,
            "status": "error",
            "error": error_text,
        }


    def _dedup_and_validate_tool_calls(
        self,
        tool_calls: list,
        thread_id: str
    ) -> list:
        seen_ids: set[str] = set()
        valid = []

        for tc in tool_calls:
            call_id = tc.get("id")
            name = tc.get("name")

            if not call_id or not name:
                log_event(
                    self.config.log_file,
                    "tool_call_malformed",
                    thread_id=thread_id,
                    raw=str(tc)[:300]
                )
                continue

            if call_id in seen_ids:
                log_event(
                    self.config.log_file,
                    "tool_call_duplicate_id",
                    thread_id=thread_id,
                    tool_name=name,
                    call_id=call_id,
                )
                continue

            seen_ids.add(call_id)
            valid.append(tc)

        if len(valid) != len(tool_calls):
            log_event(
                self.config.log_file,
                "tool_calls_validated",
                thread_id=thread_id,
                input_count=len(tool_calls),
                valid_count=len(valid),
                dropped_count=len(tool_calls) - len(valid),
            )
        return valid


    def confirmation_node(self, state: AgentState) -> dict[str, Any]:
        """Gate any pending tool calls that need human approval.

        Uses LangGraph's interrupt()/Command(resume=...), so the graph
        actually pauses here (requires the checkpointer from _build_graph)
        rather than blocking on input() inside the node. Approved calls are
        left in place for tool_node to execute; denied calls are stripped
        from the AIMessage and get a synthetic ToolMessage explaining the
        denial instead, since every tool_call_id must have a matching
        tool response.
        """
        thread_id = _thread_id()
        node_started_at = time.monotonic()

        log_event(
            self.config.log_file,
            "node_entered",
            node="confirmation",
            thread_id=thread_id,
        )

        last_ai = _last_ai_message_with_tool_calls(state.messages)
        if last_ai is None or not last_ai.tool_calls:
            log_event(
                self.config.log_file,
                "node_exited",
                node="confirmation",
                thread_id=thread_id,
                reason="no_pending_tool_calls",
                duration_ms=int((time.monotonic() - node_started_at) * 1000),
            )
            return {}

        valid_calls = self._dedup_and_validate_tool_calls(last_ai.tool_calls, thread_id)

        kept_calls = []
        denial_messages: list[ToolMessage] = []

        for tc in valid_calls:
            call_id = tc["id"]
            name = tc["name"]
            args = tc.get("args") or {}

            if not needs_confirmation(name, args, confirm_all=self.config.confirm_all_tools):
                kept_calls.append(tc)
                continue

            log_event(
                self.config.log_file,
                "tool_confirmation_requested",
                thread_id=thread_id,
                tool_name=name,
                call_id=call_id,
                args=safe_args(args),
            )

            request = ConfirmationRequest(
                tool_name=name,
                tool_args=args,
                call_id=call_id,
            )
            raw_decision = interrupt(confirmation_request_payload(request))
            decision = parse_confirmation_decision(raw_decision)

            log_event(
                self.config.log_file,
                "tool_confirmation_decided",
                thread_id=thread_id,
                tool_name=name,
                call_id=call_id,
                approved=decision.approved,
                args=safe_args(args),
            )

            if decision.approved:
                kept_calls.append(tc)
            else:
                reason = f" — {decision.reason}" if decision.reason else ""
                denial_messages.append(
                    ToolMessage(
                        content=f"BLOCKED: user declined this call{reason}.",
                        tool_call_id=call_id,
                        name=name,
                    )
                )

        updated_ai = last_ai.model_copy(update={"tool_calls": kept_calls})

        log_event(
            self.config.log_file,
            "node_exited",
            node="confirmation",
            thread_id=thread_id,
            reason="confirmation_processed",
            kept_call_count=len(kept_calls),
            denied_call_count=len(denial_messages),
            duration_ms=int((time.monotonic() - node_started_at) * 1000),
        )

        return {"messages": [updated_ai, *denial_messages]}


    def tool_node(self, state: AgentState) -> dict[str, Any]:
        """Execute whatever tool calls survived confirmation_node."""
        thread_id = _thread_id()
        node_started_at = time.monotonic()

        log_event(
            self.config.log_file,
            "node_entered",
            node="tools",
            thread_id=thread_id,
        )

        last_ai = _last_ai_message_with_tool_calls(state.messages)
        if last_ai is None or not last_ai.tool_calls:
            log_event(
                self.config.log_file,
                "node_exited",
                node="tools",
                thread_id=thread_id,
                reason="no_tool_calls_to_execute",
                duration_ms=int((time.monotonic() - node_started_at) * 1000),
            )
            return {}

        valid_calls = self._dedup_and_validate_tool_calls(last_ai.tool_calls, thread_id)

        tool_messages: list[ToolMessage] = []
        tool_logs: list[dict[str, Any]] = []

        project_root = _project_root()

        for tc in valid_calls:
            call_id = tc["id"]
            name = tc["name"]
            args = tc.get("args") or {}
            call_started_at = time.monotonic()

            tool = self.tools_by_name.get(name)
            if tool is None:
                output = f"ERROR: unknown tool '{name}'. Valid tools: {', '.join(sorted(self.tools_by_name))}"
                success = False
                log_event(
                    self.config.log_file,
                    "tool_call_unknown_tool",
                    thread_id=thread_id,
                    tool_name=name,
                    call_id=call_id,
                )
            else:
                try:
                    resolved_args = resolve_tool_path_args(name, args, project_root)
                except ProjectRootViolation as exc:
                    output = f"ERROR: {exc}"
                    success = False
                    log_event(
                        self.config.log_file,
                        "tool_call_path_rejected",
                        thread_id=thread_id,
                        tool_name=name,
                        call_id=call_id,
                        project_root=project_root,
                        error=str(exc),
                    )
                else:
                    try:
                        raw_output = tool.invoke(resolved_args)
                        output = (
                            str(raw_output)
                            if name == "read_file"
                            else _truncate(raw_output, continuation_hint_tool=name)
                        )
                        success = True
                    except Exception as exc:  # noqa: BLE001 - tool errors are reported to the model, not raised
                        output = f"ERROR: tool '{name}' raised an exception: {exc}"
                        success = False

            log_event(
                self.config.log_file,
                "tool_call_executed",
                thread_id=thread_id,
                tool_name=name,
                success=success,
                args=safe_args(args),
                project_root=project_root,
                duration_ms=int((time.monotonic() - call_started_at) * 1000),
            )

            tool_messages.append(
                ToolMessage(content=output, tool_call_id=call_id, name=name)
            )
            tool_logs.append({
                "call_id": call_id,
                "tool_name": name,
                "args": args,
                "success": success,
            })

        log_event(
            self.config.log_file,
            "node_exited",
            node="tools",
            thread_id=thread_id,
            reason="tool_calls_executed",
            call_count=len(tool_logs),
            success_count=sum(1 for t in tool_logs if t["success"]),
            failure_count=sum(1 for t in tool_logs if not t["success"]),
            duration_ms=int((time.monotonic() - node_started_at) * 1000),
        )

        return {"messages": tool_messages, "tool_logs": tool_logs}


    def route_after_agent(self, state: AgentState) -> Literal["confirmation", "end"]:
        thread_id = _thread_id()

        if state.status in {"cancelled", "error", "max_iterations_reached"}:
            log_event(
                self.config.log_file,
                "route_decided",
                thread_id=thread_id,
                from_node="agent",
                to_node="end",
                reason=f"terminal_status:{state.status}",
            )
            return "end"

        last = state.messages[-1] if state.messages else None
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            log_event(
                self.config.log_file,
                "route_decided",
                thread_id=thread_id,
                from_node="agent",
                to_node="confirmation",
                reason="pending_tool_calls",
                tool_call_count=len(last.tool_calls),
            )
            return "confirmation"

        log_event(
            self.config.log_file,
            "route_decided",
            thread_id=thread_id,
            from_node="agent",
            to_node="end",
            reason="no_tool_calls",
        )
        return "end"


    def _build_graph(self, checkpointer: Optional[Any] = None) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("agent", self.agent_node)
        graph.add_node("confirmation", self.confirmation_node)
        graph.add_node("tools", self.tool_node)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self.route_after_agent,
            {"confirmation": "confirmation", "end": END},
        )
        graph.add_edge("confirmation", "tools")
        graph.add_edge("tools", "agent")

        compiled = graph.compile(checkpointer=checkpointer or InMemorySaver())

        log_event(
            self.config.log_file,
            "agent_graph_built",
            task_mode=self.TASK_MODE,
            nodes=["agent", "confirmation", "tools"],
            using_provided_checkpointer=checkpointer is not None,
        )

        return compiled
