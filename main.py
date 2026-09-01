#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from config import Config
from agent import AgentRunner
from agent.confirmation import (
    confirmation_resume_payload,
    default_cli_confirmation_handler,
)
from schema.agent_schema import ConfirmationRequest


VALID_AGENTS = ("backend", "ml", "git", "algorithms")
console = Console()


def _select_agent(initial: Optional[str]) -> str:
    if initial and initial in VALID_AGENTS:
        return initial
    
    console.print("\n[bold cyan]Available Agents[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Agent", style="cyan")
    table.add_column("Description")
    
    agent_info = {
        "backend": "Flask/FastAPI routes, business logic, integrations",
        "ml": "Training, evaluation, data pipelines, experiments",
        "git": "Inspect, branch, commit, push version-control work",
        "algorithms": "Correctness- and complexity-sensitive implementation",
    }
    
    for agent in VALID_AGENTS:
        table.add_row(agent, agent_info.get(agent, ""))
    
    console.print(table)
    
    while True:
        try:
            choice = Prompt.ask(
                "[bold]Select agent[/bold]",
                choices=list(VALID_AGENTS),
                default="backend"
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[red]bye[/red]")
            sys.exit(0)
        if choice in VALID_AGENTS:
            return choice
        console.print(f"[yellow]Unknown agent '{choice}'. Choose one of: {', '.join(VALID_AGENTS)}[/yellow]")


def _extract_interrupt(result: Any) -> Optional[ConfirmationRequest]:
    """Best-effort extraction of a pending confirmation request from a run
    result - see the interface-assumptions note at the top of this file."""
    state = getattr(result, "state", result)
    interrupts = state.get("__interrupt__") if isinstance(state, dict) else None
    if not interrupts:
        return None

    value = interrupts[0]
    value = getattr(value, "value", value)  # unwrap a langgraph Interrupt object if that's what this is

    if isinstance(value, ConfirmationRequest):
        return value
    if isinstance(value, dict):
        return ConfirmationRequest.model_validate(value)
    return None


def _print_response(result: Any) -> None:
    state = getattr(result, "state", result)
    messages = state.get("messages") if isinstance(state, dict) else None
    if messages:
        content = messages[-1].content
        panel = Panel(
            content,
            title="[bold green]Agent Response[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
        console.print(panel)
    else:
        console.print(
            "[yellow](agent produced no message this turn)[/yellow]"
        )


def run_cli(project_path: Optional[str] = None, initial_agent: Optional[str] = None) -> None:
    config = Config.from_env()
    runner = AgentRunner(config)

    agent_key = _select_agent(initial_agent)
    thread_id: Optional[str] = None

    header_text = Text()
    header_text.append(f"Coding agent ready (")
    header_text.append(agent_key, style="bold cyan")
    header_text.append("). Type ")
    header_text.append("exit", style="bold yellow")
    header_text.append(" or ")
    header_text.append("Ctrl-C", style="bold yellow")
    header_text.append(" to quit.")
    
    panel = Panel(
        header_text,
        border_style="blue",
        padding=(1, 2),
        title="[bold blue]Ready[/bold blue]",
    )
    console.print(panel)

    while True:
        try:
            user_message = Prompt.ask("[bold cyan]you[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[red]bye[/red]")
            return

        if user_message.lower() in ("exit", "quit"):
            console.print("[red]bye[/red]")
            return
        if not user_message:
            continue

        try:
            result = runner.run(
                agent_key=agent_key,
                user_message=user_message,
                thread_id=thread_id,
                project_path=project_path,
            )
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive on a bad turn
            err_panel = Panel(
                f"[red]{exc}[/red]",
                title="[bold red]ERROR: agent run failed[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
            console.print(err_panel)
            continue

        thread_id = getattr(result, "thread_id", thread_id)

        # Drain however many confirmation pauses this turn produces - a
        # single user message can trigger more than one gated tool call in
        # sequence (e.g. delete_path then git_push), each its own interrupt.
        pending = _extract_interrupt(result)
        while pending is not None:
            decision = default_cli_confirmation_handler(pending)
            try:
                result = runner.resume_confirmation(
                    agent_key=agent_key,
                    thread_id=thread_id,
                    decision=confirmation_resume_payload(decision.approved, decision.reason),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"ERROR: resume failed: {exc}")
                break
            pending = _extract_interrupt(result)

        _print_response(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local coding agent CLI")
    parser.add_argument("--agent", choices=VALID_AGENTS, default=None, help="Agent mode to use")
    parser.add_argument("--project", default=None, help="Project root the agent should operate in")
    args = parser.parse_args()

    try:
        run_cli(project_path=args.project, initial_agent=args.agent)
    except KeyboardInterrupt:
        print("\nbye")
        sys.exit(0)


if __name__ == "__main__":
    main()