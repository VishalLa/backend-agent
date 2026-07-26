#!/usr/bin/env python3
"""Interactive REPL for the local coding agent.

Keeps a conversation thread alive across turns (so the agent remembers
earlier context in the session), and shows tool calls as they happen.
Confirmation prompts (shell commands, deletes, git push, background jobs)
still go through the normal stdin y/N handler from agent.confirmation.

Run from the project root (same level as main.py):
    python interactive_cli.py

Optional, for nicer output:
    pip install rich

Commands:
    /new              start a fresh conversation (new thread id, clears context)
    /model <name>     switch the model for the rest of this session
    /tools            list available tools
    /history          show all tool calls made so far in this session
    /help             show this message
    /exit, /quit      exit the REPL
"""

import sys
import uuid

from agent import AgentConfig, run_agent
from tools import ALL_TOOLS

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table

    _console = Console()
    _RICH = True
except ImportError:
    _console = None
    _RICH = False

HELP_TEXT = """
Commands:
  /new              start a fresh conversation (new thread id, clears context)
  /model <name>     switch the model for the rest of this session
  /tools            list available tools
  /history          show all tool calls made so far in this session
  /help             show this message
  /exit, /quit      exit the REPL
"""


def _print(text: str, style: str = None) -> None:
    if _RICH:
        _console.print(text, style=style)
    else:
        print(text)


def _print_panel(title: str, body: str, style: str = "cyan") -> None:
    body = body if body and body.strip() else "(empty)"
    if _RICH:
        _console.print(Panel(Markdown(body), title=title, border_style=style))
    else:
        print(f"\n--- {title} ---\n{body}\n")


def _print_tool_calls(tool_calls, title: str = "Tool calls this turn") -> None:
    if not tool_calls:
        return
    if _RICH:
        table = Table(title=title, show_lines=False)
        table.add_column("Status", width=6)
        table.add_column("Tool")
        table.add_column("Args", overflow="fold")
        table.add_column("Result", overflow="fold")
        for log in tool_calls:
            status = "[green]OK[/]" if log.success else "[red]FAIL[/]"
            table.add_row(status, log.tool_name, str(log.args)[:60], log.result[:100].replace("\n", " "))
        _console.print(table)
    else:
        print(f"\n-- {title} --")
        for log in tool_calls:
            flag = "OK" if log.success else "FAIL"
            print(f"[{flag}] {log.tool_name}({log.args}) -> {log.result[:150]}")


def _list_tools() -> str:
    lines = []
    for t in ALL_TOOLS:
        first_line = (t.description or "").strip().splitlines()[0] if t.description else ""
        lines.append(f"- **{t.name}** — {first_line}")
    return "\n".join(lines)


def main() -> None:
    try:
        config = AgentConfig.from_env()
    except Exception as e:  # noqa: BLE001
        print(f"Config error: {e}")
        sys.exit(1)

    thread_id = str(uuid.uuid4())
    session_tool_log: list = []

    _print_panel(
        "Local Coding Agent",
        f"model: **{config.model_name}**  |  thread: `{thread_id[:8]}`\n\n"
        "Type `/help` for commands, or just describe a task.",
    )

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not user_input:
            continue

        if user_input in ("/exit", "/quit"):
            print("bye.")
            break

        if user_input == "/help":
            print(HELP_TEXT)
            continue

        if user_input == "/new":
            thread_id = str(uuid.uuid4())
            session_tool_log = []
            _print(f"started a new conversation: {thread_id[:8]}")
            continue

        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                try:
                    config = config.model_copy(update={"model_name": parts[1].strip()})
                    _print(f"model switched to: {config.model_name}")
                except Exception as e:  # noqa: BLE001
                    _print(f"couldn't switch model: {e}", style="red")
            else:
                _print(f"current model: {config.model_name}")
            continue

        if user_input == "/tools":
            _print_panel("Available tools", _list_tools())
            continue

        if user_input == "/history":
            _print_tool_calls(session_tool_log, title="All tool calls this session")
            continue

        if user_input.startswith("/"):
            _print(f"unknown command: {user_input}  (try /help)", style="yellow")
            continue

        try:
            result = run_agent(user_input, config=config, thread_id=thread_id, tools=ALL_TOOLS)
        except Exception as e:  # noqa: BLE001 - last-resort net so a bad turn doesn't kill the REPL
            _print(f"agent run crashed: {e}", style="red")
            continue

        _print_panel("agent", result.output)
        _print_tool_calls(result.tool_calls)
        session_tool_log.extend(result.tool_calls)

        if result.status == "error":
            _print(f"[status: error] {result.error}", style="red")
        elif result.status == "max_iterations_reached":
            _print(f"[status: stopped — hit max_iterations ({config.max_iterations})]", style="yellow")


if __name__ == "__main__":
    main()
    