#!/usr/bin/env python3
"""Interactive REPL for the local coding agent.

Keeps a conversation thread alive across turns (so the agent remembers
earlier context in the session), and shows tool calls as they happen.
Confirmation prompts (shell commands, deletes, git push, background jobs,
overwriting files) still go through the normal stdin y/N handler from
agent.confirmation.

Model calls fall back automatically, in order: Groq -> OpenRouter (if
OPENROUTER_API_KEY is set) -> local Ollama Qwen2.5-Coder (if enabled and
`ollama serve` is running with the model pulled). Every tool call,
confirmation decision, and LLM call/fallback is logged to config.log_file
(default: agent_events.log) — use /logs to tail it from here.

Run from the project root (same level as main.py):
    python main.py

Optional, for nicer output:
    pip install rich

Commands:
    /new              start a fresh conversation (new thread id, clears context)
    /model <name>     switch the model for the rest of this session
    /provider [api|local]   switch between the API chain and local-only mode
    /confirm-all [on|off]   require confirmation before EVERY tool call
    /tools            list available tools
    /history          show all tool calls made so far in this session
    /logs             tail the recent entries in the event log file
    /help             show this message
    /exit, /quit      exit the REPL
"""

import argparse
import sys
import uuid
from pathlib import Path

from agent import AgentConfig, run_agent
from agent.logging_utils import log_event
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
  /provider [api|local]   switch between the API chain (Groq->OpenRouter->
                    local fallback) and local-only mode (Ollama only)
  /confirm-all [on|off]   require interactive confirmation before EVERY tool
                    call (not just shell/delete/push/overwrite); no argument
                    shows the current setting
  /tools            list available tools
  /history          show all tool calls made so far in this session
  /logs             tail the recent entries in the event log file
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


def _fallback_summary(config: AgentConfig) -> str:
    if config.provider_mode == "local":
        return f"local-only: Ollama ({config.ollama_model})  [mode: local]"
    tiers = [f"1. Groq ({config.model_name})"]
    if config.openrouter_key_str():
        tiers.append(f"2. OpenRouter ({config.fallback_model_name})")
    if config.enable_ollama_fallback:
        tiers.append(f"3. Local Ollama ({config.ollama_model}) — last resort")
    chain = " → ".join(tiers) if len(tiers) > 1 else f"{tiers[0]}  (no fallback tiers configured)"
    return f"{chain}  [mode: api]"


def _tail_log(path: str, n: int = 25) -> str:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return f"(no log file yet at `{path}` — it's created on the first run)"
    except Exception as e:  # noqa: BLE001
        return f"(couldn't read log file: {e})"
    if not lines:
        return "(log file is empty)"
    tail = lines[-n:]
    return "```\n" + "\n".join(tail) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description="Local coding agent REPL")
    parser.add_argument(
        "--provider", choices=["api", "local"], default=None,
        help="Force provider mode for this session: 'api' (Groq->OpenRouter->local "
             "fallback) or 'local' (Ollama only). Defaults to AGENT_PROVIDER_MODE env "
             "var, or 'api'.",
    )
    parser.add_argument(
        "--confirm-all", action="store_true",
        help="Require interactive confirmation before every tool call, not just "
             "destructive ones (shell/delete/git push/background jobs/file overwrite). "
             "Can also be toggled mid-session with /confirm-all on|off, or set "
             "persistently via AGENT_CONFIRM_ALL_TOOLS=true.",
    )
    args = parser.parse_args()

    try:
        config = AgentConfig.from_env()
    except Exception as e:  # noqa: BLE001
        print(f"Config error: {e}")
        sys.exit(1)

    if args.provider:
        try:
            config = config.model_copy(update={"provider_mode": args.provider})
        except Exception as e:  # noqa: BLE001
            print(f"Invalid --provider value: {e}")
            sys.exit(1)

    if args.confirm_all:
        config = config.model_copy(update={"confirm_all_tools": True})

    if config.provider_mode == "local" and not config.enable_ollama_fallback:
        print(
            "provider_mode is 'local' but Ollama fallback is disabled "
            "(enable_ollama_fallback=False / AGENT_ENABLE_OLLAMA_FALLBACK=false). "
            "Enable it, or drop --provider local to use the API chain."
        )
        sys.exit(1)

    thread_id = str(uuid.uuid4())
    session_tool_log: list = []

    _print_panel(
        "Local Coding Agent",
        f"model chain: {_fallback_summary(config)}\n"
        f"thread: `{thread_id[:8]}`  |  log: `{config.log_file}`  |  "
        f"confirm-all: `{'ON' if config.confirm_all_tools else 'OFF'}`\n\n"
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
                _print(f"current model chain: {_fallback_summary(config)}")
            continue

        if user_input.startswith("/provider"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip().lower() in ("api", "local"):
                new_mode = parts[1].strip().lower()
                if new_mode == "local" and not config.enable_ollama_fallback:
                    _print(
                        "can't switch to local mode: Ollama fallback is disabled "
                        "(enable_ollama_fallback=False).",
                        style="red",
                    )
                    continue
                old_mode = config.provider_mode
                config = config.model_copy(update={"provider_mode": new_mode})
                log_event(
                    config.log_file, "provider_mode_changed", thread_id=thread_id,
                    from_mode=old_mode, to_mode=new_mode,
                )
                _print(f"provider mode switched to: {new_mode}\n{_fallback_summary(config)}")
            else:
                _print(f"current provider mode: {config.provider_mode}\n{_fallback_summary(config)}\n\nusage: /provider api | /provider local")
            continue

        if user_input.startswith("/confirm-all"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip().lower() in ("on", "off"):
                new_val = parts[1].strip().lower() == "on"
                old_val = config.confirm_all_tools
                config = config.model_copy(update={"confirm_all_tools": new_val})
                log_event(
                    config.log_file, "confirm_all_tools_changed", thread_id=thread_id,
                    from_value=old_val, to_value=new_val,
                )
                if new_val:
                    _print("confirm-all: ON — every tool call (including reads/searches) will ask for approval first.")
                else:
                    _print("confirm-all: OFF — only destructive tools (shell/delete/git push/background jobs/file overwrite) need approval.")
            else:
                state = "ON" if config.confirm_all_tools else "OFF"
                _print(f"confirm-all is currently: {state}\n\nusage: /confirm-all on | /confirm-all off")
            continue

        if user_input == "/tools":
            _print_panel("Available tools", _list_tools())
            continue

        if user_input == "/history":
            _print_tool_calls(session_tool_log, title="All tool calls this session")
            continue

        if user_input == "/logs":
            _print_panel(f"Recent events — {config.log_file}", _tail_log(config.log_file))
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
        elif result.status == "completed_with_errors":
            last_failed = next((tc for tc in reversed(result.tool_calls) if not tc.success), None)
            detail = f" last failed call: {last_failed.tool_name} -> {last_failed.result[:150]}" if last_failed else ""
            _print(f"[status: completed, but the last tool call failed — double-check the result.{detail}]", style="yellow")


if __name__ == "__main__":
    main()
    