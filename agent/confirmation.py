from __future__ import annotations

from typing import Any, Mapping, Optional
import json

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.text import Text

from schema.agent_schema import ConfirmationDecision, ConfirmationRequest

console = Console()

ALWAYS_CONFIRM_TOOLS = {
    "run_shell_command",
    "git_push",
    "delete_path",
    "launch_background_process",
}

CONDITIONAL_CONFIRM_TOOLS = {
    "write_file": lambda args: bool(args.get("overwrite")),
}

def needs_confirmation(
    tool_name: str,
    tool_args: dict,
    confirm_all: bool = False
) -> bool:
    if confirm_all:
        return True

    if tool_name in ALWAYS_CONFIRM_TOOLS:
        return True    

    check = CONDITIONAL_CONFIRM_TOOLS.get(tool_name)
    return bool(check and check(tool_args or {}))


def confirmation_request_payload(request: ConfirmationRequest) -> dict[str, Any]:
    """Return the JSON-safe interrupt payload consumed by a Streamlit UI."""
    return {
        "type": "confirmation_required",
        **request.model_dump(mode="json"),
    }


def parse_confirmation_decision(value: Any) -> ConfirmationDecision:
    """Validate a value supplied through ``Command(resume=...)``."""
    if isinstance(value, ConfirmationDecision):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("Confirmation resume value must be an object with an 'approved' boolean")
    return ConfirmationDecision.model_validate(value)


def confirmation_resume_payload(
    approved: bool,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    """Build the exact JSON-safe payload to pass to ``Command(resume=...)``."""
    return ConfirmationDecision(approved=approved, reason=reason).model_dump(mode="json")


# def render_streamlit_confirmation(
#     request: ConfirmationRequest,
#     *,
#     key: Optional[str] = None,
# ) -> Optional[ConfirmationDecision]:
#     """Render an approval control for a future Streamlit dashboard.

#     Returns ``None`` until the user chooses an action. The dashboard resumes
#     the graph with ``Command(resume=confirmation_resume_payload(...))``.
#     Streamlit is imported lazily so agent and worker processes do not need the
#     dashboard dependency installed.
#     """
#     try:
#         import streamlit as st
#     except ImportError as exc:
#         raise RuntimeError("Streamlit is required to render the dashboard confirmation control") from exc

#     widget_key = key or f"confirmation-{request.call_id}"
#     st.warning(f"Confirmation required: `{request.tool_name}`")
#     st.caption(request.reason)
#     st.json(confirmation_request_payload(request))
#     reason = st.text_input("Reason (optional)", key=f"{widget_key}-reason")
#     approve_column, deny_column = st.columns(2)
#     if approve_column.button("Approve", key=f"{widget_key}-approve", type="primary"):
#         return ConfirmationDecision(approved=True, reason=reason or None)
#     if deny_column.button("Deny", key=f"{widget_key}-deny"):
#         return ConfirmationDecision(approved=False, reason=reason or None)
#     return None


def default_cli_confirmation_handler(
    request: ConfirmationRequest
) -> ConfirmationDecision:
    # Build content panel with tool details
    content = Text()
    content.append("Tool: ", style="bold")
    content.append(f"{request.tool_name}\n", style="cyan")
    content.append("\nReason: ", style="bold")
    content.append(f"{request.reason}\n\n", style="yellow")
    content.append("Arguments:\n", style="bold")
    
    # Display tool info panel
    console.print(
        Panel(
            content,
            title="[bold yellow]⚠️  CONFIRMATION REQUIRED[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )
    
    # Format and display arguments as JSON
    try:
        args_json = json.dumps(request.tool_args, indent=2)
        syntax = Syntax(args_json, "json", theme="monokai", line_numbers=False)
        console.print(syntax)
    except Exception:
        # Fallback if JSON formatting fails
        console.print(f"[yellow]{request.tool_args}[/yellow]")

    try:
        answer = Prompt.ask(
            "[bold red]Allow this?[/bold red]",
            choices=["y", "n"],
            default="n",
        ).strip().lower()
    except EOFError:
        return ConfirmationDecision(
            approved=False,
            reason="No interactive input available (non-interactive session) — auto-declined for safety.",
        )
    except KeyboardInterrupt:
        return ConfirmationDecision(
            approved=False,
            reason="cancelled by user (Ctrl-C)"
        )
    if answer in ("y", "yes"):
        console.print("[green]✓ Approved[/green]")
        return ConfirmationDecision(approved=True)
    console.print("[red]✗ Denied[/red]")
    return ConfirmationDecision(
        approved=False, 
        reason="declined by user"
    )
    
