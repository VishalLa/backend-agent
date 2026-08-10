from __future__ import annotations

from typing import Any, Mapping, Optional

from schema.agent_schema import ConfirmationDecision, ConfirmationRequest

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


def render_streamlit_confirmation(
    request: ConfirmationRequest,
    *,
    key: Optional[str] = None,
) -> Optional[ConfirmationDecision]:
    """Render an approval control for a future Streamlit dashboard.

    Returns ``None`` until the user chooses an action. The dashboard resumes
    the graph with ``Command(resume=confirmation_resume_payload(...))``.
    Streamlit is imported lazily so agent and worker processes do not need the
    dashboard dependency installed.
    """
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Streamlit is required to render the dashboard confirmation control") from exc

    widget_key = key or f"confirmation-{request.call_id}"
    st.warning(f"Confirmation required: `{request.tool_name}`")
    st.caption(request.reason)
    st.json(confirmation_request_payload(request))
    reason = st.text_input("Reason (optional)", key=f"{widget_key}-reason")
    approve_column, deny_column = st.columns(2)
    if approve_column.button("Approve", key=f"{widget_key}-approve", type="primary"):
        return ConfirmationDecision(approved=True, reason=reason or None)
    if deny_column.button("Deny", key=f"{widget_key}-deny"):
        return ConfirmationDecision(approved=False, reason=reason or None)
    return None


def default_cli_confirmation_handler(
    request: ConfirmationRequest
) -> ConfirmationDecision:
    print("\n--- CONFIRMATION REQUIRED ---")
    print(f"Tool:   {request.tool_name}")
    print(f"Args:   {request.tool_args}")
    print(f"Reason: {request.reason}")

    try: 
        answer = input("Allow this? [y/N]: ").strip().lower()
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
        return ConfirmationDecision(approved=True)
    return ConfirmationDecision(
        approved=False, 
        reason="declined by user"
    )
    
