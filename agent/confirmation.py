from .schemas import ConfirmationDecision, ConfirmationRequest

ALWAYS_CONFIRM_TOOLS = {
    "run_shell_command",
    "git_push",
    "delete_path",
    "launch_background_process",
}


def default_cli_confirmation_handler(request: ConfirmationRequest) -> ConfirmationDecision:
    """Blocking CLI confirmation prompt.

    Edge cases handled:
    - Non-interactive session (no stdin, e.g. piped/CI run): EOFError is
      caught and the action is auto-declined rather than hanging forever.
    - Ctrl-C while the prompt is open: treated as a decline, not a crash.
    """
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
        return ConfirmationDecision(approved=False, reason="cancelled by user (Ctrl-C)")

    if answer in ("y", "yes"):
        return ConfirmationDecision(approved=True)
    return ConfirmationDecision(approved=False, reason="declined by user")
    