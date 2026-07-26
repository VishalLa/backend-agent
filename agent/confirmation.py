from .schemas import ConfirmationDecision, ConfirmationRequest

ALWAYS_CONFIRM_TOOLS = {
    "run_shell_command",
    "git_push",
    "delete_path",
    "launch_background_process",
}

CONDITIONAL_CONFIRM_TOOLS = {
    "write_file": lambda args: bool(args.get("overwrite")),
}


def needs_confirmation(tool_name: str, tool_args: dict, confirm_all: bool = False) -> bool:
    """True if this specific tool call requires human approval before running.

    Checked per-call (not just per-tool-name) so conditional gates like
    write_file's overwrite flag only trigger when the destructive argument
    is actually set.

    confirm_all=True (see AgentConfig.confirm_all_tools) overrides everything
    below and requires confirmation for every tool call, regardless of name
    or args — useful when you want to review each action one at a time.
    """
    if confirm_all:
        return True
    if tool_name in ALWAYS_CONFIRM_TOOLS:
        return True
    check = CONDITIONAL_CONFIRM_TOOLS.get(tool_name)
    return bool(check and check(tool_args or {}))


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
    