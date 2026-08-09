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
    
    
