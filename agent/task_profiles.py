TASK_PROFILES = {
    "backend": {
        "model_field": "backend_model_name",
        "description": "Flask/FastAPI backend development",
        "tool_names": {
            "run_shell_command",
            "read_file", "write_file", "append_file", "edit_file", "list_dir",
            "git_status", "git_diff", "git_log", "git_branch", "git_checkout", "git_commit", "git_push",
            "ripgrep_search", "web_search",
            "http_request", "fetch_openapi_schema",
            "delete_path",
        },
    },
    "ml": {
        "model_field": "ml_model_name",
        "description": "ML/AI and data-analysis workflows",
        "tool_names": {
            "run_shell_command",
            "read_file", "write_file", "append_file", "edit_file", "list_dir",
            "git_status", "git_diff", "git_log", "git_branch", "git_checkout", "git_commit", "git_push",
            "ripgrep_search", "web_search",
            "execute_code", "restart_kernel",
            "check_gpu_status", "launch_background_process", "tail_log",
            "delete_path",
        },
    },
}


def filter_tools_for_task(task_mode: str, all_tools: list) -> list:
    """Return the subset of all_tools relevant to task_mode. Falls back to
    the full toolset if task_mode is unrecognized or filtering would leave
    nothing usable — a task profile narrowing tools to zero is always a
    configuration bug, not something that should silently break the agent.
    """
    profile = TASK_PROFILES.get(task_mode)
    if profile is None:
        return all_tools
    names = profile["tool_names"]
    filtered = [t for t in all_tools if t.name in names]
    return filtered if filtered else all_tools
