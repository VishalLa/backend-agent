TASK_PROFILES = {
    "backend": {
        "model_field": "backend_model_name",
        "description": "Flask/FastAPI backend development",
        "tool_names": {
            "run_shell_command",
            "read_file",
            "write_file",
            "append_file",
            "edit_file",
            "list_dir",
            "ripgrep_search",
            "web_search",
            "http_request",
            "fetch_openapi_schema",
            "delete_path"
        }
    },
    "ml": {
        "model_field": "ml_model_name",
        "description": "ML/AI and data-analysis workflow",
        "tool_names": {
            "read_file",
            "write_file",
            "append_file",
            "edit_file",
            "list_dir",
            "ripgrep_search",
            "web_search",
            "execute_code",
            "restart_kernel",
            "check_gpu_status",
            "launch_background_process",
            "tail_log",
            "delete_path",
        }
    },
    "git": {
        "model_field": "git_model_name",
        "description": "git workflow",
        "tool_names": {
            "git_status",
            "git_diff",
            "git_log",
            "git_branch",
            "git_checkout",
            "git_commit",
            "git_push",
        }
    },
    "algorithms": {
        "model_field": "algo_model_name",
        "description": "complex algo work flow",
        "tool_names": {
            "run_shell_command",
            "read_file",
            "write_file",
            "append_file",
            "edit_file",
            "list_dir",
            "ripgrep_search",
            "web_search",
            "http_request",
            "fetch_openapi_schema",
            "delete_path"
        }
    }
}


def filter_tools_for_task(task_mode: str, all_tools: list) -> list:
    profile = TASK_PROFILES.get(task_mode)
    if profile is None:
        return all_tools
    names = profile["tool_names"]
    filtered = [t for t in all_tools if t.name in names]
    return filtered if filtered else all_tools
