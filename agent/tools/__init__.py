from .api_tool import fetch_openapi_schema, http_request
from .file_tool import append_file, edit_file, list_dir, read_file, write_file
from .git_tool import git_branch, git_checkout, git_commit, git_diff, git_log, git_push, git_status
from .jupyter_tool import execute_code, restart_kernel
from .search_tool import ripgrep_search, web_search
from .shell_tool import run_shell_command
from .sandbox_tool import execute_in_sandbox
from .system_tool import check_gpu_status, delete_path, launch_background_process, tail_log


BACKEND_TOOLS = [
    run_shell_command,

    read_file,
    write_file,
    append_file,
    edit_file,
    list_dir,
    delete_path,

    ripgrep_search,
    web_search,

    http_request,
    fetch_openapi_schema,
]

ALGO_TOOLS = list(BACKEND_TOOLS)

GIT_TOOLS = [
    git_status,
    git_diff,
    git_log,
    git_branch,
    git_checkout,
    git_commit,
    git_push,
]

ML_TOOLS = [
    read_file,
    write_file,
    append_file,
    edit_file,
    list_dir,

    ripgrep_search,
    web_search,

    execute_code,
    restart_kernel,

    check_gpu_status,
    launch_background_process,
    tail_log,
    delete_path,
]

SANDBOX_TOOLS = [execute_in_sandbox]

TOOLS_BY_TASK = {
    "backend": BACKEND_TOOLS,
    "ml": ML_TOOLS,
    "git": GIT_TOOLS,
    "algorithms": ALGO_TOOLS,
}

__all__ = [
    "BACKEND_TOOLS", "ALGO_TOOLS", "GIT_TOOLS", "ML_TOOLS", "SANDBOX_TOOLS", "TOOLS_BY_TASK",
]
