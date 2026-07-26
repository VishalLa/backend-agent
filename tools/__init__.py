from .api_tool import fetch_openapi_schema, http_request
from .file_tool import edit_file, list_dir, read_file, write_file
from .git_tool import git_branch, git_checkout, git_commit, git_diff, git_log, git_push, git_status
from .jupyter_tool import execute_code, restart_kernel
from .search_tool import ripgrep_search, web_search
from .shell_tool import run_shell_command
from .system_tool import check_gpu_status, delete_path, launch_background_process, tail_log


ALL_TOOLS = [
    # shell 
    run_shell_command,
    # file ops
    read_file,
    write_file,
    edit_file,
    list_dir,
    # git 
    git_status,
    git_diff,
    git_log,
    git_branch,
    git_checkout,
    git_commit,
    git_push,
    # code + web search
    ripgrep_search,
    web_search,
    # persistent python/jupyter kernel
    execute_code,
    restart_kernel,
    # ML/system safety
    check_gpu_status,
    launch_background_process,
    tail_log,
    delete_path,
    # Flask/FastAPI helpers
    http_request,
    fetch_openapi_schema,
]

__all__ = ["ALL_TOOLS"]
