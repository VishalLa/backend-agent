import subprocess
from langchain_core.tools import tool


def _run_git(args: list, cwd: str = None) -> str:
    result = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    out, err = result.stdout.strip(), result.stderr.strip()

    if result.returncode != 0:
        return f"ERROR (exit {result.returncode}: {err or out})"
    combined = "\n".join(part for part in (out, err) if part)
    return combined or "(no output)"


@tool
def git_status(cwd: str = None) -> str:
    """Show staged, unstaged, and untracked files.

    Args:
        cwd: Repo directory. Default current dir.
    """
    return _run_git(["status", "--short", "--branch"], cwd)


@tool
def git_diff(
    staged: bool = False, 
    path: str = None, 
    cwd: str = None
) -> str:
    """Show diff between working tree/index/HEAD.

    Args:
        staged: Show staged changes instead of unstaged.
        path: Optional file/dir to limit diff to.
        cwd: Repo directory.
    """
    args = ["diff"]
    if staged:
        args.append("--cached")
    if path:
        args.append(path)
    return _run_git(args, cwd)



@tool
def git_log(n: int = 10, cwd: str = None) -> str:
    """Show recent commits, one line each.

    Args:
        n: Number of commits. Default 10.
        cwd: Repo directory.
    """

    return _run_git(["log", f"-{n}", "--oneline", "--decorate"], cwd)


@tool
def git_branch(create: str = None, cwd: str = None) -> str:
    """List branches, or create+switch to a new one if create= is given.

    Args:
        create: New branch name to create and switch to (e.g. "agent/task").
        cwd: Repo directory.
    """

    if create:
        return _run_git(["checkout", "-b", create], cwd)
    return _run_git(["branch", "-vv"], cwd)


@tool
def git_checkout(ref: str, cwd: str = None) -> str:
    """Switch branches or restore files to a given ref.

    Args:
        ref: Branch name, commit hash, or ref.
        cwd: Repo directory.
    """

    return _run_git(["checkout", ref], cwd)


@tool
def git_commit(message: str, add_all: bool = True, cwd: str = None) -> str:
    """Stage and commit changes (checkpoint, reversible via git reset/revert).

    Args:
        message: Commit message.
        add_all: Stage all changes first (git add -A). Default True.
        cwd: Repo directory.
    """

    if add_all:
        add_result = _run_git(["add", "-A"], cwd)
        if add_result.startswith("ERROR"):
            return add_result
    return _run_git(["commit", "-m", message], cwd)


@tool
def git_push(
    remote: str = "origin", 
    branch: str = None, 
    confirm: bool = False, 
    cwd: str = None
) -> str:
    """Push to a remote. REQUIRES confirm=True (only after the user
    explicitly agrees) — hard to undo cleanly.

    Args:
        remote: Remote name. Default "origin".
        branch: Branch to push. Default current branch.
        confirm: Must be True to actually push. Default False.
        cwd: Repo directory.
    """
    
    if not confirm:
        return "BLOCKED: git_push requires explicit confirm=True. Confirm with the user before proceeding."
    args = ["push", remote]
    if branch:
        args.append(branch)
    return _run_git(args, cwd)
