import subprocess

from langchain_core.tools import tool


def _run_git(args: list[str], cwd: str = None) -> str:
    result = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    out, err = result.stdout.strip(), result.stderr.strip()
    if result.returncode != 0:
        return f"ERROR (exit {result.returncode}): {err or out}"
    combined = "\n".join(part for part in (out, err) if part)
    return combined or "(no output)"


@tool
def git_status(cwd: str = None) -> str:
    """Show the working tree status: staged, unstaged, and untracked files.

    Args:
        cwd: Repository directory. Defaults to the current directory.
    """
    return _run_git(["status", "--short", "--branch"], cwd)


@tool
def git_diff(
    staged: bool = False, 
    path: str = None, 
    cwd: str = None
) -> str:
    """Show changes between the working tree, index, and HEAD.

    Args:
        staged: If True, show staged (index) changes instead of unstaged.
        path: Optional file or directory to limit the diff to.
        cwd: Repository directory.
    """
    args = ["diff"]
    if staged:
        args.append("--cached")
    if path:
        args.append(path)
    return _run_git(args, cwd)


@tool
def git_log(n: int = 10, cwd: str = None) -> str:
    """Show recent commit history, one line per commit.

    Args:
        n: Number of commits to show. Defaults to 10.
        cwd: Repository directory.
    """

    return _run_git(["log", f"-{n}", "--oneline", "--decorate"], cwd)


@tool
def git_branch(create: str = None, cwd: str = None) -> str:
    """List branches with tracking info, or create and switch to a new one.

    Args:
        create: If given (e.g. "agent/task-name"), create and switch to a
            new branch with this name instead of listing branches. Use a
            dedicated branch per agent session so changes can be reviewed
            before merging to main.
        cwd: Repository directory.
    """

    if create:
        return _run_git(["checkout", "-b", create], cwd)
    return _run_git(["branch", "-vv"], cwd)


@tool
def git_checkout(ref: str, cwd: str = None) -> str:
    """Switch branches or restore working tree files to a given ref.

    Args:
        ref: Branch name, commit hash, or ref to check out.
        cwd: Repository directory.
    """

    return _run_git(["checkout", ref], cwd)


@tool
def git_commit(message: str, add_all: bool = True, cwd: str = None) -> str:
    """Stage and commit changes. Use this to create a checkpoint before or
    after making edits, so any change is reversible with git reset/revert.

    Args:
        message: Commit message.
        add_all: If True (default), stage all changes (git add -A) before committing.
        cwd: Repository directory.
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
    """Push commits to a remote. This is hard to undo cleanly, so it is on
    the confirm-before-run list: it REQUIRES confirm=True to actually run.
    Never set confirm=True on your own initiative — only after the user has
    explicitly agreed to the push.

    Args:
        remote: Remote name. Defaults to "origin".
        branch: Branch to push. Defaults to the current branch.
        confirm: Must be explicitly True to actually push. Defaults to False.
        cwd: Repository directory.
    """
    
    if not confirm:
        return "BLOCKED: git_push requires explicit confirm=True. Confirm with the user before proceeding."
    args = ["push", remote]
    if branch:
        args.append(branch)
    return _run_git(args, cwd)
    