from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

try:
    import pathspec
    _HAS_PATHSPEC = True
except ImportError:
    _HAS_PATHSPEC = False

DEFAULT_IGNORE = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints",
}


def _load_gitignore(root: Path):
    gi_path = root / ".gitignore"
    if _HAS_PATHSPEC and gi_path.exists():
        with open(gi_path) as f:
            return pathspec.PathSpec.from_lines("gitwildmatch", f.readlines())
    return None


@tool
def read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """Read a file's contents with 1-indexed line numbers, for easy reference
    when constructing later edit_file calls.

    Args:
        path: Path to the file to read.
        start_line: Optional 1-indexed line to start from (inclusive).
        end_line: Optional 1-indexed line to end at (inclusive).
    """

    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    except FileNotFoundError:
        return f"ERROR: file not found: {path}"
    except Exception as e:
        return f"ERROR: {e}"

    start = max(0, (start_line - 1) if start_line else 0)
    end = min(len(lines), end_line if end_line else len(lines))
    numbered = [f"{i + 1}\t{lines[i]}" for i in range(start, end)]
    return "".join(numbered) if numbered else "(empty range)"


@tool
def write_file(
    path: str,
    content: str,
    overwrite: bool = False,
) -> str:
    """Create a new file with the given content. Fails if the file already
    exists unless overwrite=True is explicitly passed — use edit_file for
    targeted changes to existing files instead of clobbering them.

    NOTE: calling this with overwrite=True on an existing file requires
    human confirmation (gated at the graph level, see
    agent.confirmation.needs_confirmation) — it will pause and prompt before
    running, same as a delete. Plain creation of a new file (overwrite=False,
    the default) is never gated.

    Args:
        path: Path to the file to create.
        content: Full text content to write.
        overwrite: If True, allow overwriting an existing file. Requires
            human confirmation. Defaults to False.
    """

    p = Path(path)
    if p.exists() and not overwrite:
        return f"ERROR: {path} already exists. Use edit_file to modify it, or pass overwrite=True."

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def append_file(
    path: str,
    content: str,
    create_if_missing: bool = True,
) -> str:
    """Append content to the end of an existing file, creating it first if
    it doesn't exist and create_if_missing=True.

    Use this to build large files (roughly 150+ lines) across several
    smaller tool calls instead of generating the whole thing in one
    write_file completion — a single huge completion is much more likely to
    hit a model's per-request token cap (this is especially tight on
    low-tier free API plans). Typical pattern: write_file for the first
    chunk (imports, first class/section), then one or more append_file
    calls for the rest.

    Args:
        path: Path to the file to append to.
        content: Text to append at the end of the file.
        create_if_missing: If True (default) and the file doesn't exist yet,
            create it with this content instead of erroring.
    """
    p = Path(path)
    if not p.exists():
        if not create_if_missing:
            return f"ERROR: {path} does not exist. Pass create_if_missing=True or use write_file first."
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"OK: created {path} and wrote {len(content)} chars"
        except Exception as e:
            return f"ERROR: {e}"

    try:
        with open(p, "a", encoding="utf-8") as f:
            f.write(content)
        return f"OK: appended {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def edit_file(
    path: str,
    old_str: str,
    new_str: str,
) -> str:
    """Replace an exact, unique string in a file with a new string
    (patch-based edit). old_str must match the file content exactly,
    including whitespace, and must appear exactly once — this avoids
    clobbering unrelated code and keeps diffs reviewable.

    Args:
        path: Path to the file to edit.
        old_str: Exact existing text to replace. Must be unique in the file.
        new_str: Text to replace it with.
    """

    try:
        content = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"ERROR: file not found: {path}"
    except Exception as e:
        return f"ERROR: {e}"

    count = content.count(old_str)
    if count == 0:
        return "ERROR: old_str not found in file"
    if count > 1:
        return f"ERROR: old_str is not unique ({count} occurrences) — include more surrounding context"

    Path(path).write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
    return f"OK: edited {path}"


@tool
def list_dir(path: str = ".", depth: int = 2) -> str:
    """List a directory tree up to a given depth, respecting .gitignore
    rules and skipping common noise directories (.git, __pycache__, venv,
    node_modules, etc).

    Args:
        path: Root directory to list. Defaults to the current directory.
        depth: Max depth to recurse. Defaults to 2.
    """
    root = Path(path)
    if not root.exists():
        return f"ERROR: path not found: {path}"

    spec = _load_gitignore(root)
    lines: list[str] = []

    def _walk(current: Path, prefix: str, level: int) -> None:
        if level > depth:
            return
        try:
            entries = sorted(current.iterdir(), key=lambda e: (e.is_file(), e.name))
        except PermissionError:
            return
        for entry in entries:
            if entry.name in DEFAULT_IGNORE:
                continue
            rel = str(entry.relative_to(root))
            match_target = rel + "/" if entry.is_dir() else rel
            if spec and spec.match_file(match_target):
                continue
            marker = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{entry.name}{marker}")
            if entry.is_dir() and level < depth:
                _walk(entry, prefix + "  ", level + 1)

    _walk(root, "", 1)
    return "\n".join(lines) if lines else "(empty)"
