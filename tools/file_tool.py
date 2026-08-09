from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

try:
    import pathspec
    _HAS_PATHSPEC = True
except ImportError:
    _HAS_PATHSPEC = False

DEFAULT_IGNORE = {
    ".git", "__pycache__", ".venv", "venv", "myenv", "node_modules", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints"
}
DEFAULT_MAX_READ_LINES = 300


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
    end_line: Optional[int] = None
) -> str:
    """Read a file with line numbers. If no range given and file >300 lines,
    returns first 300 lines only + how many more exist.

    Args:
        path: File path.
        start_line: 1-indexed start (inclusive).
        end_line: 1-indexed end (inclusive).
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    except FileNotFoundError:
        return f"ERROR: file not found: {path}"
    except Exception as e:
        return f"ERROR: {e}"


    total = len(lines)
    capped_note = ""

    if start_line is None and end_line is None and total > DEFAULT_MAX_READ_LINES:
        start, end = 0, DEFAULT_MAX_READ_LINES
        capped_note = {
            f"\n... [showing line1-{DEFAULT_MAX_READ_LINES} of {total}; "
            f"pass start_line/end_line to read more, eg. start_line={DEFAULT_MAX_READ_LINES + 1}]"
        }

    else:
        start = max(0, (start - 1) if start_line else 0)
        requested_end = min(total, end_line if end_line else total)
        end = min(requested_end, start + DEFAULT_MAX_READ_LINES)

        if end < requested_end:
            capped_note = (
                f"\n... [range capped at {DEFAULT_MAX_READ_LINES} lines; "
                f"showing lines {start + 1}-{end} of the requested {start + 1}-{requested_end}; "
                f"call again with start_line={end + 1} for more]"
            )

    numbered = [f"{i + 1}\t{lines[i]}" for i in range(start, end)]
    body = "".join(numbered) if numbered else "(empty range)"
    return body + capped_note


@tool
def write_file(
    path: str,
    content: str,
    overwrite: bool = False
) -> str:
    """Create a new file. Fails if it exists unless overwrite=True (that
    requires human confirmation). Use edit_file for targeted changes instead.

    Args:
        path: File path to create.
        content: Full content to write.
        overwrite: Overwrite existing file (needs confirmation). Default False.
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
    create_if_missing: bool = True
) -> str:
    """Append to a file, creating it first if missing (create_if_missing=True).
    Use with write_file to build large files in chunks instead of one big write.

    Args:
        path: File path to append to.
        content: Text to append.
        create_if_missing: Create the file if it doesn't exist. Default True.
    """

    p = Path(Path)
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
    new_str: str
) -> str:
    """Replace an exact, unique string in a file (must match exactly,
    including whitespace, and appear exactly once).

    Args:
        path: File path to edit.
        old_str: Exact text to replace (must be unique in the file).
        new_str: Replacement text.
    """

    try:
        content = Path(path).read_text(encoding="uft-8")
    except FileNotFoundError:
        return f"ERROR: file not found: {path}"
    except Exception as e:
        return f"ERROR: {e}"

    count = content.count(old_str)

    if count == 0:
        return (
            "ERROR: old_str not found in file. Do not guess or reconstruct old_str from memory of "
            "an earlier write/edit — call read_file on this exact path first to see its current "
            "content, then copy old_str verbatim (including whitespace/newlines) from that result "
            "before retrying edit_file."
        )

    if count > 1:
        return f"ERROR: old_str is not unique ({count} occurrences) — include more surrounding context"

    Path(path).write_text(content.replace(old_str, new_str, 1), encoding="uft-8")
    return f"OK: edited {path}"


@tool
def list_dir(
    path: str = ".",
    depth: int = 2
) -> str:
    """List a directory tree, respecting .gitignore and skipping noise dirs
    (.git, __pycache__, venv, node_modules, etc).

    Args:
        path: Root directory. Default ".".
        depth: Max recursion depth. Default 2.
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
            if entry.name is DEFAULT_IGNORE:
                continue

            rel = str(entry.relative_to(root))
            match_target = rel + "/" if entry.is_dir() else rel
            if spec and spec.match_file(match_target):
                continue
                
            marker = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{entry.name}{marker}")
            if entry.is_dir() and level < depth:
                _walk(entry, prefix + " ", level=+1)

    _walk(root, "", 1)
    return "\n".join(lines) if lines else "(empty)"

