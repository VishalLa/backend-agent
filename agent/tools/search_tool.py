from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import tool

from agent.codebase_index import CodebaseIndex

try:
    from langchain_community.tools import DuckDuckGoSearchRun

    web_search = DuckDuckGoSearchRun()
    web_search.name = "web_search"
    web_search.description = (
        "Search the web for current docs, errors, or API references. "
        "Prefer over training data for fast-moving libraries. Requires `ddgs`."
    )
except ImportError:

    @tool
    def web_search(query: str) -> str:
        """Search the web. NOT CONFIGURED: install `ddgs` + `langchain-community`.

        Args:
            query: Search query.
        """
        return "ERROR: web_search is not configured. Install ddgs + langchain-community."


@tool
def ripgrep_search(
    pattern: str,
    path: str = ".",
    file_type: str = None,
    case_insensitive: bool = False,
    max_results: int = 100,
) -> str:
    """Fast regex/text search via ripgrep. Prefer over reading whole files
    when looking for a symbol/function/import/string.

    Args:
        pattern: Regex or plain-text pattern.
        path: File/dir to search. Default current dir.
        file_type: Optional file-type filter (e.g. "py").
        case_insensitive: Ignore case.
        max_results: Cap on matching lines returned.
    """
    cmd = ["rg", "--line-number", "--no-heading"]
    if case_insensitive:
        cmd.append("-i")
    if file_type:
        cmd += ["-t", file_type]
    cmd += [pattern, path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return "ERROR: ripgrep (rg) is not installed."
    except subprocess.TimeoutExpired:
        return "ERROR: search timed out"

    if result.returncode not in (0, 1):  # 1 = no matches, still a valid run
        return f"ERROR: {result.stderr.strip()}"

    lines = result.stdout.splitlines()
    if not lines:
        return "(no matches)"

    truncated = lines[:max_results]
    out = "\n".join(truncated)
    if len(lines) > max_results:
        out += f"\n... ({len(lines) - max_results} more matches truncated)"
    return out


@tool
def search_codebase(
    query: str,
    path: str = ".",
    max_results: int = 5,
    force_reindex: bool = False,
) -> str:
    """Retrieve relevant source-code chunks from a local project index.

    Uses a persistent local BM25 index in the system temporary directory and
    returns exact file/line citations. Use this first for questions spanning multiple files,
    then use read_file to verify the cited source.

    Args:
        query: Natural-language question, symbol, error, or implementation topic.
        path: Project root to index. Defaults to the current project root.
        max_results: Number of relevant chunks to return (1-10).
        force_reindex: Rebuild even if the source tree is unchanged.
    """
    if not 1 <= max_results <= 10:
        return "ERROR: max_results must be between 1 and 10."
    try:
        index = CodebaseIndex(Path(path))
        if force_reindex:
            index.build(force=True)
        return index.format_results(query, limit=max_results)
    except (OSError, ValueError) as exc:
        return f"ERROR: could not search codebase: {exc}"
