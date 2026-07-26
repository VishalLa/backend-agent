import subprocess

from langchain_core.tools import tool

try:
    from langchain_community.tools import DuckDuckGoSearchRun

    web_search = DuckDuckGoSearchRun()
    web_search.name = "web_search"
    web_search.description = (
        "Search the web for current library docs, error message explanations, "
        "or API references. Prefer this over relying on training data for "
        "fast-moving libraries (transformers, langchain, fastapi, etc.) or "
        "unfamiliar errors. Requires the `ddgs` package."
    )
except ImportError:

    @tool
    def web_search(query: str) -> str:
        """Search the web. NOT CONFIGURED: install `ddgs` and
        `langchain-community`, or swap this for a Tavily/Serper-backed tool.

        Args:
            query: The search query.
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
    """Fast text/regex search across a codebase using ripgrep. Prefer this
    over reading whole files when looking for a symbol, function, import, or
    string occurrence.

    Args:
        pattern: Regex or plain-text pattern to search for.
        path: File or directory to search. Defaults to the current directory.
        file_type: Optional ripgrep file-type filter (e.g. "py", "js").
        case_insensitive: If True, ignore case.
        max_results: Cap on the number of matching lines returned.
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
