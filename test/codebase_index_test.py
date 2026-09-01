from __future__ import annotations

from agent.codebase_index import CodebaseIndex
from agent.graphs.helper import resolve_tool_path_args
from agent.tools.search_tool import search_codebase


def test_codebase_index_returns_ranked_line_citations(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def reset_password(token: str) -> bool:\n    return bool(token)\n",
        encoding="utf-8",
    )
    (tmp_path / "other.py").write_text("def unrelated():\n    return 1\n", encoding="utf-8")

    index = CodebaseIndex(tmp_path)
    assert index.build() is True
    result = index.format_results("password reset token")

    assert "auth.py:1-2" in result
    assert "reset_password" in result
    assert index.build() is False


def test_codebase_index_ignores_generated_directories(tmp_path):
    (tmp_path / "main.py").write_text("def application_entrypoint(): pass\n", encoding="utf-8")
    ignored = tmp_path / "node_modules"
    ignored.mkdir()
    (ignored / "secret.js").write_text("application_entrypoint fake\n", encoding="utf-8")

    result = CodebaseIndex(tmp_path).format_results("application entrypoint")

    assert "main.py" in result
    assert "secret.js" not in result


def test_codebase_index_respects_gitignore_and_skips_dotenv(tmp_path):
    (tmp_path / ".gitignore").write_text("private.py\n", encoding="utf-8")
    (tmp_path / "private.py").write_text("def private_token(): pass\n", encoding="utf-8")
    (tmp_path / ".env").write_text("API_TOKEN=super-secret\n", encoding="utf-8")
    (tmp_path / "public.py").write_text("def public_token(): pass\n", encoding="utf-8")

    result = CodebaseIndex(tmp_path).format_results("token")

    assert "public.py" in result
    assert "private.py" not in result
    assert "super-secret" not in result


def test_search_tool_and_project_root_path_resolution(tmp_path):
    (tmp_path / "routes.py").write_text("def health_route(): return 'ok'\n", encoding="utf-8")

    resolved = resolve_tool_path_args(
        "search_codebase", {"path": ".", "query": "health route"}, str(tmp_path)
    )
    output = search_codebase.invoke({"query": "health route", **resolved})

    assert resolved["path"] == str(tmp_path)
    assert "routes.py" in output
