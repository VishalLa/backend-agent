"""Tests for the project-root path guard (agent/graphs/helper.py).

The guard's job: every tool call a model makes must run inside the project
root. These tests cover the case the original guard missed: the model simply
leaves out `cwd` / `path`, so there is nothing to check and the tool falls back
to the directory the agent process was launched from.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent.graphs.helper import ProjectRootViolation, resolve_tool_path_args
from agent.tools.git_tool import git_commit, git_status

# Tools that take a `cwd` and run a subprocess there.
CWD_TOOLS = [
    "run_shell_command",
    "git_status",
    "git_diff",
    "git_log",
    "git_branch",
    "git_checkout",
    "git_commit",
    "git_push",
    "launch_background_process",
]

# Tools whose `path` defaults to "." (the launch directory) when omitted.
PATH_DEFAULTS_TO_DOT_TOOLS = ["list_dir", "ripgrep_search", "search_codebase"]

# Tools where a missing `path` must stay missing so the tool itself rejects it.
PATH_REQUIRED_TOOLS = ["read_file", "write_file", "append_file", "edit_file", "delete_path"]


@pytest.mark.parametrize("tool_name", CWD_TOOLS)
def test_missing_cwd_is_filled_with_project_root(tool_name, tmp_path):
    resolved = resolve_tool_path_args(tool_name, {}, str(tmp_path))

    assert resolved["cwd"] == str(tmp_path.resolve())


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_blank_cwd_is_treated_as_missing(blank, tmp_path):
    resolved = resolve_tool_path_args("git_status", {"cwd": blank}, str(tmp_path))

    assert resolved["cwd"] == str(tmp_path.resolve())


@pytest.mark.parametrize("tool_name", PATH_DEFAULTS_TO_DOT_TOOLS)
def test_missing_path_is_filled_for_tools_that_default_to_dot(tool_name, tmp_path):
    resolved = resolve_tool_path_args(tool_name, {"query": "x"}, str(tmp_path))

    assert resolved["path"] == str(tmp_path.resolve())


@pytest.mark.parametrize("tool_name", PATH_REQUIRED_TOOLS)
def test_missing_path_is_not_invented_for_file_tools(tool_name, tmp_path):
    """A read/write/delete with no path must fail on the tool's own validation,
    never silently act on the project root itself."""
    resolved = resolve_tool_path_args(tool_name, {}, str(tmp_path))

    assert "path" not in resolved


def test_explicit_relative_cwd_still_resolves_inside_root(tmp_path):
    resolved = resolve_tool_path_args("git_status", {"cwd": "sub"}, str(tmp_path))

    assert resolved["cwd"] == str((tmp_path / "sub").resolve())


@pytest.mark.parametrize("outside", ["..", "../sibling", "/etc"])
def test_explicit_paths_outside_root_are_still_rejected(outside, tmp_path):
    with pytest.raises(ProjectRootViolation):
        resolve_tool_path_args("git_status", {"cwd": outside}, str(tmp_path))


def test_no_project_root_leaves_args_untouched():
    args = {"message": "x"}

    assert resolve_tool_path_args("git_commit", args, None) == args


# --- end to end: the exact failure from the logs ---------------------------


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout


@pytest.fixture
def project_and_launch_dir(tmp_path, monkeypatch):
    """Two real git repos. `project` is where the agent should work;
    `launch` is the directory the agent process was started from (in the real
    incident, the agent's own repo)."""
    for key, value in {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }.items():
        monkeypatch.setenv(key, value)

    project = tmp_path / "project"
    launch = tmp_path / "launch"
    for repo in (project, launch):
        repo.mkdir()
        _git(repo, "init", "-q")
        (repo / "seed.txt").write_text("seed", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "seed")

    (project / "only_in_project.txt").write_text("hi", encoding="utf-8")
    (launch / "only_in_launch_dir.txt").write_text("hi", encoding="utf-8")

    monkeypatch.chdir(launch)
    return project, launch


def test_git_status_without_cwd_inspects_project_not_launch_dir(project_and_launch_dir):
    project, _launch = project_and_launch_dir

    args = resolve_tool_path_args("git_status", {}, str(project))
    output = git_status.invoke(args)

    assert "only_in_project.txt" in output
    assert "only_in_launch_dir.txt" not in output


def test_git_commit_without_cwd_commits_in_project_not_launch_dir(project_and_launch_dir):
    project, launch = project_and_launch_dir

    args = resolve_tool_path_args("git_commit", {"message": "agent commit"}, str(project))
    git_commit.invoke(args)

    assert "agent commit" in _git(project, "log", "--oneline")
    assert "agent commit" not in _git(launch, "log", "--oneline")
    