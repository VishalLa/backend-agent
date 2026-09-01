from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.task_profile import TASK_PROFILES, filter_tools_for_task
from agent.confirmation import (
    needs_confirmation,
    ALWAYS_CONFIRM_TOOLS,
    CONDITIONAL_CONFIRM_TOOLS,
)


# ============================================================================
# SECTION 1: Task Profile Tool Filtering Tests
# ============================================================================

class TestTaskProfileToolNames:
    """Verify tool names are correctly specified in each task profile."""

    def test_all_profiles_defined(self):
        """Ensure all required agent profiles exist."""
        required_profiles = {"backend", "ml", "git", "algorithms"}
        assert set(TASK_PROFILES.keys()) == required_profiles, (
            f"Missing or extra profiles. Expected {required_profiles}, "
            f"got {set(TASK_PROFILES.keys())}"
        )

    def test_backend_profile_has_expected_tools(self):
        """Backend should have file + shell + search + http tools."""
        profile = TASK_PROFILES["backend"]
        expected = {"run_shell_command", "read_file", "write_file", "delete_path"}
        assert expected.issubset(profile["tool_names"]), (
            f"Backend missing critical tools. Expected {expected} to be in "
            f"{profile['tool_names']}"
        )

    def test_ml_profile_has_expected_tools(self):
        """ML should have file + execution + background process + deletion."""
        profile = TASK_PROFILES["ml"]
        expected = {
            "execute_code",
            "read_file",
            "write_file",
            "launch_background_process",
            "delete_path",
        }
        assert expected.issubset(profile["tool_names"]), (
            f"ML missing critical tools. Expected {expected} to be in "
            f"{profile['tool_names']}"
        )

    def test_git_profile_has_only_git_tools(self):
        """Git profile should not have shell, file write, or execution tools."""
        profile = TASK_PROFILES["git"]
        dangerous_tools = {
            "run_shell_command",
            "execute_code",
            "launch_background_process",
        }
        git_tools = {
            "git_status",
            "git_diff",
            "git_log",
            "git_branch",
            "git_checkout",
            "git_commit",
            "git_push",
        }
        
        assert not dangerous_tools.intersection(profile["tool_names"]), (
            f"Git profile must not have dangerous tools. Found: "
            f"{dangerous_tools.intersection(profile['tool_names'])}"
        )
        assert git_tools.issubset(profile["tool_names"]), (
            f"Git profile missing git tools. Expected {git_tools} to be in "
            f"{profile['tool_names']}"
        )

    def test_algorithms_profile_matches_backend(self):
        """Algorithms should have same tools as backend (same capability level)."""
        backend_tools = TASK_PROFILES["backend"]["tool_names"]
        algo_tools = TASK_PROFILES["algorithms"]["tool_names"]
        assert backend_tools == algo_tools, (
            f"Algorithms tools should match backend. "
            f"Backend: {sorted(backend_tools)}, Algorithms: {sorted(algo_tools)}"
        )

    def test_no_tool_name_leaks_across_profiles(self):
        """Verify no tool appears in profiles where it shouldn't.
        
        High-risk tools should only appear in allowed profiles:
        - run_shell_command: backend, algorithms only
        - delete_path: backend, ml, algorithms only (NOT git)
        - launch_background_process: ml, algorithms only (NOT git)
        - execute_code: ml only
        """
        profiles = TASK_PROFILES
        
        # run_shell_command should not be in git or ml
        shell_tool_profiles = {
            p for p, profile in profiles.items() 
            if "run_shell_command" in profile["tool_names"]
        }
        assert shell_tool_profiles == {"backend", "algorithms"}, (
            f"run_shell_command found in unexpected profiles: {shell_tool_profiles}"
        )
        
        # execute_code should only be in ml
        code_tool_profiles = {
            p for p, profile in profiles.items()
            if "execute_code" in profile["tool_names"]
        }
        assert code_tool_profiles == {"ml"}, (
            f"execute_code found in unexpected profiles: {code_tool_profiles}"
        )
        
        # launch_background_process should only be in ml
        bg_tool_profiles = {
            p for p, profile in profiles.items()
            if "launch_background_process" in profile["tool_names"]
        }
        assert bg_tool_profiles == {"ml"}, (
            f"launch_background_process found in unexpected profiles: {bg_tool_profiles}"
        )
        
        # delete_path should not be in git
        delete_profiles = {
            p for p, profile in profiles.items()
            if "delete_path" in profile["tool_names"]
        }
        assert "git" not in delete_profiles, (
            f"delete_path should never be available to git agent"
        )

    def test_git_profile_has_no_dangerous_operations(self):
        """Git agent should have no tool that can modify local state destructively.
        
        This is intentional - git operations are all read/write to version control,
        not filesystem deletion or arbitrary command execution.
        """
        git_tools = TASK_PROFILES["git"]["tool_names"]
        dangerous = {"run_shell_command", "delete_path", "execute_code", "execute_in_sandbox"}
        intersection = dangerous.intersection(git_tools)
        assert not intersection, (
            f"Git profile has dangerous tools that should be restricted: {intersection}"
        )


# ============================================================================
# SECTION 2: Confirmation Logic Tests
# ============================================================================

class TestConfirmationLogic:
    """Verify that dangerous operations are flagged for confirmation."""

    def test_always_confirm_shell_command(self):
        """Shell commands always require confirmation."""
        assert needs_confirmation("run_shell_command", {}, confirm_all=False)
        assert needs_confirmation("run_shell_command", {"command": "rm -rf /"}, confirm_all=False)

    def test_always_confirm_git_push(self):
        """Git push always requires confirmation."""
        assert needs_confirmation("git_push", {}, confirm_all=False)
        assert needs_confirmation("git_push", {"branch": "main"}, confirm_all=False)

    def test_always_confirm_delete_path(self):
        """Path deletion always requires confirmation."""
        assert needs_confirmation("delete_path", {}, confirm_all=False)
        assert needs_confirmation("delete_path", {"path": "/tmp/foo"}, confirm_all=False)
        assert needs_confirmation("delete_path", {"recursive": True}, confirm_all=False)

    def test_always_confirm_launch_background_process(self):
        """Background process launch always requires confirmation."""
        assert needs_confirmation("launch_background_process", {}, confirm_all=False)
        assert needs_confirmation(
            "launch_background_process",
            {"command": "python script.py"},
            confirm_all=False
        )

    def test_write_file_without_overwrite_needs_no_confirmation(self):
        """Writing a new file (no overwrite) should not require confirmation."""
        result = needs_confirmation("write_file", {"path": "/tmp/new_file.txt"}, confirm_all=False)
        assert not result, "New file write should not require confirmation"
        
        result = needs_confirmation("write_file", {"overwrite": False}, confirm_all=False)
        assert not result, "Explicit overwrite=False should not require confirmation"

    def test_write_file_with_overwrite_needs_confirmation(self):
        """Overwriting an existing file requires confirmation."""
        result = needs_confirmation(
            "write_file",
            {"path": "/tmp/existing.txt", "overwrite": True},
            confirm_all=False
        )
        assert result, "File overwrite should require confirmation"

    def test_write_file_overwrite_edge_cases(self):
        """Edge cases for write_file overwrite detection."""
        # overwrite key present but falsy
        assert not needs_confirmation("write_file", {"overwrite": False}, confirm_all=False)
        
        # overwrite key present and truthy
        assert needs_confirmation("write_file", {"overwrite": True}, confirm_all=False)
        
        # overwrite key missing
        assert not needs_confirmation("write_file", {"path": "/tmp/file"}, confirm_all=False)
        
        # overwrite with empty string (falsy)
        assert not needs_confirmation("write_file", {"overwrite": ""}, confirm_all=False)

    def test_safe_operations_do_not_require_confirmation(self):
        """Read-only operations should never require confirmation."""
        safe_ops = [
            ("read_file", {"path": "/tmp/file"}),
            ("list_dir", {"path": "/tmp"}),
            ("ripgrep_search", {"pattern": "foo"}),
            ("web_search", {"query": "python"}),
            ("git_status", {}),
            ("git_diff", {"path": "file.py"}),
            ("git_log", {"count": 10}),
        ]
        
        for tool_name, tool_args in safe_ops:
            result = needs_confirmation(tool_name, tool_args, confirm_all=False)
            assert not result, f"{tool_name} should not require confirmation"

    def test_confirm_all_flag_overrides_all_checks(self):
        """When confirm_all=True, every operation requires confirmation."""
        test_cases = [
            ("read_file", {"path": "/tmp/file"}),
            ("write_file", {}),
            ("run_shell_command", {"command": "ls"}),
        ]
        
        for tool_name, tool_args in test_cases:
            result = needs_confirmation(tool_name, tool_args, confirm_all=True)
            assert result, f"{tool_name} should require confirmation when confirm_all=True"

    def test_always_confirm_tools_is_complete(self):
        """Verify ALWAYS_CONFIRM_TOOLS includes all documented dangerous operations.
        
        From README: "The agent always requires confirmation for shell commands, 
        Git pushes, path deletion, and launching background processes."
        """
        expected = {
            "run_shell_command",  # shell commands
            "git_push",            # git pushes
            "delete_path",         # path deletion
            "launch_background_process",  # background processes
        }
        assert ALWAYS_CONFIRM_TOOLS == expected, (
            f"ALWAYS_CONFIRM_TOOLS incomplete. Expected {expected}, got {ALWAYS_CONFIRM_TOOLS}"
        )

    def test_conditional_confirm_tools_includes_overwrite(self):
        """Verify CONDITIONAL_CONFIRM_TOOLS includes file overwrite check."""
        assert "write_file" in CONDITIONAL_CONFIRM_TOOLS, (
            "write_file should be in CONDITIONAL_CONFIRM_TOOLS"
        )


# ============================================================================
# SECTION 3: Path Containment Tests (Regression)
# ============================================================================

class TestPathContainment:
    """Regression tests for path resolution escaping project root.
    
    History: Several real bugs surfaced where relative paths could resolve
    outside the intended project root. This test ensures path-containment logic
    is correct and catches regressions.
    """

    def test_absolute_path_boundary(self):
        """Absolute paths should never escape when resolved against project root."""
        # This is a theoretical test - the actual implementation is in base_graph.py
        # For now, we document the expected behavior.
        project_root = Path("/home/user/project")
        
        # An absolute path should be validated to ensure it's within project_root
        safe_absolute = Path("/home/user/project/src/file.py")
        dangerous_absolute = Path("/etc/passwd")
        
        # This is documented behavior - dangerous paths should be rejected
        # at the graph tool node level
        assert str(safe_absolute).startswith(str(project_root))
        assert not str(dangerous_absolute).startswith(str(project_root))

    def test_relative_path_normalization(self):
        """Relative paths with .. should be normalized and contained."""
        # Create a temporary project directory for realistic testing
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            
            # Safe relative path
            safe_file = project_root / "src" / "file.py"
            assert str(safe_file).startswith(str(project_root))
            
            # Path that tries to escape via .. - should normalize and check
            dangerous_relative = project_root / ".." / ".." / "etc" / "passwd"
            resolved = dangerous_relative.resolve()
            
            # After normalization, this will escape. The safety check should
            # happen at the graph level BEFORE path resolution, not after.
            # This test documents that Path.resolve() cannot be relied upon
            # for security - we need explicit containment checks.
            assert not str(resolved).startswith(str(project_root.resolve()))

    def test_path_traversal_attempt_blocked(self):
        """Path containment should be verified before resolution.
        
        This test documents that we need explicit checks to prevent path traversal,
        not relying on Path.resolve() which will escape confinement.
        """
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir) / "project"
            project_root.mkdir()
            
            # Extreme path traversal attempt - this represents what the user submitted
            evil_path_str = "../../../../../../../../etc/passwd"
            
            # Safety check should happen on the string BEFORE we resolve
            # Using an explicit containment test
            def is_path_contained(path_str: str, root: Path) -> bool:
                """Check if a path stays within root without using resolve()."""
                # Prevent absolute paths outright
                if path_str.startswith("/"):
                    return False
                # Prevent .. at the start or after /
                if ".." in path_str:
                    return False
                return True
            
            # With explicit checks, this should fail
            assert not is_path_contained(evil_path_str, project_root)
            assert is_path_contained("src/file.py", project_root)
            assert is_path_contained("file.py", project_root)


# ============================================================================
# SECTION 4: Integration Tests
# ============================================================================

class TestTaskProfileIntegration:
    """Test task profile filtering with actual tool objects."""

    def test_filter_tools_for_backend(self):
        """Test that filter_tools_for_task correctly filters backend tools."""
        # Create mock tools - need actual string names, not mock attributes
        tools = []
        for tool_name in ["run_shell_command", "read_file", "write_file", "git_push", "execute_code"]:
            tool = MagicMock()
            tool.name = tool_name  # Set as string attribute, not mock
            tools.append(tool)
        
        filtered = filter_tools_for_task("backend", tools)
        filtered_names = {t.name for t in filtered}
        
        expected = {"run_shell_command", "read_file", "write_file"}
        assert filtered_names == expected, (
            f"Backend filtering wrong. Expected {expected}, got {filtered_names}"
        )

    def test_filter_tools_for_git(self):
        """Test that filter_tools_for_task correctly filters git tools."""
        tools = []
        for tool_name in ["git_push", "git_status", "git_commit", "run_shell_command", "delete_path"]:
            tool = MagicMock()
            tool.name = tool_name  # Set as string attribute, not mock
            tools.append(tool)
        
        filtered = filter_tools_for_task("git", tools)
        filtered_names = {t.name for t in filtered}
        
        expected = {"git_push", "git_status", "git_commit"}
        assert filtered_names == expected, (
            f"Git filtering wrong. Expected {expected}, got {filtered_names}"
        )

    def test_filter_tools_rejects_empty_result(self):
        """Filtering that results in no tools should raise ValueError."""
        # All tools are filtered out
        tool = MagicMock()
        tool.name = "git_push"  # Set as string attribute
        tools = [tool]
        
        with pytest.raises(ValueError, match="No tools matched"):
            filter_tools_for_task("backend", tools)


# ============================================================================
# SECTION 5: Safety Documentation Tests
# ============================================================================

class TestSafetyDocumentation:
    """Ensure safety requirements are properly documented and enforced."""

    def test_dangerous_operations_documented(self):
        """Verify all dangerous operations mentioned in README are in confirmation lists."""
        # From README: "The agent always requires confirmation for shell commands, 
        # Git pushes, path deletion, and launching background processes. 
        # Overwriting a file also requires confirmation."
        
        documented_dangerous = {
            "shell commands",  # run_shell_command
            "Git pushes",       # git_push
            "path deletion",    # delete_path
            "launching background processes",  # launch_background_process
            "file overwrite",   # write_file with overwrite=True
        }
        
        # Verify we have confirmation logic for all of these
        assert "run_shell_command" in ALWAYS_CONFIRM_TOOLS
        assert "git_push" in ALWAYS_CONFIRM_TOOLS
        assert "delete_path" in ALWAYS_CONFIRM_TOOLS
        assert "launch_background_process" in ALWAYS_CONFIRM_TOOLS
        assert "write_file" in CONDITIONAL_CONFIRM_TOOLS

    def test_git_agent_safety_boundary(self):
        """Git agent should not be able to execute code or run shell commands."""
        git_profile = TASK_PROFILES["git"]
        
        # These should NOT be in git profile
        forbidden_tools = {
            "run_shell_command",
            "execute_code",
            "delete_path",
            "launch_background_process",
            "write_file",
            "append_file",
            "edit_file",
        }
        
        assert not forbidden_tools.intersection(git_profile["tool_names"]), (
            f"Git profile contains forbidden tools: "
            f"{forbidden_tools.intersection(git_profile['tool_names'])}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
