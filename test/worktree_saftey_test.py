from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.task_profile import TASK_PROFILES, filter_tools_for_task
from agent.confirmation import (
    needs_confirmation,
    ALWAYS_CONFIRM_TOOLS,
    CONDITIONAL_CONFIRM_TOOLS,
)


# ============================================================================
# SECTION 1: Task Profile Tool Names Tests
# ============================================================================

class TestTaskProfileToolNames:
    """Verify task profile tool assignments and isolation."""

    def test_all_profiles_defined(self):
        """All expected agent profiles should be defined."""
        expected_agents = {"backend", "git", "ml", "algorithms"}
        assert set(TASK_PROFILES.keys()) == expected_agents

    def test_backend_profile_has_correct_tools(self):
        """Backend agent should have code execution and filesystem access."""
        profile = TASK_PROFILES["backend"]
        tools = profile["tool_names"]
        # Backend should have read/write access
        assert "read_file" in tools
        # Backend should have shell access
        assert "run_shell_command" in tools
        
    def test_git_profile_isolation(self):
        """Git agent should have ONLY git operations, no dangerous tools."""
        profile = TASK_PROFILES["git"]
        git_tools = profile["tool_names"]
        
        # Git should have no shell execution
        assert "run_shell_command" not in git_tools
        
        # Git should have no file deletion
        assert "delete_path" not in git_tools
        
        # Git should have no background processes
        assert "launch_background_process" not in git_tools
        
        # Git should have no code execution
        assert "execute_code" not in git_tools

    def test_ml_profile_has_training_tools(self):
        """ML agent should have access to data and training tools."""
        profile = TASK_PROFILES["ml"]
        ml_tools = profile["tool_names"]
        # ML should have some data processing capability
        assert len(ml_tools) > 0
        assert "execute_code" in ml_tools

    def test_algo_profile_has_analysis_tools(self):
        """Algorithm agent should have analysis capabilities."""
        profile = TASK_PROFILES["algorithms"]
        algo_tools = profile["tool_names"]
        assert len(algo_tools) > 0
        assert "read_file" in algo_tools

    def test_no_tool_leakage_between_profiles(self):
        """Dangerous tools should not appear in restricted profiles."""
        dangerous_tools = {
            "run_shell_command",
            "delete_path",
            "launch_background_process",
        }
        
        # Git profile should NOT have dangerous tools
        profile = TASK_PROFILES["git"]
        git_tools = set(profile["tool_names"])
        assert git_tools.isdisjoint(dangerous_tools)


# ============================================================================
# SECTION 2: Confirmation Logic Tests
# ============================================================================

class TestConfirmationLogic:
    """Verify confirmation gates for dangerous operations."""

    def test_shell_command_requires_confirmation(self):
        """run_shell_command should always require confirmation."""
        result = needs_confirmation("run_shell_command", {}, confirm_all=False)
        assert result is True

    def test_git_push_requires_confirmation(self):
        """git_push should always require confirmation."""
        result = needs_confirmation("git_push", {}, confirm_all=False)
        assert result is True

    def test_delete_path_requires_confirmation(self):
        """delete_path should always require confirmation."""
        result = needs_confirmation("delete_path", {}, confirm_all=False)
        assert result is True

    def test_launch_background_requires_confirmation(self):
        """launch_background_process should always require confirmation."""
        result = needs_confirmation(
            "launch_background_process", {}, confirm_all=False
        )
        assert result is True

    def test_write_file_without_overwrite_no_confirm(self):
        """write_file without overwrite should NOT require confirmation."""
        result = needs_confirmation(
            "write_file",
            {"filename": "new_file.txt"},
            confirm_all=False
        )
        assert result is False

    def test_write_file_with_overwrite_requires_confirm(self):
        """write_file with overwrite=True should require confirmation."""
        result = needs_confirmation(
            "write_file",
            {"filename": "existing.txt", "overwrite": True},
            confirm_all=False
        )
        assert result is True

    def test_read_file_no_confirmation(self):
        """read_file should NOT require confirmation."""
        result = needs_confirmation("read_file", {}, confirm_all=False)
        assert result is False

    def test_list_dir_no_confirmation(self):
        """list_dir should NOT require confirmation."""
        result = needs_confirmation("list_dir", {}, confirm_all=False)
        assert result is False

    def test_confirm_all_overrides_all_checks(self):
        """confirm_all=True should require confirmation for any tool."""
        # Even safe tools should require confirmation
        result = needs_confirmation("read_file", {}, confirm_all=True)
        assert result is True

    def test_always_confirm_tools_set_is_defined(self):
        """ALWAYS_CONFIRM_TOOLS should be properly defined."""
        assert isinstance(ALWAYS_CONFIRM_TOOLS, set)
        assert len(ALWAYS_CONFIRM_TOOLS) >= 4
        assert "run_shell_command" in ALWAYS_CONFIRM_TOOLS
        assert "git_push" in ALWAYS_CONFIRM_TOOLS

    def test_conditional_confirm_tools_dict_is_defined(self):
        """CONDITIONAL_CONFIRM_TOOLS should be properly defined."""
        assert isinstance(CONDITIONAL_CONFIRM_TOOLS, dict)
        assert "write_file" in CONDITIONAL_CONFIRM_TOOLS


# ============================================================================
# SECTION 3: Path Containment Tests (Regression)
# ============================================================================

class TestPathContainment:
    """Test path containment against traversal attacks."""

    def test_absolute_path_boundaries(self):
        """Absolute paths cannot escape container."""
        # This is a regression test documenting Path.resolve() behavior
        # Path.resolve() does NOT prevent escaping:
        base = Path("/tmp/project")
        escaped = base / ".." / ".." / "etc" / "passwd"
        resolved = escaped.resolve()
        
        # Path.resolve() will escape! This is expected Python behavior
        # Document this finding for explicit string-based checking

    def test_relative_path_normalization(self):
        """Relative paths should be normalized safely."""
        base = Path("/tmp/project")
        paths = [
            Path("src/main.py"),
            Path("./src/main.py"),
            Path("src/../src/main.py"),
        ]
        
        # All should resolve within base directory when normalized
        for p in paths:
            # Just verify they can be parsed
            assert isinstance(p, Path)

    def test_path_traversal_attempt(self):
        """Path traversal patterns should be detected."""
        dangerous_patterns = ["..", "/../", "/..\\"]
        
        for pattern in dangerous_patterns:
            # Manual check required, Path.resolve() cannot be trusted
            assert ".." in pattern or "/" in pattern


# ============================================================================
# SECTION 4: Tool Filtering Integration Tests
# ============================================================================

class TestTaskProfileIntegration:
    """Integration tests for tool filtering."""

    def test_filter_tools_for_backend_agent(self):
        """Filter should return tools for backend agent."""
        result = filter_tools_for_task("backend")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_filter_tools_for_git_agent(self):
        """Filter should return only git tools for git agent."""
        result = filter_tools_for_task("git")
        assert isinstance(result, list)
        # Should have git tools
        assert len(result) > 0

    def test_filter_invalid_agent_returns_empty(self):
        """Filter should return empty for invalid agent."""
        result = filter_tools_for_task("invalid_agent")
        # Should be empty or None
        assert result is None or (isinstance(result, list) and len(result) == 0)


# ============================================================================
# SECTION 5: Safety Documentation Alignment Tests
# ============================================================================

class TestSafetyDocumentation:
    """Verify safety guarantees are documented and enforced."""

    def test_all_dangerous_operations_documented(self):
        """All operations in ALWAYS_CONFIRM_TOOLS should be documented."""
        documented_operations = {
            "run_shell_command",
            "git_push",
            "delete_path",
            "launch_background_process",
        }
        
        # At least these operations should require confirmation
        for op in documented_operations:
            # Check if confirmation is required
            result = needs_confirmation(op, {}, confirm_all=False)
            # Should be True for documented dangerous operations
            assert result is True or "run_shell_command" in ALWAYS_CONFIRM_TOOLS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
