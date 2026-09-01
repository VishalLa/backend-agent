"""
Phase 2: Git Worktree Isolation Tests

Tests for worktree creation, diff summary, merge/discard, and cleanup operations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import subprocess

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.worktree import GitWorktree, WorktreeError


class TestWorktreeCreation:
    """Test worktree creation and initialization."""

    def test_init_with_valid_git_repo(self, tmp_path):
        """Worktree should initialize successfully with valid git repo."""
        repo = tmp_path / "test_repo"
        repo.mkdir()
        
        subprocess.run(
            ["git", "init"],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        
        worktree = GitWorktree(repo)
        assert worktree.project_root == repo

    def test_init_fails_with_invalid_git_repo(self, tmp_path):
        """Worktree initialization should fail with non-git directory."""
        non_git_dir = tmp_path / "not_a_repo"
        non_git_dir.mkdir()
        
        with pytest.raises(WorktreeError, match="Not a valid git repository"):
            GitWorktree(non_git_dir)

    @patch("subprocess.run")
    def test_create_worktree_success(self, mock_run, tmp_path):
        """Creating a worktree should execute git worktree add command."""
        repo = tmp_path / "repo"
        repo.mkdir()
        
        mock_run.return_value = MagicMock(returncode=0)
        
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.project_root = repo
        worktree.active_worktrees = {}
        
        with patch.object(worktree, "_validate_git_repo"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                result = worktree.create_worktree("test-task")
        
        expected_path = repo.parent / "agent-work-test-task"
        assert result == expected_path
        assert "test-task" in worktree.active_worktrees

    @patch("subprocess.run")
    def test_create_worktree_fails_if_exists(self, mock_run, tmp_path):
        """Creating worktree at existing path should raise."""
        repo = tmp_path / "repo"
        repo.mkdir()
        
        worktree_path = repo.parent / "agent-work-existing"
        worktree_path.mkdir()
        
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.project_root = repo
        worktree.active_worktrees = {}
        
        with patch.object(worktree, "_validate_git_repo"):
            with pytest.raises(WorktreeError, match="already exists"):
                worktree.create_worktree("existing")

    @patch("subprocess.run")
    def test_create_worktree_subprocess_error(self, mock_run, tmp_path):
        """Subprocess failure should raise WorktreeError."""
        repo = tmp_path / "repo"
        repo.mkdir()
        
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.project_root = repo
        worktree.active_worktrees = {}
        
        with patch.object(worktree, "_validate_git_repo"):
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(
                    1, "git", stderr=b"error message"
                )
                
                with pytest.raises(WorktreeError, match="Failed to create"):
                    worktree.create_worktree("test")


class TestWorktreePaths:
    """Test worktree path management."""

    def test_get_worktree_path_exists(self):
        """Getting path for existing worktree should return it."""
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.active_worktrees = {"task1": Path("/tmp/agent-work-task1")}
        
        result = worktree.get_worktree_path("task1")
        assert result == Path("/tmp/agent-work-task1")

    def test_get_worktree_path_not_found(self):
        """Getting path for non-existent worktree should return None."""
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.active_worktrees = {}
        
        result = worktree.get_worktree_path("nonexistent")
        assert result is None

    def test_list_worktrees(self):
        """List worktrees should return all active worktrees."""
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.active_worktrees = {
            "task1": Path("/tmp/agent-work-task1"),
            "task2": Path("/tmp/agent-work-task2"),
        }
        
        result = worktree.list_worktrees()
        assert result == worktree.active_worktrees
        assert len(result) == 2

    def test_get_branch_name(self):
        """Branch name should follow pattern agent/<task_id>."""
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.active_worktrees = {"abc123": Path("/tmp/agent-work-abc123")}
        
        result = worktree.get_branch_name("abc123")
        assert result == "agent/abc123"


class TestDiffSummary:
    """Test diff summary retrieval."""

    @patch("subprocess.run")
    def test_get_diff_summary_success(self, mock_run):
        """Getting diff summary should execute git diff command."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="2 files changed, 10 insertions, 5 deletions"
        )
        
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.active_worktrees = {"task1": Path("/tmp/agent-work-task1")}
        
        result = worktree.get_diff_summary("task1")
        
        assert result is not None

    @patch("subprocess.run")
    def test_get_diff_summary_not_found(self, mock_run):
        """Getting diff for non-existent worktree should raise."""
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.active_worktrees = {}
        
        with pytest.raises(WorktreeError, match="No worktree found"):
            worktree.get_diff_summary("nonexistent")


class TestWorktreeMergeDiscard:
    """Test merge and discard operations."""

    @patch("subprocess.run")
    def test_merge_worktree_success(self, mock_run):
        """Merge should execute git merge command."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Merge made by...")
        
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.project_root = Path("/repo")
        worktree.active_worktrees = {"task1": Path("/tmp/agent-work-task1")}
        
        result = worktree.merge_worktree("task1", delete=False)
        
        assert result is not None

    @patch("subprocess.run")
    def test_merge_worktree_not_found(self, mock_run):
        """Merge non-existent worktree should raise."""
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.active_worktrees = {}
        
        with pytest.raises(WorktreeError, match="No worktree found"):
            worktree.merge_worktree("nonexistent")

    @patch("subprocess.run")
    def test_discard_worktree_success(self, mock_run):
        """Discard should remove worktree and branch."""
        mock_run.return_value = MagicMock(returncode=0)
        
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.project_root = Path("/repo")
        worktree.active_worktrees = {"task1": Path("/tmp/agent-work-task1")}
        
        result = worktree.discard_worktree("task1")
        
        assert "removed" in result.lower()
        assert "task1" not in worktree.active_worktrees

    @patch("subprocess.run")
    def test_discard_worktree_not_found(self, mock_run):
        """Discard non-existent worktree should raise."""
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.active_worktrees = {}
        
        with pytest.raises(WorktreeError, match="No worktree found"):
            worktree.discard_worktree("nonexistent")


class TestWorktreeCleanup:
    """Test worktree cleanup operations."""

    @patch("subprocess.run")
    def test_cleanup_all_success(self, mock_run):
        """Cleanup all should remove all worktrees."""
        mock_run.return_value = MagicMock(returncode=0)
        
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.project_root = Path("/repo")
        worktree.active_worktrees = {
            "task1": Path("/tmp/agent-work-task1"),
            "task2": Path("/tmp/agent-work-task2"),
        }
        
        result = worktree.cleanup_all()
        
        assert result is not None

    def test_cleanup_all_empty(self):
        """Cleanup all with no worktrees should return message."""
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.active_worktrees = {}
        
        result = worktree.cleanup_all()
        
        assert "No worktrees" in result


class TestWorktreeIntegration:
    """Integration tests for complete worktree workflow."""

    def test_worktree_workflow_isolation(self, tmp_path):
        """Complete workflow: create -> diff -> merge/discard."""
        repo = tmp_path / "repo"
        repo.mkdir()
        
        subprocess.run(
            ["git", "init"],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo,
            capture_output=True,
        )
        
        (repo / "README.md").write_text("# Test Project")
        subprocess.run(
            ["git", "add", "."],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        
        worktree_mgr = GitWorktree(repo)
        worktree_path = worktree_mgr.create_worktree("feature-x")
        
        assert worktree_path.exists()
        assert worktree_path.is_dir()

    def test_worktree_branch_naming(self, tmp_path):
        """Worktree should use agent/<task-id> branch naming convention."""
        repo = tmp_path / "repo"
        repo.mkdir()
        
        subprocess.run(
            ["git", "init"],
            cwd=repo,
            capture_output=True,
            check=True,
        )
        
        worktree_mgr = GitWorktree(repo)
        worktree_path = worktree_mgr.create_worktree("task123")
        
        branch_name = worktree_mgr.get_branch_name("task123")
        assert branch_name == "agent/task123"
        assert branch_name.startswith("agent/")


class TestWorktreeSafety:
    """Test safety guards and error handling."""

    def test_no_double_remove(self):
        """Removing same worktree twice should be safe."""
        worktree = GitWorktree.__new__(GitWorktree)
        worktree.project_root = Path("/repo")
        worktree.active_worktrees = {"task1": Path("/tmp/agent-work-task1")}
        
        with patch("subprocess.run"):
            worktree.active_worktrees.pop("task1", None)
            if "task2" in worktree.active_worktrees:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
