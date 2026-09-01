"""
Git Worktree Isolation

Manages isolated git worktrees for each agent task, allowing safe experimentation
and easy review/rollback of changes. Each task runs in a separate branch on a
worktree at ../agent-work-<task-id>, keeping the main working directory clean.

Workflow:
1. Before task starts: Create worktree on scratch branch
2. Agent operates on isolated worktree
3. On completion: Summarize diff for review
4. User decides: merge, discard, or iterate
5. After decision: Clean up worktree
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional
import uuid


class WorktreeError(Exception):
    """Raised when worktree operations fail."""
    pass


class GitWorktree:
    """Manages isolated git worktrees for agent tasks."""

    def __init__(
        self, 
        project_root: Optional[Path] = None
    ) -> None:
        """Initialize worktree manager.
        
        Args:
            project_root: Path to the git repository root. If None, uses current directory.
        """
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self._validate_git_repo()
        self.active_worktrees: dict[str, Path] = {}  # task_id -> worktree_path
        

    def _validate_git_repo(self) -> None:
        """Ensure we're in a valid git repository."""
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.project_root,
                capture_output=True,
                check=True,
                timeout=5,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise WorktreeError(
                f"Not a valid git repository at {self.project_root}: {e}"
            )


    def create_worktree(self, task_id: Optional[str] = None) -> Path:
        """Create a new git worktree for a task.
        
        Args:
            task_id: Optional task identifier. If None, generates a random UUID.
        
        Returns:
            Path to the created worktree.
        
        Raises:
            WorktreeError: If worktree creation fails.
        """
        if not task_id:
            task_id = str(uuid.uuid4())[:12]  # Use first 12 chars of UUID
        
        # Worktree will be created as ../agent-work-<task-id> relative to project root
        worktree_parent = self.project_root.parent
        worktree_path = worktree_parent / f"agent-work-{task_id}"
        branch_name = f"agent/{task_id}"
        
        # Check if worktree already exists
        if worktree_path.exists():
            raise WorktreeError(
                f"Worktree already exists at {worktree_path}. "
                f"Remove it with: git worktree remove {worktree_path}"
            )
        
        try:
            # Create worktree on a new branch based on current HEAD
            subprocess.run(
                ["git", "worktree", "add", str(worktree_path), "-b", branch_name],
                cwd=self.project_root,
                capture_output=True,
                check=True,
                timeout=30,
            )
        except subprocess.CalledProcessError as e:
            raise WorktreeError(
                f"Failed to create worktree at {worktree_path}: "
                f"{e.stderr.decode() if e.stderr else e}"
            )
        except subprocess.TimeoutExpired:
            raise WorktreeError(f"Worktree creation timed out for {task_id}")
        
        self.active_worktrees[task_id] = worktree_path
        return worktree_path
    

    def get_worktree_path(self, task_id: str) -> Optional[Path]:
        """Get the path to a worktree by task ID.
        
        Args:
            task_id: Task identifier.
        
        Returns:
            Path to the worktree, or None if it doesn't exist.
        """
        return self.active_worktrees.get(task_id)


    def list_worktrees(self) -> dict[str, Path]:
        """List all active worktrees managed by this instance.
        
        Returns:
            Dictionary mapping task_id to worktree_path.
        """
        return dict(self.active_worktrees)


    def get_diff_summary(self, task_id: str) -> str:
        """Get a summary of changes in the worktree.
        
        Args:
            task_id: Task identifier.
        
        Returns:
            String containing the diff summary (files changed, insertions, deletions).
        
        Raises:
            WorktreeError: If worktree doesn't exist or diff retrieval fails.
        """
        worktree_path = self.active_worktrees.get(task_id)
        if not worktree_path:
            raise WorktreeError(f"No worktree found for task {task_id}")
        
        try:
            # Get diff stats between main/master and current branch
            result = subprocess.run(
                ["git", "diff", "--stat", "HEAD..."],
                cwd=worktree_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if result.returncode != 0:
                # Try alternative: diff against parent branch
                result = subprocess.run(
                    ["git", "diff", "--stat", "origin/HEAD"],
                    cwd=worktree_path,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            
            return result.stdout or "(no changes)"
        
        except subprocess.TimeoutExpired:
            raise WorktreeError(f"Diff retrieval timed out for {task_id}")
        except Exception as e:
            raise WorktreeError(f"Failed to get diff for {task_id}: {e}")


    def get_branch_name(self, task_id: str) -> Optional[str]:
        """Get the branch name for a worktree.
        
        Args:
            task_id: Task identifier.
        
        Returns:
            Branch name (e.g., "agent/abc123def456"), or None if worktree doesn't exist.
        """
        return f"agent/{task_id}" if task_id in self.active_worktrees else None


    def merge_worktree(self, task_id: str, delete: bool = True) -> str:
        """Merge worktree changes back to the main branch.
        
        Args:
            task_id: Task identifier.
            delete: If True, remove the worktree after merge.
        
        Returns:
            String with merge results.
        
        Raises:
            WorktreeError: If merge or cleanup fails.
        """
        worktree_path = self.active_worktrees.get(task_id)
        if not worktree_path:
            raise WorktreeError(f"No worktree found for task {task_id}")
        
        try:
            # Get current branch to restore later
            current_branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            current_branch = current_branch_result.stdout.strip()
            
            # Merge the worktree branch into current branch
            branch_name = self.get_branch_name(task_id)
            merge_result = subprocess.run(
                ["git", "merge", branch_name],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            result_msg = merge_result.stdout or merge_result.stderr or "Merge completed"
            
            if delete:
                self.remove_worktree(task_id)
            
            return result_msg
        
        except subprocess.CalledProcessError as e:
            raise WorktreeError(
                f"Failed to merge worktree {task_id}: "
                f"{e.stderr.decode() if e.stderr else e}"
            )
        except subprocess.TimeoutExpired:
            raise WorktreeError(f"Merge operation timed out for {task_id}")


    def discard_worktree(self, task_id: str) -> str:
        """Discard all changes and remove the worktree.
        
        Args:
            task_id: Task identifier.
        
        Returns:
            Confirmation message.
        
        Raises:
            WorktreeError: If removal fails.
        """
        worktree_path = self.active_worktrees.get(task_id)
        if not worktree_path:
            raise WorktreeError(f"No worktree found for task {task_id}")
        
        try:
            # Remove the worktree (and optionally the branch)
            subprocess.run(
                ["git", "worktree", "remove", str(worktree_path), "--force"],
                cwd=self.project_root,
                capture_output=True,
                check=True,
                timeout=10,
            )
            
            # Also delete the branch
            branch_name = self.get_branch_name(task_id)
            subprocess.run(
                ["git", "branch", "-D", branch_name],
                cwd=self.project_root,
                capture_output=True,
                timeout=5,
            )
            
            del self.active_worktrees[task_id]
            return f"Worktree and branch {branch_name} removed"
        
        except subprocess.CalledProcessError as e:
            raise WorktreeError(
                f"Failed to remove worktree {task_id}: "
                f"{e.stderr.decode() if e.stderr else e}"
            )
        except subprocess.TimeoutExpired:
            raise WorktreeError(f"Worktree removal timed out for {task_id}")
        except KeyError:
            pass  # Already removed


    def remove_worktree(self, task_id: str) -> str:
        """Remove a worktree (alias for discard_worktree).
        
        Args:
            task_id: Task identifier.
        
        Returns:
            Confirmation message.
        """
        return self.discard_worktree(task_id)


    def cleanup_all(self) -> str:
        """Remove all active worktrees.
        
        Returns:
            Summary of cleanup operations.
        """
        task_ids = list(self.active_worktrees.keys())
        if not task_ids:
            return "No worktrees to clean up"
        
        results = []
        for task_id in task_ids:
            try:
                msg = self.discard_worktree(task_id)
                results.append(f"✓ {task_id}: {msg}")
            except WorktreeError as e:
                results.append(f"✗ {task_id}: {e}")
        
        return "\n".join(results)


    def __enter__(self):
        """Context manager entry."""
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleans up all worktrees."""
        self.cleanup_all()
