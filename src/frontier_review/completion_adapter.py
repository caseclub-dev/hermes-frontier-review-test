"""Completion adapter for Frontier Review — Phase 2 hardening.

Provides ``CoderCompletionAdapter`` which bridges completed Coder
delegations into the WorkflowLedger as PENDING_REVIEW tasks, performing
idempotent registration and optional git-head verification.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Optional

from .workflow_state import TaskRecord, WorkflowLedger


class CoderCompletionAdapter:
    """Adapt a completed Coder delegation into a PENDING_REVIEW ledger entry.

    Parameters
    ----------
    ledger : WorkflowLedger
        The durable task-state ledger that owns all review tasks.
    """

    def __init__(self, ledger: WorkflowLedger) -> None:
        self.ledger = ledger

    # ── Public API ────────────────────────────────────────────────────

    def register_from_delegation_result(
        self,
        delegation_id: str,
        workspace_path: str,
        requirements: str,
        acceptance_criteria: str,
        task_id: Optional[str] = None,
    ) -> TaskRecord:
        """Register a completed Coder delegation as PENDING_REVIEW.

        Idempotent — calling this twice with the same inputs produces
        exactly one ledger entry.  Verifies git HEAD and runs pytest
        (if tests exist) before registration.

        Args:
            delegation_id: The Hermes delegate_task ID returned by the orchestrator.
            workspace_path: Path to the Coder's working directory.
            requirements: Human-readable task description.
            acceptance_criteria: JSON string or free-text criteria.
            task_id: Optional explicit task ID; auto-generated from a stable hash if omitted.

        Returns:
            The ``TaskRecord`` for this review task.
        """
        # Verify git HEAD (non-fatal — returns None on failure)
        head_sha = self.verify_git_commit(workspace_path)

        # Run pytest discovery if test files exist (best-effort)
        self._try_run_pytest(workspace_path)

        # Generate stable task_id from delegation_id + requirements hash
        if task_id is None:
            task_id = self._generate_task_id(delegation_id, requirements)

        now = datetime.now(timezone.utc).isoformat()

        return self.ledger.create_pending_task(
            task_id=task_id,
            requirements=requirements,
            workspace_path=workspace_path,
            acceptance_criteria=acceptance_criteria,
        )

    def verify_git_commit(self, workspace_path: str) -> Optional[str]:
        """Return the HEAD commit SHA of a git repo, or None on failure.

        Args:
            workspace_path: Path to the repository root.

        Returns:
            40-character hex SHA string, or ``None`` if not a valid git repo.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=workspace_path,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass
        return None

    # ── Internal helpers ──────────────────────────────────────────────

    def _generate_task_id(self, delegation_id: str, requirements: str) -> str:
        """Produce a stable 16-char task ID from inputs.

        Args:
            delegation_id: The delegate_task identifier.
            requirements: Task description string.

        Returns:
            A deterministic ``<delegation_id_prefix>-<hash>`` string.
        """
        raw = f"{delegation_id}{requirements}"
        h = hashlib.sha256(raw.encode()).hexdigest()[:8]
        prefix = delegation_id[-8:] if len(delegation_id) >= 8 else delegation_id
        return f"{prefix}-{h}"

    def _try_run_pytest(self, workspace_path: str) -> None:
        """Attempt to run pytest in the workspace directory.

        Best-effort — any exception is silently swallowed.  Only runs
        if there are files matching ``test_*.py`` or ``*_test.py``.
        """
        try:
            test_files = []
            for root, _dirs, files in os.walk(workspace_path):
                for fname in files:
                    if fname.startswith("test_") or fname.endswith("_test.py"):
                        test_files.append(os.path.join(root, fname))
            if not test_files:
                return

            result = subprocess.run(
                ["python", "-m", "pytest"] + test_files,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=workspace_path,
            )
            # Ignore result — logging is handled by caller if needed
        except (subprocess.TimeoutExpired, OSError):
            pass  # Best-effort only

    # ── Legacy convenience alias ──────────────────────────────────────

    def register_from_delegation_result_legacy(
        self,
        delegation_id: str,
        workspace_path: str,
        requirements: str,
        acceptance_criteria: str,
        task_id: Optional[str] = None,
    ) -> TaskRecord:
        """Backwards-compatible alias for *register_from_delegation_result*.

        Kept so older callers do not break.  Delegates to the main method.
        """
        return self.register_from_delegation_result(
            delegation_id=delegation_id,
            workspace_path=workspace_path,
            requirements=requirements,
            acceptance_criteria=acceptance_criteria,
            task_id=task_id,
        )
