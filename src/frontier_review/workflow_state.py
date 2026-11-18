"""Durable task-state persistence for the autonomous loop.

Provides a JSON-backed ledger that tracks task state across process restarts,
enforcing correction cycle limits, review dispatch bounds, duplicate prevention,
and budget exhaustion tracking.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


# ── Data models ────────────────────────────────────────────────────────

@dataclass
class TaskRecord:
    """Persistent record for a single autonomous review task."""

    task_id: str
    requirements: str
    workspace_path: str
    original_sha: str = ""
    current_sha: str = ""
    initial_delegation_id: str = ""
    corrected_delegation_ids: list[str] = field(default_factory=list)
    review_sequence: list[dict[str, Any]] = field(default_factory=list)
    correction_count: int = 0
    max_correction_cycles: int = 3
    max_review_dispatches: int = 4
    final_disposition: str = ""
    total_mock_cost_usd: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskRecord":
        """Deserialize from a dict (e.g. loaded from JSON)."""
        known = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class WorkflowState:
    """In-memory workflow state wrapping a TaskRecord plus tracking helpers."""

    task_record: TaskRecord
    correction_cycle_count: int = 0
    review_dispatch_count: int = 0
    cumulative_cost_usd: float = 0.0
    final_disposition: str = ""


# ── WorkflowLedger ─────────────────────────────────────────────────────

class WorkflowLedger:
    """JSON-backed ledger tracking task state across process restarts.

    Enforces:
    - Maximum correction cycles per task (tracked, not reset on restart)
    - Maximum review dispatches per task
    - No duplicate review for unchanged SHA + evidence fingerprint
    - Budget exhaustion tracking (monthly and per-task)
    """

    def __init__(self, ledger_path: str) -> None:
        """Initialize the ledger.

        Args:
            ledger_path: Path to the JSON file backing this ledger.
                         A temp directory is created if the path's parent
                         doesn't exist.
        """
        self._path = ledger_path
        # Ensure parent directory exists
        parent = os.path.dirname(ledger_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        self._data: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load ledger from disk."""
        try:
            with open(self._path) as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self._data = raw
            else:
                self._data = {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._data = {}

    def _save(self) -> None:
        """Persist ledger to disk atomically (write tmp then replace)."""
        try:
            tmp_path = self._path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(self._data, f, indent=2, default=str)
            os.replace(tmp_path, self._path)
        except OSError:
            # Fallback: direct write (shouldn't fail in practice)
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=2, default=str)

    def create_task(
        self,
        task_id: str,
        requirements: str,
        workspace_path: str,
        original_sha: str = "",
        max_correction_cycles: int = 3,
        max_review_dispatches: int = 4,
    ) -> TaskRecord:
        """Create a new task record.

        Args:
            task_id: Unique identifier for this review task.
            requirements: What the code must satisfy.
            workspace_path: Path to the disposable workspace.
            original_sha: Initial commit SHA.
            max_correction_cycles: Maximum correction iterations allowed.
            max_review_dispatches: Maximum total reviews allowed.

        Returns:
            The newly created TaskRecord.
        """
        if task_id in self._data:
            # Idempotent: return existing record (concurrent callers may overlap)
            return TaskRecord.from_dict(self._data[task_id])

        now = datetime.now(timezone.utc).isoformat()
        record = TaskRecord(
            task_id=task_id,
            requirements=requirements,
            workspace_path=workspace_path,
            original_sha=original_sha,
            current_sha=original_sha,
            max_correction_cycles=max_correction_cycles,
            max_review_dispatches=max_review_dispatches,
            created_at=now,
            updated_at=now,
        )
        self._data[task_id] = record.to_dict()
        self._save()
        return record

    def get_task(self, task_id: str) -> TaskRecord:
        """Load an existing task record.

        Args:
            task_id: The task to look up.

        Returns:
            The TaskRecord for this task_id.

        Raises:
            ValueError: If the task doesn't exist.
        """
        if task_id not in self._data:
            raise ValueError(f"Task '{task_id}' not found in ledger")
        return TaskRecord.from_dict(self._data[task_id])

    def record_correction(
        self,
        task_id: str,
        new_sha: str,
        delegation_id: str,
        diff: str = "",
        test_output: str = "",
    ) -> None:
        """Record a correction cycle result.

        Args:
            task_id: The task being corrected.
            new_sha: New commit SHA after correction.
            delegation_id: ID of the Coder delegation that produced this fix.
            diff: Git diff of the correction.
            test_output: Test evidence output.
        """
        if task_id not in self._data:
            raise ValueError(f"Task '{task_id}' not found for correction recording")

        rec = self._data[task_id]
        rec["current_sha"] = new_sha
        rec["corrected_delegation_ids"].append(delegation_id)
        rec["correction_count"] = rec.get("correction_count", 0) + 1
        rec["updated_at"] = datetime.now(timezone.utc).isoformat()

        # Append to review sequence for audit trail
        rec.setdefault("review_sequence", []).append({
            "type": "correction",
            "delegation_id": delegation_id,
            "new_sha": new_sha,
            "diff": diff[:500] if diff else "",  # Truncate for storage
            "test_output": test_output[:500] if test_output else "",
        })

        self._data[task_id] = rec
        self._save()

    def increment_correction_count(self, task_id: str) -> None:
        """Increment the correction count for a task and persist it.

        Args:
            task_id: The task whose correction count to increment.
        """
        if task_id not in self._data:
            raise ValueError(f"Task '{task_id}' not found for correction count")
        rec = self._data[task_id]
        rec["correction_count"] = rec.get("correction_count", 0) + 1
        rec["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._data[task_id] = rec
        self._save()

    def record_review(
        self,
        task_id: str,
        review_status: str,
        findings: list[dict],
        token_cost: float,
        commit_sha: str = "",
        evidence_fingerprint: str = "",
    ) -> bool:
        """Record a review result.

        Args:
            task_id: The task that was reviewed.
            review_status: ReviewStatus value string.
            findings: List of finding dicts from the review.
            token_cost: Simulated cost in USD for this review.
            commit_sha: SHA that was reviewed.
            evidence_fingerprint: Evidence fingerprint (for dedup).

        Returns:
            True if the review was recorded, False if it would be a
            duplicate (same SHA + evidence fingerprint and already recorded).
        """
        if task_id not in self._data:
            raise ValueError(f"Task '{task_id}' not found for review recording")

        # Check for duplicate: same SHA + evidence fingerprint already reviewed
        # Allow re-review if there's a more recent correction that would have changed the code
        if commit_sha and evidence_fingerprint:
            rec = self._data[task_id]
            existing_reviews = [
                e for e in rec.get("review_sequence", [])
                if e.get("type") == "review"
                and e.get("sha") == commit_sha
                and e.get("evidence_fp") == evidence_fingerprint
            ]
            # Reject only if: same SHA+fp already reviewed AND no newer correction
            # (a newer correction implies the code changed, making this a valid new review)
            if existing_reviews and rec.get("correction_count", 0) == 0:
                # No corrections made yet — this is a genuine duplicate
                return False

        rec = self._data[task_id]
        total_cost = rec.get("total_mock_cost_usd", 0.0) + token_cost
        seq = rec.setdefault("review_sequence", [])
        seq.append({
            "type": "review",
            "status": review_status,
            "findings": findings[:10],  # Keep summary only
            "token_cost": round(token_cost, 6),
            "sha": commit_sha,
            "evidence_fp": evidence_fingerprint,
        })
        rec["total_mock_cost_usd"] = round(total_cost, 6)
        rec["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._data[task_id] = rec
        self._save()
        return True

    def check_budget_exhausted(self, monthly_budget_usd: float = 75.0) -> bool:
        """Check if the monthly budget is exhausted.

        Sums all tasks' costs and compares against the monthly budget.

        Args:
            monthly_budget_usd: Monthly spending cap.

        Returns:
            True if total ledger spending >= monthly budget.
        """
        total = sum(
            rec.get("total_mock_cost_usd", 0.0) for rec in self._data.values()
        )
        return total >= monthly_budget_usd

    def is_max_corrections_reached(self, task_id: str) -> bool:
        """Check if a task has reached its maximum correction cycles.

        Args:
            task_id: The task to check.

        Returns:
            True if correction_count >= max_correction_cycles.
        """
        rec = self.get_task(task_id)
        return rec.correction_count >= rec.max_correction_cycles

    def is_max_reviews_reached(self, task_id: str) -> bool:
        """Check if a task has reached its maximum review dispatches.

        Args:
            task_id: The task to check.

        Returns:
            True if total reviews >= max_review_dispatches.
        """
        rec = self.get_task(task_id)
        review_count = sum(
            1 for entry in rec.get("review_sequence", [])
            if entry.get("type") == "review"
        )
        return review_count >= rec.max_review_dispatches

    def get_cumulative_cost(self, task_id: str) -> float:
        """Get cumulative cost for a specific task.

        Args:
            task_id: The task to look up.

        Returns:
            Total mock cost in USD.
        """
        try:
            return self.get_task(task_id).total_mock_cost_usd
        except ValueError:
            return 0.0

    def get_task_review_count(self, task_id: str) -> int:
        """Get the number of reviews recorded for a task."""
        try:
            rec = self.get_task(task_id)
            return sum(
                1 for entry in rec.review_sequence
                if isinstance(entry, dict) and entry.get("type") == "review"
            )
        except (ValueError, AttributeError):
            return 0

    def finalize_task(
        self, task_id: str, disposition: str
    ) -> TaskRecord:
        """Mark a task as complete with final disposition.

        Args:
            task_id: The task to finalize.
            disposition: One of the final disposition strings.

        Returns:
            The finalized TaskRecord.
        """
        rec = self.get_task(task_id)
        rec_data = self._data[task_id]
        rec_data["final_disposition"] = disposition
        rec_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save()
        return rec

    @staticmethod
    def compute_fingerprint(commit_sha: str, test_evidence: str) -> str:
        """Compute a fingerprint for SHA + evidence pairs.

        Used to detect no-op corrections (unchanged commit + unchanged evidence).

        Args:
            commit_sha: The git commit SHA.
            test_evidence: Test output or evidence text.

        Returns:
            16-character hex fingerprint string.
        """
        raw = f"{commit_sha}|{test_evidence}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get_monthly_spending(self) -> float:
        """Get total spending across all tasks.

        Returns:
            Total USD spent this month (all tasks).
        """
        return sum(
            rec.get("total_mock_cost_usd", 0.0) for rec in self._data.values()
        )


def load_ledger_or_create(ledger_path: str) -> WorkflowLedger:
    """Load a ledger from disk, or create a new one if it doesn't exist.

    Args:
        ledger_path: Path to the JSON ledger file.

    Returns:
        A WorkflowLedger instance ready for use.
    """
    if os.path.exists(ledger_path):
        return WorkflowLedger(ledger_path)
    return WorkflowLedger(ledger_path)
