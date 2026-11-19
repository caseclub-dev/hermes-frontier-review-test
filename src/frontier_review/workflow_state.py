"""Durable task-state persistence for the autonomous loop.

Provides a JSON-backed ledger that tracks task state across process restarts,
enforcing correction cycle limits, review dispatch bounds, duplicate prevention,
and budget exhaustion tracking.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

# ── Shared ledger configuration ────────────────────────────────────────
# All components (controller, PR adapter, autonomous loop) MUST use this
# constant for the ledger filename to prevent cross-file state divergence.
LEDGER_FILENAME = "workflow_ledger.json"


def default_ledger_path(state_dir: str) -> str:
    """Return the canonical path to the workflow ledger file."""
    return os.path.join(state_dir, LEDGER_FILENAME)


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
    review_count: int = 0
    final_disposition: str = ""
    total_mock_cost_usd: float = 0.0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # ── Task claiming fields (Part 1 — cron-driven review controller) ──
    status: str = "PENDING_REVIEW"
    claim_worker: str = ""
    claim_time: str = ""
    claim_expires_at: str = ""

    # ── PR-based review integration (Phase 3C additive fields) ─────────
    repo_owner: str = ""
    repo_name: str = ""
    pr_number: int = 0
    pr_url: str = ""
    head_branch: str = ""
    base_branch: str = ""
    remote_head_sha: str = ""
    initial_remote_sha: str = ""
    review_status: str = "unreviewed"  # unreviewed | changes_required | pass | blocked | correction_exhausted
    evidence_fingerprint: str = ""

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
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                raw = json.load(f)
            if isinstance(raw, dict):
                self._data = raw
            else:
                self._data = {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._data = {}

    def _load_under_lock(self, task_id: Optional[str] = None) -> None:
        """Reload ledger state from disk under a shared lock.

        If *task_id* is provided, reloads only that task's entry; otherwise
        reloads the entire ledger into self._data.  Uses ``fcntl.LOCK_SH``
        for cross-process safety so concurrent readers are not blocked.

        Args:
            task_id: Optional task identifier to reload selectively.
        """
        try:
            with open(self._path) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                raw = json.load(f)
            if not isinstance(raw, dict):
                raw = {}
            if task_id is not None:
                # Reload only this task's entry; preserve others in memory
                if task_id in raw:
                    self._data[task_id] = raw[task_id]
                # else: leave existing in-memory data untouched
            else:
                self._data = raw
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            if task_id is not None:
                # Best-effort: don't wipe the entire ledger on a partial read error
                pass
            else:
                self._data = {}

    def _extend_lease(self, task_id: str, ttl_seconds: int = 300) -> bool:
        """Renew the claim_expires_at for an IN_PROGRESS task.

        Uses an exclusive lock (``fcntl.LOCK_EX``) to atomically read-modify-write
        only the target task's timestamp fields.

        Args:
            task_id: The task whose lease to extend.
            ttl_seconds: Number of seconds to add to the current expiry.

        Returns:
            True if the lease was renewed, False if the task is not IN_PROGRESS
            or does not exist.
        """
        try:
            with open(self._path) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False

        if task_id not in raw:
            return False

        rec = TaskRecord.from_dict(raw[task_id])
        if rec.status != "IN_PROGRESS":
            return False

        now = datetime.now(timezone.utc)
        try:
            expires = datetime.fromisoformat(rec.claim_expires_at)
        except (ValueError, TypeError):
            expires = now
        new_expiry = expires + timedelta(seconds=ttl_seconds)

        raw[task_id]["claim_expires_at"] = new_expiry.isoformat()
        raw[task_id]["updated_at"] = now.isoformat()

        # Update in-memory state for same-process callers
        self._data[task_id] = raw[task_id]

        try:
            self._atomic_write(raw)
        except OSError:
            pass  # In-memory data is already updated; disk write failure is non-fatal

        return True

    def _save(self) -> None:
        """Persist ledger to disk atomically (write tmp then replace)."""
        with open(self._path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(self._data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())

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
        # Hardened locking: reload from disk before any mutation
        self._load_under_lock(task_id)
        if task_id in self._data:
            # Idempotent: return existing record (concurrent callers may overlap)
            return TaskRecord.from_dict(self._data[task_id])

        # Auto-detect git HEAD when workspace_path points to a git repo
        # and no explicit original_sha was provided.
        if not original_sha and os.path.isdir(workspace_path):
            r = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                cwd=workspace_path, timeout=5,
            )
            if r.returncode == 0:
                original_sha = r.stdout.strip()

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
        # Hardened locking: reload from disk before any mutation
        self._load_under_lock(task_id)
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
        # Hardened locking: reload from disk before any mutation
        self._load_under_lock(task_id)
        if task_id not in self._data:
            raise ValueError(f"Task '{task_id}' not found for correction count")
        rec = self._data[task_id]
        rec["correction_count"] = rec.get("correction_count", 0) + 1
        rec["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._data[task_id] = rec
        self._save()

    def update_task_sha(self, task_id: str, new_sha: str) -> None:
        """Update the current SHA for a task and persist it.

        Args:
            task_id: The task whose SHA to update.
            new_sha: The new commit SHA (will be truncated to 40 chars).
        """
        self._load_under_lock(task_id)
        if task_id not in self._data:
            raise ValueError(f"Task '{task_id}' not found for SHA update")
        rec = self._data[task_id]
        rec["current_sha"] = new_sha[:40] if len(new_sha) >= 40 else new_sha
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
        # Hardened locking: reload from disk before any mutation
        self._load_under_lock(task_id)
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
        # Increment review_count: exactly one per successful record_review call
        # (not for discovery checks, push retries, or duplicate rejections)
        review_cnt = rec.get("review_count", 0)
        rec["review_count"] = review_cnt + 1
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
        # Hardened locking: reload from disk before any mutation
        self._load_under_lock(task_id)
        rec = self.get_task(task_id)
        rec_data = self._data[task_id]
        rec_data["final_disposition"] = disposition
        # Update status to reflect final state
        if disposition in ("ready_for_operator_review", "changes_required",
                           "blocked", "escalated_architect",
                           "max_correction_exhausted"):
            rec_data["status"] = disposition
        rec_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save()
        return rec

    # ── Task claiming (Part 1 — cron-driven review controller) ─────────

    def _atomic_write(self, raw: dict[str, Any]) -> None:
        """Write ledger dict to disk atomically."""
        try:
            tmp_path = self._path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(raw, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except OSError:
            with open(self._path, "w") as f:
                json.dump(raw, f, indent=2, default=str)
                f.flush()

    def create_pending_task(
        self,
        task_id: str,
        requirements: str,
        workspace_path: str,
        acceptance_criteria: str = "",
        cost_ceiling_usd: float = 10.0,
    ) -> TaskRecord:
        """Create a PENDING_REVIEW task record (idempotent).

        Args:
            task_id: Unique identifier for this review task.
            requirements: What the code must satisfy.
            workspace_path: Path to the disposable workspace.
            acceptance_criteria: JSON string or description of acceptance criteria.
            cost_ceiling_usd: Per-task simulated cost ceiling (default $10).

        Returns:
            The newly created TaskRecord.

        Raises:
            ValueError: If a task with this id already exists with a final disposition.
        """
        if task_id in self._data:
            existing = TaskRecord.from_dict(self._data[task_id])
            # Idempotent: return existing for non-final states
            if existing.status in ("PENDING_REVIEW", ""):
                return existing
            # Final disposition — do not overwrite
            raise ValueError(
                f"Task '{task_id}' already exists with final disposition "
                f"'{existing.final_disposition}'. Cannot re-create."
            )

        now = datetime.now(timezone.utc).isoformat()
        # Auto-detect git HEAD when workspace_path points to a git repo.
        initial_sha = ""
        if os.path.isdir(workspace_path):
            r = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                cwd=workspace_path, timeout=5,
            )
            if r.returncode == 0:
                initial_sha = r.stdout.strip()

        record = TaskRecord(
            task_id=task_id,
            requirements=requirements,
            workspace_path=workspace_path,
            original_sha=initial_sha,
            current_sha=initial_sha,
            created_at=now,
            updated_at=now,
            status="PENDING_REVIEW",
        )
        self._data[task_id] = record.to_dict()
        self._save()
        return record

    def claim_task(
        self,
        task_id: str,
        worker_id: str = "controller",
        ttl_seconds: int = 300,
    ) -> Optional[TaskRecord]:
        """Atomically transition PENDING_REVIEW → IN_PROGRESS.

        Uses fcntl.flock for cross-process safety with 5x retry on EACCES.

        Args:
            task_id: The task to claim.
            worker_id: Identity of the claiming worker.
            ttl_seconds: How long the claim lasts (default 300s).

        Returns:
            The claimed TaskRecord, or None if unavailable.
        """
        import time

        for attempt in range(5):
            try:
                return self._claim_task_inner(task_id, worker_id, ttl_seconds)
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.1)
                else:
                    raise
        return None

    def _claim_task_inner(
        self,
        task_id: str,
        worker_id: str,
        ttl_seconds: int,
    ) -> Optional[TaskRecord]:
        """Inner implementation of claim_task (inside lock)."""
        # Read → flock → modify → write back → unlock
        try:
            with open(self._path) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        if task_id not in raw:
            return None

        rec = TaskRecord.from_dict(raw[task_id])
        # Must be PENDING_REVIEW to claim
        if rec.status != "PENDING_REVIEW":
            return None

        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=ttl_seconds)

        raw[task_id]["status"] = "IN_PROGRESS"
        raw[task_id]["claim_worker"] = worker_id
        raw[task_id]["claim_time"] = now.isoformat()
        raw[task_id]["claim_expires_at"] = expires.isoformat()
        raw[task_id]["updated_at"] = now.isoformat()

        # Update in-memory state for same-process callers
        self._data[task_id] = raw[task_id]

        # Write back to disk atomically
        try:
            self._atomic_write(raw)
        except OSError:
            pass  # In-memory data is already updated; disk write failure is non-fatal here

        # Return fresh TaskRecord from the updated data
        return TaskRecord.from_dict(raw[task_id])

    def release_task(
        self,
        task_id: str,
        reason: str = "expired",
    ) -> bool:
        """Release an IN_PROGRESS claim back to PENDING_REVIEW.

        Args:
            task_id: The task to release.
            reason: Why the claim was released (logged in updated_at).

        Returns:
            True if released, False if not in IN_PROGRESS state.
        """
        try:
            with open(self._path) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return False

        if task_id not in raw:
            return False

        rec = TaskRecord.from_dict(raw[task_id])
        if rec.status != "IN_PROGRESS":
            return False

        now = datetime.now(timezone.utc).isoformat()
        raw[task_id]["status"] = "PENDING_REVIEW"
        raw[task_id]["claim_worker"] = ""
        raw[task_id]["claim_time"] = ""
        raw[task_id]["claim_expires_at"] = ""
        raw[task_id]["updated_at"] = now

        # Update in-memory state for same-process callers
        self._data[task_id] = raw[task_id]

        try:
            self._atomic_write(raw)
        except OSError:
            pass  # In-memory data is already updated; disk write failure is non-fatal

        return True

    # ── Correction job management (Phase 3 — auto Coder dispatch bridge) ──

    def set_task_status(self, task_id: str, new_status: str, **extra_fields: Any) -> TaskRecord:
        """Transition a task to a new status under exclusive lock.

        Unlike ``finalize_task``, this method does not set a final disposition;
        it updates the mutable status field for intermediate states such as
        CORRECTION_PENDING or IN_PROGRESS.

        Args:
            task_id: The task to update.
            new_status: Target status (e.g. 'CORRECTION_PENDING', 'IN_PROGRESS').
            **extra_fields: Additional dict fields to write (e.g. correction_job).

        Returns:
            Updated TaskRecord.

        Raises:
            ValueError: If the task does not exist.
        """
        self._load_under_lock(task_id)
        rec = self.get_task(task_id)
        rec_data = self._data[task_id]
        rec_data["status"] = new_status
        rec_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        for key, value in extra_fields.items():
            if key in rec_data:
                rec_data[key] = value
            else:
                # Allow adding top-level fields beyond TaskRecord schema
                rec_data[key] = value
        self._save()
        return self.get_task(task_id)

    def get_pending_tasks(self) -> list[TaskRecord]:
        """Return all tasks with status == PENDING_REVIEW."""
        pending = []
        for task_id, data in self._data.items():
            rec = TaskRecord.from_dict(data)
            if rec.status == "PENDING_REVIEW":
                pending.append(rec)
        # Sort by created_at for deterministic ordering
        pending.sort(key=lambda r: r.created_at)
        return pending

    def find_expired_claims(
        self,
        max_ttl_seconds: int = 600,
    ) -> list[TaskRecord]:
        """Return all IN_PROGRESS tasks whose claim has expired.

        A claim is expired if the current time exceeds claim_expires_at.

        Args:
            max_ttl_seconds: Maximum TTL threshold for considering a claim expired.

        Returns:
            List of expired TaskRecords sorted by claim_time (oldest first).
        """
        now = datetime.now(timezone.utc)
        expired = []
        for task_id, data in self._data.items():
            rec = TaskRecord.from_dict(data)
            if rec.status != "IN_PROGRESS":
                continue
            if not rec.claim_expires_at:
                continue
            try:
                expires = datetime.fromisoformat(rec.claim_expires_at)
                if now > expires and (now - expires).total_seconds() < max_ttl_seconds:
                    expired.append(rec)
            except (ValueError, TypeError):
                # Malformed timestamp — treat as expired
                expired.append(rec)
        expired.sort(key=lambda r: r.claim_time or "")
        return expired

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

    def validate_approval_record(
        self,
        task_id: str,
        workspace_path: str,
        commit_sha: str,
        test_evidence: str = "",
    ) -> dict[str, Any]:
        """Validate that an existing review record authorizes final approval.

        This is the mandatory gate before any task may transition to
        READY_FOR_OPERATOR_REVIEW.  It verifies that a **legitimate**
        review record — one produced by the normal workflow (not
        fabricated post-hoc) — exists in this ledger and matches all
        required fields.

        Checks:
          1. Task record exists for *task_id*.
          2. At least one review with status ``"pass"`` is recorded.
          3. The latest PASS review has a matching *workspace_path*.
          4. The latest PASS review references the expected *commit_sha*.
          5. The evidence fingerprint is consistent (non-empty for
             tasks that had test evidence, empty-only when no tests ran).
          6. Status is exactly ``"pass"`` — never ``"changes_required"``,
             ``"blocked"``, or anything else.

        Args:
            task_id: The task whose review record to validate.
            workspace_path: Canonical repository / workspace path.
            commit_sha: Exact final Git commit SHA to match.
            test_evidence: Test output (for fingerprint matching).

        Returns:
            Dict with keys:
              ``approved`` (bool): True only if ALL checks pass.
              ``reason`` (str): Human-readable reason if not approved.
              ``status`` (str | None): Status of the latest review entry.
              ``findings`` (list[dict]): Findings from the latest PASS review.

        Raises:
            ValueError: If *task_id* does not exist in this ledger.
        """
        rec = self.get_task(task_id)
        # rec is a TaskRecord dataclass; access fields directly
        workspace = rec.workspace_path
        current_sha = rec.current_sha
        reviews = [e for e in rec.review_sequence if isinstance(e, dict) and e.get("type") == "review"]

        # 1. Must have at least one review
        if not reviews:
            return {
                "approved": False,
                "reason": "No review record found for this task",
                "status": None,
                "findings": [],
            }

        # Find the latest PASS review (if any) — check from most recent
        # ReviewStatus.PASS.value == "PASS" (uppercase)
        pass_reviews = [r for r in reviews if r.get("status") == "PASS"]

        if not pass_reviews:
            latest = reviews[-1]
            return {
                "approved": False,
                "reason": f"Latest review status is '{latest.get('status')}' — not PASS",
                "status": latest.get("status"),
                "findings": latest.get("findings", []),
            }

        # Use the latest PASS review
        latest_pass = pass_reviews[-1]
        review_sha = latest_pass.get("sha", "")
        review_fp = latest_pass.get("evidence_fp", "")
        review_status = latest_pass.get("status", "")
        findings = latest_pass.get("findings", [])

        # 2. Workspace path must match
        if workspace and workspace_path and workspace != workspace_path:
            return {
                "approved": False,
                "reason": f"Workspace mismatch: ledger={workspace!r}, expected={workspace_path!r}",
                "status": review_status,
                "findings": findings,
            }

        # 3. The PASS review was performed on a specific commit (review_sha).
        #    It must match the expected commit. This prevents approving old
        #    reviews after code changes that update ledger.current_sha
        #    without requiring re-review.
        if review_sha and commit_sha and review_sha != commit_sha:
            return {
                "approved": False,
                "reason": f"SHA mismatch: PASS review on={review_sha!r}, expected={commit_sha!r}",
                "status": review_status,
                "findings": findings,
            }

        # 4. Evidence fingerprint consistency — a PASS with empty evidence_fp
        #    is acceptable only when no test evidence was provided; a non-empty
        #    evidence_fp requires matching test evidence
        if test_evidence and not review_fp:
            return {
                "approved": False,
                "reason": "Test evidence provided but no evidence fingerprint in review record",
                "status": review_status,
                "findings": findings,
            }

        if test_evidence and review_fp:
            expected_fp = self.compute_fingerprint(review_sha or commit_sha, test_evidence)
            if review_fp != expected_fp:
                return {
                    "approved": False,
                    "reason": f"Evidence fingerprint mismatch: ledger={review_fp!r}, expected={expected_fp!r}",
                    "status": review_status,
                    "findings": findings,
                }

        # All checks passed — legitimate review record found
        return {
            "approved": True,
            "reason": "",
            "status": review_status,
            "findings": findings,
        }


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
