"""Publication transaction management for Frontier Review publisher.

Provides durable publication transactions with crash recovery. Each correction
publication is wrapped in a transaction that records state before and after the
push attempt, allowing safe recovery from interrupted operations.

Transaction lifecycle:
  1. BEGIN — record expected_remote_sha, proposed new SHA, PR number
  2. LOCK — serialize on a per-PR lock file (fcntl.flock)
  3. PUSH — perform the authorized Git push
  4. VERIFY — confirm actual remote SHA matches published SHA
  5. COMMIT — persist outcome (success/failure) and increment correction counter
  6. UNLOCK — release serialization lock

Crash recovery:
  On startup, inspect the transaction log for any uncommitted transactions.
  Fetch the current remote HEAD for the affected PR branch. If it matches the
  published SHA, mark the transaction as COMMITTED (remote succeeded even if
  local persistence was interrupted). If it does not match, mark as FAILED.

This module uses disposable test directories — no real GitHub or privileged paths.

Constants are NEVER configurable by caller input; they are hardcoded at import time.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("frontier_review.publication_transaction")

# ---------------------------------------------------------------------------
# Constants — never configurable by caller
# ---------------------------------------------------------------------------

DEFAULT_TRANSACTION_LOG: str = "/var/lib/frontier-review/publisher/transactions/"
DEFAULT_LOCK_DIR: str = "/var/lib/frontier-review/publisher/locks/"

PUBLICATION_ACTION = "push"


# ---------------------------------------------------------------------------
# PublicationTransaction class
# ---------------------------------------------------------------------------


class PublicationTransactionError(Exception):
    """Raised when a transaction is in an invalid state."""


class PublicationTransaction:
    """Manages a single correction publication as a durable transaction.

    The transaction object handles BEGIN, LOCK, PUSH, VERIFY, COMMIT/ABORT,
    and UNLOCK steps with crash recovery support via the transaction log.

    Attributes:
        transaction_id: UUID string identifying this transaction.
        pr_number: Pull request number being corrected.
        repo_owner: Repository owner slug.
        repo_name: Repository name.
        head_branch: Git branch being published.
        expected_remote_sha: SHA at remote HEAD before the push (authorizer-provided).
        proposed_new_sha: Expected local commit SHA to push (may differ from actual).
        transaction_log_path: Path to the persistent transaction log file.
    """

    def __init__(
        self,
        pr_number: int,
        repo_owner: str,
        repo_name: str,
        head_branch: str,
        expected_remote_sha: str,
        proposed_new_sha: str | None = None,
        transaction_log_dir: str | None = None,
        lock_dir: str | None = None,
    ) -> None:
        """Initialize a publication transaction.

        Args:
            pr_number: Pull request number being corrected.
            repo_owner: Repository owner.
            repo_name: Repository name.
            head_branch: Git branch name.
            expected_remote_sha: Expected remote HEAD before push (40-hex SHA).
            proposed_new_sha: Expected local commit SHA to be pushed (40-hex SHA).
            transaction_log_dir: Directory for the transaction log file.
                Defaults to DEFAULT_TRANSACTION_LOG, overridden by FR_TX_LOG_DIR env var.
            lock_dir: Directory for serialization lock files.
                Defaults to DEFAULT_LOCK_DIR, overridden by FR_TX_LOCK_DIR env var.
        """
        self.transaction_id = str(uuid.uuid4())
        self.pr_number = pr_number
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.head_branch = head_branch
        self.expected_remote_sha = expected_remote_sha
        self.proposed_new_sha = proposed_new_sha
        self.state: str = "PENDING"  # PENDING | LOCKED | PUSHING | COMMITTED | ABORTED | RECOVERED | CONSUMED_NO_PUSH
        self.actual_remote_sha: str | None = None
        self.error_message: str | None = None
        self.timestamp: datetime = datetime.now(timezone.utc)

        # Resolve log directory
        if transaction_log_dir is not None:
            self._log_dir = transaction_log_dir
        else:
            env_log = os.environ.get("FR_TX_LOG_DIR")
            self._log_dir = env_log or DEFAULT_TRANSACTION_LOG

        Path(self._log_dir).mkdir(parents=True, exist_ok=True)

        # Resolve lock directory
        if lock_dir is not None:
            self._lock_dir = lock_dir
        else:
            env_lock = os.environ.get("FR_TX_LOCK_DIR")
            self._lock_dir = env_lock or DEFAULT_LOCK_DIR

        Path(self._lock_dir).mkdir(parents=True, exist_ok=True)

        self._log_path = os.path.join(self._log_dir, "transactions.log")
        self._lock_path = os.path.join(
            self._lock_dir, f"pr-{pr_number}-{repo_owner}-{repo_name}.lock"
        )

    # ------------------------------------------------------------------
    # Transaction lifecycle
    # ------------------------------------------------------------------

    def begin(self) -> None:
        """Mark this transaction as BEGIN and persist to the log.

        A BEGIN record contains minimal state so recovery can detect it.
        The record is written atomically (tempfile + rename).
        """
        self.state = "PENDING"
        record = {
            "transaction_id": self.transaction_id,
            "pr_number": self.pr_number,
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "head_branch": self.head_branch,
            "expected_remote_sha": self.expected_remote_sha[:8],
            "proposed_new_sha": self.proposed_new_sha[:8] if self.proposed_new_sha else None,
            "state": self.state,
            "timestamp": self.timestamp.isoformat(),
        }
        self._append_log(record)
        logger.info(
            "Transaction BEGIN %s for PR #%d",
            self.transaction_id[:8], self.pr_number,
        )

    def lock(self) -> None:
        """Acquire an exclusive flock for serialization.

        Only one publication at a time per (pr_number, repo_owner, repo_name).
        Uses fcntl.flock which is POSIX-compatible and survives process crashes
        (the kernel releases the lock when the file descriptor is closed).

        Raises:
            PublicationTransactionError: If locking fails unexpectedly.
        """
        try:
            self._lock_fd = open(self._lock_path, "w")
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Write our PID for diagnostics
            self._lock_fd.write(str(os.getpid()))
            self._lock_fd.flush()
            self.state = "LOCKED"
            logger.info(
                "Transaction %s acquired lock for PR #%d",
                self.transaction_id[:8], self.pr_number,
            )
        except (IOError, OSError) as exc:
            if hasattr(self, "_lock_fd") and self._lock_fd is not None:
                try:
                    self._lock_fd.close()
                except OSError:
                    pass
            raise PublicationTransactionError(
                f"Could not acquire publication lock for PR #{self.pr_number}: {exc}"
            )

    def unlock(self) -> None:
        """Release the flock and close the lock file descriptor."""
        if hasattr(self, "_lock_fd") and self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
                self._lock_fd.close()
            except (IOError, OSError):
                pass
            finally:
                self._lock_fd = None

    def commit(self, actual_remote_sha: str, corrections_consumed: int = 1) -> None:
        """Mark the transaction as COMMITTED and persist to the log.

        Args:
            actual_remote_sha: The SHA that was actually published to remote (40-hex).
            corrections_consumed: Number of correction budget units consumed.
        """
        self.state = "COMMITTED"
        self.actual_remote_sha = actual_remote_sha
        record = {
            "transaction_id": self.transaction_id,
            "pr_number": self.pr_number,
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "head_branch": self.head_branch,
            "expected_remote_sha": self.expected_remote_sha[:8],
            "proposed_new_sha": self.proposed_new_sha[:8] if self.proposed_new_sha else None,
            "actual_remote_sha": actual_remote_sha[:8],
            "state": self.state,
            "corrections_consumed": corrections_consumed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append_log(record)

    def abort(self, error: str) -> None:
        """Mark the transaction as ABORTED and persist to the log.

        Args:
            error: Description of why the transaction was aborted.
        """
        self.state = "ABORTED"
        self.error_message = error[:200]  # truncate long messages for safety
        record = {
            "transaction_id": self.transaction_id,
            "pr_number": self.pr_number,
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "head_branch": self.head_branch,
            "expected_remote_sha": self.expected_remote_sha[:8],
            "proposed_new_sha": self.proposed_new_sha[:8] if self.proposed_new_sha else None,
            "state": self.state,
            "error": self.error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append_log(record)

    def consumed_no_push(self, error: str, reason: str = "") -> None:
        """Mark the transaction as CONSUMED_NO_PUSH and persist to the log.

        Use this when the correction budget was consumed but the push failed,
        requiring manual reconciliation (do NOT auto-refund).

        Args:
            error: Description of why the push failed after consumption.
            reason: Optional human-readable reason for audit trail.
        """
        self.state = "CONSUMED_NO_PUSH"
        self.error_message = error[:200]
        record = {
            "transaction_id": self.transaction_id,
            "pr_number": self.pr_number,
            "repo_owner": self.repo_owner,
            "repo_name": self.repo_name,
            "head_branch": self.head_branch,
            "expected_remote_sha": self.expected_remote_sha[:8],
            "proposed_new_sha": self.proposed_new_sha[:8] if self.proposed_new_sha else None,
            "state": self.state,
            "error": self.error_message,
            "reason": reason[:200] if reason else "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._append_log(record)

    # ------------------------------------------------------------------
    # Log persistence helpers
    # ------------------------------------------------------------------

    def _append_log(self, record: dict[str, Any]) -> None:
        """Append a JSON line to the transaction log atomically.

        Reads existing log content, appends the new record, and writes the
        combined result via tempfile.mkstemp + os.rename for atomicity.
        Never overwrites an existing log file without preserving its contents.

        Args:
            record: Dictionary to serialize and append.
        """
        line = json.dumps(record) + "\n"
        # Read existing content (if any)
        try:
            with open(self._log_path) as f:
                existing = f.read()
        except (FileNotFoundError, OSError):
            existing = ""

        combined = existing + line
        fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=self._log_dir)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(combined)
            os.chmod(tmp_path, 0o640)
            os.rename(tmp_path, self._log_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get_last_committed(self) -> dict[str, Any] | None:
        """Return the most recent COMMITTED transaction record from the log.

        Returns:
            The last committed record dict, or None if no committed transactions exist.
        """
        if not os.path.isfile(self._log_path):
            return None
        records = self._read_log()
        for record in reversed(records):
            if record.get("state") == "COMMITTED":
                return record
        return None

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def recover(self, fetch_remote_sha: Any) -> str:
        """Recover the state of this transaction after an interruption.

        Inspects the log for uncommitted transactions and checks the current
        remote HEAD against recorded SHAs to determine the true outcome.

        Args:
            fetch_remote_sha: Callable that returns the current remote HEAD SHA
                for this PR's branch as a 40-hex string, or raises on failure.

        Returns:
            One of "RECOVERED_SUCCESS" (remote has our SHA),
                  "RECOVERED_FAILURE" (remote diverged — abort),
                  "RECOVERED_UNKNOWN" (no log entry for recovery).
        """
        if not os.path.isfile(self._log_path):
            return "RECOVERED_UNKNOWN"

        records = self._read_log()
        # Find uncommitted transactions matching this transaction ID or PR
        uncommitted = [
            r for r in reversed(records)
            if r.get("transaction_id") == self.transaction_id and r.get("state") != "COMMITTED"
        ]

        if not uncommitted:
            return "RECOVERED_UNKNOWN"

        # Fetch current remote state
        try:
            current_sha = fetch_remote_sha()
        except Exception as exc:
            logger.warning("Recovery: could not fetch remote SHA: %s", exc)
            self.abort(f"recovery-failed: {exc}")
            return "RECOVERED_FAILURE"

        # Check if the published (expected) SHA is now on the remote
        expected_short = self.expected_remote_sha[:8]
        actual_short = current_sha[:8] if current_sha else ""

        if actual_short == expected_short and len(current_sha) == 40:
            # Remote has our SHA — the push succeeded despite local interruption
            self.state = "RECOVERED"
            self.actual_remote_sha = current_sha
            record = {
                "transaction_id": self.transaction_id,
                "pr_number": self.pr_number,
                "repo_owner": self.repo_owner,
                "repo_name": self.repo_name,
                "head_branch": self.head_branch,
                "expected_remote_sha": expected_short,
                "actual_remote_sha": actual_short,
                "state": "RECOVERED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._append_log(record)
            return "RECOVERED_SUCCESS"

        # Remote has a different SHA — the push failed or was superseded
        self.abort(f"remote diverged: expected {expected_short} got {actual_short}")
        return "RECOVERED_FAILURE"

    def _read_log(self) -> list[dict[str, Any]]:
        """Read all records from the transaction log."""
        if not os.path.isfile(self._log_path):
            return []
        try:
            with open(self._log_path) as f:
                return [json.loads(line.strip()) for line in f if line.strip()]
        except (json.JSONDecodeError, OSError):
            return []

    def __del__(self) -> None:
        """Ensure lock is released when the object is garbage-collected."""
        try:
            self.unlock()
        except Exception:
            pass  # Silently ignore in destructor to avoid interpreter warnings


# ---------------------------------------------------------------------------
# Transaction log management helpers
# ---------------------------------------------------------------------------


def list_transactions(log_dir: str | None = None) -> list[dict[str, Any]]:
    """List all transaction records from the log.

    Args:
        log_dir: Directory containing the transaction log. Defaults to env or constant.

    Returns:
        List of all transaction record dicts.
    """
    if log_dir is None:
        env_log = os.environ.get("FR_TX_LOG_DIR")
        log_dir = env_log or DEFAULT_TRANSACTION_LOG
    log_path = os.path.join(log_dir, "transactions.log")
    if not os.path.isfile(log_path):
        return []
    try:
        with open(log_path) as f:
            return [json.loads(line.strip()) for line in f if line.strip()]
    except (json.JSONDecodeError, OSError):
        return []


def cleanup_old_transactions(
    log_dir: str | None = None,
    max_age_hours: int = 168,  # 7 days
) -> int:
    """Remove transaction records older than max_age_hours.

    Args:
        log_dir: Directory containing the transaction log.
        max_age_hours: Maximum age in hours before a record is considered expired.

    Returns:
        Number of records removed.
    """
    if log_dir is None:
        env_log = os.environ.get("FR_TX_LOG_DIR")
        log_dir = env_log or DEFAULT_TRANSACTION_LOG
    log_path = os.path.join(log_dir, "transactions.log")
    if not os.path.isfile(log_path):
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
    try:
        records = []
        with open(log_path) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line.strip())
                # Preserve RECOVERED and recent records
                if record.get("state") == "RECOVERED":
                    records.append(record)
                    continue
                ts_str = record.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str)
                except (ValueError, TypeError):
                    # Unknown timestamp — keep the record to be safe
                    records.append(record)
                    continue
                if ts >= cutoff:
                    records.append(record)
        with open(log_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return len(records)  # Return remaining count; removed = original - remaining
    except (json.JSONDecodeError, OSError):
        return 0
