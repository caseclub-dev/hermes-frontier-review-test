"""Tests for the publication transaction module.

Covers:
  - Transaction lifecycle (BEGIN, COMMIT, ABORT)
  - Atomic log persistence
  - Crash recovery from interrupted transactions
  - Remote SHA reconciliation after interruption
  - Concurrent lock serialization
  - Old record cleanup
  - Unknown/missing state handling
"""

import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

sys_path = os.path.join(os.path.dirname(__file__), "..", "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from frontier_review.publication_transaction import (  # noqa: E402
    PublicationTransaction,
    PublicationTransactionError,
    cleanup_old_transactions,
    list_transactions,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _setup_dirs(tmp_path: Path) -> tuple[str, str]:
    """Create temporary transaction log dir and lock dir.

    Returns:
        Tuple of (log_dir, lock_dir).
    """
    log_dir = str(tmp_path / "tx_log")
    lock_dir = str(tmp_path / "locks")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(lock_dir, exist_ok=True)
    return log_dir, lock_dir


def _mock_bare_repo_sha(bare_repo: Path, branch: str) -> str | None:
    """Return the current HEAD SHA of a branch from a mock bare repo.

    Returns None if the branch does not exist or cannot be read.
    """
    r = subprocess.run(
        ["git", "ls-remote", str(bare_repo), f"refs/heads/{branch}"],
        capture_output=True, text=True, timeout=10,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return None
    return r.stdout.strip().split()[0]


def _make_env(tmp_path: Path) -> dict[str, str]:
    """Return env with FR_TX_* vars pointing to temp dirs."""
    log_dir, lock_dir = _setup_dirs(tmp_path)
    return {**os.environ, "FR_TX_LOG_DIR": log_dir, "FR_TX_LOCK_DIR": lock_dir}


# ------------------------------------------------------------------
# Test: basic lifecycle
# ------------------------------------------------------------------


class TestTransactionLifecycle:
    """Test BEGIN, COMMIT, ABORT with atomic log persistence."""

    def test_begin_creates_log_entry(self, tmp_path: Path) -> None:
        """BEGIN writes a PENDING record to the transaction log."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        tx = PublicationTransaction(1, "owner", "repo", "hermes/test-1", "a" * 40)
        tx.begin()

        records = list_transactions()
        assert len(records) == 1
        assert records[0]["state"] == "PENDING"
        assert records[0]["transaction_id"] == tx.transaction_id
        assert records[0]["expected_remote_sha"] == ("a" * 40)[:8]

    def test_commit_records_success(self, tmp_path: Path) -> None:
        """COMMIT writes a COMMITTED record with the actual SHA."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        tx = PublicationTransaction(1, "owner", "repo", "hermes/test-2", "b" * 40)
        tx.begin()
        tx.commit(actual_remote_sha="c" * 40, corrections_consumed=1)

        records = list_transactions()
        assert len(records) == 2
        last = records[-1]
        assert last["state"] == "COMMITTED"
        assert last["actual_remote_sha"] == ("c" * 40)[:8]

    def test_abort_records_failure(self, tmp_path: Path) -> None:
        """ABORT writes an ABORTED record with error message."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        tx = PublicationTransaction(1, "owner", "repo", "hermes/test-3", "d" * 40)
        tx.begin()
        tx.abort("remote SHA mismatch")

        records = list_transactions()
        last = records[-1]
        assert last["state"] == "ABORTED"
        assert "remote SHA mismatch" in last.get("error", "")

    def test_atomic_log_write(self, tmp_path: Path) -> None:
        """Each log write creates a new file via tempfile+rename (atomic)."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        tx = PublicationTransaction(1, "owner", "repo", "hermes/test-4", "e" * 40)
        tx.begin()

        log_dir = os.environ["FR_TX_LOG_DIR"]
        # After begin(), there should be exactly one .json file (the log)
        files = list(Path(log_dir).glob("*.json")) + list(Path(log_dir).glob("transactions.log*"))
        assert len(files) == 1

    def test_file_permissions_640(self, tmp_path: Path) -> None:
        """Transaction log file has 0o640 permissions."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        tx = PublicationTransaction(1, "owner", "repo", "hermes/test-5", "f" * 40)
        tx.begin()

        log_dir = os.environ["FR_TX_LOG_DIR"]
        log_path = Path(log_dir) / "transactions.log"
        assert log_path.exists()
        mode = oct(log_path.stat().st_mode & 0o777)
        assert mode == "0o640", f"Expected 0o640, got {mode}"

    def test_preserves_existing_records(self, tmp_path: Path) -> None:
        """New transactions are appended without destroying old records."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        tx1 = PublicationTransaction(1, "owner", "repo", "hermes/test-6a", "g" * 40)
        tx1.begin()
        tx2 = PublicationTransaction(1, "owner", "repo", "hermes/test-6b", "h" * 40)
        tx2.begin()

        records = list_transactions()
        assert len(records) == 2
        assert records[0]["transaction_id"] != records[1]["transaction_id"]


# ------------------------------------------------------------------
# Test: crash recovery
# ------------------------------------------------------------------


class TestCrashRecovery:
    """Test recovery from interrupted transactions."""

    def test_recovery_unknown_no_log(self, tmp_path: Path) -> None:
        """No log → RECOVERED_UNKNOWN."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        tx = PublicationTransaction(1, "owner", "repo", "hermes/test-7", "i" * 40)
        result = tx.recover(lambda: "j" * 40)
        assert result == "RECOVERED_UNKNOWN"

    def test_recovery_success_sha_matches(self, tmp_path: Path) -> None:
        """Remote SHA matches expected → RECOVERED_SUCCESS."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        tx = PublicationTransaction(1, "owner", "repo", "hermes/test-8", "k" * 40)
        tx.begin()
        # Simulate interruption: remote succeeded but commit was never recorded
        result = tx.recover(lambda: "k" * 40)
        assert result == "RECOVERED_SUCCESS"

        records = list_transactions()
        last = records[-1]
        assert last["state"] == "RECOVERED"
        assert last["actual_remote_sha"] == ("k" * 40)[:8]

    def test_recovery_failure_sha_mismatch(self, tmp_path: Path) -> None:
        """Remote SHA differs → RECOVERED_FAILURE (abort)."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        tx = PublicationTransaction(1, "owner", "repo", "hermes/test-9", "l" * 40)
        tx.begin()
        # Remote has diverged
        result = tx.recover(lambda: "m" * 40)
        assert result == "RECOVERED_FAILURE"

        records = list_transactions()
        last = records[-1]
        assert last["state"] == "ABORTED"

    def test_recovery_fetch_fails(self, tmp_path: Path) -> None:
        """Remote fetch raises → RECOVERED_FAILURE (abort)."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        tx = PublicationTransaction(1, "owner", "repo", "hermes/test-10", "n" * 40)
        tx.begin()
        result = tx.recover(lambda: (_ for _ in ()).throw(ConnectionError("network down")))
        assert result == "RECOVERED_FAILURE"

        records = list_transactions()
        last = records[-1]
        assert last["state"] == "ABORTED"


# ------------------------------------------------------------------
# Test: lock serialization
# ------------------------------------------------------------------


class TestLockSerialization:
    """Test that publication locks are properly serialized."""

    def test_lock_acquire_release(self, tmp_path: Path) -> None:
        """Lock can be acquired and released successfully."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        tx = PublicationTransaction(1, "owner", "repo", "hermes/test-11", "o" * 40)
        tx.begin()
        tx.lock()
        assert tx.state == "LOCKED"
        tx.unlock()
        # After unlock, lock file should be closed

    def test_lock_blocking_on_contention(self, tmp_path: Path) -> None:
        """Second lock acquisition raises when first holder holds it."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        tx1 = PublicationTransaction(1, "owner", "repo", "hermes/test-12a", "p" * 40)
        tx2 = PublicationTransaction(1, "owner", "repo", "hermes/test-12b", "q" * 40)

        tx1.begin()
        tx1.lock()

        with pytest.raises(PublicationTransactionError):
            tx2.lock()

        tx1.unlock()


# ------------------------------------------------------------------
# Test: cleanup
# ------------------------------------------------------------------


class TestCleanup:
    """Test old transaction record cleanup."""

    def test_cleanup_removes_old_records(self, tmp_path: Path) -> None:
        """Records older than max_age_hours are removed."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        # Create a record with an old timestamp by writing directly
        log_dir = env["FR_TX_LOG_DIR"]
        old_ts = "2020-01-01T00:00:00+00:00"
        log_path = Path(log_dir) / "transactions.log"
        with open(log_path, "w") as f:
            f.write(json.dumps({
                "transaction_id": "old-id",
                "state": "COMMITTED",
                "timestamp": old_ts,
            }) + "\n")
            f.write(json.dumps({
                "transaction_id": "new-id",
                "state": "PENDING",
                "timestamp": "2099-01-01T00:00:00+00:00",
            }) + "\n")

        removed = cleanup_old_transactions(log_dir=log_dir, max_age_hours=24)
        # Returns remaining count; old record should be gone, new stays
        records = list_transactions(log_dir=log_dir)
        assert len(records) == 1
        assert records[0]["transaction_id"] == "new-id"

    def test_cleanup_preserves_recovered(self, tmp_path: Path) -> None:
        """RECOVERED state records are never removed by cleanup."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        old_ts = "2020-01-01T00:00:00+00:00"
        log_dir = env["FR_TX_LOG_DIR"]
        log_path = Path(log_dir) / "transactions.log"
        with open(log_path, "w") as f:
            f.write(json.dumps({
                "transaction_id": "recovered-id",
                "state": "RECOVERED",
                "timestamp": old_ts,
            }) + "\n")

        cleanup_old_transactions(log_dir=log_dir, max_age_hours=24)
        records = list_transactions(log_dir=log_dir)
        assert len(records) == 1
        assert records[0]["state"] == "RECOVERED"


# ------------------------------------------------------------------
# Test: SHA masking in logs
# ------------------------------------------------------------------


class TestLogSafety:
    """Verify SHAs are never fully logged."""

    def test_sha_truncated_in_log(self, tmp_path: Path) -> None:
        """Transaction log contains at most first 8 chars of SHA."""
        env = _make_env(tmp_path)
        os.environ["FR_TX_LOG_DIR"] = env["FR_TX_LOG_DIR"]
        os.environ["FR_TX_LOCK_DIR"] = env["FR_TX_LOCK_DIR"]

        tx = PublicationTransaction(1, "owner", "repo", "hermes/test-safety", "abcdef0123456789" * 5)
        tx.begin()
        tx.commit(actual_remote_sha="9876543210fedcba" * 5)

        records = list_transactions()
        assert len(records) == 2
        # Check BEGIN record (proposed_new_sha is None so we check expected_remote_sha)
        begin_record = records[0]
        assert len(begin_record["expected_remote_sha"]) <= 8
        # Check COMMIT record
        commit_record = records[1]
        assert len(commit_record.get("actual_remote_sha", "")) <= 8
