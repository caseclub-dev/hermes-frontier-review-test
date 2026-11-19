"""P1D-2c: Authorization enforcement integration tests for the publisher.

Tests that perform_push() correctly enforces operator authorization through:
  - AuthPolicy.check_authorized() — verifies active, unexpired, in-budget authorizations
  - AuthPolicy.consume_correction() — decrements budget atomically
  - PublicationTransaction — durable recording of push outcomes

Uses disposable local Git repos and mock PR metadata. No real GitHub access.

Scenarios:
  1. Authorized publication succeeds
  2. Unauthorized publication rejected (no auth record)
  3. Expired authorization rejected
  4. Revoked authorization rejected
  5. Exhausted budget rejected
  6. Stale expected_remote_sha rejected
  7. Unknown remote transitions detected (crash recovery)
  8. Concurrent push rejection (lock contention)
"""

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys_path = os.path.join(os.path.dirname(__file__), "..", "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

# Set audit override BEFORE importing publisher
os.environ["FR_PUBLISH_AUDIT_DIR"] = str(Path(__file__).parent / "_audit_output")

from frontier_review.publisher import (  # noqa: E402
    ACTION_PUSH,
    _make_commit_in_repo,
    _run_git,
    perform_push,
)


# ------------------------------------------------------------------
# Test helpers — mirror test_publisher.py pattern
# ------------------------------------------------------------------


def _create_bare_repo(tmp_path: Path, name: str = "bare.git") -> Path:
    """Create a bare Git repository (mock GitHub remote)."""
    bare = tmp_path / name
    bare.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--bare", str(bare)], capture_output=True, check=True)
    return bare


def _setup_workspace(tmp_path: Path, name: str = "ws", initial_text: str = "v1\n") -> tuple[Path, str]:
    """Create a Git workspace and return (path, sha_a)."""
    ws = tmp_path / name
    ws.mkdir(parents=True, exist_ok=True)
    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@localhost",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@localhost",
    }
    (ws / "file.txt").write_text(initial_text)
    _run_git(["init"], cwd=str(ws), env_override=env)
    _run_git(["config", "user.name", "test"], cwd=str(ws), env_override=env)
    _run_git(["config", "user.email", "test@localhost"], cwd=str(ws), env_override=env)
    _run_git(["add", "."], cwd=str(ws))
    r = _run_git(["commit", "-m", "initial"], cwd=str(ws), env_override=env)
    sha_a = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(ws), capture_output=True, text=True
    ).stdout.strip()
    return ws, sha_a


def _setup_bare_with_branch(bare: Path, ws: Path, branch: str = "hermes/test-001") -> tuple[str, str]:
    """Connect workspace to bare repo, push branch, return (bare_path, remote_sha)."""
    _run_git(["remote", "add", "origin", str(bare)], cwd=str(ws))
    _run_git(["push", "origin", f"HEAD:refs/heads/{branch}"], cwd=str(ws))
    r = _run_git(["rev-parse", f"origin/{branch}"], cwd=str(ws))
    remote_sha = r.stdout.strip()
    return str(bare), remote_sha


def _create_auth_record(auth_dir: str, repo_owner: str, repo_name: str, pr_number: int) -> str:
    """Create a raw auth JSON file in the given directory."""
    auth_id = f"auth-{repo_owner}-{repo_name}-{pr_number}"
    now = datetime.now(timezone.utc)
    record = {
        "authorization_id": auth_id,
        "repository_owner": repo_owner,
        "repository_name": repo_name,
        "pr_number": pr_number,
        "head_branch": "hermes/test-001",
        "initial_remote_sha": "a" * 40,
        "max_corrections": 3,
        "corrections_used": 0,
        "ttl_hours": 720.0,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=720)).isoformat(),
        "status": "active",
    }
    path = os.path.join(auth_dir, f"{auth_id}.json")
    with open(path, "w") as f:
        json.dump(record, f)
    return auth_id


def _make_correction_commit(ws: Path) -> str:
    """Add a correction commit and return the new SHA."""
    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@localhost",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@localhost",
    }
    (ws / "file.txt").write_text("v2\n")
    _run_git(["add", "."], cwd=str(ws))
    _run_git(["commit", "-m", "correction"], cwd=str(ws), env_override=env)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(ws), capture_output=True, text=True
    ).stdout.strip()


# ------------------------------------------------------------------
# Test 1: Authorized publication succeeds
# ------------------------------------------------------------------


class TestAuthorizedPublication:
    """Scenario 1: Valid authorization allows push."""

    def test_authorized_push_succeeds(self, tmp_path: Path):
        ws, sha_a = _setup_workspace(tmp_path, "ws1")
        bare = _create_bare_repo(tmp_path, "bare1.git")
        _, remote_sha = _setup_bare_with_branch(bare, ws, "hermes/test-auth-1")
        new_sha = _make_correction_commit(ws)

        # Create authorization with correct initial_remote_sha
        auth_id = _create_auth_record(
            os.environ["FR_AUTH_DIR"], "caseclub-dev", "hermes-frontier-review-test", 42
        )
        # Patch the auth record with actual SHA
        record_path = os.path.join(os.environ["FR_AUTH_DIR"], f"{auth_id}.json")
        with open(record_path) as f:
            rec = json.load(f)
        rec["initial_remote_sha"] = remote_sha[:40]
        rec["head_branch"] = "hermes/test-auth-1"
        with open(record_path, "w") as f:
            json.dump(rec, f)

        try:
            result = perform_push(
                {
                    "action": ACTION_PUSH,
                    "repo_owner": "caseclub-dev",
                    "repo_name": "hermes-frontier-review-test",
                    "head_branch": "hermes/test-auth-1",
                    "workspace_path": str(ws),
                    "expected_remote_sha": remote_sha,
                    "pr_number": 42,
                },
                mock_mode=True,
                mock_bare_repo=str(bare),
            )
        finally:
            shutil.rmtree(str(ws), ignore_errors=True)

        assert result["success"] is True
        assert result["action"] == "push"
        assert "remote_sha" in result


# ------------------------------------------------------------------
# Test 2: Unauthorized publication rejected (no auth record)
# ------------------------------------------------------------------


class TestUnauthorizedPublication:
    """Scenario 2: Push fails when no authorization exists."""

    def test_no_auth_record_rejected(self, tmp_path: Path):
        ws, sha_a = _setup_workspace(tmp_path, "ws2")
        bare = _create_bare_repo(tmp_path, "bare2.git")
        _, remote_sha = _setup_bare_with_branch(bare, ws, "hermes/test-auth-2")
        _make_correction_commit(ws)

        # NO auth record created — push should fail
        try:
            result = perform_push(
                {
                    "action": ACTION_PUSH,
                    "repo_owner": "caseclub-dev",
                    "repo_name": "hermes-frontier-review-test",
                    "head_branch": "hermes/test-auth-2",
                    "workspace_path": str(ws),
                    "expected_remote_sha": remote_sha,
                    "pr_number": 99,
                },
                mock_mode=True,
                mock_bare_repo=str(bare),
            )
        finally:
            shutil.rmtree(str(ws), ignore_errors=True)

        assert result["success"] is False
        assert "authorization" in result.get("error", "").lower()


# ------------------------------------------------------------------
# Test 3: Expired authorization rejected
# ------------------------------------------------------------------


class TestExpiredAuthorization:
    """Scenario 3: Push fails when auth record has expired."""

    def test_expired_auth_rejected(self, tmp_path: Path):
        ws, sha_a = _setup_workspace(tmp_path, "ws3")
        bare = _create_bare_repo(tmp_path, "bare3.git")
        _, remote_sha = _setup_bare_with_branch(bare, ws, "hermes/test-auth-3")
        _make_correction_commit(ws)

        now_iso = "2020-01-01T00:00:00+00:00"
        expired_iso = "2020-01-02T00:00:00+00:00"
        auth_id = f"auth-expired-repo-3"
        record = {
            "authorization_id": auth_id,
            "repository_owner": "caseclub-dev",
            "repository_name": "hermes-frontier-review-test",
            "pr_number": 42,
            "head_branch": "hermes/test-auth-3",
            "initial_remote_sha": remote_sha[:40],
            "max_corrections": 3,
            "corrections_used": 0,
            "ttl_hours": 1.0,
            "created_at": now_iso,
            "expires_at": expired_iso,
            "status": "active",
        }
        with open(os.path.join(os.environ["FR_AUTH_DIR"], f"{auth_id}.json"), "w") as f:
            json.dump(record, f)

        try:
            result = perform_push(
                {
                    "action": ACTION_PUSH,
                    "repo_owner": "caseclub-dev",
                    "repo_name": "hermes-frontier-review-test",
                    "head_branch": "hermes/test-auth-3",
                    "workspace_path": str(ws),
                    "expected_remote_sha": remote_sha,
                    "pr_number": 42,
                },
                mock_mode=True,
                mock_bare_repo=str(bare),
            )
        finally:
            shutil.rmtree(str(ws), ignore_errors=True)

        assert result["success"] is False


# ------------------------------------------------------------------
# Test 4: Revoked authorization rejected
# ------------------------------------------------------------------


class TestRevokedAuthorization:
    """Scenario 4: Push fails when auth record is revoked."""

    def test_revoked_auth_rejected(self, tmp_path: Path):
        ws, sha_a = _setup_workspace(tmp_path, "ws4")
        bare = _create_bare_repo(tmp_path, "bare4.git")
        _, remote_sha = _setup_bare_with_branch(bare, ws, "hermes/test-auth-4")
        _make_correction_commit(ws)

        auth_id = f"auth-revoked-repo-4"
        record = {
            "authorization_id": auth_id,
            "repository_owner": "caseclub-dev",
            "repository_name": "hermes-frontier-review-test",
            "pr_number": 42,
            "head_branch": "hermes/test-auth-4",
            "initial_remote_sha": remote_sha[:40],
            "max_corrections": 3,
            "corrections_used": 0,
            "ttl_hours": 720.0,
            "created_at": "2026-01-01T00:00:00+00:00",
            "expires_at": "2027-01-01T00:00:00+00:00",
            "status": "revoked",
        }
        with open(os.path.join(os.environ["FR_AUTH_DIR"], f"{auth_id}.json"), "w") as f:
            json.dump(record, f)

        try:
            result = perform_push(
                {
                    "action": ACTION_PUSH,
                    "repo_owner": "caseclub-dev",
                    "repo_name": "hermes-frontier-review-test",
                    "head_branch": "hermes/test-auth-4",
                    "workspace_path": str(ws),
                    "expected_remote_sha": remote_sha,
                    "pr_number": 42,
                },
                mock_mode=True,
                mock_bare_repo=str(bare),
            )
        finally:
            shutil.rmtree(str(ws), ignore_errors=True)

        assert result["success"] is False


# ------------------------------------------------------------------
# Test 5: Exhausted budget rejected
# ------------------------------------------------------------------


class TestExhaustedBudget:
    """Scenario 5: Push fails when correction budget is exhausted."""

    def test_exhausted_budget_rejected(self, tmp_path: Path):
        ws, sha_a = _setup_workspace(tmp_path, "ws5")
        bare = _create_bare_repo(tmp_path, "bare5.git")
        _, remote_sha = _setup_bare_with_branch(bare, ws, "hermes/test-auth-5")
        _make_correction_commit(ws)

        # Write exhausted counter to state_dir BEFORE creating auth record
        # (conftest sets FR_STATE_DIR=state dir in tmp_path)
        state_dir = os.environ.get("FR_STATE_DIR", "")
        if not state_dir:
            pytest.skip("FR_STATE_DIR not configured")

        auth_id_base = f"auth-caseclub-dev-hermes-frontier-review-test-42"
        counter_file = os.path.join(state_dir, f"{auth_id_base}.counter")
        with open(counter_file, "w") as f:
            f.write("3\n")  # Already at max (default 3)

        auth_id = _create_auth_record(
            os.environ["FR_AUTH_DIR"], "caseclub-dev", "hermes-frontier-review-test", 42
        )
        # Patch with actual SHA and branch
        record_path = os.path.join(os.environ["FR_AUTH_DIR"], f"{auth_id}.json")
        with open(record_path) as f:
            rec = json.load(f)
        rec["initial_remote_sha"] = remote_sha[:40]
        rec["head_branch"] = "hermes/test-auth-5"
        with open(record_path, "w") as f:
            json.dump(rec, f)

        try:
            result = perform_push(
                {
                    "action": ACTION_PUSH,
                    "repo_owner": "caseclub-dev",
                    "repo_name": "hermes-frontier-review-test",
                    "head_branch": "hermes/test-auth-5",
                    "workspace_path": str(ws),
                    "expected_remote_sha": remote_sha,
                    "pr_number": 42,
                },
                mock_mode=True,
                mock_bare_repo=str(bare),
            )
        finally:
            shutil.rmtree(str(ws), ignore_errors=True)

        assert result["success"] is False


# ------------------------------------------------------------------
# Test 6: Stale expected_remote_sha rejected
# ------------------------------------------------------------------


class TestStaleSHA:
    """Scenario 6: Push fails when remote SHA doesn't match expected."""

    def test_stale_expected_sha_rejected(self, tmp_path: Path):
        ws, sha_a = _setup_workspace(tmp_path, "ws6")
        bare = _create_bare_repo(tmp_path, "bare6.git")
        _, remote_sha = _setup_bare_with_branch(bare, ws, "hermes/test-auth-6")
        _make_correction_commit(ws)

        _create_auth_record(
            os.environ["FR_AUTH_DIR"], "caseclub-dev", "hermes-frontier-review-test", 42
        )
        # Patch auth with actual SHA
        record_path = os.path.join(os.environ["FR_AUTH_DIR"], "auth-caseclub-dev-hermes-frontier-review-test-42.json")
        with open(record_path) as f:
            rec = json.load(f)
        rec["initial_remote_sha"] = remote_sha[:40]
        rec["head_branch"] = "hermes/test-auth-6"
        with open(record_path, "w") as f:
            json.dump(rec, f)

        # Use deliberately wrong expected_remote_sha
        try:
            result = perform_push(
                {
                    "action": ACTION_PUSH,
                    "repo_owner": "caseclub-dev",
                    "repo_name": "hermes-frontier-review-test",
                    "head_branch": "hermes/test-auth-6",
                    "workspace_path": str(ws),
                    "expected_remote_sha": "d" * 40,  # Wrong SHA!
                    "pr_number": 42,
                },
                mock_mode=True,
                mock_bare_repo=str(bare),
            )
        finally:
            shutil.rmtree(str(ws), ignore_errors=True)

        assert result["success"] is False
        assert "SHA" in result.get("error", "") or "sha" in result.get("error", "").lower()


# ------------------------------------------------------------------
# Test 7: Unknown remote transitions (crash recovery)
# ------------------------------------------------------------------


class TestCrashRecoveryIntegration:
    """Scenario 7: Crash recovery detects unknown remote transitions."""

    def test_recovery_detects_unknown_state(self, tmp_path: Path):
        from frontier_review.publication_transaction import PublicationTransaction, cleanup_old_transactions

        log_dir = str(tmp_path / "tx_log")
        lock_dir = str(tmp_path / "locks")

        # Simulate an interrupted transaction: no record exists for this PR
        tx = PublicationTransaction(
            99,
            "caseclub-dev",
            "hermes-frontier-review-test",
            "hermes/test-crash",
            "e" * 40,
            transaction_log_dir=log_dir,
            lock_dir=lock_dir,
        )

        # No BEGIN called — simulate crash before transaction started
        result = tx.recover(lambda: "f" * 40)
        assert result == "RECOVERED_UNKNOWN"


# ------------------------------------------------------------------
# Test 8: Concurrent lock serialization
# ------------------------------------------------------------------


class TestConcurrentLocks:
    """Scenario 8: Concurrent push requests are serialized."""

    def test_lock_contention_detected(self, tmp_path: Path):
        from frontier_review.publication_transaction import PublicationTransaction, PublicationTransactionError

        log_dir = str(tmp_path / "tx_log2")
        lock_dir = str(tmp_path / "locks2")
        os.makedirs(log_dir)
        os.makedirs(lock_dir)

        tx1 = PublicationTransaction(
            99,
            "caseclub-dev",
            "hermes-frontier-review-test",
            "hermes/test-lock-1",
            "g" * 40,
            transaction_log_dir=log_dir,
            lock_dir=lock_dir,
        )
        tx2 = PublicationTransaction(
            99,
            "caseclub-dev",
            "hermes-frontier-review-test",
            "hermes/test-lock-2",
            "h" * 40,
            transaction_log_dir=log_dir,
            lock_dir=lock_dir,
        )

        tx1.begin()
        tx1.lock()

        # Second lock should raise on contention
        with pytest.raises(PublicationTransactionError):
            tx2.lock()

        tx1.unlock()
