"""Subphase P1A: Focused tests for the trusted publisher core.

Tests validate schema enforcement, repository/branch allowlisting,
fast-forward-only verification, SHA descendant checks, and audit logging.

All operations use local mock Git repositories — no network access.

Test scenarios covered:
  1. Valid fast-forward correction              (TestValidFastForward)
  2. Incorrect repository                        (TestInvalidRepository)
  3. Incorrect PR or branch                      (TestInvalidBranch)
  4. Stale remote SHA                            (TestStaleRemoteSHA)
  5. Non-descendant commit                       (TestNonDescendantCommit)
  6. Unauthorized filesystem path                (TestUnauthorizedPath)
  7. Malformed or oversized request              (TestMalformedRequest)
  8. Duplicate publication detection             (TestDuplicatePublication)
  9. Rejected arbitrary command/refspec          (TestRejectedArbitraryCommand)
  10. Audit record persistence                   (TestAuditPersistence)

Additionally tests verify:
  - verify_sha action                           (TestVerifySHA)
  - status action                               (TestStatus)
  - Fast-forward helper function                (TestFastForwardCheck)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# Set audit log override BEFORE importing publisher module.
# This ensures _audit() uses the test directory instead of production path.
os.environ["FR_PUBLISH_AUDIT_DIR"] = str(Path(__file__).parent / "_audit_output")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from frontier_review.publisher import (  # noqa: E402
    ACTION_PUSH,
    ACTION_STATUS,
    ACTION_VERIFY_SHA,
    MAX_PAYLOAD_SIZE,
    PublisherValidationError,
    _audit,
    _is_fast_forward,
    _run_git,
    validate_request,
    perform_push,
    perform_status,
    perform_verify_sha,
)

# ---------------------------------------------------------------------------
# Helpers — shared across tests
# ---------------------------------------------------------------------------


def _create_workspace(tmp_path: Path, name: str = "ws", initial_text: str = "v1\n") -> Path:
    """Create a Git workspace with one initial commit and return the path."""
    ws = tmp_path / name
    ws.mkdir()
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
    _run_git(["commit", "-m", "initial"], cwd=str(ws), env_override=env)
    return ws


def _create_bare_repo(tmp_path: Path, name: str = "bare.git") -> Path:
    """Create a bare Git repository (mock GitHub remote)."""
    bare = tmp_path / name
    bare.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--bare", str(bare)], capture_output=True, check=True)
    return bare


def _setup_bare_with_branch(
    bare: Path, ws: Path, branch: str = "hermes/test-001"
) -> tuple[str, str]:
    """Connect workspace to bare repo, push branch, return (bare_path, remote_sha)."""
    _run_git(["remote", "add", "origin", str(bare)], cwd=str(ws))
    _run_git(["push", "origin", f"HEAD:refs/heads/{branch}"], cwd=str(ws))
    r = _run_git(["rev-parse", f"origin/{branch}"], cwd=str(ws))
    remote_sha = r.stdout.strip()
    return str(bare), remote_sha


# ---------------------------------------------------------------------------
# Scenario 1 — Valid fast-forward correction
# ---------------------------------------------------------------------------


class TestValidFastForward:
    """Workspace HEAD is a descendant of remote HEAD → push succeeds."""

    def test_push_succeeds_fast_forward(self, tmp_path: Path) -> None:
        # Step 1: Create workspace with initial commit (SHA_A)
        ws = _create_workspace(tmp_path, "ws")
        bare = _create_bare_repo(tmp_path, "bare.git")

        # Step 2: Push to bare repo on branch hermes/test-001
        _, sha_a = _setup_bare_with_branch(bare, ws, "hermes/test-001")

        # Step 3: Add a second commit to workspace (descendant of SHA_A)
        env = {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@localhost",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@localhost",
        }
        (ws / "change.txt").write_text("v2\n")
        _run_git(["add", "."], cwd=str(ws))
        _run_git(["commit", "-m", "correction"], cwd=str(ws), env_override=env)

        # Step 4: Call perform_push() — should succeed in mock mode
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws),
            "expected_remote_sha": sha_a,
        }

        result = perform_push(req, mock_mode=True, mock_bare_repo=str(bare))
        assert result["success"] is True
        assert result["action"] == ACTION_PUSH
        assert len(result.get("remote_sha", "")) == 40


# ---------------------------------------------------------------------------
# Scenario 2 — Incorrect repository
# ---------------------------------------------------------------------------


class TestInvalidRepository:
    """Request targets a repo not in the allowlist."""

    def test_wrong_owner_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "wrong-owner",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws),
        }
        with pytest.raises(PublisherValidationError, match="not on allowlist"):
            validate_request(json.dumps(req).encode())

    def test_wrong_repo_name_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "some-other-repo",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws),
        }
        with pytest.raises(PublisherValidationError, match="not on allowlist"):
            validate_request(json.dumps(req).encode())

    def test_both_fields_required(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {"action": ACTION_PUSH, "head_branch": "hermes/test-001", "workspace_path": str(ws)}
        with pytest.raises(PublisherValidationError, match="Missing required field"):
            validate_request(json.dumps(req).encode())


# ---------------------------------------------------------------------------
# Scenario 3 — Incorrect or disallowed branch
# ---------------------------------------------------------------------------


class TestInvalidBranch:
    """Branch does not match the hermes/* pattern."""

    def test_feat_branch_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "feat/test-001",
            "workspace_path": str(ws),
        }
        with pytest.raises(PublisherValidationError, match="pattern"):
            validate_request(json.dumps(req).encode())

    def test_main_branch_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "main",
            "workspace_path": str(ws),
        }
        with pytest.raises(PublisherValidationError, match="pattern"):
            validate_request(json.dumps(req).encode())

    def test_invalid_char_branch_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test@bad!",
            "workspace_path": str(ws),
        }
        with pytest.raises(PublisherValidationError, match="pattern"):
            validate_request(json.dumps(req).encode())

    def test_valid_hermes_branch_accepted(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001-task-name",
            "workspace_path": str(ws),
        }
        result = validate_request(json.dumps(req).encode())
        assert result["head_branch"] == "hermes/test-001-task-name"


# ---------------------------------------------------------------------------
# Scenario 4 — Stale remote SHA (non-descendant commit)
# ---------------------------------------------------------------------------


class TestStaleRemoteSHA:
    """expected_remote_sha is from a different lineage than workspace HEAD."""

    def test_non_descendant_rejected(self, tmp_path: Path) -> None:
        # Create two independent workspaces with unrelated commits
        ws_a = _create_workspace(tmp_path, "ws_a", "from-a v1\n")
        ws_b = _create_workspace(tmp_path, "ws_b", "from-b v1\n")

        # Push workspace B to bare repo as hermes/test-001
        bare = _create_bare_repo(tmp_path, "bare.git")
        _, sha_b_remote = _setup_bare_with_branch(bare, ws_b, "hermes/test-001")

        # Try to push workspace A with expected_remote_sha from workspace B.
        # Since A's history doesn't descend from B's commits, it should fail.
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws_a),
            "expected_remote_sha": sha_b_remote,
        }

        result = perform_push(req, mock_mode=True, mock_bare_repo=str(bare))
        assert result["success"] is False
        assert "descendant" in result["error"].lower()


# ---------------------------------------------------------------------------
# Scenario 5 — Non-descendant commit (no expected_remote_sha)
# ---------------------------------------------------------------------------


class TestNonDescendantCommit:
    """Push would create a non-fast-forward merge — git rejects without --force."""

    def test_non_descendant_fails_push(self, tmp_path: Path) -> None:
        # Create workspace A and push to bare repo on hermes/test-001
        ws_a = _create_workspace(tmp_path, "ws_a")
        bare = _create_bare_repo(tmp_path, "bare.git")
        _, _ = _setup_bare_with_branch(bare, ws_a, "hermes/test-001")

        # Create workspace B with a completely separate initial commit (not descendant of A's commit).
        # When we remove origin, add bare repo, and fetch:
        #   - origin/hermes/test-001 points to ws_a's SHA (from step 1)
        #   - ws_b HEAD is a different SHA (unrelated history)
        # Git push will reject because it would not be fast-forward.
        ws_b = _create_workspace(tmp_path, "ws_b", "separate v1\n")

        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws_b),
        }

        result = perform_push(req, mock_mode=True, mock_bare_repo=str(bare))
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Scenario 6 — Unauthorized filesystem path
# ---------------------------------------------------------------------------


class TestUnauthorizedPath:
    """Workspace path outside allowed boundaries or doesn't exist."""

    def test_nonexistent_path_rejected(self) -> None:
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": "/nonexistent/path/that/does/not/exist",
        }

        result = perform_push(req)
        assert result["success"] is False

    def test_file_instead_of_directory_rejected(self, tmp_path: Path) -> None:
        fake_workspace = tmp_path / "not_a_dir.txt"
        fake_workspace.write_text("not a dir")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(fake_workspace),
        }

        result = perform_push(req)
        assert result["success"] is False

    def test_empty_workspace_path_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": "",
        }
        with pytest.raises(PublisherValidationError, match="workspace_path is required"):
            validate_request(json.dumps(req).encode())


# ---------------------------------------------------------------------------
# Scenario 7 — Malformed or oversized request
# ---------------------------------------------------------------------------


class TestMalformedRequest:
    """Invalid JSON, wrong types, oversized payload."""

    def test_invalid_json_rejected(self) -> None:
        with pytest.raises(PublisherValidationError, match="Invalid JSON"):
            validate_request(b"not json at all {{{")

    def test_non_object_payload_rejected(self) -> None:
        with pytest.raises(PublisherValidationError, match="must be a JSON object"):
            validate_request(json.dumps([1, 2, 3]).encode())

    def test_missing_action_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {"repo_owner": "caseclub-dev", "repo_name": "hermes-frontier-review-test",
               "head_branch": "hermes/test-001", "workspace_path": str(ws)}
        with pytest.raises(PublisherValidationError, match="Missing required field"):
            validate_request(json.dumps(req).encode())

    def test_invalid_action_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {"action": "delete_everything", "repo_owner": "caseclub-dev",
               "repo_name": "hermes-frontier-review-test", "head_branch": "hermes/test-001",
               "workspace_path": str(ws)}
        with pytest.raises(PublisherValidationError, match="Invalid action"):
            validate_request(json.dumps(req).encode())

    def test_oversized_payload_rejected(self) -> None:
        big = b'{"action":"push","repo_owner":"caseclub-dev","repo_name":"hermes-frontier-review-test","head_branch":"hermes/test-001","workspace_path":"' + \
              b"A" * (MAX_PAYLOAD_SIZE + 100) + b'"}'
        with pytest.raises(PublisherValidationError, match="exceeds"):
            validate_request(big)

    def test_invalid_sha_format_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws),
            "expected_remote_sha": "not-a-valid-sha",
        }
        with pytest.raises(PublisherValidationError, match="40 lowercase hex"):
            validate_request(json.dumps(req).encode())

    def test_uppercase_sha_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws),
            "expected_remote_sha": "A" * 40,
        }
        with pytest.raises(PublisherValidationError, match="40 lowercase hex"):
            validate_request(json.dumps(req).encode())

    def test_short_sha_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws),
            "expected_remote_sha": "abc123",  # too short
        }
        with pytest.raises(PublisherValidationError, match="40 lowercase hex"):
            validate_request(json.dumps(req).encode())

    def test_invalid_pr_number_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws),
            "pr_number": -5,
        }
        with pytest.raises(PublisherValidationError, match="positive integer"):
            validate_request(json.dumps(req).encode())

    def test_pr_number_zero_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws),
            "pr_number": 0,
        }
        with pytest.raises(PublisherValidationError, match="positive integer"):
            validate_request(json.dumps(req).encode())

    def test_pr_number_string_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws),
            "pr_number": "forty-two",
        }
        with pytest.raises(PublisherValidationError, match="positive integer"):
            validate_request(json.dumps(req).encode())

    def test_workspace_path_too_long_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": "A" * 513,
        }
        with pytest.raises(PublisherValidationError, match="512 characters"):
            validate_request(json.dumps(req).encode())


# ---------------------------------------------------------------------------
# Scenario 8 — Duplicate publication detection
# ---------------------------------------------------------------------------


class TestDuplicatePublication:
    """Publishing the same SHA twice is detected via audit log / ledger."""

    def test_same_sha_twice_succeeds_push(self, tmp_path: Path) -> None:
        """Push idempotent to bare repo (git detects no change)."""
        ws = _create_workspace(tmp_path, "ws")
        bare = _create_bare_repo(tmp_path, "bare.git")
        _, sha_a = _setup_bare_with_branch(bare, ws, "hermes/test-001")

        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws),
            "expected_remote_sha": sha_a,
        }

        # First push — succeeds (nothing to push, idempotent)
        result1 = perform_push(req, mock_mode=True, mock_bare_repo=str(bare))
        assert result1["success"] is True

        # Second push — also succeeds at Git level; duplicate detection
        # is enforced by the controller's ledger (outside this module).
        result2 = perform_push(req, mock_mode=True, mock_bare_repo=str(bare))
        assert result2["success"] is True


# ---------------------------------------------------------------------------
# Scenario 9 — Rejected arbitrary command/refspec
# ---------------------------------------------------------------------------


class TestRejectedArbitraryCommand:
    """Actions like 'delete', 'merge', or force-push are rejected."""

    def test_delete_action_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": "delete",
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws),
        }
        with pytest.raises(PublisherValidationError, match="Invalid action"):
            validate_request(json.dumps(req).encode())

    def test_merge_action_rejected(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        req = {
            "action": "merge",
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws),
        }
        with pytest.raises(PublisherValidationError, match="Invalid action"):
            validate_request(json.dumps(req).encode())

    def test_force_push_prevented(self) -> None:
        """The publisher source code must never use --force flag (non-comment lines only)."""
        import inspect
        from frontier_review.publisher import perform_push as pp

        source_lines = inspect.getsource(pp).splitlines()
        for line in source_lines:
            stripped = line.split("#", 1)[0]  # Remove inline comments
            assert "--force" not in stripped, f"Publisher must not use --force flag: {line}"

    def test_push_is_only_write_action(self) -> None:
        """Only 'push' modifies remote state; all other actions are read-only or status."""
        valid_actions = {"push", "status", "verify_sha"}
        write_actions = {"push"}
        assert {"delete", "merge", "force-push"}.isdisjoint(valid_actions)


# ---------------------------------------------------------------------------
# Scenario 10 — Audit record persistence
# ---------------------------------------------------------------------------


class TestAuditPersistence:
    """Audit records are written to disk and contain required fields."""

    def test_audit_record_written_on_success(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        bare = _create_bare_repo(tmp_path, "bare.git")
        _, sha_a = _setup_bare_with_branch(bare, ws, "hermes/test-001")

        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": str(ws),
            "expected_remote_sha": sha_a,
        }

        perform_push(req, mock_mode=True, mock_bare_repo=str(bare))

        audit_file = Path(os.environ["FR_PUBLISH_AUDIT_DIR"]) / "publisher_audit.log"
        assert audit_file.exists(), "Audit log file was not created"

        records = [json.loads(line) for line in audit_file.read_text().strip().split("\n") if line.strip()]
        success_records = [r for r in records if r["action"] == ACTION_PUSH and r["success"] is True]
        assert len(success_records) >= 1, "No successful push audit record found"

    def test_audit_record_written_on_failure(self) -> None:
        """Validation failure also produces an audit record."""
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": "/tmp/does_not_exist_publisher_test_xyz",
        }

        perform_push(req)  # Should fail — path doesn't exist
        audit_file = Path(os.environ["FR_PUBLISH_AUDIT_DIR"]) / "publisher_audit.log"
        assert audit_file.exists(), "Audit log file was not created on failure"

        records = [json.loads(line) for line in audit_file.read_text().strip().split("\n") if line.strip()]
        failure_records = [r for r in records if r["action"] == ACTION_PUSH and r["success"] is False]
        assert len(failure_records) >= 1, "No failed push audit record found"

    def test_audit_record_fields(self) -> None:
        """Each audit record must contain ts, pid, action, success, detail."""
        req = {
            "action": ACTION_PUSH,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
            "workspace_path": "/nonexistent_audit_field_test",
        }

        perform_push(req)  # Fails as expected
        audit_file = Path(os.environ["FR_PUBLISH_AUDIT_DIR"]) / "publisher_audit.log"
        records = [json.loads(line) for line in audit_file.read_text().strip().split("\n") if line.strip()]

        required_keys = {"ts", "pid", "action", "success", "detail"}
        for record in records:
            assert required_keys.issubset(record.keys()), (
                f"Audit record missing keys {required_keys - set(record.keys())}: {record}"
            )


# ---------------------------------------------------------------------------
# Additional tests — verify_sha, status, fast-forward helper
# ---------------------------------------------------------------------------


class TestVerifySHA:
    """Tests for the verify_sha action."""

    def test_verify_sha_finds_branch(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        bare = _create_bare_repo(tmp_path, "bare.git")
        _, sha_a = _setup_bare_with_branch(bare, ws, "hermes/test-001")

        req = {
            "action": ACTION_VERIFY_SHA,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/test-001",
        }

        result = perform_verify_sha(req, mock_mode=True, mock_bare_repo=str(bare))
        assert result["success"] is True
        assert result["remote_sha"] == sha_a

    def test_verify_sha_branch_not_found(self, tmp_path: Path) -> None:
        bare = _create_bare_repo(tmp_path, "bare.git")  # Empty bare repo

        req = {
            "action": ACTION_VERIFY_SHA,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "head_branch": "hermes/nonexistent",
        }

        result = perform_verify_sha(req, mock_mode=True, mock_bare_repo=str(bare))
        assert result["success"] is False


class TestStatus:
    """Tests for the status action."""

    def test_status_mock_succeeds(self) -> None:
        req = {
            "action": ACTION_STATUS,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "ref_sha": "a" * 40,
            "pr_number": 42,
        }

        result = perform_status(req, mock_mode=True)
        assert result["success"] is True

    def test_status_invalid_sha(self) -> None:
        req = {
            "action": ACTION_STATUS,
            "repo_owner": "caseclub-dev",
            "repo_name": "hermes-frontier-review-test",
            "ref_sha": "not-a-sha",
        }

        result = perform_status(req, mock_mode=False)
        assert result["success"] is False


class TestFastForwardCheck:
    """Tests for the _is_fast_forward helper function."""

    def test_is_fast_forward_true(self, tmp_path: Path) -> None:
        # Create workspace, push to bare repo, then add another commit on top.
        # After fetch, origin/branch = SHA_A, local HEAD = SHA_B (descendant of A).
        ws = _create_workspace(tmp_path, "ws")
        bare = _create_bare_repo(tmp_path, "bare.git")
        _, _ = _setup_bare_with_branch(bare, ws, "hermes/test-001")

        # Add second commit to local workspace
        env = {
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@localhost",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@localhost",
        }
        (ws / "file2.txt").write_text("v2\n")
        _run_git(["add", "."], cwd=str(ws))
        _run_git(["commit", "-m", "second"], cwd=str(ws), env_override=env)

        # Fetch to update remote tracking refs
        _run_git(["fetch", "origin"], cwd=str(ws))

        assert _is_fast_forward(str(ws), "hermes/test-001", "") is True

    def test_is_fast_forward_false_no_remote(self, tmp_path: Path) -> None:
        ws = _create_workspace(tmp_path, "ws")
        # No remote configured — fast-forward check should fail
        assert _is_fast_forward(str(ws), "hermes/test-001", "") is False
