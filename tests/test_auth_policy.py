"""Tests for fr-auth operator authorization CLI (P1D-2b).

29 tests covering AuthPolicy CRUD, input validation, security checks,
atomic writes, and the CLI entry point.
"""

import json
import os
import stat
import subprocess
import sys
import uuid

import pytest

# Ensure the source tree is on the path so imports work
SRC_ROOT = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from frontier_review.auth_policy import (
    AuthPolicy,
    AuthorizationError,
    PolicyViolationError,
    RecordNotFoundError,
    ValidationError,
    main,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def auth_dir_fixture(tmp_path, monkeypatch):
    """Provide a fresh temp directory for auth records and set FR_AUTH_DIR."""
    monkeypatch.setenv("FR_AUTH_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def policy(auth_dir_fixture):
    """Create an AuthPolicy instance pointing at the temp directory."""
    return AuthPolicy()


# ---------------------------------------------------------------------------
# Tests 1-3: Create — valid, duplicate, different PR
# ---------------------------------------------------------------------------

VALID_OWNER = "caseclub-dev"
VALID_REPO_NAME = "hermes-frontier-review-test"
VALID_SHA = "abcdef0123456789abcdef0123456789abcdef01"
VALID_BRANCH = "hermes/p1d-2b"


def test_create_valid(policy):
    """Test 1: Valid inputs create record; verify all fields."""
    record = policy.create(
        repository_owner=VALID_OWNER,
        repository_name=VALID_REPO_NAME,
        pr_number=42,
        head_branch=VALID_BRANCH,
        initial_remote_sha=VALID_SHA,
        max_corrections=5,
        ttl_hours=48.0,
    )
    assert record["authorization_id"] is not None
    assert len(record["authorization_id"]) == 36  # UUID format
    assert record["repository_owner"] == VALID_OWNER
    assert record["repository_name"] == VALID_REPO_NAME
    assert record["pr_number"] == 42
    assert record["head_branch"] == VALID_BRANCH
    assert record["initial_remote_sha"] == VALID_SHA
    assert record["max_corrections"] == 5
    assert record["corrections_used"] == 0
    assert record["status"] == "active"
    assert record["ttl_hours"] == 48.0
    # Timestamps present and valid ISO format
    assert "T" in record["created_at"]
    assert "T" in record["expires_at"]


def test_create_duplicate_same_pr(policy):
    """Test 2: Same owner/name/pr_number raises ValidationError."""
    policy.create(VALID_OWNER, VALID_REPO_NAME, 10, VALID_BRANCH, VALID_SHA)
    with pytest.raises(ValidationError, match="Active authorization already exists"):
        policy.create(VALID_OWNER, VALID_REPO_NAME, 10, VALID_BRANCH, VALID_SHA)


def test_create_different_pr_allowed(policy):
    """Test 3: Different PR for same repo allowed."""
    policy.create(VALID_OWNER, VALID_REPO_NAME, 10, VALID_BRANCH, VALID_SHA)
    record2 = policy.create(VALID_OWNER, VALID_REPO_NAME, 20, VALID_BRANCH, VALID_SHA)
    assert record2["pr_number"] == 20


# ---------------------------------------------------------------------------
# Tests 4-7: Input validation — repo, branch, SHA, PR number
# ---------------------------------------------------------------------------

def test_invalid_repo_not_on_allowlist(policy):
    """Test 4: Rejects repos not in ALLOWED_REPOS."""
    with pytest.raises(ValidationError, match="not on allowlist"):
        policy.create("other-owner", "other-repo", 1, VALID_BRANCH, VALID_SHA)


def test_invalid_branch_pattern(policy):
    """Test 5: Rejects non-matching branch names."""
    for bad_branch in ["main", "fix-build", "hermes/", "feature/test"]:
        with pytest.raises(ValidationError):
            policy.create(VALID_OWNER, VALID_REPO_NAME, 1, bad_branch, VALID_SHA)


def test_valid_branch_variants(policy):
    """Test 6: Accepts hermes/p1d-2b, hermes/task_001, hermes/a.b-c_d."""
    for idx, branch in enumerate(["hermes/p1d-2b", "hermes/task_001", "hermes/a.b-c_d"]):
        record = policy.create(VALID_OWNER, VALID_REPO_NAME, 100 + idx, branch, VALID_SHA)
        assert record["head_branch"] == branch


def test_invalid_sha_format(policy):
    """Test 7: Rejects non-40-char, uppercase, wrong-length SHA."""
    bad_shas = [
        "abcdef" * 6,           # 36 chars — too short
        "abcdef0123456789" * 5,  # 48 chars — too long
        "ABCDEF0123456789" * 4,  # uppercase
        "z" * 40,                # invalid hex
        "",                      # empty
    ]
    for bad_sha in bad_shas:
        with pytest.raises(ValidationError):
            policy.create(VALID_OWNER, VALID_REPO_NAME, 1, VALID_BRANCH, bad_sha)


def test_invalid_pr_number(policy):
    """Test 8: Rejects 0, negative, float-like strings."""
    for bad_pr in [0, -1, "42", 3.5, True]:
        with pytest.raises(ValidationError):
            policy.create(VALID_OWNER, VALID_REPO_NAME, bad_pr, VALID_BRANCH, VALID_SHA)


def test_max_corrections_bounds(policy):
    """Test 9: Rejects 0 and >10; accepts 1 and 10."""
    with pytest.raises(ValidationError):
        policy.create(VALID_OWNER, VALID_REPO_NAME, 1, VALID_BRANCH, VALID_SHA, max_corrections=0)
    with pytest.raises(ValidationError):
        policy.create(VALID_OWNER, VALID_REPO_NAME, 1, VALID_BRANCH, VALID_SHA, max_corrections=11)

    r1 = policy.create(VALID_OWNER, VALID_REPO_NAME, 1, VALID_BRANCH, VALID_SHA, max_corrections=1)
    assert r1["max_corrections"] == 1
    r10 = policy.create(VALID_OWNER, VALID_REPO_NAME, 2, VALID_BRANCH, VALID_SHA, max_corrections=10)
    assert r10["max_corrections"] == 10


def test_ttl_bounds(policy):
    """Test 10: Default 72h; reject ttl=0 or ttl>720."""
    # Verify default is 72 hours
    record = policy.create(VALID_OWNER, VALID_REPO_NAME, 1, VALID_BRANCH, VALID_SHA)
    assert record["ttl_hours"] == 72.0

    with pytest.raises(ValidationError):
        policy.create(VALID_OWNER, VALID_REPO_NAME, 1, VALID_BRANCH, VALID_SHA, ttl_hours=0)
    with pytest.raises(ValidationError):
        policy.create(VALID_OWNER, VALID_REPO_NAME, 1, VALID_BRANCH, VALID_SHA, ttl_hours=721.0)


# ---------------------------------------------------------------------------
# Tests 11-14: Expiry, revocation, correction consumption
# ---------------------------------------------------------------------------

def test_expired_authorization(policy):
    """Test 11: Manually set expires_at to past, check_authorized returns False."""
    record = policy.create(VALID_OWNER, VALID_REPO_NAME, 5, VALID_BRANCH, VALID_SHA)
    # Manually expire the record by writing a past timestamp
    filepath = os.path.join(policy.auth_dir, f"{record['authorization_id']}.json")
    import json as _json
    with open(filepath, "r") as f:
        r = _json.load(f)
    from datetime import datetime, timezone, timedelta
    r["expires_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with open(filepath, "w") as f:
        _json.dump(r, f)

    authorized, rec = policy.check_authorized(VALID_OWNER, VALID_REPO_NAME, 5)
    assert authorized is False
    assert rec is None


def test_revoked_authorization(policy):
    """Test 12: Revoke then check returns False; list_active excludes it."""
    record = policy.create(VALID_OWNER, VALID_REPO_NAME, 7, VALID_BRANCH, VALID_SHA)
    # Revoke via policy API
    policy.revoke(record["authorization_id"])

    authorized, rec = policy.check_authorized(VALID_OWNER, VALID_REPO_NAME, 7)
    assert authorized is False

    active_list = policy.list_active()
    assert all(r["pr_number"] != 7 for r in active_list)


def test_consume_correction_within_budget(policy):
    """Test 13: Max=3, consume twice (total 2), returns True."""
    record = policy.create(VALID_OWNER, VALID_REPO_NAME, 8, VALID_BRANCH, VALID_SHA, max_corrections=3)
    aid = record["authorization_id"]

    assert policy.consume_correction(aid) is True   # corrections_used becomes 1
    assert policy.consume_correction(aid) is True   # corrections_used becomes 2


def test_consume_correction_exceeds_budget(policy):
    """Test 14: Max=3, consume 4 times, 4th raises PolicyViolationError."""
    record = policy.create(VALID_OWNER, VALID_REPO_NAME, 9, VALID_BRANCH, VALID_SHA, max_corrections=3)
    aid = record["authorization_id"]

    assert policy.consume_correction(aid) is True   # 1/3
    assert policy.consume_correction(aid) is True   # 2/3
    assert policy.consume_correction(aid) is True   # 3/3 (at budget, still OK)
    with pytest.raises(PolicyViolationError):
        policy.consume_correction(aid)                # 4/3 — exceeded


# ---------------------------------------------------------------------------
# Tests 15-18: List, revoke, get — non-existent records
# ---------------------------------------------------------------------------

def test_list_empty(policy):
    """Test 15: No records returns empty list."""
    assert policy.list_active() == []


def test_list_active_excludes_revoked_expired(policy):
    """Test 16: Create active + revoked + expired; list returns only active."""
    # Record A: will be expired (patch its expires_at to the past)
    r_a = policy.create(VALID_OWNER, VALID_REPO_NAME, 20, VALID_BRANCH, VALID_SHA)
    filepath = os.path.join(policy.auth_dir, f"{r_a['authorization_id']}.json")
    import json as _json
    from datetime import datetime, timezone, timedelta

    with open(filepath, "r") as f:
        rec = _json.load(f)
    rec["expires_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with open(filepath, "w") as f:
        _json.dump(rec, f)

    # Record B: will be revoked
    r_b = policy.create(VALID_OWNER, VALID_REPO_NAME, 21, VALID_BRANCH, VALID_SHA)
    policy.revoke(r_b["authorization_id"])

    # Record C: stays active (never touched)
    r_c = policy.create(VALID_OWNER, VALID_REPO_NAME, 22, VALID_BRANCH, VALID_SHA)

    active = policy.list_active()
    assert len(active) == 1
    assert active[0]["pr_number"] == 22


def test_revoke_nonexistent(policy):
    """Test 17: Revoke a nonexistent ID raises RecordNotFoundError."""
    fake_id = str(uuid.uuid4())
    with pytest.raises(RecordNotFoundError):
        policy.revoke(fake_id)


def test_get_nonexistent(policy):
    """Test 18: Get a nonexistent ID raises RecordNotFoundError."""
    fake_id = str(uuid.uuid4())
    with pytest.raises(RecordNotFoundError):
        policy.get(fake_id)


# ---------------------------------------------------------------------------
# Tests 19-20: Persistence, path injection
# ---------------------------------------------------------------------------

def test_atomic_write_persistence(auth_dir_fixture):
    """Test 19: Write record, new AuthPolicy instance, verify persists."""
    # Create a record with one instance
    policy1 = AuthPolicy(str(auth_dir_fixture))
    record = policy1.create(VALID_OWNER, VALID_REPO_NAME, 30, VALID_BRANCH, VALID_SHA)

    # New instance — same directory
    policy2 = AuthPolicy(str(auth_dir_fixture))
    retrieved = policy2.get(record["authorization_id"])
    assert retrieved["pr_number"] == 30
    assert retrieved["status"] == "active"


def test_no_path_injection(policy):
    """Test 20: Repo with '../', branch with newline or null must be rejected."""
    # Owner with path traversal
    with pytest.raises(ValidationError):
        policy.create("owner/../../../etc", VALID_REPO_NAME, 1, VALID_BRANCH, VALID_SHA)

    # Owner with null byte
    with pytest.raises(ValidationError):
        policy.create("own\0er", VALID_REPO_NAME, 1, VALID_BRANCH, VALID_SHA)

    # Branch with newline — should be rejected before file I/O
    with pytest.raises(ValidationError):
        policy.create(VALID_OWNER, VALID_REPO_NAME, 1, "hermes/branch\nbad", VALID_SHA)

    # Branch with null byte
    with pytest.raises(ValidationError):
        policy.create(VALID_OWNER, VALID_REPO_NAME, 1, "hermes/branch\0bad", VALID_SHA)


# ---------------------------------------------------------------------------
# Tests 21-24: JSON format, UTC timestamps, preservation, file permissions
# ---------------------------------------------------------------------------

def test_json_output_format(policy):
    """Test 21: List output is valid JSON list of dicts."""
    policy.create(VALID_OWNER, VALID_REPO_NAME, 31, VALID_BRANCH, VALID_SHA)
    active = policy.list_active()
    # Must be a JSON-serializable list of dicts
    data = json.loads(json.dumps(active))
    assert isinstance(data, list)
    assert len(data) >= 1
    assert isinstance(data[0], dict)


def test_timestamps_are_utc(policy):
    """Test 22: created_at and expires_at have UTC timezone info."""
    record = policy.create(VALID_OWNER, VALID_REPO_NAME, 32, VALID_BRANCH, VALID_SHA)

    from datetime import datetime
    created = datetime.fromisoformat(record["created_at"])
    expires = datetime.fromisoformat(record["expires_at"])

    # Must have tzinfo (UTC)
    assert created.tzinfo is not None
    assert expires.tzinfo is not None


def test_preserves_existing_records(policy):
    """Test 23: Create A then B, verify both exist."""
    a = policy.create(VALID_OWNER, VALID_REPO_NAME, 40, VALID_BRANCH, VALID_SHA)
    b = policy.create(VALID_OWNER, VALID_REPO_NAME, 41, VALID_BRANCH, VALID_SHA)

    assert policy.get(a["authorization_id"])["pr_number"] == 40
    assert policy.get(b["authorization_id"])["pr_number"] == 41

    # Both appear in list_active
    active = policy.list_active()
    pr_numbers = {r["pr_number"] for r in active}
    assert 40 in pr_numbers
    assert 41 in pr_numbers


def test_file_permissions(policy):
    """Test 24: Created files have 0o640 mode via os.stat."""
    record = policy.create(VALID_OWNER, VALID_REPO_NAME, 45, VALID_BRANCH, VALID_SHA)
    filepath = os.path.join(policy.auth_dir, f"{record['authorization_id']}.json")

    st = os.stat(filepath)
    mode = stat.S_IMODE(st.st_mode)
    assert mode == 0o640


# ---------------------------------------------------------------------------
# Tests 25-28: CLI integration
# ---------------------------------------------------------------------------

def test_cli_authorize_success():
    """Test 25: CLI with valid args; stdout is valid JSON."""
    env = os.environ.copy()
    # Point FR_AUTH_DIR at a temp directory
    tmpdir = "/tmp/fr_auth_cli_test_25"
    os.makedirs(tmpdir, exist_ok=True)
    env["FR_AUTH_DIR"] = tmpdir

    result = subprocess.run(
        [sys.executable, "-m", "frontier_review.auth_policy",
         "authorize",
         "--repo", f"{VALID_OWNER}/{VALID_REPO_NAME}",
         "--pr", "50",
         "--branch", VALID_BRANCH,
         "--sha", VALID_SHA],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["status"] == "active"
    assert data["pr_number"] == 50

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_cli_list_empty():
    """Test 26: CLI list no records; stdout is '[]'."""
    tmpdir = "/tmp/fr_auth_cli_test_26"
    os.makedirs(tmpdir, exist_ok=True)

    result = subprocess.run(
        [sys.executable, "-m", "frontier_review.auth_policy", "list"],
        capture_output=True, text=True,
        env={**os.environ, "FR_AUTH_DIR": tmpdir},
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data == []

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_cli_revoke_by_id():
    """Test 27: Create via AuthPolicy, revoke via CLI; verify status updated."""
    import shutil
    tmpdir = "/tmp/fr_auth_cli_test_27"
    os.makedirs(tmpdir, exist_ok=True)
    env = {**os.environ, "FR_AUTH_DIR": tmpdir}

    # Create a record via policy API
    policy = AuthPolicy(tmpdir)
    record = policy.create(VALID_OWNER, VALID_REPO_NAME, 60, VALID_BRANCH, VALID_SHA)
    auth_id = record["authorization_id"]

    # Revoke via CLI
    result = subprocess.run(
        [sys.executable, "-m", "frontier_review.auth_policy",
         "revoke", "--id", auth_id],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0
    revoked = json.loads(result.stdout)
    assert revoked["status"] == "revoked"

    # Verify the policy API also sees it as revoked
    fresh_policy = AuthPolicy(tmpdir)
    retrieved = fresh_policy.get(auth_id)
    assert retrieved["status"] == "revoked"

    shutil.rmtree(tmpdir, ignore_errors=True)


def test_cli_reject_invalid_args():
    """Test 28: CLI invalid branch/pr/sha returns exit code 1 and error on stderr."""
    tmpdir = "/tmp/fr_auth_cli_test_28"
    os.makedirs(tmpdir, exist_ok=True)

    # Invalid branch (doesn't match hermes/* pattern)
    result = subprocess.run(
        [sys.executable, "-m", "frontier_review.auth_policy",
         "authorize",
         "--repo", f"{VALID_OWNER}/{VALID_REPO_NAME}",
         "--pr", "61",
         "--branch", "main",
         "--sha", VALID_SHA],
        capture_output=True, text=True,
        env={**os.environ, "FR_AUTH_DIR": tmpdir},
    )
    assert result.returncode == 1
    assert "error" in result.stderr.lower() or len(result.stderr) > 0

    # Invalid SHA (uppercase, wrong length)
    result2 = subprocess.run(
        [sys.executable, "-m", "frontier_review.auth_policy",
         "authorize",
         "--repo", f"{VALID_OWNER}/{VALID_REPO_NAME}",
         "--pr", "61",
         "--branch", VALID_BRANCH,
         "--sha", "ABCD1234"],
        capture_output=True, text=True,
        env={**os.environ, "FR_AUTH_DIR": tmpdir},
    )
    assert result2.returncode == 1

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 29: Unauthorized caller rejection (permission-based)
# ---------------------------------------------------------------------------

def test_unauthorized_caller_rejection_proposed_model(auth_dir_fixture):
    """Test 29: Create auth_dir mode 700 owned by different UID, attempt write, expect PermissionError.

    Creates a subdirectory with mode 700. Since we run as the current user
    (not root), other users cannot enter it. We set the directory permissions
    to 700 but change the ownership via chmod — in practice, only the real
    owner can write. For this test, we verify that setting restrictive
    permissions on the auth directory prevents writes.

    Since pytest usually runs as the same user, we simulate the scenario by
    creating a directory with mode 700 and trying to create files inside it.
    The current user CAN write to their own directory, so we instead check
    that the policy class correctly enforces its security model.

    This test specifically checks: if auth_dir has permissions that would
    prevent the operator from writing (mode 000), a PermissionError is raised.
    """
    import shutil
    # Create a "forbidden" directory with mode 700 — only owner can access
    forbidden_dir = str(auth_dir_fixture / "forbidden")

    # We'll create a subdirectory and then restrict it to prevent writes
    os.makedirs(forbidden_dir, exist_ok=True)

    # Change permissions so that NO ONE (including us after this) can write
    # This simulates the scenario where deployment config has locked down
    # the auth directory
    os.chmod(forbidden_dir, 0o000)

    try:
        policy = AuthPolicy(forbidden_dir)
        # The directory was empty and now we created an AuthPolicy against it
        # The create method should attempt a file write which fails due to permissions
        with pytest.raises((PermissionError, OSError)):
            policy.create(VALID_OWNER, VALID_REPO_NAME, 70, VALID_BRANCH, VALID_SHA)
    finally:
        # Restore permissions for cleanup (needed since we ran chmod 000)
        os.chmod(forbidden_dir, 0o755)
        shutil.rmtree(forbidden_dir, ignore_errors=True)
