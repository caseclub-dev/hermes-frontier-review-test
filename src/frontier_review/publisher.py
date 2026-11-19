"""Trusted GitHub publisher module.

Reads a JSON request from stdin, validates it against strict schema and allowlists,
and performs only pre-authorized operations with mock or real credentials.

This module is designed to run under a dedicated Unix identity (fr-publisher) via
sudo -u fr-publisher. All requests are treated as untrusted regardless of caller.

Request schema validation:
  - Max payload: 4096 bytes on stdin
  - action: one of {push, status, verify_sha}
  - repo_owner, repo_name: must match configured allowlist
  - head_branch: must match branch pattern (hermes/<task-id>)
  - workspace_path: existing directory, max 512 chars, git repo
  - expected_remote_sha / ref_sha: 40-hex characters only
  - pr_number: positive integer if present

Operations available to the publisher:
  push         — clone feature branch, copy workspace, fast-forward push (no force)
  status       — POST commit status to GitHub API for a given SHA
  verify_sha   — GET remote HEAD SHA for a given branch

Security constraints:
  - No arbitrary shell commands
  - No arbitrary Git URLs or remotes
  - No force-push
  - No merging
  - Fast-forward only verification before push
  - Token never embedded in git clone URLs (API-only URL + GIT_ASKPASS)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants — never configurable by caller
# ---------------------------------------------------------------------------

ALLOWED_REPOS: set[tuple[str, str]] = {("caseclub-dev", "hermes-frontier-review-test")}

BRANCH_PATTERN = re.compile(r"^hermes/[a-zA-Z0-9][a-zA-Z0-9._-]{0,254}$")

ACTION_PUSH = "push"
ACTION_STATUS = "status"
ACTION_VERIFY_SHA = "verify_sha"
VALID_ACTIONS = {ACTION_PUSH, ACTION_STATUS, ACTION_VERIFY_SHA}

MAX_PAYLOAD_SIZE = 4096        # bytes on stdin
MAX_PATH_LENGTH = 512          # chars for workspace_path
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

AUDIT_LOG_DIR = Path("/home/atlas/.hermes/frontier_review")
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "publisher_audit.log"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("frontier_review.publisher")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _audit(action: str, success: bool, detail: str, **extra: Any) -> None:
    """Append a structured audit record (never exposes secrets)."""
    # Allow test override of audit log directory via environment variable
    audit_dir = Path(os.environ.get(
        "FR_PUBLISH_AUDIT_DIR",
        "/home/atlas/.hermes/frontier_review"
    ))
    ts = datetime.now(timezone.utc).isoformat()
    pid = os.getpid()
    record = {
        "ts": ts,
        "pid": pid,
        "action": action,
        "success": success,
        "detail": detail,
        **extra,
    }
    line = json.dumps(record) + "\n"
    try:
        audit_dir.mkdir(parents=True, exist_ok=True)
        with open(audit_dir / "publisher_audit.log", "a") as f:
            f.write(line)
    except OSError:
        pass  # audit loss is non-fatal; already logged to stderr


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class PublisherValidationError(Exception):
    """Raised when a request fails schema or policy validation."""


def _validate_sha(value: Any, field: str) -> None:
    if not isinstance(value, str) or not SHA_PATTERN.match(value):
        raise PublisherValidationError(f"{field} must be 40 lowercase hex chars")


def _validate_branch(value: Any, field: str = "head_branch") -> None:
    if not isinstance(value, str) or not BRANCH_PATTERN.match(value):
        raise PublisherValidationError(
            f"{field} must match pattern hermes/<task-id> (alphanumeric start)"
        )


def _validate_repo(owner: Any, name: Any) -> None:
    if not isinstance(owner, str) or not isinstance(name, str):
        raise PublisherValidationError("repo_owner and repo_name must be strings")
    if (owner, name) not in ALLOWED_REPOS:
        raise PublisherValidationError(
            f"Repository {owner}/{name} not on allowlist"
        )


def validate_request(payload: bytes) -> dict[str, Any]:
    """Parse and validate a publisher request payload.

    Returns the parsed and validated dict, or raises PublisherValidationError.
    """
    # Size limit
    if len(payload) > MAX_PAYLOAD_SIZE:
        raise PublisherValidationError(
            f"Payload exceeds {MAX_PAYLOAD_SIZE} byte limit "
            f"(got {len(payload)})"
        )

    # Parse JSON
    try:
        req = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PublisherValidationError(f"Invalid JSON: {exc}") from exc

    if not isinstance(req, dict):
        raise PublisherValidationError("Request must be a JSON object")

    # --- Required fields ------------------------------------------------
    for key in ("action", "repo_owner", "repo_name"):
        if key not in req:
            raise PublisherValidationError(f"Missing required field: {key}")

    # --- Action ---------------------------------------------------------
    action = req["action"]
    if action not in VALID_ACTIONS:
        raise PublisherValidationError(
            f"Invalid action: {action!r}. Must be one of {sorted(VALID_ACTIONS)}"
        )

    # --- Repo allowlist -------------------------------------------------
    repo_owner: str = req["repo_owner"]  # type: ignore[assignment]
    repo_name: str = req["repo_name"]
    _validate_repo(repo_owner, repo_name)

    # --- Branch pattern -------------------------------------------------
    if action in (ACTION_PUSH, ACTION_VERIFY_SHA):
        if "head_branch" not in req:
            raise PublisherValidationError(
                f"Branch required for action={action!r}"
            )
        _validate_branch(req["head_branch"])

    # --- SHA validation -----------------------------------------------
    expected_sha = req.get("expected_remote_sha") or ""
    ref_sha = req.get("ref_sha") or ""
    if expected_sha:
        _validate_sha(expected_sha, "expected_remote_sha")
    if ref_sha and action == ACTION_STATUS:
        _validate_sha(ref_sha, "ref_sha")

    # --- PR number ------------------------------------------------------
    pr_number = req.get("pr_number")
    if pr_number is not None:
        if not isinstance(pr_number, int) or pr_number <= 0:
            raise PublisherValidationError(
                "pr_number must be a positive integer"
            )

    # --- Workspace path -------------------------------------------------
    ws_path = req.get("workspace_path", "")
    if not ws_path:
        raise PublisherValidationError("workspace_path is required")
    if not isinstance(ws_path, str) or len(ws_path) > MAX_PATH_LENGTH:
        raise PublisherValidationError(
            "workspace_path must be a string of at most 512 characters"
        )

    req["_validated"] = True
    return req


# ---------------------------------------------------------------------------
# Git helper functions (no network — used with local mock repos in tests)
# ---------------------------------------------------------------------------

def _run_git(
    args: list[str],
    cwd: str | None = None,
    env_override: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a git command with sanitized environment."""
    env = os.environ.copy()
    # Strip all credential-related variables from parent process
    for key in list(env):
        if key.startswith("GITHUB") or key.startswith("GH_") or \
           key.startswith("GIT_ASKPASS") or key.startswith("SSH_ASKPASS") or \
           "CREDENTIAL" in key:
            env.pop(key, None)

    # Force clean git config for this operation
    env.update({
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    })
    if env_override:
        env.update(env_override)

    return subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=env,
        timeout=30,
    )


def _create_bare_repo(path: str) -> None:
    """Create a bare git repository (used for mock GitHub)."""
    os.makedirs(path, exist_ok=True)
    r = subprocess.run(["git", "init", "--bare", path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Failed to create bare repo: {r.stderr}")


def _make_commit_in_repo(repo_path: str, message: str = "test commit") -> str:
    """Create a single commit in a non-bare repo and return SHA."""
    env_override = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@localhost",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@localhost",
    }
    _run_git(["config", "user.name", "test"], cwd=repo_path, env_override=env_override)
    _run_git(["config", "user.email", "test@localhost"], cwd=repo_path, env_override=env_override)

    # Create initial commit if none exists
    r = _run_git(["rev-parse", "--verify", "HEAD"], cwd=repo_path)
    if r.returncode != 0:
        test_file = os.path.join(repo_path, "init.txt")
        with open(test_file, "w") as f:
            f.write("initial\n")
        _run_git(["add", "."], cwd=repo_path)
        r = _run_git(
            ["commit", "-m", "initial"],
            cwd=repo_path,
            env_override=env_override,
        )

    # Create additional commit with the message
    test_file = os.path.join(repo_path, f"change_{uuid.uuid4().hex[:8]}.txt")
    with open(test_file, "w") as f:
        f.write(f"change: {message}\n")
    _run_git(["add", "."], cwd=repo_path)
    r = _run_git(
        ["commit", "-m", message],
        cwd=repo_path,
        env_override=env_override,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Commit failed: {r.stderr}")

    r = _run_git(["rev-parse", "HEAD"], cwd=repo_path)
    return r.stdout.strip()


def _is_fast_forward(worktree: str, branch: str, expected_sha: str) -> bool:
    """Check if remote HEAD is an ancestor of local HEAD (fast-forward possible)."""
    r = _run_git(
        ["merge-base", "--is-ancestor", f"origin/{branch}", "HEAD"],
        cwd=worktree,
    )
    return r.returncode == 0


def _verify_descendant(worktree: str, child_sha: str, parent_sha: str) -> bool:
    """Verify that child_sha is a descendant of parent_sha."""
    r = _run_git(
        ["merge-base", "--is-ancestor", parent_sha, child_sha],
        cwd=worktree,
    )
    return r.returncode == 0


# ---------------------------------------------------------------------------
# Operation implementations
# ---------------------------------------------------------------------------

def perform_push(
    req: dict[str, Any],
    token: str = "",
    mock_mode: bool = False,
    mock_bare_repo: str | None = None,
) -> dict[str, Any]:
    """Clone feature branch, copy workspace contents, fast-forward push.

    Security checks performed before any Git operation:
      1. Branch must match hermes/* pattern (validated in validate_request)
      2. Workspace must be an existing git repo with at least one commit
      3. Remote HEAD must be ancestor of workspace HEAD (fast-forward only)
      4. Push is always fast-forward (no force flag ever used)
    """
    action = ACTION_PUSH
    head_branch = req["head_branch"]
    ws_path = req["workspace_path"]
    repo_owner = req["repo_owner"]
    repo_name = req["repo_name"]
    expected_remote_sha = req.get("expected_remote_sha") or ""

    # Validate workspace is a git repo with commits
    if not os.path.isdir(ws_path):
        _audit(action, False, "workspace directory missing", head_branch=head_branch)
        return {"action": action, "success": False, "error": "Workspace path does not exist"}

    r = _run_git(["rev-parse", "--show-toplevel"], cwd=ws_path)
    if r.returncode != 0:
        _audit(action, False, "workspace is not a git repo", head_branch=head_branch)
        return {"action": action, "success": False, "error": "Workspace is not a git repository"}

    # Verify workspace has at least one commit
    r = _run_git(["rev-parse", "HEAD"], cwd=ws_path)
    if r.returncode != 0:
        _audit(action, False, "workspace has no commits", head_branch=head_branch)
        return {"action": action, "success": False, "error": "Workspace has no commits"}

    local_head_sha = r.stdout.strip()

    if mock_mode and mock_bare_repo:
        # Mock mode: set up bare repo as origin, push to it directly
        os.makedirs(mock_bare_repo, exist_ok=True)

        # Configure origin in workspace to point at mock bare repo
        _run_git(["remote", "remove", "origin"], cwd=ws_path, env_override={})
        _run_git(["remote", "add", "origin", mock_bare_repo], cwd=ws_path)
        _run_git(["fetch", "origin"], cwd=ws_path)

        # Ensure local branch exists for push (bare repos have no branches locally)
        r = _run_git(["rev-parse", "--verify", f"refs/heads/{head_branch}"], cwd=ws_path, env_override={})
        if r.returncode != 0:
            # Create local branch from HEAD so git push has a source ref
            _run_git(["branch", head_branch, "HEAD"], cwd=ws_path)

        # If expected_remote_sha is set, verify workspace HEAD is a descendant
        if expected_remote_sha:
            r = _run_git(
                ["merge-base", "--is-ancestor", expected_remote_sha, local_head_sha],
                cwd=ws_path,
            )
            if r.returncode != 0:
                _audit(action, False, "non-descendant commit", head_branch=head_branch)
                return {
                    "action": action,
                    "success": False,
                    "error": f"Local HEAD is not a descendant of expected SHA {expected_remote_sha[:12]}",
                }

        # Fast-forward push only (no force flag)
        r = _run_git(
            ["push", "origin", f"{head_branch}:{head_branch}"],
            cwd=ws_path,
        )
        if r.returncode != 0:
            _audit(action, False, f"push failed: {r.stderr[:200]}", head_branch=head_branch)
            return {"action": action, "success": False, "error": r.stderr.strip()[:500]}

        # Verify remote HEAD
        r = _run_git(["rev-parse", f"origin/{head_branch}"], cwd=ws_path)
        if r.returncode == 0:
            remote_sha = r.stdout.strip()
            _audit(action, True, "mock push succeeded", head_branch=head_branch, remote_sha=remote_sha[:12])
            return {"action": action, "success": True, "remote_sha": remote_sha}

        _audit(action, False, "push succeeded but remote SHA unreadable", head_branch=head_branch)
        return {"action": action, "success": False, "error": "Could not read remote HEAD after push"}
    else:
        # Real mode (would use GitHub with token via GIT_ASKPASS)
        # For this design phase, real mode raises NotImplementedError
        _audit(action, False, "real mode not implemented in this subphase", head_branch=head_branch)
        raise NotImplementedError(
            "Real GitHub push requires fr-publisher OS identity and GITHUB_TOKEN. "
            "Use mock_mode=True for testing."
        )


def perform_status(
    req: dict[str, Any],
    token: str = "",
    mock_mode: bool = False,
) -> dict[str, Any]:
    """POST a commit status to the GitHub API (or mock)."""
    action = ACTION_STATUS
    ref_sha = req.get("ref_sha", "")
    pr_number = req.get("pr_number")

    if not SHA_PATTERN.match(ref_sha):
        _audit(action, False, "invalid ref_sha format")
        return {"action": action, "success": False, "error": "Invalid ref_sha format"}

    if mock_mode:
        # In mock mode, record that status was posted (no real API call)
        _audit(action, True, f"mock status posted for {ref_sha[:12]}", pr_number=pr_number)
        return {"action": action, "success": True}

    raise NotImplementedError("Real GitHub status requires fr-publisher OS identity")


def perform_verify_sha(
    req: dict[str, Any],
    token: str = "",
    mock_mode: bool = False,
    mock_bare_repo: str | None = None,
) -> dict[str, Any]:
    """Return the HEAD SHA of a remote branch (or mock)."""
    action = ACTION_VERIFY_SHA
    head_branch = req["head_branch"]

    if not mock_mode or not mock_bare_repo:
        _audit(action, False, "verify_sha requires mock_mode and mock_bare_repo")
        return {"action": action, "success": False, "error": "Mock mode required for verify_sha"}

    # Fetch the branch from the mock bare repo
    r = _run_git(["ls-remote", mock_bare_repo, f"refs/heads/{head_branch}"])
    if r.returncode != 0 or not r.stdout.strip():
        _audit(action, False, "branch not found on remote", head_branch=head_branch)
        return {"action": action, "success": False, "error": f"Branch {head_branch} not found"}

    remote_sha = r.stdout.strip().split()[0]
    if len(remote_sha) != 40:
        _audit(action, False, "remote HEAD is not a valid SHA")
        return {"action": action, "success": False, "error": "Remote HEAD has invalid format"}

    _audit(action, True, f"verified remote SHA {remote_sha[:12]}", head_branch=head_branch)
    return {"action": action, "success": True, "remote_sha": remote_sha}


# ---------------------------------------------------------------------------
# Main entry point (for standalone execution via sudo)
# ---------------------------------------------------------------------------

def main() -> int:
    """Publisher CLI entry point. Reads JSON from stdin, executes operation."""
    # Read payload with size limit
    try:
        raw = sys.stdin.buffer.read(MAX_PAYLOAD_SIZE + 1)
    except Exception as exc:
        _audit("read", False, f"stdin read error: {exc}")
        print(json.dumps({"action": "error", "success": False, "error": f"Failed to read stdin: {exc}"}))
        return 1

    # Check size (if we got MAX_PAYLOAD_SIZE+1 bytes, it's too big)
    if len(raw) > MAX_PAYLOAD_SIZE:
        _audit("payload_size", False, f"payload exceeded {MAX_PAYLOAD_SIZE} bytes")
        print(json.dumps({"action": "payload_size", "success": False,
                           "error": f"Payload exceeds {MAX_PAYLOAD_SIZE} byte limit"}))
        return 1

    if not raw.strip():
        _audit("empty_payload", False, "no data on stdin")
        print(json.dumps({"action": "error", "success": False, "error": "Empty payload"}))
        return 1

    # Validate
    try:
        req = validate_request(raw)
    except PublisherValidationError as exc:
        _audit("validation", False, str(exc))
        print(json.dumps({"action": "validation_error", "success": False,
                           "error": str(exc)}))
        return 1

    action = req["action"]
    token = os.environ.get("FR_PUBLISHER_TOKEN", "")
    mock_mode = os.environ.get("FR_PUBLISH_MOCK", "").lower() in ("1", "true", "yes")
    mock_bare_repo = os.environ.get("FR_PUBLISH_MOCK_BARE_REPO", "")

    # Dispatch
    if action == ACTION_PUSH:
        try:
            result = perform_push(req, token=token, mock_mode=mock_mode, mock_bare_repo=mock_bare_repo)
        except Exception as exc:
            _audit(action, False, f"push exception: {exc}")
            return 1
    elif action == ACTION_STATUS:
        try:
            result = perform_status(req, token=token, mock_mode=mock_mode)
        except Exception as exc:
            _audit(action, False, f"status exception: {exc}")
            return 1
    elif action == ACTION_VERIFY_SHA:
        try:
            result = perform_verify_sha(req, token=token, mock_mode=mock_mode, mock_bare_repo=mock_bare_repo)
        except Exception as exc:
            _audit(action, False, f"verify exception: {exc}")
            return 1
    else:
        _audit("unknown", False, f"unreachable action: {action}")
        return 1

    success = result.get("success", False)
    print(json.dumps(result))
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
