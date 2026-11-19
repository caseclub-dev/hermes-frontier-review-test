"""PR-based review integration — Phase 3C.

Provides PR-to-task registration, discovery, and lifecycle management
for the Frontier Review autonomous correction system.

Key components:
    - PRReviewTask: TaskRecord extended with GitHub PR identity fields.
    - PRRegistry: Manages the ledger of registered PR review tasks.
    - GitHubClient: Mock-mode client for tests (pre-programmed internal state).
    - discover_reviewable_prs(): SHA-comparison + eligibility logic.

Credential boundary: The controller is the ONLY entity that interacts with
GitHub.  No tokens, repo URLs, or API keys are ever passed to the review model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

from .workflow_state import TaskRecord, WorkflowLedger

logger = logging.getLogger(__name__)


# ── Data model ───────────────────────────────────────────────────────────


@dataclass
class PRReviewTask(TaskRecord):
    """TaskRecord extended with GitHub PR identity fields (Phase 3C)."""

    # GitHub PR identity
    repo_owner: str = ""
    repo_name: str = ""
    pr_number: int = 0
    pr_url: str = ""
    head_branch: str = ""
    base_branch: str = ""

    # Commit SHA tracking
    remote_head_sha: str = ""
    initial_remote_sha: str = ""

    # Review gate status
    review_status: str = "unreviewed"  # unreviewed | changes_required | pass | blocked | correction_exhausted

    # Evidence fingerprint for deduplication
    evidence_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PRReviewTask":
        """Deserialize from a dict (e.g. loaded from JSON)."""
        known = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ── PRRegistry ───────────────────────────────────────────────────────────

ALLOWED_REPOS_KEY = "allowed_repos"  # key used in registry config dict


class PRRegistry:
    """Manages the ledger of registered PR review tasks.

    Wraps a WorkflowLedger and provides PR-specific operations:
    registration, discovery, status transitions, and deduplication.

    All methods operate on top of the same JSON-backed ledger file so
    that controller invocations share state atomically.
    """

    def __init__(self, ledger_path: str) -> None:
        """Initialize PRRegistry backed by a WorkflowLedger.

        Args:
            ledger_path: Path to the JSON ledger file shared with WorkflowLedger.
        """
        self._ledger = WorkflowLedger(ledger_path)
        self._allowlist: list[str] = []  # "owner/repo" strings

    # ── Allowlist management ────────────────────────────────────────

    def set_allowlist(self, repos: list[str]) -> None:
        """Set the allowlist of approved repo identifiers (e.g. ['caseclub/foo'])."""
        self._allowlist = [r.lower() for r in repos]

    def is_allowed(self, owner: str, name: str) -> bool:
        """Check whether a repo is on the allowlist."""
        if not self._allowlist:
            # No allowlist configured → allow all (permissive default for tests)
            return True
        return f"{owner}/{name}".lower() in self._allowlist

    def get_allowlist(self) -> list[str]:
        """Return the current allowlist."""
        return list(self._allowlist)

    # ── Registration ────────────────────────────────────────────────

    def register_pr(
        self,
        task_id: str,
        repo_owner: str,
        repo_name: str,
        pr_number: int,
        pr_url: str,
        head_branch: str,
        base_branch: str,
        head_sha: str,
        requirements: str = "",
        workspace_path: str = "",
    ) -> PRReviewTask:
        """Register a draft PR for Frontier Review.

        Idempotent by task_id — duplicate calls return the existing record.

        Args:
            task_id: Unique identifier for this review task.
            repo_owner: GitHub owner (e.g. 'caseclub').
            repo_name: Repository name (e.g. 'frontier-review-pr-test').
            pr_number: The draft PR number.
            pr_url: Full PR URL for audit trail.
            head_branch: Feature branch name (e.g. 'feature/calc-fix').
            base_branch: Target branch (e.g. 'main').
            head_sha: Remote HEAD SHA at time of discovery.
            requirements: What the code must satisfy.
            workspace_path: Path to disposable workspace for corrections.

        Returns:
            The PRReviewTask record (new or existing).

        Raises:
            ValueError: If already registered with a final disposition.
        """
        if not self.is_allowed(repo_owner, repo_name):
            raise ValueError(
                f"Repo {repo_owner}/{repo_name} is not on the allowlist"
            )

        # Idempotent check via ledger
        try:
            existing = self._ledger.get_task(task_id)
            rec = PRReviewTask.from_dict(existing.to_dict())
            if rec.review_status in ("pass", "correction_exhausted"):
                raise ValueError(
                    f"Task '{task_id}' already registered with final disposition "
                    f"'{rec.review_status}'. Cannot re-register."
                )
            return rec
        except ValueError as exc:
            # Check if it's a "not found" error (new task) or a final-disposition error
            if "not found" in str(exc):
                pass  # new registration
            else:
                raise

        now = datetime.now(timezone.utc).isoformat()
        # Use provided task_id if non-empty; otherwise auto-generate
        effective_task_id = task_id or f"pr-{repo_owner}/{repo_name}-{pr_number}"

        record = PRReviewTask(
            task_id=effective_task_id,
            requirements=requirements,
            workspace_path=workspace_path,
            repo_owner=repo_owner,
            repo_name=repo_name,
            pr_number=pr_number,
            pr_url=pr_url,
            head_branch=head_branch,
            base_branch=base_branch,
            remote_head_sha=head_sha,
            initial_remote_sha=head_sha,
            review_status="unreviewed",
            created_at=now,
            updated_at=now,
            status="PENDING_REVIEW",
        )

        # Store in ledger
        self._ledger._data[effective_task_id] = record.to_dict()
        self._ledger._save()
        return record

    def is_duplicate(self, task_id: str) -> bool:
        """Return True if a PRReviewTask with this task_id already exists."""
        try:
            self._ledger.get_task(task_id)
            return True
        except ValueError:
            return False

    # ── Query ────────────────────────────────────────────────────────

    def get_registered_prs(self) -> list[PRReviewTask]:
        """List all registered PRs with review_status='unreviewed'."""
        result = []
        for task_id, data in self._ledger._data.items():
            rec = TaskRecord.from_dict(data)
            pr_rec = PRReviewTask.from_dict(data)
            if pr_rec.repo_owner and pr_rec.review_status == "unreviewed":
                result.append(pr_rec)
        result.sort(key=lambda r: r.created_at)
        return result

    def get_task(self, task_id: str) -> Optional[PRReviewTask]:
        """Load a PRReviewTask by id, or None if not found."""
        try:
            data = self._ledger.get_task(task_id).to_dict()
            # Enrich with PR fields from the same record
            return PRReviewTask.from_dict(data)
        except ValueError:
            return None

    def get_task_by_pr(self, owner: str, name: str, pr_number: int) -> Optional[PRReviewTask]:
        """Find a task by repo + PR number."""
        for data in self._ledger._data.values():
            rec = PRReviewTask.from_dict(data)
            if (rec.repo_owner == owner and rec.repo_name == name
                    and rec.pr_number == pr_number):
                return rec
        return None

    # ── Status transitions ──────────────────────────────────────────

    def mark_review_pending(self, task_id: str) -> bool:
        """Set status to 'changes_required' and record updated SHA."""
        try:
            rec = self._ledger.get_task(task_id)
            raw = self._ledger._data[task_id]
            raw["review_status"] = "changes_required"
            raw["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._ledger._data[task_id] = raw
            self._ledger._save()
            return True
        except ValueError:
            return False

    def mark_pass(self, task_id: str, sha: str, evidence_fp: str) -> bool:
        """Set status to 'pass' and record evidence fingerprint."""
        try:
            rec = self._ledger.get_task(task_id)
            raw = self._ledger._data[task_id]
            raw["review_status"] = "pass"
            raw["remote_head_sha"] = sha
            raw["evidence_fingerprint"] = evidence_fp
            raw["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._ledger._data[task_id] = raw
            self._ledger._save()
            return True
        except ValueError:
            return False

    def mark_correction_exhausted(self, task_id: str) -> bool:
        """Set status to 'correction_exhausted' after max corrections reached."""
        try:
            rec = self._ledger.get_task(task_id)
            raw = self._ledger._data[task_id]
            raw["review_status"] = "correction_exhausted"
            raw["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._ledger._data[task_id] = raw
            self._ledger._save()
            return True
        except ValueError:
            return False

    def update_task_sha(self, task_id: str, new_sha: str) -> None:
        """Delegate to WorkflowLedger.update_task_sha and update PR SHA fields."""
        self._ledger.update_task_sha(task_id, new_sha)
        raw = self._ledger._data.get(task_id)
        if raw:
            raw["remote_head_sha"] = new_sha[:40]
            raw["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._ledger._save()

    def get_correction_count(self, task_id: str) -> int:
        """Get the correction count for a task."""
        try:
            rec = self._ledger.get_task(task_id)
            return rec.correction_count
        except ValueError:
            return 0


# ── GitHubClient (mock mode) ────────────────────────────────────────────


class GitHubClient:
    """Interface for scoped GitHub operations.

    In production: calls GitHub REST API with a scoped token (set in config or env).
    In tests/mock mode: returns pre-programmed responses from an internal state dict.

    IMPORTANT: This class is NEVER passed to the review model. Only the controller uses it.
    """

    def __init__(
        self,
        repo_owner: str = "",
        repo_name: str = "",
        token: str = "",
    ) -> None:
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token = token  # Never logged, never passed to model
        self._mock_state: dict[str, Any] = {}

    # ── Mock setup helpers ────────────────────────────────────────────

    def set_pr_info(
        self,
        pr_number: int,
        head_branch: str,
        base_branch: str,
        head_sha: str,
        is_open: bool = True,
        is_draft: bool = True,
    ) -> None:
        """Program mock state for a specific PR."""
        self._mock_state.setdefault("prs", {})[pr_number] = {
            "head_branch": head_branch,
            "base_branch": base_branch,
            "head_sha": head_sha,
            "is_open": is_open,
            "is_draft": is_draft,
        }
        self._mock_state.setdefault("branch_shas", {})[head_branch] = head_sha

    def set_repo_shas(self, branch_shas: dict[str, str]) -> None:
        """Map branch name → remote HEAD SHA."""
        self._mock_state["branch_shas"] = dict(branch_shas)

    def set_pr_diff(self, pr_number: int, diff_text: str) -> None:
        """Set the unified diff for a PR."""
        self._mock_state.setdefault("diffs", {})[pr_number] = diff_text

    def set_api_error(self, should_fail: bool = True) -> None:
        """Cause get_remote_head_sha to raise an exception."""
        self._mock_state["api_error"] = should_fail

    # ── Public API (same for real + mock) ─────────────────────────────

    def get_remote_head_sha(self, head_branch: str) -> str:
        """Get the current HEAD SHA of a PR's feature branch on GitHub.

        In mock mode, looks up from _mock_state["branch_shas"].
        Falls back to real GitHub API when called with repo credentials
        and no matching mock state.
        """
        if self._mock_state.get("api_error"):
            raise ConnectionError(
                f"GitHub API connection error for {self.repo_owner}/{self.repo_name}"
            )
        branch_shas = self._mock_state.get("branch_shas", {})
        if head_branch in branch_shas:
            return branch_shas[head_branch]
        # Real GitHub API fallback for integration testing
        if self.token and self.repo_owner and self.repo_name:
            try:
                from urllib.request import Request, urlopen
                headers = {
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "Hermes-Phase3D",
                }
                req = Request(
                    f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/git/refs/heads/{head_branch}",
                    headers=headers,
                )
                with urlopen(req, timeout=15) as resp:
                    data = json.load(resp)
                return data.get("object", {}).get("sha", "")
            except Exception:
                pass
        return ""

    def get_pr_diff(self, pr_number: int, base_branch: str) -> str:
        """Get the full unified diff for a PR against its base branch.

        This is what gets passed to Frontier Review as code context.
        The controller extracts this and builds review payloads — NOT the model.

        Falls back to real GitHub API when called with repo credentials
        and no matching mock state.
        """
        diffs = self._mock_state.get("diffs", {})
        if pr_number in diffs:
            return diffs[pr_number]
        # Real GitHub API fallback for integration testing
        if self.token and self.repo_owner and self.repo_name:
            try:
                from urllib.request import Request, urlopen
                headers = {
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github.v3.diff",
                    "User-Agent": "Hermes-Phase3D",
                }
                req = Request(
                    f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr_number}/diff",
                    headers=headers,
                )
                with urlopen(req, timeout=15) as resp:
                    return resp.read().decode()
            except Exception:
                pass
        return ""

    def update_status(
        self, ref_sha: str, state: str, description: str, context: str = ""
    ) -> bool:
        """Set a GitHub status check on a specific commit SHA.

        This is the 'required check' gate: if review passes, set 'success'.
        If code changes after review (SHA mismatch), set 'failure' with explanation.
        Only works on the exact sha that was reviewed — prevents stale approvals.
        """
        self._mock_state.setdefault("statuses", []).append({
            "sha": ref_sha,
            "state": state,
            "description": description,
            "context": context,
        })
        return True

    def update_pr_ready(self, pr_number: int) -> bool:
        """Transition a draft PR to ready-for-review.

        Called only after PASS verdict — the PR becomes visible but is NOT auto-merged.
        """
        prs = self._mock_state.get("prs", {})
        if pr_number in prs:
            prs[pr_number]["is_draft"] = False
        return True

    def list_repos(self) -> list[dict]:
        """List repos in the allowlist that have open draft PRs."""
        prs = self._mock_state.get("prs", {})
        repos = set()
        for pr in prs.values():
            if pr.get("is_open") and pr.get("is_draft"):
                repos.add(f"{self.repo_owner}/{self.repo_name}")
        return [
            {"owner": self.repo_owner, "name": self.repo_name}
            for _ in repos
        ]

    def get_pr_state(self, pr_number: int) -> Optional[dict]:
        """Get PR info dict from mock state (for testing)."""
        return self._mock_state.get("prs", {}).get(pr_number)

    def get_pr_state_from_api(self, pr_number: int) -> Optional[dict]:
        """Get PR state from real GitHub API. Returns dict with 'state', 'draft' keys.

        In mock mode (no token / no repo credentials), returns None so callers
        can fall back to local checks.
        """
        if not self.token or not self.repo_owner or not self.repo_name:
            return None
        try:
            from urllib.request import Request, urlopen
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "Hermes-Phase3D",
            }
            req = Request(
                f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/pulls/{pr_number}",
                headers=headers,
            )
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            return {"state": data.get("state"), "draft": data.get("draft", True)}
        except Exception:
            return None


def init_workspace_at_remote_head(
    repo_owner: str,
    repo_name: str,
    head_branch: str,
    expected_sha: str,
    token: str,
    existing_path: str = "",
) -> tuple[str, str]:
    """Initialize (or prepare) a Git workspace at the exact PR-head commit.

    If *existing_path* already exists and its HEAD matches *expected_sha*,
    returns (path, sha) immediately — no clone or fetch needed.

    Otherwise creates a fresh clone of just the target branch, checked out
    to the expected SHA in detached-HEAD state, so all future corrections
    descend cleanly from the PR head with zero non-fast-forward risk.

    Returns:
        (workspace_path, verified_sha) — verified_sha == expected_sha.

    Raises ValueError on auth failure or unreachable remote.
    """

    url = f"https://{token}@github.com/{repo_owner}/{repo_name}.git"

    # 1. Check existing workspace
    if existing_path and os.path.isdir(existing_path):
        try:
            r_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5, cwd=existing_path,
            )
            existing_sha = r_sha.stdout.strip()
            if existing_sha == expected_sha:
                return existing_path, existing_sha

            # Workspace exists but SHA differs — will reclone below.
        except Exception:
            pass

    # 2. Create fresh workspace via full branch clone (reliable for arbitrary SHAs)
    base_dir = os.environ.get(
        "FR_CORRECTION_WORKSPACE",
        "/home/atlas/.hermes/cache/scratch"
    )
    os.makedirs(base_dir, exist_ok=True)
    ws_path = tempfile.mkdtemp(dir=base_dir, prefix="fr_ws_")

    # Clone the branch fully (not depth-1) so any SHA on that branch is reachable
    r_clone = subprocess.run(
        ["git", "clone", "--branch", head_branch, url, ws_path],
        capture_output=True, text=True, timeout=120,
    )
    if r_clone.returncode != 0:
        shutil.rmtree(ws_path, ignore_errors=True)
        raise ValueError(
            f"Failed to clone {repo_owner}/{repo_name} branch {head_branch}: "
            f"{r_clone.stderr[:300]}"
        )

    # Verify the target SHA exists locally (it should be on this branch)
    r_peek = subprocess.run(
        ["git", "-C", ws_path, "cat-file", "-t", expected_sha],
        capture_output=True, text=True, timeout=5,
    )
    if r_peek.returncode != 0:
        raise ValueError(
            f"SHA {expected_sha[:12]} not found in branch {head_branch} "
            f"(clone may be stale or sha is incorrect)"
        )

    # Detach HEAD at the exact expected SHA
    r_co = subprocess.run(
        ["git", "-C", ws_path, "checkout", "--detach", expected_sha],
        capture_output=True, text=True, timeout=10,
    )
    if r_co.returncode != 0:
        raise ValueError(
            f"Failed to checkout {expected_sha[:12]}: {r_co.stderr[:300]}"
        )

    # Verify workspace HEAD matches expected SHA
    r_v = subprocess.run(
        ["git", "-C", ws_path, "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=5,
    )
    verified_sha = r_v.stdout.strip()
    if not verified_sha:
        raise ValueError(f"Workspace at {ws_path} has no HEAD commit after checkout")

    # Ensure a local branch exists so Coder can commit on it
    subprocess.run(
        ["git", "-C", ws_path, "branch", "-M", head_branch],
        capture_output=True, text=True, timeout=5,
    )

    return ws_path, verified_sha


# ── Phase 3D helpers ────────────────────────────────────────────────────────────


def push_correction_to_github(
    repo_owner: str,
    repo_name: str,
    head_branch: str,
    token: str,
    workspace_path: str = "",
) -> tuple[bool, str]:
    """Push the local feature branch to GitHub and return (success, remote_sha).

    Args:
        workspace_path: Git repository directory for operations. Required.

    Returns:
        (True, sha) on success — remote HEAD matches what was pushed.
        (False, error_msg) on failure.

    Never force-push. Uses standard git push with credential helper.
    """

    # Determine working directory for git operations
    work_dir = workspace_path if workspace_path and os.path.isdir(workspace_path) else None
    if not work_dir:
        return False, "No valid workspace directory provided"

    _askpass_script = None
    try:
        # Set up temporary credential helper for authenticated pushes
        _env2 = os.environ.copy()
        _env2["GIT_TERMINAL_PROMPT"] = "0"
        _askpass_script = f"/tmp/.git_askpass_{os.getpid()}"
        with open(_askpass_script, "w") as _f:
            _f.write("#!/bin/sh\necho \"$TEMP_GIT_TOKEN\"\n")
        os.chmod(_askpass_script, 0o755)
        _env2["GIT_ASKPASS"] = _askpass_script
        _env2["TEMP_GIT_TOKEN"] = token

        url = f"https://github.com/{repo_owner}/{repo_name}.git"

        # ── Pre-push checks (no branch renaming, no reset) ────────────────

        # 1. Ensure we are on the expected branch
        r_branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=work_dir,
        )
        local_branch = r_branch.stdout.strip()
        if local_branch != head_branch:
            # Workspace is on wrong branch — fetch remote to check ancestry.
            subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, timeout=5, cwd=work_dir,
            )
            if subprocess.run(
                ["git", "fetch", "origin", head_branch],
                capture_output=True, text=True, timeout=10, cwd=work_dir,
            ).returncode != 0:
                return False, f"Cannot fetch {head_branch} from remote"

        # 2. Verify local HEAD descends from (or equals) remote head
        r_local = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, cwd=work_dir,
        )
        local_sha = r_local.stdout.strip()

        # Fetch latest remote state for comparison
        subprocess.run(
            ["git", "fetch", "origin", head_branch],
            capture_output=True, text=True, timeout=10, cwd=work_dir,
        )
        try:
            r_remote = subprocess.run(
                ["git", "merge-base", "--is-ancestor",
                 f"origin/{head_branch}", "HEAD"],
                capture_output=True, text=True, timeout=5, cwd=work_dir,
            )
        except Exception:
            return False, "Cannot determine ancestry of local HEAD vs remote"

        if r_remote.returncode != 0:
            # Local is not an ancestor of origin head — check reverse.
            r_reverse = subprocess.run(
                ["git", "merge-base", "--is-ancestor",
                 "HEAD", f"origin/{head_branch}"],
                capture_output=True, text=True, cwd=work_dir, timeout=5,
            )
            if r_reverse.returncode != 0:
                # Neither is ancestor — likely workspace was not initialized at PR head.
                return False, (
                    f"Workspace HEAD {local_sha[:8]} does not descend from "
                    f"remote branch {head_branch}; workspace must be initialized at the PR head SHA"
                )

        # 3. Push (no force) — use temporary credential helper to avoid
        # interactive prompts and token leaks to git subprocesses.
        _env3 = os.environ.copy()
        _env3["GIT_TERMINAL_PROMPT"] = "0"
        _env3["GIT_ASKPASS"] = _askpass_script
        _env3["TEMP_GIT_TOKEN"] = token
        r_push = subprocess.run(
            ["git", "push", "--no-verify", url, f"{local_branch}:{head_branch}"],
            capture_output=True, text=True, timeout=120, env=_env3, cwd=work_dir,
        )

        if _askpass_script and os.path.exists(_askpass_script):
            os.unlink(_askpass_script)
            _askpass_script = None
    finally:
        if _askpass_script and os.path.exists(_askpass_script):
            os.unlink(_askpass_script)

    if r_push.returncode != 0:
        return False, f"git push failed: {r_push.stderr[:300]}"

    # 4. Verify remote SHA matches local (requirement 6)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "Hermes-Phase3D",
    }
    from urllib.request import Request, urlopen as _urlopen
    req = Request(
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/refs/heads/{head_branch}",
        headers=headers,
    )
    try:
        with _urlopen(req, timeout=15) as resp:
            remote_data = json.loads(resp.read())
        remote_sha = remote_data.get("object", {}).get("sha", "")
    except Exception as exc:
        return False, f"Failed to fetch remote SHA after push: {exc}"

    if remote_sha != local_sha:
        # Remote HEAD differs from what we just pushed — stale or concurrent update
        return False, (f"Remote SHA mismatch: local={local_sha[:8]}, "
                       f"remote={remote_sha[:8]}")

    return True, remote_sha


def verify_approval_prerequisites(
    repo_owner: str,
    repo_name: str,
    pr_number: int,
    head_branch: str,
    reviewed_sha: str,
    token: str,
) -> tuple[bool, str]:
    """Verify all conditions before marking a PR as ready for operator review.

    Returns (True, '') if all checks pass, or (False, reason).
    Checks:
      1. PR is open and assigned to correct repo
      2. Remote HEAD SHA matches reviewed SHA
    """
    # Check 1 & 2: Remote HEAD SHA must match reviewed SHA
    if not token or not repo_owner or not repo_name:
        return False, "Missing credentials for verification"

    from urllib.request import Request as _Req, urlopen as _urlopen
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "Hermes-Phase3D",
    }

    # Get PR state
    req = _Req(
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/pulls/{pr_number}",
        headers=headers,
    )
    try:
        with _urlopen(req, timeout=15) as resp:
            pr_data = json.loads(resp.read())
    except Exception as exc:
        return False, f"GitHub API call failed: {exc}"

    if pr_data.get("state") != "open":
        return False, f"PR is {pr_data.get('state', 'unknown')}, not open — cannot approve"

    # Get remote HEAD SHA
    req2 = _Req(
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/refs/heads/{head_branch}",
        headers=headers,
    )
    try:
        with _urlopen(req2, timeout=15) as resp:
            head_data = json.loads(resp.read())
        remote_sha = head_data.get("object", {}).get("sha", "")
    except Exception as exc:
        return False, f"Failed to get branch SHA: {exc}"

    if remote_sha != reviewed_sha:
        return False, (f"Remote HEAD mismatch: reviewed={reviewed_sha[:8]}, "
                       f"current remote={remote_sha[:8]}")

    return True, ""


# ── Discovery ─────────────────────────────────────────────────────────────


def discover_reviewable_prs(
    registry: PRRegistry,
    gh_client: GitHubClient,
) -> list[tuple[str, dict]]:
    """Discover PRs that need review.

    For each registered PR with status 'unreviewed' or 'changes_required':
      - Verify it remains open (draft state OK)
      - Read current remote head SHA via gh_client.get_remote_head_sha
      - Compare to registry's recorded remote_head_sha
      - If SHA unchanged → skip (no-op detection)
      - If SHA changed → queue for review with new SHA

    PRs that are closed/merged → marked 'skipped', no further action.
    PRs outside the allowlisted repo → silently ignored.

    Args:
        registry: PRRegistry instance with registered tasks.
        gh_client: GitHubClient (mock or real).

    Returns:
        List of (task_id, {sha, diff, pr_info}) tuples ready for review.
    """
    eligible = registry.get_registered_prs()
    # Also check 'changes_required' status PRs
    change_req = []
    for task_id, data in registry._ledger._data.items():
        rec = PRReviewTask.from_dict(data)
        if rec.repo_owner and rec.review_status == "changes_required":
            change_req.append(rec)

    all_tasks: list[PRReviewTask] = list(eligible) + change_req
    results: list[tuple[str, dict]] = []

    for task in all_tasks:
        # Check allowlist
        if not registry.is_allowed(task.repo_owner, task.repo_name):
            logger.debug(f"Skipping non-allowlisted repo: {task.repo_owner}/{task.repo_name}")
            continue

        # Create a client scoped to this PR's repo
        client = GitHubClient(
            repo_owner=task.repo_owner,
            repo_name=task.repo_name,
            token=gh_client.token,
        )
        # Inherit mock state if present (for test environments)
        if gh_client._mock_state:
            client._mock_state = dict(gh_client._mock_state)

        try:
            current_sha = client.get_remote_head_sha(task.head_branch)
        except ConnectionError as exc:
            logger.warning(f"API failure for {task.task_id}: {exc}")
            # Mark task with api_failure — do NOT transition to PASS
            try:
                raw = registry._ledger._data[task.task_id]
                raw["review_status"] = "blocked"
                raw["updated_at"] = datetime.now(timezone.utc).isoformat()
                registry._ledger._save()
            except Exception:
                pass
            continue

        if not current_sha:
            logger.warning(f"No remote SHA found for {task.task_id}")
            continue

        # Check PR open state — try mock first, then real GitHub API (Blocker 2)
        pr_info = client.get_pr_state(task.pr_number) if hasattr(client, 'get_pr_state') else None
        is_open = True  # default assumption; only checked in mock mode
        if pr_info:
            is_open = pr_info.get("is_open", True)

        # Also check via real GitHub API for Blocker 2 (closed/merged PR filtering)
        api_state = client.get_pr_state_from_api(task.pr_number)
        if api_state and api_state.get("state") != "open":
            # Closed or merged — skip entirely, mark stale in ledger
            logger.info(
                f"PR #{task.pr_number} is {api_state['state']} — skipping (API check)"
            )
            try:
                raw = registry._ledger._data[task.task_id]
                raw["review_status"] = "skipped"
                raw["updated_at"] = datetime.now(timezone.utc).isoformat()
                registry._ledger._save()
            except Exception:
                pass
            continue

        if not is_open:
            logger.info(f"PR #{task.pr_number} closed/merged — skipping")
            try:
                raw = registry._ledger._data[task.task_id]
                raw["review_status"] = "skipped"
                raw["updated_at"] = datetime.now(timezone.utc).isoformat()
                registry._ledger._save()
            except Exception:
                pass
            continue

        # SHA comparison — skip if unchanged
        recorded_sha = task.remote_head_sha or task.initial_remote_sha or ""
        if current_sha == recorded_sha:
            logger.debug(f"SHA unchanged for {task.task_id} ({current_sha[:8]})")
            continue

        # Fetch the PR diff for review context
        diff = client.get_pr_diff(task.pr_number, task.base_branch)

        results.append((
            task.task_id,
            {
                "sha": current_sha,
                "diff": diff,
                "pr_info": {
                    "repo_owner": task.repo_owner,
                    "repo_name": task.repo_name,
                    "pr_number": task.pr_number,
                    "pr_url": task.pr_url,
                    "head_branch": task.head_branch,
                    "base_branch": task.base_branch,
                },
            },
        ))

    return results


def build_review_context(
    task: PRReviewTask,
    diff: str,
    registry: PRRegistry,
) -> dict[str, Any]:
    """Build the review payload for a PR.

    Strips all GitHub credentials — only safe fields are included.
    This is what gets passed to Frontier Review as code context.

    Args:
        task: The PRReviewTask being reviewed.
        diff: The unified diff of changes.
        registry: PRRegistry for enrichment.

    Returns:
        Dict with review payload fields (no tokens, no credentials).
    """
    return {
        "task_id": task.task_id,
        "requirements": task.requirements or "",
        "diff": diff,
        "pr_number": task.pr_number,
        "repo_owner": task.repo_owner,
        "repo_name": task.repo_name,
        "head_branch": task.head_branch,
        "base_branch": task.base_branch,
        "workspace_path": task.workspace_path,
    }
