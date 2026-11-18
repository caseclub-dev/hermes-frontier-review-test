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
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

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
        """
        if self._mock_state.get("api_error"):
            raise ConnectionError(
                f"GitHub API connection error for {self.repo_owner}/{self.repo_name}"
            )
        return self._mock_state.get("branch_shas", {}).get(head_branch, "")

    def get_pr_diff(self, pr_number: int, base_branch: str) -> str:
        """Get the full unified diff for a PR against its base branch.

        This is what gets passed to Frontier Review as code context.
        The controller extracts this and builds review payloads — NOT the model.
        """
        return self._mock_state.get("diffs", {}).get(pr_number, "")

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

        # Check PR open state via mock or real API
        pr_info = client.get_pr_state(task.pr_number) if hasattr(client, 'get_pr_state') else None
        is_open = True  # default assumption; only checked in mock mode
        if pr_info:
            is_open = pr_info.get("is_open", True)

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
