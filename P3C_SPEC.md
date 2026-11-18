Phase 3C: GitHub PR-Based Review Integration — Implementation Spec

## Overview

Extend Frontier Review to use GitHub draft pull requests as the central, auditable code-delivery artifact. A scheduled controller discovers registered draft PRs, detects new commits, invokes Frontier Review (mock only), dispatches real Coder corrections when needed, and pushes correction commits to the SAME PR branch.

## CRITICAL CONSTRAINTS

- NO real Anthropic API calls ever
- NO cron activation yet (controller tested with --once)
- NO production repo modifications
- NEVER merge PRs, force-push protected branches, or modify main/default directly
- Frontier Review model must NEVER receive GitHub credentials, tokens, repos, or API access
- The controller is the sole entity that interacts with GitHub — it passes diffs + context to the review pipeline; model output is structured feedback only

## Files to Create

### 1. src/frontier_review/pr_adapter.py

A new module providing PR-to-task registration, discovery, and lifecycle management.

#### Data Model

```python
@dataclass
class PRReviewTask(TaskRecord):
    """TaskRecord extended with GitHub PR fields."""
    # ── GitHub PR identity ────────────────────────
    repo_owner: str              # e.g. "caseclub"
    repo_name: str               # e.g. "frontier-review-pr-test"
    pr_number: int               # The draft PR number
    pr_url: str                  # Full URL for audit trail
    head_branch: str             # Feature branch (e.g. "feature/calc-fix")
    base_branch: str             # Target branch (e.g. "main")

    # ── Commit SHA tracking ───────────────────────
    remote_head_sha: str         # Current remote HEAD at time of discovery
    initial_remote_sha: str      # SHA when first registered

    # ── Review gate status ────────────────────────
    review_status: str = "unreviewed"  # "unreviewed", "changes_required", "pass", "blocked", "correction_exhausted"

    # ── Evidence fingerprint ──────────────────────
    evidence_fingerprint: str = ""  # For deduplication

    def to_dict(self):
        ...
    @classmethod
    def from_dict(cls, data):
        ...
```

#### PRRegistry

Manages the ledger of registered PR review tasks. Provides:
- `register_pr(task_id, repo_owner, repo_name, pr_number, pr_url, head_branch, base_branch, head_sha)` — Create registration (idempotent by task_id; raises ValueError if already registered)
- `get_registered_prs()` — List all registered PENDING_REVIEW PRs with review_status="unreviewed"
- `mark_review_pending(task_id)` — Set status to "changes_required", update SHA
- `mark_pass(task_id, sha, evidence_fp)` — Set status to "pass", record fingerprint
- `get_task(task_id)` / `update_task_sha(task_id, new_sha)` — inherited from WorkflowLedger
- `is_duplicate(task_id)` — boolean

Key invariant: no two registrations for the same task_id. Duplicate register() calls return the existing record (idempotent).

#### GitHubClient (mock-only for tests)

```python
class GitHubClient:
    """Interface for scoped GitHub operations.

    In production: calls GitHub REST API with a scoped token (set in config or env).
    In tests/mock mode: returns pre-programmed responses from an internal state dict.

    IMPORTANT: This class is NEVER passed to the review model. Only the controller uses it.
    """

    def __init__(self, repo_owner: str = "", repo_name: str = "", token: str = ""):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.token = token  # Never logged, never passed to model
        self._mock_state: dict = {}  # For testing

    # ── Public API (same for real + mock) ────────────

    def get_remote_head_sha(self, head_branch: str) -> str:
        """Get the current HEAD SHA of a PR's feature branch on GitHub."""
        ...

    def get_pr_diff(self, pr_number: int, base_branch: str) -> str:
        """Get the full unified diff for a PR against its base branch.

        This is what gets passed to Frontier Review as code context.
        The controller extracts this and builds review payloads — NOT the model.
        """
        ...

    def update_status(self, ref_sha: str, state: str, description: str, context: str = "") -> bool:
        """Set a GitHub status check on a specific commit SHA.

        This is the "required check" gate: if review passes, set 'success'.
        If code changes after review (SHA mismatch), set 'failure' with explanation.
        Only works on the exact sha that was reviewed — prevents stale approvals.
        """
        ...

    def update_pr_ready(self, pr_number: int) -> bool:
        """Transition a draft PR to ready-for-review.

        Called only after PASS verdict — the PR becomes visible but is NOT auto-merged.
        """
        ...

    def list_repos(self) -> list[dict]:
        """List repos in the allowlist that have open draft PRs."""
        ...
```

#### Discovery + Eligibility Logic

`discover_reviewable_prs(registry: PRRegistry, gh_client: GitHubClient) -> list[tuple[str, dict]]`:
- For each registered PR with status "unreviewed" or "changes_required":
  - Verify it remains open (draft state OK)
  - Read current remote head SHA via `gh_client.get_remote_head_sha(head_branch)`
  - Compare to registry's recorded `remote_head_sha`
  - If SHA unchanged → skip (no-op detection)
  - If SHA changed → queue for review with new SHA
- PRs that are closed/merged → marked "skipped", no further action
- PRs outside the allowlisted repo → silently ignored
- Returns list of `(task_id, {sha, diff, pr_info})` tuples

## Files to Modify

### 2. /home/atlas/.hermes/scripts/frontier-review-controller.py

Add a `--pr-mode` flag and integration with the PR adapter. When `--pr-mode --once`:
- Load the PRRegistry from the state directory
- Call `discover_reviewable_prs()` to find eligible PRs
- For each eligible PR:
  - Claim atomically via existing ledger claim mechanism
  - Build review context: extract diff via GitHubClient, gather code metadata
  - Run Frontier Review with mock transport (use FR_REVIEW_SEQUENCE)
  - If CHANGES_REQUIRED:
    - Update registry to "changes_required", record findings
    - Dispatch real Coder CLI subprocess targeting the feature branch workspace
    - On success: update SHA in ledger/registry, re-discover for next --once cycle
  - If PASS:
    - Mark registry as "pass"
    - Set GitHub status check to 'success' on the reviewed SHA
    - Update PR from draft → ready (not merged)
  - Release claim

### 3. src/frontier_review/workflow_state.py

Add optional fields to TaskRecord:
- `repo_owner: str = ""`
- `repo_name: str = ""`
- `pr_number: int = 0`
- `pr_url: str = ""`
- `head_branch: str = ""`
- `base_branch: str = ""`
- `remote_head_sha: str = ""`
- `initial_remote_sha: str = ""`
- `review_status: str = "unreviewed"`  (extends the status field)
- `evidence_fingerprint: str = ""`

These fields are additive — they don't break existing callers since TaskRecord.from_dict() filters unknown fields via `known`.

### 4. tests/test_pr_integration.py

Full acceptance test suite using an in-memory mocked GitHub API (no network, no real repos).

## Acceptance Test Scenarios

All tests use a local git workspace and an in-memory mock GitHub client. Tests verify the full logic chain without requiring a live GitHub repo.

### Scenario 1: Full PR Review Cycle
1. Create local git workspace with buggy code → commit SHA_A
2. Mock GitHub API to report SHA_A as remote head, return PR info for pr#42
3. Register PR via PRRegistry.register_pr()
4. Run controller --once → discovers unreviewed PR, runs mock review → CHANGES_REQUIRED
5. Registry updated to "changes_required", findings persisted
6. Controller dispatches Coder CLI subprocess on workspace
7. Coder fixes code → new commit SHA_B
8. Coder pushes correction to feature branch (mock: update local git + GitHub mock SHA)
9. Verify ledger has SHA_B, remote_head_sha updated

### Scenario 2: Mock PASS After Correction
10. Controller --once again → discovers SHA_B (different from SHA_A)
11. Mock review returns PASS
12. Registry marked "pass", evidence_fingerprint recorded
13. GitHub status check set to 'success' on SHA_B
14. PR draft → ready transition recorded

### Scenario 3: Unchanged SHA Skipped
15. Controller --once again with same SHA_B → reports "no_new_commits" in JSON output
16. No duplicate review dispatch, no spurious ledger entries

### Scenario 4: New Commit Invalidates PASS
17. Push new commit SHA_C to feature branch (simulate another developer's change)
18. Controller --once → discovers SHA_C differs from reviewed SHA_B
19. Review runs again with SHA_C → CHANGES_REQUIRED (mock sequence exhausted)
20. Ledger shows re-review was triggered by stale pass

### Scenario 5: Duplicate Cron/Controller Invocations
21. Run two controller invocations concurrently (different workers, same state dir)
22. OS-level flock prevents double-claiming
23. Only one worker processes; other sees "already_claimed" and skips

### Scenario 6: Closed/Merged PRs Skipped
24. Mock GitHub reports PR #42 is now "closed"
25. Controller --once → skips it, logs "pr_closed", no review attempted

### Scenario 7: Non-Allowlisted Repo Ignored
26. Register a task with repo_owner="unknown", repo_name="unauthorized"
27. Controller --once → ignores this PR entirely (not in allowlist)

### Scenario 8: GitHub API Failure Cannot Approve
28. Mock GitHub client raises connection error during get_remote_head_sha
29. Controller catches exception, records "api_failure", does NOT transition to PASS
30. Task remains unreviewed for next retry

### Scenario 9: Correction Limit Blocks PR
31. Feed a sequence of CHANGES_REQUIRED responses (exceeding max_corrections=3)
32. After 3 corrections with no PASS, ledger marks "max_correction_exhausted"
33. Registry shows "correction_exhausted"
34. GitHub status set to 'failure' with context "Max corrections exhausted"

### Scenario 10: No Credentials Exposed to Reviewer
35. During review pipeline execution, verify the model receives:
    - diff content ✓
    - requirements ✓
    - test evidence ✓
    - NO github_token ✓
    - NO repo URL ✓
    - NO api_key ✓
36. Audit log confirms credential fields were stripped before model invocation

### Scenario 11: SHA Verification — PASS Matches Exact Reviewed SHA
37. After PASS on SHA_B, verify ledger.current_sha == review.sha == SHA_B
38. Attempt to approve SHA_C (different from reviewed) → validation fails
39. validate_approval_record() returns approved=False with SHA mismatch reason

## Test Implementation Notes

- Use `unittest.mock.patch` or a simple dict-based mock client for GitHub operations
- Create real local git repos for workspace code + commit creation
- Controller runs in --once mode; test harness manually sets FR_REVIEW_SEQUENCE env var
- All file I/O (ledger, job files) uses temporary directories — no production state touched

## Credential-Boundary Verification

The PR adapter must NEVER pass:
- `token` field of GitHubClient to review payloads
- Any env vars containing GH_*, GITHUB_*, *_TOKEN
- The GitHubClient instance itself to the review pipeline

The controller explicitly strips these before building code_context for Frontier Review.

## Remaining Blockers Before Cron Activation

1. Real GitHub API credentials configured and tested (not mocked)
2. GitHub branch protection rule on base branch requiring "Frontier Review" status check
3. Allowlist of approved repos populated in config.yaml under frontier_review sections
4. Controller cron schedule set to */1 * * * * for monitoring
5. Health monitoring / alerting if controller fails
6. FR_REVIEW_SEQUENCE default changed from mock sequences to real Anthropic model
7. Credential scoping: token must have only repo scope (read:status-checks, pull-requests:write)

## Safe Rollback Procedure

If Phase 3C introduces issues:
1. Revert the pr_adapter.py and workflow_state.py modifications: `git checkout master -- src/frontier_review/pr_adapter.py src/frontier_review/workflow_state.py`
2. Remove tests/test_pr_integration.py
3. Restore controller script from git: `git checkout HEAD~1 -- /home/atlas/.hermes/scripts/frontier-review-controller.py`
4. No state migration needed (JSON ledger is additive-field compatible)
5. The core autonomous loop, transport, and existing pipelines remain untouched

## Delivery Checklist for Coder

- [ ] pr_adapter.py created with PRRegistry, GitHubClient, Discovery functions
- [ ] workflow_state.py extended with new TaskRecord fields (additive only)
- [ ] Controller script updated with --pr-mode support
- [ ] tests/test_pr_integration.py created with all 11 scenarios passing
- [ ] Credential boundary verified: no tokens in review payloads
- [ ] git commit on feature branch, PR created against master
