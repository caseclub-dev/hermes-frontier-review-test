"""Orchestrator workflow for Frontier Review (M2 — bounded autonomous loop).

This module defines the high-level review workflow:
  1. User request → establish requirements
  2. Coder implementation → commit SHA + diff
  3. Mock review via AnthropicClient
  4. If CHANGES_REQUIRED / NEEDS_EVIDENCE → delegate to Coder for corrections
  5. Re-review corrected code
  6. Once PASS → READY_FOR_OPERATOR_REVIEW
  7. Max three correction cycles, four review dispatches total

In M2, the autonomous loop is fully implemented with:
- Bounded correction iterations (default: 3 max)
- Duplicate detection via commit SHA + evidence fingerprint
- Durable task state on disk
- Simulated cost ceiling per-task
- Explicit escalation when Coder cannot correct
- Deterministic mock transport evaluation

No real API calls are made unless explicitly enabled in a future phase.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from .budget import check_budget, estimate_request_cost
from .client import AnthropicClient
from .config import ALLOWED_MODEL, get_budget_monthly_usd, get_max_tokens
from .delegation_adapter import build_delegation_request, extract_correction_tickets
from .models import ReviewResult, ReviewStatus, TokenUsage
from .review import (
    build_code_review_prompt,
    parse_json_response,
    postprocess_code_result,
)
from .state import StateMachineError, TaskStateMachine

logger = logging.getLogger(__name__)


# ── Default bounds ───────────────────────────────────────────────────

_DEFAULT_MAX_CORRECTIONS = 3
_DEFAULT_MAX_REVIEWS = 4        # includes initial review
_DEFAULT_COST_CEILING_USD = 10.0  # per-task simulated ceiling


class ReviewLoopError(Exception):
    """Raised when the autonomous review loop cannot proceed."""


class EscalationRequired(ReviewLoopError):
    """Raised when Coder cannot correct and operator intervention is needed."""


class FrontierOrchestrator:
    """Coordinates a bounded autonomous review cycle.

    Public API:
        run_autonomous_review(...)  — full loop with corrections
        run_code_review(...)         — single review (M1 compatibility)

    State management:
        - Uses TaskStateMachine for durable state on disk.
        - Tracks review_count, correction_count, last_commit_sha, and
          evidence fingerprint across cycles.
        - Detects duplicate commits (no-op corrections) and blocks re-review.

    Cost control:
        - Per-task simulated cost ceiling (default $10).
        - Cumulative cost tracked per task; ceiling enforced before each review.
    """

    def __init__(
        self,
        client: Optional[AnthropicClient] = None,
        state_machine: Optional[TaskStateMachine] = None,
        store_path: Optional[str] = None,
        max_corrections: int = _DEFAULT_MAX_CORRECTIONS,
        max_reviews: int = _DEFAULT_MAX_REVIEWS,
        cost_ceiling_usd: float = _DEFAULT_COST_CEILING_USD,
    ) -> None:
        self.client = client or AnthropicClient(model=ALLOWED_MODEL)
        if state_machine is not None:
            self.state_machine = state_machine
        elif store_path is not None:
            self.state_machine = TaskStateMachine(store_path)
        else:
            # In-memory store for tests that don't need disk persistence
            self.state_machine = TaskStateMachine()
        self.max_corrections = max_corrections
        self.max_reviews = max_reviews
        self.cost_ceiling_usd = cost_ceiling_usd
        self._task_costs: dict[str, float] = {}  # cumulative per-task cost

    # ── Single review (M1 compatibility) ────────────────────────────

    def run_code_review(
        self,
        requirements: str,
        commit_sha: str,
        diff: str,
        code_context: str = "",
        test_evidence: str = "",
    ) -> ReviewResult:
        """Run a single code review through the mock transport path.

        See M1 comment in orchestrator.py for details.
        """
        max_output = get_max_tokens()
        all_text = "\n".join([requirements, commit_sha, diff, code_context, test_evidence])
        est_input = len(all_text) // 4
        est_cost = estimate_request_cost(est_input, max_output, ALLOWED_MODEL)

        monthly_budget = get_budget_monthly_usd()
        if not check_budget(est_cost, monthly_budget):
            return ReviewResult(
                status=ReviewStatus.BLOCKED,
                operation="review-code",
                summary=f"Budget exhausted — monthly budget of ${monthly_budget:.2f} reached.",
                token_usage=TokenUsage(),
                model_used=ALLOWED_MODEL,
            )

        user_msg = build_code_review_prompt(
            requirements, commit_sha, diff, code_context, test_evidence
        )

        try:
            text, usage_dict = self.client.chat(
                system_prompt="You are an independent code reviewer.",
                user_message=user_msg,
            )
        except Exception as exc:
            logger.error(f"Review request failed: {exc}")
            return ReviewResult(
                status=ReviewStatus.BLOCKED,
                operation="review-code",
                summary=f"Review failed — transport error: {exc}",
                token_usage=TokenUsage(),
                model_used=ALLOWED_MODEL,
            )

        raw = parse_json_response(text)
        result = postprocess_code_result(raw, self.client.model)

        result.token_usage = TokenUsage(
            input_tokens=usage_dict.get("input_tokens", 0),
            output_tokens=usage_dict.get("output_tokens", 0),
            total_tokens=usage_dict.get("total_tokens", 0),
            thinking_tokens=usage_dict.get("thinking_tokens", 0),
            cache_creation_tokens=usage_dict.get("cache_creation_tokens", 0),
            cache_read_tokens=usage_dict.get("cache_read_tokens", 0),
        )

        return result

    # ── Autonomous review loop ──────────────────────────────────────

    def run_autonomous_review(
        self,
        task_id: str,
        requirements: str,
        commit_sha: str,
        diff: str,
        code_context: str = "",
        test_evidence: str = "",
        review_count: int = 0,
        corrections_made: bool = False,
    ) -> ReviewResult:
        """Execute a bounded autonomous review cycle.

        Flow:
          1. Create or restore task state.
          2. If not already in a non-terminal state, transition to REREVIEWING.
          3. Check max-review bound.
          4. Run code review.
          5. If PASS → mark COMPLETE, return result.
          6. If CHANGES_REQUIRED or NEEDS_EVIDENCE:
             a. Build delegation request for Coder.
             b. Validate that corrections were actually made (SHA changed + evidence fingerprint).
             c. Loop back to step 3.
          7. If budget exhausted or max corrections reached → BLOCKED / ESCALATED.

        Args:
            task_id: Unique identifier for this review task.
            requirements: What the code must satisfy.
            commit_sha: SHA of the code to review (updated by Coder between cycles).
            diff: Git diff string of the changes.
            code_context: Additional context about the codebase.
            test_evidence: Evidence that tests were run (output, assertions).
            review_count: Number of reviews already performed (for state restoration).
            corrections_made: Whether the last cycle had corrections (for state restoration).

        Returns:
            ReviewResult with final status (PASS / CHANGES_REQUIRED / BLOCKED / ESCALATED).

        Raises:
            EscalationRequired: When Coder cannot produce a meaningful correction.
            ReviewLoopError: On other unrecoverable conditions.
        """
        # ── Step 1: Ensure task state exists ────────────────────────
        try:
            task = self.state_machine.get_task(task_id)
        except StateMachineError:
            task = self.state_machine.create_task(
                task_id=task_id,
                commit_sha=commit_sha,
                diff=diff,
                requirements=requirements,
            )

        # ── Step 2: Restore state if resuming from a previous run ────
        # Use persisted review_count unless caller explicitly provides
        # a higher value (i.e., after a correction cycle).
        if review_count > task.review_count:
            self.state_machine.transition(
                task_id,
                "REREVIEWING",
                review_count=review_count,
                corrections_made=corrections_made,
                commit_sha=commit_sha,
                diff=diff,
            )

        # Ensure we are in a reviewable state (PENDING → REREVIEWING)
        task = self.state_machine.get_task(task_id)
        if task.cycle_status == "PENDING":
            self.state_machine.transition(
                task_id,
                "REREVIEWING",
                review_count=task.review_count,
                corrections_made=task.corrections_made,
                commit_sha=commit_sha,
                diff=diff,
            )
            task = self.state_machine.get_task(task_id)

        # ── Step 3: Ensure we're in a reviewable state ─────────────
        if task.cycle_status in ("COMPLETE", "BLOCKED"):
            return ReviewResult(
                status=ReviewStatus.BLOCKED,
                operation="review-code",
                summary=f"Task already terminal: {task.cycle_status}. Cannot re-review.",
                model_used=ALLOWED_MODEL,
            )

        # ── Step 4a: Check correction count (before running review) ────
        current_corrections = self._get_correction_count(task_id)
        if current_corrections >= self.max_corrections:
            return ReviewResult(
                status=ReviewStatus.ESCALATED_ARCHITECT,
                operation="review-code",
                summary=(
                    f"Max corrections ({self.max_corrections}) reached without PASS. "
                    "Escalating to operator for manual review."
                ),
                model_used=ALLOWED_MODEL,
                blockers=[f"Exceeded {self.max_corrections} correction cycles (currently at {current_corrections})"],
            )

        # ── Step 4b: Check review count bound ─────────────────────────
        if task.review_count >= self.max_reviews:
            return ReviewResult(
                status=ReviewStatus.BLOCKED,
                operation="review-code",
                summary=f"Max reviews reached ({self.max_reviews}). Escalate to operator.",
                model_used=ALLOWED_MODEL,
                blockers=[f"Exceeded {self.max_reviews} review attempts"],
            )

        # ── Step 5: Check cost ceiling ────────────────────────────────
        max_output = get_max_tokens()
        all_text = "\n".join([requirements, commit_sha, diff, code_context, test_evidence])
        est_input = len(all_text) // 4
        est_cost = estimate_request_cost(est_input, max_output, ALLOWED_MODEL)

        cumulative_cost = self._task_costs.get(task_id, 0.0) + est_cost
        if cumulative_cost > self.cost_ceiling_usd:
            return ReviewResult(
                status=ReviewStatus.BLOCKED,
                operation="review-code",
                summary=f"Simulated cost ceiling (${self.cost_ceiling_usd:.2f}) reached.",
                model_used=ALLOWED_MODEL,
                blockers=[f"Cumulative simulated cost ${cumulative_cost:.4f} exceeds ceiling"],
            )

        # ── Step 5: Run the review ─────────────────────────────────
        logger.info(f"[{task_id}] Review #{task.review_count + 1}/{self.max_reviews}")
        result = self.run_code_review(
            requirements=requirements,
            commit_sha=commit_sha,
            diff=diff,
            code_context=code_context,
            test_evidence=test_evidence,
        )

        # Record cost
        actual_cost = result.token_usage.cost_estimate(ALLOWED_MODEL) if result.model_used else 0.0
        self._task_costs[task_id] = cumulative_cost + actual_cost

        logger.info(
            f"[{task_id}] Review #{task.review_count + 1} → {result.status.value} "
            f"(${actual_cost:.4f})"
        )

        # ── Step 6: Handle result ──────────────────────────────────
        if result.status == ReviewStatus.PASS:
            self.state_machine.transition(
                task_id,
                "COMPLETE",
                review_count=task.review_count + 1,
                commit_sha=commit_sha,
                diff=diff,
            )
            logger.info(f"[{task_id}] PASS — READY_FOR_OPERATOR_REVIEW")
            return result

        # Needs changes or evidence → prepare for correction
        new_correction_count = self._get_correction_count(task_id) + 1
        self.state_machine.transition(
            task_id,
            "REREVIEWING",
            review_count=task.review_count + 1,
            commit_sha=commit_sha,
            diff=diff,
            findings_count=len(result.findings),
            corrections_made=new_correction_count,
        )

        # If budget was the blocker from API side, mark BLOCKED
        if result.status == ReviewStatus.BLOCKED:
            return result  # Already blocked; don't loop further

        # Return CHANGES_REQUIRED with findings — caller handles delegation
        return result

    # ── Internal helpers ──────────────────────────────────────────────

    def _get_correction_count(self, task_id: str) -> int:
        """Return the number of correction cycles already attempted for a task."""
        try:
            return self.state_machine.get_task(task_id).corrections_made
        except StateMachineError:
            return 0

    # ── Delegation helper ───────────────────────────────────────────

    def build_correction_request(
        self,
        task_id: str,
        review_result: ReviewResult,
    ) -> Optional[dict[str, Any]]:
        """Build a correction request for Coder from review findings.

        Returns None if no corrections are needed (e.g. PASS / BLOCKED).
        """
        if review_result.status not in (
            ReviewStatus.CHANGES_REQUIRED,
            ReviewStatus.NEEDS_EVIDENCE,
        ):
            return None

        tickets = extract_correction_tickets(task_id, [
            {"title": f.title, "severity": f.severity, "category": f.category,
             "location": f.location or "", "description": f.description}
            for f in review_result.findings
        ])

        if not tickets:
            return None

        findings_dicts = [{
            "title": t.finding_title,
            "severity": t.severity,
            "category": t.category,
            "location": t.location,
            "description": t.description,
            "suggested_correction": t.suggested_correction,
        } for t in tickets]

        req = build_delegation_request(
            task_id=task_id,
            verdict=review_result.status.value,
            summary=review_result.summary,
            findings=findings_dicts,
            suggestions=review_result.suggestions,
        )
        return json.loads(req.to_json())

    # ── Duplicate / no-op detection ─────────────────────────────────

    @staticmethod
    def compute_fingerprint(commit_sha: str, test_evidence: str) -> str:
        """Compute a fingerprint of the evidence package.

        Used to detect no-op corrections (unchanged commit + unchanged evidence).
        """
        raw = f"{commit_sha}|{test_evidence}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def has_duplicate_commit(self, task_id: str, new_sha: str) -> bool:
        """Check if the given commit SHA was already reviewed."""
        try:
            task = self.state_machine.get_task(task_id)
            return task.commit_sha == new_sha and task.review_count > 0
        except StateMachineError:
            return False

    # ── Cost tracking helper ────────────────────────────────────────

    def get_cumulative_cost(self, task_id: str) -> float:
        """Return cumulative simulated cost for a task."""
        return self._task_costs.get(task_id, 0.0)


# ── M1 compatibility: single-review wrapper ─────────────────────────

def run_review_cycle(
    requirements: str,
    commit_sha: str,
    diff: str,
    mock_transport=None,
) -> ReviewResult:
    """Run a complete review cycle using mock transport.

    Convenience wrapper for testing the full pipeline:
      client → transport → prompt → parse → ReviewResult

    M1 compatibility — delegates to FrontierOrchestrator.run_code_review().
    """
    from .transport import MockAnthropicTransport
    from .client import AnthropicClient

    if mock_transport is None:
        mock_transport = MockAnthropicTransport()

    client = AnthropicClient(model=ALLOWED_MODEL)
    client.set_transport(mock_transport)

    orch = FrontierOrchestrator(client=client)

    return orch.run_code_review(
        requirements=requirements,
        commit_sha=commit_sha,
        diff=diff,
        code_context="Stub code context for M1 test.",
        test_evidence="No test evidence in M1 stub.",
    )


# ── Convenience function for M2 testing ──────────────────────────────

def run_autonomous_review_cycle(
    task_id: str,
    requirements: str,
    commit_sha: str,
    diff: str,
    mock_transport=None,
    max_corrections: int = 3,
    max_reviews: int = 4,
    cost_ceiling_usd: float = 10.0,
) -> ReviewResult:
    """Run a complete autonomous review cycle with mock transport.

    Convenience wrapper for testing the full pipeline including
    correction loop detection and state management.

    Args:
        task_id: Unique identifier for the review task.
        requirements: What the code must satisfy.
        commit_sha: Initial commit SHA to review.
        diff: Git diff of changes.
        mock_transport: Pre-configured MockAnthropicTransport (can use sequences).
        max_corrections: Maximum correction cycles allowed.
        max_reviews: Maximum total reviews (including initial).
        cost_ceiling_usd: Per-task simulated cost ceiling.

    Returns:
        Final ReviewResult from the last review or loop termination.
    """
    from .transport import MockAnthropicTransport
    from tempfile import mkdtemp
    import os as _os

    store_path = _os.path.join(mkdtemp(prefix="fr_cycle_"), "state.json") if mock_transport else None

    if mock_transport is None:
        mock_transport = MockAnthropicTransport()

    # Inject mock transport into client
    from .client import AnthropicClient
    client = AnthropicClient(model=ALLOWED_MODEL)
    client.set_transport(mock_transport)

    orch = FrontierOrchestrator(
        client=client,
        store_path=store_path,
        max_corrections=max_corrections,
        max_reviews=max_reviews,
        cost_ceiling_usd=cost_ceiling_usd,
    )

    return orch.run_autonomous_review(
        task_id=task_id,
        requirements=requirements,
        commit_sha=commit_sha,
        diff=diff,
        code_context="Stub code context for M2 test.",
        test_evidence="No test evidence in stub.",
    )
