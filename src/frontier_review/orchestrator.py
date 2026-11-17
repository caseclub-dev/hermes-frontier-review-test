"""Orchestrator workflow for Frontier Review (M1 stub).

This module defines the high-level review workflow:
  1. User request → establish requirements
  2. Coder implementation → commit SHA + diff
  3. Mock review via AnthropicClient
  4. If CHANGES_REQUIRED → delegate to Coder for corrections
  5. Re-review corrected code
  6. Once PASS → prepare deliverable

In M1, the workflow is stubbed: no real API calls, no actual delegation,
only the request builder and mock review path are exercised.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .budget import check_budget, estimate_request_cost
from .client import AnthropicClient
from .config import ALLOWED_MODEL, get_max_tokens, get_budget_monthly_usd
from .models import ReviewResult, ReviewStatus, TokenUsage
from .review import build_code_review_prompt, parse_json_response, postprocess_code_result

logger = logging.getLogger(__name__)


class FrontierOrchestrator:
    """Coordinates a review cycle: request → review → correction → re-review.

    In M1: uses MockAnthropicTransport for all reviews. No real API calls.
    No actual Coder delegation (interface only).
    """

    def __init__(self, client: Optional[AnthropicClient] = None) -> None:
        self.client = client or AnthropicClient(model=ALLOWED_MODEL)

    def run_code_review(
        self,
        requirements: str,
        commit_sha: str,
        diff: str,
        code_context: str = "",
        test_evidence: str = "",
    ) -> ReviewResult:
        """Run a single code review through the mock transport path.

        1. Estimate cost and check budget
        2. Build prompt
        3. Send via client (mock transport must be injected)
        4. Parse response → ReviewResult
        5. Record token usage
        """
        # Budget check
        max_output = get_max_tokens()
        all_text = "\n".join([requirements, commit_sha, diff, code_context, test_evidence])
        est_input = len(all_text) // 4  # rough token estimate
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

        # Build and send review request
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

        # Parse and post-process
        raw = parse_json_response(text)
        result = postprocess_code_result(raw, self.client.model)

        # Fill actual token usage
        result.token_usage = TokenUsage(
            input_tokens=usage_dict.get("input_tokens", 0),
            output_tokens=usage_dict.get("output_tokens", 0),
            total_tokens=usage_dict.get("total_tokens", 0),
            thinking_tokens=usage_dict.get("thinking_tokens", 0),
            cache_creation_tokens=usage_dict.get("cache_creation_tokens", 0),
            cache_read_tokens=usage_dict.get("cache_read_tokens", 0),
        )

        return result


# ── Convenience function for M1 testing ──────────────────────────────

def run_review_cycle(
    requirements: str,
    commit_sha: str,
    diff: str,
    mock_transport=None,
) -> ReviewResult:
    """Run a complete review cycle using mock transport.

    Convenience wrapper for testing the full pipeline:
      client → transport → prompt → parse → ReviewResult
    """
    from .transport import MockAnthropicTransport

    if mock_transport is None:
        mock_transport = MockAnthropicTransport()

    # Inject mock transport into client
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
