"""Review logic — prompt building, response parsing, and post-processing.

All prompts are static strings (no f-strings at import time). The prompt
builder receives review inputs as parameters, constructs the full user
message, and returns it to AnthropicClient.chat().
"""

from __future__ import annotations

import json
from typing import Any


# ── System prompts ───────────────────────────────────────────────────

CODE_REVIEW_SYSTEM = (
    "You are an independent code reviewer for the Hermes Agent fleet. "
    "Your task is to review code changes against stated requirements.\n\n"
    "Review rules:\n"
    "- Verify correctness, security, performance, and completeness\n"
    "- Flag type hint removals, error handling gaps, naming issues\n"
    "- Check for regression risks and edge cases\n"
    "- Be objective — approve good code, flag bad code\n\n"
    "Output format (strict JSON):\n"
    '{"status": "pass|changes_required|needs_evidence|blocked", '
    '"summary": "...", "findings": [...], "requirements_met": [], '
    '"blockers": ["reason this cannot proceed"], '
    '"suggestions": ["improvement recommendations"]}\n---END---\n'
)

FLEET_REVIEW_SYSTEM = (
    "You are a fleet configuration reviewer. Analyze the fleet report for "
    "consistency, security posture, and operational readiness.\n\n"
    'Output format: {"status": "...", "summary": "...", "findings": [...]}\n---END---\n'
)


# ── Prompt builders ─────────────────────────────────────────────────

def build_code_review_prompt(
    requirements: str,
    commit_sha: str,
    diff: str,
    code_context: str,
    test_evidence: str,
) -> str:
    """Build the user message for a code review request."""
    return (
        f"REQUIREMENTS:\n{requirements or 'No specific requirements stated.'}\n\n"
        f"COMMIT SHA: {commit_sha or 'N/A'}\n\n"
        f"DIFF:\n{diff or 'No diff provided.'}\n\n"
        f"CODE CONTEXT:\n{code_context or 'No code context provided.'}\n\n"
        f"TEST EVIDENCE:\n{test_evidence or 'No test evidence provided.'}\n\n"
        "Review this code against the stated requirements. Verify correctness, "
        "check for type hints (must remain), error handling (must be present), "
        "and naming conventions. Confirm tests were actually run."
    )


def build_fleet_review_prompt(fleet_report: str, fleet_issues: str) -> str:
    """Build the user message for a fleet review request."""
    return (
        f"FLEET REPORT:\n{fleet_report or 'No report provided.'}\n\n"
        f"KNOWN ISSUES:\n{fleet_issues or 'No known issues.'}\n\n"
        "Review this fleet configuration for consistency and security."
    )


# ── Response parsing ────────────────────────────────────────────────

def parse_json_response(text: str) -> dict[str, Any]:
    """Parse the structured JSON response from Claude.

    Handles both clean JSON and responses that contain wrapped content.
    Returns a safe fallback dict for unparseable input.
    """
    if not text:
        return {"status": "blocked", "summary": "Empty response.", "findings": []}

    # Find JSON boundaries in the response text
    json_start = max(text.find("{"), 0)
    json_end = text.rfind("}") + 1

    if json_start > 0 or json_end < len(text):
        candidate = text[json_start:json_end]
    else:
        candidate = text

    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return {
            "status": "blocked",
            "summary": f"Could not parse response: {text[:200]}",
            "findings": [],
        }


# ── Post-processing ─────────────────────────────────────────────────

def postprocess_code_result(raw: dict, model_used: str) -> "models.ReviewResult":  # type: ignore[name-defined]
    """Convert raw API response into a ReviewResult."""
    from .models import Finding, ReviewResult, ReviewStatus

    status_str = raw.get("status", "").lower()
    status_map = {
        "pass": ReviewStatus.PASS,
        "changes_required": ReviewStatus.CHANGES_REQUIRED,
        "needs_evidence": ReviewStatus.NEEDS_EVIDENCE,
        "blocked": ReviewStatus.BLOCKED,
    }
    status = status_map.get(status_str, ReviewStatus.BLOCKED)

    findings_list: list[Finding] = []
    for f in raw.get("findings", []):
        if isinstance(f, dict):
            try:
                findings_list.append(Finding(**f))
            except Exception:
                pass  # Malformed finding — skip silently

    return ReviewResult(
        status=status,
        operation="review-code",
        summary=raw.get("summary", "Review completed."),
        findings=findings_list,
        requirements_met=raw.get("requirements_met", []),
        blockers=raw.get("blockers", []),
        suggestions=raw.get("suggestions", []),
        model_used=model_used,
    )


def postprocess_fleet_result(raw: dict, model_used: str) -> "models.ReviewResult":  # type: ignore[name-defined]
    """Convert raw API response into a ReviewResult for fleet reviews."""
    from .models import Finding, ReviewResult, ReviewStatus

    status_str = raw.get("status", "").lower()
    status_map = {
        "pass": ReviewStatus.PASS,
        "changes_required": ReviewStatus.CHANGES_REQUIRED,
        "needs_evidence": ReviewStatus.NEEDS_EVIDENCE,
        "blocked": ReviewStatus.BLOCKED,
    }
    status = status_map.get(status_str, ReviewStatus.BLOCKED)

    findings_list: list[Finding] = []
    for f in raw.get("findings", []):
        if isinstance(f, dict):
            try:
                findings_list.append(Finding(**f))
            except Exception:
                pass

    return ReviewResult(
        status=status,
        operation="review-fleet",
        summary=raw.get("summary", "Fleet review completed."),
        findings=findings_list,
        blockers=raw.get("blockers", []),
        suggestions=raw.get("suggestions", []),
        model_used=model_used,
    )
