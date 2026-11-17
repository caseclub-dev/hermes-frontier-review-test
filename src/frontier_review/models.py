"""Pydantic data models for Frontier Review.

All review results, findings, and token accounting use these schemas.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ReviewStatus(str, Enum):
    """Possible outcomes of a code review."""

    PASS = "PASS"
    CHANGES_REQUIRED = "CHANGES_REQUIRED"
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
    BLOCKED = "BLOCKED"
    ESCALATED_ARCHITECT = "ESCALATED_ARCHITECT"


class TokenUsage(BaseModel):
    """Token usage tracking for API calls and cost estimation."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    thinking_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    def cost_estimate(self, model_name: str) -> float:
        """Estimate cost based on token counts and pricing.

        Pricing (per 1M tokens):
          - claude-opus-5-5: input $15.00, output $75.00, cache_read $3.75
        Non-Opus models are rejected.
        """
        from .config import validate_model

        if not validate_model(model_name):
            raise ValueError(
                f"Cost estimation requires the allowed Opus model; got '{model_name}'"
            )

        input_cost = (self.input_tokens + self.cache_creation_tokens) / 1_000_000 * 15.0
        output_cost = self.output_tokens / 1_000_000 * 75.0
        cache_read_cost = self.cache_read_tokens / 1_000_000 * 3.75

        return round(input_cost + output_cost + cache_read_cost, 6)


class Finding(BaseModel):
    """A single finding from a code review."""

    title: str
    severity: str  # "HIGH", "MEDIUM", "LOW"
    category: str  # e.g. "correctness", "security", "performance"
    location: Optional[str] = None
    description: str


class ReviewResult(BaseModel):
    """Structured result of a code/fleet review."""

    status: ReviewStatus
    operation: str
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    requirements_met: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    model_used: str = ""
    request_id: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __str__(self) -> str:
        cost = 0.0
        if self.model_used:
            cost = self.token_usage.cost_estimate(self.model_used)
        return (
            f"ReviewResult({self.status.value}, {self.operation}, "
            f"{len(self.findings)} findings, ${cost:.6f})"
        )


class HealthStatus(BaseModel):
    """Health check result for the Frontier Review system."""

    status: str = "ok"
    model: str = ""
    budget_monthly_usd: float = 0.0
    is_offline: bool = False
    message: str = ""


class CycleStatus(str, Enum):
    """Tracking states for the autonomous review cycle."""

    PENDING = "PENDING"
    REREVIEWING = "REREVIEWING"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"
