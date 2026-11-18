"""Frontier Review — autonomous programming review with mock-first transport.

Design principle: one client interface with interchangeable transports.
- MockAnthropicTransport  — used throughout development and testing.
- RealAnthropicTransport  — future implementation, disabled until authorized.

The request builder, structured response parser, model validation,
review classification, token accounting, budget enforcement, and state
machine are shared between transports. Mock mode never falls through to
the real API.

Target model: Claude Opus 5.5 (no actual API calls in M1).

Modules
-------
- ``config``        – model name, budget caps, allowed models
- ``models``        – Pydantic schemas (ReviewResult, Finding, TokenUsage, …)
- ``transport``     – Transport ABC + MockTransport + RealAnthropicTransport stub
- ``client``        – AnthropicClient with transport injection
- ``review``        – prompt builders, response parsing, post-processing
- ``budget``        – ledger management, cost estimation, budget checks
- ``state``         – persistent task-state machine (PENDING → COMPLETE)
- ``delegation_adapter`` – adapter for delegating review feedback to Coder
- ``orchestrator``  – workflow orchestration with bounded autonomous loop

No live API calls are made unless explicitly enabled in a future phase.
"""

from __future__ import annotations

__version__ = "0.2.0"

# ── Public symbols ---------------------------------------------------

from .config import (  # noqa: E402
    ALLOWED_MODEL, get_model, validate_model, get_max_tokens, get_budget_monthly_usd,
)
from .models import (  # noqa: E402
    ReviewStatus, TokenUsage, Finding, ReviewResult, HealthStatus, CycleStatus,
)
from .client import AnthropicClient  # noqa: E402
from .transport import (  # noqa: E402
    TransportError, MockServerError, MockRateLimitError, MockTimeoutError,
    RealApiDisabledError, ScenarioConfig, MockAnthropicTransport,
)
from .orchestrator import (  # noqa: E402
    FrontierOrchestrator, ReviewLoopError, EscalationRequired, run_review_cycle,
    run_autonomous_review_cycle,
)

__all__ = [
    "ALLOWED_MODEL",
    "get_model",
    "validate_model",
    "get_max_tokens",
    "get_budget_monthly_usd",
    "ReviewStatus",
    "TokenUsage",
    "Finding",
    "ReviewResult",
    "HealthStatus",
    "CycleStatus",
    "AnthropicClient",
    "TransportError",
    "MockServerError",
    "MockRateLimitError",
    "MockTimeoutError",
    "RealApiDisabledError",
    "ScenarioConfig",
    "MockAnthropicTransport",
    "FrontierOrchestrator",
    "ReviewLoopError",
    "EscalationRequired",
    "run_review_cycle",
    "run_autonomous_review_cycle",
]
