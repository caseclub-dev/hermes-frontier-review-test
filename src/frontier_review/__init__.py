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
|- ``delegation_adapter`` – adapter for delegating review feedback to Coder
|- ``coder_integration``  – M3: real Coder delegation via delegate_task (M3 milestone)
|- ``orchestrator``  – workflow orchestration with bounded autonomous loop
|- ``autonomous_loop``   – M4: evidence-aware autonomous correction loop
|- ``workflow_state``    – durable task-state persistence (WorkflowLedger)

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
from .coder_integration import (  # noqa: E402
    DelegationResult, CodeReviewFeedback, Severity, FindingCategory,
    build_initial_delegation_request, build_correction_delegation_request,
    parse_delegation_response, GitWorkspaceVerifier, compute_evidence_fingerprint,
    create_disposable_workspace, MockReviewSimulator, CorrectionLoopRunner,
    handle_delegation_failure, is_duplicate_review, is_max_correction_cycles,
)
from .autonomous_loop import (  # noqa: E402
    AutonomousCorrectionLoop, TaskRecord, WorkflowState, Disposition,
)
from .workflow_state import WorkflowLedger  # noqa: E402
from .completion_adapter import CoderCompletionAdapter  # noqa: E402
from .remote_delegator import (  # noqa: E402
    FrontierReviewDelegator,
    create_demo_workspace,
    run_full_m4b_pipeline,
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
    # M3 — real Coder integration
    "DelegationResult",
    "CodeReviewFeedback",
    "Severity",
    "FindingCategory",
    "build_initial_delegation_request",
    "build_correction_delegation_request",
    "parse_delegation_response",
    "GitWorkspaceVerifier",
    "compute_evidence_fingerprint",
    "create_disposable_workspace",
    "MockReviewSimulator",
    "CorrectionLoopRunner",
    "handle_delegation_failure",
    "is_duplicate_review",
    "is_max_correction_cycles",
    # M4 — autonomous correction loop
    "AutonomousCorrectionLoop",
    "TaskRecord",
    "WorkflowState",
    "Disposition",
    "WorkflowLedger",
    # Phase 2 — completion adapter
    "CoderCompletionAdapter",
    # M4C — remote delegator bridge
    "FrontierReviewDelegator",
    "create_demo_workspace",
    "run_full_m4b_pipeline",
]
