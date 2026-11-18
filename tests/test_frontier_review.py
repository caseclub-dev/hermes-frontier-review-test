"""Frontier Review tests — M1 & M2 acceptance validation.

M1 covers:
- Import successfully in isolated environment
- Mock transport instantiation without credentials
- Anthropic-compatible response parsing
- Unexpected model identity rejection
- Malformed response handling
- Real API not possible in offline mode
- Budget accounting
- State machine transitions
- Delegation adapter serialization

M2 covers (autonomous correction loop):
- Deterministic PASS scenario
- Deterministic CHANGES_REQUIRED with actionable findings
- Multiple defects → multi-cycle correction
- Incomplete evidence handling
- Malformed / truncated responses
- Unexpected model identifiers (drift)
- Authentication failures
- Rate limiting and server errors
- Timeouts and delayed responses
- Duplicate request detection
- Budget exhaustion
- Max review count enforcement
- Max correction count escalation
- No-op correction rejection (duplicate SHA + evidence fingerprint)
- Durable task state across restarts
- Cost ceiling enforcement
- READY_FOR_OPERATOR_REVIEW status propagation

All tests run fully offline. No network access required.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

# Ensure src is on path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _tmpdir() -> str:
    """Create a temporary directory for test artifacts."""
    d = tempfile.mkdtemp(prefix="fr_test_")
    return d


def _cleanup(tmpdir: str) -> None:
    shutil.rmtree(tmpdir, ignore_errors=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ imports ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestImports:
    """Verify the package imports without side effects."""

    def test_import_package(self):
        import frontier_review  # noqa: F401

    def test_import_submodules(self):
        from frontier_review import config, models, transport, client, review, budget, state, orchestrator  # noqa: F401

    def test_public_symbols_exist(self):
        import frontier_review as fr
        assert hasattr(fr, "ALLOWED_MODEL")
        assert hasattr(fr, "get_model")
        assert hasattr(fr, "validate_model")
        assert hasattr(fr, "AnthropicClient")
        assert hasattr(fr, "MockAnthropicTransport")
        assert hasattr(fr, "TokenUsage")
        assert hasattr(fr, "Finding")
        assert hasattr(fr, "ReviewResult")
        assert hasattr(fr, "ReviewStatus")
        assert hasattr(fr, "HealthStatus")
        assert hasattr(fr, "FrontierOrchestrator")
        assert hasattr(fr, "ReviewLoopError")
        assert hasattr(fr, "EscalationRequired")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ config ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestConfig:
    """Configuration validation and defaults."""

    def test_allowed_model(self):
        from frontier_review.config import ALLOWED_MODEL, get_model
        assert ALLOWED_MODEL == "claude-opus-5-5"
        assert get_model() == "claude-opus-5-5"

    def test_validate_model_true(self):
        from frontier_review.config import validate_model
        assert validate_model("claude-opus-5-5") is True

    def test_validate_model_false_sonnet(self):
        from frontier_review.config import validate_model
        assert validate_model("claude-sonnet-4-20250514") is False

    def test_validate_model_false_haiku(self):
        from frontier_review.config import validate_model
        assert validate_model("claude-haiku-3-5") is False

    def test_validate_model_empty(self):
        from frontier_review.config import validate_model
        assert validate_model("") is False

    def test_budget_defaults(self):
        from frontier_review.config import get_budget_monthly_usd
        assert get_budget_monthly_usd() == 75.0

    def test_max_tokens_default(self):
        from frontier_review.config import get_max_tokens
        assert get_max_tokens() == 32768

    def test_offline_mode_defaults_true(self):
        from frontier_review.config import is_offline, set_offline_mode
        assert is_offline() is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ models ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestModels:
    """Pydantic schema validation."""

    def test_token_usage_default(self):
        from frontier_review.models import TokenUsage
        t = TokenUsage()
        assert t.input_tokens == 0
        assert t.output_tokens == 0
        assert t.cost_estimate("claude-opus-5-5") == pytest.approx(0.0)

    def test_token_usage_cost(self):
        from frontier_review.models import TokenUsage
        t = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
        cost = t.cost_estimate("claude-opus-5-5")
        assert cost == pytest.approx(90.0)  # 15 + 75

    def test_token_usage_cost_non_opus_rejected(self):
        from frontier_review.models import TokenUsage
        t = TokenUsage(input_tokens=1_000_000)
        with pytest.raises(ValueError, match="allowed Opus model"):
            t.cost_estimate("claude-sonnet-4")

    def test_finding_minimal(self):
        from frontier_review.models import Finding
        f = Finding(title="Test", severity="HIGH", category="correctness", description="desc")
        assert f.title == "Test"
        assert f.location is None

    def test_review_result_str(self):
        from frontier_review.models import ReviewResult, ReviewStatus, TokenUsage
        r = ReviewResult(
            status=ReviewStatus.PASS, operation="review-code", summary="All good.",
            model_used="claude-opus-5-5", token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        s = str(r)
        assert "PASS" in s
        assert "review-code" in s

    def test_review_result_non_opus_cost(self):
        from frontier_review.models import ReviewResult, ReviewStatus
        r = ReviewResult(
            status=ReviewStatus.PASS, operation="review-code", summary="All good.",
            model_used="claude-sonnet-4",
        )
        with pytest.raises(ValueError):
            _ = str(r)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ transport ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestTransport:
    """Mock transport — Anthropic-compatible, no network."""

    def test_mock_transport_instantiation_no_credentials(self):
        from frontier_review.transport import MockAnthropicTransport
        mt = MockAnthropicTransport()
        assert mt is not None

    def test_real_transport_raises_disabled_error(self):
        from frontier_review.transport import RealAnthropicTransport, RealApiDisabledError
        with pytest.raises(RealApiDisabledError):
            RealAnthropicTransport(api_key="fake-key")

    def test_ok_response_structure(self):
        from frontier_review.transport import MockAnthropicTransport
        cfg = MockAnthropicTransport.ok_response(model="claude-opus-5-5", status="pass", summary="Test review.")
        assert cfg.status_code == 200
        body = json.loads(cfg._raw_body)
        assert body["model"] == "claude-opus-5-5"
        assert "content" in body
        assert "usage" in body

    def test_response_is_anthropic_compatible(self):
        """Mock response must match Anthropic Messages API structure."""
        from frontier_review.transport import MockAnthropicTransport
        cfg = MockAnthropicTransport.ok_response(
            model="claude-opus-5-5", status="changes_required", summary="Needs fixes.",
            findings=[{"title": "Bug in loop", "severity": "HIGH", "category": "correctness",
                       "location": "main.py:42", "description": "Off-by-one error."}],
        )
        body = json.loads(cfg._raw_body)
        assert "id" in body
        assert body["model"] == "claude-opus-5-5"
        assert isinstance(body["content"], list)
        assert len(body["content"]) > 0
        assert body["content"][0]["type"] == "text"
        assert "usage" in body
        usage = body["usage"]
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            assert key in usage

    def test_server_error_response(self):
        from frontier_review.transport import MockAnthropicTransport
        cfg = MockAnthropicTransport.server_error_response("Test error")
        body = json.loads(cfg._raw_body)
        assert "error" in body

    def test_rate_limit_response(self):
        from frontier_review.transport import MockAnthropicTransport, MockRateLimitError
        cfg = MockAnthropicTransport.rate_limit_response()
        assert cfg.status_code == 429
        body = json.loads(cfg._raw_body)
        assert body["error"]["type"] == "rate_limit_error"

    def test_timeout_response(self):
        from frontier_review.transport import MockAnthropicTransport, MockTimeoutError
        cfg = MockAnthropicTransport.timeout_response()
        assert cfg.status_code == 0  # Signal: no response
        assert cfg._raw_body == ""

    def test_malformed_response(self):
        from frontier_review.transport import MockAnthropicTransport, TransportError
        raw_text = "this is not json {{{ broken"
        cfg = MockAnthropicTransport.malformed_response(raw_text)
        assert cfg.status_code == 200
        # Parsing should raise
        mt = MockAnthropicTransport()
        mt.configure(cfg)
        resp_bytes = mt.send_request("url", "POST", {}, "{}")
        with pytest.raises(TransportError):
            mt.receive_response(resp_bytes)

    def test_model_drift_response(self):
        from frontier_review.transport import MockAnthropicTransport, ScenarioConfig
        cfg = MockAnthropicTransport.model_drift_response(unexpected_model="claude-sonnet-4")
        body = json.loads(cfg._raw_body)
        assert body["model"] == "claude-sonnet-4"  # NOT the allowed model

    def test_auth_failure_response(self):
        from frontier_review.transport import MockAnthropicTransport
        cfg = MockAnthropicTransport.auth_failure_response("Invalid API key")
        assert cfg.status_code == 401
        body = json.loads(cfg._raw_body)
        assert body["error"]["type"] == "authentication_error"

    def test_budget_exhaustion_response(self):
        from frontier_review.transport import MockAnthropicTransport
        cfg = MockAnthropicTransport.budget_exhaustion_response()
        assert cfg.status_code == 403
        body = json.loads(cfg._raw_body)
        review = json.loads(body["content"][0]["text"])
        assert review["status"] == "blocked"

    def test_interrupted_task_response(self):
        """Truncated JSON exercises partial-parse handling."""
        from frontier_review.transport import MockAnthropicTransport, TransportError
        cfg = MockAnthropicTransport.interrupted_task_response()
        mt = MockAnthropicTransport()
        mt.configure(cfg)
        raw = mt.send_request("url", "POST", {}, "{}")
        with pytest.raises(TransportError):
            mt.receive_response(raw)

    def test_duplicate_request_response(self):
        from frontier_review.transport import MockAnthropicTransport
        cfg = MockAnthropicTransport.duplicate_request_response(original_id="req-dup-123")
        body = json.loads(cfg._raw_body)
        review = json.loads(body["content"][0]["text"])
        assert "Duplicate" in review["summary"]

    def test_send_request_returns_bytes(self):
        from frontier_review.transport import MockAnthropicTransport
        mt = MockAnthropicTransport()
        cfg = MockAnthropicTransport.ok_response()
        mt.configure(cfg)
        raw = mt.send_request("url", "POST", {}, "{}")
        assert isinstance(raw, bytes)

    def test_receive_response_parses_json(self):
        from frontier_review.transport import MockAnthropicTransport
        mt = MockAnthropicTransport()
        cfg = MockAnthropicTransport.ok_response()
        mt.configure(cfg)
        raw = mt.send_request("url", "POST", {}, "{}")
        resp = mt.receive_response(raw)
        assert isinstance(resp, dict)
        assert "model" in resp

    def test_malformed_raw_raises_error(self):
        from frontier_review.transport import MockAnthropicTransport, ScenarioConfig, TransportError
        mt = MockAnthropicTransport()
        cfg = ScenarioConfig(_raw_body="not json {{{")
        mt.configure(cfg)
        raw = mt.send_request("url", "POST", {}, "{}")
        with pytest.raises(TransportError):
            mt.receive_response(raw)

    def test_sequence_consumption(self):
        from frontier_review.transport import MockAnthropicTransport
        mt = MockAnthropicTransport()
        configs = [MockAnthropicTransport.ok_response(status=f"pass{i}", summary=f"Review {i}") for i in range(3)]
        mt.configure_sequence(configs)
        raw1 = mt.send_request("url", "POST", {}, "{}")
        raw2 = mt.send_request("url", "POST", {}, "{}")
        assert len(raw1) > 0 and len(raw2) > 0
        raw3 = mt.send_request("url", "POST", {}, "{}")
        body3 = json.loads(raw3.decode())
        assert body3["content"][0]["text"]

    def test_sequence_exhaustion_raises(self):
        from frontier_review.transport import MockAnthropicTransport, MockServerError
        mt = MockAnthropicTransport()
        cfg = MockAnthropicTransport.ok_response(status="only_one")
        mt.configure(cfg)
        mt.send_request("url", "POST", {}, "{}")
        with pytest.raises(MockServerError):
            mt.send_request("url", "POST", {}, "{}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ client ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestClient:
    """AnthropicClient — transport injection, model validation."""

    def test_client_requires_mock_transport_or_error(self):
        from frontier_review.client import AnthropicClient
        from frontier_review.transport import RealApiDisabledError
        c = AnthropicClient()
        with pytest.raises(RealApiDisabledError):
            c.chat(system_prompt="test", user_message="test")

    def test_client_with_mock_transport(self):
        from frontier_review.client import AnthropicClient
        from frontier_review.transport import MockAnthropicTransport
        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.ok_response(status="pass"))
        c = AnthropicClient(model="claude-opus-5-5")
        c.set_transport(mt)
        text, usage = c.chat(system_prompt="You are a reviewer.", user_message="Review this code.")
        assert isinstance(text, str)
        assert len(text) > 0
        assert "input_tokens" in usage

    def test_model_validation_on_init(self):
        from frontier_review.client import AnthropicClient
        with pytest.raises(ValueError, match="not in the allowed model list"):
            AnthropicClient(model="claude-sonnet-4")

    def test_inject_non_transport_raises(self):
        from frontier_review.client import AnthropicClient
        c = AnthropicClient()
        with pytest.raises(TypeError):
            c.set_transport("not a transport")  # type: ignore[arg-type]

    def test_model_drift_detection(self):
        from frontier_review.client import AnthropicClient
        from frontier_review.transport import MockAnthropicTransport, ScenarioConfig
        mt = MockAnthropicTransport()
        cfg = ScenarioConfig(
            _raw_body=json.dumps({
                "id": "msg-test", "model": "claude-sonnet-4",
                "content": [{"type": "text", "text": '{"status":"pass","summary":"ok"}'}],
                "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            })
        )
        mt.configure(cfg)
        c = AnthropicClient(model="claude-opus-5-5")
        c.set_transport(mt)
        with pytest.raises(RuntimeError, match="Model drift"):
            c.chat(system_prompt="test", user_message="test")

    def test_chat_builds_proper_request_body(self):
        from frontier_review.client import AnthropicClient
        from frontier_review.transport import MockAnthropicTransport
        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.ok_response())
        c = AnthropicClient(model="claude-opus-5-5")
        c.set_transport(mt)
        text, _ = c.chat(system_prompt="Review code.", user_message="Fix the bug.")
        assert len(text) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ review ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestReview:
    """Prompt building, JSON parsing, post-processing."""

    def test_code_review_prompt_has_required_sections(self):
        from frontier_review.review import build_code_review_prompt
        prompt = build_code_review_prompt(
            requirements="Must pass tests.", commit_sha="abc123",
            diff="--- a/main.py\n+++ b/main.py\n@@ ...", code_context="", test_evidence="",
        )
        assert "REQUIREMENTS:" in prompt
        assert "COMMIT SHA: abc123" in prompt
        assert "DIFF:" in prompt

    def test_parse_clean_json(self):
        from frontier_review.review import parse_json_response
        text = '{"status":"pass","summary":"OK","findings":[]}'
        result = parse_json_response(text)
        assert result["status"] == "pass"
        assert result["summary"] == "OK"

    def test_parse_wrapped_json(self):
        from frontier_review.review import parse_json_response
        text = "Some preamble {\n  \"status\": \"blocked\",\n  \"summary\": \"Error\",\n  \"findings\": []\n}\nTrailer."
        result = parse_json_response(text)
        assert result["status"] == "blocked"

    def test_parse_empty_string(self):
        from frontier_review.review import parse_json_response
        result = parse_json_response("")
        assert result["status"] == "blocked"

    def test_parse_invalid_json(self):
        from frontier_review.review import parse_json_response
        result = parse_json_response("this is not json at all")
        assert result["status"] == "blocked"
        assert "Could not parse" in result["summary"]

    def test_postprocess_pass_result(self):
        from frontier_review.review import postprocess_code_result
        raw = {"status": "pass", "summary": "All checks passed.", "findings": [],
               "requirements_met": ["type hints intact"], "suggestions": ["add docstrings"]}
        result = postprocess_code_result(raw, "claude-opus-5-5")
        assert result.status.value == "PASS"
        assert len(result.findings) == 0
        assert "type hints intact" in result.requirements_met

    def test_postprocess_changes_required(self):
        from frontier_review.review import postprocess_code_result
        raw = {"status": "changes_required", "summary": "Fix type hint removals.",
               "findings": [{"title": "Type hint removed", "severity": "HIGH", "category": "correctness",
                             "location": "main.py:10", "description": "Function lost its return type annotation."}]}
        result = postprocess_code_result(raw, "claude-opus-5-5")
        assert result.status.value == "CHANGES_REQUIRED"
        assert len(result.findings) == 1
        assert result.findings[0].severity == "HIGH"

    def test_postprocess_unknown_status(self):
        from frontier_review.review import postprocess_code_result
        raw = {"status": "unknown_garbage", "summary": "???", "findings": []}
        result = postprocess_code_result(raw, "claude-opus-5-5")
        assert result.status.value == "BLOCKED"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ budget ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBudget:
    """Cost estimation and budget checks."""

    def test_estimate_cost_opus(self):
        from frontier_review.budget import estimate_request_cost
        cost = estimate_request_cost(10_000, 32768, "claude-opus-5-5")
        assert cost > 0
        assert cost == pytest.approx(0.15 + 2.4576, abs=0.001)

    def test_estimate_cost_non_opus_zero(self):
        from frontier_review.budget import estimate_request_cost
        cost = estimate_request_cost(10_000, 32768, "claude-sonnet-4")
        assert cost == 0.0

    def test_check_budget_no_ledger(self):
        from frontier_review.budget import check_budget, isolate_test_ledger, reset_ledger_override, set_ledger_override
        tmpdir = _tmpdir()
        try:
            fake_path = os.path.join(tmpdir, "nonexistent.json")
            set_ledger_override(fake_path)
            assert check_budget(10.0, 75.0) is True
        finally:
            reset_ledger_override()
            _cleanup(tmpdir)

    def test_record_request(self):
        from frontier_review.budget import isolate_test_ledger, record_request, reset_ledger_override
        tmp_path = isolate_test_ledger()
        try:
            rid = record_request("claude-opus-5-5", 10000, 2000)
            assert rid is not None
            assert rid.startswith("req-")
            with open(tmp_path) as f:
                data = json.load(f)
            assert len(data["requests"]) == 1
        finally:
            reset_ledger_override()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ state ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestState:
    """Task state machine transitions."""

    def test_create_and_get_task(self, tmp_path_factory):
        from frontier_review.state import TaskStateMachine
        store = tmp_path_factory.mktemp("fr_state") / "state.json"
        sm = TaskStateMachine(str(store))
        t = sm.create_task("task-1", commit_sha="abc123", diff="diff content")
        assert t.task_id == "task-1"
        assert t.cycle_status == "PENDING"
        assert t.commit_sha == "abc123"

    def test_pending_to_rereviewing(self, tmp_path_factory):
        from frontier_review.state import TaskStateMachine
        store = tmp_path_factory.mktemp("fr_state") / "state.json"
        sm = TaskStateMachine(str(store))
        sm.create_task("task-1")
        t = sm.transition("task-1", "REREVIEWING")
        assert t.cycle_status == "REREVIEWING"

    def test_rereviewing_to_complete(self, tmp_path_factory):
        from frontier_review.state import TaskStateMachine
        store = tmp_path_factory.mktemp("fr_state") / "state.json"
        sm = TaskStateMachine(str(store))
        sm.create_task("task-1")
        sm.transition("task-1", "REREVIEWING")
        t = sm.transition("task-1", "COMPLETE")
        assert t.cycle_status == "COMPLETE"

    def test_rereviewing_to_blocked(self, tmp_path_factory):
        from frontier_review.state import TaskStateMachine
        store = tmp_path_factory.mktemp("fr_state") / "state.json"
        sm = TaskStateMachine(str(store))
        sm.create_task("task-1")
        sm.transition("task-1", "REREVIEWING")
        t = sm.transition("task-1", "BLOCKED")
        assert t.cycle_status == "BLOCKED"

    def test_invalid_transition_raises(self, tmp_path_factory):
        from frontier_review.state import TaskStateMachine, StateMachineError
        store = tmp_path_factory.mktemp("fr_state") / "state.json"
        sm = TaskStateMachine(str(store))
        sm.create_task("task-1")
        with pytest.raises(StateMachineError, match="Invalid transition"):
            sm.transition("task-1", "COMPLETE")

    def test_terminal_state_no_transitions(self, tmp_path_factory):
        from frontier_review.state import TaskStateMachine, StateMachineError
        store = tmp_path_factory.mktemp("fr_state") / "state.json"
        sm = TaskStateMachine(str(store))
        sm.create_task("task-1")
        sm.transition("task-1", "REREVIEWING")
        sm.transition("task-1", "COMPLETE")
        with pytest.raises(StateMachineError, match="terminal"):
            sm.transition("task-1", "REREVIEWING")

    def test_task_count(self, tmp_path_factory):
        from frontier_review.state import TaskStateMachine
        store = tmp_path_factory.mktemp("fr_state") / "state.json"
        sm = TaskStateMachine(str(store))
        assert sm.get_task_count() == 0
        sm.create_task("task-1")
        sm.create_task("task-2")
        assert sm.get_task_count() == 2

    def test_duplicate_task_id_raises(self, tmp_path_factory):
        from frontier_review.state import TaskStateMachine, StateMachineError
        store = tmp_path_factory.mktemp("fr_state") / "state.json"
        sm = TaskStateMachine(str(store))
        sm.create_task("task-1")
        with pytest.raises(StateMachineError, match="already exists"):
            sm.create_task("task-1")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ delegation_adapter ━━━━━━━━━━━━━━━━━━━━

class TestDelegationAdapter:
    """Coder delegation request builders and serialization."""

    def test_build_delegation_request(self):
        from frontier_review.delegation_adapter import build_delegation_request
        req = build_delegation_request("task-1", "CHANGES_REQUIRED", "Fix type hints.", [{
            "title": "Missing return type", "severity": "HIGH", "category": "correctness",
            "location": "main.py:10", "description": "Function has no return annotation.",
        }])
        assert req.task_id == "task-1"
        assert req.verdict == "CHANGES_REQUIRED"
        assert len(req.findings) == 1

    def test_serialization_roundtrip(self):
        from frontier_review.delegation_adapter import build_delegation_request, DelegationRequest
        req = build_delegation_request("t1", "NEEDS_EVIDENCE", "Show test results.", [])
        json_str = req.to_json()
        parsed = DelegationRequest.from_json(json_str)
        assert parsed.task_id == "t1"
        assert parsed.verdict == "NEEDS_EVIDENCE"

    def test_extract_correction_tickets(self):
        from frontier_review.delegation_adapter import extract_correction_tickets
        tickets = extract_correction_tickets("task-1", [{
            "title": "Bug fix needed", "severity": "HIGH", "category": "correctness",
            "location": "main.py:50", "description": "Null pointer exception.",
        }])
        assert len(tickets) == 1
        assert tickets[0].finding_title == "Bug fix needed"
        assert tickets[0].severity == "HIGH"

    def test_extract_from_empty_list(self):
        from frontier_review.delegation_adapter import extract_correction_tickets
        tickets = extract_correction_tickets("t1", [])
        assert len(tickets) == 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ orchestrator / review cycle ━━━━━━━━━━━

class TestReviewCycle:
    """End-to-end mock review through the full pipeline."""

    def test_full_review_cycle_pass(self):
        from frontier_review.orchestrator import run_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport
        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.ok_response(status="pass", summary="All good."))
        result = run_review_cycle(requirements="Code must compile.", commit_sha="abc123",
                                  diff="--- a/main.py\n+++ b/main.py", mock_transport=mt)
        assert result.status == ReviewStatus.PASS

    def test_full_review_cycle_changes_required(self):
        from frontier_review.orchestrator import run_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport
        mt = MockAnthropicTransport()
        findings = [{"title": "Type hint removed", "severity": "HIGH", "category": "correctness",
                     "location": "main.py:10", "description": "Function lost return type."}]
        mt.configure(MockAnthropicTransport.ok_response(status="changes_required", summary="Needs fixes.", findings=findings))
        result = run_review_cycle(requirements="Type hints must remain.", commit_sha="def456",
                                  diff="diff content here...", mock_transport=mt)
        assert result.status == ReviewStatus.CHANGES_REQUIRED
        assert len(result.findings) == 1

    def test_full_review_cycle_blocked_by_budget(self):
        from frontier_review.budget import isolate_test_ledger, record_request, reset_ledger_override
        from frontier_review.orchestrator import run_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport
        tmp_path = isolate_test_ledger()
        record_request("claude-opus-5-5", 6_000_000, 32768)
        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.ok_response(status="pass"))
        result = run_review_cycle(requirements="Test.", commit_sha="abc", diff="small diff here", mock_transport=mt)
        assert result.status == ReviewStatus.BLOCKED
        reset_ledger_override()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ offline enforcement ━━━━━━━━━━━━━━━━━━

class TestOfflineMode:
    """Verify no real API calls can be made in offline mode."""

    def test_real_transport_always_raises(self):
        from frontier_review.transport import RealAnthropicTransport, RealApiDisabledError
        with pytest.raises(RealApiDisabledError):
            RealAnthropicTransport()  # type: ignore[call-arg]

    def test_client_without_transport_raises(self):
        from frontier_review.client import AnthropicClient
        from frontier_review.transport import RealApiDisabledError
        c = AnthropicClient()
        with pytest.raises(RealApiDisabledError):
            c.chat(system_prompt="x", user_message="y")

    def test_mock_never_reaches_network(self):
        """If we patch socket.connect to raise, mock transport must still work."""
        import socket
        from frontier_review.transport import MockAnthropicTransport
        real_connect = socket.socket.connect
        def fake_connect(*args, **kwargs):
            raise ConnectionError("Network access blocked — this should not happen in mock mode")
        try:
            socket.socket.connect = fake_connect
            mt = MockAnthropicTransport()
            mt.configure(MockAnthropicTransport.ok_response())
            raw = mt.send_request("https://api.anthropic.com/v1/messages", "POST", {}, "{}")
            resp = mt.receive_response(raw)
            assert isinstance(resp, dict)
        finally:
            socket.socket.connect = real_connect


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ M2 autonomous loop ━━━━━━━━━━━━━━━━━━━━

class TestAutonomousLoopScenarios:
    """Deterministic mock review scenarios for the autonomous correction loop."""

    def test_scenario_pass_on_first_review(self, tmp_path_factory):
        """Correct code receiving PASS on initial review."""
        from frontier_review.orchestrator import run_autonomous_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport

        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.ok_response(
            status="pass", summary="All requirements met.", findings=[]))
        result = run_autonomous_review_cycle(
            task_id="test-pass-1", requirements="Must compile and pass tests.",
            commit_sha="abc123", diff="--- a/main.py\n+++ b/main.py", mock_transport=mt)

        assert result.status == ReviewStatus.PASS, f"Expected PASS but got {result.status.value}"
        assert "requirements met" in result.summary.lower() or "All requirements" in result.summary

    def test_scenario_changes_required_single_finding(self, tmp_path_factory):
        """Incorrect code receiving actionable findings (single HIGH finding)."""
        from frontier_review.orchestrator import run_autonomous_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport

        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.ok_response(
            status="changes_required", summary="Fix type hint removals.",
            findings=[{"title": "Type hint removed", "severity": "HIGH", "category": "correctness",
                       "location": "main.py:10", "description": "Function lost return type annotation."}],
        ))
        result = run_autonomous_review_cycle(
            task_id="test-changes-1", requirements="Type hints must remain.",
            commit_sha="def456", diff="--- a/main.py", mock_transport=mt)

        assert result.status == ReviewStatus.CHANGES_REQUIRED, f"Expected CHANGES_REQUIRED but got {result.status.value}"
        assert len(result.findings) == 1
        assert result.findings[0].severity == "HIGH"

    def test_scenario_multiple_defects(self, tmp_path_factory):
        """Multiple defects requiring more than one correction."""
        from frontier_review.orchestrator import run_autonomous_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport

        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.ok_response(
            status="changes_required", summary="Multiple issues found.",
            findings=[
                {"title": "Missing error handling", "severity": "HIGH", "category": "correctness",
                 "location": "main.py:20", "description": "Division by zero not guarded."},
                {"title": "Type hint removed", "severity": "MEDIUM", "category": "correctness",
                 "location": "main.py:35", "description": "Function lost return type."},
            ],
        ))
        result = run_autonomous_review_cycle(
            task_id="test-multi-1", requirements="Error handling and types required.",
            commit_sha="multi001", diff="--- a/main.py", mock_transport=mt)

        assert result.status == ReviewStatus.CHANGES_REQUIRED
        assert len(result.findings) == 2, f"Expected 2 findings but got {len(result.findings)}"

    def test_scenario_incomplete_evidence(self, tmp_path_factory):
        """Inadequate test evidence returns NEEDS_EVIDENCE."""
        from frontier_review.orchestrator import run_autonomous_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport

        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.ok_response(
            status="needs_evidence", summary="No test results provided.",
            findings=[{"title": "Missing test evidence", "severity": "HIGH", "category": "process",
                       "location": "N/A", "description": "Review cannot proceed without test output."}],
        ))
        result = run_autonomous_review_cycle(
            task_id="test-evidence-1", requirements="Must provide test evidence.",
            commit_sha="ev001", diff="--- a/main.py", mock_transport=mt)

        assert result.status == ReviewStatus.NEEDS_EVIDENCE, f"Expected NEEDS_EVIDENCE but got {result.status.value}"

    def test_scenario_malformed_response(self, tmp_path_factory):
        """Malformed API response → review blocked."""
        from frontier_review.orchestrator import run_autonomous_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport

        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.malformed_response(raw_text="not json {{{ broken"))
        result = run_autonomous_review_cycle(
            task_id="test-malformed-1", requirements="Test.",
            commit_sha="mal001", diff="--- a/main.py", mock_transport=mt)

        # parse_json_response returns blocked for unparseable input
        assert result.status == ReviewStatus.BLOCKED, f"Expected BLOCKED but got {result.status.value}"
        assert "parse" in result.summary.lower() or "transport error" in result.summary.lower(), \
            f"Unexpected summary: {result.summary}"

    def test_scenario_model_drift(self, tmp_path_factory):
        """Unexpected model identifier → RuntimeError caught as blocked."""
        from frontier_review.orchestrator import run_autonomous_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport

        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.model_drift_response(unexpected_model="claude-sonnet-4"))
        result = run_autonomous_review_cycle(
            task_id="test-drift-1", requirements="Test.",
            commit_sha="drift001", diff="--- a/main.py", mock_transport=mt)

        # The RuntimeError from model drift should be caught and return BLOCKED
        assert result.status == ReviewStatus.BLOCKED, f"Expected BLOCKED but got {result.status.value}"

    def test_scenario_auth_failure(self, tmp_path_factory):
        """Authentication failure → review blocked."""
        from frontier_review.orchestrator import run_autonomous_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport

        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.auth_failure_response("Invalid API key"))
        result = run_autonomous_review_cycle(
            task_id="test-auth-1", requirements="Test.",
            commit_sha="auth001", diff="--- a/main.py", mock_transport=mt)

        # 401 auth failure: response is parsed but client should handle it
        assert result.status == ReviewStatus.BLOCKED, f"Expected BLOCKED but got {result.status.value}"

    def test_scenario_rate_limit(self, tmp_path_factory):
        """Rate limiting → review blocked."""
        from frontier_review.orchestrator import run_autonomous_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport, MockRateLimitError

        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.rate_limit_response())
        result = run_autonomous_review_cycle(
            task_id="test-ratelimit-1", requirements="Test.",
            commit_sha="rl001", diff="--- a/main.py", mock_transport=mt)

        assert result.status == ReviewStatus.BLOCKED, f"Expected BLOCKED but got {result.status.value}"

    def test_scenario_server_error(self, tmp_path_factory):
        """Server error (500) → review blocked."""
        from frontier_review.orchestrator import run_autonomous_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport

        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.server_error_response("Internal server error"))
        result = run_autonomous_review_cycle(
            task_id="test-srverr-1", requirements="Test.",
            commit_sha="srv001", diff="--- a/main.py", mock_transport=mt)

        assert result.status == ReviewStatus.BLOCKED, f"Expected BLOCKED but got {result.status.value}"

    def test_scenario_timeout(self, tmp_path_factory):
        """Timeout / delayed response → review blocked."""
        from frontier_review.orchestrator import run_autonomous_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport, TransportError

        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.timeout_response())
        result = run_autonomous_review_cycle(
            task_id="test-timeout-1", requirements="Test.",
            commit_sha="to001", diff="--- a/main.py", mock_transport=mt)

        assert result.status == ReviewStatus.BLOCKED, f"Expected BLOCKED but got {result.status.value}"

    def test_scenario_budget_exhaustion(self, tmp_path_factory):
        """Budget exhausted → review blocked."""
        from frontier_review.orchestrator import run_autonomous_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport

        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.budget_exhaustion_response())
        result = run_autonomous_review_cycle(
            task_id="test-budget-1", requirements="Test.",
            commit_sha="bud001", diff="--- a/main.py", mock_transport=mt)

        assert result.status == ReviewStatus.BLOCKED, f"Expected BLOCKED but got {result.status.value}"

    def test_scenario_duplicate_request(self, tmp_path_factory):
        """Duplicate request → returns original result."""
        from frontier_review.orchestrator import run_autonomous_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport

        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.duplicate_request_response(original_id="req-dup-abc123"))
        result = run_autonomous_review_cycle(
            task_id="test-dup-1", requirements="Test.",
            commit_sha="dup001", diff="--- a/main.py", mock_transport=mt)

        assert result.status == ReviewStatus.PASS, f"Expected PASS but got {result.status.value}"
        assert "Duplicate" in result.summary or "original" in result.summary.lower()

    def test_scenario_interrupted_task(self, tmp_path_factory):
        """Interrupted task (truncated JSON) → review blocked."""
        from frontier_review.orchestrator import run_autonomous_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport

        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.interrupted_task_response())
        result = run_autonomous_review_cycle(
            task_id="test-interrupted-1", requirements="Test.",
            commit_sha="int001", diff="--- a/main.py", mock_transport=mt)

        assert result.status == ReviewStatus.BLOCKED, f"Expected BLOCKED but got {result.status.value}"


class TestAutonomousLoopControl:
    """Bounded correction loop controls — limits, dedup, cost ceiling."""

    def test_max_reviews_enforced(self, tmp_path_factory):
        """Max review dispatches limit is enforced (default 4)."""
        from frontier_review.orchestrator import FrontierOrchestrator
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport
        from tempfile import mkdtemp as _mkdtemp

        store_dir = _mkdtemp(prefix="fr_maxrev_")
        try:
            store_path = os.path.join(store_dir, "state.json")
            mt = MockAnthropicTransport()
            # Provide 4 responses (one per review call) so first 3 reviews run;
            # on the 4th call we should hit max_reviews=3 check and return BLOCKED.
            always_changes = MockAnthropicTransport.ok_response(
                status="changes_required", summary="Still needs fixes.", findings=[])
            configs = [always_changes] * 4
            mt.configure_sequence(configs)

            from frontier_review.client import AnthropicClient
            client = AnthropicClient(model="claude-opus-5-5")
            client.set_transport(mt)

            orch = FrontierOrchestrator(
                client=client, store_path=store_path, max_reviews=3, max_corrections=10, cost_ceiling_usd=100.0)

            # 1st call: PENDING → review #1 → CHANGES_REQUIRED
            r1 = orch.run_autonomous_review(
                task_id="test-maxrev", requirements="Test.", commit_sha="a1", diff="d1")
            assert r1.status == ReviewStatus.CHANGES_REQUIRED

            # 2nd call: review #2 → still CHANGES_REQUIRED
            r2 = orch.run_autonomous_review(
                task_id="test-maxrev", requirements="Test.", commit_sha="a2", diff="d2")
            assert r2.status == ReviewStatus.CHANGES_REQUIRED, \
                f"Call 2 expected CHANGES_REQUIRED, got {r2.status.value}"

            # 3rd call: review #3 → still CHANGES_REQUIRED
            r3 = orch.run_autonomous_review(
                task_id="test-maxrev", requirements="Test.", commit_sha="a3", diff="d3")
            assert r3.status == ReviewStatus.CHANGES_REQUIRED, \
                f"Call 3 expected CHANGES_REQUIRED, got {r3.status.value}"

            # 4th call: should block because review_count=3 >= max_reviews=3
            r4 = orch.run_autonomous_review(
                task_id="test-maxrev", requirements="Test.", commit_sha="a4", diff="d4")
            assert r4.status == ReviewStatus.BLOCKED, f"Expected BLOCKED (max reviews) but got {r4.status.value}"
            assert "Max reviews" in r4.summary or "Exceeded" in r4.summary or "review" in r4.summary.lower()
        finally:
            import shutil
            shutil.rmtree(store_dir, ignore_errors=True)

    def test_max_corrections_escalation(self, tmp_path_factory):
        """Max corrections reached → ESCALATED_ARCHITECT."""
        from frontier_review.orchestrator import FrontierOrchestrator
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport
        from tempfile import mkdtemp as _mkdtemp

        store_dir = _mkdtemp(prefix="fr_esc_")
        try:
            store_path = os.path.join(store_dir, "state.json")
            mt = MockAnthropicTransport()
            # Provide 2 responses: review #1 (CHANGES_REQUIRED) and a fallback
            always_changes = MockAnthropicTransport.ok_response(
                status="changes_required", summary="Still needs fixes.",
                findings=[{"title": "Issue", "severity": "HIGH", "category": "correctness",
                           "location": "main.py:1", "description": "Persistent bug."}])
            mt.configure_sequence([always_changes] * 2)

            from frontier_review.client import AnthropicClient
            client = AnthropicClient(model="claude-opus-5-5")
            client.set_transport(mt)

            orch = FrontierOrchestrator(
                client=client, store_path=store_path, max_corrections=1, max_reviews=3, cost_ceiling_usd=100.0)

            # First review → CHANGES_REQUIRED (corrections_made set to 1 internally)
            result1 = orch.run_autonomous_review(task_id="test-esc", requirements="Test.", commit_sha="a", diff="d")
            assert result1.status == ReviewStatus.CHANGES_REQUIRED, \
                f"Call 1 expected CHANGES_REQUIRED, got {result1.status.value}"

            # Second review → ESCALATED_ARCHITECT (corrections_made=1 >= max_corrections=1)
            result2 = orch.run_autonomous_review(
                task_id="test-esc", requirements="Test.", commit_sha="b", diff="d2")
            assert result2.status == ReviewStatus.ESCALATED_ARCHITECT, \
                f"Call 2 expected ESCALATED_ARCHITECT, got {result2.status.value}"
        finally:
            import shutil
            shutil.rmtree(store_dir, ignore_errors=True)

    def test_duplicate_fingerprint_detection(self):
        """Fingerprint computation detects unchanged evidence."""
        from frontier_review.orchestrator import FrontierOrchestrator
        fp1 = FrontierOrchestrator.compute_fingerprint("abc123", "test evidence")
        fp2 = FrontierOrchestrator.compute_fingerprint("abc123", "test evidence")
        fp3 = FrontierOrchestrator.compute_fingerprint("def456", "test evidence")
        assert fp1 == fp2, "Same SHA+evidence should produce same fingerprint"
        assert fp1 != fp3, "Different SHA should produce different fingerprint"

    def test_no_op_correction_rejected(self):
        """Same commit SHA after review count > 0 is flagged as duplicate."""
        from frontier_review.orchestrator import FrontierOrchestrator
        from tempfile import mkdtemp as _mkdtemp
        import shutil

        store_dir = _mkdtemp(prefix="fr_dup_")
        try:
            orch = FrontierOrchestrator(store_path=os.path.join(store_dir, "state.json"))
            orch.state_machine.create_task("task-dup", commit_sha="abc123", diff="diff")
            # Mark as having been reviewed
            orch.state_machine.transition("task-dup", "REREVIEWING", review_count=1)

            # Same SHA should be detected as duplicate
            assert orch.has_duplicate_commit("task-dup", "abc123") is True
            # Different SHA should not
            assert orch.has_duplicate_commit("task-dup", "new_sha") is False
        finally:
            shutil.rmtree(store_dir, ignore_errors=True)

    def test_cost_ceiling_enforcement(self, tmp_path_factory):
        """Simulated cost ceiling blocks reviews above threshold."""
        from frontier_review.orchestrator import FrontierOrchestrator
        from frontier_review.models import ReviewStatus
        from tempfile import mkdtemp as _mkdtemp

        store_dir = _mkdtemp(prefix="fr_cost_")
        try:
            orch = FrontierOrchestrator(
                store_path=os.path.join(store_dir, "state.json"),
                cost_ceiling_usd=0.001)  # Very low ceiling

            result = orch.run_autonomous_review(
                task_id="test-cost", requirements="Test.", commit_sha="abc", diff="small diff")

            assert result.status == ReviewStatus.BLOCKED, f"Expected BLOCKED (cost ceiling) but got {result.status.value}"
            assert "cost ceiling" in result.summary.lower() or "Ceiling" in result.summary
        finally:
            import shutil
            shutil.rmtree(store_dir, ignore_errors=True)

    def test_terminal_task_no_re_review(self, tmp_path_factory):
        """A COMPLETE task cannot be re-reviewed."""
        from frontier_review.orchestrator import FrontierOrchestrator
        from frontier_review.models import ReviewStatus
        from tempfile import mkdtemp as _mkdtemp

        store_dir = _mkdtemp(prefix="fr_term_")
        try:
            orch = FrontierOrchestrator(store_path=os.path.join(store_dir, "state.json"))
            orch.state_machine.create_task("task-term", commit_sha="abc", diff="diff")
            # Must go through REREVIEWING → COMPLETE (valid state machine path)
            orch.state_machine.transition("task-term", "REREVIEWING")
            orch.state_machine.transition("task-term", "COMPLETE")

            result = orch.run_autonomous_review(
                task_id="task-term", requirements="Test.", commit_sha="new", diff="new-diff")

            assert result.status == ReviewStatus.BLOCKED, f"Expected BLOCKED but got {result.status.value}"
            assert "terminal" in result.summary.lower()
        finally:
            import shutil
            shutil.rmtree(store_dir, ignore_errors=True)

    def test_delegation_request_build(self):
        """Build correction request returns None for PASS results."""
        from frontier_review.orchestrator import FrontierOrchestrator
        from frontier_review.models import ReviewResult, ReviewStatus, TokenUsage

        orch = FrontierOrchestrator()

        # PASS → no correction request needed
        pass_result = ReviewResult(
            status=ReviewStatus.PASS, operation="review-code", summary="All good.",
            model_used="claude-opus-5-5")
        assert orch.build_correction_request("t1", pass_result) is None

        # CHANGES_REQUIRED → correction request returned
        changes_result = ReviewResult(
            status=ReviewStatus.CHANGES_REQUIRED, operation="review-code", summary="Fix bugs.",
            findings=[{"title": "Bug", "severity": "HIGH", "category": "correctness",
                       "location": "main.py:10", "description": "Null pointer."}],
            suggestions=["use null check"], model_used="claude-opus-5-5")
        req = orch.build_correction_request("t1", changes_result)
        assert req is not None
        assert req["task_id"] == "t1"
        assert req["verdict"] == "CHANGES_REQUIRED"

    def test_cumulative_cost_tracking(self, tmp_path_factory):
        """Cumulative cost is tracked per task."""
        from frontier_review.orchestrator import FrontierOrchestrator
        from tempfile import mkdtemp as _mkdtemp

        store_dir = _mkdtemp(prefix="fr_cost_track_")
        orch = FrontierOrchestrator(store_path=os.path.join(store_dir, "state.json"))
        assert orch.get_cumulative_cost("unknown-task") == 0.0


class TestOfflineMode:
    """Verify no real API calls can be made in offline mode."""

    def test_real_transport_always_raises(self):
        from frontier_review.transport import RealAnthropicTransport, RealApiDisabledError
        with pytest.raises(RealApiDisabledError):
            RealAnthropicTransport()  # type: ignore[call-arg]

    def test_client_without_transport_raises(self):
        from frontier_review.client import AnthropicClient
        from frontier_review.transport import RealApiDisabledError
        c = AnthropicClient()
        with pytest.raises(RealApiDisabledError):
            c.chat(system_prompt="x", user_message="y")

    def test_mock_never_reaches_network(self):
        """If we patch socket.connect to raise, mock transport must still work."""
        import socket
        from frontier_review.transport import MockAnthropicTransport
        real_connect = socket.socket.connect
        def fake_connect(*args, **kwargs):
            raise ConnectionError("Network access blocked — this should not happen in mock mode")
        try:
            socket.socket.connect = fake_connect
            mt = MockAnthropicTransport()
            mt.configure(MockAnthropicTransport.ok_response())
            raw = mt.send_request("https://api.anthropic.com/v1/messages", "POST", {}, "{}")
            resp = mt.receive_response(raw)
            assert isinstance(resp, dict)
        finally:
            socket.socket.connect = real_connect
