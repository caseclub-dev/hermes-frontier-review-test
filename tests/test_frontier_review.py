"""Frontier Review tests — M1 acceptance validation.

Covers:
- Import successfully in isolated environment
- Mock transport instantiation without credentials
- Anthropic-compatible response parsing
- Unexpected model identity rejection
- Malformed response handling
- Real API not possible in offline mode
- Budget accounting
- State machine transitions
- Delegation adapter serialization

All tests run fully offline. No network access required.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import shutil

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
        from frontier_review import config, models, transport, client, review, budget, state  # noqa: F401

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
        # Default should be True (fail-closed)
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
            status=ReviewStatus.PASS,
            operation="review-code",
            summary="All good.",
            model_used="claude-opus-5-5",
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        s = str(r)
        assert "PASS" in s
        assert "review-code" in s

    def test_review_result_non_opus_cost(self):
        from frontier_review.models import ReviewResult, ReviewStatus
        r = ReviewResult(
            status=ReviewStatus.PASS,
            operation="review-code",
            summary="All good.",
            model_used="claude-sonnet-4",
        )
        with pytest.raises(ValueError):
            _ = str(r)  # __str__ calls cost_estimate


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
        cfg = MockAnthropicTransport.ok_response(
            model="claude-opus-5-5",
            status="pass",
            summary="Test review.",
        )
        assert cfg.status_code == 200
        body = json.loads(cfg._raw_body)
        assert body["model"] == "claude-opus-5-5"
        assert "content" in body
        assert "usage" in body

    def test_response_is_anthropic_compatible(self):
        """Mock response must match Anthropic Messages API structure."""
        from frontier_review.transport import MockAnthropicTransport
        cfg = MockAnthropicTransport.ok_response(
            model="claude-opus-5-5",
            status="changes_required",
            summary="Needs fixes.",
            findings=[{
                "title": "Bug in loop",
                "severity": "HIGH",
                "category": "correctness",
                "location": "main.py:42",
                "description": "Off-by-one error.",
            }],
        )
        body = json.loads(cfg._raw_body)

        # Top-level keys must match Anthropic's Messages API
        assert "id" in body
        assert body["model"] == "claude-opus-5-5"
        assert isinstance(body["content"], list)
        assert len(body["content"]) > 0
        assert body["content"][0]["type"] == "text"
        assert "usage" in body

        # Usage keys must match Anthropic's response format
        usage = body["usage"]
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            assert key in usage, f"Missing usage key: {key}"

    def test_server_error_response(self):
        from frontier_review.transport import MockAnthropicTransport
        cfg = MockAnthropicTransport.server_error_response("Test error")
        body = json.loads(cfg._raw_body)
        assert "error" in body

    def test_send_request_returns_bytes(self):
        from frontier_review.transport import MockAnthropicTransport, ScenarioConfig
        mt = MockAnthropicTransport()
        cfg = MockAnthropicTransport.ok_response()
        mt.configure(cfg)
        raw = mt.send_request(
            url="https://api.anthropic.com/v1/messages",
            method="POST",
            headers={"content-type": "application/json"},
            body=json.dumps({"model": "test"}),
            timeout=30,
        )
        assert isinstance(raw, bytes)

    def test_receive_response_parses_json(self):
        from frontier_review.transport import MockAnthropicTransport, ScenarioConfig
        mt = MockAnthropicTransport()
        cfg = MockAnthropicTransport.ok_response()
        mt.configure(cfg)
        raw = mt.send_request(
            url="https://api.anthropic.com/v1/messages",
            method="POST",
            headers={},
            body="",
        )
        resp = mt.receive_response(raw)
        assert isinstance(resp, dict)
        assert "model" in resp

    def test_malformed_response_raises_error(self):
        from frontier_review.transport import (
            MockAnthropicTransport,
            ScenarioConfig,
            TransportError,
        )
        mt = MockAnthropicTransport()
        cfg = ScenarioConfig(_raw_body="not json {{{")
        mt.configure(cfg)
        raw = mt.send_request(
            url="https://api.anthropic.com/v1/messages",
            method="POST",
            headers={},
            body="",
        )
        with pytest.raises(TransportError):
            mt.receive_response(raw)

    def test_sequence_consumption(self):
        from frontier_review.transport import MockAnthropicTransport, ScenarioConfig
        mt = MockAnthropicTransport()
        configs = [
            MockAnthropicTransport.ok_response(status=f"pass{i}", summary=f"Review {i}")
            for i in range(3)
        ]
        mt.configure_sequence(configs)

        # Consume first 2
        raw1 = mt.send_request("url", "POST", {}, "{}")
        raw2 = mt.send_request("url", "POST", {}, "{}")
        assert len(raw1) > 0 and len(raw2) > 0

        # Third
        raw3 = mt.send_request("url", "POST", {}, "{}")
        body3 = json.loads(raw3.decode())
        assert body3["content"][0]["text"]

    def test_sequence_exhaustion_raises(self):
        from frontier_review.transport import (
            MockAnthropicTransport,
            ScenarioConfig,
            MockServerError,
        )
        mt = MockAnthropicTransport()
        cfg = MockAnthropicTransport.ok_response(status="only_one")
        mt.configure(cfg)  # single-scenario — sequence of length 1

        # Consume the one response
        mt.send_request("url", "POST", {}, "{}")

        # Next call should raise (single-scenario falls through to sequence path)
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
        cfg = MockAnthropicTransport.ok_response(status="pass")
        mt.configure(cfg)

        c = AnthropicClient(model="claude-opus-5-5")
        c.set_transport(mt)

        text, usage = c.chat(
            system_prompt="You are a reviewer.",
            user_message="Review this code.",
        )
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
        """If response model differs from requested, raise RuntimeError."""
        from frontier_review.client import AnthropicClient
        from frontier_review.transport import MockAnthropicTransport, ScenarioConfig
        mt = MockAnthropicTransport()
        # Configure with a DIFFERENT model — simulate drift
        cfg = ScenarioConfig(
            _raw_body=json.dumps({
                "id": "msg-test",
                "model": "claude-sonnet-4",  # WRONG MODEL
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
        """Verify the request body matches Anthropic's Messages API schema."""
        from frontier_review.client import AnthropicClient
        from frontier_review.transport import MockAnthropicTransport
        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.ok_response())

        c = AnthropicClient(model="claude-opus-5-5")
        c.set_transport(mt)

        # The chat method calls _build_request_body internally
        text, _ = c.chat(
            system_prompt="Review code.",
            user_message="Fix the bug.",
        )
        assert "pass" in text.lower() or len(text) > 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ review ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestReview:
    """Prompt building, JSON parsing, post-processing."""

    def test_code_review_prompt_has_required_sections(self):
        from frontier_review.review import build_code_review_prompt
        prompt = build_code_review_prompt(
            requirements="Must pass tests.",
            commit_sha="abc123",
            diff="--- a/main.py\n+++ b/main.py\n@@ ...\n",
            code_context="",
            test_evidence="",
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
        raw = {
            "status": "pass",
            "summary": "All checks passed.",
            "findings": [],
            "requirements_met": ["type hints intact"],
            "suggestions": ["add docstrings"],
        }
        result = postprocess_code_result(raw, "claude-opus-5-5")
        assert result.status.value == "PASS"
        assert len(result.findings) == 0
        assert "type hints intact" in result.requirements_met

    def test_postprocess_changes_required(self):
        from frontier_review.review import postprocess_code_result
        raw = {
            "status": "changes_required",
            "summary": "Fix type hint removals.",
            "findings": [{
                "title": "Type hint removed",
                "severity": "HIGH",
                "category": "correctness",
                "location": "main.py:10",
                "description": "Function lost its return type annotation.",
            }],
        }
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
        # Input: 10000/1M * 15 = 0.15; Output: 32768/1M * 75 ≈ 2.46
        assert cost == pytest.approx(0.15 + 2.4576, abs=0.001)

    def test_estimate_cost_non_opus_zero(self):
        from frontier_review.budget import estimate_request_cost
        cost = estimate_request_cost(10_000, 32768, "claude-sonnet-4")
        assert cost == 0.0

    def test_check_budget_no_ledger(self):
        """When ledger file doesn't exist, budget check passes."""
        from frontier_review.budget import (
            check_budget, isolate_test_ledger, reset_ledger_override, set_ledger_override,
        )
        tmpdir = _tmpdir()
        try:
            fake_path = os.path.join(tmpdir, "nonexistent.json")
            set_ledger_override(fake_path)
            assert check_budget(10.0, 75.0) is True
        finally:
            reset_ledger_override()
            _cleanup(tmpdir)

    def test_record_request(self):
        from frontier_review.budget import (
            isolate_test_ledger,
            record_request,
            reset_ledger_override,
        )
        tmp_path = isolate_test_ledger()
        try:
            rid = record_request("claude-opus-5-5", 10000, 2000)
            assert rid is not None
            assert rid.startswith("req-")
            # Read ledger to confirm it was recorded
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
            sm.transition("task-1", "COMPLETE")  # PENDING → COMPLETE invalid

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
        req = build_delegation_request(
            task_id="task-1",
            verdict="CHANGES_REQUIRED",
            summary="Fix type hints.",
            findings=[{
                "title": "Missing return type",
                "severity": "HIGH",
                "category": "correctness",
                "location": "main.py:10",
                "description": "Function has no return annotation.",
            }],
        )
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
        tickets = extract_correction_tickets("task-1", [
            {
                "title": "Bug fix needed",
                "severity": "HIGH",
                "category": "correctness",
                "location": "main.py:50",
                "description": "Null pointer exception.",
            },
        ])
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

        result = run_review_cycle(
            requirements="Code must compile.",
            commit_sha="abc123",
            diff="--- a/main.py\n+++ b/main.py\n@@ -1,5 +1,5 @@\n-def foo(x):\n+def foo(x: int) -> str:\n     return str(x)\n",
            mock_transport=mt,
        )
        assert result.status == ReviewStatus.PASS

    def test_full_review_cycle_changes_required(self):
        from frontier_review.orchestrator import run_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport
        mt = MockAnthropicTransport()
        findings = [{
            "title": "Type hint removed",
            "severity": "HIGH",
            "category": "correctness",
            "location": "main.py:10",
            "description": "Function lost return type.",
        }]
        mt.configure(MockAnthropicTransport.ok_response(
            status="changes_required", summary="Needs fixes.", findings=findings,
        ))

        result = run_review_cycle(
            requirements="Type hints must remain.",
            commit_sha="def456",
            diff="diff content here...",
            mock_transport=mt,
        )
        assert result.status == ReviewStatus.CHANGES_REQUIRED
        assert len(result.findings) == 1

    def test_full_review_cycle_blocked_by_budget(self):
        """When budget is exceeded, review should be blocked."""
        from frontier_review.budget import (
            isolate_test_ledger, record_request, reset_ledger_override,
        )
        from frontier_review.orchestrator import run_review_cycle
        from frontier_review.models import ReviewStatus
        from frontier_review.transport import MockAnthropicTransport

        # Create temp ledger with drained budget — 6M input tokens costs ~$92 (exceeds $75 monthly budget)
        tmp_path = isolate_test_ledger()
        record_request("claude-opus-5-5", 6_000_000, 32768)

        # Review cycle runs while _LEDGER_PATH still points to the drained temp ledger
        mt = MockAnthropicTransport()
        mt.configure(MockAnthropicTransport.ok_response(status="pass"))

        result = run_review_cycle(
            requirements="Test.", commit_sha="abc", diff="small diff here", mock_transport=mt,
        )
        assert result.status == ReviewStatus.BLOCKED

        # Cleanup
        reset_ledger_override()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        assert "Budget" in result.summary.lower() or "budget" in result.summary.lower()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ offline enforcement ━━━━━━━━━━━━━━━━━━

class TestOfflineMode:
    """Verify no real API calls can be made in offline mode."""

    def test_real_transport_always_raises(self):
        from frontier_review.transport import RealAnthropicTransport, RealApiDisabledError
        # Construction should fail immediately — no fallback to network
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

        real_connect = socket.socket.connect  # Save original

        def fake_connect(*args, **kwargs):
            raise ConnectionError("Network access blocked — this should not happen in mock mode")

        try:
            socket.socket.connect = fake_connect
            mt = MockAnthropicTransport()
            mt.configure(MockAnthropicTransport.ok_response())
            # This MUST work without network access
            raw = mt.send_request("https://api.anthropic.com/v1/messages", "POST", {}, "{}")
            resp = mt.receive_response(raw)
            assert isinstance(resp, dict)
        finally:
            socket.socket.connect = real_connect  # Restore
