"""M3 Integration Tests — Real Coder Delegation Reference.

These tests document the expected behavior of real Coder delegation via
the Hermes delegate_task mechanism. The unit-test classes here serve as a
specification; actual execution of delegate_task calls is performed by
Orchestrator in its own context (not from within pytest).

Key flows:
  1. Initial delegation — Orchestrator delegates to Coder, verifies commit + evidence
  2. Mock review — same pipeline as production, uses MockAnthropicTransport
  3. Correction loop — feedback → delegate_task again → re-verify
  4. PASS → READY_FOR_OPERATOR_REVIEW
  5. Failure handling — timeout, no_commit, no_changes, max_cycles, malformed

M3 milestone: This module provides the adapter layer that connects the
existing mock-first review engine to real Coder delegations. The actual
delegate_task invocations are executed by the Orchestrator in its live
session, not from within these test functions.

No Anthropic API key is used at any point in this file.
"""

from __future__ import annotations

import json
import sys
import os
import shutil
import tempfile
import hashlib
import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ helper imports ━━━━━━━━━━━━━━━━━━━━━━━━

from frontier_review import (
    DelegationResult, CodeReviewFeedback, Severity, FindingCategory,
    build_initial_delegation_request, build_correction_delegation_request,
    parse_delegation_response, GitWorkspaceVerifier, compute_evidence_fingerprint,
    create_disposable_workspace, MockReviewSimulator, CorrectionLoopRunner,
    handle_delegation_failure, is_duplicate_review, is_max_correction_cycles,
)

from frontier_review.transport import MockAnthropicTransport


# ━━━━━━━━━━━━━━━━━━━━━━━ delegation result parsing ━━━━━━━━━━━━━━━━

class TestDelegationResultParsing:
    """Test parse_delegation_response against expected Coder output formats."""

    def test_parse_successful_with_files_and_sha(self):
        response = (
            "Implementation complete.\n"
            "Files created: `main.py`, `utils.py`\n"
            "Commit SHA: abc123def4567890123456789012345678abcdef\n"
            "Tests passed: 5/5\n"
        )
        result = parse_delegation_response(response)
        assert result.success is True
        assert set(result.files_created) == {"main.py", "utils.py"}
        assert result.commit_sha == "abc123def4567890123456789012345678abcdef"
        assert result.has_changes is True

    def test_parse_with_test_output(self):
        response = (
            "Done.\n"
            "Files: `module.py`\n"
            "SHA: deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\n"
            "\n"
            "Test output:\n"
            "passed 5 tests in module.py\n"
        )
        result = parse_delegation_response(response)
        assert result.commit_sha == "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
        assert "passed 5 tests" in result.test_output

    def test_parse_failure_no_commit(self):
        response = (
            "Failed to implement.\n"
            "Error: cannot write to the target directory.\n"
            "The workspace path does not exist.\n"
        )
        result = parse_delegation_response(response)
        assert result.success is False
        assert not result.has_changes

    def test_parse_finds_py_files_in_backticks(self):
        response = (
            "Created files: `src/tools/calc.py` and `src/tests/test_calc.py`\n"
            "All tests pass. Commit made.\n"
        )
        result = parse_delegation_response(response)
        assert len(result.files_created) == 2
        assert any("calc.py" in f for f in result.files_created)

    def test_parse_no_sha_detected(self):
        response = "I wrote the code but couldn't commit."
        result = parse_delegation_response(response)
        assert result.commit_sha is None
        assert not result.has_changes


# ━━━━━━━━━━━━━━━━━━━━━━ delegation request builders ━━━━━━━━━━━━━━━

class TestDelegationRequestBuilders:
    """Test that request builders produce properly structured payloads."""

    def test_initial_request_contains_workspace(self):
        req = build_initial_delegation_request(
            task_id="test-init",
            requirements="Write a function.",
            workspace_path="/tmp/test_work",
            acceptance_criteria=["must have type hints", "tests must pass"],
        )
        assert "/tmp/test_work" in req
        assert "write the function" in req.lower() or "type hints" in req.lower()

    def test_initial_request_contains_acceptance_criteria(self):
        criteria = ["one thing", "two things"]
        req = build_initial_delegation_request(
            task_id="t1", requirements="", workspace_path="/tmp/x", acceptance_criteria=criteria,
        )
        for c in criteria:
            assert c in req

    def test_correction_request_references_findings(self):
        findings_text = (
            "[HIGH] Missing error handling\n"
            "[MEDIUM] No type hints\n"
        )
        req = build_correction_delegation_request(
            task_id="t2", workspace_path="/tmp/x", findings_text=findings_text,
            acceptance_criteria=["fix above"],
        )
        assert "Missing error handling" in req
        assert "No type hints" in req

    def test_initial_vs_correction_distinguishable(self):
        init = build_initial_delegation_request("t1", "", "/tmp/x", ["c1"])
        corr = build_correction_delegation_request(
            "t2", "/tmp/x", "[HIGH] fix this", ["c1"])
        assert "# Coder Assignment" in init
        assert "# Coder Correction" in corr


# ━━━━━━━━━━━━━━━━━━━━━━━ mock review simulator ━━━━━━━━━━━━━━━━━━━━

class TestMockReviewSimulator:
    """Test that MockReviewSimulator produces valid review payloads."""

    def test_pass_review_contains_requirements_met(self):
        payload = MockReviewSimulator.create_pass_review()
        data = json.loads(payload)
        assert data["status"] == "pass"
        assert len(data["requirements_met"]) > 0

    def test_changes_required_contains_findings(self):
        findings = [
            {"title": "Bug", "severity": "HIGH", "category": "correctness",
             "description": "Logic error"},
        ]
        payload = MockReviewSimulator.create_changes_required(findings)
        data = json.loads(payload)
        assert data["status"] == "changes_required"
        assert len(data["findings"]) == 1

    def test_block_review_has_no_findings(self):
        payload = MockReviewSimulator.create_block_review("reason")
        data = json.loads(payload)
        assert data["status"] == "blocked"
        assert "reason" in data["summary"]

    def test_pass_review_structure_anthropic_compatible(self):
        """The mock review format must match what postprocess_code_result expects."""
        payload = MockReviewSimulator.create_pass_review()
        data = json.loads(payload)
        assert "status" in data
        assert "summary" in data
        assert "findings" in data
        assert isinstance(data["findings"], list)


# ━━━━━━━━━━━━━━━━━━━━ evidence fingerprinting ━━━━━━━━━━━━━━━━━━━━━

class TestEvidenceFingerprint:
    """Test that evidence fingerprints detect duplicates."""

    def test_same_evidence_produces_same_fingerprint(self):
        fp1 = compute_evidence_fingerprint("abc123", "5 passed")
        fp2 = compute_evidence_fingerprint("abc123", "5 passed")
        assert fp1 == fp2

    def test_different_sha_different_fingerprint(self):
        fp1 = compute_evidence_fingerprint("sha-a", "test output")
        fp2 = compute_evidence_fingerprint("sha-b", "test output")
        assert fp1 != fp2

    def test_different_test_output_different_fingerprint(self):
        fp1 = compute_evidence_fingerprint("abc", "5 passed")
        fp2 = compute_evidence_fingerprint("abc", "3 passed")
        assert fp1 != fp2

    def test_fingerprint_is_16_chars_hex(self):
        fp = compute_evidence_fingerprint("x", "y")
        assert len(fp) == 16
        int(fp, 16)  # Will raise if not valid hex


# ━━━━━━━━━━━━━━━━━━━ disposable workspace creation ━━━━━━━━━━━━━━━━

class TestDisposableWorkspace:
    """Test create_disposable_workspace creates a usable Git repo."""

    def test_creates_directory(self):
        path = create_disposable_workspace(base_dir="/tmp", prefix="fr_m3_test_")
        assert os.path.isdir(path)
        shutil.rmtree(path, ignore_errors=True)

    def test_contains_git_directory(self):
        path = create_disposable_workspace(base_dir="/tmp", prefix="fr_m3_test_")
        assert os.path.isdir(os.path.join(path, ".git"))
        shutil.rmtree(path, ignore_errors=True)

    def test_unique_paths(self):
        p1 = create_disposable_workspace(base_dir="/tmp", prefix="fr_m3_test_")
        p2 = create_disposable_workspace(base_dir="/tmp", prefix="fr_m3_test_")
        assert p1 != p2
        shutil.rmtree(p1, ignore_errors=True)
        shutil.rmtree(p2, ignore_errors=True)


# ━━━━━━━━━━━━━━━━━━━━━━ failure handling utilities ━━━━━━━━━━━━━━━━

class TestFailureHandling:
    """Test handle_delegation_failure returns correct terminal states."""

    def test_timeout_is_terminal_blocked(self):
        r = handle_delegation_failure("timeout")
        assert r["status"] == "blocked"
        assert r["terminal"] is True

    def test_no_commit_is_non_terminal(self):
        r = handle_delegation_failure("no_commit", "nothing happened")
        assert r["status"] == "changes_required"
        assert r["terminal"] is False

    def test_no_changes_is_non_terminal(self):
        r = handle_delegation_failure("no_changes")
        assert r["status"] == "changes_required"
        assert r["terminal"] is False

    def test_malformed_review_is_terminal_blocked(self):
        r = handle_delegation_failure("malformed_review", "{not json")
        assert r["status"] == "blocked"
        assert r["terminal"] is True

    def test_malformed_review_without_details(self):
        r = handle_delegation_failure("malformed_review", None)
        assert r["status"] == "blocked"
        assert r["terminal"] is True
        assert "no details" in r["summary"]

    def test_max_cycles_exhausted_esc_always(self):
        r = handle_delegation_failure("max_cycles_exhausted")
        assert r["status"] == "escalated_architect"
        assert r["terminal"] is True
        assert any("max" in b.lower() for b in r.get("blockers", []))

    def test_tests_failed_is_non_terminal(self):
        r = handle_delegation_failure("tests_failed", "FAILED: assertion error")
        assert r["status"] == "changes_required"
        assert r["terminal"] is False

    def test_unknown_error_is_terminal_blocked(self):
        r = handle_delegation_failure("unknown_type_xyz")
        assert r["status"] == "blocked"
        assert r["terminal"] is True


class TestLimitChecks:
    """Test review and correction limit checks."""

    def test_duplicate_review_at_limit(self):
        assert is_duplicate_review(4, max_reviews=4) is True
        assert is_duplicate_review(3, max_reviews=4) is False

    def test_max_correction_cycles(self):
        assert is_max_correction_cycles(3, current=3) is True
        assert is_max_correction_cycles(3, current=2) is False


# ━━━━━━━━━━━━━━━━━━━━━━ verification utilities ━━━━━━━━━━━━━━━━━━━━

class TestGitWorkspaceVerifier:
    """Test GitWorkspaceVerifier works against an actual repo."""

    def test_empty_workspace(self):
        tmpdir = tempfile.mkdtemp(prefix="fr_verif_")
        try:
            import subprocess as sp
            sp.run(["git", "init", tmpdir], capture_output=True, timeout=5)
            v = GitWorkspaceVerifier(tmpdir)
            assert v.list_created_files() == []
            assert v.get_commit_count() == 0
            assert v.get_last_commit_sha() is None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_workspace_with_commits(self):
        tmpdir = tempfile.mkdtemp(prefix="fr_verif_")
        try:
            import subprocess as sp
            sp.run(["git", "init", tmpdir], capture_output=True, timeout=5)
            sp.run(["git", "-C", tmpdir, "config", "user.name", "test"], capture_output=True, timeout=5)
            sp.run(["git", "-C", tmpdir, "config", "user.email", "t@t.t"], capture_output=True, timeout=5)

            # Create a file and commit
            with open(os.path.join(tmpdir, "a.py"), "w") as f:
                f.write("# first\n")
            sp.run(["git", "-C", tmpdir, "add", "."], capture_output=True, timeout=5)
            sp.run(["git", "-C", tmpdir, "commit", "-m", "first"], capture_output=True, timeout=5)

            v = GitWorkspaceVerifier(tmpdir)
            assert len(v.list_created_files()) == 1
            assert v.get_commit_count() >= 1
            assert v.get_last_commit_sha() is not None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ━━━━━━━━━━━━━━━━━━━━━━ code review feedback model ━━━━━━━━━━━━━━━━

class TestCodeReviewFeedback:
    """Test CodeReviewFeedback model methods."""

    def test_is_changes_required_true(self):
        fb = CodeReviewFeedback(status="changes_required", summary="fix this")
        assert fb.is_changes_required() is True

    def test_is_changes_required_false(self):
        fb = CodeReviewFeedback(status="pass", summary="ok")
        assert fb.is_changes_required() is False

    def test_correctness_findings_filter(self):
        findings = [
            {"severity": "HIGH", "category": "correctness", "title": "A"},
            {"severity": "MEDIUM", "category": "style", "title": "B"},
            {"severity": "HIGH", "category": "security", "title": "C"},
        ]
        fb = CodeReviewFeedback(status="changes_required", summary="", findings=findings)
        correct = fb.correctness_findings()
        assert len(correct) == 1
        assert correct[0]["title"] == "A"

    def test_as_description_format(self):
        findings = [
            {"severity": "HIGH", "title": "Bug A", "description": "First"},
            {"severity": "LOW", "title": "Nit B", "description": "Second"},
        ]
        fb = CodeReviewFeedback(status="changes_required", summary="Overview", findings=findings)
        desc = fb.as_description()
        assert "Overview" in desc
        assert "Bug A" in desc


# ━━━━━━━━━━━━━━━━━━━━━┓ correction loop planning ┏━━━━━━━━━━━━━━━━━

class TestCorrectionLoopPlanning:
    """Test CorrectionLoopRunner plan generation (no real Coder calls)."""

    def test_pass_review_short_circuits(self):
        workspace = create_disposable_workspace(prefix="fr_plan_")
        try:
            runner = CorrectionLoopRunner(workspace)
            mock_payload = MockReviewSimulator.create_pass_review()
            plan = runner.run(
                initial_request_text="# Assignment\nRequirements: build something.\nAcceptance Criteria:\n1. must work\n",
                mock_review_result=mock_payload,
            )
            assert plan["status"] == "ready_for_operator_review"
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def test_changes_required_generates_correction_plan(self):
        workspace = create_disposable_workspace(prefix="fr_plan_")
        try:
            runner = CorrectionLoopRunner(workspace)
            mock_payload = MockReviewSimulator.create_changes_required([
                {"title": "Bug", "severity": "HIGH", "category": "correctness",
                 "description": "Logic error in loop"},
            ])
            plan = runner.run(
                initial_request_text="Requirements: build.\nAcceptance Criteria:\n1. pass tests\n",
                mock_review_result=mock_payload,
            )
            assert plan["status"] == "changes_required"
            assert len(plan["steps"]) >= 2  # init + correction + verify
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━ reference: real workflow ━━━━━━━━━━━━━━━━

# Real M3 workflow: execute correction loop via delegate_task tool calls.
# See documentation in M3_WORKFLOW_REFERENCE.md or the orchestrator implementation.
