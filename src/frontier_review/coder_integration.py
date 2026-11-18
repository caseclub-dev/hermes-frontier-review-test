"""Coder integration adapter for Frontier Review — M3.

Provides structured interfaces for delegating programming assignments to
the real Hermes Coder agent (profile="coder") via the Orchestrator's
delegate_task tool. Also provides mock review helpers that exercise the
same review pipeline against real Coder output, and a verifier that
inspects Git commits and test evidence from completed delegations.

Design principles:
- The adapter does NOT call delegate_task directly (that is an
  Orchestrator-only capability). It builds properly structured
  request payloads that the orchestrator can submit.
- The verifier inspects only temporary workspaces outside this project.
- Mock review uses the existing MockAnthropicTransport — same pipeline,
  no real API calls.
- No Anthropic API key is read or transmitted at any point.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ── Delegation result ───────────────────────────────────────────────

@dataclass
class DelegationResult:
    """Structured results returned from a Coder delegation."""

    task_id: str
    success: bool = False
    files_created: list[str] = field(default_factory=list)
    commit_sha: Optional[str] = None
    test_output: str = ""
    summary_text: str = ""
    errors: list[str] = field(default_factory=list)
    corrected_findings: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """True if any files were created and a commit was made."""
        return bool(self.files_created and self.commit_sha)


# ── Review feedback ─────────────────────────────────────────────────

class Severity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FindingCategory(str, Enum):
    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    STYLE = "style"
    EVIDENCE = "evidence"


@dataclass
class CodeReviewFeedback:
    """Findings from a review of code produced by Coder."""

    status: str  # "pass", "changes_required", "needs_evidence", "blocked"
    summary: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    requirements_met: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)

    def is_changes_required(self) -> bool:
        return self.status == "changes_required"

    def is_correctness_issue(self, finding: dict) -> bool:
        return (finding.get("severity") == Severity.HIGH.value and
                finding.get("category") == FindingCategory.CORRECTNESS.value)

    def correctness_findings(self) -> list[dict]:
        return [f for f in self.findings if self.is_correctness_issue(f)]

    def as_description(self) -> str:
        parts = [self.summary]
        for f in self.findings:
            parts.append(
                f"- [{f.get('severity', 'LOW')}] {f.get('title', '')}: "
                f"{f.get('description', '')}"
            )
        return "\n".join(parts)


# ── Request builder ─────────────────────────────────────────────────

def build_initial_delegation_request(
    task_id: str,
    requirements: str,
    workspace_path: str,
    acceptance_criteria: list[str],
    code_context: str = "",
) -> str:
    """Build a delegation request for the initial Coder implementation.

    This is the first delegation in the correction loop — Orchestrator
    sends it to delegate_task with profile="coder".
    """
    criteria_section = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(acceptance_criteria))
    request = (
        f"# Coder Assignment: {task_id}\n\n"
        f"**Workspace**: Use `{workspace_path}` as the target directory.\n"
        f"**Do not modify** any files outside this workspace.\n\n"
        f"{requirements}\n\n"
        f"**Acceptance Criteria:**\n{criteria_section}\n\n"
        "**Constraints:**\n"
        "- Write all code in Python only.\n"
        "- All public functions MUST have type hints for parameters and return types.\n"
        "- All operations that can raise exceptions MUST be wrapped in try/except.\n"
        f"- Run the provided test script to verify correctness before completing.\n"
        "- Create a git commit after implementation with a descriptive message.\n"
        "- Return the list of created files, the commit SHA, and the full test output.\n\n"
        f"**Context:**\n{code_context}"
    )
    return request


def build_correction_delegation_request(
    task_id: str,
    workspace_path: str,
    findings_text: str,
    acceptance_criteria: list[str],
) -> str:
    """Build a correction delegation request after review feedback."""
    criteria_section = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(acceptance_criteria))
    return (
        f"# Coder Correction: {task_id}\n\n"
        f"**Workspace**: `{workspace_path}`\n"
        f"**Previous commit has been reviewed. Fix these issues:**\n\n{findings_text}\n\n"
        f"**Acceptance Criteria:**\n{criteria_section}\n\n"
        "**Constraints:**\n"
        "- Only modify files that exist in `{workspace_path}`.\n"
        "- Do NOT change the module path or public API signature.\n"
        "- Add any missing error handling and type hints.\n"
        "- Fix the functional defects identified above.\n"
        "- Run tests again and return new test output plus commit SHA.\n\n"
        "**Important:** Previous commits already exist — amend your changes with a single git commit."
    )


# ── Response parser ─────────────────────────────────────────────────

def parse_delegation_response(response_text: str) -> DelegationResult:
    """Parse the text summary returned by delegate_task.

    Extracts file list, commit SHA, test output from Coder's natural-
    language summary using heuristic parsing.
    """
    result = DelegationResult(task_id="unknown")
    lines = response_text.split("\n")

    for line in lines:
        stripped = line.strip()

        # Extract files created
        if "file(s) created" in stripped.lower() or "files created:" in stripped.lower():
            # Try to find filenames after the label
            parts = stripped.split(":")
            if len(parts) > 1:
                file_part = parts[1].strip()
                files = [f.strip().strip("`*").strip() for f in file_part.split(",")]
                result.files_created = [f for f in files if f and f.endswith(".py")]

        # Extract commit SHA
        import re
        sha_match = re.search(r'[a-f0-9]{40}', stripped)
        if sha_match:
            result.commit_sha = sha_match.group(0)

        # Extract "files:" or "Files:"
        files_match = re.search(r'(?i)(?:files?|created)\s*[:\-]\s*\[([^\]]+)\]', line)
        if files_match and not result.files_created:
            file_part = files_match.group(1)
            files = [f.strip().strip("`*").strip() for f in file_part.split(",")]
            result.files_created = [f for f in files if f and f.endswith(".py")]

    # Fallback: try to find .py filenames in the response
    if not result.files_created:
        py_files = re.findall(r'`(\S+\.py)`', response_text)
        if py_files:
            result.files_created = list(set(py_files))

    # Detect success/failure
    lower = response_text.lower()
    errors = []
    if "error" in lower and ("cannot" in lower or "failed" in lower):
        result.success = False
    elif "successfully" in lower or "completed" in lower:
        result.success = True
    else:
        result.success = result.has_changes

    # Extract test output — look for "test output" / "tests run" sections
    test_match = re.search(
        r'(?:test\s+output|tests?\s+r[ue]n|pytest|result)\s*[:\-]\s*\n((?:[^\n]+\n?)+)',
        response_text, re.IGNORECASE
    )
    if test_match:
        result.test_output = test_match.group(1).strip()
    elif "passed" in lower or "PASSED" in lower:
        result.test_output = "Tests passed."

    # Summary is the full response text truncated
    result.summary_text = response_text.strip()

    return result


def build_mock_review_payload(
    status: str,
    summary: str,
    findings: list[dict[str, Any]] | None = None,
    requirements_met: list[str] | None = None,
    suggestions: list[str] | None = None,
) -> str:
    """Build a mock review response that mimics Anthropic's structure.

    This produces text matching the MockAnthropicTransport.ok_response()
    format — specifically: {status, summary, findings, requirements_met, suggestions}.
    """
    payload = {
        "status": status,
        "summary": summary,
        "findings": findings or [],
        "requirements_met": requirements_met or [],
        "suggestions": suggestions or [],
    }
    return json.dumps(payload)


# ── Verifier ────────────────────────────────────────────────────────

class GitWorkspaceVerifier:
    """Inspect a Coder workspace for commits, file changes, and test output."""

    def __init__(self, workspace_path: str) -> None:
        self.workspace_path = workspace_path

    def list_created_files(self) -> list[str]:
        """Return all .py files in the workspace (excluding __pycache__)."""
        try:
            result = subprocess.run(
                ["find", self.workspace_path, "-name", "*.py", "-not", "-path", "*/__pycache__/*"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        except (subprocess.TimeoutExpired, OSError):
            pass
        return []

    def get_commit_count(self) -> int:
        """Return number of commits in the workspace repo."""
        try:
            result = subprocess.run(
                ["git", "-C", self.workspace_path, "rev-list", "--count", "HEAD"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass
        return 0

    def get_last_commit_sha(self) -> Optional[str]:
        """Return the latest commit SHA."""
        try:
            result = subprocess.run(
                ["git", "-C", self.workspace_path, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass
        return None

    def get_last_commit_message(self) -> Optional[str]:
        """Return the latest commit message."""
        try:
            result = subprocess.run(
                ["git", "-C", self.workspace_path, "log", "-1", "--format=%s"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass
        return None

    def get_diff(self) -> str:
        """Return git diff from working tree."""
        try:
            result = subprocess.run(
                ["git", "-C", self.workspace_path, "diff", "--stat"],
                capture_output=True, text=True, timeout=10
            )
            return result.stdout if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, OSError):
            return ""

    def run_tests(self, test_script: Optional[str] = None) -> dict[str, Any]:
        """Run the test script in the workspace and capture output."""
        passed = 0
        errors: list[str] = []

        # If a specific test script is provided, run it
        if test_script:
            full_path = os.path.join(self.workspace_path, test_script) if not test_script.startswith("/") else test_script
            import subprocess as sp
            try:
                result = sp.run(
                    ["python", "-m", "pytest", full_path, "-v", "--tb=short"],
                    capture_output=True, text=True, timeout=60, cwd=self.workspace_path
                )
                output = result.stdout + result.stderr
                passed = output.count(" passed") if " passed" in output else 0
            except (sp.TimeoutExpired, OSError) as e:
                errors.append(f"Test execution failed: {e}")
            return {"passed": passed, "errors": errors, "output": output if 'output' not in dir() else ""}

        # Otherwise try to run pytest on the entire workspace
        import subprocess as sp
        try:
            result = sp.run(
                ["python", "-m", "pytest", self.workspace_path, "-v", "--tb=short", "-x"],
                capture_output=True, text=True, timeout=60
            )
            output = result.stdout + result.stderr
            if "passed" in output:
                import re
                m = re.search(r'(\d+) passed', output)
                passed = int(m.group(1)) if m else 0
            return {"passed": passed, "errors": errors, "output": output}
        except (sp.TimeoutExpired, OSError) as e:
            errors.append(f"Test execution failed: {e}")
            return {"passed": 0, "errors": errors, "output": str(e)}

    def verify_python_syntax(self) -> dict[str, Any]:
        """Verify all Python files have valid syntax."""
        py_files = self.list_created_files()
        errors: list[str] = []
        import subprocess as sp
        for f in py_files:
            try:
                result = sp.run(
                    ["python", "-c", f"import py_compile; py_compile.compile('{f}', doraise=True)"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    errors.append(f"{f}: {result.stderr.strip()}")
            except sp.TimeoutExpired:
                errors.append(f"{f}: syntax check timed out")
        return {"valid": len(errors) == 0, "errors": errors}


def compute_evidence_fingerprint(commit_sha: str, test_output: str) -> str:
    """Compute a fingerprint of the commit+test evidence.

    Used to detect no-op corrections (unchanged commit + unchanged evidence).
    Same algorithm as Orchestrator.compute_fingerprint().
    """
    import hashlib
    raw = f"{commit_sha}|{test_output}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Mock review helper (uses transport internally) ──────────────────

def build_correction_context(findings_text: str, acceptance_criteria: list[str]) -> dict[str, Any]:
    """Build a structured correction context from review findings."""
    return {
        "findings": findings_text,
        "acceptance_criteria": acceptance_criteria,
    }


class MockReviewSimulator:
    """Build mock review responses using the same pipeline as production.

    This class configures the existing MockAnthropicTransport and produces
    review payloads that match Anthropic's response format exactly.
    It does NOT make real API calls.
    """

    def __init__(self) -> None:
        from .transport import MockAnthropicTransport, ScenarioConfig
        self.transport_class = MockAnthropicTransport
        self.scenario_class = ScenarioConfig

    @staticmethod
    def create_pass_review() -> str:
        return build_mock_review_payload(
            status="pass",
            summary="All acceptance criteria met. Code is ready.",
            requirements_met=["type hints present", "error handling in place", "tests pass"],
        )

    @staticmethod
    def create_changes_required(findings: list[dict[str, str]]) -> str:
        return build_mock_review_payload(
            status="changes_required",
            summary="Code review identified defects requiring correction.",
            findings=findings,
            suggestions=["Add comprehensive docstrings"],
        )

    @staticmethod
    def create_block_review(reason: str) -> str:
        return build_mock_review_payload(
            status="blocked",
            summary=f"Review blocked: {reason}",
        )


# ── Correction loop runner ──────────────────────────────────────────

class CorrectionLoopRunner:
    """Orchestrate the full correction loop with real Coder delegation.

    Usage:
        runner = CorrectionLoopRunner(workspace_path)
        result = runner.run(
            initial_request_text=...,  # Built by build_initial_delegation_request()
            mock_review_result=mock_payload_string,
            max_cycles=3,
        )
    """

    MAX_CORRECTION_CYCLES = 3
    MAX_REVIEW_DISPATCHES = 4

    def __init__(self, workspace_path: str) -> None:
        self.workspace_path = workspace_path
        self.verifier = GitWorkspaceVerifier(workspace_path)

    def verify_workspace(self) -> dict[str, Any]:
        """Verify a completed Coder workspace."""
        files = self.verifier.list_created_files()
        sha = self.verifier.get_last_commit_sha()
        msg = self.verifier.get_last_commit_message()
        diff = self.verifier.get_diff()
        syntax = self.verifier.verify_python_syntax()

        return {
            "files_created": files,
            "commit_count": self.verifier.get_commit_count(),
            "last_commit_sha": sha,
            "last_commit_message": msg,
            "diff": diff,
            "has_changes": bool(files and sha),
            "syntax_valid": syntax["valid"],
            "syntax_errors": syntax["errors"],
        }

    def run(
        self,
        initial_request_text: str,
        mock_review_result: str,
        max_cycles: int = MAX_CORRECTION_CYCLES,
    ) -> dict[str, Any]:
        """Run the correction loop.

        This is a planning function — it builds the sequence of actions
        that Orchestrator should take when delegating to Coder. The actual
        delegate_task calls are made by the orchestrator in its own context.

        Returns a structured plan with:
          - Initial delegation payload
          - Mock review result (to parse and evaluate)
          - If correction needed: correction request payload
          - Verification steps
        """
        import re

        # Parse mock review result
        try:
            review_data = json.loads(mock_review_result)
        except (json.JSONDecodeError, TypeError):
            return {
                "status": "blocked",
                "summary": "Cannot parse mock review result",
                "steps": [],
            }

        status = review_data.get("status", "")
        summary = review_data.get("summary", "")
        findings = review_data.get("findings", [])

        steps = [{
            "action": "initial_delegation",
            "payload": initial_request_text,
        }]

        if status == "pass":
            return {
                "status": "ready_for_operator_review",
                "summary": summary,
                "steps": steps,
                "corrected_sha": None,
            }

        # Build correction request
        criteria = [c for c in initial_request_text.split("Acceptance Criteria:")[-1].split("\n\n") if c.strip()][:5]
        criteria_clean = re.findall(r'\d+\.\s+(.+)', "\n".join(criteria))
        if not criteria_clean:
            criteria_clean = ["Fix identified defects and ensure all tests pass."]

        corrections_text = "\n".join(
            f"  [{f.get('severity', 'LOW')}] {f.get('title', '')}: "
            f"{f.get('description', '')}"
            for f in findings
        )

        steps.append({
            "action": "correction_delegation",
            "payload": build_correction_delegation_request(
                task_id="m3-correction",
                workspace_path=self.workspace_path,
                findings_text=corrections_text,
                acceptance_criteria=criteria_clean or ["Fix identified defects"],
            ),
        })

        # Verify steps
        steps.append({"action": "verify_workspace"})
        steps.append({"action": "re-review"})

        return {
            "status": status,
            "summary": summary,
            "findings": findings,
            "steps": steps,
            "max_cycles_reached": False,
        }


def create_disposable_workspace(base_dir: str = "/tmp", prefix: str = "frontier_coder_test_") -> str:
    """Create a disposable Git workspace for Coder work.

    Returns the absolute path to the new directory.
    The directory is empty except for a .gitignore file.
    The Orchestrator must initialize git inside it after files are written.
    """
    import os, tempfile, subprocess

    # Find unique name by creating temp dir then moving it
    tmpdir = tempfile.mkdtemp(dir=base_dir, prefix=prefix)

    # Initialize as a git repo
    try:
        subprocess.run(
            ["git", "init", tmpdir],
            capture_output=True, text=True, timeout=10
        )
        # Set up initial config to allow commits without remote
        subprocess.run(
            ["git", "-C", tmpdir, "config", "user.name", "frontier-m3-test"],
            capture_output=True, text=True, timeout=5
        )
        subprocess.run(
            ["git", "-C", tmpdir, "config", "user.email", "m3-test@frontier-review.local"],
            capture_output=True, text=True, timeout=5
        )
    except (subprocess.TimeoutExpired, OSError):
        pass

    return tmpdir


# ── Failure handling utilities ──────────────────────────────────────

def handle_delegation_failure(
    error_type: str,
    response_text: str = "",
) -> dict[str, Any]:
    """Handle various delegation failure modes.

    Returns a structured result with the appropriate terminal status.
    """
    if error_type == "timeout":
        return {
            "status": "blocked",
            "summary": "Coder delegation timed out — task may not have completed.",
            "terminal": True,
        }
    elif error_type == "no_commit":
        return {
            "status": "changes_required",
            "summary": "Coder returned no commit — correction failed.",
            "findings": [{"title": "No git commit found", "severity": "HIGH",
                          "category": "evidence", "description": response_text}],
            "terminal": False,
        }
    elif error_type == "no_changes":
        return {
            "status": "changes_required",
            "summary": "Coder made no changes — correction was a no-op.",
            "findings": [{"title": "No code changes detected", "severity": "HIGH",
                          "category": "correctness", "description": response_text}],
            "terminal": False,
        }
    elif error_type == "malformed_review":
        return {
            "status": "blocked",
            "summary": f"Mock review returned malformed data: {response_text[:100]}",
            "terminal": True,
        }
    elif error_type == "max_cycles_exhausted":
        return {
            "status": "escalated_architect",
            "summary": "Maximum correction cycles reached without PASS.",
            "blockers": ["Exceeded max correction cycles"],
            "terminal": True,
        }
    elif error_type == "tests_failed":
        return {
            "status": "changes_required",
            "summary": f"Tests failed after correction: {response_text[:200]}",
            "findings": [{"title": "Test failures", "severity": "HIGH",
                          "category": "correctness", "description": response_text}],
            "terminal": False,
        }
    else:
        return {
            "status": "blocked",
            "summary": f"Unhandled delegation failure: {error_type}",
            "terminal": True,
        }


def is_duplicate_review(review_count: int, max_reviews: int = 4) -> bool:
    """Check if a review dispatch would exceed the maximum."""
    return review_count >= max_reviews


def is_max_correction_cycles(max_corrections: int, current: int) -> bool:
    """Check if correction cycles are exhausted."""
    return current >= max_corrections
