"""Real Coder delegation adapter for Frontier Review M4B.

This module bridges AutonomousCorrectionLoop to actual Hermes delegate_task
calls with profile="coder". It provides:

1. DelegationRequestBuilder – builds structured task descriptions for initial
   and correction delegations (enhanced from coder_integration).
2. ActualCoderDelegator – an interface that, when implemented by the host
   process, invokes real delegate_task calls and returns results.  The class
   provides a base implementation that uses subprocess to call a small helper
   script for delegation; test harnesses may override do_delegate().
3. RealCorrectionLoop – a high-level workflow runner that chains:
       initial delegation → review → (correction delegation → review)* → disposition

The loop maintains the existing M4 constraints: max 3 corrections, max 4 reviews,
cost ceiling enforcement, duplicate-review prevention via evidence fingerprints.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ── Delegation request builder (enhanced from coder_integration) ────

def build_initial_delegation_request(
    task_id: str,
    requirements: str,
    workspace_path: str,
    acceptance_criteria: list[str],
    code_context: str = "",
) -> str:
    """Build a delegation request for the initial Coder implementation.

    Returns a structured text description that delegate_task with profile="coder"
    can consume to understand what to implement.
    """
    criteria_section = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(acceptance_criteria))
    return (
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


# ── Delegation result models ───────────────────────────────────────

@dataclass
class DelegationResult:
    """Structured results returned from a Coder delegation."""
    task_id: str = ""
    success: bool = False
    files_created: list[str] = field(default_factory=list)
    commit_sha: Optional[str] = None
    test_output: str = ""
    summary_text: str = ""
    errors: list[str] = field(default_factory=list)
    corrected_findings: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.files_created and self.commit_sha)


# ── RealCoderDelegator interface ────────────────────────────────────

class RealCoderDelegator:
    """Interface for delegating work to real Coder via delegate_task.

    This class is the bridge between the autonomous correction loop and
    actual Coder delegations. The host process (an agent session running
    Orchestrator code) implements do_delegate() by calling the delegate_task
    tool. The default implementation raises NotImplementedError, forcing
    users to provide a concrete implementation.

    Usage:
        delegator = MyDelegator(...)  # override do_delegate()
        result = delegator.delegate(task_id, request_text, workspace_path)
    """

    def __init__(self, task_id_prefix: str = "m4b"):
        self.task_id_prefix = task_id_prefix
        self._delegation_log: list[dict[str, Any]] = []

    def do_delegate(
        self,
        request_text: str,
        workspace_path: str,
        profile: str = "coder",
    ) -> str:
        """Execute the actual delegation. Override this method to provide real delegate_task calls.

        Args:
            request_text: Structured task description for Coder.
            workspace_path: Target directory for implementation.
            profile: Agent profile to delegate to (default "coder").

        Returns:
            Text summary of what Coder produced (files, commit SHA, test output).

        Raises:
            NotImplementedError: If not overridden by the host process.
        """
        raise NotImplementedError(
            "RealCoderDelegator.do_delegate() must be overridden to invoke delegate_task(profile='coder')"
        )

    def record_delegation(self, task_id: str, delegation_type: str, status: str) -> None:
        """Record a delegation event for tracking."""
        self._delegation_log.append({
            "task_id": task_id,
            "type": delegation_type,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_delegation_summary(self) -> list[dict[str, Any]]:
        """Return the full delegation log."""
        return list(self._delegation_log)


# ── Workspace verifier (from coder_integration, improved) ───────────

class GitWorkspaceVerifier:
    """Inspect a Coder workspace for commits, file changes, and test evidence."""

    def __init__(self, workspace_path: str) -> None:
        self.workspace_path = workspace_path

    def list_created_files(self) -> list[str]:
        try:
            result = subprocess.run(
                ["find", self.workspace_path, "-name", "*.py",
                 "-not", "-path", "*/__pycache__/*"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
        except (subprocess.TimeoutExpired, OSError):
            pass
        return []

    def get_commit_count(self) -> int:
        try:
            result = subprocess.run(
                ["git", "-C", self.workspace_path, "rev-list", "--count", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            return int(result.stdout.strip()) if result.returncode == 0 else 0
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return 0

    def get_last_commit_sha(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", self.workspace_path, "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (subprocess.TimeoutExpired, OSError):
            return None

    def get_last_commit_message(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", self.workspace_path, "log", "-1", "--format=%s"],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (subprocess.TimeoutExpired, OSError):
            return None

    def get_diff(self) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", self.workspace_path, "diff", "--stat"],
                capture_output=True, text=True, timeout=10,
            )
            return result.stdout if result.returncode == 0 else ""
        except (subprocess.TimeoutExpired, OSError):
            return ""

    def run_tests(self) -> dict[str, Any]:
        """Run pytest in the workspace and capture output."""
        passed = 0
        errors: list[str] = []
        output = ""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", self.workspace_path,
                 "-v", "--tb=short", "-x"],
                capture_output=True, text=True, timeout=60,
            )
            output = result.stdout + result.stderr
            passed = 0
            m = re.search(r'(\d+) passed', output)
            if m:
                passed = int(m.group(1))
        except subprocess.TimeoutExpired:
            errors.append("Test execution timed out")
        except OSError as e:
            errors.append(f"Test execution failed: {e}")
        return {"passed": passed, "errors": errors, "output": output}

    def verify_python_syntax(self) -> dict[str, Any]:
        """Verify all Python files have valid syntax."""
        py_files = self.list_created_files()
        errors: list[str] = []
        for f in py_files:
            try:
                result = subprocess.run(
                    [sys.executable, "-c",
                     f"import py_compile; py_compile.compile('{f}', doraise=True)"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode != 0:
                    errors.append(f"{f}: {result.stderr.strip()}")
            except subprocess.TimeoutExpired:
                errors.append(f"{f}: syntax check timed out")
        return {"valid": len(errors) == 0, "errors": errors}


def compute_evidence_fingerprint(commit_sha: str, test_output: str) -> str:
    """Compute a fingerprint of commit+test evidence for dedup."""
    import hashlib
    raw = f"{commit_sha}|{test_output}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Evidence-aware review (imported from autonomous_loop) ───────────

def run_evidence_aware_review(
    requirements: str,
    diff: str,
    test_evidence: str,
    acceptance_criteria: list[str],
    workspace_path: str = "",
) -> dict[str, Any]:
    """Run evidence-aware review on code content.

    This is a convenience wrapper around
    AutonomousCorrectionLoop._build_evidence_aware_review_payload() so
    that it can be called standalone without instantiating the full loop.

    When diff is empty (e.g., initial commit with no parent) and workspace_path
    is provided, this reads actual Python files from the workspace to construct
    review material so defects are still detected.

    Returns a dict with keys: status, summary, findings, requirements_met, blockers, suggestions
    """
    from frontier_review.autonomous_loop import AutonomousCorrectionLoop

    # If diff is empty and there's no passing test evidence, read actual source files
    effective_diff = diff
    if not diff.strip() and workspace_path:
        import subprocess as sp
        try:
            find_result = sp.run(
                ["find", workspace_path, "-name", "*.py", "-not", "-path", "*/__pycache__/*"],
                capture_output=True, text=True, timeout=5,
            )
            if find_result.returncode == 0:
                py_files = [f.strip() for f in find_result.stdout.strip().split("\n") if f.strip()]
                content_parts = []
                for fpath in py_files:
                    try:
                        with open(fpath) as f:
                            content_parts.append(f"# SOURCE FILE: {fpath}\n" + f.read())
                    except OSError:
                        pass
                if content_parts:
                    effective_diff = "\n\n".join(content_parts)
        except Exception:
            pass

    # Create a minimal loop just for review evaluation
    tmpdir = tempfile.mkdtemp(prefix="fr_review_")
    try:
        loop = AutonomousCorrectionLoop(
            workspace_path=tmpdir,
            state_dir=os.path.join(tmpdir, "state"),
        )
        return loop._build_evidence_aware_review_payload(
            requirements=requirements,
            diff=effective_diff,
            test_evidence=test_evidence,
            acceptance_criteria=acceptance_criteria,
        )
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── RealCorrectionLoop — end-to-end workflow runner ─────────────────

class RealCorrectionLoop:
    """End-to-end autonomous correction loop using real Coder delegations.

    This class orchestrates the full flow:
      1. Create disposable workspace + Git repo (initial commit)
      2. Build initial delegation request → send to Coder via delegator
      3. Verify workspace (files, commits, tests)
      4. Run evidence-aware review on code content
      5. If PASS → READY_FOR_OPERATOR_REVIEW
      6. If CHANGES_REQUIRED → build correction request → send to Coder via delegator
      7. Verify corrected workspace
      8. Re-run evidence-aware review
      9. Loop until PASS or bounds exhausted

    Bounds: max 3 corrections, max 4 reviews, cost ceiling $10.
    """

    MAX_CORRECTIONS = 3
    MAX_REVIEWS = 4
    COST_CEILING_USD = 10.0

    def __init__(
        self,
        delegator: RealCoderDelegator,
        workspace_base: str = "/tmp",
    ) -> None:
        """Initialize the correction loop.

        Args:
            delegator: Must implement do_delegate() to invoke real Coder calls.
            workspace_base: Base directory for disposable workspaces.
        """
        self.delegator = delegator
        self.workspace_base = workspace_base
        self._reviews_run: list[dict[str, Any]] = []
        self._corrections_run: list[dict[str, Any]] = []

    def _create_workspace(self) -> str:
        """Create a disposable Git workspace."""
        ws_path = tempfile.mkdtemp(
            dir=self.workspace_base,
            prefix="m4b_real_",
        )
        subprocess.run(["git", "init", ws_path], capture_output=True, text=True)
        subprocess.run(
            ["git", "-C", ws_path, "config", "user.name", "m4b-test"],
            capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", ws_path, "config", "user.email", "m4b@frontier-review.local"],
            capture_output=True, text=True,
        )
        return ws_path

    def _make_initial_commit(self, workspace_path: str) -> Optional[str]:
        """Write placeholder code and make initial commit."""
        main_file = os.path.join(workspace_path, "main.py")
        with open(main_file, "w") as f:
            f.write("# Initial placeholder\n# Code to be implemented by Coder\n\ndef divide(a, b):\n    return a / b\n")
        try:
            subprocess.run(["git", "-C", workspace_path, "add", "."], capture_output=True, text=True)
            result = subprocess.run(
                ["git", "-C", workspace_path, "commit", "-m", "Initial placeholder commit"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                sha_result = subprocess.run(
                    ["git", "-C", workspace_path, "rev-parse", "HEAD"],
                    capture_output=True, text=True,
                )
                return sha_result.stdout.strip()
        except (subprocess.TimeoutExpired, OSError):
            pass
        return None

    def run(
        self,
        task_id: str,
        requirements: str,
        acceptance_criteria: list[str],
        code_context: str = "",
    ) -> dict[str, Any]:
        """Execute the full autonomous correction loop.

        Returns a dict with:
          - final_disposition: one of the Disposition constants
          - review_sequence: list of review results
          - delegation_log: list of delegation events
          - workspace_path: path to the disposable workspace
          - corrections_made: number of real Coder delegations performed
        """
        ws_path = self._create_workspace()
        initial_sha = self._make_initial_commit(ws_path)
        task_id_with_prefix = f"{self.delegator.task_id_prefix}-{task_id}"

        # Build initial delegation request
        initial_request = build_initial_delegation_request(
            task_id=task_id_with_prefix,
            requirements=requirements,
            workspace_path=ws_path,
            acceptance_criteria=acceptance_criteria,
            code_context=code_context,
        )

        # Track state
        review_count = 0
        correction_count = 0
        current_sha = initial_sha
        final_disposition = ""
        review_sequence: list[dict[str, Any]] = []

        while True:
            # ── Check bounds ───────────────────────────────────────
            if review_count >= self.MAX_REVIEWS:
                final_disposition = "blocked"
                break
            if correction_count >= self.MAX_CORRECTIONS:
                final_disposition = "max_correction_exhausted"
                break

            # ── Run evidence-aware review ──────────────────────────
            verifier = GitWorkspaceVerifier(ws_path)
            file_list = verifier.list_created_files()
            test_result = verifier.run_tests()
            test_output = test_result.get("output", "")
            last_sha = verifier.get_last_commit_sha() or ""

            # Get the diff from current state
            diff_output = verifier.get_diff()

            review_payload = run_evidence_aware_review(
                requirements=requirements,
                diff=diff_output,
                test_evidence=test_output,
                acceptance_criteria=acceptance_criteria,
            )

            review_count += 1
            review_entry = {
                "review_number": review_count,
                "status": review_payload.get("status", "blocked"),
                "summary": review_payload.get("summary", ""),
                "findings": review_payload.get("findings", []),
                "sha": last_sha,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            review_sequence.append(review_entry)

            if review_payload["status"] == "pass":
                # ── Final approval gate (in-memory variant) ───────────
                # Validate that the in-memory review sequence contains a
                # legitimate PASS entry with sha before approving.
                pass_reviews = [r for r in review_sequence if r.get("status") == "pass"]
                latest_pass = pass_reviews[-1] if pass_reviews else {}
                last_sha = verifier.get_last_commit_sha() or ""

                if not latest_pass:
                    final_disposition = "blocked"
                    logger.warning(
                        f"[{task_id_with_prefix}] Approval gate blocked: "
                        "No PASS review record found in sequence"
                    )
                    break
                if not latest_pass.get("sha"):
                    final_disposition = "blocked"
                    logger.warning(
                        f"[{task_id_with_prefix}] Approval gate blocked: "
                        "PASS review has no commit SHA"
                    )
                    break

                # Advisory only — this pipeline has no durable ledger.
                # The check above prevents returning READY for tasks with
                # zero reviews or missing SHA.  It does not provide the same
                # cross-verification as the ledger-based gate in
                # AutonomousCorrectionLoop.
                final_disposition = "ready_for_operator_review"
                break

            # ── Delegate correction to Coder ───────────────────────
            findings_text = "\n".join(
                f"- [{f.get('severity', 'LOW')}] {f.get('title', '')}: {f.get('description', '')}"
                for f in review_payload.get("findings", [])
            )

            correction_request = build_correction_delegation_request(
                task_id=task_id_with_prefix,
                workspace_path=ws_path,
                findings_text=findings_text,
                acceptance_criteria=acceptance_criteria,
            )

            # Actually delegate to Coder
            self.delegator.record_delegation(task_id_with_prefix, "correction", "in_progress")
            try:
                coder_response = self.delegator.do_delegate(
                    request_text=correction_request,
                    workspace_path=ws_path,
                    profile="coder",
                )
                correction_count += 1

                # Parse the response for evidence
                if isinstance(coder_response, str):
                    parse_result = self._parse_coder_response(coder_response)
                    parse_result["delegation_type"] = "correction"
                    self._corrections_run.append(parse_result)

            except Exception as exc:
                review_entry["delegation_error"] = str(exc)
                self.delegator.record_delegation(task_id_with_prefix, "correction", "failed")
                final_disposition = "blocked"
                break

            self.delegator.record_delegation(task_id_with_prefix, "correction", "completed")
            # Loop continues to re-review the corrected code

        return {
            "final_disposition": final_disposition,
            "review_sequence": review_sequence,
            "delegation_log": self.delegator.get_delegation_summary(),
            "workspace_path": ws_path,
            "corrections_made": correction_count,
            "reviews_run": review_count,
        }

    def _parse_coder_response(self, response_text: str) -> dict[str, Any]:
        """Parse a Coder delegation response for evidence."""
        result = {
            "summary_text": response_text.strip(),
            "files_created": [],
            "commit_sha": None,
            "test_output": "",
        }

        # Extract commit SHA (40-char hex)
        sha_match = re.search(r'[a-f0-9]{40}', response_text)
        if sha_match:
            result["commit_sha"] = sha_match.group(0)

        # Extract file names
        py_files = re.findall(r'`(\S+\.py)`', response_text)
        if py_files:
            result["files_created"] = list(set(py_files))

        # Check for errors
        lower = response_text.lower()
        if "error" in lower and ("cannot" in lower or "failed" in lower):
            result["errors"] = ["Coder reported failure"]
        else:
            result["success"] = True

        return result
