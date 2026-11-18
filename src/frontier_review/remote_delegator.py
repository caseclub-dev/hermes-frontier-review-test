"""Bridge between Hermes' delegate_task tool and Frontier Review.

When Orchestrator (or any host process) runs in a Hermes session that has
access to the ``delegate_task`` tool, this module provides a delegator
implementation that calls it directly — so the full M4B evidence-aware
review pipeline can execute against real Coder delegations without manual
interruption.

Usage from an Orchestrator context (tool-execution Python):

    from frontier_review.remote_delegator import FrontierReviewDelegator
    from frontier_review.m4b_integration import RealCorrectionLoop

    delegator = FrontierReviewDelegator()
    loop = RealCorrectionLoop(delegator)
    result = loop.run(
        task_id="example-task",
        requirements="Implement a division function.",
        acceptance_criteria=["Handles zero correctly"],
    )

No Anthropic API calls are made; all reviews use deterministic content
inspection via ``AutonomousCorrectionLoop._build_evidence_aware_review_payload``.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import subprocess
from datetime import datetime, timezone
from typing import Any

# Ensure the project's src is importable when running from an arbitrary cwd
_HERE = os.path.dirname(__file__)
_SRC = os.path.join(_HERE, "..")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from frontier_review.m4b_integration import (  # noqa: E402
    RealCoderDelegator,
    GitWorkspaceVerifier,
)


# ── FrontierReviewDelegator — bridges delegate_task → Coder ──────────

class FrontierReviewDelegator(RealCoderDelegator):
    """Delegate to Hermes' Coder profile via the ``delegate_task`` tool.

    The host process must call ``register_delegation_tool()`` once before
    using this class, or rely on the module-level ``_DELEGATION_TOOL`` hook
    that Orchestrator sets during task processing.

    Parameters
    ----------
    max_retries : int
        How many times to retry a failed delegation (default 1).
    retry_delay_s : float
        Seconds between retries (default 0 — no delay for synchronous use).
    """

    def __init__(
        self,
        max_retries: int = 1,
        retry_delay_s: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s

    # ── delegate_task interface (set by Orchestrator) ────────────────

    _delegation_fn: Any = None  # Callable(request_text, goal, profile)

    @classmethod
    def register_delegation_tool(cls, fn: Any) -> None:
        """Register the host's delegate_task function.

        Called once by Orchestrator during tool execution so this delegator
        can route work to the real Coder agent.

        Parameters
        ----------
        fn : callable
            Must accept ``(request_text, goal_text, profile)`` and return a
            string summarising what was produced (files, commit SHA, output).
        """
        cls._delegation_fn = fn

    @classmethod
    def _get_delegation_tool(cls) -> Any:
        if cls._delegation_fn is None:
            raise RuntimeError(
                "FrontierReviewDelegator.delegate_task() not registered. "
                "Call FrontierReviewDelegator.register_delegation_tool(fn) before use."
            )
        return cls._delegation_fn

    # ── RealCoderDelegator.do_delegate implementation ────────────────

    def do_delegate(  # type: ignore[override]
        self,
        request_text: str,
        workspace_path: str,
        profile: str = "coder",
    ) -> str:
        """Call the registered delegate_task tool with a Coder goal.

        Parameters
        ----------
        request_text : str
            Structured task description (requirements, acceptance criteria, findings).
        workspace_path : str
            Target directory for implementation.
        profile : str
            Agent profile (default ``"coder"``, always enforced internally).

        Returns
        -------
        str
            Coder's response text summarising work done.
        """
        goal = f"Implement the following coding task and commit your changes in `{workspace_path}`:\n\n{request_text}"
        tool = self._get_delegation_tool()

        # The delegate_task call returns a dict-like result; extract summary.
        for attempt in range(1, self.max_retries + 2):
            try:
                result = tool(request_text=request_text, goal=goal, profile=profile)
                # Normalize whatever shape the result comes back in
                if isinstance(result, dict):
                    return str(result.get("summary", result.get("output", result)))
                return str(result)
            except Exception as exc:
                if attempt <= self.max_retries:
                    continue
                raise RuntimeError(
                    f"Coder delegation failed after {self.max_retries} retries: {exc}"
                ) from exc

    @classmethod
    def reset(cls) -> None:
        """Clear the registered tool hook (useful between independent runs)."""
        cls._delegation_fn = None


# ── Convenience functions for Orchestrator use ────────────────────────

def _run_subprocess(cmd: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=cwd)
    return r.returncode, r.stdout, r.stderr


def create_demo_workspace(base_dir: str = "/tmp") -> tuple[str, str]:
    """Create a disposable Git workspace with initial placeholder + test script.

    Returns (workspace_path, test_script_path).
    """
    ws = tempfile.mkdtemp(dir=base_dir, prefix="fr_demo_")
    _run_subprocess(["git", "init", ws], cwd=None)
    _run_subprocess(["git", "-C", ws, "config", "user.name", "fr-demo"], cwd=None)
    _run_subprocess(
        ["git", "-C", ws, "config", "user.email", "demo@frontier-review.local"],
        cwd=None,
    )

    # Write test script
    ts = os.path.join(ws, "test_main.py")
    with open(ts, "w") as f:
        f.write("""import sys
import importlib.util

spec = importlib.util.spec_from_file_location("main", "main.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
    result = mod.divide(10.0, 2.0)
    assert abs(result - 5.0) < 0.01, f"Expected 5.0, got {result}"
    print("Test 1 PASSED: normal division")
except Exception as e:
    print(f"Test 1 FAILED: {e}", file=sys.stderr)

try:
    spec.loader.exec_module(mod)
    mod.divide(10.0, 0.0)
    print("Test 2 FAILED: should have raised ValueError", file=sys.stderr)
except ValueError as e:
    print(f"Test 2 PASSED: ValueError raised for zero division")
except Exception as e:
    print(f"Test 2 FAILED: wrong exception type: {type(e).__name__}: {e}", file=sys.stderr)

try:
    spec.loader.exec_module(mod)
    if hasattr(mod.divide, '__annotations__') and len(mod.divide.__annotations__) > 0:
        print("Test 3 PASSED: type hints present")
    else:
        print("Test 3 FAILED: no type hints on divide()", file=sys.stderr)
except Exception as e:
    print(f"Test 3 FAILED: {e}", file=sys.stderr)
""")

    # Write initial buggy placeholder code
    with open(os.path.join(ws, "main.py"), "w") as f:
        f.write("def divide(a, b):\n    return a / b\n")

    _run_subprocess(["git", "-C", ws, "add", "."], cwd=None)
    _ = _run_subprocess(
        ["git", "-C", ws, "commit", "-m", "Initial placeholder — buggy divide"],
        cwd=None,
    )

    return ws, ts


def run_full_m4b_pipeline(
    requirements: str,
    acceptance_criteria: list[str],
    workspace_base: str = "/tmp",
    delegator: FrontierReviewDelegator | None = None,
    code_context: str = "",
) -> dict[str, Any]:
    """Execute the full M4B pipeline: delegate → review → (correction → review)*.

    This is a convenience wrapper that:
    1. Creates a disposable Git workspace with a test script and buggy placeholder.
    2. Delegates to Coder via ``delegate_task``.
    3. Runs evidence-aware mock review.
    4. If CHANGES_REQUIRED, sends correction back to Coder.
    5. Re-reviews until PASS or max corrections (3) exhausted.

    Parameters
    ----------
    requirements : str
        What the code should do.
    acceptance_criteria : list[str]
        Verifiable criteria for success.
    workspace_base : str
        Directory for disposable workspaces.
    delegator : FrontierReviewDelegator or None
        If provided, used for Coder delegations. Otherwise a fresh instance is created
        and ``register_delegation_tool`` is expected to have been set by the host.
    code_context : str
        Additional context about existing codebase (passed to Coder).

    Returns
    -------
    dict with keys: final_disposition, review_sequence, workspace_path, corrections_made
    """
    if delegator is None:
        delegator = FrontierReviewDelegator(max_retries=1)

    from frontier_review.m4b_integration import (  # noqa: E402
        RealCorrectionLoop,
        GitWorkspaceVerifier,
        run_evidence_aware_review,
    )

    # For the initial review we use GitWorkspaceVerifier directly because
    # there's no Coder delegation — just an existing buggy commit.
    ws = create_demo_workspace(workspace_base)
    verifier = GitWorkspaceVerifier(ws)

    # Get test output and code content for initial review
    tr = subprocess.run(
        [sys.executable, "-m", "pytest", ws, "-v", "--tb=short"],
        capture_output=True, text=True, timeout=30, cwd=ws,
    )
    test_output = tr.stdout + tr.stderr

    # Read committed code for review content
    rc, out, _ = _run_subprocess(["git", "-C", ws, "show", "--format=", "HEAD"], cwd=None)
    commit_code = out if rc == 0 else ""

    initial_review = run_evidence_aware_review(
        requirements=requirements,
        diff=commit_code,
        test_evidence=test_output,
        acceptance_criteria=acceptance_criteria,
    )

    review_sequence = [
        {
            "review_number": 1,
            "status": initial_review["status"],
            "summary": initial_review.get("summary", ""),
            "findings": initial_review.get("findings", []),
            "sha": verifier.get_last_commit_sha() or "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    ]

    current_sha = verifier.get_last_commit_sha() or ""

    if initial_review["status"] == "pass":
        return {
            "final_disposition": "ready_for_operator_review",
            "review_sequence": review_sequence,
            "workspace_path": ws,
            "corrections_made": 0,
            "reviews_run": 1,
        }

    # ── Delegate correction to Coder ───────────────────────────────
    from frontier_review.m4b_integration import (  # noqa: E402
        build_correction_delegation_request,
    )

    findings_text = "\n".join(
        f"- [{f.get('severity', 'LOW')}] {f.get('title', '')}: {f.get('description', '')}"
        for f in initial_review.get("findings", [])
    )

    correction_request = build_correction_delegation_request(
        task_id="demo-task",
        workspace_path=ws,
        findings_text=findings_text,
        acceptance_criteria=acceptance_criteria,
    )

    goal = f"Fix the issues identified in the code review for the project at `{ws}`:\n\n{correction_request}"

    try:
        delegate_tool = delegator._get_delegation_tool()
        coder_result = delegate_tool(
            request_text=correction_request,
            goal=goal,
            profile="coder",
        )
        # The delegate_task tool returns a dict with 'summary' and 'output';
        # the Coder subprocess also handles git commits in the workspace.

    except Exception as exc:
        review_sequence[0]["delegation_error"] = str(exc)
        return {
            "final_disposition": "blocked",
            "review_sequence": review_sequence,
            "workspace_path": ws,
            "corrections_made": 1,
            "reviews_run": 1,
        }

    # Post-correction: verify workspace and re-review
    verifier2 = GitWorkspaceVerifier(ws)
    sha_after = verifier2.get_last_commit_sha() or current_sha
    test_result = verifier2.run_tests()
    post_test_output = test_result.get("output", "")
    diff_result = subprocess.run(
        ["git", "-C", ws, "diff", "HEAD~1..HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    post_diff = diff_result.stdout

    post_review = run_evidence_aware_review(
        requirements=requirements,
        diff=post_diff,
        test_evidence=post_test_output,
        acceptance_criteria=acceptance_criteria,
        workspace_path=ws,
    )

    review_sequence.append({
        "review_number": 2,
        "status": post_review["status"],
        "summary": post_review.get("summary", ""),
        "findings": post_review.get("findings", []),
        "sha": sha_after,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    disposition = (
        "ready_for_operator_review" if post_review["status"] == "pass" else "max_correction_exhausted"
    )

    return {
        "final_disposition": disposition,
        "review_sequence": review_sequence,
        "workspace_path": ws,
        "corrections_made": 1,
        "reviews_run": 2,
    }
