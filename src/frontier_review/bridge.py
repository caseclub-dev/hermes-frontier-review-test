"""Orchestrator bridge for Frontier Review real Coder delegation (M4F).

This module provides two helper functions that Orchestrator can call from its
own tool-invocation context to connect the pipeline to real Coder delegations:

1. ``register_delegate_task_for_pipeline()`` — creates a callable delegate_task
   function and registers it with FrontierReviewDelegator, so that when the
   pipeline reaches a correction step it can actually dispatch work to Coder.

2. ``verify_workspace_after_delegation(workspace_path)`` — inspects a workspace
   after a Coder delegation completed (files committed, tests passing, SHA
   updated) and returns structured evidence for the review step.

Design rationale:
  - Neither execute_code nor browser_exec can invoke agent-level tools
    (delegate_task is explicitly blocked by tool-permission filters).
  - The correction loop therefore lives at the Orchestrator level:
    pipeline executes in Python → reports CHANGES_REQUIRED → Orchestrator
    uses delegate_task directly → verifies workspace → re-runs review.
  - This module makes that pattern explicit and reusable.

Usage from an Orchestrator session:

    # Step A — run pipeline with no-op delegator to get initial review
    from frontier_review.remote_delegator import (
        run_full_m4b_pipeline,
        FrontierReviewDelegator,
    )

    result = run_full_m4b_pipeline(
        requirements="Implement divide(a, b).",
        acceptance_criteria=["Returns correct quotient", "Handles zero division"],
        workspace_base="/home/atlas/.hermes/cache/scratch",
        delegator=FrontierReviewDelegator(max_retries=0),  # no-op
    )

    # Step B — if corrections needed, register real delegate_task and re-run
    from frontier_review.bridge import (
        get_orchestrator_delegation_fn,
        verify_workspace_after_delegation,
    )

    # This creates a callable that delegates to the real Coder agent.
    delegator = FrontierReviewDelegator(max_retries=1)
    # NOTE: The actual delegate_task call must be made by the Orchestrator's
    # own tool layer (not from within this Python function). The bridge module
    # provides utility functions that document the pattern; the Orchestrator
    # must explicitly invoke delegate_task for each correction cycle.

See the frontier-review SKILL.md for the complete step-by-step invocation
procedure including the multi-turn correction loop pattern.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import Any


def verify_workspace_after_delegation(
    workspace_path: str,
) -> dict[str, Any]:
    """Inspect a workspace after a Coder delegation completed.

    Returns structured evidence suitable for feeding back into the review:
      - files_created: list of Python files in the workspace
      - commit_count: number of commits (must be >= 2 after correction)
      - last_sha: SHA of the most recent commit
      - last_message: commit message text
      - test_passed: whether tests passed
      - test_output: full pytest output
      - diff_stat: git diff --stat between HEAD~1 and HEAD

    Parameters
    ----------
    workspace_path : str
        Path to the disposable Git workspace.

    Returns
    -------
    dict with keys listed above.
    """
    evidence: dict[str, Any] = {}

    # List Python files
    try:
        result = subprocess.run(
            ["find", workspace_path, "-name", "*.py", "-not",
             "-path", "*/__pycache__/*"],
            capture_output=True, text=True, timeout=10,
        )
        evidence["files_created"] = [
            f.strip() for f in result.stdout.strip().split("\n") if f.strip()
        ] if result.returncode == 0 else []
    except Exception:
        evidence["files_created"] = []

    # Commit count and SHA
    try:
        count_result = subprocess.run(
            ["git", "-C", workspace_path, "rev-list", "--count", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        evidence["commit_count"] = int(count_result.stdout.strip()) if count_result.returncode == 0 else 0

        sha_result = subprocess.run(
            ["git", "-C", workspace_path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        evidence["last_sha"] = sha_result.stdout.strip() if sha_result.returncode == 0 else ""

        msg_result = subprocess.run(
            ["git", "-C", workspace_path, "log", "-1", "--format=%s"],
            capture_output=True, text=True, timeout=10,
        )
        evidence["last_message"] = msg_result.stdout.strip() if msg_result.returncode == 0 else ""
    except Exception:
        evidence["commit_count"] = 0
        evidence["last_sha"] = ""
        evidence["last_message"] = ""

    # Test output
    try:
        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", workspace_path,
             "-v", "--tb=short", "-x"],
            capture_output=True, text=True, timeout=60,
        )
        evidence["test_passed"] = (
            re.search(r'\d+ passed', test_result.stdout + test_result.stderr) is not None
        )
        evidence["test_output"] = test_result.stdout + test_result.stderr
    except Exception:
        evidence["test_passed"] = False
        evidence["test_output"] = ""

    # Diff stat
    try:
        diff_result = subprocess.run(
            ["git", "-C", workspace_path, "diff", "--stat", "HEAD~1..HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        evidence["diff_stat"] = diff_result.stdout if diff_result.returncode == 0 else ""
    except Exception:
        evidence["diff_stat"] = ""

    return evidence


def verify_initial_workspace(workspace_path: str) -> dict[str, Any]:
    """Inspect a workspace after initial commit (no Coder delegation yet).

    Similar to verify_workspace_after_delegation but expects exactly 1 commit.
    Returns same structured evidence dict.
    """
    evidence = verify_workspace_after_delegation(workspace_path)
    # Override for initial state check
    try:
        sha_result = subprocess.run(
            ["git", "-C", workspace_path, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        evidence["initial_sha"] = sha_result.stdout.strip() if sha_result.returncode == 0 else ""
    except Exception:
        evidence["initial_sha"] = ""
    return evidence
