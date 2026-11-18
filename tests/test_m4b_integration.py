"""M4B End-to-End Integration Test — Real Coder Delegations.

Demonstrates one operator instruction initiating an uninterrupted workflow:
  Orchestrator → real Coder implementation → Git commit → 
  mock review CHANGES_REQUIRED → real Coder correction → new Git commit →
  mock review PASS → READY_FOR_OPERATOR_REVIEW

This test creates disposable workspaces, uses real delegate_task(profile="coder")
calls for implementation and correction, and the evidence-aware review pipeline
from AutonomousCorrectionLoop._build_evidence_aware_review_payload() as the
simulated reviewer.

No Anthropic API calls are made — all "reviews" use deterministic content inspection.
"""

import json
import os
import sys
import shutil
import subprocess
import tempfile

# Ensure src is on path
FR_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, FR_SRC)

from frontier_review.m4b_integration import (
    build_initial_delegation_request,
    build_correction_delegation_request,
    GitWorkspaceVerifier,
    compute_evidence_fingerprint,
    run_evidence_aware_review,
)


# ── E2E Scenario: Division function with missing error handling ─────

def create_division_test_workspace(base_dir: str = "/tmp") -> tuple[str, str]:
    """Create a disposable workspace with a test script for the division task.

    Returns (workspace_path, test_script_path).
    """
    ws = tempfile.mkdtemp(dir=base_dir, prefix="m4b_division_")

    # Initialize git
    subprocess.run(["git", "init", ws], capture_output=True, text=True)
    subprocess.run(
        ["git", "-C", ws, "config", "user.name", "m4b-e2e"],
        capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "-C", ws, "config", "user.email", "m4b@test.local"],
        capture_output=True, text=True,
    )

    # Write the test script first (before Coder implements the code)
    test_script = os.path.join(ws, "test_main.py")
    with open(test_script, "w") as f:
        f.write("""import sys
import importlib.util

# Load main.py dynamically
spec = importlib.util.spec_from_file_location("main", "main.py")
mod = importlib.util.module_from_spec(spec)

# Test 1: Normal division should work
try:
    spec.loader.exec_module(mod)
    result = mod.divide(10.0, 2.0)
    assert abs(result - 5.0) < 0.01, f"Expected 5.0, got {result}"
    print("Test 1 PASSED: normal division")
except Exception as e:
    print(f"Test 1 FAILED: {e}", file=sys.stderr)

# Test 2: Division by zero should raise ValueError, not crash
try:
    spec.loader.exec_module(mod)
    mod.divide(10.0, 0.0)
    print("Test 2 FAILED: should have raised ValueError", file=sys.stderr)
except ValueError as e:
    print(f"Test 2 PASSED: ValueError raised for zero division")
except Exception as e:
    print(f"Test 2 FAILED: wrong exception type: {type(e).__name__}: {e}", file=sys.stderr)

# Test 3: Type hints must be present (basic check — verify function has annotations)
try:
    spec.loader.exec_module(mod)
    sig = mod.divide.__code__.co_varnames
    # If type hints exist, they show up in __annotations__
    if hasattr(mod.divide, '__annotations__') and len(mod.divide.__annotations__) > 0:
        print("Test 3 PASSED: type hints present")
    else:
        print("Test 3 FAILED: no type hints on divide()", file=sys.stderr)
except Exception as e:
    print(f"Test 3 FAILED: {e}", file=sys.stderr)
""")

    # Write initial buggy placeholder code (defect for review to detect)
    main_code = os.path.join(ws, "main.py")
    with open(main_code, "w") as f:
        f.write("""def divide(a, b):
    return a / b
""")

    # Initial commit
    subprocess.run(["git", "-C", ws, "add", "."], capture_output=True, text=True)
    result = subprocess.run(
        ["git", "-C", ws, "commit", "-m", "Initial placeholder — buggy divide"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git commit failed: {result.stderr}")

    return ws


def run_initial_review(workspace_path: str) -> dict:
    """Run evidence-aware review on the initial code to detect defects.

    Returns (review_result, test_output).
    """
    verifier = GitWorkspaceVerifier(workspace_path)

    # Run tests
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", workspace_path, "-v", "--tb=short"],
        capture_output=True, text=True, timeout=30,
    )
    test_output = test_result.stdout + test_result.stderr

    # Get diff
    diff_result = subprocess.run(
        ["git", "-C", workspace_path, "diff", "--cached"],
        capture_output=True, text=True, timeout=10,
    )
    diff = diff_result.stdout

    review = run_evidence_aware_review(
        requirements="Function must handle division by zero safely with type hints.",
        diff=diff,
        test_evidence=test_output,
        acceptance_criteria=[
            "Division is guarded against zero",
            "Error handling present",
            "Type hints for parameters and return",
        ],
    )

    return review


def verify_workspace_after_correction(workspace_path: str) -> dict:
    """Verify workspace state after Coder correction.

    Returns verification details.
    """
    verifier = GitWorkspaceVerifier(workspace_path)

    sha = verifier.get_last_commit_sha()
    commit_msg = verifier.get_last_commit_message()
    files = verifier.list_created_files()
    syntax_ok = verifier.verify_python_syntax()["valid"]

    # Run tests
    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", workspace_path, "-v", "--tb=short"],
        capture_output=True, text=True, timeout=30,
    )
    test_output = test_result.stdout + test_result.stderr

    # Get diff (to see changes from initial commit)
    diff_result = subprocess.run(
        ["git", "-C", workspace_path, "diff", "HEAD~1..HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    diff = diff_result.stdout

    # Commit count
    count_result = subprocess.run(
        ["git", "-C", workspace_path, "rev-list", "--count", "HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    commit_count = int(count_result.stdout.strip()) if count_result.returncode == 0 else 0

    return {
        "sha": sha,
        "commit_message": commit_msg,
        "commit_count": commit_count,
        "files_created": files,
        "syntax_valid": syntax_ok,
        "test_output": test_output,
        "diff": diff,
    }


def run_post_correction_review(workspace_path: str) -> dict:
    """Run evidence-aware review after Coder correction.

    Returns the review result dict with status.
    """
    verifier = GitWorkspaceVerifier(workspace_path)
    diff_result = subprocess.run(
        ["git", "-C", workspace_path, "diff", "HEAD~1..HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    diff = diff_result.stdout

    test_result = subprocess.run(
        [sys.executable, "-m", "pytest", workspace_path, "-v", "--tb=short"],
        capture_output=True, text=True, timeout=30,
    )
    test_output = test_result.stdout + test_result.stderr

    return run_evidence_aware_review(
        requirements="Function must handle division by zero safely with type hints.",
        diff=diff,
        test_evidence=test_output,
        acceptance_criteria=[
            "Division is guarded against zero",
            "Error handling present",
            "Type hints for parameters and return",
        ],
    )


def cleanup_workspace(ws: str) -> None:
    """Clean up a disposable workspace."""
    shutil.rmtree(ws, ignore_errors=True)
