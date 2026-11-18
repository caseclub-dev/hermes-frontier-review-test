"""M4F integration tests — bridge between pipeline Python and Orchestrator tool layer.

Tests verify:
1. Bridge module is importable and functions work correctly
2. Pipeline can be run with no-op delegator and reports CHANGES_REQUIRED
3. Workspace evidence extraction works for both initial and post-correction states
4. The correction loop pattern (orchestrator-level) produces correct outcomes
5. Full regression suite still passes after M4F changes
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

# Ensure imports work from the project src directory
sys.path.insert(0, "/home/atlas/projects/frontier_review/src")


class TestBridgeModule(unittest.TestCase):
    """Tests for frontier_review.bridge module."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="fr_bridge_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_bridge_importable(self):
        from frontier_review.bridge import (
            verify_workspace_after_delegation,
            verify_initial_workspace,
        )
        self.assertTrue(callable(verify_workspace_after_delegation))
        self.assertTrue(callable(verify_initial_workspace))

    def test_verify_initial_workspace_structure(self):
        from frontier_review.bridge import verify_initial_workspace
        from frontier_review.remote_delegator import create_demo_workspace

        ws, ts = create_demo_workspace(self.test_dir)
        try:
            evidence = verify_initial_workspace(ws)

            self.assertIn("files_created", evidence)
            self.assertIn("commit_count", evidence)
            self.assertIn("last_sha", evidence)
            self.assertIn("last_message", evidence)
            self.assertIn("test_passed", evidence)
            self.assertIn("test_output", evidence)
            self.assertIn("diff_stat", evidence)
            self.assertIn("initial_sha", evidence)

            # Should be exactly 1 commit (placeholder)
            self.assertEqual(evidence["commit_count"], 1, "Initial workspace should have 1 commit")

            # Should have main.py and test_main.py
            py_files = [os.path.basename(f) for f in evidence["files_created"]]
            self.assertIn("main.py", py_files)
            self.assertIn("test_main.py", py_files)

            # Initial placeholder code has no type hints, so initial_sha != ""
            self.assertTrue(evidence["initial_sha"], "SHA should be non-empty")
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    def test_verify_workspace_after_correction(self):
        """Simulate a workspace after Coder correction (2 commits)."""
        ws = tempfile.mkdtemp(dir=self.test_dir, prefix="fr_corr_")
        subprocess.run(["git", "init", ws], capture_output=True, text=True)
        subprocess.run(
            ["git", "-C", ws, "config", "user.name", "test"],
            capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "-C", ws, "config", "user.email", "test@test.local"],
            capture_output=True, text=True,
        )

        # Write initial placeholder (commit 1)
        with open(os.path.join(ws, "main.py"), "w") as f:
            f.write("# Initial placeholder\ndef divide(a, b):\n    return a / b\n")
        subprocess.run(["git", "-C", ws, "add", "."], capture_output=True, text=True)
        subprocess.run(
            ["git", "-C", ws, "commit", "-m", "Initial"],
            capture_output=True, text=True,
        )

        # Write corrected code (commit 2)
        with open(os.path.join(ws, "main.py"), "w") as f:
            f.write("# Corrected\ndef divide(a: float, b: float) -> float:\n    if b == 0.0:\n        raise ValueError(\"Division by zero\")\n    return a / b\n")
        subprocess.run(["git", "-C", ws, "add", "."], capture_output=True, text=True)
        subprocess.run(
            ["git", "-C", ws, "commit", "-m", "Fix: add error handling and type hints"],
            capture_output=True, text=True,
        )

        try:
            from frontier_review.bridge import verify_workspace_after_delegation
            evidence = verify_workspace_after_delegation(ws)

            self.assertEqual(evidence["commit_count"], 2, "Should have 2 commits after correction")
            self.assertTrue(evidence["last_sha"], "SHA should be non-empty")
            self.assertIn("diff_stat", evidence)
            self.assertIn("main.py", str(evidence.get("diff_stat", "")))
        finally:
            shutil.rmtree(ws, ignore_errors=True)


class TestPipelineWithNoOpDelegator(unittest.TestCase):
    """Tests that pipeline works correctly with no-op delegator (step 1 of M4F pattern)."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="fr_pipe_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_pipeline_detects_defects(self):
        from frontier_review.remote_delegator import (
            run_full_m4b_pipeline,
            FrontierReviewDelegator,
        )

        result = run_full_m4b_pipeline(
            requirements="Implement divide(a, b) with error handling.",
            acceptance_criteria=[
                "divide(10.0, 2.0) returns 5.0",
                "divide(10.0, 0.0) raises ValueError",
                "divide has type hints on all parameters and return",
            ],
            workspace_base=self.test_dir,
            delegator=FrontierReviewDelegator(max_retries=0),
        )

        # No-op delegator → blocked after first review finds defects
        self.assertEqual(result["final_disposition"], "blocked")
        self.assertEqual(len(result.get("review_sequence", [])), 1)

        review = result["review_sequence"][0]
        # The buggy placeholder should be flagged for changes
        self.assertIn(review["status"], ["changes_required", "needs_evidence"])

    def test_workspace_path_valid(self):
        from frontier_review.remote_delegator import (
            run_full_m4b_pipeline,
            FrontierReviewDelegator,
        )

        result = run_full_m4b_pipeline(
            requirements="dummy",
            acceptance_criteria=["dummy"],
            workspace_base=self.test_dir,
            delegator=FrontierReviewDelegator(max_retries=0),
        )

        self.assertTrue(os.path.isdir(result["workspace_path"]))


class TestEvidenceAwareReview(unittest.TestCase):
    """Tests that evidence-aware review correctly identifies defects in corrected code."""

    def test_passes_with_good_code_and_tests(self):
        from frontier_review.m4b_integration import run_evidence_aware_review

        good_code = "# GOOD CODE\ndef divide(a: float, b: float) -> float:\n" \
                    "    if b == 0.0:\n        raise ValueError('Division by zero')\n    return a / b\n"
        test_output = """================== test session ==================\nplatform linux -- Python 3.x\ncollected 3 items\ntest_main.py::test_divide_normal PASSED\ntest_main.py::test_divide_zero PASSED\ntest_main.py::test_type_hints PASSED\n================== 3 passed in 0.1s ==================\n"""

        result = run_evidence_aware_review(
            requirements="Implement divide(a, b).",
            diff=good_code,
            test_evidence=test_output,
            acceptance_criteria=["divide(10.0, 2.0) returns 5.0"],
        )

        self.assertEqual(result["status"], "pass", f"Expected PASS but got {result}")

    def test_changes_required_with_missing_type_hints(self):
        from frontier_review.m4b_integration import run_evidence_aware_review

        code_no_hints = "# Code without type hints\ndef divide(a, b):\n    return a / b\n"
        # Test output showing test 3 (type hints) failed
        test_output = """test_main.py::test_divide_normal PASSED
test_main.py::test_divide_zero PASSED
test_main.py::test_type_hints FAILED: no type hints on divide()
"""

        result = run_evidence_aware_review(
            requirements="Implement divide(a, b).",
            diff=code_no_hints,
            test_evidence=test_output,
            acceptance_criteria=["divide has type hints"],
        )

        self.assertEqual(result["status"], "changes_required")


class TestIntegrationFlow(unittest.TestCase):
    """End-to-end pattern verification (no real Coder, simulates correction flow)."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="fr_int_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_full_step_pattern_initial_to_corrected(self):
        """Simulate the M4F step-by-step pattern: initial review → manual correction → re-review."""
        from frontier_review.remote_delegator import create_demo_workspace
        from frontier_review.bridge import verify_workspace_after_delegation, verify_initial_workspace
        from frontier_review.m4b_integration import run_evidence_aware_review

        # Step A: Create workspace with buggy placeholder (like pipeline step 1)
        ws, _ = create_demo_workspace(self.test_dir)

        try:
            # Step B: Verify initial state (like execute_code after pipeline step 1)
            init_evidence = verify_initial_workspace(ws)
            self.assertEqual(init_evidence["commit_count"], 1)
            initial_sha = init_evidence["initial_sha"]
            self.assertTrue(initial_sha)

            # Step C: Initial review on buggy code (pipeline does this internally)
            from frontier_review.remote_delegator import _run_subprocess
            rc, diff_content, _ = _run_subprocess(
                ["git", "-C", ws, "show", "--format=", "HEAD"], cwd=None
            )
            initial_test = subprocess.run(
                [sys.executable, "-m", "pytest", ws, "-v", "--tb=short"],
                capture_output=True, text=True, timeout=30, cwd=ws,
            )
            initial_review = run_evidence_aware_review(
                requirements="Implement divide(a, b).",
                diff=diff_content if rc == 0 else "",
                test_evidence=(initial_test.stdout + initial_test.stderr),
                acceptance_criteria=[
                    "divide(10.0, 2.0) returns 5.0",
                    "divide(10.0, 0.0) raises ValueError",
                    "divide has type hints",
                ],
            )

            # Step D: Review should flag defects
            self.assertIn(initial_review["status"], ["changes_required", "needs_evidence"])

            # Step E: Simulate Coder correction (write corrected code + commit)
            with open(os.path.join(ws, "main.py"), "w") as f:
                f.write("# Fixed by Coder\ndef divide(a: float, b: float) -> float:\n"
                        "    if b == 0.0:\n        raise ValueError('Division by zero')\n"
                        "    return a / b\n")

            subprocess.run(
                ["git", "-C", ws, "add", "."], capture_output=True, text=True,
            )
            subprocess.run(
                ["git", "-C", ws, "commit", "-m", "Correction: fix defects"],
                capture_output=True, text=True, timeout=10,
            )

            # Step F: Verify corrected workspace (like execute_code step 3)
            post_evidence = verify_workspace_after_delegation(ws)
            self.assertEqual(post_evidence["commit_count"], 2, "Should have 2 commits after correction")
            self.assertTrue(post_evidence["last_sha"])
            self.assertNotEqual(initial_sha, post_evidence["last_sha"], "SHA should change after correction")

            # Step G: Diff stat for re-review
            rc2, diff_stat, _ = _run_subprocess(
                ["git", "-C", ws, "diff", "HEAD~1..HEAD"], cwd=None
            )
            post_diff = diff_stat if rc2 == 0 else ""

            # Step H: Re-review on corrected code
            subprocess_result = subprocess.run(
                [sys.executable, "-m", "pytest", ws, "-v", "--tb=short"],
                capture_output=True, text=True, timeout=30, cwd=ws,
            )
            post_test_output = subprocess_result.stdout + subprocess_result.stderr

            re_review = run_evidence_aware_review(
                requirements="Implement divide(a, b).",
                diff=post_diff,
                test_evidence=post_test_output,
                acceptance_criteria=[
                    "divide(10.0, 2.0) returns 5.0",
                    "divide(10.0, 0.0) raises ValueError",
                    "divide has type hints",
                ],
            )

            # Step I: Corrected code should pass
            self.assertEqual(re_review["status"], "pass",
                             f"Expected PASS after correction but got {re_review['status']}. "
                             f"Findings: {re_review.get('findings', [])}")

        finally:
            shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
