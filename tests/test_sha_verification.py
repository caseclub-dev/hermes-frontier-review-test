#!/usr/bin/env python3
"""Phase 4A Remote SHA Verification — Regression Tests.

Tests for verify_remote_sha() bounded retries and push_correction_to_github()
remote SHA verification behavior, including transient API staleness scenarios.

All tests are self-contained and order-independent.
"""

import json
import os
import subprocess
import tempfile
import time
import unittest
from contextlib import AbstractContextManager
from unittest.mock import MagicMock, patch

import pytest

from frontier_review.pr_adapter import (
    verify_remote_sha,
    push_correction_to_github,
)


# ━━━━━━━━━━━━━━━━━━━  FIXTURES ────────────────────────────────────────


class _TestSHA:
    """Shared SHA constants for tests."""
    CORRECT = "aaaa" * 10
    STALE = "bbbb" * 10


def _make_ctx_mgr(json_data):
    """Build a mock context manager that yields an object with .read() returning JSON."""
    response_obj = MagicMock()
    response_obj.read.return_value = json.dumps(json_data).encode()

    ctx = MagicMock(spec=AbstractContextManager)
    ctx.__enter__ = MagicMock(return_value=response_obj)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _git_mock_success(*args, **kwargs):
    """Return a successful subprocess.CompletedProcess (used for mocking git calls)."""
    proc = subprocess.CompletedProcess(args=args[0] if isinstance(args, tuple) else args,
                                       returncode=0, stdout=b"ok", stderr=b"")
    return proc


# ━━━━━━━━━━━━━━━━━━━  verify_remote_sha UNIT TESTS ────────────────────


class TestVerifyRemoteShaDirect(unittest.TestCase):

    @patch("frontier_review.pr_adapter._urlopen")
    def test_immediate_match_returns_success(self, mock_urlopen):
        """SHA matches on first call → (True, sha)."""
        mock_urlopen.return_value = _make_ctx_mgr({"object": {"sha": _TestSHA.CORRECT}})

        success, sha = verify_remote_sha(
            repo_owner="caseclub-dev", repo_name="test-repo",
            head_branch="feature/x", expected_sha=_TestSHA.CORRECT,
            token="fake", max_retries=1,
        )

        self.assertTrue(success)
        self.assertEqual(sha, _TestSHA.CORRECT)
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("frontier_review.pr_adapter._urlopen")
    def test_transient_staleness_then_match(self, mock_urlopen):
        """Stale SHA twice, then correct → retry succeeds."""
        responses = [
            _make_ctx_mgr({"object": {"sha": _TestSHA.STALE}}),
            _make_ctx_mgr({"object": {"sha": _TestSHA.STALE}}),
            _make_ctx_mgr({"object": {"sha": _TestSHA.CORRECT}}),
        ]
        idx = [0]

        def side_effect(*args, **kwargs):
            resp = responses[idx[0]]
            idx[0] += 1
            return resp

        mock_urlopen.side_effect = side_effect

        success, sha = verify_remote_sha(
            repo_owner="caseclub-dev", repo_name="test-repo",
            head_branch="feature/x", expected_sha=_TestSHA.CORRECT,
            token="fake", max_retries=3, retry_interval=0.01,
        )

        self.assertTrue(success)
        self.assertEqual(sha, _TestSHA.CORRECT)
        self.assertEqual(mock_urlopen.call_count, 3)

    @patch("frontier_review.pr_adapter._urlopen")
    def test_persistent_staleness_fails_after_retries(self, mock_urlopen):
        """Wrong SHA always → fails after max_retries."""
        stale = _make_ctx_mgr({"object": {"sha": _TestSHA.STALE}})
        mock_urlopen.return_value = stale

        success, msg = verify_remote_sha(
            repo_owner="caseclub-dev", repo_name="test-repo",
            head_branch="feature/x", expected_sha=_TestSHA.CORRECT,
            token="fake", max_retries=2, retry_interval=0.01,
        )

        self.assertFalse(success)
        self.assertIn("mismatch", msg.lower())
        self.assertIn("2 retries", msg.lower())
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("frontier_review.pr_adapter._urlopen")
    def test_transient_api_error_then_match(self, mock_urlopen):
        """API error on first call, then match → retry succeeds."""
        match_ctx = _make_ctx_mgr({"object": {"sha": _TestSHA.CORRECT}})
        errors = [ConnectionError("GitHub API unavailable"), match_ctx]

        def side_effect(*args, **kwargs):
            item = errors.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        mock_urlopen.side_effect = side_effect

        success, sha = verify_remote_sha(
            repo_owner="caseclub-dev", repo_name="test-repo",
            head_branch="feature/x", expected_sha=_TestSHA.CORRECT,
            token="fake", max_retries=3, retry_interval=0.01,
        )

        self.assertTrue(success)
        self.assertEqual(sha, _TestSHA.CORRECT)
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("frontier_review.pr_adapter._urlopen")
    def test_all_retries_fail_on_api_error(self, mock_urlopen):
        """API errors always → fails with clear message."""
        mock_urlopen.side_effect = ConnectionError("GitHub API unavailable")

        success, msg = verify_remote_sha(
            repo_owner="caseclub-dev", repo_name="test-repo",
            head_branch="feature/x", expected_sha=_TestSHA.CORRECT,
            token="fake", max_retries=2, retry_interval=0.01,
        )

        self.assertFalse(success)
        self.assertIn("Failed to fetch remote SHA", msg)


# ━━━━━━━━━━━━━━━━━━━  push_correction INTEGRATION TESTS ────────────────


class TestPushCorrectionSHAVerification(unittest.TestCase):
    """Test push_correction_to_github() with mocked subprocess calls.

    By mocking all git commands we isolate testing of verify_remote_sha()
    which is the focus of Subphase 4A-3. The actual git operations are
    real and verified by other existing tests (test_phase4a_branch_init).
    """

    @patch("subprocess.run", side_effect=_git_mock_success)
    @patch("frontier_review.pr_adapter.verify_remote_sha")
    def test_push_succeeds_with_immediate_sha_match(self, mock_verify, _mock_git):
        """SHA match → push_correction returns success."""
        mock_verify.return_value = (True, "cccc" * 10)

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            os.makedirs(ws)
            # Minimal git workspace so workspace validation passes
            subprocess.run(["git", "init"], cwd=ws, capture_output=True, check=True)
            subprocess.run(
                ["git", "-C", ws, "config", "user.email", "test@test.com"],
                capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "-C", ws, "config", "user.name", "Test User"],
                capture_output=True, check=True,
            )
            # Create a feature/x branch with a commit (push_correction creates/checkout)
            with open(os.path.join(ws, "main.py"), "w") as f:
                f.write("# initial\n")
            subprocess.run(["git", "-C", ws, "add", "."], capture_output=True, check=True)
            subprocess.run(
                ["git", "-C", ws, "commit", "-m", "initial"], capture_output=True, check=True
            )
            # Ensure feature/x branch exists in local repo (for merge-base to work)
            subprocess.run(
                ["git", "-C", ws, "branch", "feature/x", "HEAD"], capture_output=True, check=True
            )

            success, msg = push_correction_to_github(
                repo_owner="caseclub-dev",
                repo_name="test-repo",
                head_branch="feature/x",
                token="fake",
                workspace_path=ws,
            )

            self.assertTrue(success)
            mock_verify.assert_called_once()

    @patch("subprocess.run", side_effect=_git_mock_success)
    @patch("frontier_review.pr_adapter.verify_remote_sha")
    def test_stale_api_sha_returns_error(self, mock_verify, _mock_git):
        """Stale SHA → push_correction returns error."""
        mock_verify.return_value = (
            False,
            "Remote SHA mismatch after 3 retries: local=aaaaaa remote=bbbbbb",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.join(tmpdir, "ws")
            os.makedirs(ws)
            subprocess.run(["git", "init"], cwd=ws, capture_output=True, check=True)
            subprocess.run(
                ["git", "-C", ws, "config", "user.email", "test@test.com"],
                capture_output=True, check=True,
            )
            subprocess.run(
                ["git", "-C", ws, "config", "user.name", "Test User"],
                capture_output=True, check=True,
            )
            with open(os.path.join(ws, "main.py"), "w") as f:
                f.write("# initial\n")
            subprocess.run(["git", "-C", ws, "add", "."], capture_output=True, check=True)
            subprocess.run(
                ["git", "-C", ws, "commit", "-m", "initial"], capture_output=True, check=True
            )
            subprocess.run(
                ["git", "-C", ws, "branch", "feature/x", "HEAD"], capture_output=True, check=True
            )

            success, msg = push_correction_to_github(
                repo_owner="caseclub-dev",
                repo_name="test-repo",
                head_branch="feature/x",
                token="fake",
                workspace_path=ws,
            )

            self.assertFalse(success)
            self.assertIn("Remote SHA verification failed", msg)

    def test_empty_workspace_rejected(self):
        """workspace_path='' → (False, 'No valid workspace')."""
        success, msg = push_correction_to_github(
            repo_owner="caseclub-dev",
            repo_name="test-repo",
            head_branch="feature/x",
            token="fake",
            workspace_path="",
        )

        self.assertFalse(success)
        self.assertIn("No valid workspace", msg)


# ━━━━━━━━━━━━━━━━━━━  VERIFY_RETRIES SAFETY TESTS ─────────────────────


class TestVerifyRetrySafety(unittest.TestCase):

    @patch("frontier_review.pr_adapter._urlopen")
    def test_empty_sha_not_found_in_response(self, mock_urlopen):
        """Response without 'object.sha' → SHA is empty → mismatch."""
        mock_urlopen.return_value = _make_ctx_mgr({"ref": "refs/heads/main"})

        success, _ = verify_remote_sha(
            repo_owner="o", repo_name="r", head_branch="main",
            expected_sha=_TestSHA.CORRECT, token="t", max_retries=1,
        )
        self.assertFalse(success)

    @patch("frontier_review.pr_adapter._urlopen")
    def test_retry_delays_are_bounded(self, mock_urlopen):
        """Total time bounded: 3 retries × 0.1s = ~0.2-0.3s."""
        stale = _make_ctx_mgr({"object": {"sha": _TestSHA.STALE}})
        mock_urlopen.return_value = stale

        start = time.time()
        verify_remote_sha(
            repo_owner="o", repo_name="r", head_branch="h",
            expected_sha=_TestSHA.CORRECT, token="t",
            max_retries=3, retry_interval=0.1,
        )
        elapsed = time.time() - start

        self.assertGreaterEqual(elapsed, 0.2)
        self.assertLess(elapsed, 5.0)


# ━━━━━━━━━━━━━━━━━━━  TEST ENTRY POINT ────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
