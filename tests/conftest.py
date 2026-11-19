"""conftest.py — shared fixtures for Frontier Review tests."""
import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _setup_test_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Set up test directories and patch publisher module variables.

    This runs before every test function in any test file that imports from this conftest.
    It ensures FR_AUTH_DIR and FR_STATE_DIR point to writable temp dirs and patches
    the publisher module's PUB_AUTH_DIR / PUB_STATE_DIR after import.
    """
    # Create directories
    auth_dir = str(tmp_path / "auth")
    state_dir = str(tmp_path / "state")
    tx_log_dir = str(tmp_path / "tx_log")
    tx_lock_dir = str(tmp_path / "locks")
    os.makedirs(auth_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(tx_log_dir, exist_ok=True)
    os.makedirs(tx_lock_dir, exist_ok=True)

    # Set environment variables
    monkeypatch.setenv("FR_AUTH_DIR", auth_dir)
    monkeypatch.setenv("FR_STATE_DIR", state_dir)
    monkeypatch.setenv("PUB_AUTH_DIR", auth_dir)
    monkeypatch.setenv("PUB_STATE_DIR", state_dir)
    monkeypatch.setenv("FR_TX_LOG_DIR", tx_log_dir)
    monkeypatch.setenv("FR_TX_LOCK_DIR", tx_lock_dir)

    # Patch publisher module variables (set at import time from env, so we patch directly)
    try:
        import frontier_review.publisher as pub_mod

        pub_mod.PUB_AUTH_DIR = auth_dir
        pub_mod.PUB_STATE_DIR = state_dir
    except ImportError:
        pass

    # Enable LEGACY_PUBLISH_ENABLED for all tests that exercise the guard path.
    # Individual tests can override by setting pa_mod.LEGACY_PUBLISH_ENABLED = False.
    try:
        import frontier_review.pr_adapter as pa_mod

        pa_mod.LEGACY_PUBLISH_ENABLED = True
        monkeypatch.setattr(pa_mod, "LEGACY_PUBLISH_ENABLED", True)
        # Mock verify_pr_state to return success for tests that call push_correction_to_github
        monkeypatch.setattr(
            pa_mod,
            "verify_pr_state",
            lambda *a, **k: (True, "PR verification passed (test mock)"),
        )
    except ImportError:
        pass
