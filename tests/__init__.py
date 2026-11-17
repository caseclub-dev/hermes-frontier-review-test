"""Test fixtures for Frontier Review.

Provides shared setup/teardown for test isolation (temporary ledger files,
state store paths, etc.).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Ensure src is discoverable when running tests from any directory
SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
