"""Budget ledger management for Frontier Review.

Provides token cost estimation, budget checks, and a JSON-based ledger
for tracking spending. Ledger path is configurable for test isolation.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Optional


# ── Ledger path configuration ────────────────────────────────────────

_DEFAULT_LEDGER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".budget_ledger.json"
)
_LEDGER_PATH: str = _DEFAULT_LEDGER_PATH

# Headroom reserved per review to ensure we don't exceed budget
RESERVATION_HEADROOM = 0.50  # USD


# ── Test isolation helpers ───────────────────────────────────────────

_ledger_override_path: Optional[str] = None


def set_ledger_override(path: str) -> None:
    """Redirect ledger I/O to a temporary path (for test isolation)."""
    global _LEDGER_PATH  # noqa: PLW0603
    _LEDGER_PATH = path


def reset_ledger_override() -> None:
    """Restore production ledger path."""
    global _LEDGER_PATH  # noqa: PLW0603
    _LEDGER_PATH = _DEFAULT_LEDGER_PATH


def isolate_test_ledger() -> str:
    """Create a temporary ledger file and override the default path.

    Returns the path to the temporary ledger for cleanup by the caller.
    """
    led_path = tempfile.NamedTemporaryFile(
        suffix="_budget_ledger.json", delete=False, prefix="fr_"
    )
    led_path.close()
    # Write an empty ledger
    with open(led_path.name, "w") as f:
        json.dump({"monthly_spending_usd": 0.0, "requests": [], "pending_reservations": []}, f)
    _LEDGER_PATH = led_path.name
    return led_path.name


# ── Cost estimation ─────────────────────────────────────────────────

def estimate_request_cost(input_tokens: int, max_output: int, model_name: str) -> float:
    """Estimate the cost of an API request based on token counts.

    Pricing (per 1M tokens):
      - Input: $15.00 for claude-opus-5-5
      - Output: $75.00 for claude-opus-5-5
      - Cache read: $3.75 per 1M tokens

    Non-Opus models return 0.0 cost (they will be rejected by validation elsewhere).
    """
    from .config import get_model, validate_model

    if not validate_model(model_name):
        # If the model doesn't match, estimate zero — it'll be blocked anyway
        return 0.0

    input_cost = input_tokens / 1_000_000 * 15.0
    output_cost = max_output / 1_000_000 * 75.0
    return round(input_cost + output_cost, 8)


def check_budget(requested_cost: float, monthly_budget: float) -> bool:
    """Check if a requested cost fits within the monthly budget.

    Args:
        requested_cost: Total estimated cost including headroom.
        monthly_budget: Monthly spending cap in USD.

    Returns True if the cost is within budget (including any already spent).
    """
    try:
        with open(_LEDGER_PATH) as f:
            ledger = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        # Ledger doesn't exist or is corrupted — treat as if no spending
        return True

    total_spent = ledger.get("monthly_spending_usd", 0.0)
    return (total_spent + requested_cost + RESERVATION_HEADROOM) <= monthly_budget


def record_request(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> Optional[str]:
    """Record a request in the budget ledger.

    Returns the generated request ID, or None if the model is not allowed.
    """
    from .config import validate_model

    if not validate_model(model_name):
        return None

    input_cost = (input_tokens + cache_creation_tokens) / 1_000_000 * 15.0
    output_cost = output_tokens / 1_000_000 * 75.0
    cache_read_cost = cache_read_tokens / 1_000_000 * 3.75
    cost = round(input_cost + output_cost + cache_read_cost, 6)

    request_id = f"req-{os.urandom(4).hex()}"

    try:
        with open(_LEDGER_PATH) as f:
            ledger = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        ledger = {"monthly_spending_usd": 0.0, "requests": [], "pending_reservations": []}

    ledger["requests"].append({
        "id": request_id,
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cost_usd": cost,
    })
    ledger["monthly_spending_usd"] = round(ledger["monthly_spending_usd"] + cost, 6)

    with open(_LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)

    return request_id


def release_reservation(reservation_id: str) -> None:
    """Release a pending budget reservation."""
    try:
        with open(_LEDGER_PATH) as f:
            ledger = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return

    ledger["pending_reservations"] = [
        r for r in ledger.get("pending_reservations", []) if r["id"] != reservation_id
    ]

    with open(_LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)


def reset_ledger() -> None:
    """Reset the ledger to empty state (useful for tests)."""
    reset_ledger_override()
    try:
        with open(_DEFAULT_LEDGER_PATH, "w") as f:
            json.dump({"monthly_spending_usd": 0.0, "requests": [], "pending_reservations": []}, f)
    except FileNotFoundError:
        pass
