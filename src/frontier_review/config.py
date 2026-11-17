"""Configuration constants for Frontier Review.

All tunables live here so test overrides are a single variable assignment.
"""

from __future__ import annotations

# ── Model configuration ──────────────────────────────────────────────

ALLOWED_MODEL = "claude-opus-5-5"


def get_model() -> str:
    """Return the allowed model name."""
    return ALLOWED_MODEL


def validate_model(model_name: str) -> bool:
    """Check if a model name matches the allowed Opus model.

    Only the exact allowed Opus model is permitted for code review.
    Returns False for all other models (including sonnet, haiku, etc.).
    """
    return model_name == ALLOWED_MODEL


# ── Token limits ─────────────────────────────────────────────────────

_MAX_OUTPUT_TOKENS = 32768


def get_max_tokens() -> int:
    """Return the maximum output tokens for review responses."""
    return _MAX_OUTPUT_TOKENS


# ── Budget configuration ────────────────────────────────────────────

_BUDGET_MONTHLY_USD = 75.0

# Public alias for use by tests and other modules
BUDGET_USD = _BUDGET_MONTHLY_USD


def get_budget_monthly_usd() -> float:
    """Return the monthly budget cap in USD."""
    return _BUDGET_MONTHLY_USD


# ── Offline / transport gating ──────────────────────────────────────

_OFFLINE_MODE = True  # Default to fail-closed; no real API access


def set_offline_mode(value: bool) -> None:
    """Enable or disable offline (fail-closed) mode."""
    global _OFFLINE_MODE
    _OFFLINE_MODE = value


def is_offline() -> bool:
    """Return True when offline mode is active (default)."""
    return _OFFLINE_MODE
