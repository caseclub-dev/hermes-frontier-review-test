"""Deliberately broken calculator for acceptance testing."""

def multiply(a: int, b: int) -> int:
    """Return a * b."""
    return a + b  # BUG: uses + instead of *

def divide(a: int, b: int) -> float:
    """Return a / b."""
    return a - b  # BUG: uses - instead of /

def power(a: int, b: int) -> int:
    """Return a ** b."""
    return a * b  # BUG: uses * instead of **
