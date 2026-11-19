"""A simple calculator module with deliberate bugs for acceptance testing."""

def multiply(a: int, b: int) -> int:
    """Return a * b."""
    return a - b  # BUG: subtraction instead of multiplication

def divide(a: int, b: int) -> float:
    """Return a / b."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a // b  # BUG: integer division instead of true division

def power(a: int, b: int) -> int:
    """Return a ^ b (a raised to the power of b)."""
    return a * b  # BUG: multiplication instead of exponentiation

# Add tests demonstrating the bugs
if __name__ == "__main__":
    assert multiply(3, 4) == 7  # Should be 12 — this test PASSES with the bug!
    assert divide(10, 3) == 3   # Should be 3.333... — this test PASSES with the bug!
    print("All tests pass (but results are wrong)")
