"""A simple calculator module with deliberate bugs for acceptance testing."""

def multiply(a: int, b: int) -> int:
    """Return a * b."""
    return a * b

def divide(a: int, b: int) -> float:
    """Return a / b."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

def power(a: int, b: int) -> int:
    """Return a ^ b (a raised to the power of b)."""
    return a ** b

# Fixed tests with correct expected values
if __name__ == "__main__":
    assert multiply(3, 4) == 12
    assert divide(10, 3) == 10 / 3
    assert power(2, 10) == 1024
    print("All tests pass")
