"""Calculator module — intentionally has a bug for acceptance testing.

BUG: calculate_multiply uses addition instead of multiplication.
"""


def add(a, b):
    """Return sum of two numbers."""
    return a + b


def subtract(a, b):
    """Return difference of two numbers."""
    return a - b


def multiply(a, b):
    """BUG: Returns sum instead of product.

    This is the bug that Coder must fix during the acceptance test.
    Expected behavior: multiply(a, b) should return a * b.
    Actual (buggy) behavior: returns a + b.
    """
    return a + b


def divide(a, b):
    """Return quotient. Raises ValueError on division by zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
