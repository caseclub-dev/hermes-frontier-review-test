"""Simple calculator module with edge-case-aware divide function."""


def divide(a, b):
    """Divide a by b.

    Args:
        a: The dividend (any numeric type).
        b: The divisor (any numeric type). Must not be zero.

    Returns:
        float: The result of a divided by b.

    Raises:
        TypeError: If a or b is not a number (int, float, etc.).
        ZeroDivisionError: If b is zero.
    """
    # Validate input types
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        raise TypeError(
            f"divide() requires numeric arguments, "
            f"got {type(a).__name__} and {type(b).__name__}"
        )

    # Guard against division by zero
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")

    return a / b
