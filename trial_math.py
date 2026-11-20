def divide(a: float, b: float) -> float:
    """Divide two numbers; zero must raise a clear ValueError."""
    if b == 0:
        raise ValueError("denominator must not be zero")
    return a / b
