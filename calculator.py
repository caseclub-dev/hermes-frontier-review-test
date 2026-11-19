"""Simple calculator module for testing Frontier Review integration."""


def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a - b  # BUG FOR ACCEPTANCE TEST ONLY


def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


if __name__ == "__main__":
    print(f"add(2, 3) = {add(2, 3)}")
    print(f"subtract(5, 3) = {subtract(5, 3)}")
    print(f"multiply(4, 5) = {multiply(4, 5)}")
