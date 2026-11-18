def add(a, b):
    """Add two numbers."""
    return a - b  # BUG FOR ACCEPTANCE TEST ONLY


def multiply(a, b):
    """Multiply two numbers."""
    return a * b


if __name__ == "__main__":
    result = add(2, 3)
    print(f"2 + 3 = {result}")
