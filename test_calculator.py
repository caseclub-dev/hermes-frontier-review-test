"""Tests for calculator module."""
from calculator import add, subtract, multiply


def test_add():
    """Test add function - should fail with the buggy implementation."""
    assert add(2, 3) == 5


def test_subtract():
    """Test subtract function - should pass."""
    assert subtract(5, 3) == 2


def test_multiply():
    """Test multiply function - should pass."""
    assert multiply(4, 5) == 20


if __name__ == "__main__":
    test_add()
    test_subtract()
    test_multiply()
    print("All tests passed")
