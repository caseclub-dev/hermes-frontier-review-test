"""Tests for calculator module."""
import sys
sys.path.insert(0, '/tmp/frontier_test_workspace')
from calculator import add, subtract, multiply


def test_add():
    """Test add function."""
    assert add(2, 3) == 5


def test_subtract():
    """Test subtract function - correct."""
    assert subtract(5, 3) == 2


def test_multiply():
    """Test multiply function - correct."""
    assert multiply(4, 5) == 20


if __name__ == "__main__":
    test_add()
    test_subtract()
    test_multiply()
    print("All tests passed")
