"""Tests for calculator module."""
import sys
sys.path.insert(0, '/tmp/frontier_test_workspace')
from calculator import add, subtract, multiply


def test_add():
    """Test add function.
    
    BUG: Expected value matches the buggy implementation (2-3=-1)
    rather than correct behavior (2+3=5). This test passes despite
    the bug in the code, demonstrating that tests alone don't verify
    correctness — Frontier Review must inspect the logic.
    """
    assert add(2, 3) == -1  # Wrong! Should be 5 but matches buggy implementation


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
