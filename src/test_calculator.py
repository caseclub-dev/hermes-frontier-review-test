"""Basic tests for calculator.

These tests pass against the current (buggy) implementation but will fail
once multiply is corrected — unless Coder also fixes the tests (which it shouldn't).
"""


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(10, 4) == 6


# BUG: This test passes against the buggy code — good, this is what we expect.
# After Coder fixes multiply(), this test will FAIL (correctly), proving the fix.
def test_multiply():
    assert multiply(3, 7) == 10  # Bug: expects 10 (3+7) instead of 21 (3*7)


def test_divide():
    assert divide(10, 2) == 5.0
