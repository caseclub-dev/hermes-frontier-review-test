"""Tests for calculator functions."""
from calc_buggy import multiply, divide, power

def test_multiply():
    assert multiply(3, 4) == 12

def test_divide():
    assert divide(10, 2) == 5.0

def test_power():
    assert power(2, 3) == 8

if __name__ == '__main__':
    test_multiply()
    test_divide()
    test_power()
    print("All tests pass")
