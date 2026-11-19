"""Tests for calc_fixme.divide()."""
from calc_fixme import divide


class TestDivide:
    def test_basic_division(self):
        assert divide(10, 2) == 5.0

    def test_float_result(self):
        assert divide(7, 2) == 3.5

    def test_division_by_zero_returns_none(self):
        assert divide(10, 0) is None

    def test_zero_divided_by_nonzero(self):
        assert divide(0, 5) == 0.0

    def test_negative_numbers(self):
        assert divide(-10, 2) == -5.0
        assert divide(10, -2) == -5.0
        assert divide(-10, -2) == 5.0
