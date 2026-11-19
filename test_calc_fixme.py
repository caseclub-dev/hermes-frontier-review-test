"""Tests for calc_fixme.divide() edge cases."""

import unittest
from calc_fixme import divide


class TestDivide(unittest.TestCase):
    """Test suite for the divide function edge cases."""

    # --- Happy-path cases ---

    def test_divide_positive_integers(self):
        self.assertEqual(divide(10, 2), 5.0)

    def test_divide_negative_result(self):
        self.assertEqual(divide(-10, 2), -5.0)

    def test_divide_both_negative(self):
        self.assertEqual(divide(-10, -2), 5.0)

    def test_divide_float(self):
        self.assertAlmostEqual(divide(7.0, 2.0), 3.5)

    def test_divide_by_one(self):
        self.assertEqual(divide(42, 1), 42.0)

    def test_divide_zero_by_nonzero(self):
        self.assertEqual(divide(0, 7), 0.0)

    # --- Edge cases ---

    def test_divide_by_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            divide(10, 0)

    def test_divide_float_zero_raises(self):
        with self.assertRaises(ZeroDivisionError):
            divide(10, 0.0)

    def test_divide_type_error_string_arg(self):
        with self.assertRaises(TypeError):
            divide("10", 2)

    def test_divide_type_error_none_arg(self):
        with self.assertRaises(TypeError):
            divide(10, None)

    def test_divide_type_error_list_arg(self):
        with self.assertRaises(TypeError):
            divide([], 2)

    # --- Large/small values ---

    def test_divide_large_numbers(self):
        self.assertAlmostEqual(divide(1e15, 1e10), 1e5)

    def test_divide_small_divisor(self):
        result = divide(1, 0.001)
        self.assertEqual(result, 1000.0)


if __name__ == "__main__":
    unittest.main()
