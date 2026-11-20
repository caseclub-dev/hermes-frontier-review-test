import pytest
from clamp import clamp


# --- normal range cases ---

def test_value_inside_range():
    assert clamp(5, 0, 10) == 5


def test_value_at_lower_bound():
    assert clamp(0, 0, 10) == 0


def test_value_at_upper_bound():
    assert clamp(10, 0, 10) == 10


# --- clamping to bounds ---

def test_clamp_below_lower():
    assert clamp(-5, 0, 10) == 0


def test_clamp_above_upper():
    assert clamp(15, 0, 10) == 10


# --- negative and fractional values ---

def test_negative_value_and_range():
    assert clamp(-3, -10, -1) == -3


def test_fractional_clamp():
    assert clamp(2.7, 0.0, 5.5) == 2.7


def test_fractional_clamp_below():
    assert clamp(-0.5, 0.0, 1.0) == 0.0


# --- equal lower/upper ---

def test_equal_bounds():
    assert clamp(5, 3, 3) == 3


def test_value_at_equal_bounds():
    assert clamp(3, 3, 3) == 3


# --- ValueError: lower > upper ---

def test_lower_exceeds_upper():
    with pytest.raises(ValueError, match="^lower must not exceed upper$"):
        clamp(1, 5, 0)


def test_lower_exceeds_upper_equal():
    """equal is fine; strictly greater triggers the error."""
    assert clamp(1, 2, 1)  # should NOT raise


def test_negative_bounds_violation():
    with pytest.raises(ValueError, match="^lower must not exceed upper$"):
        clamp(-5, 0, -10)
