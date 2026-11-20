import pytest
from trial_math import divide

@pytest.mark.parametrize("a,b,expected", [(6, 3, 2), (-6, 3, -2), (1, 2, 0.5)])
def test_normal(a, b, expected):
    assert divide(a, b) == expected

def test_zero():
    with pytest.raises(ValueError, match="^denominator must not be zero$"):
        divide(1, 0)
