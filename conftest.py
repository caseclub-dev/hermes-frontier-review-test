"""Auto-inject calculator functions into test module globals at collection time."""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def pytest_collection_modifyitems(config, items):
    """Inject calculator functions into each test module's globals."""
    from src.calculator import add, subtract, multiply, divide

    for item in items:
        mod = item.module
        if any(name in mod.__name__ for name in ("test_", "_test")):
            mod.add = add
            mod.subtract = subtract
            mod.multiply = multiply
            mod.divide = divide
