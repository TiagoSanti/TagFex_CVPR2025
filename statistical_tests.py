"""Compatibility import and CLI; implementation in analysis.statistics.statistical_tests."""
import sys
from analysis.statistics import statistical_tests as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
