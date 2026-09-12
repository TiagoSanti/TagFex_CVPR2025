"""Compatibility import and CLI; implementation in analysis.reporting.generate_html_pdf_report."""
import sys
from analysis.reporting import generate_html_pdf_report as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
