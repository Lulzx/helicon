"""Automated validation suite for Helicon."""

from helicon.validate.proximity import ProximityResult, config_proximity
from helicon.validate.regression import (
    RegressionSuite,
    compare_to_baseline,
    generate_markdown_report,
    load_baseline,
    save_baseline,
    save_regression_report,
)
from helicon.validate.runner import run_validation

__all__ = [
    "ProximityResult",
    "RegressionSuite",
    "compare_to_baseline",
    "config_proximity",
    "generate_markdown_report",
    "load_baseline",
    "run_validation",
    "save_baseline",
    "save_regression_report",
]
