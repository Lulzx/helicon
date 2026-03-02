"""Automated validation suite for Helicon."""

from helicon.validate.proximity import ProximityResult, config_proximity
from helicon.validate.runner import run_validation

__all__ = ["run_validation", "config_proximity", "ProximityResult"]
