"""Validation case definitions."""

from magnozzlex.validate.cases.free_expansion import FreeExpansionCase
from magnozzlex.validate.cases.guiding_center import GuidingCenterCase
from magnozzlex.validate.cases.merino_ahedo import MerinoAhedoCase
from magnozzlex.validate.cases.mn1d_comparison import MN1DComparisonCase
from magnozzlex.validate.cases.resistive_dimov import ResistiveDimovCase
from magnozzlex.validate.cases.vasimr_plume import VASIMRPlumeCase

__all__ = [
    "FreeExpansionCase",
    "GuidingCenterCase",
    "MN1DComparisonCase",
    "MerinoAhedoCase",
    "ResistiveDimovCase",
    "VASIMRPlumeCase",
]
