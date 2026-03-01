"""Magnetic field computation module."""

from magnozzlex.fields.biot_savart import BField, Coil, Grid, compute_bfield
from magnozzlex.fields.field_lines import (
    FieldLine,
    FieldLineSet,
    FieldLineType,
    compute_flux_function,
    trace_field_line,
    trace_field_lines,
)

__all__ = [
    "BField",
    "Coil",
    "FieldLine",
    "FieldLineSet",
    "FieldLineType",
    "Grid",
    "compute_bfield",
    "compute_flux_function",
    "trace_field_line",
    "trace_field_lines",
]
