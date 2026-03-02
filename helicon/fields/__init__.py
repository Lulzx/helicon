"""Magnetic field computation module."""

from helicon.fields.biot_savart import BField, Coil, Grid, compute_bfield
from helicon.fields.cache import FieldCache, compute_bfield_cached
from helicon.fields.field_lines import (
    FieldLine,
    FieldLineSet,
    FieldLineType,
    compute_flux_function,
    trace_field_line,
    trace_field_lines,
)
from helicon.fields.frc_topology import FRCTopologyResult, find_frc_topology
from helicon.fields.import_external import load_csv_bfield, load_femm_bfield

__all__ = [
    "BField",
    "Coil",
    "FRCTopologyResult",
    "FieldCache",
    "FieldLine",
    "FieldLineSet",
    "FieldLineType",
    "Grid",
    "compute_bfield",
    "compute_bfield_cached",
    "compute_flux_function",
    "find_frc_topology",
    "load_csv_bfield",
    "load_femm_bfield",
    "trace_field_line",
    "trace_field_lines",
]
