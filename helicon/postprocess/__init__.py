"""Post-processing pipeline for extracting propulsion metrics from WarpX output."""

from helicon.postprocess.detachment import compute_detachment
from helicon.postprocess.moments import compute_moments
from helicon.postprocess.plume import compute_plume_metrics
from helicon.postprocess.propbench import (
    PropBenchResult,
    load_propbench,
    save_propbench,
    to_propbench,
)
from helicon.postprocess.normalize import (
    PlasmaScales,
    compute_plasma_scales,
    normalize_bfield,
    normalize_density,
    normalize_length,
    normalize_pressure,
    normalize_time,
    normalize_velocity,
)
from helicon.postprocess.pulsed import compute_pulsed_metrics
from helicon.postprocess.report import generate_report, load_report, save_report
from helicon.postprocess.thrust import compute_thrust

__all__ = [
    "PlasmaScales",
    "PropBenchResult",
    "compute_detachment",
    "compute_moments",
    "compute_plasma_scales",
    "compute_plume_metrics",
    "compute_pulsed_metrics",
    "compute_thrust",
    "generate_report",
    "load_propbench",
    "load_report",
    "normalize_bfield",
    "normalize_density",
    "normalize_length",
    "normalize_pressure",
    "normalize_time",
    "normalize_velocity",
    "save_propbench",
    "save_report",
    "to_propbench",
]
