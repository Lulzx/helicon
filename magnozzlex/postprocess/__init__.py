"""Post-processing pipeline for extracting propulsion metrics from WarpX output."""

from magnozzlex.postprocess.detachment import compute_detachment
from magnozzlex.postprocess.moments import compute_moments
from magnozzlex.postprocess.plume import compute_plume_metrics
from magnozzlex.postprocess.propbench import PropBenchResult, load_propbench, save_propbench, to_propbench
from magnozzlex.postprocess.pulsed import compute_pulsed_metrics
from magnozzlex.postprocess.report import generate_report, load_report, save_report
from magnozzlex.postprocess.thrust import compute_thrust

__all__ = [
    "PropBenchResult",
    "compute_detachment",
    "compute_moments",
    "compute_plume_metrics",
    "compute_pulsed_metrics",
    "compute_thrust",
    "generate_report",
    "load_propbench",
    "load_report",
    "save_propbench",
    "save_report",
    "to_propbench",
]
