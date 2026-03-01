"""Summary report generation.

Collects all computed metrics into a structured JSON report and
optionally generates summary CSV for parameter scans.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import magnozzlex


@dataclass
class RunReport:
    """Complete report of a simulation run."""

    magnozzlex_version: str
    config_hash: str | None
    thrust_N: float | None
    isp_s: float | None
    exhaust_velocity_ms: float | None
    mass_flow_rate_kgs: float | None
    detachment_momentum: float | None
    detachment_particle: float | None
    detachment_energy: float | None
    plume_half_angle_deg: float | None
    beam_efficiency: float | None
    thrust_coefficient: float | None
    radial_loss_fraction: float | None


def generate_report(
    output_dir: str | Path,
    *,
    config_hash: str | None = None,
) -> RunReport:
    """Generate a summary report from all available postprocessing results.

    Attempts to compute each metric; missing data results in None values.
    """
    output_dir = Path(output_dir)

    report = RunReport(
        magnozzlex_version=magnozzlex.__version__,
        config_hash=config_hash,
        thrust_N=None,
        isp_s=None,
        exhaust_velocity_ms=None,
        mass_flow_rate_kgs=None,
        detachment_momentum=None,
        detachment_particle=None,
        detachment_energy=None,
        plume_half_angle_deg=None,
        beam_efficiency=None,
        thrust_coefficient=None,
        radial_loss_fraction=None,
    )

    # Thrust
    try:
        from magnozzlex.postprocess.thrust import compute_thrust

        thrust = compute_thrust(output_dir)
        report.thrust_N = thrust.thrust_N
        report.isp_s = thrust.isp_s
        report.exhaust_velocity_ms = thrust.exhaust_velocity_ms
        report.mass_flow_rate_kgs = thrust.mass_flow_rate_kgs
    except (FileNotFoundError, ValueError):
        pass

    # Detachment
    try:
        from magnozzlex.postprocess.detachment import compute_detachment

        det = compute_detachment(output_dir)
        report.detachment_momentum = det.momentum_based
        report.detachment_particle = det.particle_based
        report.detachment_energy = det.energy_based
    except (FileNotFoundError, ValueError):
        pass

    # Plume
    try:
        from magnozzlex.postprocess.plume import compute_plume_metrics

        plume = compute_plume_metrics(output_dir)
        report.plume_half_angle_deg = plume.divergence_half_angle_deg
        report.beam_efficiency = plume.beam_efficiency
        report.thrust_coefficient = plume.thrust_coefficient
        report.radial_loss_fraction = plume.radial_loss_fraction
    except (FileNotFoundError, ValueError):
        pass

    return report


def save_report(report: RunReport, path: str | Path) -> None:
    """Save a report to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(report)
    path.write_text(json.dumps(data, indent=2, default=str))


def load_report(path: str | Path) -> dict:
    """Load a report from JSON."""
    return json.loads(Path(path).read_text())
