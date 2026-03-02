"""Summary report generation.

Collects all computed metrics into a structured JSON report and
optionally generates summary CSV for parameter scans.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
    # Optional spec §6.3 fields
    nozzle_type: str | None = None
    plasma_source: dict | None = None

    def to_spec_dict(
        self,
        *,
        config: Any = None,
        timestamp: str | None = None,
    ) -> dict:
        """Return a spec §6.3 compliant nested dictionary.

        Parameters
        ----------
        config : SimConfig, optional
            Original simulation configuration; used to populate
            ``nozzle_type`` and ``plasma_source`` when not already set.
        timestamp : str, optional
            ISO-8601 timestamp string.  Defaults to the current UTC time.

        Returns
        -------
        dict
        """
        import datetime

        if timestamp is None:
            timestamp = (
                datetime.datetime.now(datetime.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            )

        # Resolve nozzle_type
        nozzle_type: str | None = self.nozzle_type
        plasma_source: dict | None = self.plasma_source
        if config is not None:
            if nozzle_type is None and hasattr(config, "nozzle"):
                nozzle_type = getattr(config.nozzle, "type", None)
            if plasma_source is None and hasattr(config, "plasma"):
                try:
                    plasma_source = config.plasma.model_dump(mode="python")
                except AttributeError:
                    plasma_source = None

        return {
            "magnozzlex_version": self.magnozzlex_version,
            "config_hash": self.config_hash,
            "timestamp": timestamp,
            "nozzle_type": nozzle_type,
            "plasma_source": plasma_source,
            "results": {
                "thrust_N": self.thrust_N,
                "isp_s": self.isp_s,
                "detachment_efficiency": {
                    "momentum_based": self.detachment_momentum,
                    "particle_based": self.detachment_particle,
                    "energy_based": self.detachment_energy,
                },
                "plume_half_angle_deg": self.plume_half_angle_deg,
                "beam_efficiency": self.beam_efficiency,
                "radial_loss_fraction": self.radial_loss_fraction,
                "convergence": {
                    "thrust_relative_change_last_10pct": None,
                    "particle_count_exit": None,
                },
            },
            "validation_flags": {
                "steady_state_reached": None,
                "particle_statistics_sufficient": None,
                "energy_conservation_error": None,
            },
        }


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


def save_report(report: RunReport, path: str | Path, *, config: Any = None) -> None:
    """Save a report to JSON using the spec §6.3 nested format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = report.to_spec_dict(config=config)
    path.write_text(json.dumps(data, indent=2, default=str))


def load_report(path: str | Path) -> dict:
    """Load a report from JSON."""
    return json.loads(Path(path).read_text())


def save_scan_csv(
    reports: list[RunReport],
    param_labels: list[str],
    param_values: list[list[float]],
    path: str | Path,
) -> None:
    """Save a parameter scan summary to CSV.

    Parameters
    ----------
    reports : list of RunReport
        One report per scan point.
    param_labels : list of str
        Names of the varied parameters.
    param_values : list of list of float
        Parameter values for each scan point.
    path : path
        Output CSV file path.
    """
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_keys = [
        "thrust_N",
        "isp_s",
        "exhaust_velocity_ms",
        "mass_flow_rate_kgs",
        "plume_half_angle_deg",
        "beam_efficiency",
        "thrust_coefficient",
        "detachment_momentum",
        "detachment_particle",
        "detachment_energy",
    ]

    fieldnames = list(param_labels) + metric_keys
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for params, report in zip(param_values, reports):
            row: dict = dict(zip(param_labels, params))
            for key in metric_keys:
                row[key] = getattr(report, key, None)
            writer.writerow(row)
