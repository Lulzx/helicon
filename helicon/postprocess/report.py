"""Summary report generation.

Collects all computed metrics into a structured JSON report and
optionally generates summary CSV for parameter scans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import helicon


@dataclass
class RunReport:
    """Complete report of a simulation run."""

    helicon_version: str
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
    # Convergence diagnostics (§6.3 results.convergence)
    thrust_relative_change_last_10pct: float | None = None
    particle_count_exit: int | None = None
    # §14 reproducibility flag
    mass_ratio_reduced: bool = False
    # §7.3 validation proximity (populated via config_proximity())
    validation_proximity: dict | None = None

    def _compute_validation_flags(self) -> dict:
        """Derive spec §6.3 validation_flags from convergence diagnostics.

        Rules (per spec §7.1):
        - steady_state_reached: True when thrust_relative_change_last_10pct < 0.01 (1%)
        - particle_statistics_sufficient: True when particle_count_exit >= 1_000_000
        - energy_conservation_error: returned directly from convergence data
          (None when not yet computed)
        """
        steady = None
        if self.thrust_relative_change_last_10pct is not None:
            steady = self.thrust_relative_change_last_10pct < 0.01

        sufficient = None
        if self.particle_count_exit is not None:
            sufficient = self.particle_count_exit >= 1_000_000

        return {
            "steady_state_reached": steady,
            "particle_statistics_sufficient": sufficient,
            # Energy conservation error is a PIC diagnostic not yet piped
            # through postprocess; None until WarpX diagnostic integration.
            "energy_conservation_error": None,
        }

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
            timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

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

        # Compute validation proximity if config provided and not already set
        validation_proximity = self.validation_proximity
        if validation_proximity is None and config is not None:
            try:
                from helicon.validate.proximity import config_proximity

                prox = config_proximity(config)
                validation_proximity = {
                    "nearest_case": prox.nearest_case,
                    "distance": prox.distance,
                    "in_validated_region": prox.in_validated_region,
                    "parameter_distances": prox.parameter_distances,
                    "warning": prox.warning,
                }
            except Exception:
                pass

        return {
            "helicon_version": self.helicon_version,
            "config_hash": self.config_hash,
            "timestamp": timestamp,
            "nozzle_type": nozzle_type,
            "plasma_source": plasma_source,
            "mass_ratio_reduced": self.mass_ratio_reduced,
            "results": {
                "thrust_N": self.thrust_N,
                "isp_s": self.isp_s,
                "exhaust_velocity_ms": self.exhaust_velocity_ms,
                "mass_flow_rate_kgs": self.mass_flow_rate_kgs,
                "detachment_efficiency": {
                    "momentum_based": self.detachment_momentum,
                    "particle_based": self.detachment_particle,
                    "energy_based": self.detachment_energy,
                },
                "plume_half_angle_deg": self.plume_half_angle_deg,
                "beam_efficiency": self.beam_efficiency,
                "radial_loss_fraction": self.radial_loss_fraction,
                "convergence": {
                    "thrust_relative_change_last_10pct": (
                        self.thrust_relative_change_last_10pct
                    ),
                    "particle_count_exit": self.particle_count_exit,
                },
            },
            "validation_flags": self._compute_validation_flags(),
            "validation_proximity": validation_proximity,
        }


def generate_report(
    output_dir: str | Path,
    *,
    config_hash: str | None = None,
    config: Any = None,
) -> RunReport:
    """Generate a summary report from all available postprocessing results.

    Attempts to compute each metric; missing data results in None values.
    """
    output_dir = Path(output_dir)

    # Derive mass_ratio_reduced from config if provided
    mass_ratio_reduced = False
    if config is not None and hasattr(config, "plasma"):
        mr = getattr(config.plasma, "mass_ratio", None)
        mass_ratio_reduced = mr is not None and mr < 1836.0

    report = RunReport(
        helicon_version=helicon.__version__,
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
        mass_ratio_reduced=mass_ratio_reduced,
    )

    # Thrust
    try:
        from helicon.postprocess.thrust import compute_thrust

        thrust = compute_thrust(output_dir)
        report.thrust_N = thrust.thrust_N
        report.isp_s = thrust.isp_s
        report.exhaust_velocity_ms = thrust.exhaust_velocity_ms
        report.mass_flow_rate_kgs = thrust.mass_flow_rate_kgs
    except (FileNotFoundError, ValueError):
        pass

    # Detachment
    try:
        from helicon.postprocess.detachment import compute_detachment

        det = compute_detachment(output_dir)
        report.detachment_momentum = det.momentum_based
        report.detachment_particle = det.particle_based
        report.detachment_energy = det.energy_based
    except (FileNotFoundError, ValueError):
        pass

    # Plume
    try:
        from helicon.postprocess.plume import compute_plume_metrics

        plume = compute_plume_metrics(output_dir)
        report.plume_half_angle_deg = plume.divergence_half_angle_deg
        report.beam_efficiency = plume.beam_efficiency
        report.thrust_coefficient = plume.thrust_coefficient
        report.radial_loss_fraction = plume.radial_loss_fraction
    except (FileNotFoundError, ValueError):
        pass

    # Convergence diagnostics from time-series snapshots
    report.thrust_relative_change_last_10pct, report.particle_count_exit = (
        _compute_convergence_diagnostics(output_dir)
    )

    return report


def _compute_convergence_diagnostics(
    output_dir: Path,
) -> tuple[float | None, int | None]:
    """Compute convergence diagnostics from time-series HDF5 snapshots.

    Returns
    -------
    thrust_relative_change_last_10pct : float or None
        Relative std-dev of thrust over the final 10% of timesteps.
        A small value indicates the simulation reached steady state.
    particle_count_exit : int or None
        Total macro-particle count near the exit plane in the last snapshot.
    """
    import h5py
    import numpy as np

    h5_files = sorted(output_dir.glob("**/*.h5"))
    if len(h5_files) < 2:
        return None, None

    momentum_series: list[float] = []
    particle_counts: list[int] = []

    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                base = f["data"][sorted(f["data"].keys(), key=int)[-1]] if "data" in f else f
                if "particles" not in base:
                    continue
                total_pz = 0.0
                total_n = 0
                for sp_name in base["particles"]:
                    sp = base["particles"][sp_name]
                    if "momentum" not in sp or "z" not in sp["momentum"]:
                        continue
                    pz = sp["momentum"]["z"][:]
                    w = sp["weighting"][:] if "weighting" in sp else np.ones_like(pz)
                    total_pz += float(np.sum(w * pz))
                    # Exit plane: top 10% of z range
                    if "position" in sp and "z" in sp["position"]:
                        z = sp["position"]["z"][:]
                        z_threshold = z.min() + 0.9 * (z.max() - z.min())
                        total_n += int(np.sum(z > z_threshold))
                momentum_series.append(total_pz)
                particle_counts.append(total_n)
        except Exception:
            continue

    thrust_rc = None
    if len(momentum_series) >= 5:
        arr = np.array(momentum_series)
        n_last = max(1, len(arr) // 10)  # last 10%
        tail = arr[-n_last:]
        mean_tail = float(np.mean(np.abs(tail)))
        if mean_tail > 0:
            thrust_rc = float(np.std(tail) / mean_tail)

    particle_count_exit = particle_counts[-1] if particle_counts else None

    return thrust_rc, particle_count_exit


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
