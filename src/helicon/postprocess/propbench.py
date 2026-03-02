"""PropBench JSON schema for propulsion benchmark data exchange.

PropBench is an emerging standard for sharing and comparing
propulsion simulation results across different codes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from helicon.postprocess.report import RunReport


@dataclass
class PropBenchResult:
    """PropBench-compatible result record."""

    propbench_version: str = "0.1"
    code_name: str = "Helicon"
    code_version: str = ""
    timestamp: str = ""
    config_hash: str | None = None
    nozzle_type: str | None = None
    species: list[str] | None = None
    n0_m3: float | None = None
    T_i_eV: float | None = None
    T_e_eV: float | None = None
    thrust_N: float | None = None
    isp_s: float | None = None
    exhaust_velocity_ms: float | None = None
    mass_flow_rate_kgs: float | None = None
    detachment_efficiency_momentum: float | None = None
    detachment_efficiency_particle: float | None = None
    detachment_efficiency_energy: float | None = None
    plume_half_angle_deg: float | None = None
    beam_efficiency: float | None = None
    mass_ratio_reduced: bool = False
    electron_model: str = "kinetic"

    def __post_init__(self) -> None:
        if self.species is None:
            self.species = []


def to_propbench(
    run_report: RunReport,
    config: object | None = None,
    *,
    mass_ratio_reduced: bool = False,
    electron_model: str = "kinetic",
) -> PropBenchResult:
    """Convert a RunReport to a PropBenchResult.

    Parameters
    ----------
    run_report : RunReport
        Simulation results from postprocess/report.py.
    config : SimConfig or None
        Optional simulation configuration to extract plasma parameters.
    mass_ratio_reduced : bool
        Whether a reduced mass ratio was used.
    electron_model : str
        Electron treatment model used (e.g., "kinetic", "fluid").

    Returns
    -------
    PropBenchResult
    """
    result = PropBenchResult(
        code_version=run_report.helicon_version,
        timestamp=datetime.now(tz=UTC).isoformat(),
        config_hash=run_report.config_hash,
        thrust_N=run_report.thrust_N,
        isp_s=run_report.isp_s,
        exhaust_velocity_ms=run_report.exhaust_velocity_ms,
        mass_flow_rate_kgs=run_report.mass_flow_rate_kgs,
        detachment_efficiency_momentum=run_report.detachment_momentum,
        detachment_efficiency_particle=run_report.detachment_particle,
        detachment_efficiency_energy=run_report.detachment_energy,
        plume_half_angle_deg=run_report.plume_half_angle_deg,
        beam_efficiency=run_report.beam_efficiency,
        mass_ratio_reduced=mass_ratio_reduced,
        electron_model=electron_model,
    )

    if config is not None:
        # Extract from SimConfig-like object
        if hasattr(config, "nozzle"):
            result.nozzle_type = getattr(config.nozzle, "type", None)
        if hasattr(config, "plasma"):
            plasma = config.plasma
            result.species = list(getattr(plasma, "species", []))
            result.n0_m3 = getattr(plasma, "n0", None)
            result.T_i_eV = getattr(plasma, "T_i_eV", None)
            result.T_e_eV = getattr(plasma, "T_e_eV", None)

    return result


def save_propbench(result: PropBenchResult, path: str | Path) -> None:
    """Write a PropBenchResult to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(result)
    path.write_text(json.dumps(data, indent=2, default=str))


def load_propbench(path: str | Path) -> PropBenchResult:
    """Load a PropBenchResult from a JSON file."""
    data = json.loads(Path(path).read_text())
    return PropBenchResult(**data)
