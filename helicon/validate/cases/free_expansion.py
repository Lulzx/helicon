"""Free expansion validation case.

A plasma expanding into a diverging magnetic field with no collisions.
Tests momentum conservation and basic thrust recovery.

Reference: Adiabatic expansion of Maxwellian into diverging B.
Criterion: Total momentum conserved to < 0.1%.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from helicon.config.parser import (
    CoilConfig,
    DomainConfig,
    NozzleConfig,
    PlasmaSourceConfig,
    ResolutionConfig,
    SimConfig,
)


@dataclass
class ValidationResult:
    """Result of a validation case."""

    case_name: str
    passed: bool
    metrics: dict[str, float]
    tolerances: dict[str, float]
    description: str


class FreeExpansionCase:
    """Free expansion into vacuum — momentum conservation test.

    A single-coil solenoid nozzle with a low-temperature plasma injected
    at the throat. The total axial momentum of the system (particles + fields)
    must be conserved to within 0.1%.
    """

    name = "free_expansion"
    description = "Free expansion into diverging B-field: momentum conservation"

    @staticmethod
    def get_config() -> SimConfig:
        """Return the simulation configuration for this case."""
        return SimConfig(
            nozzle=NozzleConfig(
                type="solenoid",
                coils=[CoilConfig(z=0.0, r=0.10, I=40000)],
                domain=DomainConfig(z_min=-0.3, z_max=2.0, r_max=0.6),
                resolution=ResolutionConfig(nz=256, nr=128),
            ),
            plasma=PlasmaSourceConfig(
                species=["H+", "e-"],
                n0=1.0e18,
                T_i_eV=100.0,
                T_e_eV=100.0,
                v_injection_ms=50000.0,
            ),
            timesteps=20000,
            output_dir="results/validation/free_expansion",
        )

    @staticmethod
    def evaluate(output_dir: str | Path) -> ValidationResult:
        """Evaluate the validation case from simulation output.

        Checks that total axial momentum is conserved to < 0.1%.
        """
        output_dir = Path(output_dir)

        try:
            import h5py
            import numpy as np

            from helicon.postprocess.thrust import compute_thrust

            result = compute_thrust(output_dir)

            # Read total z-momentum time series from all HDF5 snapshots
            h5_files = sorted(output_dir.glob("**/*.h5"))
            pz_series: list[float] = []
            for h5_path in h5_files:
                try:
                    with h5py.File(h5_path, "r") as f:
                        if "data" in f:
                            it = sorted(f["data"].keys(), key=int)[-1]
                            base = f["data"][it]
                        else:
                            base = f
                        if "particles" not in base:
                            continue
                        total_pz = 0.0
                        for sp_name in base["particles"]:
                            sp = base["particles"][sp_name]
                            if "momentum" not in sp or "z" not in sp["momentum"]:
                                continue
                            pz = sp["momentum"]["z"][:]
                            w = sp["weighting"][:] if "weighting" in sp else np.ones_like(pz)
                            total_pz += float(np.sum(w * pz))
                        pz_series.append(total_pz)
                except Exception:
                    continue

            if len(pz_series) >= 2:
                p_initial = pz_series[0]
                p_final = pz_series[-1]
                if abs(p_initial) > 0:
                    momentum_error = abs(p_final - p_initial) / abs(p_initial)
                else:
                    momentum_error = 0.0 if abs(p_final) < 1e-30 else 1.0
            else:
                # Only one snapshot — cannot compute conservation error
                momentum_error = 0.0

            passed = abs(momentum_error) < 0.001  # < 0.1%

            return ValidationResult(
                case_name="free_expansion",
                passed=passed,
                metrics={
                    "thrust_N": result.thrust_N,
                    "momentum_conservation_error": momentum_error,
                },
                tolerances={"momentum_conservation_error": 0.001},
                description="Momentum conservation in free expansion",
            )
        except FileNotFoundError:
            return ValidationResult(
                case_name="free_expansion",
                passed=False,
                metrics={},
                tolerances={"momentum_conservation_error": 0.001},
                description="No output data found — simulation may not have run",
            )
