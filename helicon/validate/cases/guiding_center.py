"""Guiding-center orbit validation case.

A single charged particle in a known magnetic field. The trajectory
must match Littlejohn (1983) guiding-center theory to within expected
Boris-pusher accuracy (2nd order in dt).

Reference: Littlejohn, R.G. (1983) J. Plasma Physics 29, 111.
Criterion: Orbit error decreases as O(dt^2) with timestep refinement.
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


class GuidingCenterCase:
    """Single-particle guiding center orbit in a mirror field.

    A single ion is launched along a magnetic mirror axis. The guiding
    center approximation predicts the trajectory analytically. The PIC
    Boris pusher must converge to this solution at 2nd order in dt.
    """

    name = "guiding_center"
    description = "Single-particle guiding center orbit vs Littlejohn theory"

    @staticmethod
    def get_config() -> SimConfig:
        """Return the simulation configuration for this case."""
        return SimConfig(
            nozzle=NozzleConfig(
                type="solenoid",
                coils=[
                    CoilConfig(z=0.0, r=0.05, I=10000),
                    CoilConfig(z=0.5, r=0.05, I=10000),
                ],
                domain=DomainConfig(z_min=-0.1, z_max=0.6, r_max=0.1),
                resolution=ResolutionConfig(nz=128, nr=64),
            ),
            plasma=PlasmaSourceConfig(
                species=["H+", "e-"],
                n0=1.0e10,  # very low density — single particle regime
                T_i_eV=10.0,
                T_e_eV=10.0,
                v_injection_ms=10000.0,
            ),
            timesteps=10000,
            output_dir="results/validation/guiding_center",
        )

    @staticmethod
    def analytic_gyroradius(B: float, v_perp: float, mass: float, charge: float) -> float:
        """Compute the analytic Larmor radius.

        r_L = m * v_perp / (|q| * B)
        """
        return mass * v_perp / (abs(charge) * B)

    @staticmethod
    def evaluate(output_dir: str | Path) -> ValidationResult:
        """Evaluate the guiding-center validation case.

        Checks that the Boris pusher orbit converges to the guiding-center
        solution at 2nd order in dt.
        """
        output_dir = Path(output_dir)

        try:
            # Real implementation would:
            # 1. Read particle trajectory from multiple dt runs
            # 2. Compute guiding-center prediction analytically
            # 3. Measure convergence order

            # Framework for convergence check
            convergence_order = 2.0  # expected for Boris pusher
            measured_order = 0.0  # placeholder
            passed = abs(measured_order - convergence_order) / convergence_order < 0.2

            return ValidationResult(
                case_name="guiding_center",
                passed=passed,
                metrics={
                    "expected_convergence_order": convergence_order,
                    "measured_convergence_order": measured_order,
                },
                tolerances={"order_relative_error": 0.2},
                description="Boris pusher convergence to guiding-center theory",
            )
        except FileNotFoundError:
            return ValidationResult(
                case_name="guiding_center",
                passed=False,
                metrics={},
                tolerances={"order_relative_error": 0.2},
                description="No output data found — simulation may not have run",
            )
