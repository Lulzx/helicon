"""VASIMR VX-200 plume validation case.

Validates plume divergence and thrust efficiency against published
experimental measurements from the VX-200 magnetoplasma thruster:

    Olsen, C.S., et al. (2015). "Investigation of plasma detachment from
    a magnetic nozzle in the plume of the VX-200 magnetoplasma thruster."
    IEEE Transactions on Plasma Science, 43(1), 252-268.

Reference operating point: 200 kW RF power, argon propellant.
Reported values: thrust ~5.7 N, thrust efficiency ~69%,
plume half-angle ~31°.

Pass criteria:
- Thrust efficiency within 15% of reference
- Plume half-angle within 20% of reference
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from magnozzlex.config.parser import (
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


# Reference data from Olsen et al. (2015) Table I / Fig. 9
# VX-200 at 200 kW, argon propellant
VASIMR_REFERENCE = {
    "thrust_N": 5.7,
    "thrust_efficiency": 0.69,
    "plume_half_angle_deg": 31.0,
    "specific_power_kW_per_N": 35.1,
}

# Tolerances from spec (VASIMR comparison is an external benchmark)
TOLERANCES = {
    "thrust_efficiency": 0.15,
    "plume_half_angle_deg": 0.20,
}


class VASIMRPlumeCase:
    """VASIMR VX-200 plume benchmark (Olsen et al. 2015).

    Tests thrust efficiency and plume divergence angle against
    published experimental measurements at 200 kW operating point.
    """

    name = "vasimr_plume"
    description = "VX-200 plume: thrust efficiency and divergence angle (Olsen 2015)"

    @staticmethod
    def get_config() -> SimConfig:
        """Return the VASIMR-equivalent simulation configuration.

        The VX-200 uses a helicon + ICH RF system.  We model the nozzle
        region only, with the plasma injected at the magnetic throat.

        Parameters approximate the VX-200 geometry and plasma conditions
        from Olsen (2015): B_throat ≈ 0.5 T, argon plasma.
        """
        # VX-200 nozzle coil approximation: two mirror coils
        # B_throat ~ 0.5 T → I ≈ B * 2a / μ₀ ≈ 400 kA-turns per coil
        mu0 = 4.0e-7 * np.pi
        B_throat = 0.5  # T
        r_coil = 0.12  # m
        I_coil = B_throat * 2.0 * r_coil / mu0

        return SimConfig(
            nozzle=NozzleConfig(
                type="converging_diverging",
                coils=[
                    CoilConfig(z=-0.05, r=r_coil, I=float(I_coil)),
                    CoilConfig(z=0.05, r=r_coil, I=float(I_coil) * 0.5),
                ],
                domain=DomainConfig(z_min=-0.3, z_max=2.0, r_max=0.8),
                resolution=ResolutionConfig(nz=512, nr=256),
            ),
            plasma=PlasmaSourceConfig(
                species=["Ar+", "e-"],
                n0=3.0e18,
                T_i_eV=50.0,
                T_e_eV=5.0,
                v_injection_ms=40000.0,  # argon ion thermal speed at 50 eV
            ),
            timesteps=100000,
            output_dir="results/validation/vasimr_plume",
        )

    @staticmethod
    def evaluate(output_dir: str | Path) -> ValidationResult:
        """Compare simulated plume metrics against VX-200 reference data."""
        output_dir = Path(output_dir)
        metrics: dict[str, float] = {}
        errors: dict[str, float] = {}

        try:
            from magnozzlex.postprocess.plume import compute_plume_metrics

            plume = compute_plume_metrics(output_dir)
            metrics["plume_half_angle_deg"] = plume.divergence_half_angle_deg
            metrics["beam_efficiency"] = plume.beam_efficiency
            metrics["thrust_coefficient"] = plume.thrust_coefficient

            # Beam efficiency as proxy for thrust efficiency
            ref_eff = VASIMR_REFERENCE["thrust_efficiency"]
            errors["thrust_efficiency"] = abs(
                plume.beam_efficiency - ref_eff
            ) / ref_eff

            ref_angle = VASIMR_REFERENCE["plume_half_angle_deg"]
            errors["plume_half_angle_deg"] = abs(
                plume.divergence_half_angle_deg - ref_angle
            ) / ref_angle

        except (FileNotFoundError, ValueError):
            return ValidationResult(
                case_name="vasimr_plume",
                passed=False,
                metrics={},
                tolerances=TOLERANCES,
                description="No output data found",
            )

        passed = all(errors[k] < TOLERANCES[k] for k in errors if k in TOLERANCES)
        metrics.update({f"error_{k}": v for k, v in errors.items()})

        return ValidationResult(
            case_name="vasimr_plume",
            passed=passed,
            metrics=metrics,
            tolerances=TOLERANCES,
            description="VX-200 plume comparison (Olsen 2015)",
        )
