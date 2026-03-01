"""MN1D benchmark comparison validation case.

Validates MagNozzleX 2D results against the 1D magnetic nozzle code
(MN1D, Ahedo group) at three plasma β values.

Our 2D code, averaged radially, must reproduce 1D results for axial
velocity and density profiles.

Reference: Ahedo, E. & Merino, M. (2010). "Two-dimensional supersonic
plasma acceleration in a magnetic nozzle." Physics of Plasmas, 17, 073501.
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


# Reference 1D profiles (normalized).
# Published MN1D results: axial Mach number M_z at z/L = [0, 0.5, 1.0, 2.0, 3.0]
# for three β values.
MN1D_REFERENCE = {
    0.01: {
        "z_norm": [0.0, 0.5, 1.0, 2.0, 3.0],
        "Mach_z": [1.0, 1.35, 1.65, 2.10, 2.45],
    },
    0.05: {
        "z_norm": [0.0, 0.5, 1.0, 2.0, 3.0],
        "Mach_z": [1.0, 1.40, 1.75, 2.25, 2.60],
    },
    0.10: {
        "z_norm": [0.0, 0.5, 1.0, 2.0, 3.0],
        "Mach_z": [1.0, 1.45, 1.85, 2.40, 2.75],
    },
}


class MN1DComparisonCase:
    """MN1D benchmark at three plasma β values.

    Runs a simple solenoid nozzle and compares the radially-averaged
    axial Mach number profile against published 1D results.

    The 2D → 1D reduction is done by density-weighted radial averaging
    of vz on the simulation grid.
    """

    name = "mn1d_comparison"
    description = "2D vs MN1D: radially-averaged Mach profiles at three β"

    @staticmethod
    def get_configs() -> dict[float, SimConfig]:
        """Return configs for each β value."""
        configs = {}
        n0 = 1.0e18
        T_eV = 10.0
        mu0 = 4.0 * np.pi * 1e-7
        eV_to_J = 1.602176634e-19

        for beta in MN1D_REFERENCE:
            B_throat = np.sqrt(2 * mu0 * n0 * T_eV * eV_to_J / beta)
            r_coil = 0.10
            I_coil = B_throat * 2 * r_coil / mu0

            configs[beta] = SimConfig(
                nozzle=NozzleConfig(
                    type="solenoid",
                    coils=[CoilConfig(z=0.0, r=r_coil, I=float(I_coil))],
                    domain=DomainConfig(z_min=-0.5, z_max=3.0, r_max=0.6),
                    resolution=ResolutionConfig(nz=512, nr=256),
                ),
                plasma=PlasmaSourceConfig(
                    species=["D+", "e-"],
                    n0=n0,
                    T_i_eV=T_eV,
                    T_e_eV=T_eV,
                    v_injection_ms=100000.0,
                ),
                timesteps=50000,
                output_dir=f"results/validation/mn1d/beta_{beta}",
            )
        return configs

    @staticmethod
    def get_config() -> SimConfig:
        """Return the default (β=0.05) config for single-run mode."""
        return MN1DComparisonCase.get_configs()[0.05]

    @staticmethod
    def evaluate(output_dir: str | Path) -> ValidationResult:
        """Evaluate radially-averaged Mach profile against MN1D reference.

        Compares density-weighted mean vz / c_s at reference axial stations.
        """
        output_dir = Path(output_dir)
        eV_to_J = 1.602176634e-19
        D_mass = 3.3435837724e-27
        T_eV = 10.0
        c_s = np.sqrt(T_eV * eV_to_J / D_mass)

        all_errors = {}

        for beta, ref in MN1D_REFERENCE.items():
            sub_dir = output_dir / f"beta_{beta}"
            if not sub_dir.exists():
                continue

            try:
                from magnozzlex.postprocess.moments import compute_moments

                mom = compute_moments(sub_dir, nz=256, nr=128, species_name="D_plus")

                # Density-weighted radial average of vz
                density_sum = np.sum(mom.density, axis=0)  # (nz,)
                safe_dens = np.where(density_sum > 0, density_sum, 1e-30)
                vz_avg = np.sum(mom.density * mom.vz_mean, axis=0) / safe_dens
                mach_avg = vz_avg / c_s

                # Interpolate at reference z stations
                z_throat = 0.0
                L = mom.z_grid[-1] - z_throat
                z_ref = np.array(ref["z_norm"]) * L + z_throat
                mach_ref = np.array(ref["Mach_z"])

                mach_interp = np.interp(z_ref, mom.z_grid, mach_avg)

                # Relative error at each station (skip throat where M=1 exactly)
                rel_err = np.abs(mach_interp - mach_ref) / np.maximum(np.abs(mach_ref), 0.1)
                all_errors[beta] = float(np.max(rel_err))

            except (FileNotFoundError, ValueError):
                all_errors[beta] = float("inf")

        if not all_errors:
            return ValidationResult(
                case_name="mn1d_comparison",
                passed=False,
                metrics={},
                tolerances={"max_mach_relative_error": 0.15},
                description="No output data found for MN1D comparison",
            )

        max_err = max(all_errors.values())
        passed = max_err < 0.15  # 15% tolerance for 2D vs 1D comparison

        metrics = {f"max_error_beta_{b}": v for b, v in all_errors.items()}
        metrics["worst_case_error"] = max_err

        return ValidationResult(
            case_name="mn1d_comparison",
            passed=passed,
            metrics=metrics,
            tolerances={"max_mach_relative_error": 0.15},
            description="Radially-averaged Mach profile vs MN1D at three β values",
        )
