"""Physics sanity checks for simulation configurations.

These go beyond schema validation — they check whether the configuration
is physically reasonable for a magnetic nozzle simulation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from helicon.config.parser import SimConfig
from helicon.fields.biot_savart import MU_0


@dataclass
class ValidationResult:
    """Result of physics validation."""

    passed: bool
    warnings: list[str]
    errors: list[str]


def validate_config(config: SimConfig) -> ValidationResult:
    """Run all physics sanity checks on a simulation configuration.

    Returns a ``ValidationResult`` with any warnings or errors found.
    """
    warns: list[str] = []
    errors: list[str] = []

    # --- Domain checks ---
    domain = config.nozzle.domain
    domain_length = domain.z_max - domain.z_min
    if domain_length < 0.1:
        warns.append(
            f"Domain axial length ({domain_length:.3f} m) is very short. "
            "Detachment may not be captured."
        )

    # --- Coil checks ---
    for i, coil in enumerate(config.nozzle.coils):
        if coil.r > domain.r_max:
            errors.append(
                f"Coil {i} radius ({coil.r:.3f} m) exceeds domain r_max "
                f"({domain.r_max:.3f} m)."
            )
        if not (domain.z_min <= coil.z <= domain.z_max):
            warns.append(
                f"Coil {i} at z={coil.z:.3f} m is outside the domain "
                f"[{domain.z_min}, {domain.z_max}]."
            )

    # --- Resolution checks ---
    res = config.nozzle.resolution
    dr = domain.r_max / res.nr
    min_coil_r = min(c.r for c in config.nozzle.coils)
    if min_coil_r / dr < 5:
        warns.append(
            f"Radial resolution may be insufficient: smallest coil radius "
            f"({min_coil_r:.3f} m) is only {min_coil_r / dr:.1f} cells wide."
        )

    # --- Plasma beta estimate at throat ---
    # beta = n * k_B * (T_i + T_e) / (B^2 / 2 mu_0)
    # Rough B at throat: use on-axis field from strongest coil
    strongest = max(config.nozzle.coils, key=lambda c: abs(c.I))
    B_throat_approx = MU_0 * abs(strongest.I) / (2.0 * strongest.r)

    eV_to_J = 1.602176634e-19
    n0 = config.plasma.n0
    T_total_J = (config.plasma.T_i_eV + config.plasma.T_e_eV) * eV_to_J
    p_plasma = n0 * T_total_J
    p_mag = B_throat_approx**2 / (2.0 * MU_0)

    if p_mag > 0:
        beta = p_plasma / p_mag
        if beta > 1.0:
            warns.append(
                f"Estimated plasma beta at throat ≈ {beta:.2f} > 1. "
                "Magnetic confinement may be insufficient."
            )

    # --- Mass ratio warning ---
    if config.plasma.mass_ratio is not None and config.plasma.mass_ratio < 100:
        warns.append(
            f"Mass ratio {config.plasma.mass_ratio} is very low. "
            "Results will be qualitative only."
        )

    # --- Timestep check ---
    if config.timesteps < 1000:
        warns.append(f"Only {config.timesteps} timesteps — may not reach steady state.")

    # Emit warnings to Python warning system as well
    for w in warns:
        warnings.warn(w, UserWarning, stacklevel=2)

    return ValidationResult(
        passed=len(errors) == 0,
        warnings=warns,
        errors=errors,
    )
