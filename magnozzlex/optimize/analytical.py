"""Fast analytical pre-screening for magnetic nozzle geometries.

Implements the Tier 1 analytical model described in the v0.3 roadmap:

* **Mirror ratio** R_B = B_throat / B_exit computed from Biot-Savart
* **Paraxial thrust coefficient** C_T from Little & Choueiri (2013)
* **Plume divergence half-angle** from field-line geometry

These models evaluate in milliseconds on CPU (no WarpX required) and
are used to pre-screen thousands of coil configurations before running
the expensive Tier 2 WarpX PIC simulations.

References
----------
1. Breizman, B.N. & Arefiev, A.V. (2008). Paraxial model of a magnetic
   nozzle. *Physics of Plasmas*, 15, 057103.
2. Little, J.M. & Choueiri, E.Y. (2013). Thrust and efficiency model for
   electron-driven magnetic nozzles. *Physics of Plasmas*, 20, 103501.
3. Ahedo, E. & Merino, M. (2010). Two-dimensional supersonic plasma
   acceleration in a magnetic nozzle. *Physics of Plasmas*, 17, 073501.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class NozzleScreeningResult:
    """Fast analytical screening metrics for a coil configuration.

    Attributes
    ----------
    mirror_ratio : float
        R_B = B_throat / B_exit.  Should be >> 1 for good confinement.
    thrust_coefficient : float
        C_T = F / (ṁ c_s).  Dimensionless nozzle performance metric.
    divergence_half_angle_deg : float
        Plume half-angle estimate [°].  Lower is better.
    thrust_efficiency : float
        η_T = 1 − 1/√R_B.  Fraction of thermal energy converted to thrust.
    """

    mirror_ratio: float
    thrust_coefficient: float
    divergence_half_angle_deg: float
    thrust_efficiency: float


def mirror_ratio(
    coils: list,
    *,
    z_min: float,
    z_max: float,
    n_pts: int = 200,
    backend: str = "auto",
) -> float:
    """Compute B_z,max / B_z,exit on axis from Biot-Savart.

    Parameters
    ----------
    coils : list of Coil
        Coil configuration.
    z_min, z_max : float
        Axial extent to evaluate [m].
    n_pts : int
        Number of on-axis evaluation points.
    backend : str
        Biot-Savart backend (``"auto"``, ``"numpy"``, ``"mlx"``).

    Returns
    -------
    R_B : float
        Mirror ratio  B_max / B_exit  (B_exit = B_z at z_max).
    """
    from magnozzlex.fields.biot_savart import Grid, compute_bfield

    grid = Grid(z_min=z_min, z_max=z_max, r_max=0.001, nz=n_pts, nr=2)
    bf = compute_bfield(coils, grid, backend=backend)
    Bz_axis = bf.Bz[0, :]  # r=0 row
    B_max = float(np.max(np.abs(Bz_axis)))
    B_exit = float(np.abs(Bz_axis[-1]))
    if B_exit < 1e-20:
        return float("inf")
    return B_max / B_exit


def thrust_coefficient_paraxial(
    R_B: float,
    gamma: float = 5.0 / 3.0,
) -> float:
    """Paraxial thrust coefficient from Little & Choueiri (2013).

    Simplified model for a polytropic electron gas expanding through a
    magnetic nozzle with mirror ratio R_B = B_throat / B_exit.

    For γ = 5/3 (adiabatic): C_T = √(2(γ+1)/(γ−1)) * √(1 − R_B^(1−γ))
    Reduces to the Breizman-Arefiev result in the cold-plasma limit.

    Parameters
    ----------
    R_B : float
        Mirror ratio B_throat / B_exit.  Must be > 1.
    gamma : float
        Polytropic index for electrons (5/3 adiabatic, 1 isothermal).

    Returns
    -------
    C_T : float
        Thrust coefficient (dimensionless).
    """
    if R_B <= 1.0:
        return 0.0
    if math.isinf(R_B):
        return math.sqrt(2.0 * (gamma + 1.0) / (gamma - 1.0))
    # Field expansion ratio at exit: B_exit/B_throat = 1/R_B
    # Energy conservation: C_T = √(2) * √(η_T) * normalization
    eta_T = thrust_efficiency(R_B, gamma=gamma)
    prefactor = math.sqrt(2.0 * (gamma + 1.0) / (gamma - 1.0))
    return prefactor * math.sqrt(eta_T)


def thrust_efficiency(R_B: float, gamma: float = 5.0 / 3.0) -> float:
    """Fraction of upstream thermal energy converted to directed thrust.

    For a polytropic gas expanding from B_throat to B_exit:

        η_T = 1 − (1/R_B)^((γ−1)/γ)

    Parameters
    ----------
    R_B : float
        Mirror ratio.
    gamma : float
        Polytropic index.
    """
    if R_B <= 1.0:
        return 0.0
    if math.isinf(R_B):
        return 1.0
    return 1.0 - (1.0 / R_B) ** ((gamma - 1.0) / gamma)


def divergence_half_angle(R_B: float) -> float:
    """Plume divergence half-angle estimate from field line geometry [deg].

    In the paraxial approximation, the field-line half-angle at the loss
    cone boundary is sin(θ) = 1/√R_B (mirror loss-cone formula).

    Parameters
    ----------
    R_B : float
        Mirror ratio.

    Returns
    -------
    theta_deg : float
        Half-angle in degrees.
    """
    if R_B <= 1.0:
        return 90.0
    if math.isinf(R_B):
        return 0.0
    return math.degrees(math.asin(1.0 / math.sqrt(R_B)))


def screen_geometry(
    coils: list,
    *,
    z_min: float,
    z_max: float,
    n_pts: int = 200,
    gamma: float = 5.0 / 3.0,
    backend: str = "auto",
) -> NozzleScreeningResult:
    """Run the full Tier 1 analytical screening for a coil geometry.

    Computes mirror ratio from Biot-Savart, then derives all analytical
    performance metrics in < 100 ms.

    Parameters
    ----------
    coils : list of Coil
        Coil definitions.
    z_min, z_max : float
        Axial domain [m].
    n_pts : int
        On-axis evaluation resolution.
    gamma : float
        Electron polytropic index.
    backend : str
        Biot-Savart backend.

    Returns
    -------
    NozzleScreeningResult
    """
    R_B = mirror_ratio(coils, z_min=z_min, z_max=z_max, n_pts=n_pts, backend=backend)
    C_T = thrust_coefficient_paraxial(R_B, gamma=gamma)
    eta_T = thrust_efficiency(R_B, gamma=gamma)
    theta = divergence_half_angle(R_B)

    return NozzleScreeningResult(
        mirror_ratio=R_B,
        thrust_coefficient=C_T,
        divergence_half_angle_deg=theta,
        thrust_efficiency=eta_T,
    )
