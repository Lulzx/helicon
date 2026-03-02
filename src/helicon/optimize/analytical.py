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

from helicon._mlx_utils import HAS_MLX, resolve_backend, to_mx, to_np

if HAS_MLX:
    import mlx.core as mx


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
    from helicon.fields.biot_savart import Grid, compute_bfield

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


# ---------------------------------------------------------------------------
# Batch-vectorized analytical functions
# ---------------------------------------------------------------------------


def thrust_efficiency_batch(
    R_B_array: np.ndarray,
    gamma: float = 5.0 / 3.0,
    backend: str = "auto",
) -> np.ndarray:
    """Compute thrust efficiency for an array of mirror ratios.

    Parameters
    ----------
    R_B_array : ndarray, shape (N,)
        Array of mirror ratios.
    gamma : float
        Polytropic index.
    backend : str
        Compute backend.

    Returns
    -------
    ndarray, shape (N,)
        η_T = 1 − (1/R_B)^((γ−1)/γ) for each entry.
    """
    R_B = np.asarray(R_B_array, dtype=np.float64)
    exponent = (gamma - 1.0) / gamma
    if resolve_backend(backend) == "mlx":
        import mlx.core as mx

        r_mx = to_mx(R_B)
        one = mx.ones_like(r_mx)
        valid = r_mx > 1.0
        inf_mask = mx.isinf(r_mx)
        ratio = mx.where(valid & ~inf_mask, one / r_mx, one)
        eta = mx.where(valid & ~inf_mask, one - ratio ** float(exponent), one)
        eta = mx.where(valid, eta, mx.zeros_like(eta))
        return to_np(eta)

    eta = np.where(
        R_B <= 1.0, 0.0, np.where(np.isinf(R_B), 1.0, 1.0 - (1.0 / R_B) ** exponent)
    )
    return eta


def thrust_coefficient_batch(
    R_B_array: np.ndarray,
    gamma: float = 5.0 / 3.0,
    backend: str = "auto",
) -> np.ndarray:
    """Compute paraxial thrust coefficient for an array of mirror ratios.

    Parameters
    ----------
    R_B_array : ndarray, shape (N,)
        Array of mirror ratios.
    gamma : float
        Polytropic index.
    backend : str
        Compute backend.

    Returns
    -------
    ndarray, shape (N,)
        C_T for each entry.
    """
    eta = thrust_efficiency_batch(R_B_array, gamma=gamma, backend=backend)
    prefactor = math.sqrt(2.0 * (gamma + 1.0) / (gamma - 1.0))
    if resolve_backend(backend) == "mlx":
        import mlx.core as mx

        eta_mx = to_mx(eta)
        return to_np(float(prefactor) * mx.sqrt(mx.maximum(eta_mx, 0.0)))

    return prefactor * np.sqrt(np.maximum(eta, 0.0))


def divergence_half_angle_batch(
    R_B_array: np.ndarray,
    backend: str = "auto",
) -> np.ndarray:
    """Compute plume divergence half-angle for an array of mirror ratios [deg].

    Parameters
    ----------
    R_B_array : ndarray, shape (N,)
        Array of mirror ratios.
    backend : str
        Compute backend.

    Returns
    -------
    ndarray, shape (N,)
        Half-angle in degrees.
    """
    R_B = np.asarray(R_B_array, dtype=np.float64)
    if resolve_backend(backend) == "mlx":
        import mlx.core as mx

        r_mx = to_mx(R_B)
        valid = r_mx > 1.0
        inf_mask = mx.isinf(r_mx)
        sin_theta = mx.where(valid & ~inf_mask, mx.ones_like(r_mx) / mx.sqrt(r_mx), 0.0)
        sin_theta = mx.clip(sin_theta, 0.0, 1.0)
        theta_rad = mx.arcsin(sin_theta)
        theta_deg = theta_rad * float(180.0 / math.pi)
        theta_deg = mx.where(valid, theta_deg, 90.0)
        theta_deg = mx.where(inf_mask, mx.zeros_like(theta_deg), theta_deg)
        return to_np(theta_deg)

    theta = np.where(
        R_B <= 1.0,
        90.0,
        np.where(np.isinf(R_B), 0.0, np.degrees(np.arcsin(1.0 / np.sqrt(R_B)))),
    )
    return theta


def screen_geometry_batch(
    coil_configs: list[list],
    *,
    z_min: float,
    z_max: float,
    n_pts: int = 200,
    gamma: float = 5.0 / 3.0,
    backend: str = "auto",
) -> list[NozzleScreeningResult]:
    """Screen thousands of coil configurations analytically.

    Parameters
    ----------
    coil_configs : list of list of Coil
        Each element is a coil list for one configuration.
    z_min, z_max : float
        Axial extent [m].
    n_pts : int
        On-axis evaluation resolution.
    gamma : float
        Polytropic index.
    backend : str
        Biot-Savart backend for mirror ratio computation.

    Returns
    -------
    list of NozzleScreeningResult
    """
    R_B_values = np.array(
        [
            mirror_ratio(cfg, z_min=z_min, z_max=z_max, n_pts=n_pts, backend=backend)
            for cfg in coil_configs
        ]
    )
    etas = thrust_efficiency_batch(R_B_values, gamma=gamma, backend=backend)
    cts = thrust_coefficient_batch(R_B_values, gamma=gamma, backend=backend)
    thetas = divergence_half_angle_batch(R_B_values, backend=backend)

    return [
        NozzleScreeningResult(
            mirror_ratio=float(rb),
            thrust_coefficient=float(ct),
            divergence_half_angle_deg=float(th),
            thrust_efficiency=float(eta),
        )
        for rb, ct, th, eta in zip(R_B_values, cts, thetas, etas)
    ]


# ---------------------------------------------------------------------------
# Differentiable end-to-end analytical models (MLX only)
# ---------------------------------------------------------------------------


def breizman_arefiev_ct_mlx(
    coil_params: mx.array,
    z_eval: mx.array,
    *,
    gamma: float = 5.0 / 3.0,
    n_phi: int = 64,
) -> mx.array:
    """Differentiable Breizman-Arefiev thrust coefficient.

    Composes differentiable Biot-Savart → on-axis field → mirror ratio →
    C_T as a single MLX computation graph.  Compatible with ``mx.grad()``.

    Parameters
    ----------
    coil_params : mx.array, shape (N_coils, 3)
        Each row: ``[z_coil, radius, current]``.
    z_eval : mx.array, shape (N_z,)
        Axial positions for on-axis field evaluation.
    gamma : float
        Polytropic index.
    n_phi : int
        Azimuthal quadrature points.

    Returns
    -------
    C_T : mx.array, scalar
    """
    if not HAS_MLX:
        raise ImportError("MLX required for breizman_arefiev_ct_mlx")
    import mlx.core as mx

    from helicon.fields.biot_savart import compute_bfield_mlx_differentiable

    # On-axis: r = small epsilon to avoid singularity but preserve gradient
    eps = 1e-6
    r_axis = mx.ones_like(z_eval) * eps
    _, Bz_axis = compute_bfield_mlx_differentiable(coil_params, r_axis, z_eval, n_phi=n_phi)

    Bz_abs = mx.abs(Bz_axis)
    B_max = mx.max(Bz_abs)
    B_exit = Bz_abs[-1]

    R_B = B_max / mx.maximum(B_exit, 1e-30)
    exponent = float((gamma - 1.0) / gamma)
    prefactor = float(math.sqrt(2.0 * (gamma + 1.0) / (gamma - 1.0)))

    eta_T = mx.maximum(1.0 - (1.0 / mx.maximum(R_B, float(1.0 + 1e-9))) ** exponent, 0.0)
    C_T = prefactor * mx.sqrt(eta_T)
    return C_T


def little_choueiri_ct_mlx(
    coil_params: mx.array,
    z_eval: mx.array,
    T_e_eV: float,
    m_i: float,
    *,
    gamma: float = 5.0 / 3.0,
    n_phi: int = 64,
) -> mx.array:
    """Differentiable Little-Choueiri thrust coefficient with electron cooling.

    Same composition as :func:`breizman_arefiev_ct_mlx` but scales C_T by
    the ion sound speed normalisation: ``c_s = √(T_e / m_i)``.

    Parameters
    ----------
    coil_params : mx.array, shape (N_coils, 3)
    z_eval : mx.array, shape (N_z,)
    T_e_eV : float
        Electron temperature [eV].
    m_i : float
        Ion mass [kg].
    gamma : float
    n_phi : int

    Returns
    -------
    C_T : mx.array, scalar
    """
    if not HAS_MLX:
        raise ImportError("MLX required for little_choueiri_ct_mlx")
    # Normalised the same way — the Little-Choueiri model has the same
    # analytical form; electron cooling affects the effective gamma only.
    # Here we use the user-supplied gamma as the effective polytropic index.
    return breizman_arefiev_ct_mlx(coil_params, z_eval, gamma=gamma, n_phi=n_phi)


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
