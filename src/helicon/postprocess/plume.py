"""Plume divergence, beam efficiency, and thrust coefficient.

Computes exhaust plume metrics from particle data at the exit plane:
- Plume divergence half-angle θ
- Beam efficiency η_b
- Thrust coefficient C_T
- Radial loss fraction
- Electron magnetization parameter Ω_e
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from helicon._mlx_utils import HAS_MLX, resolve_backend, to_mx, to_np


@dataclass
class PlumeResult:
    """Plume and beam metrics."""

    divergence_half_angle_deg: float
    beam_efficiency: float
    thrust_coefficient: float
    radial_loss_fraction: float


def _plume_reduce_mlx(
    wt: np.ndarray,
    species_mass: float,
    vz: np.ndarray,
    vr: np.ndarray,
    c_s: float,
) -> tuple[float, float, float, float]:
    """Compute plume force sums on Metal GPU via MLX.

    Returns
    -------
    F_axial, F_total, ke_axial, ke_total, mdot :
        Force and kinetic energy sums.
    """
    if not HAS_MLX:
        raise ImportError("MLX required for _plume_reduce_mlx")
    import mlx.core as mx

    wt_mx = to_mx(wt)
    vz_mx = to_mx(vz)
    vr_mx = to_mx(vr)
    m = float(species_mass)

    v_total_mx = mx.sqrt(vr_mx * vr_mx + vz_mx * vz_mx)
    F_axial = float(to_np(mx.sum(wt_mx * m * vz_mx * vz_mx)))
    F_total = float(to_np(mx.sum(wt_mx * m * v_total_mx * mx.abs(vz_mx))))
    ke_axial = float(to_np(mx.sum(wt_mx * vz_mx * vz_mx)))
    ke_total = float(to_np(mx.sum(wt_mx * (v_total_mx * v_total_mx))))
    mdot = float(to_np(mx.sum(wt_mx * m * mx.abs(vz_mx))))
    return F_axial, F_total, ke_axial, ke_total, mdot


def compute_plume_metrics(
    output_dir: str | Path,
    *,
    species_name: str = "D_plus",
    species_mass: float = 3.3435837724e-27,
    T_e_eV: float = 2000.0,
    z_exit: float | None = None,
    r_max: float | None = None,
    backend: str = "auto",
) -> PlumeResult:
    """Compute plume divergence and beam efficiency from particle data.

    Parameters
    ----------
    output_dir : path
        WarpX output directory.
    species_name : str
        Species to analyze.
    species_mass : float
        Species mass [kg].
    T_e_eV : float
        Electron temperature at throat [eV], for C_T calculation.
    z_exit : float, optional
        Exit plane z [m].
    r_max : float, optional
        Radial domain boundary [m].
    backend : ``"auto"`` | ``"mlx"`` | ``"numpy"``
        Compute backend for force and kinetic-energy reductions.
    """
    use_mlx = resolve_backend(backend) == "mlx"
    import h5py

    output_dir = Path(output_dir)
    h5_files = sorted(output_dir.glob("**/*.h5"))
    if not h5_files:
        msg = f"No HDF5 files found in {output_dir}"
        raise FileNotFoundError(msg)

    with h5py.File(h5_files[-1], "r") as f:
        base = _navigate_openpmd(f)
        if "particles" not in base or species_name not in base["particles"]:
            return PlumeResult(
                divergence_half_angle_deg=0.0,
                beam_efficiency=0.0,
                thrust_coefficient=0.0,
                radial_loss_fraction=0.0,
            )

        sp = base["particles"][species_name]
        z = sp["position"]["z"][:]
        r = sp["position"]["r"][:] if "r" in sp["position"] else np.zeros_like(z)
        pz = sp["momentum"]["z"][:]
        pr = sp["momentum"]["r"][:] if "r" in sp["momentum"] else np.zeros_like(pz)
        w = sp["weighting"][:] if "weighting" in sp else np.ones_like(z)

    if z_exit is None:
        z_exit = z.max() * 0.9
    if r_max is None:
        r_max = r.max() * 0.95

    n_total = np.sum(w)

    # Select particles near exit plane
    dz_tol = (z.max() - z.min()) / 100.0
    near_exit = np.abs(z - z_exit) < dz_tol

    if np.sum(near_exit) == 0:
        return PlumeResult(
            divergence_half_angle_deg=0.0,
            beam_efficiency=0.0,
            thrust_coefficient=0.0,
            radial_loss_fraction=0.0,
        )

    vz = pz[near_exit] / species_mass
    vr = pr[near_exit] / species_mass
    wt = w[near_exit]

    eV_to_J = 1.602176634e-19
    c_s = math.sqrt(T_e_eV * eV_to_J / species_mass)

    if use_mlx and len(wt) > 0:
        F_axial, F_total, ke_axial, ke_total, mdot = _plume_reduce_mlx(
            wt, species_mass, vz, vr, c_s
        )
    else:
        v_total = np.sqrt(vr**2 + vz**2)
        F_axial = float(np.sum(wt * species_mass * vz**2))
        F_total = float(np.sum(wt * species_mass * v_total * np.abs(vz)))
        ke_axial = float(np.sum(wt * vz**2))
        ke_total = float(np.sum(wt * v_total**2))
        mdot = float(np.sum(wt * species_mass * np.abs(vz)))

    if F_total > 0:
        cos_theta = F_axial / F_total
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        theta_deg = float(np.degrees(np.arccos(cos_theta)))
    else:
        theta_deg = 0.0

    # Beam efficiency: η_b = (½ṁv_z²) / (½ṁ|v|²)
    eta_beam = float(ke_axial / ke_total) if ke_total > 0 else 0.0

    # Thrust coefficient: C_T = F / (ṁ c_s)
    C_T = float(F_axial / (mdot * c_s)) if mdot > 0 and c_s > 0 else 0.0

    # Radial loss fraction
    radial_lost = r >= r_max
    radial_fraction = float(np.sum(w[radial_lost]) / n_total) if n_total > 0 else 0.0

    return PlumeResult(
        divergence_half_angle_deg=theta_deg,
        beam_efficiency=eta_beam,
        thrust_coefficient=C_T,
        radial_loss_fraction=radial_fraction,
    )


def compute_electron_magnetization(
    Br: np.ndarray,
    Bz: np.ndarray,
    n_e: np.ndarray,
    T_e_eV: float,
    *,
    collision_freq: float | None = None,
    backend: str = "auto",
) -> np.ndarray:
    """Compute electron magnetization parameter Ω_e on the grid.

    Omega_e = omega_ce / nu_eff, where omega_ce is the electron cyclotron
    frequency and nu_eff is the effective collision frequency.

    Detachment occurs where Ω_e ~ 1.

    Parameters
    ----------
    Br, Bz : ndarray, shape (nr, nz)
        Magnetic field components [T].
    n_e : ndarray, shape (nr, nz)
        Electron density [m^-3].
    T_e_eV : float
        Electron temperature [eV].
    collision_freq : float, optional
        Fixed collision frequency [Hz]. If None, uses Coulomb collision
        estimate.
    backend : ``"auto"`` | ``"mlx"`` | ``"numpy"``
        Compute backend for grid-array operations.
    """
    use_mlx = resolve_backend(backend) == "mlx"

    e = 1.602176634e-19
    m_e = 9.1093837139e-31
    eps0 = 8.854187817e-12

    if use_mlx:
        import mlx.core as mx

        Br_mx = to_mx(Br)
        Bz_mx = to_mx(Bz)
        ne_mx = to_mx(n_e)
        B_mag_mx = mx.sqrt(Br_mx * Br_mx + Bz_mx * Bz_mx)
        omega_ce_mx = float(e) / float(m_e) * B_mag_mx

        if collision_freq is not None:
            nu_mx = mx.ones_like(B_mag_mx) * float(collision_freq)
        else:
            T_J = T_e_eV * e
            v_th = math.sqrt(T_J / m_e)
            ln_lambda = 10.0
            coeff = float(e**4 * ln_lambda / (4 * math.pi * eps0**2 * m_e**2 * v_th**3))
            nu_mx = mx.where(ne_mx > 0, ne_mx * coeff, 1e-10)

        result_mx = mx.where(nu_mx > 0, omega_ce_mx / nu_mx, float("inf"))
        return to_np(result_mx)

    B_mag = np.sqrt(Br**2 + Bz**2)
    omega_ce = e * B_mag / m_e  # cyclotron frequency

    if collision_freq is not None:
        nu_eff = collision_freq
    else:
        # Coulomb collision frequency estimate:
        # nu_ei ~ n_e e^4 ln(Lambda) / (4 pi eps0^2 m_e^2 v_th^3)
        T_J = T_e_eV * e
        v_th = np.sqrt(T_J / m_e)
        ln_lambda = 10.0  # typical Coulomb logarithm
        nu_eff = np.where(
            n_e > 0,
            n_e * e**4 * ln_lambda / (4 * np.pi * eps0**2 * m_e**2 * v_th**3),
            1e-10,
        )

    safe_nu = np.where(nu_eff > 0, nu_eff, 1.0)
    return np.where(nu_eff > 0, omega_ce / safe_nu, np.inf)


def compute_pressure_anisotropy(
    P_perp: np.ndarray,
    P_parallel: np.ndarray,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Compute pressure anisotropy A = P_perp / P_parallel - 1.

    Parameters
    ----------
    P_perp : ndarray
        Perpendicular pressure (P_rr for axisymmetric) [Pa].
    P_parallel : ndarray
        Parallel pressure (P_zz along B) [Pa].
    backend : ``"auto"`` | ``"mlx"`` | ``"numpy"``
        Compute backend.
    """
    if resolve_backend(backend) == "mlx":
        import mlx.core as mx

        pp_mx = to_mx(P_perp)
        par_mx = to_mx(P_parallel)
        safe = mx.where(par_mx > 0, par_mx, 1e-30)
        return to_np(pp_mx / safe - 1.0)

    safe_parallel = np.where(P_parallel > 0, P_parallel, 1e-30)
    return P_perp / safe_parallel - 1.0


def _navigate_openpmd(f: Any) -> Any:
    if "data" in f:
        iterations = sorted(f["data"].keys(), key=int)
        return f["data"][iterations[-1]]
    return f
