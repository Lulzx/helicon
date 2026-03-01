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


@dataclass
class PlumeResult:
    """Plume and beam metrics."""

    divergence_half_angle_deg: float
    beam_efficiency: float
    thrust_coefficient: float
    radial_loss_fraction: float


def compute_plume_metrics(
    output_dir: str | Path,
    *,
    species_name: str = "D_plus",
    species_mass: float = 3.3435837724e-27,
    T_e_eV: float = 2000.0,
    z_exit: float | None = None,
    r_max: float | None = None,
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
    """
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

    # Plume divergence half-angle: cos(θ) = F / ∫ n m |v|² dA
    # Simplified: θ = arctan(⟨|vr|⟩ / ⟨vz⟩) momentum-weighted
    v_total = np.sqrt(vr**2 + vz**2)
    F_axial = np.sum(wt * species_mass * vz**2)
    F_total = np.sum(wt * species_mass * v_total * np.abs(vz))

    if F_total > 0:
        cos_theta = F_axial / F_total
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_deg = float(np.degrees(np.arccos(cos_theta)))
    else:
        theta_deg = 0.0

    # Beam efficiency: η_b = (½ṁv_z²) / (½ṁ|v|²)
    ke_axial = np.sum(wt * vz**2)
    ke_total = np.sum(wt * v_total**2)
    eta_beam = float(ke_axial / ke_total) if ke_total > 0 else 0.0

    # Thrust coefficient: C_T = F / (ṁ c_s)
    # c_s = √(T_e / m_i) — ion sound speed
    eV_to_J = 1.602176634e-19
    c_s = math.sqrt(T_e_eV * eV_to_J / species_mass)
    mdot = np.sum(wt * species_mass * np.abs(vz))
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
    """
    e = 1.602176634e-19
    m_e = 9.1093837139e-31
    eps0 = 8.854187817e-12

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

    return np.where(nu_eff > 0, omega_ce / nu_eff, np.inf)


def compute_pressure_anisotropy(
    P_perp: np.ndarray,
    P_parallel: np.ndarray,
) -> np.ndarray:
    """Compute pressure anisotropy A = P_perp / P_parallel - 1.

    Parameters
    ----------
    P_perp : ndarray
        Perpendicular pressure (P_rr for axisymmetric) [Pa].
    P_parallel : ndarray
        Parallel pressure (P_zz along B) [Pa].
    """
    safe_parallel = np.where(P_parallel > 0, P_parallel, 1e-30)
    return P_perp / safe_parallel - 1.0


def _navigate_openpmd(f: Any) -> Any:
    if "data" in f:
        iterations = sorted(f["data"].keys(), key=int)
        return f["data"][iterations[-1]]
    return f
