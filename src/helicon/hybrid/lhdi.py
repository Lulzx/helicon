"""Lower-Hybrid Drift Instability (LHDI) anomalous transport sub-grid model.

Implements cross-field electron transport driven by the LHDI as a
sub-grid closure on the fluid-hybrid electron path (spec v1.2).

The LHDI grows at density gradients ∇n perpendicular to B and drives
anomalous cross-field diffusion of electrons. It is the dominant
non-classical transport mechanism in magnetic nozzle edge regions.

This model computes an effective cross-field diffusion coefficient
D_eff from local plasma parameters and applies it as a correction
to the CGL electron fluid state.

Physics:
    ω_LH = ω_ci * sqrt(1 + ω_pe^2 / ω_ce^2)  ≈ sqrt(ω_ci * ω_ce)
    γ_LHDI ~ ω_LH * (ε_d) / (1 + ε_d)  where ε_d = ρ_i * |∇n|/n
    D_LHDI ~ γ_LHDI * ρ_e^2  (Bohm-like scaling)

References
----------
- Davidson & Gladd (1975) — original LHDI theory
- Huba et al. (1978) — saturation and anomalous transport
- Janhunen et al. (2018) — LHDI in magnetic nozzle context
- Choueiri (2001) — review of anomalous transport in EPT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from helicon._mlx_utils import HAS_MLX, resolve_backend, to_mx, to_np

_QE = 1.6021766340e-19   # C
_ME = 9.1093837015e-31   # kg
_MP = 1.6726219236951e-27  # kg
_MU0 = 4.0 * np.pi * 1e-7  # H/m
_EPS0 = 8.8541878128e-12   # F/m


@dataclass
class LHDIParams:
    """Local LHDI parameters at a grid point.

    Attributes
    ----------
    omega_lh : ndarray
        Lower-hybrid frequency [rad/s]
    gamma_lhdi : ndarray
        LHDI growth rate [s⁻¹]
    D_eff : ndarray
        Effective cross-field diffusion coefficient [m²/s]
    kappa_n : ndarray
        Normalized density gradient scale length |∇n|/n [m⁻¹]
    """

    omega_lh: np.ndarray
    gamma_lhdi: np.ndarray
    D_eff: np.ndarray
    kappa_n: np.ndarray


class LHDITransport:
    """LHDI anomalous cross-field transport model.

    Computes the effective diffusion coefficient for cross-field electron
    transport due to the LHDI and applies it as a density diffusion term.

    Parameters
    ----------
    ion_mass_kg : float
        Ion mass [kg]
    backend : str
        'auto', 'mlx', or 'numpy'
    saturation_factor : float
        Saturation multiplier α such that D_eff = α * γ_LHDI * ρ_e².
        Default 0.1 from Huba et al. calibration.
    """

    def __init__(
        self,
        ion_mass_kg: float = _MP,
        backend: str = "auto",
        saturation_factor: float = 0.1,
    ) -> None:
        self.m_i = ion_mass_kg
        self.backend = backend
        self.alpha = saturation_factor

    def compute_lhdi_params(
        self,
        n_e: np.ndarray,
        B_mag: np.ndarray,
        T_e_eV: np.ndarray,
        T_i_eV: float | np.ndarray,
        dr: float,
    ) -> LHDIParams:
        """Compute LHDI growth rate and diffusion coefficient on grid.

        Parameters
        ----------
        n_e : ndarray of shape (nr, nz)
            Electron density [m⁻³]
        B_mag : ndarray of shape (nr, nz)
            Magnetic field magnitude [T]
        T_e_eV : ndarray
            Electron temperature [eV]
        T_i_eV : float or ndarray
            Ion temperature [eV]
        dr : float
            Radial grid spacing [m] for computing ∇n

        Returns
        -------
        LHDIParams
        """
        backend = resolve_backend(self.backend)
        if backend == "mlx":
            return self._compute_mlx(n_e, B_mag, T_e_eV, T_i_eV, dr)
        return self._compute_numpy(n_e, B_mag, T_e_eV, T_i_eV, dr)

    def _compute_numpy(
        self,
        n_e: np.ndarray,
        B_mag: np.ndarray,
        T_e_eV: np.ndarray,
        T_i_eV: float | np.ndarray,
        dr: float,
    ) -> LHDIParams:
        """NumPy LHDI parameter computation."""
        B_safe = np.where(B_mag > 0, B_mag, 1e-10)
        n_safe = np.where(n_e > 0, n_e, 1e6)

        # Electron/ion cyclotron frequencies
        omega_ce = _QE * B_safe / _ME  # [rad/s]
        omega_ci = _QE * B_safe / self.m_i  # [rad/s]

        # Electron plasma frequency
        omega_pe = np.sqrt(n_safe * _QE**2 / (_EPS0 * _ME))  # [rad/s]

        # Lower-hybrid frequency: ω_LH ≈ sqrt(ω_ci * ω_ce) for ω_pe >> ω_ce
        # Full expression: ω_LH = ω_ci / sqrt(1 + ω_ci^2/ω_ce^2 + ω_pe^2/ω_ce^2)
        # Approximate: ω_LH ≈ sqrt(ω_ci * ω_ce) when ω_pe >> ω_ce
        omega_lh = np.sqrt(omega_ci * omega_ce / (1.0 + omega_pe**2 / omega_ce**2 + omega_ci / omega_ce))

        # Electron Larmor radius: ρ_e = v_th_e / ω_ce
        T_e_J = np.asarray(T_e_eV, dtype=float) * _QE
        v_th_e = np.sqrt(2.0 * T_e_J / _ME)
        rho_e = v_th_e / np.maximum(omega_ce, 1.0)

        # Ion Larmor radius: ρ_i = v_th_i / ω_ci
        T_i_J = np.asarray(T_i_eV, dtype=float) * _QE
        v_th_i = np.sqrt(2.0 * T_i_J / self.m_i)
        rho_i = v_th_i / np.maximum(omega_ci, 1.0)

        # Density gradient scale length κ_n = |∇n|/n  [m⁻¹]
        # Radial gradient dominates in nozzle geometry
        dn_dr = np.gradient(n_safe, dr, axis=0)
        kappa_n = np.abs(dn_dr) / n_safe

        # LHDI drive parameter: ε_d = ρ_i * κ_n
        eps_d = rho_i * kappa_n

        # LHDI growth rate (Davidson & Gladd 1975 simplified):
        # γ_LHDI ~ ω_LH * ε_d / (1 + ε_d)
        gamma_lhdi = omega_lh * eps_d / (1.0 + eps_d)

        # Effective diffusion coefficient: D_eff = α * γ_LHDI * ρ_e^2
        D_eff = self.alpha * gamma_lhdi * rho_e**2

        return LHDIParams(
            omega_lh=omega_lh,
            gamma_lhdi=gamma_lhdi,
            D_eff=D_eff,
            kappa_n=kappa_n,
        )

    def _compute_mlx(
        self,
        n_e: np.ndarray,
        B_mag: np.ndarray,
        T_e_eV: np.ndarray,
        T_i_eV: float | np.ndarray,
        dr: float,
    ) -> LHDIParams:
        """MLX-accelerated LHDI computation on Metal GPU."""
        import mlx.core as mx

        B_mx = mx.maximum(to_mx(B_mag), mx.array(1e-10))
        n_mx = mx.maximum(to_mx(n_e), mx.array(1e6))

        qe = float(_QE)
        me = float(_ME)
        mi = float(self.m_i)
        eps0 = float(_EPS0)

        omega_ce = qe * B_mx / me
        omega_ci = qe * B_mx / mi
        omega_pe = mx.sqrt(n_mx * qe**2 / (eps0 * me))

        omega_lh = mx.sqrt(
            omega_ci * omega_ce / (
                mx.array(1.0) + omega_pe * omega_pe / (omega_ce * omega_ce) + omega_ci / omega_ce
            )
        )

        T_e_J = float(np.mean(T_e_eV)) * qe if np.ndim(T_e_eV) == 0 else None
        if T_e_J is None:
            T_e_J_arr = to_mx(np.asarray(T_e_eV, dtype=float) * qe)
        else:
            T_e_J_arr = mx.ones_like(B_mx) * mx.array(T_e_J)

        T_i_J = float(np.mean(T_i_eV)) * qe

        v_th_e = mx.sqrt(mx.array(2.0) * T_e_J_arr / me)
        rho_e = v_th_e / mx.maximum(omega_ce, mx.array(1.0))

        v_th_i = float(np.sqrt(2.0 * T_i_J / mi))
        rho_i = mx.array(v_th_i) / mx.maximum(omega_ci, mx.array(1.0))

        # Density gradient — computed in NumPy (indexing-heavy)
        n_np = to_np(n_mx)
        dn_dr = np.gradient(n_np, dr, axis=0)
        kappa_n_np = np.abs(dn_dr) / np.where(n_np > 0, n_np, 1.0)
        kappa_n_mx = to_mx(kappa_n_np)

        eps_d = rho_i * kappa_n_mx
        gamma_lhdi = omega_lh * eps_d / (mx.array(1.0) + eps_d)
        D_eff = mx.array(self.alpha) * gamma_lhdi * rho_e * rho_e

        mx.eval(omega_lh, gamma_lhdi, D_eff)

        return LHDIParams(
            omega_lh=to_np(omega_lh),
            gamma_lhdi=to_np(gamma_lhdi),
            D_eff=to_np(D_eff),
            kappa_n=kappa_n_np,
        )

    def apply_diffusion(
        self,
        n_e: np.ndarray,
        D_eff: np.ndarray,
        dt: float,
        dr: float,
    ) -> np.ndarray:
        """Apply LHDI cross-field diffusion to electron density.

        Solves ∂n/∂t = ∇_⊥ · (D_eff ∇_⊥ n) for one timestep using
        explicit finite differences. For stability, dt < dr^2 / (2 * D_max).

        Parameters
        ----------
        n_e : ndarray of shape (nr, nz)
            Electron density [m⁻³]
        D_eff : ndarray of shape (nr, nz)
            Diffusion coefficient [m²/s]
        dt : float
            Timestep [s]
        dr : float
            Radial grid spacing [m]

        Returns
        -------
        ndarray
            Updated electron density [m⁻³]
        """
        nr, nz = n_e.shape

        # Radial diffusion only (cross-field in 2D-RZ)
        D_avg = 0.5 * (D_eff[:-1, :] + D_eff[1:, :])  # (nr-1, nz)
        flux_plus = D_avg * (n_e[1:, :] - n_e[:-1, :]) / dr  # inward flux
        flux_minus = np.zeros_like(n_e)
        flux_minus[1:, :] = flux_plus
        flux_plus_padded = np.zeros_like(n_e)
        flux_plus_padded[:-1, :] = flux_plus

        dn = (flux_plus_padded - flux_minus) / dr
        n_new = n_e + dt * dn

        return np.maximum(n_new, 0.0)
