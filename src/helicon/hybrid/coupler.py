"""Fluid-hybrid coupler: kinetic ions ↔ CGL electron fluid.

Couples the WarpX kinetic ion moments (density, bulk velocity, pressure
tensor) to the CGL electron fluid state, providing the two-fluid closure
needed for the fluid-hybrid electron path.

The coupling follows a predictor-corrector scheme:
1. Read ion moments from WarpX (or from simulated openPMD data in tests)
2. Compute quasi-neutrality: n_e = sum_s(Z_s * n_s)
3. Advance CGL electron pressures
4. Optionally apply LHDI closure
5. Compute ambipolar potential from electron pressure gradient
6. Return updated electron moments for next WarpX timestep

References
----------
- Bittencourt (2004) — fundamentals of plasma physics (two-fluid theory)
- Toth et al. (2012) — fluid-hybrid methods review
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from helicon._mlx_utils import resolve_backend, to_mx, to_np

_QE = 1.6021766340e-19  # C
_ME = 9.1093837015e-31  # kg


@dataclass
class IonMoments:
    """Ion fluid moments on a grid.

    Attributes
    ----------
    density : ndarray of shape (nr, nz)
        Ion number density [m⁻³] (sum over all species if multi-species)
    vr, vz : ndarray
        Bulk ion velocity components [m/s]
    p_perp, p_par : ndarray
        Ion pressure components [Pa]
    charge_number : int
        Ion charge number Z (default 1 for singly-charged)
    """

    density: np.ndarray
    vr: np.ndarray
    vz: np.ndarray
    p_perp: np.ndarray
    p_par: np.ndarray
    charge_number: int = 1


@dataclass
class HybridState:
    """Combined fluid-hybrid state for one timestep.

    Attributes
    ----------
    n_e : ndarray
        Electron density (= Z * n_i by quasi-neutrality) [m⁻³]
    phi_amb : ndarray
        Ambipolar potential [V]
    E_r, E_z : ndarray
        Ambipolar electric field components [V/m]
    p_perp_e, p_par_e : ndarray
        Updated CGL electron pressures [Pa]
    T_perp_eV, T_par_eV : ndarray
        Electron temperatures [eV]
    """

    n_e: np.ndarray
    phi_amb: np.ndarray
    E_r: np.ndarray
    E_z: np.ndarray
    p_perp_e: np.ndarray
    p_par_e: np.ndarray
    T_perp_eV: np.ndarray
    T_par_eV: np.ndarray


class HybridCoupler:
    """Couples kinetic ion moments to the CGL electron fluid.

    This is the central coupling object for the fluid-hybrid path:
    - Enforces quasi-neutrality: n_e = Z * n_i
    - Advances CGL electron pressures via CGLElectronFluid
    - Computes ambipolar electric field from electron pressure gradient
    - Optionally applies LHDI anomalous transport as a closure

    Parameters
    ----------
    cgl_fluid : CGLElectronFluid
        Pre-initialized CGL electron fluid solver
    lhdi_transport : LHDITransport or None
        Optional LHDI anomalous transport model
    backend : str
        'auto', 'mlx', or 'numpy'
    dr, dz : float
        Grid spacings [m] for field gradient computation
    """

    def __init__(
        self,
        cgl_fluid,
        lhdi_transport=None,
        backend: str = "auto",
        dr: float = 1e-3,
        dz: float = 1e-3,
    ) -> None:
        self.cgl = cgl_fluid
        self.lhdi = lhdi_transport
        self.backend = backend
        self.dr = dr
        self.dz = dz

    def step(
        self,
        ions: IonMoments,
        B_mag: np.ndarray,
        dt: float,
    ) -> HybridState:
        """Advance the fluid-hybrid state by one timestep.

        Parameters
        ----------
        ions : IonMoments
            Current kinetic ion moments from WarpX
        B_mag : ndarray of shape (nr, nz)
            Updated magnetic field magnitude [T]
        dt : float
            Timestep [s]

        Returns
        -------
        HybridState
            Updated hybrid state with electron pressures and ambipolar fields
        """
        # Quasi-neutrality: n_e = Z * n_i
        n_e_new = ions.charge_number * ions.density

        # Advance CGL electron pressures
        cgl_state = self.cgl.update(n_e_new, B_mag, dt)

        # Optionally apply LHDI diffusion to electron density
        if self.lhdi is not None:
            T_e_eV = cgl_state.T_perp_eV  # use T_perp for LHDI drive
            lhdi_params = self.lhdi.compute_lhdi_params(
                n_e_new,
                B_mag,
                T_e_eV,
                T_i_eV=np.maximum(ions.p_perp / (np.maximum(ions.density, 1.0) * _QE), 0.1),
                dr=self.dr,
            )
            n_e_new = self.lhdi.apply_diffusion(n_e_new, lhdi_params.D_eff, dt, self.dr)

        # Compute ambipolar potential from generalized Ohm's law (simplified):
        # E_amb = -(1/n_e * e) * ∇p_e  (electron pressure gradient force)
        phi_amb, E_r, E_z = self._compute_ambipolar_field(
            n_e_new, cgl_state.p_perp, cgl_state.p_par
        )

        n_safe = np.where(n_e_new > 0, n_e_new, 1.0)

        return HybridState(
            n_e=n_e_new,
            phi_amb=phi_amb,
            E_r=E_r,
            E_z=E_z,
            p_perp_e=cgl_state.p_perp,
            p_par_e=cgl_state.p_par,
            T_perp_eV=cgl_state.p_perp / (n_safe * _QE),
            T_par_eV=cgl_state.p_par / (n_safe * _QE),
        )

    def _compute_ambipolar_field(
        self,
        n_e: np.ndarray,
        p_perp: np.ndarray,
        p_par: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ambipolar potential and electric field.

        Generalized Ohm's law (electron momentum, massless limit):
            0 = -e*n_e*E - ∇p_e
            E = -∇p_e / (e * n_e)

        Uses isotropic approximation p_e = (p_perp + p_par/2) / ...
        actually uses p_perp for radial and p_par for axial gradients,
        consistent with field-aligned pressure anisotropy.

        Returns
        -------
        phi_amb : ndarray
            Ambipolar potential [V] (integrated from downstream BC)
        E_r, E_z : ndarray
            Ambipolar electric field [V/m]
        """
        backend = resolve_backend(self.backend)
        n_safe = np.where(n_e > 0, n_e, 1.0)

        if backend == "mlx":
            import mlx.core as mx

            n_mx = to_mx(n_safe)
            p_perp_mx = to_mx(p_perp)
            p_par_mx = to_mx(p_par)

            # Compute gradients in NumPy (finite differences with indexing)
            dp_dr_np = np.gradient(to_np(p_perp_mx), self.dr, axis=0)
            dp_dz_np = np.gradient(to_np(p_par_mx), self.dz, axis=1)

            dp_dr = to_mx(dp_dr_np)
            dp_dz = to_mx(dp_dz_np)

            E_r_mx = -dp_dr / (mx.array(float(_QE)) * n_mx)
            E_z_mx = -dp_dz / (mx.array(float(_QE)) * n_mx)
            mx.eval(E_r_mx, E_z_mx)

            E_r = to_np(E_r_mx)
            E_z = to_np(E_z_mx)
        else:
            dp_dr = np.gradient(p_perp, self.dr, axis=0)
            dp_dz = np.gradient(p_par, self.dz, axis=1)
            E_r = -dp_dr / (_QE * n_safe)
            E_z = -dp_dz / (_QE * n_safe)

        # Ambipolar potential: integrate E_z from downstream boundary
        phi_amb = -np.cumsum(E_z * self.dz, axis=1)

        return phi_amb, E_r, E_z
