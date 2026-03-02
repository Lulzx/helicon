"""Tests for helicon.hybrid — CGL electron fluid + LHDI (spec v1.2)."""

from __future__ import annotations

import numpy as np
import pytest

from helicon.hybrid.cgl_electron import CGLElectronFluid, CGLState
from helicon.hybrid.coupler import HybridCoupler, IonMoments
from helicon.hybrid.lhdi import LHDITransport

_QE = 1.6021766340e-19
_MP = 1.6726219236951e-27


# ---------------------------------------------------------------------------
# CGLState
# ---------------------------------------------------------------------------


class TestCGLState:
    def test_from_isotropic(self):
        nr, nz = 10, 20
        n_e = np.ones((nr, nz)) * 1e18
        B_mag = np.ones((nr, nz)) * 0.1
        state = CGLState.from_isotropic(n_e, T_eV=10.0, B_mag=B_mag)
        assert state.n_e.shape == (nr, nz)
        assert state.p_perp.shape == (nr, nz)
        assert state.p_par.shape == (nr, nz)
        np.testing.assert_allclose(state.p_perp, state.p_par)

    def test_temperature_properties(self):
        nr, nz = 5, 5
        n_e = np.ones((nr, nz)) * 1e18
        B_mag = np.ones((nr, nz)) * 0.1
        T_eV = 10.0
        state = CGLState.from_isotropic(n_e, T_eV, B_mag)
        np.testing.assert_allclose(state.T_perp_eV, T_eV, rtol=1e-6)
        np.testing.assert_allclose(state.T_par_eV, T_eV, rtol=1e-6)

    def test_anisotropy_initially_zero(self):
        nr, nz = 5, 5
        n_e = np.ones((nr, nz)) * 1e18
        B_mag = np.ones((nr, nz)) * 0.1
        state = CGLState.from_isotropic(n_e, 10.0, B_mag)
        np.testing.assert_allclose(state.anisotropy, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# CGLElectronFluid — adiabatic invariants
# ---------------------------------------------------------------------------


class TestCGLElectronFluid:
    def _make_fluid(self, nr=10, nz=20, T_eV=10.0, B0=0.1):
        n_e = np.ones((nr, nz)) * 1e18
        B_mag = np.ones((nr, nz)) * B0
        state = CGLState.from_isotropic(n_e, T_eV, B_mag)
        return CGLElectronFluid(state, backend="numpy")

    def test_uniform_B_preserves_isotropic(self):
        """If B and n are uniform, CGL update should preserve isotropy."""
        fluid = self._make_fluid()
        state0 = fluid.state
        T_perp0 = state0.T_perp_eV[0, 0]

        # Same B, same n → no change
        n_new = np.ones((10, 20)) * 1e18
        B_new = np.ones((10, 20)) * 0.1
        state_new = fluid.update(n_new, B_new, dt=1e-9)

        np.testing.assert_allclose(state_new.T_perp_eV, T_perp0, rtol=1e-6)
        np.testing.assert_allclose(state_new.T_par_eV, T_perp0, rtol=1e-6)

    def test_perp_pressure_increases_with_B(self):
        """p_perp should increase when B increases (magnetic mirror heating)."""
        fluid = self._make_fluid(T_eV=10.0, B0=0.1)
        n_new = np.ones((10, 20)) * 1e18
        B_new = np.ones((10, 20)) * 0.2  # double B

        state_new = fluid.update(n_new, B_new, dt=1e-9)
        # μ_1 = p_perp/(n*B) = const → p_perp_new = μ_1 * n * B_new = 2× p_perp_old
        expected_T_perp = 20.0  # eV
        np.testing.assert_allclose(state_new.T_perp_eV, expected_T_perp, rtol=1e-5)

    def test_par_pressure_increases_with_density(self):
        """p_par should increase when density increases."""
        fluid = self._make_fluid(T_eV=10.0, B0=0.1)
        n_new = np.ones((10, 20)) * 2e18  # double density
        B_new = np.ones((10, 20)) * 0.1

        state_new = fluid.update(n_new, B_new, dt=1e-9)
        # J_2 = p_par * B^2 / n^3 = const → p_par scales as n^3
        # T_par = p_par / (n * k) scales as n^2
        expected_T_par = 10.0 * 4.0  # eV (2^2 = 4)
        np.testing.assert_allclose(state_new.T_par_eV, expected_T_par, rtol=1e-5)

    def test_isotropization_relaxes_anisotropy(self):
        """With large nu_iso and long dt, anisotropy should decay toward zero."""
        nr, nz = 5, 5
        n_e = np.ones((nr, nz)) * 1e18
        B_mag = np.ones((nr, nz)) * 0.1
        state = CGLState.from_isotropic(n_e, 10.0, B_mag)

        # Manually create anisotropic state
        state.p_perp[:] = state.p_perp * 3.0  # 3× perpendicular

        fluid = CGLElectronFluid(state, backend="numpy", isotropization_rate=1e10)

        B_new = np.ones((nr, nz)) * 0.1
        state_new = fluid.update(n_e, B_new, dt=1.0)  # 1 second → fully relaxed

        # After strong isotropization, anisotropy should be small
        assert np.max(np.abs(state_new.anisotropy)) < 1.0

    def test_positivity_enforced(self):
        """Pressures must remain non-negative."""
        fluid = self._make_fluid()
        n_new = np.ones((10, 20)) * 0.0  # zero density
        B_new = np.ones((10, 20)) * 0.1
        state_new = fluid.update(n_new, B_new, dt=1e-9)
        assert np.all(state_new.p_perp >= 0)
        assert np.all(state_new.p_par >= 0)

    def test_heat_flux_returns_arrays(self):
        fluid = self._make_fluid()
        q_par, q_perp = fluid.compute_heat_flux()
        assert q_par.shape == (10, 20)
        assert q_perp.shape == (10, 20)

    def test_pressure_tensor(self):
        fluid = self._make_fluid()
        P_rr, P_phi, P_zz = fluid.electron_pressure_tensor()
        np.testing.assert_array_equal(P_rr, P_phi)  # perp symmetry
        # P_zz = p_par may differ from P_rr if anisotropic


# ---------------------------------------------------------------------------
# LHDITransport
# ---------------------------------------------------------------------------


class TestLHDITransport:
    def _make_grid(self, nr=20, nz=40):
        n_e = np.ones((nr, nz)) * 1e18
        # Add radial gradient: n peaks at axis
        r = np.linspace(0.01, 0.5, nr)
        n_e = n_e * np.exp(-r[:, None] / 0.1)
        B_mag = np.ones((nr, nz)) * 0.05
        T_e_eV = np.ones((nr, nz)) * 10.0
        return n_e, B_mag, T_e_eV

    def test_compute_lhdi_params_numpy(self):
        lhdi = LHDITransport(ion_mass_kg=2.0 * _MP, backend="numpy")
        n_e, B_mag, T_e_eV = self._make_grid()
        params = lhdi.compute_lhdi_params(n_e, B_mag, T_e_eV, T_i_eV=5.0, dr=0.025)

        assert params.omega_lh.shape == n_e.shape
        assert params.gamma_lhdi.shape == n_e.shape
        assert params.D_eff.shape == n_e.shape
        assert params.kappa_n.shape == n_e.shape

    def test_omega_lh_positive(self):
        lhdi = LHDITransport(ion_mass_kg=2.0 * _MP, backend="numpy")
        n_e, B_mag, T_e_eV = self._make_grid()
        params = lhdi.compute_lhdi_params(n_e, B_mag, T_e_eV, T_i_eV=5.0, dr=0.025)
        assert np.all(params.omega_lh > 0)

    def test_gamma_lhdi_nonneg(self):
        lhdi = LHDITransport(ion_mass_kg=2.0 * _MP, backend="numpy")
        n_e, B_mag, T_e_eV = self._make_grid()
        params = lhdi.compute_lhdi_params(n_e, B_mag, T_e_eV, T_i_eV=5.0, dr=0.025)
        assert np.all(params.gamma_lhdi >= 0)

    def test_D_eff_nonneg(self):
        lhdi = LHDITransport(ion_mass_kg=2.0 * _MP, backend="numpy")
        n_e, B_mag, T_e_eV = self._make_grid()
        params = lhdi.compute_lhdi_params(n_e, B_mag, T_e_eV, T_i_eV=5.0, dr=0.025)
        assert np.all(params.D_eff >= 0)

    def test_gradient_drives_growth(self):
        """Higher density gradient → higher LHDI growth rate."""
        lhdi = LHDITransport(ion_mass_kg=2.0 * _MP, backend="numpy")
        B_mag = np.ones((20, 40)) * 0.05
        T_e_eV = np.ones((20, 40)) * 10.0

        # Low gradient
        n_low_grad = np.ones((20, 40)) * 1e18
        params_low = lhdi.compute_lhdi_params(n_low_grad, B_mag, T_e_eV, T_i_eV=5.0, dr=0.025)

        # High gradient
        r = np.linspace(0.01, 0.5, 20)
        n_high_grad = 1e18 * np.exp(-r[:, None] / 0.02)  # very steep
        params_high = lhdi.compute_lhdi_params(n_high_grad, B_mag, T_e_eV, T_i_eV=5.0, dr=0.025)

        # Mean gamma should be higher for steeper gradient
        assert np.mean(params_high.gamma_lhdi) > np.mean(params_low.gamma_lhdi)

    def test_apply_diffusion(self):
        lhdi = LHDITransport(ion_mass_kg=2.0 * _MP, backend="numpy")
        n_e = np.ones((20, 40)) * 1e18
        D_eff = np.ones((20, 40)) * 1.0  # 1 m²/s (large for testing)
        n_new = lhdi.apply_diffusion(n_e, D_eff, dt=1e-6, dr=0.025)
        assert n_new.shape == (20, 40)
        assert np.all(n_new >= 0)

    def test_saturation_factor(self):
        """Higher saturation factor → proportionally higher D_eff."""
        lhdi_low = LHDITransport(ion_mass_kg=2.0 * _MP, backend="numpy", saturation_factor=0.1)
        lhdi_high = LHDITransport(ion_mass_kg=2.0 * _MP, backend="numpy", saturation_factor=0.5)
        n_e, B_mag, T_e_eV = self._make_grid()
        p_low = lhdi_low.compute_lhdi_params(n_e, B_mag, T_e_eV, T_i_eV=5.0, dr=0.025)
        p_high = lhdi_high.compute_lhdi_params(n_e, B_mag, T_e_eV, T_i_eV=5.0, dr=0.025)
        np.testing.assert_allclose(p_high.D_eff, p_low.D_eff * 5.0, rtol=1e-10)


# ---------------------------------------------------------------------------
# HybridCoupler
# ---------------------------------------------------------------------------


class TestHybridCoupler:
    def _make_coupler(self, nr=10, nz=20):
        n_e = np.ones((nr, nz)) * 1e18
        B_mag = np.ones((nr, nz)) * 0.1
        state = CGLState.from_isotropic(n_e, 10.0, B_mag)
        fluid = CGLElectronFluid(state, backend="numpy")
        return HybridCoupler(cgl_fluid=fluid, backend="numpy", dr=0.05, dz=0.01)

    def test_step_returns_hybrid_state(self):
        coupler = self._make_coupler()
        nr, nz = 10, 20
        ions = IonMoments(
            density=np.ones((nr, nz)) * 1e18,
            vr=np.zeros((nr, nz)),
            vz=np.ones((nr, nz)) * 1e4,
            p_perp=np.ones((nr, nz)) * 1e18 * 5.0 * _QE,
            p_par=np.ones((nr, nz)) * 1e18 * 5.0 * _QE,
        )
        B_mag = np.ones((nr, nz)) * 0.1
        state = coupler.step(ions, B_mag, dt=1e-9)

        assert state.n_e.shape == (nr, nz)
        assert state.phi_amb.shape == (nr, nz)
        assert state.E_r.shape == (nr, nz)
        assert state.E_z.shape == (nr, nz)

    def test_quasi_neutrality(self):
        """n_e should equal Z * n_i (quasi-neutrality)."""
        coupler = self._make_coupler()
        nr, nz = 10, 20
        n_ion = 2.5e18
        ions = IonMoments(
            density=np.ones((nr, nz)) * n_ion,
            vr=np.zeros((nr, nz)),
            vz=np.zeros((nr, nz)),
            p_perp=np.ones((nr, nz)) * n_ion * 5.0 * _QE,
            p_par=np.ones((nr, nz)) * n_ion * 5.0 * _QE,
            charge_number=1,
        )
        B_mag = np.ones((nr, nz)) * 0.1
        state = coupler.step(ions, B_mag, dt=1e-9)
        np.testing.assert_allclose(state.n_e, n_ion * np.ones((nr, nz)), rtol=1e-10)

    def test_doubly_charged_ions(self):
        """For Z=2, n_e should be 2× n_i."""
        coupler = self._make_coupler()
        nr, nz = 10, 20
        n_ion = 1e18
        ions = IonMoments(
            density=np.ones((nr, nz)) * n_ion,
            vr=np.zeros((nr, nz)),
            vz=np.zeros((nr, nz)),
            p_perp=np.ones((nr, nz)) * n_ion * 10.0 * _QE,
            p_par=np.ones((nr, nz)) * n_ion * 10.0 * _QE,
            charge_number=2,
        )
        B_mag = np.ones((nr, nz)) * 0.1
        state = coupler.step(ions, B_mag, dt=1e-9)
        np.testing.assert_allclose(state.n_e, 2.0 * n_ion * np.ones((nr, nz)), rtol=1e-10)

    def test_temperatures_positive(self):
        coupler = self._make_coupler()
        nr, nz = 10, 20
        ions = IonMoments(
            density=np.ones((nr, nz)) * 1e18,
            vr=np.zeros((nr, nz)),
            vz=np.zeros((nr, nz)),
            p_perp=np.ones((nr, nz)) * 1e18 * 5.0 * _QE,
            p_par=np.ones((nr, nz)) * 1e18 * 5.0 * _QE,
        )
        B_mag = np.ones((nr, nz)) * 0.1
        state = coupler.step(ions, B_mag, dt=1e-9)
        assert np.all(state.T_perp_eV >= 0)
        assert np.all(state.T_par_eV >= 0)

    def test_with_lhdi(self):
        """Coupler with LHDI should still return valid state."""
        nr, nz = 10, 20
        n_e = np.ones((nr, nz)) * 1e18
        B_mag = np.ones((nr, nz)) * 0.1
        state = CGLState.from_isotropic(n_e, 10.0, B_mag)
        fluid = CGLElectronFluid(state, backend="numpy")
        lhdi = LHDITransport(ion_mass_kg=2.0 * _MP, backend="numpy")
        coupler = HybridCoupler(
            cgl_fluid=fluid, lhdi_transport=lhdi, backend="numpy", dr=0.05, dz=0.01
        )

        ions = IonMoments(
            density=n_e.copy(),
            vr=np.zeros((nr, nz)),
            vz=np.zeros((nr, nz)),
            p_perp=np.ones((nr, nz)) * 1e18 * 5.0 * _QE,
            p_par=np.ones((nr, nz)) * 1e18 * 5.0 * _QE,
        )
        result = coupler.step(ions, B_mag, dt=1e-9)
        assert result.n_e.shape == (nr, nz)


# ---------------------------------------------------------------------------
# MLX paths
# ---------------------------------------------------------------------------


class TestHybridMlx:
    mlx = pytest.importorskip("mlx.core")

    def test_cgl_update_mlx(self):
        nr, nz = 5, 5
        n_e = np.ones((nr, nz)) * 1e18
        B_mag = np.ones((nr, nz)) * 0.1
        state = CGLState.from_isotropic(n_e, 10.0, B_mag)
        fluid = CGLElectronFluid(state, backend="mlx")

        n_new = np.ones((nr, nz)) * 1e18
        B_new = np.ones((nr, nz)) * 0.2  # double B
        state_new = fluid.update(n_new, B_new, dt=1e-9)

        np.testing.assert_allclose(state_new.T_perp_eV, 20.0, rtol=1e-4)

    def test_lhdi_compute_mlx(self):
        lhdi = LHDITransport(ion_mass_kg=2.0 * _MP, backend="mlx")
        nr, nz = 10, 20
        r = np.linspace(0.01, 0.5, nr)
        n_e = 1e18 * np.exp(-r[:, None] / 0.1) * np.ones((nr, nz))
        B_mag = np.ones((nr, nz)) * 0.05
        T_e_eV = np.ones((nr, nz)) * 10.0

        params = lhdi.compute_lhdi_params(n_e, B_mag, T_e_eV, T_i_eV=5.0, dr=0.05)
        assert params.omega_lh.shape == (nr, nz)
        assert np.all(params.D_eff >= 0)
