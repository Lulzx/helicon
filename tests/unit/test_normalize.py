"""Tests for helicon.postprocess.normalize (spec §13.1)."""

from __future__ import annotations

import math

import pytest

from helicon.postprocess.normalize import (
    PlasmaScales,
    compute_plasma_scales,
    normalize_bfield,
    normalize_density,
    normalize_length,
    normalize_pressure,
    normalize_time,
    normalize_velocity,
)

# Physical constants
_C = 2.997924e8
_E = 1.602176634e-19
_EPS0 = 8.854187817e-12
_AMU = 1.66053906660e-27


class TestComputePlasmaScales:
    """Tests for compute_plasma_scales()."""

    def test_returns_plasma_scales(self) -> None:
        scales = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5)
        assert isinstance(scales, PlasmaScales)

    def test_d_i_positive(self) -> None:
        scales = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5)
        assert scales.d_i_m > 0.0

    def test_d_i_formula(self) -> None:
        """d_i = c / omega_pi where omega_pi = sqrt(n q^2 / (eps0 m_i))."""
        n0 = 1e19
        m_i = 2.014 * _AMU
        omega_pi = math.sqrt(n0 * _E**2 / (_EPS0 * m_i))
        expected_d_i = _C / omega_pi
        scales = compute_plasma_scales(n0=n0, T_e_eV=100.0, B_T=0.5)
        assert scales.d_i_m == pytest.approx(expected_d_i, rel=1e-6)

    def test_tau_ci_positive(self) -> None:
        scales = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5)
        assert scales.tau_ci_s > 0.0

    def test_tau_ci_formula(self) -> None:
        """tau_ci = 2pi / (q B / m_i)."""
        B = 0.5
        m_i = 2.014 * _AMU
        omega_ci = _E * B / m_i
        expected_tau_ci = 2.0 * math.pi / omega_ci
        scales = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=B)
        assert scales.tau_ci_s == pytest.approx(expected_tau_ci, rel=1e-6)

    def test_c_s_positive(self) -> None:
        scales = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5)
        assert scales.c_s_ms > 0.0

    def test_c_s_formula(self) -> None:
        """c_s = sqrt(T_e / m_i)."""
        T_e_eV = 100.0
        m_i = 2.014 * _AMU
        T_e_J = T_e_eV * _E
        expected_c_s = math.sqrt(T_e_J / m_i)
        scales = compute_plasma_scales(n0=1e19, T_e_eV=T_e_eV, B_T=0.5)
        assert scales.c_s_ms == pytest.approx(expected_c_s, rel=1e-6)

    def test_higher_density_smaller_d_i(self) -> None:
        s1 = compute_plasma_scales(n0=1e18, T_e_eV=100.0, B_T=0.5)
        s2 = compute_plasma_scales(n0=1e20, T_e_eV=100.0, B_T=0.5)
        assert s1.d_i_m > s2.d_i_m

    def test_custom_ion_mass(self) -> None:
        """Xenon (131 amu) has larger d_i than deuterium (d_i ∝ √m_i)."""
        s_d = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5, ion_mass_amu=2.014)
        s_xe = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5, ion_mass_amu=131.0)
        assert s_xe.d_i_m > s_d.d_i_m

    def test_tau_ci_infinite_when_b_zero(self) -> None:
        """Zero magnetic field → infinite cyclotron period."""
        scales = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.0)
        assert scales.tau_ci_s == float("inf")

    def test_charge_number_doubles_cyclotron_frequency(self) -> None:
        """Z=2 ions have twice the cyclotron frequency, half the period."""
        s1 = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5, charge_number=1)
        s2 = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5, charge_number=2)
        assert s2.omega_ci_rads == pytest.approx(2.0 * s1.omega_ci_rads, rel=1e-6)
        assert s2.tau_ci_s == pytest.approx(0.5 * s1.tau_ci_s, rel=1e-6)


class TestNormalizeLength:
    def test_at_one_d_i(self) -> None:
        scales = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5)
        assert normalize_length(scales.d_i_m, scales) == pytest.approx(1.0, rel=1e-9)

    def test_zero_length(self) -> None:
        scales = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5)
        assert normalize_length(0.0, scales) == pytest.approx(0.0)

    def test_two_d_i(self) -> None:
        scales = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5)
        assert normalize_length(2.0 * scales.d_i_m, scales) == pytest.approx(2.0, rel=1e-9)


class TestNormalizeTime:
    def test_at_one_tau_ci(self) -> None:
        scales = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5)
        assert normalize_time(scales.tau_ci_s, scales) == pytest.approx(1.0, rel=1e-9)

    def test_scaling(self) -> None:
        scales = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5)
        assert normalize_time(5.0 * scales.tau_ci_s, scales) == pytest.approx(5.0, rel=1e-9)


class TestNormalizeVelocity:
    def test_at_one_c_s(self) -> None:
        scales = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5)
        assert normalize_velocity(scales.c_s_ms, scales) == pytest.approx(1.0, rel=1e-9)

    def test_supersonic(self) -> None:
        scales = compute_plasma_scales(n0=1e19, T_e_eV=100.0, B_T=0.5)
        v = 3.0 * scales.c_s_ms
        assert normalize_velocity(v, scales) == pytest.approx(3.0, rel=1e-9)


class TestNormalizeBfield:
    def test_at_throat(self) -> None:
        assert normalize_bfield(1.0, B_throat_T=1.0) == pytest.approx(1.0)

    def test_downstream_weakened(self) -> None:
        assert normalize_bfield(0.1, B_throat_T=1.0) == pytest.approx(0.1)

    def test_zero_throat_raises(self) -> None:
        with pytest.raises(ValueError, match="B_throat_T must be non-zero"):
            normalize_bfield(0.5, B_throat_T=0.0)


class TestNormalizeDensity:
    def test_at_n0(self) -> None:
        assert normalize_density(1e19, n0=1e19) == pytest.approx(1.0)

    def test_half_density(self) -> None:
        assert normalize_density(5e18, n0=1e19) == pytest.approx(0.5)

    def test_zero_n0_raises(self) -> None:
        with pytest.raises(ValueError, match="n0 must be non-zero"):
            normalize_density(1e19, n0=0.0)


class TestNormalizePressure:
    def test_at_reference_pressure(self) -> None:
        n0 = 1e19
        T_e_eV = 100.0
        T_e_J = T_e_eV * _E
        P_ref = n0 * T_e_J
        assert normalize_pressure(P_ref, n0=n0, T_e_eV=T_e_eV) == pytest.approx(1.0, rel=1e-9)

    def test_zero_n0_raises(self) -> None:
        with pytest.raises(ValueError, match="n0 and T_e_eV must be non-zero"):
            normalize_pressure(1.0, n0=0.0, T_e_eV=100.0)
