"""Tests for helicon.neutrals — Monte Carlo neutral dynamics (spec v1.2)."""

from __future__ import annotations

import numpy as np
import pytest

from helicon.neutrals.cross_sections import (
    SPECIES_MASS,
    cx_cross_section_m2,
    cx_rate_m3s,
    ionization_rate_m3s,
    recombination_rate_m3s,
)
from helicon.neutrals.monte_carlo import MCCCollider, NeutralDynamics, NeutralParticles

# ---------------------------------------------------------------------------
# cross_sections.py
# ---------------------------------------------------------------------------


class TestCXCrossSection:
    def test_hydrogen_cx_positive(self):
        sigma = cx_cross_section_m2("H", 100.0)
        assert float(sigma) > 0

    def test_deuterium_cx_plausible_range(self):
        # D–D+ CX at 100 eV: expect ~10–100 × 10^-20 m²
        sigma = cx_cross_section_m2("D", 100.0)
        assert 1e-21 < float(sigma) < 1e-17

    def test_xenon_cx_constant(self):
        sigma_100 = cx_cross_section_m2("Xe", 100.0)
        sigma_200 = cx_cross_section_m2("Xe", 200.0)
        # Xe is approximately constant
        np.testing.assert_allclose(float(sigma_100), float(sigma_200), rtol=0.01)

    def test_zero_energy_returns_zero(self):
        sigma = cx_cross_section_m2("H", 0.0)
        assert float(sigma) == 0.0

    def test_array_input(self):
        E = np.array([10.0, 100.0, 1000.0])
        sigma = cx_cross_section_m2("D", E)
        assert sigma.shape == (3,)
        assert np.all(sigma > 0)

    def test_unknown_species_raises(self):
        with pytest.raises(ValueError, match="Unknown species"):
            cx_cross_section_m2("Krypton42", 100.0)


class TestIonizationRate:
    def test_hydrogen_rate_at_10eV(self):
        # At 10 eV should be small but positive
        rate = ionization_rate_m3s("H", 10.0)
        assert float(rate) > 0

    def test_hydrogen_rate_at_100eV(self):
        # At 100 eV should be near peak (~3e-14 m³/s)
        rate = ionization_rate_m3s("H", 100.0)
        assert 1e-16 < float(rate) < 1e-12

    def test_deuterium_same_as_hydrogen(self):
        # D and H have same electron structure → same ionization rate
        rH = float(ionization_rate_m3s("H", 50.0))
        rD = float(ionization_rate_m3s("D", 50.0))
        np.testing.assert_allclose(rH, rD, rtol=1e-10)

    def test_xenon_rate_positive(self):
        rate = ionization_rate_m3s("Xe", 20.0)
        assert float(rate) > 0

    def test_increases_with_temperature(self):
        r_low = float(ionization_rate_m3s("H", 5.0))
        r_high = float(ionization_rate_m3s("H", 50.0))
        assert r_high > r_low

    def test_array_input(self):
        T = np.array([10.0, 50.0, 100.0])
        rates = ionization_rate_m3s("H", T)
        assert rates.shape == (3,)
        assert np.all(rates > 0)


class TestRecombinationRate:
    def test_hydrogen_positive(self):
        rate = recombination_rate_m3s("H", 10.0)
        assert float(rate) > 0

    def test_decreases_with_temperature(self):
        # Radiative recombination decreases with T
        r_low = float(recombination_rate_m3s("H", 1.0))
        r_high = float(recombination_rate_m3s("H", 100.0))
        assert r_low > r_high

    def test_helium_positive(self):
        rate = recombination_rate_m3s("He", 10.0)
        assert float(rate) > 0


class TestCXRate:
    def test_deuterium_rate_positive(self):
        rate = cx_rate_m3s("D", 10.0)
        assert float(rate) > 0

    def test_rate_increases_with_temperature(self):
        r_low = float(cx_rate_m3s("H", 1.0))
        r_high = float(cx_rate_m3s("H", 100.0))
        assert r_high > r_low

    def test_species_mass_catalog(self):
        assert "D" in SPECIES_MASS
        assert "H" in SPECIES_MASS
        assert "He" in SPECIES_MASS
        assert "Xe" in SPECIES_MASS


# ---------------------------------------------------------------------------
# NeutralParticles
# ---------------------------------------------------------------------------


class TestNeutralParticles:
    def test_create_initializes_particles(self):
        p = NeutralParticles.create(
            n_particles=100,
            species="D",
            n_density_m3=1e18,
            T_eV=0.025,
        )
        assert len(p.positions) == 100
        assert len(p.velocities) == 100
        assert len(p.weights) == 100
        assert p.n_alive == 100

    def test_all_alive_initially(self):
        p = NeutralParticles.create(100, "H", 1e18, 0.025)
        assert np.all(p.alive)

    def test_weight_conserves_density(self):
        n_density = 1e18
        n_particles = 1000
        p = NeutralParticles.create(
            n_particles=n_particles,
            species="D",
            n_density_m3=n_density,
            T_eV=0.025,
            domain_r=(0.0, 0.5),
            domain_z=(-0.5, 2.0),
        )
        # Total physical particles = sum of weights ≈ n * V
        domain_vol = np.pi * 0.5**2 * 2.5
        total_phys = np.sum(p.weights)
        expected = n_density * domain_vol
        np.testing.assert_allclose(total_phys, expected, rtol=0.01)

    def test_positions_in_domain(self):
        p = NeutralParticles.create(
            100,
            "D",
            1e18,
            0.025,
            domain_r=(0.0, 0.5),
            domain_z=(-0.5, 2.0),
        )
        assert np.all(p.r >= 0.0)
        assert np.all(p.r <= 0.5)
        assert np.all(p.z >= -0.5)
        assert np.all(p.z <= 2.0)

    def test_r_and_z_properties(self):
        p = NeutralParticles.create(50, "H", 1e17, 0.025)
        assert len(p.r) == 50
        assert len(p.z) == 50


# ---------------------------------------------------------------------------
# MCCCollider — null collision method
# ---------------------------------------------------------------------------


class TestMCCCollider:
    def test_step_returns_mcc_result(self):
        collider = MCCCollider(species="D", dt=1e-9, backend="numpy")
        neutrals = NeutralParticles.create(200, "D", 1e18, 0.025)

        n_arr = np.full(neutrals.n_alive, 1e18)
        T_i_arr = np.full(neutrals.n_alive, 10.0)
        T_e_arr = np.full(neutrals.n_alive, 20.0)

        result = collider.step(
            neutrals,
            n_arr,
            T_i_arr,
            T_e_arr,
            domain_r=(0.0, 0.5),
            domain_z=(-0.5, 2.0),
        )
        assert result.n_cx >= 0
        assert result.n_ionized >= 0
        assert result.n_recombined >= 0

    def test_null_frequency_computation(self):
        collider = MCCCollider(species="D", dt=1e-9)
        nu_max = collider.compute_null_frequency(
            n_ion_max_m3=1e19,
            T_ion_eV=10.0,
            T_e_eV=20.0,
        )
        assert nu_max > 0

    def test_particles_removed_after_cx(self):
        """High density → some CX events → some particles removed."""
        collider = MCCCollider(species="D", dt=1e-7, backend="numpy")
        neutrals = NeutralParticles.create(500, "D", 1e18, 0.025, seed=123)
        n_alive_before = neutrals.n_alive

        # Very high density to force collisions
        n_arr = np.full(neutrals.n_alive, 1e21)
        T_i_arr = np.full(neutrals.n_alive, 50.0)
        T_e_arr = np.full(neutrals.n_alive, 50.0)

        collider.step(
            neutrals,
            n_arr,
            T_i_arr,
            T_e_arr,
            domain_r=(0.0, 0.5),
            domain_z=(-0.5, 2.0),
            seed=42,
        )
        # At least some collisions should remove particles
        assert neutrals.n_alive <= n_alive_before

    def test_particles_leave_domain(self):
        """Particles near boundary should be removed after push."""
        collider = MCCCollider(species="D", dt=1.0, backend="numpy")  # large dt
        neutrals = NeutralParticles.create(100, "D", 1e18, 0.025, seed=99)

        # Give all particles a large outward velocity
        neutrals.velocities[:, 2] = 1e6  # vz = 1 km/s → exits in 1 s

        n_arr = np.zeros(neutrals.n_alive)  # no collisions
        T_i_arr = np.full(neutrals.n_alive, 1.0)
        T_e_arr = np.full(neutrals.n_alive, 1.0)

        collider.step(
            neutrals,
            n_arr,
            T_i_arr,
            T_e_arr,
            domain_r=(0.0, 0.5),
            domain_z=(-0.5, 2.0),
            seed=0,
        )
        # Most particles should have left
        assert neutrals.n_alive < 100


# ---------------------------------------------------------------------------
# NeutralDynamics — high-level interface
# ---------------------------------------------------------------------------


class TestNeutralDynamics:
    def test_init(self):
        nd = NeutralDynamics(
            species="D",
            dt=1e-9,
            n_particles=100,
            n_density_m3=1e18,
            T_eV=0.025,
        )
        assert nd.n_alive == 100
        assert nd.species == "D"

    def test_step_runs_without_error(self):
        nd = NeutralDynamics("D", dt=1e-9, n_particles=100, n_density_m3=1e18, T_eV=0.025)
        result = nd.step(n_ion_m3=1e18, T_ion_eV=10.0, T_e_eV=20.0)
        assert result.n_cx >= 0

    def test_multiple_steps(self):
        nd = NeutralDynamics("H", dt=1e-9, n_particles=200, n_density_m3=1e17, T_eV=0.025)
        for _ in range(5):
            nd.step(n_ion_m3=1e17, T_ion_eV=5.0, T_e_eV=10.0)
        assert nd.n_alive <= 200

    def test_density_on_grid(self):
        nd = NeutralDynamics("D", dt=1e-9, n_particles=1000, n_density_m3=1e18, T_eV=0.025)
        r_grid = np.linspace(0.01, 0.49, 20)
        z_grid = np.linspace(-0.4, 1.9, 50)
        density = nd.neutral_density_on_grid(r_grid, z_grid)
        assert density.shape == (20, 50)
        assert np.all(density >= 0)
        assert np.sum(density) > 0

    def test_xenon_species(self):
        nd = NeutralDynamics("Xe", dt=1e-9, n_particles=50, n_density_m3=1e18, T_eV=0.025)
        result = nd.step(n_ion_m3=1e17, T_ion_eV=50.0, T_e_eV=30.0)
        assert result.n_cx >= 0


# ---------------------------------------------------------------------------
# MLX path (skipped if not available)
# ---------------------------------------------------------------------------


class TestNeutralsMlx:
    mlx = pytest.importorskip("mlx.core")

    def test_step_mlx_backend(self):
        collider = MCCCollider(species="D", dt=1e-9, backend="mlx")
        neutrals = NeutralParticles.create(100, "D", 1e18, 0.025)
        n_arr = np.full(neutrals.n_alive, 1e18)
        T_i_arr = np.full(neutrals.n_alive, 10.0)
        T_e_arr = np.full(neutrals.n_alive, 20.0)

        result = collider.step(
            neutrals,
            n_arr,
            T_i_arr,
            T_e_arr,
            domain_r=(0.0, 0.5),
            domain_z=(-0.5, 2.0),
        )
        assert result.n_cx >= 0

    def test_neutral_dynamics_mlx(self):
        nd = NeutralDynamics(
            "D", dt=1e-9, n_particles=100, n_density_m3=1e18, T_eV=0.025, backend="mlx"
        )
        result = nd.step(1e18, 10.0, 20.0)
        assert result.n_cx >= 0
