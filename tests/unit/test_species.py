"""Tests for helicon.postprocess.species — multi-species detachment (spec v1.2)."""

from __future__ import annotations

import numpy as np
import pytest
import tempfile
from pathlib import Path

from helicon.postprocess.species import (
    SPECIES_CATALOG,
    MultiSpeciesDetachmentResult,
    SpeciesDetachmentResult,
    compute_species_moments,
)


_QE = 1.6021766340e-19
_MP = 1.6726219236951e-27


# ---------------------------------------------------------------------------
# SPECIES_CATALOG
# ---------------------------------------------------------------------------


class TestSpeciesCatalog:
    def test_contains_key_species(self):
        for sp in ("D_plus", "He_plus2", "H_plus", "alpha"):
            assert sp in SPECIES_CATALOG

    def test_mass_plausible(self):
        for sp, info in SPECIES_CATALOG.items():
            assert info["mass"] > 0

    def test_charge_number_positive(self):
        for sp, info in SPECIES_CATALOG.items():
            assert info["Z"] >= 1


# ---------------------------------------------------------------------------
# SpeciesDetachmentResult
# ---------------------------------------------------------------------------


class TestSpeciesDetachmentResult:
    def _make(self, eta_mom=0.8, eta_part=0.85, eta_energy=0.75):
        return SpeciesDetachmentResult(
            species="D_plus",
            label="D⁺",
            mass=2.0 * _MP,
            momentum_based=eta_mom,
            particle_based=eta_part,
            energy_based=eta_energy,
            n_injected=10000,
            n_exited_downstream=8500,
            n_lost_radial=1000,
            n_reflected=500,
        )

    def test_summary_string(self):
        res = self._make()
        s = res.summary()
        assert "D⁺" in s
        assert "η_d" in s
        assert "0.8000" in s

    def test_fields_accessible(self):
        res = self._make(eta_mom=0.9)
        assert res.momentum_based == 0.9
        assert res.n_injected == 10000


# ---------------------------------------------------------------------------
# MultiSpeciesDetachmentResult
# ---------------------------------------------------------------------------


class TestMultiSpeciesDetachmentResult:
    def _make(self):
        results = {}
        for sp, eta in [("D_plus", 0.8), ("He_plus2", 0.9)]:
            cat = SPECIES_CATALOG[sp]
            results[sp] = SpeciesDetachmentResult(
                species=sp,
                label=cat["label"],
                mass=cat["mass"],
                momentum_based=eta,
                particle_based=eta,
                energy_based=eta * 0.9,
                n_injected=10000,
                n_exited_downstream=int(10000 * eta),
                n_lost_radial=int(10000 * (1 - eta) * 0.5),
                n_reflected=int(10000 * (1 - eta) * 0.5),
            )
        return MultiSpeciesDetachmentResult(results=results)

    def test_getitem(self):
        multi = self._make()
        res = multi["D_plus"]
        assert isinstance(res, SpeciesDetachmentResult)

    def test_species_names(self):
        multi = self._make()
        names = multi.species_names()
        assert "D_plus" in names
        assert "He_plus2" in names

    def test_summary_table(self):
        multi = self._make()
        s = multi.summary()
        assert "D⁺" in s
        assert "He²⁺" in s

    def test_dominant_species(self):
        multi = self._make()
        dom = multi.dominant_species()
        # He2+ has η_d=0.9 vs D+ has 0.8
        assert dom == "He_plus2"


# ---------------------------------------------------------------------------
# compute_species_moments
# ---------------------------------------------------------------------------


class TestComputeSpeciesMoments:
    def _make_particles(self, N=1000, seed=42):
        rng = np.random.default_rng(seed)
        r = rng.uniform(0.0, 0.5, N)
        z = rng.uniform(0.0, 2.0, N)
        positions = np.column_stack([r, z])
        v_th = 1e5  # 100 km/s
        pz = rng.normal(0.0, v_th * 2.0 * _MP, N)
        weights = np.ones(N)
        return positions, pz, weights

    def test_output_shapes(self):
        r_grid = np.linspace(0.01, 0.49, 20)
        z_grid = np.linspace(0.1, 1.9, 40)
        pos, pz, w = self._make_particles()
        moments = compute_species_moments(pos, pz, w, 2.0 * _MP, r_grid, z_grid)

        assert moments["density"].shape == (20, 40)
        assert moments["vz"].shape == (20, 40)
        assert moments["pz_density"].shape == (20, 40)
        assert moments["T_par_eV"].shape == (20, 40)

    def test_density_nonneg(self):
        r_grid = np.linspace(0.01, 0.49, 20)
        z_grid = np.linspace(0.1, 1.9, 40)
        pos, pz, w = self._make_particles()
        moments = compute_species_moments(pos, pz, w, 2.0 * _MP, r_grid, z_grid)
        assert np.all(moments["density"] >= 0)

    def test_temperature_nonneg(self):
        r_grid = np.linspace(0.01, 0.49, 20)
        z_grid = np.linspace(0.1, 1.9, 40)
        pos, pz, w = self._make_particles()
        moments = compute_species_moments(pos, pz, w, 2.0 * _MP, r_grid, z_grid)
        assert np.all(moments["T_par_eV"] >= 0)

    def test_1d_position_and_momentum(self):
        """Should handle 1D position/momentum arrays."""
        z = np.linspace(0.0, 2.0, 100)
        r_grid = np.linspace(0.01, 0.49, 5)
        z_grid = np.linspace(0.1, 1.9, 10)
        pz = np.ones(100) * 2.0 * _MP * 1e4
        w = np.ones(100)
        # 1D positions → treated as z-only
        moments = compute_species_moments(z, pz, w, 2.0 * _MP, r_grid, z_grid)
        assert moments["density"].shape == (5, 10)

    def test_mlx_backend(self):
        pytest.importorskip("mlx.core")
        r_grid = np.linspace(0.01, 0.49, 10)
        z_grid = np.linspace(0.1, 1.9, 20)
        pos, pz, w = self._make_particles(N=200)
        moments = compute_species_moments(pos, pz, w, 2.0 * _MP, r_grid, z_grid, backend="mlx")
        assert moments["density"].shape == (10, 20)
        assert np.all(moments["density"] >= 0)

    def test_conservation_of_total_weight(self):
        """Total deposited weight should equal total input weight (up to boundary effects)."""
        rng = np.random.default_rng(7)
        r = rng.uniform(0.01, 0.48, 5000)
        z = rng.uniform(0.01, 1.98, 5000)
        positions = np.column_stack([r, z])
        pz = np.zeros(5000)
        weights = np.ones(5000)

        r_grid = np.linspace(0.0, 0.5, 50)
        z_grid = np.linspace(0.0, 2.0, 100)
        dr = r_grid[1] - r_grid[0]
        dz = z_grid[1] - z_grid[0]

        moments = compute_species_moments(
            positions, pz, weights, 2.0 * _MP, r_grid, z_grid
        )

        # Density × cell volume → total particles
        r_2d = r_grid[:, None]
        cell_vol = 2.0 * np.pi * r_2d * dr * dz
        total = np.sum(moments["density"] * cell_vol)
        # Should be ~5000 (all particles deposited)
        np.testing.assert_allclose(total, 5000.0, rtol=0.2)
