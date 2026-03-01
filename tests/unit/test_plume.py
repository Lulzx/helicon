"""Tests for magnozzlex.postprocess.plume module."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from magnozzlex.postprocess.plume import (
    PlumeResult,
    compute_electron_magnetization,
    compute_plume_metrics,
    compute_pressure_anisotropy,
)


def _write_plume_snapshot(
    path: Path,
    *,
    species_name: str = "D_plus",
    n_particles: int = 2000,
    seed: int = 42,
) -> None:
    """Write synthetic particle data for plume tests."""
    rng = np.random.default_rng(seed)
    species_mass = 3.3435837724e-27

    z = rng.uniform(0.0, 2.0, size=n_particles)
    r = rng.exponential(0.05, size=n_particles)
    # Mostly axial momentum with some radial spread
    vz = rng.normal(200000.0, 20000.0, size=n_particles)
    vr = rng.normal(0.0, 30000.0, size=n_particles)
    pz = vz * species_mass
    pr = vr * species_mass
    w = np.ones(n_particles)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        base = f.create_group("data/0")
        sp = base.create_group(f"particles/{species_name}")
        pos = sp.create_group("position")
        pos.create_dataset("z", data=z)
        pos.create_dataset("r", data=r)
        mom = sp.create_group("momentum")
        mom.create_dataset("z", data=pz)
        mom.create_dataset("r", data=pr)
        sp.create_dataset("weighting", data=w)


@pytest.fixture
def plume_output(tmp_path: Path) -> Path:
    out = tmp_path / "plume"
    out.mkdir()
    _write_plume_snapshot(out / "snap.h5")
    return out


class TestPlumeMetrics:
    """Tests for compute_plume_metrics."""

    def test_returns_result(self, plume_output: Path) -> None:
        result = compute_plume_metrics(plume_output)
        assert isinstance(result, PlumeResult)

    def test_divergence_angle_reasonable(self, plume_output: Path) -> None:
        """For mostly axial flow, divergence should be small."""
        result = compute_plume_metrics(plume_output)
        assert 0.0 <= result.divergence_half_angle_deg <= 90.0
        # With 200 km/s axial, 30 km/s radial spread, angle ~ 8-15 deg
        assert result.divergence_half_angle_deg < 30.0

    def test_beam_efficiency_high_for_axial_flow(self, plume_output: Path) -> None:
        """With mostly axial momentum, beam efficiency should be > 0.5."""
        result = compute_plume_metrics(plume_output)
        assert result.beam_efficiency > 0.5

    def test_thrust_coefficient_positive(self, plume_output: Path) -> None:
        result = compute_plume_metrics(plume_output)
        assert result.thrust_coefficient > 0.0

    def test_radial_loss_fraction_bounded(self, plume_output: Path) -> None:
        result = compute_plume_metrics(plume_output)
        assert 0.0 <= result.radial_loss_fraction <= 1.0

    def test_no_files(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_plume_metrics(tmp_path)


class TestElectronMagnetization:
    """Tests for compute_electron_magnetization."""

    def test_shape_preserved(self) -> None:
        nr, nz = 32, 64
        Br = np.random.default_rng(0).standard_normal((nr, nz)) * 0.01
        Bz = np.ones((nr, nz)) * 0.5
        n_e = np.ones((nr, nz)) * 1e18
        Omega = compute_electron_magnetization(Br, Bz, n_e, T_e_eV=10.0)
        assert Omega.shape == (nr, nz)

    def test_strong_field_high_magnetization(self) -> None:
        """Strong B → high Ω_e (well magnetized)."""
        nr, nz = 16, 16
        Br = np.zeros((nr, nz))
        Bz = np.ones((nr, nz)) * 1.0  # 1 Tesla — very strong
        n_e = np.ones((nr, nz)) * 1e16  # low density
        Omega = compute_electron_magnetization(Br, Bz, n_e, T_e_eV=10.0)
        assert np.all(Omega > 1.0)  # well magnetized

    def test_weak_field_low_magnetization(self) -> None:
        """Weak B → low Ω_e (demagnetized)."""
        nr, nz = 16, 16
        Br = np.zeros((nr, nz))
        Bz = np.ones((nr, nz)) * 1e-6  # very weak field
        n_e = np.ones((nr, nz)) * 1e20  # high density → high collisions
        Omega = compute_electron_magnetization(Br, Bz, n_e, T_e_eV=10.0)
        # Most should be demagnetized
        assert np.median(Omega) < 100.0

    def test_fixed_collision_freq(self) -> None:
        """With fixed collision frequency, result should be deterministic."""
        nr, nz = 8, 8
        Br = np.zeros((nr, nz))
        Bz = np.ones((nr, nz)) * 0.1
        n_e = np.ones((nr, nz)) * 1e18
        Omega = compute_electron_magnetization(Br, Bz, n_e, T_e_eV=10.0, collision_freq=1e9)
        # omega_ce = eB/m_e = 1.6e-19 * 0.1 / 9.1e-31 ~ 1.76e10
        # Omega_e = omega_ce / nu = 1.76e10 / 1e9 ~ 17.6
        expected = 1.602e-19 * 0.1 / 9.109e-31 / 1e9
        assert np.allclose(Omega, expected, rtol=0.01)


class TestPressureAnisotropy:
    """Tests for compute_pressure_anisotropy."""

    def test_isotropic_gives_zero(self) -> None:
        P = np.ones((16, 16)) * 100.0
        A = compute_pressure_anisotropy(P, P)
        assert np.allclose(A, 0.0)

    def test_perp_dominated(self) -> None:
        """P_perp > P_parallel → A > 0."""
        P_perp = np.ones((8, 8)) * 200.0
        P_par = np.ones((8, 8)) * 100.0
        A = compute_pressure_anisotropy(P_perp, P_par)
        assert np.all(A > 0)
        assert np.allclose(A, 1.0)  # 200/100 - 1 = 1

    def test_parallel_dominated(self) -> None:
        """P_parallel > P_perp → A < 0."""
        P_perp = np.ones((8, 8)) * 50.0
        P_par = np.ones((8, 8)) * 100.0
        A = compute_pressure_anisotropy(P_perp, P_par)
        assert np.all(A < 0)
        assert np.allclose(A, -0.5)  # 50/100 - 1 = -0.5
