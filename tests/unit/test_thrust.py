"""Tests for helicon.postprocess.thrust module."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from helicon.fields.biot_savart import HAS_MLX
from helicon.postprocess.thrust import ThrustResult, _thrust_reduce_mlx, compute_thrust

skip_no_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")


def _write_thrust_snapshot(
    path: Path,
    *,
    species_name: str = "D_plus",
    n_particles: int = 2000,
    seed: int = 42,
) -> None:
    """Write synthetic particle data for thrust tests."""
    rng = np.random.default_rng(seed)
    species_mass = 3.3435837724e-27

    z = rng.uniform(0.0, 2.0, size=n_particles)
    vz = rng.normal(200_000.0, 20_000.0, size=n_particles)
    pz = vz * species_mass
    w = np.ones(n_particles)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        base = f.create_group("data/0")
        sp = base.create_group(f"particles/{species_name}")
        pos = sp.create_group("position")
        pos.create_dataset("z", data=z)
        mom = sp.create_group("momentum")
        mom.create_dataset("z", data=pz)
        sp.create_dataset("weighting", data=w)


@pytest.fixture
def thrust_output(tmp_path: Path) -> Path:
    out = tmp_path / "thrust"
    out.mkdir()
    _write_thrust_snapshot(out / "snap.h5")
    return out


class TestThrustResult:
    def test_returns_result(self, thrust_output: Path) -> None:
        result = compute_thrust(thrust_output)
        assert isinstance(result, ThrustResult)

    def test_thrust_positive(self, thrust_output: Path) -> None:
        result = compute_thrust(thrust_output)
        assert result.thrust_N > 0.0

    def test_isp_positive(self, thrust_output: Path) -> None:
        result = compute_thrust(thrust_output)
        assert result.isp_s > 0.0

    def test_particle_count(self, thrust_output: Path) -> None:
        result = compute_thrust(thrust_output)
        assert result.n_particles_counted > 0

    def test_no_files_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            compute_thrust(tmp_path)


@skip_no_mlx
class TestThrustMLXMatchesNumPy:
    """MLX path should produce same thrust as NumPy for equal inputs."""

    def test_thrust_reduce_mlx_matches_numpy(self) -> None:
        rng = np.random.default_rng(7)
        wt = rng.uniform(0.5, 2.0, 10_000).astype(np.float32)
        vz = rng.normal(200_000.0, 20_000.0, 10_000).astype(np.float32)
        mass = 3.3435837724e-27

        np_thrust = float(np.sum(wt * mass * vz**2))
        np_mdot = float(np.sum(wt * mass * np.abs(vz)))

        mlx_thrust, mlx_mdot = _thrust_reduce_mlx(wt, mass, vz)

        # float32 accumulation; allow 1% relative error
        assert abs(mlx_thrust - np_thrust) / max(abs(np_thrust), 1e-30) < 0.01
        assert abs(mlx_mdot - np_mdot) / max(abs(np_mdot), 1e-30) < 0.01

    def test_compute_thrust_mlx_backend(self, thrust_output: Path) -> None:
        result_np = compute_thrust(thrust_output, backend="numpy")
        result_mlx = compute_thrust(thrust_output, backend="mlx")
        rel = abs(result_np.thrust_N - result_mlx.thrust_N) / max(result_np.thrust_N, 1e-30)
        assert rel < 0.02
