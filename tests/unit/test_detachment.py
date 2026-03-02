"""Tests for helicon.postprocess.detachment module.

Since detachment computation requires HDF5 particle data from WarpX,
these tests create synthetic openPMD-like HDF5 files for testing.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from helicon.postprocess.detachment import DetachmentResult, compute_detachment


def _write_synthetic_snapshot(
    path: Path,
    *,
    species_name: str = "D_plus",
    n_particles: int = 1000,
    z_range: tuple[float, float] = (0.0, 2.0),
    r_range: tuple[float, float] = (0.0, 0.3),
    pz_mean: float = 1e-20,
    seed: int = 42,
) -> None:
    """Write a synthetic openPMD-like HDF5 snapshot."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(*z_range, size=n_particles)
    r = rng.uniform(*r_range, size=n_particles)
    pz = rng.normal(pz_mean, abs(pz_mean) * 0.3, size=n_particles)
    w = np.ones(n_particles)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        base = f.create_group("data/0")
        sp = base.create_group(f"particles/{species_name}")
        pos = sp.create_group("position")
        pos.create_dataset("z", data=z.astype(np.float64))
        pos.create_dataset("r", data=r.astype(np.float64))
        mom = sp.create_group("momentum")
        mom.create_dataset("z", data=pz.astype(np.float64))
        sp.create_dataset("weighting", data=w.astype(np.float64))


@pytest.fixture
def synthetic_output(tmp_path: Path) -> Path:
    """Create synthetic initial + final snapshots."""
    out = tmp_path / "output"
    out.mkdir()

    # Initial snapshot: particles near injection
    _write_synthetic_snapshot(
        out / "snapshot_0000.h5",
        z_range=(0.0, 0.2),
        r_range=(0.0, 0.1),
        pz_mean=1e-20,
        seed=1,
    )

    # Final snapshot: particles spread out, most downstream
    _write_synthetic_snapshot(
        out / "snapshot_0001.h5",
        z_range=(0.0, 2.0),
        r_range=(0.0, 0.3),
        pz_mean=1e-20,
        seed=2,
    )

    return out


class TestDetachment:
    """Tests for compute_detachment."""

    def test_returns_result(self, synthetic_output: Path) -> None:
        result = compute_detachment(synthetic_output)
        assert isinstance(result, DetachmentResult)

    def test_efficiencies_bounded(self, synthetic_output: Path) -> None:
        """All three efficiencies should be in [0, 1]."""
        result = compute_detachment(synthetic_output)
        assert 0.0 <= result.momentum_based <= 1.0
        assert 0.0 <= result.particle_based <= 1.0
        assert 0.0 <= result.energy_based <= 1.0

    def test_particle_counts_consistent(self, synthetic_output: Path) -> None:
        result = compute_detachment(synthetic_output)
        assert result.n_injected > 0
        # Sum of classified particles <= injected (some may be in between)
        total = result.n_exited_downstream + result.n_lost_radial + result.n_reflected
        assert total <= result.n_injected * 1.1  # allow small overlap

    def test_summary_string(self, synthetic_output: Path) -> None:
        result = compute_detachment(synthetic_output)
        summary = result.summary()
        assert "η_d (momentum)" in summary
        assert "η_d (particle)" in summary
        assert "η_d (energy)" in summary

    def test_insufficient_snapshots(self, tmp_path: Path) -> None:
        """Should raise with < 2 snapshots."""
        out = tmp_path / "single"
        out.mkdir()
        _write_synthetic_snapshot(out / "snap.h5")
        with pytest.raises(FileNotFoundError, match="2 snapshots"):
            compute_detachment(out)

    def test_no_files(self, tmp_path: Path) -> None:
        """Should raise with no HDF5 files."""
        with pytest.raises(FileNotFoundError):
            compute_detachment(tmp_path)

    def test_high_downstream_particles(self, tmp_path: Path) -> None:
        """With particles all downstream, particle efficiency should be high."""
        out = tmp_path / "downstream"
        out.mkdir()

        # Initial: particles at z ~ 0
        _write_synthetic_snapshot(
            out / "snap_0000.h5",
            z_range=(0.0, 0.1),
            r_range=(0.0, 0.05),
            pz_mean=5e-20,
            seed=10,
        )
        # Final: particles all far downstream (tightly clustered at z~2.0)
        _write_synthetic_snapshot(
            out / "snap_0001.h5",
            z_range=(1.95, 2.0),
            r_range=(0.0, 0.02),
            pz_mean=5e-20,
            seed=11,
        )

        result = compute_detachment(out)
        assert result.particle_based > 0.5
