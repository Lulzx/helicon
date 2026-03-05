"""Tests for helicon.postprocess.moments — moment computation from particle data."""

from __future__ import annotations

import numpy as np
import pytest

from helicon.postprocess.moments import MomentData


class TestMomentDataclass:
    def test_fields_accessible(self):
        nr, nz = 8, 16
        m = MomentData(
            species="D_plus",
            density=np.zeros((nr, nz)),
            vr_mean=np.zeros((nr, nz)),
            vz_mean=np.zeros((nr, nz)),
            pressure_rr=np.zeros((nr, nz)),
            pressure_zz=np.zeros((nr, nz)),
            r_grid=np.linspace(0, 0.1, nr),
            z_grid=np.linspace(-0.3, 1.0, nz),
        )
        assert m.species == "D_plus"
        assert m.density.shape == (nr, nz)
        assert m.vr_mean.shape == (nr, nz)
        assert m.vz_mean.shape == (nr, nz)
        assert m.pressure_rr.shape == (nr, nz)
        assert m.pressure_zz.shape == (nr, nz)
        assert m.r_grid.shape == (nr,)
        assert m.z_grid.shape == (nz,)

    def test_with_nonzero_values(self):
        nr, nz = 4, 8
        density = np.random.default_rng(42).random((nr, nz)) * 1e18
        m = MomentData(
            species="He_2plus",
            density=density,
            vr_mean=np.zeros((nr, nz)),
            vz_mean=np.ones((nr, nz)) * 50000.0,
            pressure_rr=np.zeros((nr, nz)),
            pressure_zz=np.zeros((nr, nz)),
            r_grid=np.linspace(0, 0.2, nr),
            z_grid=np.linspace(0.0, 2.0, nz),
        )
        assert np.all(m.density >= 0)
        assert np.allclose(m.vz_mean, 50000.0)


class TestBinIndexNumpyPath:
    """Test the numpy bin-index logic that underpins moment binning."""

    def _bin_indices(self, pos, edges, n_bins):
        """Replicate the numpy path from compute_moments."""
        return np.clip(np.searchsorted(edges, pos) - 1, 0, n_bins - 1)

    def test_center_of_first_bin(self):
        edges = np.linspace(0.0, 1.0, 11)  # 10 bins
        pos = np.array([0.05])  # midpoint of first bin
        idx = self._bin_indices(pos, edges, 10)
        assert idx[0] == 0

    def test_center_of_last_bin(self):
        edges = np.linspace(0.0, 1.0, 11)
        pos = np.array([0.95])  # midpoint of last bin
        idx = self._bin_indices(pos, edges, 10)
        assert idx[0] == 9

    def test_below_minimum_clips_to_zero(self):
        edges = np.linspace(0.0, 1.0, 11)
        pos = np.array([-0.5])
        idx = self._bin_indices(pos, edges, 10)
        assert idx[0] == 0

    def test_above_maximum_clips_to_last(self):
        edges = np.linspace(0.0, 1.0, 11)
        pos = np.array([2.0])
        idx = self._bin_indices(pos, edges, 10)
        assert idx[0] == 9

    def test_all_particles_binned_in_range(self):
        rng = np.random.default_rng(0)
        n = 1000
        edges = np.linspace(0.0, 5.0, 101)  # 100 bins
        pos = rng.uniform(0.0, 5.0, size=n)
        idx = self._bin_indices(pos, edges, 100)
        assert np.all(idx >= 0)
        assert np.all(idx < 100)


class TestComputeMomentsErrors:
    def test_raises_on_missing_h5_files(self, tmp_path):
        """compute_moments raises FileNotFoundError if no HDF5 files present."""
        from helicon.postprocess.moments import compute_moments

        with pytest.raises(FileNotFoundError, match="No HDF5 files"):
            compute_moments(tmp_path)
