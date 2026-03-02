"""Tests for helicon.fields.frc_topology — FRC topology classification."""

from __future__ import annotations

import numpy as np
import pytest

from helicon.fields.biot_savart import BField, Coil, Grid, compute_bfield
from helicon.fields.frc_topology import (
    FRCTopologyResult,
    compute_flux_function,
    find_frc_topology,
)

# A simple solenoid (no field reversal)
SOLENOID_COILS = [Coil(z=0.0, r=0.1, I=1000.0)]
GRID = Grid(z_min=-0.3, z_max=0.3, r_max=0.2, nz=32, nr=16)


def _make_solenoid_bfield():
    return compute_bfield(SOLENOID_COILS, GRID, backend="numpy")


def _make_frc_bfield():
    """Two opposing coils to create an FRC-like field reversal."""
    coils = [
        Coil(z=-0.15, r=0.1, I=1000.0),
        Coil(z=0.15, r=0.1, I=1000.0),
        # Mirror coil with reversed current to create field reversal
        Coil(z=0.0, r=0.05, I=-3000.0),
    ]
    grid = Grid(z_min=-0.3, z_max=0.3, r_max=0.2, nz=64, nr=32)
    return compute_bfield(coils, grid, backend="numpy")


# --- compute_flux_function ---


class TestComputeFluxFunction:
    def test_shape(self):
        bf = _make_solenoid_bfield()
        dr = bf.r[1] - bf.r[0]
        psi = compute_flux_function(bf.Bz, bf.r, dr)
        assert psi.shape == bf.Bz.shape

    def test_zero_at_r0(self):
        """psi should be zero at r=0 (first row)."""
        bf = _make_solenoid_bfield()
        dr = bf.r[1] - bf.r[0]
        psi = compute_flux_function(bf.Bz, bf.r, dr)
        np.testing.assert_array_equal(psi[0, :], 0.0)

    def test_monotonic_for_uniform_Bz(self):
        """For uniform Bz>0, psi = integral(-r*Bz) dr should be monotonically
        decreasing (more negative) in r."""
        nr, nz = 20, 10
        r_arr = np.linspace(0.0, 0.2, nr)
        dr = r_arr[1] - r_arr[0]
        Bz = np.ones((nr, nz))  # uniform positive Bz
        psi = compute_flux_function(Bz, r_arr, dr)
        # Each row should be <= previous (integrand is -r*Bz < 0 for r>0)
        for i in range(1, nr):
            assert np.all(psi[i, :] <= psi[i - 1, :] + 1e-15)


# --- find_frc_topology ---


class TestFindFrcTopology:
    def test_returns_result(self):
        bf = _make_solenoid_bfield()
        result = find_frc_topology(bf)
        assert isinstance(result, FRCTopologyResult)

    def test_shapes(self):
        bf = _make_solenoid_bfield()
        result = find_frc_topology(bf)
        assert result.is_open.shape == bf.Bz.shape
        assert result.is_closed.shape == bf.Bz.shape
        assert result.psi_rz.shape == bf.Bz.shape

    def test_bool_arrays(self):
        bf = _make_solenoid_bfield()
        result = find_frc_topology(bf)
        assert result.is_open.dtype == bool
        assert result.is_closed.dtype == bool

    def test_mutually_exclusive(self):
        """No grid point should be both open and closed."""
        bf = _make_solenoid_bfield()
        result = find_frc_topology(bf)
        assert not np.any(result.is_open & result.is_closed)

    def test_exhaustive(self):
        """Every grid point is either open or closed."""
        bf = _make_solenoid_bfield()
        result = find_frc_topology(bf)
        assert np.all(result.is_open | result.is_closed)

    def test_degenerate_solenoid(self):
        """A simple solenoid has no FRC — expect mostly open or mostly closed
        depending on psi sign, but is_open and is_closed should still be valid."""
        bf = _make_solenoid_bfield()
        result = find_frc_topology(bf)
        # Just verify consistency: mutually exclusive + exhaustive
        assert not np.any(result.is_open & result.is_closed)
        assert np.all(result.is_open | result.is_closed)

    def test_frc_has_closed_region(self):
        """An FRC-like field should have some closed field lines."""
        bf = _make_frc_bfield()
        result = find_frc_topology(bf)
        assert np.any(result.is_closed), "FRC field should have closed region"
        assert np.any(result.is_open), "FRC field should have open region"
