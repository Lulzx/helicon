"""Tests for helicon.fields.biot_savart_3d — 3D Biot-Savart solver."""

from __future__ import annotations

import numpy as np
import pytest

from helicon.fields.biot_savart_3d import BField3D, Coil3D, Grid3D, compute_bfield_3d


def _small_grid() -> Grid3D:
    """A small 8×8×16 grid suitable for fast unit tests."""
    return Grid3D(
        x_min=-0.2, x_max=0.2,
        y_min=-0.2, y_max=0.2,
        z_min=-0.3, z_max=1.5,
        nx=8, ny=8, nz=16,
    )


def _single_coil(z: float = 0.0, r: float = 0.1, I: float = 10000.0) -> Coil3D:
    return Coil3D(z=z, r=r, I=I)


class TestCoil3DDataclass:
    def test_frozen(self):
        coil = _single_coil()
        with pytest.raises(AttributeError):
            coil.I = 999  # type: ignore[misc]

    def test_fields(self):
        coil = Coil3D(z=0.5, r=0.15, I=5000.0)
        assert coil.z == 0.5
        assert coil.r == 0.15
        assert coil.I == 5000.0


class TestGrid3DDataclass:
    def test_fields(self):
        g = _small_grid()
        assert g.nx == 8
        assert g.ny == 8
        assert g.nz == 16

    def test_frozen(self):
        g = _small_grid()
        with pytest.raises(AttributeError):
            g.nx = 32  # type: ignore[misc]


class TestComputeBfield3D:
    def test_returns_bfield3d(self):
        bfield = compute_bfield_3d([_single_coil()], _small_grid(), backend="numpy")
        assert isinstance(bfield, BField3D)

    def test_output_shape(self):
        grid = _small_grid()
        bfield = compute_bfield_3d([_single_coil()], grid, backend="numpy")
        assert bfield.Bx.shape == (grid.nx, grid.ny, grid.nz)
        assert bfield.By.shape == (grid.nx, grid.ny, grid.nz)
        assert bfield.Bz.shape == (grid.nx, grid.ny, grid.nz)

    def test_coordinate_arrays(self):
        grid = _small_grid()
        bfield = compute_bfield_3d([_single_coil()], grid, backend="numpy")
        assert bfield.x.shape == (grid.nx,)
        assert bfield.y.shape == (grid.ny,)
        assert bfield.z.shape == (grid.nz,)

    def test_field_nonzero(self):
        bfield = compute_bfield_3d([_single_coil()], _small_grid(), backend="numpy")
        assert np.any(bfield.Bz != 0.0)

    def test_zero_current_gives_zero_field(self):
        coil = Coil3D(z=0.0, r=0.1, I=0.0)
        bfield = compute_bfield_3d([coil], _small_grid(), backend="numpy")
        assert np.allclose(bfield.Bx, 0.0)
        assert np.allclose(bfield.By, 0.0)
        assert np.allclose(bfield.Bz, 0.0)

    def test_backend_stored(self):
        bfield = compute_bfield_3d([_single_coil()], _small_grid(), backend="numpy")
        assert bfield.backend == "numpy"

    def test_coils_stored(self):
        coil = _single_coil()
        bfield = compute_bfield_3d([coil], _small_grid(), backend="numpy")
        assert len(bfield.coils) == 1
        assert bfield.coils[0] is coil

    def test_linearity_two_coils(self):
        """Field from 2 identical coils should be ~2× field from 1 coil."""
        grid = _small_grid()
        coil = _single_coil(I=1000.0)
        b1 = compute_bfield_3d([coil], grid, backend="numpy")
        b2 = compute_bfield_3d([coil, coil], grid, backend="numpy")
        np.testing.assert_allclose(b2.Bz, 2.0 * b1.Bz, rtol=1e-10)


class TestBField3DProperties:
    def test_bmag_shape(self):
        bfield = compute_bfield_3d([_single_coil()], _small_grid(), backend="numpy")
        assert bfield.Bmag.shape == bfield.Bx.shape

    def test_bmag_nonnegative(self):
        bfield = compute_bfield_3d([_single_coil()], _small_grid(), backend="numpy")
        assert np.all(bfield.Bmag >= 0.0)

    def test_on_axis_shape(self):
        grid = _small_grid()
        bfield = compute_bfield_3d([_single_coil()], grid, backend="numpy")
        assert bfield.on_axis().shape == (grid.nz,)

    def test_mirror_ratio_positive(self):
        bfield = compute_bfield_3d([_single_coil()], _small_grid(), backend="numpy")
        assert bfield.mirror_ratio() >= 1.0
