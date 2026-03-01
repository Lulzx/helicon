"""Tests for magnozzlex.fields.field_lines module."""

from __future__ import annotations

import numpy as np
import pytest

from magnozzlex.fields.biot_savart import BField, Coil, Grid, compute_bfield
from magnozzlex.fields.field_lines import (
    FieldLine,
    FieldLineSet,
    FieldLineType,
    classify_point,
    compute_flux_function,
    trace_field_line,
    trace_field_lines,
)


@pytest.fixture
def single_coil_bfield() -> BField:
    """Simple single-coil magnetic field for testing."""
    coils = [Coil(z=0.0, r=0.10, I=50000)]
    grid = Grid(z_min=-0.5, z_max=1.5, r_max=0.4, nz=128, nr=64)
    return compute_bfield(coils, grid, backend="numpy")


@pytest.fixture
def mirror_bfield() -> BField:
    """Two-coil magnetic mirror configuration."""
    coils = [
        Coil(z=0.0, r=0.10, I=50000),
        Coil(z=0.5, r=0.10, I=50000),
    ]
    grid = Grid(z_min=-0.3, z_max=0.8, r_max=0.3, nz=128, nr=64)
    return compute_bfield(coils, grid, backend="numpy")


class TestFluxFunction:
    """Tests for compute_flux_function."""

    def test_shape(self, single_coil_bfield: BField) -> None:
        psi = compute_flux_function(single_coil_bfield)
        assert psi.shape == single_coil_bfield.Bz.shape

    def test_zero_on_axis(self, single_coil_bfield: BField) -> None:
        """ψ(r=0, z) = 0 since integral starts from r=0."""
        psi = compute_flux_function(single_coil_bfield)
        assert np.allclose(psi[0, :], 0.0)

    def test_positive_for_positive_Bz(self, single_coil_bfield: BField) -> None:
        """ψ should be positive where Bz is predominantly positive."""
        psi = compute_flux_function(single_coil_bfield)
        # At some inner radius, ψ should be positive (Bz is positive on axis)
        mid_r = len(single_coil_bfield.r) // 4
        mid_z = len(single_coil_bfield.z) // 2
        assert psi[mid_r, mid_z] > 0

    def test_monotonic_in_r_near_axis(self, single_coil_bfield: BField) -> None:
        """Near the axis ψ should increase with r (cumulative integral of positive Bz)."""
        psi = compute_flux_function(single_coil_bfield)
        mid_z = len(single_coil_bfield.z) // 2
        # First ~quarter of r should be monotonically increasing
        n = len(single_coil_bfield.r) // 4
        diffs = np.diff(psi[:n, mid_z])
        assert np.all(diffs >= 0)


class TestTraceFieldLine:
    """Tests for trace_field_line."""

    def test_returns_field_line(self, single_coil_bfield: BField) -> None:
        line = trace_field_line(single_coil_bfield, 0.05, 0.0)
        assert isinstance(line, FieldLine)
        assert len(line.r) > 1
        assert len(line.z) > 1

    def test_r_nonnegative(self, single_coil_bfield: BField) -> None:
        """r should remain >= 0 during tracing."""
        line = trace_field_line(single_coil_bfield, 0.02, 0.0)
        assert np.all(line.r >= -1e-6)

    def test_axis_stays_near_axis(self, single_coil_bfield: BField) -> None:
        """A line started very near r=0 should stay near the axis."""
        line = trace_field_line(single_coil_bfield, 0.001, 0.0)
        # Allow some drift but should stay within reason
        assert np.max(line.r) < 0.1

    def test_line_type_assigned(self, single_coil_bfield: BField) -> None:
        line = trace_field_line(single_coil_bfield, 0.05, 0.0)
        assert isinstance(line.line_type, FieldLineType)

    def test_open_line_classified(self, single_coil_bfield: BField) -> None:
        """A line starting near the coil should be classified as open."""
        line = trace_field_line(single_coil_bfield, 0.05, 0.0)
        assert line.line_type == FieldLineType.OPEN
        # Should have traveled a meaningful distance
        arc_length = np.sum(np.sqrt(np.diff(line.r) ** 2 + np.diff(line.z) ** 2))
        assert arc_length > 0.1


class TestTraceFieldLines:
    """Tests for trace_field_lines."""

    def test_returns_field_line_set(self, single_coil_bfield: BField) -> None:
        fls = trace_field_lines(single_coil_bfield, n_lines=5)
        assert isinstance(fls, FieldLineSet)
        assert len(fls.lines) == 5
        assert fls.psi.shape == single_coil_bfield.Bz.shape

    def test_n_lines_correct(self, single_coil_bfield: BField) -> None:
        fls = trace_field_lines(single_coil_bfield, n_lines=10)
        assert len(fls.lines) == 10

    def test_different_start_r(self, single_coil_bfield: BField) -> None:
        """Each line should start at a different r."""
        fls = trace_field_lines(single_coil_bfield, n_lines=5)
        starts = [l.start_r for l in fls.lines]
        assert len(set(starts)) == len(starts)

    def test_mirror_has_mixed_topology(self, mirror_bfield: BField) -> None:
        """A mirror configuration may have both open and closed lines."""
        fls = trace_field_lines(mirror_bfield, n_lines=15)
        types = {l.line_type for l in fls.lines}
        # At minimum we expect open lines
        assert FieldLineType.OPEN in types


class TestClassifyPoint:
    """Tests for classify_point."""

    def test_returns_field_line_type(self, single_coil_bfield: BField) -> None:
        psi = compute_flux_function(single_coil_bfield)
        result = classify_point(
            0.05,
            0.5,
            psi,
            single_coil_bfield.r,
            single_coil_bfield.z,
            separatrix_psi=None,
        )
        assert isinstance(result, FieldLineType)

    def test_no_separatrix_returns_open(self, single_coil_bfield: BField) -> None:
        """Without separatrix info, all points should be classified as open."""
        psi = compute_flux_function(single_coil_bfield)
        result = classify_point(
            0.05,
            0.5,
            psi,
            single_coil_bfield.r,
            single_coil_bfield.z,
            separatrix_psi=None,
        )
        assert result == FieldLineType.OPEN

    def test_on_separatrix(self, single_coil_bfield: BField) -> None:
        """A point with ψ ≈ separatrix_psi should be classified as separatrix."""
        psi = compute_flux_function(single_coil_bfield)
        # Pick a psi value from the grid as the "separatrix"
        mid_r = len(single_coil_bfield.r) // 3
        mid_z = len(single_coil_bfield.z) // 2
        sep_psi = float(psi[mid_r, mid_z])

        result = classify_point(
            single_coil_bfield.r[mid_r],
            single_coil_bfield.z[mid_z],
            psi,
            single_coil_bfield.r,
            single_coil_bfield.z,
            separatrix_psi=sep_psi,
        )
        assert result == FieldLineType.SEPARATRIX
