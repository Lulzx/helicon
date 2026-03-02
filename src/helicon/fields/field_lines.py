"""Field line tracer and topology classifier.

Traces magnetic field lines on the 2D axisymmetric (r, z) grid using
RK45 integration, and classifies them as open, closed, or separatrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

from helicon.fields.biot_savart import BField


class FieldLineType(Enum):
    """Classification of a magnetic field line."""

    OPEN = "open"
    CLOSED = "closed"
    SEPARATRIX = "separatrix"


@dataclass
class FieldLine:
    """A single traced field line."""

    r: np.ndarray  # radial coordinates [m]
    z: np.ndarray  # axial coordinates [m]
    line_type: FieldLineType
    start_r: float
    start_z: float
    psi_start: float  # magnetic flux function value at start


@dataclass
class FieldLineSet:
    """Collection of traced field lines with topology info."""

    lines: list[FieldLine]
    psi: np.ndarray  # flux function ψ(r,z), shape (nr, nz)
    r: np.ndarray  # 1-D coordinate
    z: np.ndarray  # 1-D coordinate
    separatrix_psi: float | None  # ψ value at separatrix (if found)


def compute_flux_function(bfield: BField) -> np.ndarray:
    """Compute the magnetic flux function ψ(r,z).

    For axisymmetric fields, ψ = ∫₀ʳ Bz(r',z) r' dr'.
    Contours of ψ are field lines in the poloidal plane.

    Returns array of shape (nr, nz).
    """
    r = bfield.r
    nr = len(r)
    dr = r[1] - r[0] if nr > 1 else 1.0

    psi = np.zeros_like(bfield.Bz)
    # Integrate Bz * r from r=0 outward using trapezoidal rule
    for i in range(1, nr):
        psi[i, :] = psi[i - 1, :] + 0.5 * dr * (
            bfield.Bz[i - 1, :] * r[i - 1] + bfield.Bz[i, :] * r[i]
        )

    return psi


def trace_field_line(
    bfield: BField,
    start_r: float,
    start_z: float,
    *,
    max_length: float = 100.0,
    max_steps: int = 10000,
    ds: float = 0.001,
) -> FieldLine:
    """Trace a single field line from a starting point using RK45.

    Parameters
    ----------
    bfield : BField
        Magnetic field data.
    start_r, start_z : float
        Starting point [m].
    max_length : float
        Maximum arc length to trace [m].
    max_steps : int
        Maximum integration steps.
    ds : float
        Initial step size [m].
    """
    r_grid = bfield.r
    z_grid = bfield.z

    # Build interpolators for Br and Bz
    Br_interp = RegularGridInterpolator(
        (r_grid, z_grid), bfield.Br, bounds_error=False, fill_value=0.0
    )
    Bz_interp = RegularGridInterpolator(
        (r_grid, z_grid), bfield.Bz, bounds_error=False, fill_value=0.0
    )

    def rhs(s: float, y: np.ndarray) -> np.ndarray:
        ri, zi = y
        ri = max(ri, 0.0)  # keep r >= 0
        pt = np.array([[ri, zi]])
        br = float(Br_interp(pt).item())
        bz = float(Bz_interp(pt).item())
        bmag = np.sqrt(br**2 + bz**2)
        if bmag < 1e-20:
            return np.array([0.0, 0.0])
        return np.array([br / bmag, bz / bmag])

    def out_of_bounds(s: float, y: np.ndarray) -> float:
        ri, zi = y
        # Stop if outside domain
        margin = 0.0
        if ri < -1e-6 or ri > r_grid[-1] + margin:
            return -1.0
        if zi < z_grid[0] - margin or zi > z_grid[-1] + margin:
            return -1.0
        return 1.0

    out_of_bounds.terminal = True  # type: ignore[attr-defined]

    sol = solve_ivp(
        rhs,
        [0.0, max_length],
        np.array([start_r, start_z]),
        method="RK45",
        max_step=ds * 10,
        events=out_of_bounds,
        dense_output=False,
        rtol=1e-6,
        atol=1e-9,
    )

    r_line = sol.y[0]
    z_line = sol.y[1]

    # Classify the line
    psi = compute_flux_function(bfield)
    psi_interp = RegularGridInterpolator(
        (r_grid, z_grid), psi, bounds_error=False, fill_value=np.nan
    )
    psi_start = float(psi_interp(np.array([[start_r, start_z]])).item())

    line_type = _classify_line(r_line, z_line, r_grid, z_grid)

    return FieldLine(
        r=r_line,
        z=z_line,
        line_type=line_type,
        start_r=start_r,
        start_z=start_z,
        psi_start=psi_start,
    )


def _classify_line(
    r_line: np.ndarray,
    z_line: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
) -> FieldLineType:
    """Classify a field line based on its endpoints."""
    if len(r_line) < 2:
        return FieldLineType.OPEN

    z_start = z_line[0]
    z_end = z_line[-1]
    r_end = r_line[-1]

    z_min, z_max = z_grid[0], z_grid[-1]
    r_max = r_grid[-1]

    # If it exits through downstream boundary → open
    exits_downstream = z_end >= z_max * 0.99
    exits_radially = r_end >= r_max * 0.99
    exits_upstream = z_end <= z_min * 0.99 if z_min < 0 else z_end <= z_min * 1.01

    if exits_downstream or exits_upstream:
        return FieldLineType.OPEN

    # If it returns to the axis or to a similar z → closed
    r_start = r_line[0]
    if abs(r_end - r_start) < 0.01 * r_max and abs(z_end - z_start) < 0.01 * (z_max - z_min):
        return FieldLineType.CLOSED

    # Near domain boundary in r → could be separatrix
    if exits_radially:
        return FieldLineType.SEPARATRIX

    return FieldLineType.OPEN


def trace_field_lines(
    bfield: BField,
    *,
    n_lines: int = 20,
    z_start: float | None = None,
    r_range: tuple[float, float] | None = None,
    max_length: float = 100.0,
    max_steps: int = 10000,
    ds: float = 0.001,
) -> FieldLineSet:
    """Trace multiple field lines from evenly spaced starting points.

    Parameters
    ----------
    bfield : BField
        Magnetic field data.
    n_lines : int
        Number of field lines to trace.
    z_start : float, optional
        Axial position to launch lines from. Defaults to domain midpoint.
    r_range : (r_min, r_max), optional
        Radial range for starting points. Defaults to (0, r_max).
    max_length : float
        Maximum arc length to trace [m].
    max_steps : int
        Maximum integration steps.
    ds : float
        Initial step size [m].
    """
    if z_start is None:
        z_start = 0.5 * (bfield.z[0] + bfield.z[-1])
    if r_range is None:
        r_range = (0.01 * bfield.r[-1], 0.95 * bfield.r[-1])

    r_starts = np.linspace(r_range[0], r_range[1], n_lines)

    lines = []
    for r0 in r_starts:
        line = trace_field_line(
            bfield,
            float(r0),
            z_start,
            max_length=max_length,
            max_steps=max_steps,
            ds=ds,
        )
        lines.append(line)

    psi = compute_flux_function(bfield)

    # Find separatrix ψ (if any separatrix lines exist)
    sep_psi = None
    sep_lines = [l for l in lines if l.line_type == FieldLineType.SEPARATRIX]
    if sep_lines:
        sep_psi = float(np.mean([l.psi_start for l in sep_lines]))

    return FieldLineSet(
        lines=lines,
        psi=psi,
        r=bfield.r,
        z=bfield.z,
        separatrix_psi=sep_psi,
    )


def classify_point(
    r: float,
    z: float,
    psi: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    separatrix_psi: float | None,
) -> FieldLineType:
    """Classify a single point's field line topology using ψ value.

    Parameters
    ----------
    r, z : float
        Point coordinates.
    psi : 2-D array
        Flux function.
    r_grid, z_grid : 1-D arrays
        Grid coordinates.
    separatrix_psi : float or None
        ψ value at the separatrix.
    """
    interp = RegularGridInterpolator(
        (r_grid, z_grid), psi, bounds_error=False, fill_value=np.nan
    )
    psi_val = float(interp(np.array([[r, z]])).item())

    if np.isnan(psi_val) or separatrix_psi is None:
        return FieldLineType.OPEN

    if abs(psi_val - separatrix_psi) < abs(separatrix_psi) * 0.01:
        return FieldLineType.SEPARATRIX
    elif psi_val < separatrix_psi:
        return FieldLineType.CLOSED
    else:
        return FieldLineType.OPEN
