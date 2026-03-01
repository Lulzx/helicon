"""FRC magnetic topology classification.

Identifies the separatrix, O-point, X-point, and classifies every grid
point as belonging to open or closed field-line regions using the magnetic
flux function psi(r, z).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magnozzlex.fields.biot_savart import BField


@dataclass
class FRCTopologyResult:
    """Topology classification of an FRC magnetic field."""

    is_open: np.ndarray  # (nr, nz), bool
    is_closed: np.ndarray  # (nr, nz), bool
    psi_rz: np.ndarray  # (nr, nz), flux function
    o_point_z: float
    o_point_r: float
    x_point_z: float
    psi_separatrix: float


def compute_flux_function(
    Bz: np.ndarray, r_arr: np.ndarray, dr: float
) -> np.ndarray:
    """Compute psi(r,z) = integral_0^r (-r' * Bz(r',z)) dr'.

    Parameters
    ----------
    Bz : ndarray, shape (nr, nz)
    r_arr : ndarray, shape (nr,)
    dr : float

    Returns
    -------
    psi : ndarray, shape (nr, nz)
    """
    nr = Bz.shape[0]
    psi = np.zeros_like(Bz)
    for i in range(1, nr):
        psi[i, :] = psi[i - 1, :] + 0.5 * dr * (
            -r_arr[i - 1] * Bz[i - 1, :] + (-r_arr[i] * Bz[i, :])
        )
    return psi


def find_frc_topology(bfield: BField) -> FRCTopologyResult:
    """Classify every grid point as open or closed for an FRC field.

    Parameters
    ----------
    bfield : BField
        Magnetic field on an axisymmetric (r, z) grid.

    Returns
    -------
    FRCTopologyResult
    """
    r = bfield.r
    nr = len(r)
    dr = r[1] - r[0] if nr > 1 else 1.0

    psi = compute_flux_function(bfield.Bz, r, dr)

    # --- O-point: max |psi| in interior (exclude boundary rows/cols) ---
    interior = psi[1:-1, 1:-1] if nr > 2 and psi.shape[1] > 2 else psi
    abs_interior = np.abs(interior)
    if interior.size == 0:
        # Degenerate grid
        o_idx_r, o_idx_z = 0, 0
    else:
        flat_idx = np.argmax(abs_interior)
        o_idx_r_int, o_idx_z_int = np.unravel_index(flat_idx, interior.shape)
        # Offset back to full-grid indices
        o_idx_r = o_idx_r_int + (1 if nr > 2 and psi.shape[1] > 2 else 0)
        o_idx_z = o_idx_z_int + (1 if nr > 2 and psi.shape[1] > 2 else 0)

    o_point_r = float(r[o_idx_r])
    o_point_z = float(bfield.z[o_idx_z])
    psi_o = psi[o_idx_r, o_idx_z]

    # --- X-point / separatrix ---
    # For FRC: separatrix is where psi = 0 (reversed field makes psi cross zero).
    # Find the X-point as the location on the axis (r index closest to O-point r)
    # where psi crosses zero downstream of the O-point.
    psi_separatrix = 0.0

    # Look for sign change along the r = o_idx_r slice, downstream of O-point
    psi_slice = psi[o_idx_r, :]
    x_point_z = float(bfield.z[-1])  # default: end of domain

    # Search for zero crossing downstream (z > o_point_z)
    z_arr = bfield.z
    downstream = np.where(z_arr > o_point_z)[0]
    found_x = False
    for j in range(len(downstream) - 1):
        j0 = downstream[j]
        j1 = downstream[j + 1]
        if psi_slice[j0] * psi_slice[j1] < 0:
            # Linear interpolation for zero crossing
            frac = psi_slice[j0] / (psi_slice[j0] - psi_slice[j1])
            x_point_z = float(z_arr[j0] + frac * (z_arr[j1] - z_arr[j0]))
            found_x = True
            break

    if not found_x:
        # Also search upstream (z < o_point_z) as fallback
        upstream = np.where(z_arr < o_point_z)[0]
        for j in range(len(upstream) - 1, 0, -1):
            j0 = upstream[j - 1]
            j1 = upstream[j]
            if psi_slice[j0] * psi_slice[j1] < 0:
                frac = psi_slice[j0] / (psi_slice[j0] - psi_slice[j1])
                x_point_z = float(z_arr[j0] + frac * (z_arr[j1] - z_arr[j0]))
                found_x = True
                break

    # --- Classify open / closed ---
    if psi_o > 0:
        is_closed = psi > psi_separatrix
    elif psi_o < 0:
        is_closed = psi < psi_separatrix
    else:
        # Degenerate: no FRC topology, everything open
        is_closed = np.zeros_like(psi, dtype=bool)

    is_open = ~is_closed

    return FRCTopologyResult(
        is_open=is_open,
        is_closed=is_closed,
        psi_rz=psi,
        o_point_z=o_point_z,
        o_point_r=o_point_r,
        x_point_z=x_point_z,
        psi_separatrix=psi_separatrix,
    )
