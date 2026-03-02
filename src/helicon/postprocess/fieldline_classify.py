"""Particle classification by field line topology.

Tags each particle with its field line type (open/closed/separatrix)
based on the magnetic flux function ψ. This is essential for computing
detachment efficiency correctly: only particles on open field lines
contribute to net thrust.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from helicon.fields.biot_savart import BField
from helicon.fields.field_lines import (
    compute_flux_function,
    trace_field_lines,
)


@dataclass
class ParticleClassification:
    """Result of particle field-line classification."""

    species: str
    n_open: int
    n_closed: int
    n_separatrix: int
    n_total: int
    labels: np.ndarray  # 0=open, 1=closed, 2=separatrix
    positions_r: np.ndarray
    positions_z: np.ndarray


def classify_particles(
    output_dir: str | Path,
    bfield: BField,
    *,
    species_name: str = "D_plus",
    separatrix_psi: float | None = None,
) -> ParticleClassification:
    """Classify particles by their field line topology.

    Uses the magnetic flux function ψ(r,z) to determine whether each
    particle sits on an open, closed, or separatrix field line.

    Parameters
    ----------
    output_dir : path
        WarpX output directory with particle data.
    bfield : BField
        Pre-computed magnetic field (for flux function).
    species_name : str
        Species to classify.
    separatrix_psi : float, optional
        ψ value at the separatrix. If None, auto-detected from field
        line tracing.
    """
    import h5py

    output_dir = Path(output_dir)

    # Compute flux function
    psi = compute_flux_function(bfield)

    # Auto-detect separatrix if not provided
    if separatrix_psi is None:
        fls = trace_field_lines(bfield, n_lines=30)
        separatrix_psi = fls.separatrix_psi

    # Build ψ interpolator
    psi_interp = RegularGridInterpolator(
        (bfield.r, bfield.z), psi, bounds_error=False, fill_value=np.nan
    )

    # Read particle positions
    h5_files = sorted(output_dir.glob("**/*.h5"))
    if not h5_files:
        msg = f"No HDF5 files found in {output_dir}"
        raise FileNotFoundError(msg)

    with h5py.File(h5_files[-1], "r") as f:
        base = _navigate_openpmd(f)
        sp = base["particles"][species_name]
        z = sp["position"]["z"][:]
        r = sp["position"]["r"][:] if "r" in sp["position"] else np.zeros_like(z)

    # Evaluate ψ at each particle location
    points = np.column_stack([np.abs(r), z])
    psi_vals = psi_interp(points)

    # Classify
    labels = np.zeros(len(z), dtype=np.int32)  # default: open
    if separatrix_psi is not None:
        tol = abs(separatrix_psi) * 0.01 if separatrix_psi != 0 else 1e-10
        is_closed = psi_vals < separatrix_psi
        is_sep = np.abs(psi_vals - separatrix_psi) < tol
        labels[is_closed & ~is_sep] = 1  # closed
        labels[is_sep] = 2  # separatrix

    n_open = int(np.sum(labels == 0))
    n_closed = int(np.sum(labels == 1))
    n_sep = int(np.sum(labels == 2))

    return ParticleClassification(
        species=species_name,
        n_open=n_open,
        n_closed=n_closed,
        n_separatrix=n_sep,
        n_total=len(z),
        labels=labels,
        positions_r=r,
        positions_z=z,
    )


def _navigate_openpmd(f: Any) -> Any:
    if "data" in f:
        iterations = sorted(f["data"].keys(), key=int)
        return f["data"][iterations[-1]]
    return f
