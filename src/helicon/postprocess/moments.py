"""Velocity-space moment computation from particle data.

Computes density, bulk velocity, and pressure tensor from openPMD
particle output on the simulation grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from helicon._mlx_utils import HAS_MLX, resolve_backend, to_mx, to_np


@dataclass
class MomentData:
    """Grid-based moment quantities for a single species."""

    species: str
    density: np.ndarray  # (nr, nz) [m^-3]
    vr_mean: np.ndarray  # (nr, nz) [m/s]
    vz_mean: np.ndarray  # (nr, nz) [m/s]
    pressure_rr: np.ndarray  # (nr, nz) [Pa]
    pressure_zz: np.ndarray  # (nr, nz) [Pa]
    r_grid: np.ndarray  # (nr,) [m]
    z_grid: np.ndarray  # (nz,) [m]


def _compute_bin_indices_mlx(
    pos: np.ndarray,
    edges: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Compute bin indices for particle positions using MLX (faster cast-to-int).

    MLX evaluates ``(pos - min) / dz`` using float32 on Metal, then we cast to
    int32 and clip.  The scatter-add (``np.add.at``) still runs on CPU since
    MLX lacks a direct scatter-add equivalent.

    Parameters
    ----------
    pos : ndarray
        Particle positions along one axis.
    edges : ndarray
        Bin edges (length n_bins + 1).
    n_bins : int
        Number of bins.

    Returns
    -------
    ndarray of int32
        Bin index for each particle, clipped to [0, n_bins-1].
    """
    if not HAS_MLX:
        raise ImportError("MLX required for _compute_bin_indices_mlx")

    pos_mx = to_mx(pos)
    e_min = float(edges[0])
    dz = float(edges[1] - edges[0])
    idx_mx = (pos_mx - e_min) / dz
    idx_np = to_np(idx_mx).astype(np.int32)
    return np.clip(idx_np, 0, n_bins - 1)


def compute_moments(
    output_dir: str | Path,
    *,
    nz: int = 128,
    nr: int = 64,
    species_name: str = "D_plus",
    species_mass: float = 3.3435837724e-27,
    backend: str = "auto",
) -> MomentData:
    """Compute velocity-space moments on a uniform grid.

    Reads particle data from the latest openPMD snapshot, bins particles
    onto a (nr, nz) grid, and computes density, mean velocity, and
    pressure tensor components.

    Parameters
    ----------
    output_dir : path
        WarpX output directory.
    nz, nr : int
        Grid dimensions for moment computation.
    species_name : str
        Name of species to process.
    species_mass : float
        Particle mass [kg].
    backend : ``"auto"`` | ``"mlx"`` | ``"numpy"``
        Compute backend for bin index computation.  Scatter-add always runs
        on CPU regardless.
    """
    use_mlx = resolve_backend(backend) == "mlx"
    import h5py

    output_dir = Path(output_dir)

    # Find latest openPMD file
    h5_files = sorted(output_dir.glob("**/*.h5"))
    if not h5_files:
        msg = f"No HDF5 files found in {output_dir}"
        raise FileNotFoundError(msg)

    latest = h5_files[-1]

    with h5py.File(latest, "r") as f:
        # Navigate openPMD structure
        if "data" in f:
            iterations = sorted(f["data"].keys(), key=int)
            base = f["data"][iterations[-1]]
        else:
            base = f

        if "particles" not in base or species_name not in base["particles"]:
            msg = f"Species {species_name!r} not found in {latest}"
            raise ValueError(msg)

        sp = base["particles"][species_name]

        z_pos = sp["position"]["z"][:]
        r_pos = (
            sp["position"]["r"][:]
            if "r" in sp["position"]
            else (
                np.sqrt(sp["position"]["x"][:] ** 2 + sp["position"]["y"][:] ** 2)
                if "x" in sp["position"]
                else np.zeros_like(z_pos)
            )
        )

        pz = sp["momentum"]["z"][:]
        pr = sp["momentum"]["r"][:] if "r" in sp["momentum"] else (np.zeros_like(pz))

        w = sp["weighting"][:] if "weighting" in sp else np.ones_like(z_pos)

    # Convert momentum to velocity
    vz = pz / species_mass
    vr = pr / species_mass

    # Create grid
    z_min, z_max = z_pos.min(), z_pos.max()
    r_max = max(r_pos.max(), 1e-6)
    z_edges = np.linspace(z_min, z_max, nz + 1)
    r_edges = np.linspace(0.0, r_max, nr + 1)
    z_grid = 0.5 * (z_edges[:-1] + z_edges[1:])
    r_grid = 0.5 * (r_edges[:-1] + r_edges[1:])

    dz = z_edges[1] - z_edges[0]
    dr = r_edges[1] - r_edges[0]

    # Bin particles — MLX path uses (pos-min)/dz cast-to-int (faster on Metal)
    if use_mlx:
        iz = _compute_bin_indices_mlx(z_pos, z_edges, nz)
        ir = _compute_bin_indices_mlx(r_pos, r_edges, nr)
    else:
        iz = np.clip(np.searchsorted(z_edges, z_pos) - 1, 0, nz - 1)
        ir = np.clip(np.searchsorted(r_edges, r_pos) - 1, 0, nr - 1)

    density = np.zeros((nr, nz))
    vr_sum = np.zeros((nr, nz))
    vz_sum = np.zeros((nr, nz))
    vr2_sum = np.zeros((nr, nz))
    vz2_sum = np.zeros((nr, nz))

    np.add.at(density, (ir, iz), w)
    np.add.at(vr_sum, (ir, iz), w * vr)
    np.add.at(vz_sum, (ir, iz), w * vz)
    np.add.at(vr2_sum, (ir, iz), w * vr**2)
    np.add.at(vz2_sum, (ir, iz), w * vz**2)

    # Cell volume for density: 2π r dr dz (cylindrical)
    r_centers = r_grid[:, np.newaxis]
    cell_volume = 2 * np.pi * np.maximum(r_centers, 1e-10) * dr * dz

    mask = density > 0

    if use_mlx:
        import mlx.core as mx

        dens_mx = to_mx(density)
        cell_mx = to_mx(cell_volume)
        vrs_mx = to_mx(vr_sum)
        vzs_mx = to_mx(vz_sum)
        vr2s_mx = to_mx(vr2_sum)
        vz2s_mx = to_mx(vz2_sum)
        mask_mx = dens_mx > 0
        safe_dens = mx.where(mask_mx, dens_mx, 1.0)

        n_mx = mx.where(mask_mx, dens_mx / cell_mx, 0.0)
        vr_mean_mx = mx.where(mask_mx, vrs_mx / safe_dens, 0.0)
        vz_mean_mx = mx.where(mask_mx, vzs_mx / safe_dens, 0.0)
        vr2_mean_mx = mx.where(mask_mx, vr2s_mx / safe_dens, 0.0)
        vz2_mean_mx = mx.where(mask_mx, vz2s_mx / safe_dens, 0.0)

        p_rr_mx = mx.where(
            mask_mx,
            n_mx * float(species_mass) * (vr2_mean_mx - vr_mean_mx * vr_mean_mx),
            0.0,
        )
        p_zz_mx = mx.where(
            mask_mx,
            n_mx * float(species_mass) * (vz2_mean_mx - vz_mean_mx * vz_mean_mx),
            0.0,
        )

        n = to_np(n_mx)
        vr_mean = to_np(vr_mean_mx)
        vz_mean = to_np(vz_mean_mx)
        p_rr = to_np(p_rr_mx)
        p_zz = to_np(p_zz_mx)
    else:
        # Normalize
        n = np.zeros((nr, nz))
        n[mask] = density[mask] / cell_volume[mask]

        vr_mean = np.zeros((nr, nz))
        vz_mean = np.zeros((nr, nz))
        vr_mean[mask] = vr_sum[mask] / density[mask]
        vz_mean[mask] = vz_sum[mask] / density[mask]

        # Pressure = n * m * (<v^2> - <v>^2)
        p_rr = np.zeros((nr, nz))
        p_zz = np.zeros((nr, nz))
        if np.any(mask):
            vr2_mean = np.zeros((nr, nz))
            vz2_mean = np.zeros((nr, nz))
            vr2_mean[mask] = vr2_sum[mask] / density[mask]
            vz2_mean[mask] = vz2_sum[mask] / density[mask]
            p_rr[mask] = n[mask] * species_mass * (vr2_mean[mask] - vr_mean[mask] ** 2)
            p_zz[mask] = n[mask] * species_mass * (vz2_mean[mask] - vz_mean[mask] ** 2)

    return MomentData(
        species=species_name,
        density=n,
        vr_mean=vr_mean,
        vz_mean=vz_mean,
        pressure_rr=p_rr,
        pressure_zz=p_zz,
        r_grid=r_grid,
        z_grid=z_grid,
    )
