"""Biot-Savart magnetic field solver for circular coil configurations.

Computes the applied B-field on a 2D axisymmetric (r, z) grid from a set of
circular current loops.  Two backends:

* **NumPy/SciPy** — exact elliptic-integral formulae (K, E)
* **MLX** — numerical azimuthal integration, fully differentiable via
  ``mlx.core.grad`` for gradient-based coil optimisation
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Physical constant — exact by definition
MU_0 = 4.0 * math.pi * 1e-7  # T·m/A

# ---------------------------------------------------------------------------
# Try to import optional backends
# ---------------------------------------------------------------------------
try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Coil:
    """A single circular current loop in SI units.

    Parameters
    ----------
    z : float
        Axial position of the coil centre [m].
    r : float
        Radius of the coil [m].
    I : float
        Current (ampere-turns) [A].
    """

    z: float
    r: float
    I: float


@dataclass(frozen=True)
class Grid:
    """Axisymmetric (r, z) computation grid.

    The grid spans ``z ∈ [z_min, z_max]`` and ``r ∈ [0, r_max]``.

    Parameters
    ----------
    z_min, z_max : float
        Axial extent [m].
    r_max : float
        Radial extent [m].
    nz, nr : int
        Number of grid points along each axis.
    """

    z_min: float
    z_max: float
    r_max: float
    nz: int
    nr: int


@dataclass
class BField:
    """Result container — always stores NumPy arrays for interop.

    Attributes
    ----------
    Br, Bz : np.ndarray
        Radial and axial magnetic field components [T], shape ``(nr, nz)``.
    r, z : np.ndarray
        1-D coordinate arrays [m].
    coils : list[Coil]
        Coil configuration that produced this field.
    backend : str
        Backend used (``"numpy"`` or ``"mlx"``).
    """

    Br: np.ndarray
    Bz: np.ndarray
    r: np.ndarray
    z: np.ndarray
    coils: list[Coil]
    backend: str

    # -- Visualization -------------------------------------------------------
    def plot(
        self,
        *,
        component: str = "Bz",
        field_lines: bool = True,
        n_field_lines: int = 8,
        ax=None,
        cmap: str = "RdBu_r",
        figsize: tuple = (10, 5),
    ):
        """Plot the B-field on the (z, r) plane.

        Parameters
        ----------
        component : ``"Bz"`` | ``"Br"`` | ``"Bmag"``
            Field component to show as a filled contour.
        field_lines : bool
            Overlay field lines via contours of the flux function ψ.
        n_field_lines : int
            Number of field line contours to draw.
        ax : matplotlib Axes, optional
            Axes to draw on. Creates new figure if None.
        cmap : str
            Matplotlib colormap.
        figsize : tuple
            Figure size when creating a new figure.

        Returns
        -------
        fig, ax
            Matplotlib figure and axes.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for BField.plot()") from exc

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        Z, R = np.meshgrid(self.z, self.r)

        if component == "Bz":
            values = self.Bz
            label = "$B_z$ (T)"
        elif component == "Br":
            values = self.Br
            label = "$B_r$ (T)"
        elif component == "Bmag":
            values = np.sqrt(self.Br**2 + self.Bz**2)
            label = "$|B|$ (T)"
        else:
            raise ValueError(f"Unknown component {component!r}. Use 'Bz', 'Br', or 'Bmag'.")

        pcm = ax.pcolormesh(Z, R, values, cmap=cmap, shading="auto")
        fig.colorbar(pcm, ax=ax, label=label)

        if field_lines:
            # Flux function ψ(r,z) = cumulative integral of r*B_z over r (up to sign)
            dr = self.r[1] - self.r[0] if len(self.r) > 1 else 1.0
            psi = np.cumsum(self.r[:, None] * self.Bz, axis=0) * dr
            with contextlib.suppress(Exception):
                ax.contour(Z, R, psi, n_field_lines, colors="k", linewidths=0.6, alpha=0.5)

        # Mark coil positions
        for coil in self.coils:
            ax.plot(coil.z, coil.r, "ko", ms=6, zorder=5)
            ax.plot(coil.z, -coil.r if min(self.r) < 0 else coil.r, "ko", ms=6, zorder=5)

        ax.set_xlabel("z (m)")
        ax.set_ylabel("r (m)")
        ax.set_title(f"Magnetic Field — {component}")

        return fig, ax

    # -- HDF5 persistence ---------------------------------------------------
    def save(self, path: str) -> None:
        """Write field data + coil metadata to an HDF5 file."""
        if not HAS_H5PY:
            raise ImportError("h5py is required for BField.save()")
        with h5py.File(path, "w") as f:
            f.create_dataset("Br", data=self.Br)
            f.create_dataset("Bz", data=self.Bz)
            f.create_dataset("r", data=self.r)
            f.create_dataset("z", data=self.z)
            f.attrs["backend"] = self.backend
            f.attrs["n_coils"] = len(self.coils)
            for i, c in enumerate(self.coils):
                f.attrs[f"coil_{i}_z"] = c.z
                f.attrs[f"coil_{i}_r"] = c.r
                f.attrs[f"coil_{i}_I"] = c.I

    @classmethod
    def load(cls, path: str) -> BField:
        """Load a previously saved BField from HDF5."""
        if not HAS_H5PY:
            raise ImportError("h5py is required for BField.load()")
        with h5py.File(path, "r") as f:
            Br = f["Br"][:]
            Bz = f["Bz"][:]
            r = f["r"][:]
            z = f["z"][:]
            backend = f.attrs["backend"]
            n_coils = int(f.attrs["n_coils"])
            coils = [
                Coil(
                    z=float(f.attrs[f"coil_{i}_z"]),
                    r=float(f.attrs[f"coil_{i}_r"]),
                    I=float(f.attrs[f"coil_{i}_I"]),
                )
                for i in range(n_coils)
            ]
        return cls(Br=Br, Bz=Bz, r=r, z=z, coils=coils, backend=backend)


# ---------------------------------------------------------------------------
# NumPy backend — exact elliptic-integral formulae
# ---------------------------------------------------------------------------
def _bfield_numpy_single_coil(
    z_coil: float,
    a: float,
    I: float,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Field from one coil on the full 2-D meshgrid (r_grid, z_grid).

    Uses exact elliptic-integral formula (Jackson §5.5).
    """
    from scipy.special import ellipe, ellipk

    dz = z_grid - z_coil
    C = MU_0 * I / (2.0 * math.pi)

    Br = np.zeros_like(r_grid)
    Bz = np.zeros_like(r_grid)

    # --- On-axis points (r == 0) -------------------------------------------
    on_axis = r_grid == 0.0
    if np.any(on_axis):
        dz_ax = dz[on_axis]
        denom = (a**2 + dz_ax**2) ** 1.5
        Bz[on_axis] = MU_0 * I * a**2 / (2.0 * denom)
        # B_r = 0 on axis (already initialised)

    # --- Off-axis points ---------------------------------------------------
    off = ~on_axis
    if np.any(off):
        r_off = r_grid[off]
        dz_off = dz[off]

        alpha_sq = (r_off + a) ** 2 + dz_off**2
        beta_sq = (r_off - a) ** 2 + dz_off**2
        k_sq = 4.0 * a * r_off / alpha_sq

        # Singularity guard: at the coil wire (r==a, dz==0), beta_sq==0.
        # The field is physically singular there; replace with large finite value.
        singular = beta_sq < 1e-30
        beta_sq_safe = np.where(singular, 1e-30, beta_sq)

        # Clamp k_sq to [0, 1) for numerical safety
        k_sq = np.clip(k_sq, 0.0, 1.0 - 1e-15)

        K = ellipk(k_sq)
        E = ellipe(k_sq)
        alpha = np.sqrt(alpha_sq)

        Bz_off = (C / alpha) * (K + (a**2 - r_off**2 - dz_off**2) / beta_sq_safe * E)
        Br_off = (C * dz_off / (alpha * r_off)) * (
            -K + (a**2 + r_off**2 + dz_off**2) / beta_sq_safe * E
        )

        # Zero out singular points (physically: field diverges, numerically safe)
        Bz_off = np.where(singular, 0.0, Bz_off)
        Br_off = np.where(singular, 0.0, Br_off)

        Bz[off] = Bz_off
        Br[off] = Br_off

    return Br, Bz


def _compute_numpy(
    coils: Sequence[Coil], grid: Grid
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute total field using the NumPy/SciPy backend."""
    r = np.linspace(0.0, grid.r_max, grid.nr)
    z = np.linspace(grid.z_min, grid.z_max, grid.nz)
    Z, R = np.meshgrid(z, r)  # shapes (nr, nz)

    Br_total = np.zeros_like(R)
    Bz_total = np.zeros_like(R)
    for coil in coils:
        dBr, dBz = _bfield_numpy_single_coil(coil.z, coil.r, coil.I, R, Z)
        Br_total += dBr
        Bz_total += dBz

    return Br_total, Bz_total, r, z


# ---------------------------------------------------------------------------
# MLX backend — numerical azimuthal integration (differentiable)
# ---------------------------------------------------------------------------
def compute_bfield_mlx_differentiable(
    coil_params: mx.array,
    grid_r: mx.array,
    grid_z: mx.array,
    n_phi: int = 128,
) -> tuple[mx.array, mx.array]:
    """MLX-native differentiable Biot-Savart computation.

    Designed for composition with ``mlx.core.grad()``.  Does **not** call
    ``mx.eval`` so the computation graph is preserved.

    Parameters
    ----------
    coil_params : mx.array, shape (N_coils, 3)
        Each row is ``[z_coil, radius, current]`` in SI.
    grid_r, grid_z : mx.array, shape (N_pts,)
        Flattened grid coordinates.
    n_phi : int
        Number of azimuthal quadrature points.

    Returns
    -------
    Br, Bz : mx.array, shape (N_pts,)
    """
    if not HAS_MLX:
        raise ImportError("MLX is required for compute_bfield_mlx_differentiable")

    phi = mx.linspace(0.0, 2.0 * math.pi, n_phi + 1)[:-1]  # exclude endpoint
    cos_phi = mx.cos(phi)  # (N_phi,)

    N_pts = grid_r.shape[0]
    N_coils = coil_params.shape[0]

    Br_total = mx.zeros((N_pts,))
    Bz_total = mx.zeros((N_pts,))

    for i in range(N_coils):
        z_c = coil_params[i, 0]
        a = coil_params[i, 1]
        I_c = coil_params[i, 2]

        dz = grid_z - z_c  # (N_pts,)

        # Broadcast: r -> (N_pts, 1), cos_phi -> (1, N_phi)
        r_2d = mx.reshape(grid_r, (N_pts, 1))
        dz_2d = mx.reshape(dz, (N_pts, 1))
        cos_2d = mx.reshape(cos_phi, (1, n_phi))

        R_sq = r_2d * r_2d + a * a - 2.0 * a * r_2d * cos_2d + dz_2d * dz_2d
        R_sq = mx.maximum(R_sq, 1e-20)  # singularity guard

        R_inv3 = R_sq ** (-1.5)

        prefactor = MU_0 * I_c * a / (2.0 * n_phi)

        # Sum over phi (axis=1)
        Br_total = Br_total + prefactor * mx.sum(cos_2d * dz_2d * R_inv3, axis=1)
        Bz_total = Bz_total + prefactor * mx.sum((a - r_2d * cos_2d) * R_inv3, axis=1)

    return Br_total, Bz_total


def _compute_mlx(
    coils: Sequence[Coil], grid: Grid, n_phi: int = 128
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute total field using the MLX backend, return NumPy arrays."""
    r = np.linspace(0.0, grid.r_max, grid.nr)
    z = np.linspace(grid.z_min, grid.z_max, grid.nz)
    Z, R = np.meshgrid(z, r)  # (nr, nz)

    flat_r = mx.array(R.ravel().astype(np.float32))
    flat_z = mx.array(Z.ravel().astype(np.float32))

    params = mx.array(
        [[c.z, c.r, c.I] for c in coils],
        dtype=mx.float32,
    )

    Br_flat, Bz_flat = compute_bfield_mlx_differentiable(params, flat_r, flat_z, n_phi=n_phi)
    mx.eval(Br_flat, Bz_flat)

    Br = np.array(Br_flat).reshape(grid.nr, grid.nz)
    Bz = np.array(Bz_flat).reshape(grid.nr, grid.nz)
    return Br, Bz, r, z


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_bfield(
    coils: Sequence[Coil],
    grid: Grid,
    *,
    backend: str = "auto",
    n_phi: int = 128,
) -> BField:
    """Compute the applied magnetic field from a set of circular coils.

    Parameters
    ----------
    coils : sequence of Coil
        Coil definitions in SI units.
    grid : Grid
        Computational domain.
    backend : ``"auto"`` | ``"mlx"`` | ``"numpy"``
        ``"auto"`` uses MLX if available, else NumPy.
    n_phi : int
        Azimuthal quadrature points (MLX backend only).

    Returns
    -------
    BField
        Result container with NumPy arrays.
    """
    coils = list(coils)

    if backend == "auto":
        backend = "mlx" if HAS_MLX else "numpy"

    if backend == "mlx":
        if not HAS_MLX:
            raise ImportError("MLX is not installed. Install with: pip install 'helicon[mlx]'")
        Br, Bz, r, z = _compute_mlx(coils, grid, n_phi=n_phi)
    elif backend == "numpy":
        Br, Bz, r, z = _compute_numpy(coils, grid)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'auto', 'mlx', or 'numpy'.")

    return BField(Br=Br, Bz=Bz, r=r, z=z, coils=coils, backend=backend)
