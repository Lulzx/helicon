"""3D Cartesian Biot-Savart field solver for non-axisymmetric nozzles.

Computes the full B = (Bx, By, Bz) vector field on a Cartesian (x, y, z)
grid from an arbitrary set of circular current loops.  Supports coils with
arbitrary axial positions and radii; all loops are assumed to lie in planes
perpendicular to the z-axis.

Two backends:
  * NumPy — vectorised numerical integration over N_phi quadrature points
  * MLX   — same algorithm on Metal GPU; ~10–50× faster for large grids

Usage::

    from helicon.fields.biot_savart_3d import Coil3D, Grid3D, compute_bfield_3d

    coil = Coil3D(z=0.0, r=0.1, I=10000.0)
    grid = Grid3D(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2,
                  z_min=-0.3, z_max=1.5, nx=32, ny=32, nz=64)
    bfield = compute_bfield_3d([coil], grid)
    # bfield.Bx, bfield.By, bfield.Bz — shape (nx, ny, nz)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from helicon._mlx_utils import HAS_MLX, to_np

if HAS_MLX:
    import mlx.core as mx

_MU0 = 4.0 * math.pi * 1e-7  # T·m/A


@dataclass(frozen=True)
class Coil3D:
    """A single circular current loop centred on the z-axis.

    Parameters
    ----------
    z : float
        Axial position [m].
    r : float
        Coil radius [m].
    I : float
        Current (ampere-turns) [A].
    """

    z: float
    r: float
    I: float


@dataclass(frozen=True)
class Grid3D:
    """3D Cartesian computation grid.

    Parameters
    ----------
    x_min, x_max : float
        Transverse x extent [m].
    y_min, y_max : float
        Transverse y extent [m].
    z_min, z_max : float
        Axial extent [m].
    nx, ny, nz : int
        Number of grid points along each axis.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    nx: int
    ny: int
    nz: int


@dataclass
class BField3D:
    """3D magnetic field result.

    Attributes
    ----------
    Bx, By, Bz : ndarray
        Field components [T], shape ``(nx, ny, nz)``.
    Bmag : ndarray
        |B| [T], shape ``(nx, ny, nz)``.
    x, y, z : ndarray
        1-D coordinate arrays [m].
    coils : list[Coil3D]
    backend : str
    """

    Bx: np.ndarray
    By: np.ndarray
    Bz: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    coils: list[Coil3D]
    backend: str

    @property
    def Bmag(self) -> np.ndarray:
        return np.sqrt(self.Bx**2 + self.By**2 + self.Bz**2)

    def on_axis(self) -> np.ndarray:
        """On-axis |B| at x=y=0: shape (nz,)."""
        ix = len(self.x) // 2
        iy = len(self.y) // 2
        return self.Bmag[ix, iy, :]

    def mirror_ratio(self) -> float:
        """B_max / B_min(downstream) on axis."""
        bax = self.on_axis()
        idx_peak = int(np.argmax(bax))
        b_downstream = bax[idx_peak:]
        b_min = float(np.min(b_downstream)) if len(b_downstream) > 0 else float(bax[-1])
        return float(bax[idx_peak]) / max(b_min, 1e-12)


# ---------------------------------------------------------------------------
# NumPy backend
# ---------------------------------------------------------------------------


def _compute_bfield_numpy(
    coils: list[Coil3D],
    grid: Grid3D,
    n_phi: int,
) -> BField3D:
    """Vectorised NumPy implementation."""
    x_arr = np.linspace(grid.x_min, grid.x_max, grid.nx, dtype=np.float64)
    y_arr = np.linspace(grid.y_min, grid.y_max, grid.ny, dtype=np.float64)
    z_arr = np.linspace(grid.z_min, grid.z_max, grid.nz, dtype=np.float64)

    # Build 3D meshgrid: shapes (nx, ny, nz)
    X, Y, Z = np.meshgrid(x_arr, y_arr, z_arr, indexing="ij")

    Bx = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.float64)
    By = np.zeros_like(Bx)
    Bz = np.zeros_like(Bx)

    phi = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    dphi = 2.0 * math.pi / n_phi

    for coil in coils:
        R, z_c, I = coil.r, coil.z, coil.I
        prefactor = _MU0 * I * R / (4.0 * math.pi)

        cos_phi = np.cos(phi)  # (n_phi,)
        sin_phi = np.sin(phi)

        # Expand to (nx, ny, nz, n_phi) via broadcasting
        dx = X[..., np.newaxis] - R * cos_phi  # (nx,ny,nz,n_phi)
        dy = Y[..., np.newaxis] - R * sin_phi
        dz = (Z - z_c)[..., np.newaxis] * np.ones(n_phi)

        r2 = dx**2 + dy**2 + dz**2
        r3 = r2**1.5
        r3 = np.where(r3 < 1e-30, 1e-30, r3)

        # dl × r̂ / r² = (cos φ dz, sin φ dz, R - x cosφ - y sinφ) / r³
        # Sum over quadrature
        Bx += prefactor * dphi * np.sum(cos_phi * dz / r3, axis=-1)
        By += prefactor * dphi * np.sum(sin_phi * dz / r3, axis=-1)
        Bz += prefactor * dphi * np.sum(
            (R - X[..., np.newaxis] * cos_phi - Y[..., np.newaxis] * sin_phi) / r3,
            axis=-1,
        )

    return BField3D(
        Bx=Bx,
        By=By,
        Bz=Bz,
        x=x_arr,
        y=y_arr,
        z=z_arr,
        coils=coils,
        backend="numpy",
    )


# ---------------------------------------------------------------------------
# MLX backend
# ---------------------------------------------------------------------------


def _compute_bfield_mlx(
    coils: list[Coil3D],
    grid: Grid3D,
    n_phi: int,
) -> BField3D:
    """MLX Metal GPU implementation."""
    x_arr = np.linspace(grid.x_min, grid.x_max, grid.nx, dtype=np.float32)
    y_arr = np.linspace(grid.y_min, grid.y_max, grid.ny, dtype=np.float32)
    z_arr = np.linspace(grid.z_min, grid.z_max, grid.nz, dtype=np.float32)

    X_mx = mx.array(
        np.stack(np.meshgrid(x_arr, y_arr, z_arr, indexing="ij"), axis=-1)
    )  # (nx, ny, nz, 3)

    X_f = X_mx[..., 0]  # (nx, ny, nz)
    Y_f = X_mx[..., 1]
    Z_f = X_mx[..., 2]

    phi_np = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False).astype(np.float32)
    dphi = float(2.0 * math.pi / n_phi)
    cos_phi = mx.array(np.cos(phi_np))  # (n_phi,)
    sin_phi = mx.array(np.sin(phi_np))

    Bx = mx.zeros((grid.nx, grid.ny, grid.nz))
    By = mx.zeros_like(Bx)
    Bz = mx.zeros_like(Bx)

    for coil in coils:
        R = float(coil.r)
        z_c = float(coil.z)
        I = float(coil.I)
        prefactor = float(_MU0 * I * R / (4.0 * math.pi) * dphi)

        # (nx, ny, nz, n_phi) via broadcasting
        dx = X_f[..., None] - R * cos_phi
        dy = Y_f[..., None] - R * sin_phi
        dz = (Z_f - z_c)[..., None] * mx.ones((n_phi,))

        r2 = dx * dx + dy * dy + dz * dz
        r3 = mx.where(r2 > 1e-30, r2 * mx.sqrt(r2), mx.array(1e-30))

        Bx = Bx + prefactor * mx.sum(cos_phi * dz / r3, axis=-1)
        By = By + prefactor * mx.sum(sin_phi * dz / r3, axis=-1)
        Bz = Bz + prefactor * mx.sum(
            (R - X_f[..., None] * cos_phi - Y_f[..., None] * sin_phi) / r3,
            axis=-1,
        )

    mx.eval(Bx, By, Bz)
    return BField3D(
        Bx=to_np(Bx).astype(np.float64),
        By=to_np(By).astype(np.float64),
        Bz=to_np(Bz).astype(np.float64),
        x=x_arr.astype(np.float64),
        y=y_arr.astype(np.float64),
        z=z_arr.astype(np.float64),
        coils=coils,
        backend="mlx",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_bfield_3d(
    coils: list[Coil3D],
    grid: Grid3D,
    *,
    backend: str = "auto",
    n_phi: int = 64,
) -> BField3D:
    """Compute the 3D Cartesian magnetic field from circular coils.

    Parameters
    ----------
    coils : list[Coil3D]
        Coil definitions.
    grid : Grid3D
        3D Cartesian computation domain.
    backend : ``"auto"`` | ``"mlx"`` | ``"numpy"``
        Computation backend.  ``"auto"`` uses MLX if available.
    n_phi : int
        Number of azimuthal quadrature points (higher = more accurate).
        64 gives < 0.01 % error for typical nozzle geometries.

    Returns
    -------
    BField3D
    """
    if backend == "auto":
        backend = "mlx" if HAS_MLX else "numpy"

    if backend == "mlx":
        if not HAS_MLX:
            raise ImportError("MLX not available — use backend='numpy'")
        return _compute_bfield_mlx(coils, grid, n_phi)
    return _compute_bfield_numpy(coils, grid, n_phi)
