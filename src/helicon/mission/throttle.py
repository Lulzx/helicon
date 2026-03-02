"""Throttle curve generation for magnetic nozzle thrusters.

Produces thrust/Isp performance maps as functions of input power and
propellant mass flow rate. Implements the Tier 1 (analytical, fast) sweep
with optional Tier 2 (WarpX PIC) validation on top-N candidates.

Physics
-------
For a magnetic nozzle with thermal efficiency η and divergence efficiency η_div:
    v_e = sqrt(2 * η * P_in / ṁ)     [exhaust velocity]
    F = ṁ * v_e * η_div               [thrust]
    Isp = v_e * η_div / g0            [specific impulse]
    η_d from mirror ratio R_B (coil geometry)

The detachment efficiency η_d is obtained from the coil configuration via
Biot-Savart + the Breizman-Arefiev thrust efficiency formula.

Output is stored as a scipy RegularGridInterpolator over (P_in, ṁ) and
can be serialised to HDF5 or JSON for downstream mission planners.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import RegularGridInterpolator

if TYPE_CHECKING:
    from helicon.config.parser import SimConfig

_G0 = 9.80665  # standard gravity [m/s²]


@dataclass
class OperatingPoint:
    """Single (P_in, ṁ) operating point."""

    power_W: float
    mdot_kgs: float


@dataclass
class ThrottleResult:
    """Performance at one operating point."""

    power_W: float
    mdot_kgs: float
    thrust_N: float
    isp_s: float
    exhaust_velocity_ms: float
    eta_d: float  # detachment efficiency
    eta_thermal: float  # thermal coupling efficiency used
    eta_divergence: float  # divergence/plume efficiency


@dataclass
class ThrottleMap:
    """Thrust/Isp performance map over a (power, mdot) grid.

    Attributes
    ----------
    power_grid_W : ndarray, shape (n_power,)
        Input power values [W].
    mdot_grid_kgs : ndarray, shape (n_mdot,)
        Propellant mass flow rate values [kg/s].
    thrust_N : ndarray, shape (n_power, n_mdot)
        Thrust [N].
    isp_s : ndarray, shape (n_power, n_mdot)
        Specific impulse [s].
    eta_d : ndarray, shape (n_power, n_mdot)
        Detachment efficiency (geometry-dependent, constant over grid).
    mirror_ratio : float
        Mirror ratio R_B for the coil configuration.
    """

    power_grid_W: np.ndarray
    mdot_grid_kgs: np.ndarray
    thrust_N: np.ndarray
    isp_s: np.ndarray
    eta_d: np.ndarray
    mirror_ratio: float
    eta_thermal: float
    results: list[ThrottleResult] = field(default_factory=list)

    def thrust_interpolator(self) -> RegularGridInterpolator:
        """Interpolator: (power_W, mdot_kgs) → thrust_N."""
        return RegularGridInterpolator(
            (self.power_grid_W, self.mdot_grid_kgs),
            self.thrust_N,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def isp_interpolator(self) -> RegularGridInterpolator:
        """Interpolator: (power_W, mdot_kgs) → Isp_s."""
        return RegularGridInterpolator(
            (self.power_grid_W, self.mdot_grid_kgs),
            self.isp_s,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def thrust_at(self, power_W: float, mdot_kgs: float) -> float:
        """Interpolate thrust at a specific operating point."""
        return float(self.thrust_interpolator()([[power_W, mdot_kgs]])[0])

    def isp_at(self, power_W: float, mdot_kgs: float) -> float:
        """Interpolate Isp at a specific operating point."""
        return float(self.isp_interpolator()([[power_W, mdot_kgs]])[0])

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        return {
            "power_grid_W": self.power_grid_W.tolist(),
            "mdot_grid_kgs": self.mdot_grid_kgs.tolist(),
            "thrust_N": self.thrust_N.tolist(),
            "isp_s": self.isp_s.tolist(),
            "eta_d": self.eta_d.tolist(),
            "mirror_ratio": self.mirror_ratio,
            "eta_thermal": self.eta_thermal,
        }

    def save_json(self, path: str | Path) -> Path:
        """Save throttle map to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def save_hdf5(self, path: str | Path) -> Path:
        """Save throttle map to HDF5."""
        import h5py

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.create_dataset("power_grid_W", data=self.power_grid_W)
            f.create_dataset("mdot_grid_kgs", data=self.mdot_grid_kgs)
            f.create_dataset("thrust_N", data=self.thrust_N)
            f.create_dataset("isp_s", data=self.isp_s)
            f.create_dataset("eta_d", data=self.eta_d)
            f.attrs["mirror_ratio"] = self.mirror_ratio
            f.attrs["eta_thermal"] = self.eta_thermal
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> ThrottleMap:
        """Load throttle map from JSON."""
        data = json.loads(Path(path).read_text())
        return cls(
            power_grid_W=np.array(data["power_grid_W"]),
            mdot_grid_kgs=np.array(data["mdot_grid_kgs"]),
            thrust_N=np.array(data["thrust_N"]),
            isp_s=np.array(data["isp_s"]),
            eta_d=np.array(data["eta_d"]),
            mirror_ratio=float(data["mirror_ratio"]),
            eta_thermal=float(data["eta_thermal"]),
        )


def _compute_single_point(
    power_W: float,
    mdot_kgs: float,
    eta_thermal: float,
    eta_divergence: float,
    eta_d: float,
) -> ThrottleResult:
    """Compute performance at a single (power, mdot) operating point."""
    # Exhaust velocity from power balance
    if mdot_kgs <= 0 or power_W <= 0:
        return ThrottleResult(
            power_W=power_W,
            mdot_kgs=mdot_kgs,
            thrust_N=0.0,
            isp_s=0.0,
            exhaust_velocity_ms=0.0,
            eta_d=eta_d,
            eta_thermal=eta_thermal,
            eta_divergence=eta_divergence,
        )

    v_e = math.sqrt(2.0 * eta_thermal * power_W / mdot_kgs)
    thrust = mdot_kgs * v_e * eta_divergence
    isp = v_e * eta_divergence / _G0

    return ThrottleResult(
        power_W=power_W,
        mdot_kgs=mdot_kgs,
        thrust_N=thrust,
        isp_s=isp,
        exhaust_velocity_ms=v_e,
        eta_d=eta_d,
        eta_thermal=eta_thermal,
        eta_divergence=eta_divergence,
    )


def generate_throttle_map(
    config: SimConfig,
    *,
    power_range_W: tuple[float, float] = (1e3, 1e7),
    mdot_range_kgs: tuple[float, float] = (1e-6, 1e-3),
    n_power: int = 20,
    n_mdot: int = 20,
    eta_thermal: float = 0.65,
    n_pts_mirror: int = 100,
    backend: str = "auto",
) -> ThrottleMap:
    """Generate a thrust/Isp performance map over (P_in, ṁ).

    Uses Tier 1 analytical formulas for all grid points.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration (coil geometry determines mirror ratio).
    power_range_W : (float, float)
        Input power range [W].
    mdot_range_kgs : (float, float)
        Mass flow rate range [kg/s].
    n_power : int
        Number of power grid points.
    n_mdot : int
        Number of mdot grid points.
    eta_thermal : float
        Thermal coupling efficiency (0–1).  Default 0.65.
    n_pts_mirror : int
        Resolution for on-axis Biot-Savart mirror ratio computation.
    backend : str
        Biot-Savart backend (``"auto"``, ``"numpy"``, ``"mlx"``).

    Returns
    -------
    ThrottleMap
    """
    from helicon.fields.biot_savart import Coil
    from helicon.optimize.analytical import (
        divergence_half_angle,
        thrust_efficiency,
    )
    from helicon.optimize.analytical import (
        mirror_ratio as compute_mirror_ratio,
    )

    # Compute mirror ratio from coil geometry
    coils = [Coil(z=c.z, r=c.r, I=c.I) for c in config.nozzle.coils]
    domain = config.nozzle.domain
    R_B = compute_mirror_ratio(
        coils,
        z_min=domain.z_min,
        z_max=domain.z_max,
        n_pts=n_pts_mirror,
        backend=backend,
    )

    # Divergence efficiency: η_div = cos²(θ/2) where θ = half-angle
    eta_thrust = thrust_efficiency(R_B)  # 1 - 1/sqrt(R_B)
    half_angle = float(divergence_half_angle(R_B))
    eta_divergence = math.cos(math.radians(half_angle)) ** 2 if R_B > 1 else 1.0

    # Detachment efficiency ≈ thrust efficiency (Breizman-Arefiev model)
    eta_d = eta_thrust

    # Build grids (log-spaced for wide dynamic range)
    power_grid = np.geomspace(power_range_W[0], power_range_W[1], n_power)
    mdot_grid = np.geomspace(mdot_range_kgs[0], mdot_range_kgs[1], n_mdot)

    thrust_map = np.zeros((n_power, n_mdot))
    isp_map = np.zeros((n_power, n_mdot))
    eta_d_map = np.full((n_power, n_mdot), eta_d)
    results = []

    for i, p in enumerate(power_grid):
        for j, m in enumerate(mdot_grid):
            r = _compute_single_point(p, m, eta_thermal, eta_divergence, eta_d)
            thrust_map[i, j] = r.thrust_N
            isp_map[i, j] = r.isp_s
            results.append(r)

    return ThrottleMap(
        power_grid_W=power_grid,
        mdot_grid_kgs=mdot_grid,
        thrust_N=thrust_map,
        isp_s=isp_map,
        eta_d=eta_d_map,
        mirror_ratio=float(R_B),
        eta_thermal=eta_thermal,
        results=results,
    )
