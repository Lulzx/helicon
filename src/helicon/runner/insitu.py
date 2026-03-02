"""In-situ postprocessing callbacks for WarpX simulations.

Provides factory functions that create WarpX-compatible callbacks and
associated accumulators for on-the-fly thrust monitoring during a run.

WarpX integration requires ``pywarpx`` at runtime; the factory functions
themselves can be imported without WarpX installed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class InsituTimeSeries:
    """Accumulator for in-situ thrust time series.

    Attributes
    ----------
    times : list[float]
        Simulation times [s] at which thrust was sampled.
    thrust_N : list[float]
        Instantaneous thrust [N] at each sample time.
    mass_flow_kgs : list[float]
        Instantaneous mass flow rate [kg/s] at each sample time.
    """

    times: list[float] = field(default_factory=list)
    thrust_N: list[float] = field(default_factory=list)
    mass_flow_kgs: list[float] = field(default_factory=list)

    def append(self, t: float, thrust: float, mdot: float) -> None:
        """Record one sample."""
        self.times.append(t)
        self.thrust_N.append(thrust)
        self.mass_flow_kgs.append(mdot)

    def mean_thrust(self) -> float:
        """Time-averaged thrust over all samples [N]."""
        if not self.thrust_N:
            return 0.0
        import numpy as np

        return float(np.mean(self.thrust_N))

    def __len__(self) -> int:
        return len(self.times)


def make_thrust_callback(
    accumulator: InsituTimeSeries,
    *,
    species_name: str = "D_plus",
    species_mass: float = 3.3435837724e-27,
    exit_plane_z: float | None = None,
    backend: str = "auto",
) -> Callable[[], None]:
    """Create a WarpX-compatible callback that accumulates thrust in-situ.

    The returned callable is designed to be registered with
    ``pywarpx.callbacks.installafterdiagnostics()`` or equivalent.
    It reads the current particle data directly from the WarpX Python
    interface (without writing/reading HDF5) and appends to *accumulator*.

    Parameters
    ----------
    accumulator : InsituTimeSeries
        Shared state object that the callback writes to.
    species_name : str
        WarpX species to track.
    species_mass : float
        Species mass [kg].
    exit_plane_z : float, optional
        Exit plane z-coordinate [m].  If None, uses the domain maximum.
    backend : ``"auto"`` | ``"mlx"`` | ``"numpy"``
        Compute backend for momentum-flux reduction.

    Returns
    -------
    Callable[[], None]
        A no-argument callback suitable for WarpX registration.

    Notes
    -----
    Requires ``pywarpx`` at runtime.  The factory itself can be called
    without WarpX installed; import errors only occur when the callback
    is invoked inside a WarpX simulation.
    """
    # Store backend string; resolve to actual backend lazily inside the callback
    # so factory creation does not raise ImportError when MLX is absent.
    _backend = backend

    def _callback() -> None:
        from helicon._mlx_utils import resolve_backend

        use_mlx = resolve_backend(_backend) == "mlx"
        # Import pywarpx lazily — only available inside WarpX simulations
        try:
            from pywarpx import libwarpx
        except ImportError as exc:
            raise ImportError(
                "pywarpx is required for in-situ callbacks. "
                "Install WarpX with Python bindings."
            ) from exc

        import numpy as np

        t = float(libwarpx.warpx_gett_new(0))

        # Read particle data from WarpX memory (avoids HDF5 roundtrip)
        try:
            z_pos = libwarpx.get_particle_z(species_name)
            pz = libwarpx.get_particle_uz(species_name) * float(species_mass)
            w = libwarpx.get_particle_weight(species_name)
        except Exception:
            # Species may not exist at this step
            return

        if len(z_pos) == 0:
            return

        # Exit plane selection
        z_exit = float(exit_plane_z) if exit_plane_z is not None else float(z_pos.max()) * 0.9
        dz_tol = (float(z_pos.max()) - float(z_pos.min())) / 100.0
        near_exit = np.abs(z_pos - z_exit) < dz_tol

        if not np.any(near_exit):
            return

        vz = pz[near_exit] / species_mass
        wt = w[near_exit]

        if use_mlx:
            from helicon.postprocess.thrust import _thrust_reduce_mlx

            thrust, mdot = _thrust_reduce_mlx(wt, species_mass, vz)
        else:
            thrust = float(np.sum(wt * species_mass * vz**2))
            mdot = float(np.sum(wt * species_mass * np.abs(vz)))

        accumulator.append(t, thrust, mdot)

    return _callback
