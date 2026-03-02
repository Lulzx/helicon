"""Thrust and mass flow rate computation from simulation output.

Computes thrust via momentum flux integration across the downstream
exit plane: F = Σ_s ∫(n_s m_s v_z² + P_zz) dA
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

EV_TO_J = 1.602176634e-19


@dataclass
class ThrustResult:
    """Computed thrust metrics."""

    thrust_N: float
    mass_flow_rate_kgs: float
    isp_s: float
    exhaust_velocity_ms: float
    exit_plane_z: float
    n_particles_counted: int


def compute_thrust(
    output_dir: str | Path,
    *,
    exit_plane_z: float | None = None,
    species_masses: dict[str, float] | None = None,
) -> ThrustResult:
    """Compute thrust from WarpX openPMD output.

    Reads the final particle snapshot, selects particles near the exit
    plane, and integrates momentum flux.

    Parameters
    ----------
    output_dir : path
        WarpX output directory containing openPMD files.
    exit_plane_z : float, optional
        Axial position of the exit plane [m]. If None, uses the domain
        boundary (inferred from data).
    species_masses : dict, optional
        Species name -> mass [kg] mapping. Defaults to standard values.
    """
    output_dir = Path(output_dir)

    # Default species masses
    if species_masses is None:
        species_masses = {
            "D_plus": 3.3435837724e-27,
            "H_plus": 1.67262192595e-27,
            "He3_plus": 5.0082353373e-27,
            "e_minus": 9.1093837139e-31,
        }

    # Try to read openPMD data via h5py
    import h5py

    # Find the latest openPMD iteration
    diag_dir = output_dir / "diags" / "diag1"
    if not diag_dir.exists():
        # Try alternative layouts
        diag_dir = output_dir
        openpmd_files = sorted(diag_dir.glob("openpmd_*.h5"))
        if not openpmd_files:
            openpmd_files = sorted(diag_dir.glob("**/*.h5"))
        if not openpmd_files:
            msg = f"No openPMD files found in {output_dir}"
            raise FileNotFoundError(msg)
        latest = openpmd_files[-1]
    else:
        openpmd_files = sorted(diag_dir.glob("*.h5"))
        if not openpmd_files:
            msg = f"No openPMD files found in {diag_dir}"
            raise FileNotFoundError(msg)
        latest = openpmd_files[-1]

    total_momentum_z = 0.0
    total_mass_flow = 0.0
    total_particles = 0

    with h5py.File(latest, "r") as f:
        # Navigate openPMD hierarchy
        if "data" in f:
            iterations = sorted(f["data"].keys(), key=int)
            base = f["data"][iterations[-1]]
        else:
            base = f

        if "particles" not in base:
            msg = "No particle data found in openPMD file"
            raise ValueError(msg)

        for species_name in base["particles"]:
            sp = base["particles"][species_name]
            mass = species_masses.get(species_name, 1.67e-27)

            # Read positions and momenta
            if "position" in sp and "z" in sp["position"]:
                z = sp["position"]["z"][:]
            else:
                continue

            if "momentum" in sp and "z" in sp["momentum"]:
                pz = sp["momentum"]["z"][:]
            else:
                continue

            # Weight (number of real particles per macro-particle)
            w = sp["weighting"][:] if "weighting" in sp else np.ones_like(z)

            # Exit plane selection
            if exit_plane_z is None:
                exit_plane_z = z.max() * 0.9

            dz_tolerance = (z.max() - z.min()) / 100.0
            near_exit = np.abs(z - exit_plane_z) < dz_tolerance

            if np.sum(near_exit) == 0:
                continue

            vz = pz[near_exit] / mass
            wt = w[near_exit]

            # Momentum flux: sum(w * m * vz^2) is force (thrust)
            total_momentum_z += np.sum(wt * mass * vz**2)
            total_mass_flow += np.sum(wt * mass * np.abs(vz))
            total_particles += int(np.sum(near_exit))

    g0 = 9.80665  # standard gravity
    thrust = total_momentum_z
    mdot = total_mass_flow  # rough: assumes steady-state flux
    v_ex = thrust / mdot if mdot > 0 else 0.0
    isp = v_ex / g0 if g0 > 0 else 0.0

    return ThrustResult(
        thrust_N=float(thrust),
        mass_flow_rate_kgs=float(mdot),
        isp_s=float(isp),
        exhaust_velocity_ms=float(v_ex),
        exit_plane_z=float(exit_plane_z) if exit_plane_z is not None else 0.0,
        n_particles_counted=total_particles,
    )
