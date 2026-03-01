"""Detachment efficiency computation — three definitions.

MagNozzleX computes three definitions of η_d to resolve ambiguity
in the magnetic nozzle literature:

1. **Momentum-based:** Net axial momentum at exit / injected axial momentum
2. **Particle-based:** Fraction of injected particles exiting downstream
   (vs. radial loss or reflection)
3. **Energy-based:** Directed kinetic energy at exit / total injected energy
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DetachmentResult:
    """Three definitions of detachment efficiency."""

    momentum_based: float
    particle_based: float
    energy_based: float
    n_injected: int
    n_exited_downstream: int
    n_lost_radial: int
    n_reflected: int

    def summary(self) -> str:
        lines = [
            "Detachment Efficiency:",
            f"  η_d (momentum): {self.momentum_based:.4f}",
            f"  η_d (particle): {self.particle_based:.4f}",
            f"  η_d (energy):   {self.energy_based:.4f}",
            f"  Injected: {self.n_injected}",
            f"  Exited downstream: {self.n_exited_downstream}",
            f"  Lost radially: {self.n_lost_radial}",
            f"  Reflected: {self.n_reflected}",
        ]
        return "\n".join(lines)


def compute_detachment(
    output_dir: str | Path,
    *,
    species_name: str = "D_plus",
    species_mass: float = 3.3435837724e-27,
    z_inject: float | None = None,
    z_exit: float | None = None,
    r_max: float | None = None,
) -> DetachmentResult:
    """Compute detachment efficiency from openPMD particle data.

    Reads initial and final particle snapshots, classifies particles
    by exit location, and computes all three η_d definitions.

    Parameters
    ----------
    output_dir : path
        WarpX output directory.
    species_name : str
        Particle species to analyze.
    species_mass : float
        Species mass [kg].
    z_inject : float, optional
        Injection plane z-coordinate [m].
    z_exit : float, optional
        Exit plane z-coordinate [m].
    r_max : float, optional
        Radial domain boundary [m].
    """
    output_dir = Path(output_dir)
    h5_files = sorted(output_dir.glob("**/*.h5"))
    if len(h5_files) < 2:
        msg = "Need at least 2 snapshots (initial + final) for detachment analysis"
        raise FileNotFoundError(msg)

    # Read initial snapshot
    pz_inject_total, ke_inject_total, n_inject = _read_snapshot_stats(
        h5_files[0], species_name, species_mass
    )

    # Read final snapshot
    _, _, positions_final, momenta_final, weights_final = _read_final_snapshot(
        h5_files[-1], species_name, species_mass
    )

    if n_inject == 0:
        return DetachmentResult(
            momentum_based=0.0,
            particle_based=0.0,
            energy_based=0.0,
            n_injected=0,
            n_exited_downstream=0,
            n_lost_radial=0,
            n_reflected=0,
        )

    # Classify particles by exit location
    z_pos = positions_final[:, 1] if positions_final.ndim > 1 else positions_final
    r_pos = positions_final[:, 0] if positions_final.ndim > 1 else np.zeros_like(z_pos)

    if z_exit is None:
        z_exit = z_pos.max() * 0.95
    if z_inject is None:
        z_inject = z_pos.min() * 1.05
    if r_max is None:
        r_max = r_pos.max() * 0.95

    downstream = z_pos >= z_exit
    radial_loss = r_pos >= r_max
    reflected = z_pos <= z_inject

    n_downstream = int(np.sum(weights_final[downstream]))
    n_radial = int(np.sum(weights_final[radial_loss & ~downstream]))
    n_reflected_count = int(np.sum(weights_final[reflected & ~downstream & ~radial_loss]))

    # 1. Momentum-based: p_z at exit / p_z injected
    pz_exit = np.sum(weights_final[downstream] * momenta_final[downstream])
    eta_momentum = float(pz_exit / pz_inject_total) if pz_inject_total > 0 else 0.0

    # 2. Particle-based: N_downstream / N_injected
    eta_particle = float(n_downstream / n_inject) if n_inject > 0 else 0.0

    # 3. Energy-based: KE_directed_exit / KE_injected
    vz_down = momenta_final[downstream] / species_mass
    ke_directed = 0.5 * species_mass * np.sum(weights_final[downstream] * vz_down**2)
    eta_energy = float(ke_directed / ke_inject_total) if ke_inject_total > 0 else 0.0

    return DetachmentResult(
        momentum_based=np.clip(eta_momentum, 0.0, 1.0),
        particle_based=np.clip(eta_particle, 0.0, 1.0),
        energy_based=np.clip(eta_energy, 0.0, 1.0),
        n_injected=int(n_inject),
        n_exited_downstream=n_downstream,
        n_lost_radial=n_radial,
        n_reflected=n_reflected_count,
    )


def _read_snapshot_stats(
    path: Path,
    species_name: str,
    species_mass: float,
) -> tuple[float, float, int]:
    """Read total injected momentum and energy from a snapshot."""
    import h5py

    with h5py.File(path, "r") as f:
        base = _navigate_openpmd(f)
        if "particles" not in base or species_name not in base["particles"]:
            return 0.0, 0.0, 0

        sp = base["particles"][species_name]
        pz = sp["momentum"]["z"][:]
        w = sp["weighting"][:] if "weighting" in sp else np.ones_like(pz)

        vz = pz / species_mass
        total_pz = float(np.sum(w * pz))
        total_ke = float(0.5 * species_mass * np.sum(w * vz**2))
        n = int(np.sum(w))

    return total_pz, total_ke, n


def _read_final_snapshot(
    path: Path,
    species_name: str,
    species_mass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read particle data from a snapshot for classification."""
    import h5py

    with h5py.File(path, "r") as f:
        base = _navigate_openpmd(f)
        sp = base["particles"][species_name]

        z = sp["position"]["z"][:]
        r = sp["position"]["r"][:] if "r" in sp["position"] else np.zeros_like(z)
        pz = sp["momentum"]["z"][:]
        w = sp["weighting"][:] if "weighting" in sp else np.ones_like(z)

        vz = pz / species_mass
        ke = 0.5 * species_mass * vz**2
        positions = np.column_stack([r, z])

    return pz, ke, positions, pz, w


def _navigate_openpmd(f: Any) -> Any:
    """Navigate to the latest iteration in an openPMD file."""
    if "data" in f:
        iterations = sorted(f["data"].keys(), key=int)
        return f["data"][iterations[-1]]
    return f
