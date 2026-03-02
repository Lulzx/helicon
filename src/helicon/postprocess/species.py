"""Multi-ion species detachment tracking (spec v1.2).

Computes per-species η_d for D⁺, He²⁺, H⁺, and α-particles.
Heavier ions detach at different axial locations, so per-species
detachment efficiency is physically meaningful and required for
D/He3 direct-fusion-drive performance predictions.

MLX-accelerated species-resolved moment computation on Metal GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from helicon._mlx_utils import HAS_MLX, resolve_backend, to_mx, to_np
from helicon.postprocess.detachment import _navigate_openpmd

_QE = 1.6021766340e-19  # C
_MP = 1.6726219236951e-27  # kg

# Standard propulsion species masses and charge numbers
SPECIES_CATALOG: dict[str, dict[str, Any]] = {
    "D_plus": {"mass": 2.0 * _MP, "Z": 1, "label": "D⁺"},
    "He_plus2": {"mass": 4.0 * _MP, "Z": 2, "label": "He²⁺"},
    "H_plus": {"mass": _MP, "Z": 1, "label": "H⁺"},
    "alpha": {"mass": 4.0 * _MP, "Z": 2, "label": "α (He²⁺)"},
    "Xe_plus": {"mass": 131.293 * _MP, "Z": 1, "label": "Xe⁺"},
    "Ar_plus": {"mass": 39.948 * _MP, "Z": 1, "label": "Ar⁺"},
}


@dataclass
class SpeciesDetachmentResult:
    """Detachment efficiency for a single ion species.

    Attributes
    ----------
    species : str
        Species name (e.g. 'D_plus')
    label : str
        Human-readable label (e.g. 'D⁺')
    mass : float
        Ion mass [kg]
    momentum_based : float
        η_d (momentum definition)
    particle_based : float
        η_d (particle-count definition)
    energy_based : float
        η_d (energy definition)
    n_injected : int
        Number of macroparticles injected
    n_exited_downstream : int
        Number that exited through downstream boundary
    n_lost_radial : int
        Number lost radially
    n_reflected : int
        Number reflected upstream
    """

    species: str
    label: str
    mass: float
    momentum_based: float
    particle_based: float
    energy_based: float
    n_injected: int
    n_exited_downstream: int
    n_lost_radial: int
    n_reflected: int

    def summary(self) -> str:
        lines = [
            f"Species: {self.label}",
            f"  η_d (momentum): {self.momentum_based:.4f}",
            f"  η_d (particle): {self.particle_based:.4f}",
            f"  η_d (energy):   {self.energy_based:.4f}",
            f"  Injected: {self.n_injected}",
            f"  Exited downstream: {self.n_exited_downstream}",
            f"  Lost radially: {self.n_lost_radial}",
            f"  Reflected: {self.n_reflected}",
        ]
        return "\n".join(lines)


@dataclass
class MultiSpeciesDetachmentResult:
    """Detachment results for all tracked species.

    Attributes
    ----------
    results : dict[str, SpeciesDetachmentResult]
        Per-species results keyed by species name
    """

    results: dict[str, SpeciesDetachmentResult]

    def __getitem__(self, species: str) -> SpeciesDetachmentResult:
        return self.results[species]

    def species_names(self) -> list[str]:
        return list(self.results.keys())

    def summary(self) -> str:
        lines = ["Multi-Species Detachment Efficiency:"]
        lines.append(f"{'Species':<12} {'η_d(mom)':>10} {'η_d(part)':>10} {'η_d(energy)':>12}")
        lines.append("-" * 46)
        for res in self.results.values():
            lines.append(
                f"{res.label:<12} {res.momentum_based:>10.4f} "
                f"{res.particle_based:>10.4f} {res.energy_based:>12.4f}"
            )
        return "\n".join(lines)

    def dominant_species(self) -> str:
        """Return species name with highest momentum-based η_d."""
        if not self.results:
            return ""
        return max(self.results, key=lambda s: self.results[s].momentum_based)


def compute_species_moments(
    positions: np.ndarray,
    momenta: np.ndarray,
    weights: np.ndarray,
    species_mass: float,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    backend: str = "auto",
) -> dict[str, np.ndarray]:
    """Compute density, bulk velocity, and pressure tensor on grid for one species.

    MLX-accelerated particle-to-grid deposition.

    Parameters
    ----------
    positions : ndarray of shape (N, 2)
        [r, z] particle positions [m]
    momenta : ndarray of shape (N,) or (N, 3)
        Axial momentum pz (or [pr, pphi, pz]) [kg m/s]
    weights : ndarray of shape (N,)
        Macroparticle weights
    species_mass : float
        Ion mass [kg]
    r_grid : 1-D ndarray
        Radial cell centers [m]
    z_grid : 1-D ndarray
        Axial cell centers [m]
    backend : str
        'auto', 'mlx', or 'numpy'

    Returns
    -------
    dict with keys:
        'density'   : ndarray (nr, nz) [m⁻³]
        'vz'        : ndarray (nr, nz) bulk axial velocity [m/s]
        'pz_density': ndarray (nr, nz) momentum density [kg m/s / m³]
        'T_par_eV'  : ndarray (nr, nz) parallel temperature [eV]
    """
    use_mlx = resolve_backend(backend) == "mlx"

    r = positions[:, 0] if positions.ndim > 1 else np.zeros(len(weights))
    z = positions[:, 1] if positions.ndim > 1 else positions

    pz = momenta[:, 2] if momenta.ndim > 1 else momenta

    nr = len(r_grid)
    nz = len(z_grid)
    dr = r_grid[1] - r_grid[0] if nr > 1 else 1.0
    dz = z_grid[1] - z_grid[0] if nz > 1 else 1.0

    if use_mlx and HAS_MLX:
        import mlx.core as mx

        r_mx = to_mx(r)
        z_mx = to_mx(z)
        pz_mx = to_mx(pz)
        w_mx = to_mx(weights)
        r0 = float(r_grid[0])
        z0 = float(z_grid[0])
        dr_f = float(dr)
        dz_f = float(dz)
        nr_i = int(nr)
        nz_i = int(nz)

        # Clip indices
        r_idx_mx = mx.clip(
            ((r_mx - r0) / dr_f).astype(mx.int32), 0, nr_i - 1
        )
        z_idx_mx = mx.clip(
            ((z_mx - z0) / dz_f).astype(mx.int32), 0, nz_i - 1
        )
        mx.eval(r_idx_mx, z_idx_mx)
        r_idx = to_np(r_idx_mx).astype(int)
        z_idx = to_np(z_idx_mx).astype(int)
        w_np = to_np(w_mx)
        pz_np = to_np(pz_mx)
    else:
        r_idx = np.clip(
            np.floor((r - r_grid[0]) / dr).astype(int), 0, nr - 1
        )
        z_idx = np.clip(
            np.floor((z - z_grid[0]) / dz).astype(int), 0, nz - 1
        )
        w_np = weights
        pz_np = pz

    density = np.zeros((nr, nz))
    pz_density = np.zeros((nr, nz))
    pz2_density = np.zeros((nr, nz))

    np.add.at(density, (r_idx, z_idx), w_np)
    np.add.at(pz_density, (r_idx, z_idx), w_np * pz_np)
    np.add.at(pz2_density, (r_idx, z_idx), w_np * pz_np**2)

    # Cell volume: dV = 2π r dr dz
    r_2d = r_grid[:, None] * np.ones((1, nz))
    cell_vol = 2.0 * np.pi * r_2d * dr * dz
    cell_vol = np.where(cell_vol > 0, cell_vol, 1.0)

    density /= cell_vol
    pz_density /= cell_vol
    pz2_density /= cell_vol

    n_safe = np.where(density > 0, density, 1.0)
    vz = pz_density / (n_safe * species_mass)

    # Parallel temperature: T_par = (m * <(vz - <vz>)^2>) / k
    # = (m / n) * [<pz^2>/m^2 - vz^2]  [in energy units]
    vz2_mean = pz2_density / (n_safe * species_mass**2)
    var_vz = np.maximum(vz2_mean - vz**2, 0.0)
    T_par_eV = species_mass * var_vz / _QE

    return {
        "density": density,
        "vz": vz,
        "pz_density": pz_density,
        "T_par_eV": T_par_eV,
    }


def compute_multi_species_detachment(
    output_dir: str | Path,
    *,
    species_list: list[str] | None = None,
    z_inject: float | None = None,
    z_exit: float | None = None,
    r_max: float | None = None,
    backend: str = "auto",
) -> MultiSpeciesDetachmentResult:
    """Compute per-species detachment efficiency from openPMD output.

    Parameters
    ----------
    output_dir : path
        WarpX output directory.
    species_list : list[str], optional
        Species to analyze. Defaults to all species in SPECIES_CATALOG
        that are present in the output.
    z_inject : float, optional
        Injection plane z-coordinate [m].
    z_exit : float, optional
        Exit plane z-coordinate [m].
    r_max : float, optional
        Radial domain boundary [m].
    backend : str
        Compute backend.

    Returns
    -------
    MultiSpeciesDetachmentResult
    """
    import h5py

    output_dir = Path(output_dir)
    h5_files = sorted(output_dir.glob("**/*.h5"))
    if len(h5_files) < 2:
        msg = "Need at least 2 snapshots for multi-species detachment analysis"
        raise FileNotFoundError(msg)

    if species_list is None:
        species_list = list(SPECIES_CATALOG.keys())

    # Detect which species are actually in the output
    available_species = []
    with h5py.File(h5_files[0], "r") as f:
        base = _navigate_openpmd(f)
        if "particles" in base:
            for sp in species_list:
                if sp in base["particles"]:
                    available_species.append(sp)

    if not available_species:
        # Return empty results with zero efficiency
        results = {}
        for sp in species_list:
            cat = SPECIES_CATALOG.get(sp, {"mass": _MP, "Z": 1, "label": sp})
            results[sp] = SpeciesDetachmentResult(
                species=sp,
                label=cat["label"],
                mass=cat["mass"],
                momentum_based=0.0,
                particle_based=0.0,
                energy_based=0.0,
                n_injected=0,
                n_exited_downstream=0,
                n_lost_radial=0,
                n_reflected=0,
            )
        return MultiSpeciesDetachmentResult(results=results)

    results = {}
    for sp_name in available_species:
        cat = SPECIES_CATALOG.get(sp_name, {"mass": _MP, "Z": 1, "label": sp_name})
        mass = cat["mass"]
        result = _compute_single_species(
            h5_files=h5_files,
            species_name=sp_name,
            species_mass=mass,
            species_label=cat["label"],
            z_inject=z_inject,
            z_exit=z_exit,
            r_max=r_max,
            backend=backend,
        )
        results[sp_name] = result

    return MultiSpeciesDetachmentResult(results=results)


def _compute_single_species(
    h5_files: list[Path],
    species_name: str,
    species_mass: float,
    species_label: str,
    z_inject: float | None,
    z_exit: float | None,
    r_max: float | None,
    backend: str,
) -> SpeciesDetachmentResult:
    """Compute detachment for one species from openPMD snapshots."""

    use_mlx = resolve_backend(backend) == "mlx"

    # Read initial snapshot for injected totals
    pz_inject, ke_inject, n_inject = _read_species_initial(
        h5_files[0], species_name, species_mass
    )

    # Read final snapshot
    z_pos, r_pos, pz_final, w_final = _read_species_final(
        h5_files[-1], species_name, species_mass
    )

    if n_inject == 0 or len(z_pos) == 0:
        return SpeciesDetachmentResult(
            species=species_name,
            label=species_label,
            mass=species_mass,
            momentum_based=0.0,
            particle_based=0.0,
            energy_based=0.0,
            n_injected=n_inject,
            n_exited_downstream=0,
            n_lost_radial=0,
            n_reflected=0,
        )

    if z_exit is None:
        z_exit = float(np.max(z_pos) * 0.95)
    if z_inject is None:
        z_inject = float(np.min(z_pos) * 1.05)
    if r_max is None:
        r_max = float(np.max(r_pos) * 0.95)

    if use_mlx and HAS_MLX:
        import mlx.core as mx

        z_mx = to_mx(z_pos)
        r_mx = to_mx(r_pos)
        pz_mx = to_mx(pz_final)
        w_mx = to_mx(w_final)

        downstream = z_mx >= float(z_exit)
        radial = r_mx >= float(r_max)
        reflected = z_mx <= float(z_inject)

        down_f = downstream.astype(mx.float32)
        rad_f = (radial & ~downstream).astype(mx.float32)
        ref_f = (reflected & ~downstream & ~radial).astype(mx.float32)

        n_downstream = float(to_np(mx.sum(w_mx * down_f)))
        n_radial = float(to_np(mx.sum(w_mx * rad_f)))
        n_reflected = float(to_np(mx.sum(w_mx * ref_f)))
        pz_exit = float(to_np(mx.sum(w_mx * pz_mx * down_f)))
        vz_mx = pz_mx / float(species_mass)
        ke_exit = float(
            to_np(mx.sum(w_mx * down_f * 0.5 * float(species_mass) * vz_mx * vz_mx))
        )
    else:
        downstream = z_pos >= z_exit
        radial_loss = r_pos >= r_max
        reflected = z_pos <= z_inject

        n_downstream = float(np.sum(w_final[downstream]))
        n_radial = float(np.sum(w_final[radial_loss & ~downstream]))
        n_reflected = float(np.sum(w_final[reflected & ~downstream & ~radial_loss]))
        pz_exit = float(np.sum(w_final[downstream] * pz_final[downstream]))
        vz_down = pz_final[downstream] / species_mass
        ke_exit = float(0.5 * species_mass * np.sum(w_final[downstream] * vz_down**2))

    eta_mom = float(np.clip(pz_exit / pz_inject, 0.0, 1.0)) if pz_inject > 0 else 0.0
    eta_part = float(np.clip(n_downstream / n_inject, 0.0, 1.0)) if n_inject > 0 else 0.0
    eta_energy = float(np.clip(ke_exit / ke_inject, 0.0, 1.0)) if ke_inject > 0 else 0.0

    return SpeciesDetachmentResult(
        species=species_name,
        label=species_label,
        mass=species_mass,
        momentum_based=eta_mom,
        particle_based=eta_part,
        energy_based=eta_energy,
        n_injected=int(n_inject),
        n_exited_downstream=int(n_downstream),
        n_lost_radial=int(n_radial),
        n_reflected=int(n_reflected),
    )


def _read_species_initial(
    path: Path, species_name: str, species_mass: float
) -> tuple[float, float, int]:
    """Read initial (injected) totals from an openPMD snapshot."""
    import h5py

    with h5py.File(path, "r") as f:
        base = _navigate_openpmd(f)
        if "particles" not in base or species_name not in base["particles"]:
            return 0.0, 0.0, 0
        sp = base["particles"][species_name]
        pz = sp["momentum"]["z"][:]
        w = sp["weighting"][:] if "weighting" in sp else np.ones_like(pz)
        vz = pz / species_mass
        return (
            float(np.sum(w * pz)),
            float(0.5 * species_mass * np.sum(w * vz**2)),
            int(np.sum(w)),
        )


def _read_species_final(
    path: Path, species_name: str, species_mass: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read final particle data from an openPMD snapshot."""
    import h5py

    with h5py.File(path, "r") as f:
        base = _navigate_openpmd(f)
        if "particles" not in base or species_name not in base["particles"]:
            return np.array([]), np.array([]), np.array([]), np.array([])
        sp = base["particles"][species_name]
        z = sp["position"]["z"][:]
        r = sp["position"]["r"][:] if "r" in sp["position"] else np.zeros_like(z)
        pz = sp["momentum"]["z"][:]
        w = sp["weighting"][:] if "weighting" in sp else np.ones_like(z)
    return z, r, pz, w
