"""Monte Carlo Collision (MCC) neutral particle dynamics.

Implements self-consistent neutral dynamics for the v1.2 spec:
- Monte Carlo neutral particle push (ballistic, no Lorentz force)
- Null-collision method for efficient MCC sampling
- Processes: charge exchange, ionization, recombination
- MLX-accelerated collision probability evaluation on Metal GPU

The neutral particle push is intentionally decoupled from WarpX —
neutrals do not need the electromagnetic solver, only the collision
physics. This makes them well-suited for MLX acceleration.

References
----------
- Vahedi & Surendra (1995) — "A Monte Carlo collision model for the particle-in-cell method"
- Birdsall (1991) — "Particle-in-cell charged-particle simulations"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from helicon._mlx_utils import resolve_backend, to_mx, to_np
from helicon.neutrals.cross_sections import (
    SPECIES_MASS,
    cx_rate_m3s,
    ionization_rate_m3s,
    recombination_rate_m3s,
)

_QE = 1.6021766340e-19  # C


@dataclass
class NeutralParticles:
    """Neutral particle ensemble state.

    Attributes
    ----------
    positions : ndarray of shape (N, 3)
        [r, phi, z] positions [m]  (phi unused in 2D-RZ; set to 0)
    velocities : ndarray of shape (N, 3)
        [vr, vphi, vz] velocities [m/s]
    weights : ndarray of shape (N,)
        Macroparticle weights (physical particles per macroparticle)
    alive : ndarray of shape (N,) bool
        Mask for particles still in the domain
    """

    positions: np.ndarray
    velocities: np.ndarray
    weights: np.ndarray
    alive: np.ndarray

    @classmethod
    def create(
        cls,
        n_particles: int,
        species: str,
        n_density_m3: float,
        T_eV: float,
        domain_r: tuple[float, float] = (0.0, 0.5),
        domain_z: tuple[float, float] = (-0.5, 2.0),
        seed: int = 42,
    ) -> NeutralParticles:
        """Initialize uniform neutral particle distribution in the domain.

        Parameters
        ----------
        n_particles : int
            Number of macroparticles
        species : str
            Neutral species ('H', 'D', 'He', 'Xe')
        n_density_m3 : float
            Physical neutral density [m⁻³]
        T_eV : float
            Neutral temperature [eV]
        domain_r : (r_min, r_max)
        domain_z : (z_min, z_max)
        seed : int
            RNG seed
        """
        rng = np.random.default_rng(seed)
        mass = SPECIES_MASS.get(species, SPECIES_MASS["H"])

        r_min, r_max = domain_r
        z_min, z_max = domain_z
        domain_vol = np.pi * (r_max**2 - r_min**2) * (z_max - z_min)
        weight = n_density_m3 * domain_vol / n_particles

        # Uniform in volume (cylindrical: r drawn proportional to r for uniform area)
        r2 = rng.uniform(r_min**2, r_max**2, n_particles)
        r = np.sqrt(r2)
        z = rng.uniform(z_min, z_max, n_particles)
        phi = np.zeros(n_particles)  # 2D-RZ: azimuthal angle not tracked

        # Maxwellian velocity distribution
        v_th = np.sqrt(_QE * T_eV / mass)
        vr = rng.normal(0.0, v_th, n_particles)
        vphi = np.zeros(n_particles)
        vz = rng.normal(0.0, v_th, n_particles)

        positions = np.column_stack([r, phi, z])
        velocities = np.column_stack([vr, vphi, vz])
        weights = np.full(n_particles, weight)
        alive = np.ones(n_particles, dtype=bool)

        return cls(positions=positions, velocities=velocities, weights=weights, alive=alive)

    @property
    def n_alive(self) -> int:
        return int(np.sum(self.alive))

    @property
    def r(self) -> np.ndarray:
        return self.positions[:, 0]

    @property
    def z(self) -> np.ndarray:
        return self.positions[:, 2]


@dataclass
class MCCResult:
    """Result of one MCC timestep.

    Attributes
    ----------
    n_cx : int
        Number of charge-exchange events
    n_ionized : int
        Number of ionization events
    n_recombined : int
        Number of recombination events
    """

    n_cx: int = 0
    n_ionized: int = 0
    n_recombined: int = 0


@dataclass
class MCCCollider:
    """Monte Carlo Collision operator for neutral dynamics.

    Uses the null-collision method (Vahedi & Surendra 1995) for efficient
    sampling: the maximum collision frequency ν_max is pre-computed, and
    null collisions are used to maintain a constant total rate, allowing
    vectorized sampling.

    Parameters
    ----------
    species : str
        Neutral species
    dt : float
        Timestep [s]
    backend : str
        'auto', 'mlx', or 'numpy'
    """

    species: str
    dt: float
    backend: str = "auto"
    _null_freq: float = field(default=0.0, init=False, repr=False)

    def compute_null_frequency(
        self,
        n_ion_max_m3: float,
        T_ion_eV: float,
        T_e_eV: float,
    ) -> float:
        """Compute ν_max for null-collision method.

        ν_max = n_max * (σ_cx + σ_ion + σ_rec)_max * v_max

        Parameters
        ----------
        n_ion_max_m3 : float
            Maximum expected ion density in domain [m⁻³]
        T_ion_eV : float
            Representative ion temperature for CX rate
        T_e_eV : float
            Representative electron temperature for ionization rate

        Returns
        -------
        float
            Maximum collision frequency [s⁻¹] for null-collision method
        """
        rate_cx = float(cx_rate_m3s(self.species, T_ion_eV))
        rate_ion = float(ionization_rate_m3s(self.species, T_e_eV))
        rate_rec = float(recombination_rate_m3s(self.species, T_e_eV))
        total_rate = n_ion_max_m3 * (rate_cx + rate_ion + rate_rec)
        self._null_freq = total_rate
        return total_rate

    def step(
        self,
        neutrals: NeutralParticles,
        n_ion_m3: np.ndarray,
        T_ion_eV: np.ndarray,
        T_e_eV: np.ndarray,
        domain_r: tuple[float, float],
        domain_z: tuple[float, float],
        seed: int = 0,
    ) -> MCCResult:
        """Advance neutral particles by one timestep including MCC.

        1. Push all neutral particles ballistically (no E/B forces).
        2. Remove particles leaving the domain.
        3. Apply null-collision MCC for CX / ionization / recombination.

        Parameters
        ----------
        neutrals : NeutralParticles
            Current neutral state (modified in place)
        n_ion_m3 : ndarray of shape (N,)
            Ion density sampled at each neutral particle position
        T_ion_eV : ndarray
            Ion temperature at each neutral particle position [eV]
        T_e_eV : ndarray
            Electron temperature at each neutral particle position [eV]
        domain_r : (r_min, r_max)
        domain_z : (z_min, z_max)
        seed : int
            RNG seed for this step

        Returns
        -------
        MCCResult
        """
        backend = resolve_backend(self.backend)
        if backend == "mlx":
            return self._step_mlx(
                neutrals, n_ion_m3, T_ion_eV, T_e_eV, domain_r, domain_z, seed
            )
        return self._step_numpy(
            neutrals, n_ion_m3, T_ion_eV, T_e_eV, domain_r, domain_z, seed
        )

    def _step_numpy(
        self,
        neutrals: NeutralParticles,
        n_ion_m3: np.ndarray,
        T_ion_eV: np.ndarray,
        T_e_eV: np.ndarray,
        domain_r: tuple[float, float],
        domain_z: tuple[float, float],
        seed: int,
    ) -> MCCResult:
        """NumPy implementation of MCC step."""
        rng = np.random.default_rng(seed)
        alive = neutrals.alive

        # 1. Ballistic push
        neutrals.positions[alive] += neutrals.velocities[alive] * self.dt

        # 2. Domain boundary removal
        r_min, r_max = domain_r
        z_min, z_max = domain_z
        r = neutrals.r
        z = neutrals.z
        out = (r < r_min) | (r > r_max) | (z < z_min) | (z > z_max)
        neutrals.alive[out] = False
        alive = neutrals.alive

        if not np.any(alive):
            return MCCResult()

        # 3. MCC: null-collision method
        idx_alive = np.where(alive)[0]
        n_alive = len(idx_alive)

        # Compute collision probabilities for alive particles
        n_i = (
            n_ion_m3[idx_alive] if len(n_ion_m3) > 1 else np.full(n_alive, float(n_ion_m3[0]))
        )
        T_i = (
            T_ion_eV[idx_alive] if len(T_ion_eV) > 1 else np.full(n_alive, float(T_ion_eV[0]))
        )
        T_e = T_e_eV[idx_alive] if len(T_e_eV) > 1 else np.full(n_alive, float(T_e_eV[0]))

        rate_cx = cx_rate_m3s(self.species, T_i)
        rate_ion = ionization_rate_m3s(self.species, T_e)
        rate_rec = recombination_rate_m3s(self.species, T_e)

        # Total collision probability P = 1 - exp(-ν_total * dt)
        nu_cx = n_i * rate_cx
        nu_ion = n_i * rate_ion
        nu_rec = n_i * rate_rec
        nu_total = nu_cx + nu_ion + nu_rec

        P_total = 1.0 - np.exp(-nu_total * self.dt)
        rand = rng.uniform(0.0, 1.0, n_alive)
        collide = rand < P_total

        # Among colliding particles, determine which process
        n_cx = 0
        n_ion = 0
        n_rec = 0

        if np.any(collide):
            col_idx = idx_alive[collide]
            nu_t = nu_total[collide]
            nu_t_safe = np.where(nu_t > 0, nu_t, 1.0)
            rand2 = rng.uniform(0.0, 1.0, int(np.sum(collide)))

            cx_frac = nu_cx[collide] / nu_t_safe
            ion_frac = (nu_cx[collide] + nu_ion[collide]) / nu_t_safe

            is_cx = rand2 < cx_frac
            is_ion = (rand2 >= cx_frac) & (rand2 < ion_frac)
            is_rec = rand2 >= ion_frac

            # CX: neutral becomes ion (remove from neutral population)
            neutrals.alive[col_idx[is_cx]] = False
            n_cx = int(np.sum(is_cx))

            # Ionization: neutral is ionized (remove from neutral population)
            neutrals.alive[col_idx[is_ion]] = False
            n_ion = int(np.sum(is_ion))

            # Recombination: ion recombines to neutral (we don't add new particles here
            # as that is handled by the ion pusher; just count the event)
            n_rec = int(np.sum(is_rec))

        return MCCResult(n_cx=n_cx, n_ionized=n_ion, n_recombined=n_rec)

    def _step_mlx(
        self,
        neutrals: NeutralParticles,
        n_ion_m3: np.ndarray,
        T_ion_eV: np.ndarray,
        T_e_eV: np.ndarray,
        domain_r: tuple[float, float],
        domain_z: tuple[float, float],
        seed: int,
    ) -> MCCResult:
        """MLX-accelerated implementation of MCC step.

        Runs collision probability computation on Metal GPU via MLX.
        Particle removal is done in NumPy (indexing is CPU-side).
        """
        import mlx.core as mx

        alive = neutrals.alive

        # 1. Ballistic push (MLX)
        pos_mx = to_mx(neutrals.positions[alive])
        vel_mx = to_mx(neutrals.velocities[alive])
        pos_mx = pos_mx + vel_mx * float(self.dt)
        mx.eval(pos_mx)
        neutrals.positions[alive] = to_np(pos_mx)

        # 2. Domain removal
        r_min, r_max = domain_r
        z_min, z_max = domain_z
        r = neutrals.r
        z = neutrals.z
        out = (r < r_min) | (r > r_max) | (z < z_min) | (z > z_max)
        neutrals.alive[out] = False
        alive = neutrals.alive

        if not np.any(alive):
            return MCCResult()

        idx_alive = np.where(alive)[0]
        n_alive = len(idx_alive)

        # 3. Compute collision rates on GPU
        n_i = (
            n_ion_m3[idx_alive] if len(n_ion_m3) > 1 else np.full(n_alive, float(n_ion_m3[0]))
        )
        T_i = (
            T_ion_eV[idx_alive] if len(T_ion_eV) > 1 else np.full(n_alive, float(T_ion_eV[0]))
        )
        T_e = T_e_eV[idx_alive] if len(T_e_eV) > 1 else np.full(n_alive, float(T_e_eV[0]))

        n_mx = to_mx(n_i)
        rate_cx = to_mx(cx_rate_m3s(self.species, T_i))
        rate_ion = to_mx(ionization_rate_m3s(self.species, T_e))
        rate_rec = to_mx(recombination_rate_m3s(self.species, T_e))

        nu_cx_mx = n_mx * rate_cx
        nu_ion_mx = n_mx * rate_ion
        nu_rec_mx = n_mx * rate_rec
        nu_total_mx = nu_cx_mx + nu_ion_mx + nu_rec_mx

        dt_f = float(self.dt)
        P_total_mx = mx.ones_like(nu_total_mx) - mx.exp(-nu_total_mx * dt_f)

        # Evaluate on GPU then bring back for stochastic sampling
        mx.eval(P_total_mx, nu_cx_mx, nu_ion_mx, nu_total_mx)
        P_total = to_np(P_total_mx)
        nu_cx_np = to_np(nu_cx_mx)
        nu_ion_np = to_np(nu_ion_mx)
        nu_total_np = to_np(nu_total_mx)

        rng = np.random.default_rng(seed)
        rand = rng.uniform(0.0, 1.0, n_alive)
        collide = rand < P_total

        n_cx = 0
        n_ion = 0
        n_rec = 0

        if np.any(collide):
            col_idx = idx_alive[collide]
            nu_t = nu_total_np[collide]
            nu_t_safe = np.where(nu_t > 0, nu_t, 1.0)
            rand2 = rng.uniform(0.0, 1.0, int(np.sum(collide)))

            cx_frac = nu_cx_np[collide] / nu_t_safe
            ion_frac = (nu_cx_np[collide] + nu_ion_np[collide]) / nu_t_safe

            is_cx = rand2 < cx_frac
            is_ion = (rand2 >= cx_frac) & (rand2 < ion_frac)
            is_rec = rand2 >= ion_frac

            neutrals.alive[col_idx[is_cx]] = False
            neutrals.alive[col_idx[is_ion]] = False
            n_cx = int(np.sum(is_cx))
            n_ion = int(np.sum(is_ion))
            n_rec = int(np.sum(is_rec))

        return MCCResult(n_cx=n_cx, n_ionized=n_ion, n_recombined=n_rec)


class NeutralDynamics:
    """High-level interface for self-consistent neutral particle dynamics.

    Manages the neutral ensemble and MCC collision operator over a simulation.

    Parameters
    ----------
    species : str
        Neutral species ('H', 'D', 'He', 'Xe')
    dt : float
        Timestep [s]
    n_particles : int
        Number of macroparticles to initialize
    n_density_m3 : float
        Initial neutral density [m⁻³]
    T_eV : float
        Initial neutral temperature [eV]
    domain_r : (r_min, r_max)
    domain_z : (z_min, z_max)
    backend : str
        'auto', 'mlx', or 'numpy'
    seed : int
        RNG seed
    """

    def __init__(
        self,
        species: str,
        dt: float,
        n_particles: int = 10_000,
        n_density_m3: float = 1e18,
        T_eV: float = 0.025,
        domain_r: tuple[float, float] = (0.0, 0.5),
        domain_z: tuple[float, float] = (-0.5, 2.0),
        backend: str = "auto",
        seed: int = 42,
    ) -> None:
        self.species = species
        self.dt = dt
        self.backend = backend
        self._step_count = 0

        self.particles = NeutralParticles.create(
            n_particles=n_particles,
            species=species,
            n_density_m3=n_density_m3,
            T_eV=T_eV,
            domain_r=domain_r,
            domain_z=domain_z,
            seed=seed,
        )
        self.collider = MCCCollider(species=species, dt=dt, backend=backend)
        self._domain_r = domain_r
        self._domain_z = domain_z

    def step(
        self,
        n_ion_m3: float | np.ndarray,
        T_ion_eV: float | np.ndarray,
        T_e_eV: float | np.ndarray,
    ) -> MCCResult:
        """Advance neutral dynamics by one timestep.

        Parameters
        ----------
        n_ion_m3 : float or ndarray
            Ion density at each neutral particle location (or scalar for uniform)
        T_ion_eV : float or ndarray
            Ion temperature [eV]
        T_e_eV : float or ndarray
            Electron temperature [eV]
        """
        n_alive = self.particles.n_alive

        n_arr = np.atleast_1d(np.asarray(n_ion_m3, dtype=float))
        T_i_arr = np.atleast_1d(np.asarray(T_ion_eV, dtype=float))
        T_e_arr = np.atleast_1d(np.asarray(T_e_eV, dtype=float))

        # Broadcast scalar to per-particle if needed
        if len(n_arr) == 1:
            n_arr = np.full(n_alive, float(n_arr[0]))
        if len(T_i_arr) == 1:
            T_i_arr = np.full(n_alive, float(T_i_arr[0]))
        if len(T_e_arr) == 1:
            T_e_arr = np.full(n_alive, float(T_e_arr[0]))

        result = self.collider.step(
            self.particles,
            n_arr,
            T_i_arr,
            T_e_arr,
            self._domain_r,
            self._domain_z,
            seed=self._step_count,
        )
        self._step_count += 1
        return result

    @property
    def n_alive(self) -> int:
        return self.particles.n_alive

    def neutral_density_on_grid(
        self,
        r_grid: np.ndarray,
        z_grid: np.ndarray,
    ) -> np.ndarray:
        """Bin neutral macroparticles onto a (nr, nz) grid.

        Parameters
        ----------
        r_grid : 1-D array of radial cell centers [m]
        z_grid : 1-D array of axial cell centers [m]

        Returns
        -------
        ndarray of shape (nr, nz)
            Neutral number density [m⁻³] on grid
        """
        alive = self.particles.alive
        r = self.particles.r[alive]
        z = self.particles.z[alive]
        w = self.particles.weights[alive]

        nr = len(r_grid)
        nz = len(z_grid)
        dr = r_grid[1] - r_grid[0] if nr > 1 else 1.0
        dz = z_grid[1] - z_grid[0] if nz > 1 else 1.0

        r_idx = np.clip(
            np.floor((r - r_grid[0]) / dr).astype(int), 0, nr - 1
        )
        z_idx = np.clip(
            np.floor((z - z_grid[0]) / dz).astype(int), 0, nz - 1
        )

        density = np.zeros((nr, nz))
        np.add.at(density, (r_idx, z_idx), w)

        cell_vol = 2.0 * np.pi * r_grid[:, None] * dr * dz
        cell_vol = np.where(cell_vol > 0, cell_vol, 1.0)
        density /= cell_vol

        return density
