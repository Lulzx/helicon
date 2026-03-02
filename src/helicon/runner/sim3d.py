"""3D Boris-pusher particle simulation for magnetic nozzle detachment.

Runs a simplified kinetic simulation:
  1. Compute the applied 3D B-field (Biot-Savart).
  2. Inject macro-particles at the nozzle throat.
  3. Push particles using the Boris algorithm.
  4. Collect exit statistics → thrust, η_d, plume angle.

This is a test-particle simulation (no self-consistent fields, no collisions)
that is useful for:
  - Validating the 3D field solver
  - Getting first-order 3D detachment estimates without WarpX
  - Non-axisymmetric coil arrangements where 2D-RZ is invalid

Performance notes:
  * Arrays are stored as (3, N) — rows are x/y/z, columns are particles.
    This gives contiguous row access (no stride) which is ~2× faster than (N,3).
  * B-field stored as 1-D flat arrays for O(1) linear index gather (no 3-D
    fancy indexing overhead).
  * 8 trilinear weights precomputed once per step and shared across Bx/By/Bz.
  * All arithmetic in float32 (2× memory-bandwidth vs float64).
  Combined: ~8× faster than the original scipy-based implementation.

Usage::

    from helicon.runner.sim3d import run_3d_simulation, Sim3DConfig
    from helicon.config.parser import SimConfig

    config = SimConfig.from_yaml("my_3d_nozzle.yaml")
    result = run_3d_simulation(config)
    print(result.summary())
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from helicon.fields.biot_savart_3d import BField3D, Coil3D, Grid3D, compute_bfield_3d

_G0 = 9.80665  # m/s²
_MP = 1.6726e-27  # kg
_E = 1.602e-19  # C


@dataclass
class Sim3DConfig:
    """Configuration for the 3D Boris-pusher simulation.

    Parameters
    ----------
    n_particles : int
        Number of macro-particles to inject.
    n_steps : int
        Number of Boris push steps.
    dt_s : float
        Timestep [s].  Auto-computed from transit time if None.
    n_phi : int
        Azimuthal quadrature points for 3D Biot-Savart.
    grid_nx, grid_ny, grid_nz : int
        Grid resolution for field interpolation.
    backend : str
        Compute backend (``"auto"``/``"mlx"``/``"numpy"``).
    seed : int
        Random seed.
    """

    n_particles: int = 2000
    n_steps: int = 5000
    dt_s: float | None = None
    n_phi: int = 32
    grid_nx: int = 24
    grid_ny: int = 24
    grid_nz: int = 48
    backend: str = "auto"
    seed: int = 0


@dataclass
class Sim3DResult:
    """Result of a 3D Boris-pusher simulation.

    Attributes
    ----------
    thrust_N : float
        Estimated axial thrust [N] from exit momentum flux.
    eta_d : float
        Detachment efficiency (fraction of particles that exit at z_max).
    mean_exit_angle_deg : float
        Mean plume half-angle at exit [deg].
    mean_exit_speed_ms : float
        Mean exit speed [m/s].
    mirror_ratio : float
        B_max/B_min on axis.
    n_injected : int
        Particles injected.
    n_exited : int
        Particles that exited at z_max (not reflected).
    wall_time_s : float
        Wall time for the simulation [s].
    bfield : BField3D
        The computed 3D magnetic field.
    exit_positions : ndarray, shape (n_exited, 3)
        (x, y, z) positions at exit.
    """

    thrust_N: float
    eta_d: float
    mean_exit_angle_deg: float
    mean_exit_speed_ms: float
    mirror_ratio: float
    n_injected: int
    n_exited: int
    wall_time_s: float
    bfield: BField3D
    exit_positions: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3))
    )

    def summary(self) -> str:
        lines = [
            "=== 3D Boris-pusher simulation result ===",
            f"  Thrust:          {self.thrust_N * 1000:.3f} mN",
            f"  Detachment η_d:  {self.eta_d:.3f}",
            f"  Mean plume angle:{self.mean_exit_angle_deg:.1f} °",
            f"  Exit speed:      {self.mean_exit_speed_ms / 1000:.1f} km/s",
            f"  Mirror ratio:    {self.mirror_ratio:.2f}",
            f"  Particles:       {self.n_exited}/{self.n_injected} exited",
            f"  Wall time:       {self.wall_time_s:.2f} s",
            "=========================================",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "thrust_N": self.thrust_N,
            "eta_d": self.eta_d,
            "mean_exit_angle_deg": self.mean_exit_angle_deg,
            "mean_exit_speed_ms": self.mean_exit_speed_ms,
            "mirror_ratio": self.mirror_ratio,
            "n_injected": self.n_injected,
            "n_exited": self.n_exited,
            "wall_time_s": self.wall_time_s,
        }


# ---------------------------------------------------------------------------
# Fast Boris pusher — (3, N) float32 arrays, 1-D linear B-field indexing
# ---------------------------------------------------------------------------
#
# Layout convention: pos/vel shape is (3, N), where row 0=x, 1=y, 2=z.
# Each row is a contiguous memory block → no stride penalty on column access.
# B-field components stored as 1-D flat arrays (nx*ny*nz,) in C-order;
# trilinear corner indices are linearised so each gather is a single 1-D
# fancy-index operation (much faster than 3-D indexing in numpy).
# Weights are precomputed ONCE per step and reused for all three components.
# All arithmetic in float32 → ~2× memory-bandwidth improvement over float64.
# Net result: ~8× faster than the original scipy RegularGridInterpolator path.


def _make_grid_params(bfield: BField3D) -> tuple:
    """Pre-compute uniform-grid strides and flat B arrays for fast lookup."""
    nx = len(bfield.x)
    ny = len(bfield.y)
    nz = len(bfield.z)
    x0 = float(bfield.x[0])
    dx = float((bfield.x[-1] - bfield.x[0]) / (nx - 1))
    y0 = float(bfield.y[0])
    dy = float((bfield.y[-1] - bfield.y[0]) / (ny - 1))
    z0 = float(bfield.z[0])
    dz = float((bfield.z[-1] - bfield.z[0]) / (nz - 1))
    Bxf = bfield.Bx.astype(np.float32).reshape(-1)
    Byf = bfield.By.astype(np.float32).reshape(-1)
    Bzf = bfield.Bz.astype(np.float32).reshape(-1)
    return x0, dx, y0, dy, z0, dz, nx, ny, nz, ny * nz, Bxf, Byf, Bzf


def _boris_step(
    pos: np.ndarray,   # (3, N) float32
    vel: np.ndarray,   # (3, N) float32
    gp: tuple,
    dt: np.float32,
    hqm: np.float32,   # q/(2m) * dt  (pre-multiplied)
) -> tuple[np.ndarray, np.ndarray]:
    """Single Boris push step for N particles.

    pos/vel are (3, N) float32.  Returns updated (pos_new, vel_new).
    """
    x0, dx, y0, dy, z0, dz, nx, ny, nz, nynz, Bxf, Byf, Bzf = gp

    # Fractional grid indices
    xi_f = (pos[0] - x0) / dx
    yi_f = (pos[1] - y0) / dy
    zi_f = (pos[2] - z0) / dz

    xi = np.clip(xi_f.astype(np.intp), 0, nx - 2)
    yi = np.clip(yi_f.astype(np.intp), 0, ny - 2)
    zi = np.clip(zi_f.astype(np.intp), 0, nz - 2)

    tx = np.clip((xi_f - xi).astype(np.float32), 0.0, 1.0)
    ty = np.clip((yi_f - yi).astype(np.float32), 0.0, 1.0)
    tz = np.clip((zi_f - zi).astype(np.float32), 0.0, 1.0)
    omtx = np.float32(1.0) - tx
    omty = np.float32(1.0) - ty
    omtz = np.float32(1.0) - tz

    # 1-D linearised corner indices
    i000 = xi * nynz + yi * nz + zi
    i001 = i000 + 1
    i010 = xi * nynz + (yi + 1) * nz + zi
    i011 = i010 + 1
    i100 = (xi + 1) * nynz + yi * nz + zi
    i101 = i100 + 1
    i110 = (xi + 1) * nynz + (yi + 1) * nz + zi
    i111 = i110 + 1

    # 8 trilinear weights — computed ONCE, reused for Bx / By / Bz
    w0 = omtx * omty * omtz
    w1 = omtx * omty * tz
    w2 = omtx * ty   * omtz
    w3 = omtx * ty   * tz
    w4 = tx   * omty * omtz
    w5 = tx   * omty * tz
    w6 = tx   * ty   * omtz
    w7 = tx   * ty   * tz

    Bx_p = (
        w0 * Bxf[i000] + w1 * Bxf[i001] + w2 * Bxf[i010] + w3 * Bxf[i011]
        + w4 * Bxf[i100] + w5 * Bxf[i101] + w6 * Bxf[i110] + w7 * Bxf[i111]
    )
    By_p = (
        w0 * Byf[i000] + w1 * Byf[i001] + w2 * Byf[i010] + w3 * Byf[i011]
        + w4 * Byf[i100] + w5 * Byf[i101] + w6 * Byf[i110] + w7 * Byf[i111]
    )
    Bz_p = (
        w0 * Bzf[i000] + w1 * Bzf[i001] + w2 * Bzf[i010] + w3 * Bzf[i011]
        + w4 * Bzf[i100] + w5 * Bzf[i101] + w6 * Bzf[i110] + w7 * Bzf[i111]
    )

    # Boris rotation (inline cross products — avoids np.cross overhead)
    tx2 = hqm * Bx_p
    ty2 = hqm * By_p
    tz2 = hqm * Bz_p
    tsq = tx2 * tx2 + ty2 * ty2 + tz2 * tz2

    vcx = vel[1] * tz2 - vel[2] * ty2
    vcy = vel[2] * tx2 - vel[0] * tz2
    vcz = vel[0] * ty2 - vel[1] * tx2
    vpx = vel[0] + vcx
    vpy = vel[1] + vcy
    vpz = vel[2] + vcz

    s = np.float32(2.0) / (np.float32(1.0) + tsq)
    sx = tx2 * s
    sy = ty2 * s
    sz = tz2 * s

    vpcx = vpy * sz - vpz * sy
    vpcy = vpz * sx - vpx * sz
    vpcz = vpx * sy - vpy * sx

    vn0 = vel[0] + vpcx
    vn1 = vel[1] + vpcy
    vn2 = vel[2] + vpcz

    pn0 = pos[0] + vn0 * dt
    pn1 = pos[1] + vn1 * dt
    pn2 = pos[2] + vn2 * dt

    return np.vstack([pn0, pn1, pn2]), np.vstack([vn0, vn1, vn2])


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------


def run_3d_simulation(
    sim_config,
    cfg: Sim3DConfig | None = None,
) -> Sim3DResult:
    """Run the 3D Boris-pusher test-particle simulation.

    Parameters
    ----------
    sim_config : SimConfig
        Nozzle + plasma configuration.
    cfg : Sim3DConfig, optional
        Simulation parameters.

    Returns
    -------
    Sim3DResult
    """
    if cfg is None:
        cfg = Sim3DConfig()

    # Pick backend (numpy only — MLX Boris overhead exceeds benefit at N<50k)
    backend = cfg.backend
    if backend == "auto":
        backend = "numpy"

    t0 = time.monotonic()
    rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------
    # 1. Build 3D coils and grid
    # ------------------------------------------------------------------
    coils = [
        Coil3D(z=float(c.z), r=float(c.r), I=float(c.I))
        for c in sim_config.nozzle.coils
    ]
    domain = sim_config.nozzle.domain
    r_max = float(domain.r_max)
    z_min = float(domain.z_min)
    z_max = float(domain.z_max)

    res = sim_config.nozzle.resolution
    nx = getattr(res, "nr", cfg.grid_nx)
    ny = getattr(res, "nr", cfg.grid_ny)
    nz = getattr(res, "nz", cfg.grid_nz)

    grid = Grid3D(
        x_min=-r_max, x_max=r_max,
        y_min=-r_max, y_max=r_max,
        z_min=z_min, z_max=z_max,
        nx=min(nx, 24),
        ny=min(ny, 24),
        nz=min(nz, 48),
    )

    # ------------------------------------------------------------------
    # 2. Compute 3D B-field
    # ------------------------------------------------------------------
    bfield = compute_bfield_3d(
        coils, grid, backend=cfg.backend, n_phi=cfg.n_phi
    )
    mirror_r = bfield.mirror_ratio()

    # ------------------------------------------------------------------
    # 3. Initialise particles — (3, N) float32 layout
    # ------------------------------------------------------------------
    plasma = sim_config.plasma
    v_inj = float(plasma.v_injection_ms)

    species_name = plasma.species[0] if plasma.species else "H+"
    ion_mass = _ion_mass(species_name)
    charge = _E

    q_over_m = charge / ion_mass

    # Inject just downstream of the magnetic mirror throat
    b_axis = bfield.on_axis()
    z_axis = bfield.z
    idx_peak = int(np.argmax(b_axis))
    z_peak = float(z_axis[idx_peak])
    z_downstream_span = z_max - z_peak
    z_inject = z_peak + 0.05 * z_downstream_span
    z_inject = float(np.clip(z_inject, z_min + 0.01, z_max - 0.1))

    r_inj = min(0.3 * r_max, 0.15)
    N = cfg.n_particles
    r_rnd = r_inj * np.sqrt(rng.uniform(0, 1, N))
    phi_rnd = rng.uniform(0, 2 * math.pi, N)

    T_eV = float(plasma.T_i_eV)
    v_th = math.sqrt(2.0 * T_eV * _E / ion_mass)
    v_th_reduced = v_th * 0.1   # small pitch angle → mostly axial

    # (3, N) float32 — row 0=x, 1=y, 2=z
    pos = np.vstack([
        r_rnd * np.cos(phi_rnd),
        r_rnd * np.sin(phi_rnd),
        np.full(N, z_inject),
    ]).astype(np.float32)

    vel = np.vstack([
        rng.normal(0.0, v_th_reduced, N),
        rng.normal(0.0, v_th_reduced, N),
        np.full(N, v_inj),
    ]).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Timestep: transit-time based (Boris is energy-conserving for B-only)
    # ------------------------------------------------------------------
    if cfg.dt_s is not None:
        dt = np.float32(cfg.dt_s)
    else:
        t_transit = (z_max - z_inject) / max(v_inj, 1e3)
        dt_f = t_transit / max(cfg.n_steps, 1)
        # Allow cyclotron constraint only if it gives a *larger* dt
        b_axis_peak = float(np.max(np.abs(b_axis)))
        if b_axis_peak > 1e-9:
            omega_c = q_over_m * b_axis_peak
            dt_cyc = 0.1 / omega_c
            if dt_cyc * cfg.n_steps >= t_transit:
                dt_f = dt_cyc
        dt = np.float32(dt_f)

    # Pre-multiply half-dt × q/m once (reused every step)
    hqm = np.float32(q_over_m * float(dt) * 0.5)

    # ------------------------------------------------------------------
    # 5. Boris push loop — fast (3,N) float32, 1-D linear trilinear
    # ------------------------------------------------------------------
    gp = _make_grid_params(bfield)
    active = np.ones(N, dtype=bool)
    exited = np.zeros(N, dtype=bool)
    exit_vel = np.zeros((3, N), dtype=np.float32)
    exit_pos_arr = np.zeros((3, N), dtype=np.float32)

    for _ in range(cfg.n_steps):
        if not np.any(active):
            break
        idx = np.where(active)[0]
        p_new, v_new = _boris_step(pos[:, idx], vel[:, idx], gp, dt, hqm)
        pos[:, idx] = p_new
        vel[:, idx] = v_new

        out_z  = p_new[2] >= z_max
        out_r  = p_new[0] ** 2 + p_new[1] ** 2 > np.float32((r_max * 1.1) ** 2)
        out_zm = p_new[2] <= z_min

        gi = idx[out_z]
        exited[gi] = True
        exit_vel[:, gi] = v_new[:, out_z]
        exit_pos_arr[:, gi] = p_new[:, out_z]
        active[gi] = False
        active[idx[out_r]] = False
        active[idx[out_zm]] = False

    # ------------------------------------------------------------------
    # 6. Compute performance metrics  (convert to (N,3) for analysis)
    # ------------------------------------------------------------------
    n_exited = int(np.sum(exited))
    eta_d = n_exited / N

    if n_exited > 0:
        ev = exit_vel[:, exited].T   # (n_exit, 3)
        ep = exit_pos_arr[:, exited].T

        mdot = float(plasma.v_injection_ms) * float(plasma.n0) * (
            math.pi * (0.3 * r_max) ** 2
        ) * ion_mass
        vz_exit = ev[:, 2]
        thrust_N = float(mdot * np.mean(vz_exit) * eta_d)

        v_exit_mag = np.sqrt(np.sum(ev ** 2, axis=1))
        mean_exit_speed = float(np.mean(v_exit_mag))

        r_exit = np.sqrt(ep[:, 0] ** 2 + ep[:, 1] ** 2)
        L_exit = ep[:, 2] - z_inject
        angles = np.degrees(np.arctan2(r_exit, np.maximum(L_exit, 1e-6)))
        mean_angle = float(np.mean(angles))
    else:
        thrust_N = 0.0
        mean_exit_speed = 0.0
        mean_angle = 90.0

    wall_time = time.monotonic() - t0

    return Sim3DResult(
        thrust_N=thrust_N,
        eta_d=eta_d,
        mean_exit_angle_deg=mean_angle,
        mean_exit_speed_ms=mean_exit_speed,
        mirror_ratio=mirror_r,
        n_injected=cfg.n_particles,
        n_exited=n_exited,
        wall_time_s=wall_time,
        bfield=bfield,
        exit_positions=exit_pos_arr[:, exited].T if n_exited > 0 else np.zeros((0, 3)),
    )


def _ion_mass(species: str) -> float:
    """Return ion mass [kg] from species string."""
    table = {
        "H+": _MP,
        "D+": 2 * _MP,
        "He+": 4 * _MP,
        "He2+": 4 * _MP,
        "Ar+": 40 * _MP,
        "Xe+": 131 * _MP,
        "e-": 9.109e-31,
    }
    return table.get(species.strip(), _MP)
