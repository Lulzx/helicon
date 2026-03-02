"""Generate a simulation video of the Sunbird magnetic nozzle.

Loads the computed B-field, traces multi-species particle orbits using
a Boris pusher in the 2D-RZ geometry, and renders an MP4.

Usage:
    uv run python scripts/make_video.py
"""

from __future__ import annotations

import math
import pathlib

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BFIELD_H5 = pathlib.Path("results/sunbird/applied_bfield.h5")
OUT_VIDEO = pathlib.Path("results/sunbird/simulation.mp4")

N_ION = 18       # deuterium-ion particles
N_ELEC = 18      # electron particles
N_FIELD_LINES = 12
N_STEPS = 800    # total Boris-pusher steps per particle
FPS = 30
DPI = 140

# Physical constants
E = 1.602176634e-19    # elementary charge [C]
M_D = 3.3435837724e-27 # deuterium mass [kg]
M_E = 9.1093837015e-31 # electron mass [kg]

# Initial conditions
T_I_EV = 5000.0        # ion temperature [eV]
T_E_EV = 2000.0        # electron temperature [eV]
V_INJ = 200_000.0      # injection drift speed [m/s]

# ---------------------------------------------------------------------------
# Load B-field
# ---------------------------------------------------------------------------
with h5py.File(BFIELD_H5, "r") as f:
    Bz_grid = f["Bz"][:]   # shape (nr, nz)
    Br_grid = f["Br"][:]
    r_ax = f["r"][:]
    z_ax = f["z"][:]

nr, nz = Bz_grid.shape
dr = r_ax[1] - r_ax[0]
dz = z_ax[1] - z_ax[0]

# Interpolators for Bz(r,z) and Br(r,z)
interp_Bz = RegularGridInterpolator(
    (r_ax, z_ax), Bz_grid, method="linear", bounds_error=False, fill_value=0.0
)
interp_Br = RegularGridInterpolator(
    (r_ax, z_ax), Br_grid, method="linear", bounds_error=False, fill_value=0.0
)

B_mag = np.sqrt(Bz_grid**2 + Br_grid**2)  # (nr, nz)


def get_B(r: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate (Br, Bz) at positions r, z (arrays)."""
    pts = np.column_stack([np.clip(r, r_ax[0], r_ax[-1]),
                           np.clip(z, z_ax[0], z_ax[-1])])
    bz = interp_Bz(pts)
    br = interp_Br(pts)
    return br, bz


# ---------------------------------------------------------------------------
# Boris pusher in 2D-RZ (track r, z; ignore azimuthal drift for vis)
# ---------------------------------------------------------------------------
def boris_push_rz(
    r0: np.ndarray, z0: np.ndarray,
    vr0: np.ndarray, vz0: np.ndarray,
    charge: float, mass: float,
    dt: float, n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Boris pusher projected onto the r-z plane.

    Tracks N particles simultaneously. Returns (r, z) arrays of shape
    (N, n_steps+1).
    """
    N = len(r0)
    r = np.empty((N, n_steps + 1))
    z = np.empty((N, n_steps + 1))
    r[:, 0] = r0
    z[:, 0] = z0

    vr = vr0.copy()
    vz = vz0.copy()
    vt = np.zeros(N)  # azimuthal velocity

    rr, zz = r0.copy(), z0.copy()

    for i in range(n_steps):
        br, bz = get_B(np.abs(rr), zz)
        # Boris half-kick electric field = 0 (no E-field in this visualisation)
        # Boris rotation step
        q_over_m = charge / mass
        tx = 0.0
        ty = 0.0
        tz = bz * q_over_m * dt * 0.5
        tr = br * q_over_m * dt * 0.5
        t2 = tx**2 + ty**2 + tz**2 + tr**2

        # v_minus
        vm_r = vr + vt * tz - vz * 0.0
        vm_z = vz + vr * 0.0 - vt * tr
        vm_t = vt + vz * tr - vr * tz

        # v_plus via rotation
        s_r = 2 * tr / (1 + t2)
        s_z = 2 * tz / (1 + t2)

        vp_r = vm_r + vm_t * s_z - vm_z * 0.0
        vp_z = vm_z + vm_r * 0.0 - vm_t * s_r
        vp_t = vm_t + vm_z * s_r - vm_r * s_z

        vr = vp_r + vt * tz - vz * 0.0
        vz = vp_z + vr * 0.0 - vt * tr
        vt = vp_t + vz * tr - vr * tz

        rr = rr + vr * dt
        zz = zz + vz * dt
        rr = np.abs(rr)          # reflecting boundary at r=0
        r[:, i + 1] = rr
        z[:, i + 1] = zz

    return r, z


# ---------------------------------------------------------------------------
# Seed particles near injection plane (z_min of domain)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)

z_inj = z_ax[0] + 0.05 * (z_ax[-1] - z_ax[0])  # 5% into domain
r_max_inj = 0.3 * r_ax[-1]

def thermal_v(T_eV: float, mass: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    v_th = math.sqrt(T_eV * E / mass)
    vr = rng.normal(0, v_th, N)
    vz = rng.normal(V_INJ, v_th, N)
    return vr, vz


# Ions
r_i0 = rng.uniform(0.01, r_max_inj, N_ION)
z_i0 = np.full(N_ION, z_inj) + rng.normal(0, dz * 2, N_ION)
vr_i, vz_i = thermal_v(T_I_EV, M_D, N_ION)

# Electrons
r_e0 = rng.uniform(0.01, r_max_inj, N_ELEC)
z_e0 = np.full(N_ELEC, z_inj) + rng.normal(0, dz * 2, N_ELEC)
vr_e, vz_e = thermal_v(T_E_EV, M_E, N_ELEC)

# Pick timestep: ~ 1/20 of ion cyclotron period at throat
B_throat = float(B_mag.max())
omega_ci = E * B_throat / M_D
dt_ion = (2 * math.pi / omega_ci) / 20.0
# Electrons need much smaller dt; use reduced mass for visualisation
# (purely visual — 100× lighter than physical)
M_E_VIS = M_D / 100.0
omega_ce_vis = E * B_throat / M_E_VIS
dt_elec = (2 * math.pi / omega_ce_vis) / 20.0

print(f"B_throat = {B_throat:.3f} T")
print(f"Ion cyclotron period = {2*math.pi/omega_ci*1e6:.2f} µs  dt = {dt_ion*1e9:.2f} ns")
print(f"Tracing {N_ION} ions + {N_ELEC} electrons for {N_STEPS} steps …")

r_ion, z_ion = boris_push_rz(r_i0, z_i0, vr_i, vz_i,  E, M_D,     dt_ion,  N_STEPS)
r_elec, z_elec = boris_push_rz(r_e0, z_e0, vr_e, vz_e, -E, M_E_VIS, dt_elec, N_STEPS)

print("Particle tracing complete.")

# ---------------------------------------------------------------------------
# Field-line seeds for background
# ---------------------------------------------------------------------------
def trace_fieldline(r0: float, z0: float, ds: float = 5e-3, nstep: int = 2000):
    """Simple forward+backward RK4 field-line trace."""
    def step_rk4(r, z, sign):
        for _ in range(nstep):
            br, bz = get_B(np.array([r]), np.array([z]))
            b = math.hypot(float(br[0]), float(bz[0]))
            if b < 1e-6:
                break
            r += sign * float(br[0]) / b * ds
            z += sign * float(bz[0]) / b * ds
            r = max(r, 0.0)
            if z < z_ax[0] or z > z_ax[-1] or r > r_ax[-1]:
                break
            yield r, z

    fwd  = list(step_rk4(r0, z0, +1))
    bwd  = list(step_rk4(r0, z0, -1))
    pts = list(reversed(bwd)) + [(r0, z0)] + fwd
    return np.array(pts)

r_seeds = np.linspace(0.03, 0.65, N_FIELD_LINES)
z_seed = float(z_ax[len(z_ax)//2])
field_lines = [trace_fieldline(float(rs), z_seed) for rs in r_seeds]

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
TRAIL = 60   # how many past positions to show as fading trail

fig, ax = plt.subplots(figsize=(11, 5), dpi=DPI)
fig.patch.set_facecolor("#0d0d1a")
ax.set_facecolor("#0d0d1a")

# Background: B-magnitude heat-map
im = ax.pcolormesh(
    z_ax, r_ax, B_mag,
    norm=LogNorm(vmin=max(B_mag.min(), 1e-3), vmax=B_mag.max()),
    cmap="inferno", shading="gouraud", rasterized=True,
)
cbar = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.85)
cbar.set_label("|B| (T)", color="white", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

# Field lines
for fl in field_lines:
    if len(fl) > 1:
        ax.plot(fl[:, 1], fl[:, 0], color="white", lw=0.4, alpha=0.35)

ax.set_xlabel("z (m)", color="white")
ax.set_ylabel("r (m)", color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

title = ax.set_title(
    "Helicon — Sunbird Magnetic Nozzle  |  step 0", color="white", fontsize=11
)

# Particle scatter objects
scat_ion  = ax.scatter([], [], s=6, c="#00e5ff", alpha=0.9, zorder=5, label="D⁺")
scat_elec = ax.scatter([], [], s=3, c="#ff6ec7", alpha=0.8, zorder=5, label="e⁻ (vis. mass)")

# Trails as LineCollection
from matplotlib.collections import LineCollection

trail_ions  = [ax.plot([], [], lw=0.6, color="#00e5ff", alpha=0.0)[0] for _ in range(N_ION)]
trail_elecs = [ax.plot([], [], lw=0.5, color="#ff6ec7", alpha=0.0)[0] for _ in range(N_ELEC)]

ax.legend(loc="upper right", facecolor="#111", labelcolor="white", fontsize=8, framealpha=0.6)
ax.set_xlim(z_ax[0], z_ax[-1])
ax.set_ylim(0, r_ax[-1])

def init():
    scat_ion.set_offsets(np.empty((0, 2)))
    scat_elec.set_offsets(np.empty((0, 2)))
    for l in trail_ions + trail_elecs:
        l.set_data([], [])
    return [scat_ion, scat_elec] + trail_ions + trail_elecs

def update(frame):
    s = min(frame, N_STEPS)
    t0 = max(0, s - TRAIL)

    # Current positions
    zi = z_ion[:, s]; ri = r_ion[:, s]
    ze = z_elec[:, s]; re = r_elec[:, s]
    # Mask particles that left the domain
    mask_i = (zi >= z_ax[0]) & (zi <= z_ax[-1]) & (ri <= r_ax[-1])
    mask_e = (ze >= z_ax[0]) & (ze <= z_ax[-1]) & (re <= r_ax[-1])

    scat_ion.set_offsets(np.column_stack([zi[mask_i], ri[mask_i]]))
    scat_elec.set_offsets(np.column_stack([ze[mask_e], re[mask_e]]))

    # Trails with fading alpha
    for k in range(N_ION):
        zz = z_ion[k, t0:s+1]
        rr = r_ion[k, t0:s+1]
        trail_ions[k].set_data(zz, rr)
        trail_ions[k].set_alpha(0.25 if mask_i[k] else 0.0)

    for k in range(N_ELEC):
        zz = z_elec[k, t0:s+1]
        rr = r_elec[k, t0:s+1]
        trail_elecs[k].set_data(zz, rr)
        trail_elecs[k].set_alpha(0.2 if mask_e[k] else 0.0)

    title.set_text(
        f"Helicon — Sunbird Magnetic Nozzle  |  "
        f"step {s:4d} / {N_STEPS}  "
        f"(D⁺ {mask_i.sum()}, e⁻ {mask_e.sum()} in domain)"
    )
    return [scat_ion, scat_elec, title] + trail_ions + trail_elecs

ani = animation.FuncAnimation(
    fig, update, frames=N_STEPS + 1,
    init_func=init, blit=True, interval=1000 / FPS,
)

OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving video → {OUT_VIDEO} …")
writer = animation.FFMpegWriter(fps=FPS, bitrate=2000, codec="h264",
                                extra_args=["-pix_fmt", "yuv420p"])
ani.save(str(OUT_VIDEO), writer=writer, dpi=DPI)
print(f"Done. {OUT_VIDEO} ({OUT_VIDEO.stat().st_size // 1024} KB)")
plt.close(fig)
