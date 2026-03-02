"""Rocket nozzle animation — self-contained magnetic nozzle visualisation.

Analytically computes a solenoid + nozzle-coil B-field using the exact
Biot-Savart formula for circular current loops (via complete elliptic
integrals), traces magnetic field lines that form the nozzle "trumpet"
shape, then runs a Boris pusher on D⁺ ions and renders an MP4.

No HDF5 or external data files required — fully self-contained.

Usage:
    uv run python scripts/make_nozzle_video.py
"""

from __future__ import annotations

import math
import pathlib

import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import ellipe, ellipk
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT_VIDEO = pathlib.Path("results/nozzle_animation.mp4")

# Grid
NR, NZ = 150, 400
R_MAX = 0.38          # m — max radial extent
Z_MIN, Z_MAX = -0.45, 1.30   # m — throat at z≈0, plume expands right

# Animation
N_ION = 28
N_FIELD_LINES = 18    # includes the "nozzle wall" outermost line
N_STEPS = 900         # Boris steps  (≈ 30 s at 30 FPS)
TRAIL = 90            # fading history length
FPS = 30
DPI = 150

# Physical constants
MU0 = 4 * math.pi * 1e-7
E_CHARGE = 1.602176634e-19
M_D = 3.3435837724e-27   # deuterium mass [kg]

# Coil layout: (z_centre_m, radius_m, current_A)
# Five source-solenoid loops (z < 0) create an upstream mirror.
# One strong throat coil at z=0 sets peak-B.
# One guide coil downstream provides a slight focusing ridge.
COILS = [
    # ── Source solenoid ──────────────────────────────────────────────────
    (-0.38, 0.14, 22_000),
    (-0.28, 0.12, 22_000),
    (-0.19, 0.10, 22_000),
    (-0.11, 0.08, 22_000),
    # ── Throat (maximum B) ───────────────────────────────────────────────
    ( 0.00, 0.05, 65_000),
    # ── Guide / exit coil ────────────────────────────────────────────────
    ( 0.14, 0.16,  7_000),
]

# Ion injection parameters
T_ION_EV  = 12_000.0    # ion temperature [eV]
V_INJ     = 5.0e5       # axial injection drift [m/s]
Z_INJ     = -0.30       # injection z-plane [m]
R_INJ_MAX = 0.06        # injection radial aperture [m]

# ---------------------------------------------------------------------------
# Analytical B-field  ── exact current-loop formula via elliptic integrals
# ---------------------------------------------------------------------------

def _loop_field(
    r: np.ndarray, z: np.ndarray, z0: float, R: float, I: float
) -> tuple[np.ndarray, np.ndarray]:
    """Br, Bz for one circular current loop at z=z0, radius R, current I."""
    dz = z - z0
    # Protect r=0 numerically; we correct below
    safe_r = np.where(r < 1e-9, 1e-9, r)

    beta2 = (R + safe_r) ** 2 + dz ** 2
    alpha2 = (R - safe_r) ** 2 + dz ** 2
    beta = np.sqrt(beta2)
    k2 = np.clip(1.0 - alpha2 / beta2, 0.0, 1.0 - 1e-9)

    K = ellipk(k2)
    Ek = ellipe(k2)

    C = MU0 * I / (2.0 * math.pi)

    # General r ≠ 0 expressions
    Bz_gen = (C / beta) * (K + (R**2 - safe_r**2 - dz**2) / alpha2 * Ek)
    Br_gen = (C * dz / (beta * safe_r)) * (-K + (R**2 + safe_r**2 + dz**2) / alpha2 * Ek)

    # On-axis: Bz = μ₀IR²/[2(R²+dz²)^(3/2)], Br = 0
    Bz_axis = MU0 * I * R**2 / (2.0 * (R**2 + dz**2) ** 1.5)
    on_axis = r < 1e-9
    Bz = np.where(on_axis, Bz_axis, Bz_gen)
    Br = np.where(on_axis, 0.0, Br_gen)
    return Br, Bz


def _compute_grid_field(
    r_ax: np.ndarray, z_ax: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    R2D, Z2D = np.meshgrid(r_ax, z_ax, indexing="ij")  # (NR, NZ)
    Br_tot = np.zeros_like(R2D)
    Bz_tot = np.zeros_like(Z2D)
    for z0, R, I in COILS:
        br, bz = _loop_field(R2D, Z2D, z0, R, I)
        Br_tot += br
        Bz_tot += bz
    return Br_tot, Bz_tot


# Build grid & compute field
r_ax = np.linspace(0.0, R_MAX, NR)
z_ax = np.linspace(Z_MIN, Z_MAX, NZ)
print("Computing B-field on grid …")
Br_grid, Bz_grid = _compute_grid_field(r_ax, z_ax)
B_mag = np.hypot(Br_grid, Bz_grid)
B_throat = float(B_mag.max())
idx_max = np.unravel_index(B_mag.argmax(), B_mag.shape)
print(f"  Peak |B| = {B_throat:.3f} T at (r={r_ax[idx_max[0]]:.3f}, z={z_ax[idx_max[1]]:.3f}) m")

# Build interpolators
interp_Bz = RegularGridInterpolator(
    (r_ax, z_ax), Bz_grid, method="linear", bounds_error=False, fill_value=0.0
)
interp_Br = RegularGridInterpolator(
    (r_ax, z_ax), Br_grid, method="linear", bounds_error=False, fill_value=0.0
)


def get_B(r: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.column_stack(
        [np.clip(r, r_ax[0], r_ax[-1]), np.clip(z, z_ax[0], z_ax[-1])]
    )
    return interp_Br(pts), interp_Bz(pts)


# ---------------------------------------------------------------------------
# Magnetic field-line tracing  (RK4 forward + backward)
# ---------------------------------------------------------------------------

def _trace_fieldline(
    r0: float, z0: float, ds: float = 4e-3, nstep: int = 6000
) -> np.ndarray:
    def _walk(r, z, sign):
        pts = []
        for _ in range(nstep):
            br, bz = get_B(np.array([r]), np.array([z]))
            b = math.hypot(float(br[0]), float(bz[0]))
            if b < 1e-8:
                break
            r += sign * float(br[0]) / b * ds
            z += sign * float(bz[0]) / b * ds
            r = max(r, 0.0)
            if z < Z_MIN or z > Z_MAX or r > R_MAX:
                break
            pts.append((r, z))
        return pts

    fwd = _walk(r0, z0, +1)
    bwd = _walk(r0, z0, -1)
    pts = list(reversed(bwd)) + [(r0, z0)] + fwd
    return np.array(pts) if len(pts) > 1 else np.array([[r0, z0]])


print("Tracing field lines …")
r_seeds = np.linspace(0.004, R_MAX * 0.90, N_FIELD_LINES)
z_seed = 0.0
field_lines = [_trace_fieldline(float(rs), z_seed) for rs in r_seeds]
nozzle_wall = field_lines[-1]   # outermost field line = magnetic nozzle wall
print(f"  {len(field_lines)} field lines traced.")

# ---------------------------------------------------------------------------
# Boris pusher in 2D-RZ with azimuthal velocity (correct cross-product)
# ---------------------------------------------------------------------------

def _boris_push(
    r0: np.ndarray,
    z0: np.ndarray,
    vr0: np.ndarray,
    vz0: np.ndarray,
    charge: float,
    mass: float,
    dt: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised Boris pusher in cylindrical coordinates (Bθ=0).

    Returns:
        r_hist  shape (N, n_steps+1)
        z_hist  shape (N, n_steps+1)
        v_hist  |v| shape (N, n_steps+1)   (for particle colouring)
    """
    N = len(r0)
    r_h = np.empty((N, n_steps + 1))
    z_h = np.empty((N, n_steps + 1))
    v_h = np.empty((N, n_steps + 1))

    rr, zz = r0.copy(), z0.copy()
    vr, vz = vr0.copy(), vz0.copy()
    vt = np.zeros(N)   # azimuthal velocity

    r_h[:, 0] = rr
    z_h[:, 0] = zz
    v_h[:, 0] = np.sqrt(vr**2 + vt**2 + vz**2)

    q_over_2m = charge / (2.0 * mass)

    for i in tqdm(range(n_steps), desc="Boris push", unit="step", leave=False):
        br, bz = get_B(np.abs(rr), zz)

        # t-vector  (eθ component = 0 for axisymmetric B)
        tr = br * q_over_2m * dt
        tz = bz * q_over_2m * dt
        t2 = tr**2 + tz**2

        # v⁻ = v  (no E-field)
        vmr, vmt, vmz = vr, vt, vz

        # v' = v⁻ + v⁻ × t   (cylindrical cross-product with Bθ=0)
        vpr = vmr + vmt * tz
        vpt = vmt + vmz * tr - vmr * tz
        vpz = vmz - vmt * tr

        # s = 2t / (1 + |t|²)
        sr = 2.0 * tr / (1.0 + t2)
        sz = 2.0 * tz / (1.0 + t2)

        # v⁺ = v⁻ + v' × s
        vr = vmr + vpt * sz
        vt = vmt + vpz * sr - vpr * sz
        vz = vmz - vpt * sr

        rr = np.abs(rr + vr * dt)
        zz = zz + vz * dt

        r_h[:, i + 1] = rr
        z_h[:, i + 1] = zz
        v_h[:, i + 1] = np.sqrt(vr**2 + vt**2 + vz**2)

    return r_h, z_h, v_h


# Seed particles
rng = np.random.default_rng(42)
v_th = math.sqrt(T_ION_EV * E_CHARGE / M_D)
r0 = rng.uniform(0.004, R_INJ_MAX, N_ION)
z0 = np.full(N_ION, Z_INJ) + rng.normal(0.0, 3e-3, N_ION)
vr0 = rng.normal(0.0, v_th * 0.25, N_ION)
vz0 = rng.normal(V_INJ, v_th * 0.80, N_ION)

# Time-step: 1/28 of ion cyclotron period at the throat
omega_ci = E_CHARGE * B_throat / M_D
dt = (2.0 * math.pi / omega_ci) / 28.0
t_total_us = N_STEPS * dt * 1e6

print(f"ω_ci = {omega_ci:.3e} rad/s  →  dt = {dt*1e9:.2f} ns  "
      f"  t_total = {t_total_us:.2f} µs")
print(f"Tracing {N_ION} ions for {N_STEPS} steps …")
r_ion, z_ion, v_ion = _boris_push(r0, z0, vr0, vz0, E_CHARGE, M_D, dt, N_STEPS)
print("Boris pusher complete.")

# Speed extremes for particle colouring
v_lo = float(np.percentile(v_ion, 5))
v_hi = float(np.percentile(v_ion, 99))

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(14, 5), dpi=DPI)
fig.patch.set_facecolor("#060610")
ax = fig.add_axes([0.06, 0.10, 0.80, 0.82])
ax.set_facecolor("#060610")

# ── Background: |B| heat-map (log scale) ────────────────────────────────
B_plot = np.clip(B_mag, 1e-3, None)
im = ax.pcolormesh(
    z_ax, r_ax, B_plot,
    norm=mcolors.LogNorm(vmin=B_plot.min() * 1.5, vmax=B_plot.max()),
    cmap="inferno",
    shading="gouraud",
    rasterized=True,
    alpha=0.85,
)
cbar_ax = fig.add_axes([0.88, 0.12, 0.015, 0.76])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("|B|  (T)", color="#cccccc", fontsize=9, labelpad=6)
cbar.ax.yaxis.set_tick_params(color="#aaaaaa")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa", fontsize=7)

# ── Axis of symmetry ────────────────────────────────────────────────────
ax.axhline(0, color="#ffffff", lw=0.4, alpha=0.12, ls="--")

# ── Coil cross-sections ──────────────────────────────────────────────────
for z0_c, R_c, _ in COILS:
    hw = (z_ax[1] - z_ax[0]) * 1.5      # half-width in z
    rect_r = plt.Rectangle(
        (z0_c - hw, 0.001), 2 * hw, R_c - 0.001,
        linewidth=0, facecolor="#ff9900", alpha=0.18, zorder=2,
    )
    ax.add_patch(rect_r)
    ax.plot([z0_c - hw, z0_c + hw], [R_c, R_c],
            color="#ffaa33", lw=1.0, alpha=0.55, zorder=3)

# ── Magnetic field lines ─────────────────────────────────────────────────
for k, fl in enumerate(field_lines):
    if len(fl) < 2:
        continue
    is_wall = k == len(field_lines) - 1
    lw    = 1.4 if is_wall else 0.35
    alpha = 0.75 if is_wall else 0.28
    col   = "#44ddff" if is_wall else "#88eeff"
    ax.plot(fl[:, 1], fl[:, 0], color=col, lw=lw, alpha=alpha, zorder=4)

# ── Throat marker ────────────────────────────────────────────────────────
ax.axvline(0.0, color="#ffffff", lw=0.5, alpha=0.20, ls=":", zorder=2)
ax.text(0.005, R_MAX * 0.91, "throat", color="#aaaaaa", fontsize=7,
        va="top", ha="left")

# ── Particle scatter with speed-based colouring ──────────────────────────
scat = ax.scatter(
    [], [], s=12, c=[], cmap="plasma",
    vmin=v_lo, vmax=v_hi,
    alpha=0.95, zorder=8, linewidths=0,
)
# Halo layer (larger, low-alpha glow)
scat_glow = ax.scatter(
    [], [], s=55, c=[], cmap="plasma",
    vmin=v_lo, vmax=v_hi,
    alpha=0.15, zorder=7, linewidths=0,
)

# Fading trails
trails = [
    ax.plot([], [], lw=0.55, color="#22ddff", alpha=0.0, zorder=6, solid_capstyle="round")[0]
    for _ in range(N_ION)
]

# ── Labels & limits ──────────────────────────────────────────────────────
ax.set_xlabel("z  (m)", color="#cccccc", fontsize=10, labelpad=4)
ax.set_ylabel("r  (m)", color="#cccccc", fontsize=10, labelpad=4)
ax.tick_params(colors="#aaaaaa", labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor("#333344")
ax.set_xlim(Z_MIN, Z_MAX)
ax.set_ylim(0.0, R_MAX)

title_text = ax.set_title(
    "Helicon  ·  Magnetic Rocket Nozzle  |  step     0 / %d" % N_STEPS,
    color="white", fontsize=11, pad=7, loc="left",
)

# ── Legend ───────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
legend_items = [
    Line2D([0], [0], color="#44ddff", lw=1.2, label="nozzle wall (outermost B-line)"),
    Line2D([0], [0], color="#88eeff", lw=0.6, label="magnetic field lines"),
    Line2D([0], [0], marker="o", color="w", markersize=5,
           markerfacecolor="#ff55aa", lw=0, label="D⁺  (colour = speed)"),
]
ax.legend(handles=legend_items, loc="upper right",
          facecolor="#111122", edgecolor="#333355",
          labelcolor="#cccccc", fontsize=7.5, framealpha=0.75)

# Add speed colour-bar inline (small, top-right area of axes)
sm = plt.cm.ScalarMappable(cmap="plasma",
                            norm=mcolors.Normalize(vmin=v_lo / 1e5, vmax=v_hi / 1e5))
sm.set_array([])


# ---------------------------------------------------------------------------
# Animation callbacks
# ---------------------------------------------------------------------------

def init():
    scat.set_offsets(np.empty((0, 2)))
    scat.set_array(np.array([]))
    scat_glow.set_offsets(np.empty((0, 2)))
    scat_glow.set_array(np.array([]))
    for ln in trails:
        ln.set_data([], [])
    return [scat, scat_glow, title_text] + trails


def update(frame: int):
    s = min(frame, N_STEPS)
    t0 = max(0, s - TRAIL)

    zi = z_ion[:, s]
    ri = r_ion[:, s]
    vi = v_ion[:, s]
    mask = (zi >= Z_MIN) & (zi <= Z_MAX) & (ri <= R_MAX)

    if mask.any():
        pts = np.column_stack([zi[mask], ri[mask]])
        colors = vi[mask]
        scat.set_offsets(pts)
        scat.set_array(colors)
        scat_glow.set_offsets(pts)
        scat_glow.set_array(colors)
    else:
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))
        scat_glow.set_offsets(np.empty((0, 2)))
        scat_glow.set_array(np.array([]))

    # Trails
    for k in range(N_ION):
        if mask[k]:
            zz = z_ion[k, t0 : s + 1]
            rr = r_ion[k, t0 : s + 1]
            trails[k].set_data(zz, rr)
            trails[k].set_alpha(0.30)
        else:
            trails[k].set_alpha(0.0)

    n_in = int(mask.sum())
    n_exited = int((z_ion[:, s] > Z_MAX).sum())
    title_text.set_text(
        f"Helicon  ·  Magnetic Rocket Nozzle"
        f"  |  step {s:4d} / {N_STEPS}"
        f"  |  {n_in} ions in domain"
        f"  |  {n_exited} exhausted"
    )
    return [scat, scat_glow, title_text] + trails


ani = animation.FuncAnimation(
    fig, update,
    frames=N_STEPS + 1,
    init_func=init,
    blit=True,
    interval=1000 / FPS,
)

OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
total_frames = N_STEPS + 1
print(f"Saving → {OUT_VIDEO}  ({total_frames} frames @ {FPS} FPS, DPI={DPI}) …")
writer = animation.FFMpegWriter(
    fps=FPS, bitrate=4000, codec="h264",
    extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
)

pbar = tqdm(total=total_frames, desc="Rendering", unit="frame")
_orig_grab = writer.grab_frame

def _grab_with_progress(*args, **kwargs):
    _orig_grab(*args, **kwargs)
    pbar.update(1)

writer.grab_frame = _grab_with_progress
ani.save(str(OUT_VIDEO), writer=writer, dpi=DPI)
pbar.close()

sz_kb = OUT_VIDEO.stat().st_size // 1024
print(f"Done.  {OUT_VIDEO}  ({sz_kb} KB  ·  {sz_kb / 1024:.1f} MB)")
plt.close(fig)
