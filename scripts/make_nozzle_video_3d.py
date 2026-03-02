"""3D magnetic rocket nozzle animation — self-contained.

Analytically computes the B-field from circular current loops (elliptic-
integral Biot-Savart), traces 3-D magnetic field lines forming the nozzle
trumpet, runs a full 3-D Cartesian Boris pusher on D⁺ ions, and renders a
rotating-camera MP4.  No HDF5 or helicon-module imports required.

Usage:
    uv run python scripts/make_nozzle_video_3d.py
"""

from __future__ import annotations

import math
import pathlib

import matplotlib
matplotlib.use("Agg")

import matplotlib.animation as animation  # noqa: E402
import matplotlib.colors as mcolors       # noqa: E402
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402
from scipy.special import ellipe, ellipk              # noqa: E402
from tqdm import tqdm                                  # noqa: E402

# ── Output ────────────────────────────────────────────────────────────────────
OUT_VIDEO = pathlib.Path("results/nozzle_animation_3d.mp4")

# ── Grid (2-D RZ for B-field) ─────────────────────────────────────────────────
NR, NZ = 160, 420
R_MAX  = 0.38
Z_MIN, Z_MAX = -0.45, 1.30

# ── Simulation ────────────────────────────────────────────────────────────────
N_PARTICLES  = 60
N_STEPS      = 2400   # Boris steps
RECORD_EVERY = 4      # record every 4th step → 600 frames @ 25 FPS = 24 s
TRAIL        = 40     # history frames shown per particle
N_FL_RADII   = 6      # field-line seed radii
N_FL_AZ      = 8      # azimuths per radius → 48 lines total; last radius = wall
FPS          = 25
DPI          = 130

# ── Physics ───────────────────────────────────────────────────────────────────
MU0      = 4.0 * math.pi * 1e-7
E_CHARGE = 1.602176634e-19
M_D      = 3.3435837724e-27

# Coil layout: (z_m, radius_m, current_A)
COILS = [
    (-0.38, 0.14, 22_000),
    (-0.28, 0.12, 22_000),
    (-0.19, 0.10, 22_000),
    (-0.11, 0.08, 22_000),
    ( 0.00, 0.05, 65_000),   # throat — peak B
    ( 0.14, 0.16,  7_000),   # guide / exit
]

# Particle injection
# V_INJ is intentionally boosted for visualisation — we want ions to traverse
# the full nozzle (~1.6 m) within the ~1 µs simulation window.
T_ION_EV  = 12_000.0
V_INJ     = 2.5e6     # m/s  (visualisation speed; ~0.8% c)
Z_INJ     = -0.30
R_INJ_MAX = 0.06

# ── Analytical 2-D B-field ────────────────────────────────────────────────────

def _loop_field(
    r: np.ndarray, z: np.ndarray, z0: float, R: float, I: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Br, Bz for a circular current loop via elliptic integrals."""
    dz     = z - z0
    safe_r = np.where(r < 1e-9, 1e-9, r)
    beta2  = (R + safe_r) ** 2 + dz ** 2
    alpha2 = (R - safe_r) ** 2 + dz ** 2
    beta   = np.sqrt(beta2)
    k2     = np.clip(1.0 - alpha2 / beta2, 0.0, 1.0 - 1e-9)
    K  = ellipk(k2)
    Ek = ellipe(k2)
    C  = MU0 * I / (2.0 * math.pi)
    Bz_gen = (C / beta) * (K + (R**2 - safe_r**2 - dz**2) / alpha2 * Ek)
    Br_gen = (C * dz / (beta * safe_r)) * (-K + (R**2 + safe_r**2 + dz**2) / alpha2 * Ek)
    Bz_ax  = MU0 * I * R**2 / (2.0 * (R**2 + dz**2) ** 1.5)
    on_ax  = r < 1e-9
    return np.where(on_ax, 0.0, Br_gen), np.where(on_ax, Bz_ax, Bz_gen)


r_ax = np.linspace(0.0, R_MAX, NR)
z_ax = np.linspace(Z_MIN, Z_MAX, NZ)
R2D, Z2D = np.meshgrid(r_ax, z_ax, indexing="ij")

print("Computing 2-D RZ B-field …")
Br_g = np.zeros_like(R2D)
Bz_g = np.zeros_like(Z2D)
for z0, R, I in COILS:
    br, bz = _loop_field(R2D, Z2D, z0, R, I)
    Br_g += br
    Bz_g += bz

B_mag    = np.hypot(Br_g, Bz_g)
B_throat = float(B_mag.max())
print(f"  Peak |B| = {B_throat:.3f} T")

_iBz = RegularGridInterpolator((r_ax, z_ax), Bz_g, method="linear",
                                bounds_error=False, fill_value=0.0)
_iBr = RegularGridInterpolator((r_ax, z_ax), Br_g, method="linear",
                                bounds_error=False, fill_value=0.0)


def _get_B_rz(r: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.column_stack([np.clip(r, 0.0, R_MAX), np.clip(z, Z_MIN, Z_MAX)])
    return _iBr(pts), _iBz(pts)


def get_B_xyz(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian (Bx, By, Bz) by rotating the axisymmetric (Br, Bz)."""
    r    = np.hypot(x, y)
    br, bz = _get_B_rz(r, z)
    sr   = np.where(r < 1e-9, 0.0, br / np.where(r < 1e-9, 1.0, r))
    return sr * x, sr * y, bz


# ── 3-D field-line tracer (RK4) ───────────────────────────────────────────────

def _trace_3d(
    x0: float, y0: float, z0: float,
    ds: float = 4e-3, nstep: int = 5_000,
) -> np.ndarray:
    def _walk(x: float, y: float, z: float, sign: int) -> list:
        pts: list[tuple] = []
        for _ in range(nstep):
            bx, by, bz = get_B_xyz(
                np.array([x]), np.array([y]), np.array([z])
            )
            b = math.sqrt(float(bx[0])**2 + float(by[0])**2 + float(bz[0])**2)
            if b < 1e-8:
                break
            x += sign * float(bx[0]) / b * ds
            y += sign * float(by[0]) / b * ds
            z += sign * float(bz[0]) / b * ds
            if z < Z_MIN or z > Z_MAX or x**2 + y**2 > R_MAX**2:
                pts.append((x, y, z))
                break
            pts.append((x, y, z))
        return pts

    fwd = _walk(x0, y0, z0, +1)
    bwd = _walk(x0, y0, z0, -1)
    return np.array(list(reversed(bwd)) + [(x0, y0, z0)] + fwd)


print("Tracing field lines …")
radii    = np.linspace(0.012, R_MAX * 0.88, N_FL_RADII)
azimuths = np.linspace(0.0, 2.0 * math.pi, N_FL_AZ, endpoint=False)

inner_lines: list[np.ndarray] = []
wall_lines:  list[np.ndarray] = []

for phi in tqdm(azimuths, desc="Field lines", unit="az"):
    for k, r in enumerate(radii):
        fl = _trace_3d(r * math.cos(phi), r * math.sin(phi), 0.0)
        if len(fl) < 2:
            continue
        if k == len(radii) - 1:
            wall_lines.append(fl)
        else:
            inner_lines.append(fl)

print(f"  {len(inner_lines)} inner + {len(wall_lines)} wall field lines")

# ── Nozzle-wall surface (outermost 2-D field line → revolved mesh) ────────────

def _trace_2d_outer(r0: float, ds: float = 4e-3, nstep: int = 8_000) -> np.ndarray:
    """Trace the outermost 2-D field line (r, z) forward + backward from throat."""
    def _walk2(r: float, z: float, sign: int) -> list:
        pts: list[tuple] = []
        for _ in range(nstep):
            br, bz = _get_B_rz(np.array([r]), np.array([z]))
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

    fwd = _walk2(r0, 0.0, +1)
    bwd = _walk2(r0, 0.0, -1)
    pts = list(reversed(bwd)) + [(r0, 0.0)] + fwd
    return np.array(pts)


print("Building nozzle-wall surface …")
wall_2d   = _trace_2d_outer(R_MAX * 0.88)           # (M, 2): columns = [r, z]
wall_r_rz = wall_2d[:, 0]
wall_z_rz = wall_2d[:, 1]

# Sort & uniformly sample in z for a clean surface grid
sort_idx  = np.argsort(wall_z_rz)
wall_r_s  = wall_r_rz[sort_idx]
wall_z_s  = wall_z_rz[sort_idx]
n_zsamp   = 80
n_tsamp   = 36
z_samp    = np.linspace(wall_z_s.min(), wall_z_s.max(), n_zsamp)
r_samp    = np.interp(z_samp, wall_z_s, wall_r_s)

theta_samp = np.linspace(0.0, 2.0 * math.pi, n_tsamp, endpoint=False)
SURF_Z = np.tile(z_samp[:, None], (1, n_tsamp))           # (n_z, n_t)
SURF_X = r_samp[:, None] * np.cos(theta_samp[None, :])
SURF_Y = r_samp[:, None] * np.sin(theta_samp[None, :])

# ── 3-D Boris pusher ──────────────────────────────────────────────────────────

def _boris_3d(
    pos: np.ndarray, vel: np.ndarray,
    dt: float, q_over_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One Boris step.  pos/vel: (3, N) float64 arrays.  Returns updated copies."""
    bx, by, bz = get_B_xyz(pos[0], pos[1], pos[2])
    hqm = 0.5 * q_over_m * dt
    tx, ty, tz = hqm * bx, hqm * by, hqm * bz
    t2 = tx**2 + ty**2 + tz**2

    vx, vy, vz = vel[0], vel[1], vel[2]
    vpx = vx + vy * tz - vz * ty
    vpy = vy + vz * tx - vx * tz
    vpz = vz + vx * ty - vy * tx

    sx = 2.0 * tx / (1.0 + t2)
    sy = 2.0 * ty / (1.0 + t2)
    sz = 2.0 * tz / (1.0 + t2)

    vx_n = vx + vpy * sz - vpz * sy
    vy_n = vy + vpz * sx - vpx * sz
    vz_n = vz + vpx * sy - vpy * sx

    vel_new = np.array([vx_n, vy_n, vz_n])
    pos_new = pos + vel_new * dt
    return pos_new, vel_new


# Seed particles
rng   = np.random.default_rng(42)
v_th  = math.sqrt(T_ION_EV * E_CHARGE / M_D)
r_rnd = R_INJ_MAX * np.sqrt(rng.uniform(0.0, 1.0, N_PARTICLES))
p_rnd = rng.uniform(0.0, 2.0 * math.pi, N_PARTICLES)

pos = np.array([
    r_rnd * np.cos(p_rnd),
    r_rnd * np.sin(p_rnd),
    np.full(N_PARTICLES, Z_INJ),
], dtype=np.float64)

vel = np.array([
    rng.normal(0.0, v_th * 0.20, N_PARTICLES),
    rng.normal(0.0, v_th * 0.20, N_PARTICLES),
    rng.normal(V_INJ, v_th * 0.60, N_PARTICLES),
], dtype=np.float64)

# Time-step: 1/28 of ion cyclotron period at throat
omega_ci = E_CHARGE * B_throat / M_D
dt       = (2.0 * math.pi / omega_ci) / 28.0
q_over_m = E_CHARGE / M_D

N_FRAMES_MAX = N_STEPS // RECORD_EVERY + 2
print(f"dt = {dt*1e9:.2f} ns  ·  t_total = {N_STEPS*dt*1e6:.2f} µs")
print(f"Running Boris pusher: {N_PARTICLES} ions × {N_STEPS} steps …")

pos_hist = np.full((N_FRAMES_MAX, 3, N_PARTICLES), np.nan, dtype=np.float64)
spd_hist = np.full((N_FRAMES_MAX, N_PARTICLES),    np.nan, dtype=np.float64)
active   = np.ones(N_PARTICLES, dtype=bool)
exited   = np.zeros(N_PARTICLES, dtype=bool)

frame_idx = 0
pos_hist[frame_idx] = pos
spd_hist[frame_idx] = np.linalg.norm(vel, axis=0)
frame_idx += 1

for step in tqdm(range(N_STEPS), desc="Boris push", unit="step"):
    idx = np.where(active)[0]
    if len(idx) == 0:
        break
    p_new, v_new = _boris_3d(pos[:, idx], vel[:, idx], dt, q_over_m)
    pos[:, idx] = p_new
    vel[:, idx] = v_new

    out_z  = p_new[2] >= Z_MAX
    out_zm = p_new[2] <= Z_MIN
    out_r  = p_new[0]**2 + p_new[1]**2 > (R_MAX * 1.05)**2
    exited[idx[out_z]] = True
    gone = out_z | out_r | out_zm
    active[idx[gone]] = False

    if (step + 1) % RECORD_EVERY == 0 and frame_idx < N_FRAMES_MAX:
        pos_hist[frame_idx] = pos
        spd_hist[frame_idx] = np.linalg.norm(vel, axis=0)
        # NaN-mask particles that left the domain
        for i in idx[gone]:
            pos_hist[frame_idx:, :, i] = np.nan
            spd_hist[frame_idx:,    i] = np.nan
        frame_idx += 1

N_FRAMES = frame_idx
pos_hist = pos_hist[:N_FRAMES]
spd_hist = spd_hist[:N_FRAMES]
print(f"Done — {int(exited.sum())} ions exhausted  ·  {N_FRAMES} frames recorded")

v_lo = float(np.nanpercentile(spd_hist,  5))
v_hi = float(np.nanpercentile(spd_hist, 98))

# ── Build figure ──────────────────────────────────────────────────────────────
print("Building animation …")

fig = plt.figure(figsize=(13, 7.5), dpi=DPI)
fig.patch.set_facecolor("#060610")
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#060610")
ax.grid(False)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#1a1a2e")

# Axis limits: display convention  x-axis=z(axial), y-axis=x, z-axis=y
ax.set_xlim(Z_MIN, Z_MAX)
ax.set_ylim(-R_MAX * 1.05, R_MAX * 1.05)
ax.set_zlim(-R_MAX * 1.05, R_MAX * 1.05)
ax.set_xlabel("z  (m)", color="#888899", fontsize=8, labelpad=8)
ax.set_ylabel("x  (m)", color="#888899", fontsize=8, labelpad=4)
ax.set_zlabel("y  (m)", color="#888899", fontsize=8, labelpad=4)
ax.tick_params(colors="#444455", labelsize=6)
ax.view_init(elev=22, azim=35)

# ── Static geometry ───────────────────────────────────────────────────────────

# Nozzle-wall translucent surface
ax.plot_surface(
    SURF_Z, SURF_X, SURF_Y,
    alpha=0.055, color="#33aaee",
    rstride=1, cstride=1,
    linewidth=0, antialiased=False,
)

# Inner field lines (dim)
for fl in inner_lines:
    ax.plot(fl[:, 2], fl[:, 0], fl[:, 1],
            color="#aaccff", lw=0.30, alpha=0.18)

# Wall field lines (bright cyan — defines nozzle boundary)
for fl in wall_lines:
    ax.plot(fl[:, 2], fl[:, 0], fl[:, 1],
            color="#22ffee", lw=0.80, alpha=0.55)

# Coil rings with glow layers
phi_c = np.linspace(0.0, 2.0 * math.pi, 100)
for z0, R, _ in COILS:
    xc = R * np.cos(phi_c)
    yc = R * np.sin(phi_c)
    zc = np.full(100, z0)
    for lw, al in [(5.0, 0.06), (3.0, 0.18), (1.5, 0.55), (0.8, 0.90)]:
        ax.plot(zc, xc, yc, color="#ff8800", lw=lw, alpha=al)

# Axis line (z-axis / symmetry axis)
ax.plot([Z_MIN, Z_MAX], [0, 0], [0, 0],
        color="#ffffff", lw=0.4, alpha=0.12, ls="--")

# Throat marker
ax.plot([0, 0], [-R_MAX, R_MAX], [0, 0],
        color="#ffffff", lw=0.3, alpha=0.12, ls=":")

# ── Dynamic objects ───────────────────────────────────────────────────────────

# Particle scatter (speed-coloured)
scat = ax.scatter(
    np.zeros(1), np.zeros(1), np.zeros(1),
    s=10, c=[0.5], cmap="plasma", vmin=0.0, vmax=1.0,
    depthshade=False, alpha=0.95, zorder=8,
)
# Soft glow halo
scat_glow = ax.scatter(
    np.zeros(1), np.zeros(1), np.zeros(1),
    s=55, c=[0.5], cmap="plasma", vmin=0.0, vmax=1.0,
    depthshade=False, alpha=0.12, zorder=7,
)

# Trail lines (one per particle)
trail_lines = [
    ax.plot([], [], [], lw=0.45, color="#33ddff", alpha=0.0,
            solid_capstyle="round")[0]
    for _ in range(N_PARTICLES)
]

title_obj = ax.set_title(
    "Helicon  ·  3-D Magnetic Rocket Nozzle  |  frame 0",
    color="white", fontsize=10, pad=6,
)

# Camera sweep parameters
AZ_START = 35.0
AZ_SWEEP = 300.0  # degrees over full animation → near-full rotation


# ── Animation callbacks ───────────────────────────────────────────────────────

def _norm_speed(s: np.ndarray) -> np.ndarray:
    return np.clip((s - v_lo) / max(v_hi - v_lo, 1.0), 0.0, 1.0)


def init():
    scat._offsets3d      = (np.array([]), np.array([]), np.array([]))
    scat_glow._offsets3d = (np.array([]), np.array([]), np.array([]))
    scat.set_array(np.array([]))
    scat_glow.set_array(np.array([]))
    for ln in trail_lines:
        ln.set_data_3d([], [], [])
        ln.set_alpha(0.0)
    return [scat, scat_glow, title_obj] + trail_lines


def update(frame: int):
    f = min(frame, N_FRAMES - 1)

    # ── Camera rotation ──────────────────────────────────────────────────
    az = AZ_START + AZ_SWEEP * f / max(N_FRAMES - 1, 1)
    ax.view_init(elev=22, azim=az)

    # ── Current positions ────────────────────────────────────────────────
    px = pos_hist[f, 0]
    py = pos_hist[f, 1]
    pz = pos_hist[f, 2]
    sp = spd_hist[f]
    valid = ~np.isnan(pz)

    if valid.any():
        xs, ys, zs = pz[valid], px[valid], py[valid]   # display: z→X, x→Y, y→Z
        sc = _norm_speed(sp[valid])
        scat._offsets3d      = (xs, ys, zs)
        scat_glow._offsets3d = (xs, ys, zs)
        scat.set_array(sc)
        scat_glow.set_array(sc)
    else:
        scat._offsets3d      = (np.array([]), np.array([]), np.array([]))
        scat_glow._offsets3d = (np.array([]), np.array([]), np.array([]))

    # ── Trails ───────────────────────────────────────────────────────────
    t0 = max(0, f - TRAIL)
    for k in range(N_PARTICLES):
        trail_z = pos_hist[t0:f + 1, 2, k]
        trail_x = pos_hist[t0:f + 1, 0, k]
        trail_y = pos_hist[t0:f + 1, 1, k]
        ok = ~np.isnan(trail_z)
        if ok.any():
            trail_lines[k].set_data_3d(trail_z[ok], trail_x[ok], trail_y[ok])
            trail_lines[k].set_alpha(0.30 if valid[k] else 0.0)
        else:
            trail_lines[k].set_alpha(0.0)

    n_active  = int(valid.sum())
    n_exited  = int(exited.sum())
    title_obj.set_text(
        f"Helicon  ·  3-D Magnetic Rocket Nozzle"
        f"  |  step {f * RECORD_EVERY:4d}/{N_STEPS}"
        f"  |  {n_active} active  ·  {n_exited} exhausted"
    )
    return [scat, scat_glow, title_obj] + trail_lines


ani = animation.FuncAnimation(
    fig, update,
    frames=N_FRAMES,
    init_func=init,
    blit=False,   # blit=True is unreliable for Axes3D
    interval=1000 / FPS,
)

# ── Save with tqdm progress ───────────────────────────────────────────────────
OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving → {OUT_VIDEO}  ({N_FRAMES} frames @ {FPS} FPS, DPI={DPI}) …")

writer = animation.FFMpegWriter(
    fps=FPS, bitrate=4500, codec="h264",
    extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
)

pbar = tqdm(total=N_FRAMES, desc="Rendering", unit="frame")
_orig_grab = writer.grab_frame

def _grab_with_tqdm(*args, **kwargs):
    _orig_grab(*args, **kwargs)
    pbar.update(1)

writer.grab_frame = _grab_with_tqdm
ani.save(str(OUT_VIDEO), writer=writer, dpi=DPI)
pbar.close()

sz_mb = OUT_VIDEO.stat().st_size / 1024 / 1024
print(f"Done.  {OUT_VIDEO}  ({sz_mb:.1f} MB  ·  {N_FRAMES/FPS:.0f} s)")
plt.close(fig)
