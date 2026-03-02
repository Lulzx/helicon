"""Generate a 3D animation of the Sunbird magnetic nozzle simulation.

Computes the full 3D B-field (Biot-Savart), traces magnetic field lines,
runs a Boris-pusher particle simulation recording trajectories, then
renders a rotating-view MP4.

Usage:
    uv run python scripts/make_video_3d.py
"""

from __future__ import annotations

import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from helicon.fields.biot_savart_3d import Coil3D, Grid3D, compute_bfield_3d  # noqa: E402
from helicon.runner.sim3d import _boris_step, _make_grid_params  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_VIDEO = pathlib.Path("results/sunbird_3d/simulation_3d.mp4")

COILS = [
    Coil3D(z=0.0,  r=0.12, I=95000.0),
    Coil3D(z=0.08, r=0.10, I=47000.0),
]
Z_MIN, Z_MAX = -0.3, 2.0
R_MAX = 0.6
GRID = Grid3D(
    x_min=-R_MAX, x_max=R_MAX,
    y_min=-R_MAX, y_max=R_MAX,
    z_min=Z_MIN,  z_max=Z_MAX,
    nx=24, ny=24, nz=48,
)

N_PARTICLES   = 200    # visible ions
N_STEPS       = 800    # Boris steps
RECORD_EVERY  = 4      # sample interval → N_FRAMES = N_STEPS // RECORD_EVERY
TRAIL         = 18     # trailing steps shown per particle
N_FIELD_SEEDS = 5      # radii × 6 azimuths = 30 field lines
FPS           = 25
DPI           = 130

E_CHARGE = 1.602e-19
M_D      = 3.3435837724e-27   # deuterium
T_I_EV   = 100.0
V_INJ    = 50_000.0           # m/s

# ---------------------------------------------------------------------------
# 1. Compute 3D B-field
# ---------------------------------------------------------------------------
print("Computing 3D B-field …")
bfield = compute_bfield_3d(COILS, GRID, backend="auto", n_phi=32)
print(f"  Mirror ratio = {bfield.mirror_ratio():.1f}")
gp = _make_grid_params(bfield)
x0g, dxg, y0g, dyg, z0g, dzg, nxg, nyg, nzg, nynzg, Bxf, Byf, Bzf = gp


def _interp_B_scalar(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Single-point B-field trilinear interpolation (for field-line tracing)."""
    xi_f = (x - x0g) / dxg
    yi_f = (y - y0g) / dyg
    zi_f = (z - z0g) / dzg
    xi = int(np.clip(xi_f, 0, nxg - 2))
    yi = int(np.clip(yi_f, 0, nyg - 2))
    zi = int(np.clip(zi_f, 0, nzg - 2))
    tx = float(np.clip(xi_f - xi, 0.0, 1.0))
    ty = float(np.clip(yi_f - yi, 0.0, 1.0))
    tz = float(np.clip(zi_f - zi, 0.0, 1.0))
    omtx, omty, omtz = 1 - tx, 1 - ty, 1 - tz
    i000 = xi * nynzg + yi * nzg + zi
    i001 = i000 + 1
    i010 = xi * nynzg + (yi + 1) * nzg + zi
    i011 = i010 + 1
    i100 = (xi + 1) * nynzg + yi * nzg + zi
    i101 = i100 + 1
    i110 = (xi + 1) * nynzg + (yi + 1) * nzg + zi
    i111 = i110 + 1
    w = [omtx*omty*omtz, omtx*omty*tz, omtx*ty*omtz, omtx*ty*tz,
         tx*omty*omtz,   tx*omty*tz,   tx*ty*omtz,   tx*ty*tz]
    idx = [i000, i001, i010, i011, i100, i101, i110, i111]
    bx = sum(w[k] * float(Bxf[idx[k]]) for k in range(8))
    by = sum(w[k] * float(Byf[idx[k]]) for k in range(8))
    bz = sum(w[k] * float(Bzf[idx[k]]) for k in range(8))
    return bx, by, bz


# ---------------------------------------------------------------------------
# 2. Trace magnetic field lines (3D RK4)
# ---------------------------------------------------------------------------
b_axis   = bfield.on_axis()
idx_peak = int(np.argmax(b_axis))
z_seed   = float(bfield.z[idx_peak])


def trace_fieldline(x0: float, y0: float, z0: float,
                    ds: float = 0.008, nstep: int = 3000) -> np.ndarray:
    def _trace(sign: int) -> list:
        pts = []
        x, y, z = x0, y0, z0
        for _ in range(nstep):
            bx, by, bz = _interp_B_scalar(x, y, z)
            b = math.sqrt(bx*bx + by*by + bz*bz)
            if b < 1e-9:
                break
            x += sign * bx / b * ds
            y += sign * by / b * ds
            z += sign * bz / b * ds
            if (z < Z_MIN or z > Z_MAX
                    or abs(x) > R_MAX or abs(y) > R_MAX):
                pts.append((x, y, z))
                break
            pts.append((x, y, z))
        return pts

    fwd = _trace(+1)
    bwd = _trace(-1)
    return np.array(list(reversed(bwd)) + [(x0, y0, z0)] + fwd)


print(f"Tracing field lines …")
field_lines = []
radii  = np.linspace(0.06, 0.52, N_FIELD_SEEDS)
azimuths = np.linspace(0, 2 * math.pi, 6, endpoint=False)
for r in radii:
    for phi in azimuths:
        fl = trace_fieldline(r * math.cos(phi), r * math.sin(phi), z_seed)
        if len(fl) > 3:
            field_lines.append(fl)
print(f"  {len(field_lines)} field lines traced")


# ---------------------------------------------------------------------------
# 3. Run Boris simulation, recording trajectories
# ---------------------------------------------------------------------------
print(f"Running Boris pusher: {N_PARTICLES} ions × {N_STEPS} steps …")
rng = np.random.default_rng(42)

z_inject = float(np.clip(z_seed + 0.05 * (Z_MAX - z_seed), Z_MIN + 0.01, Z_MAX - 0.1))
r_inj    = min(0.3 * R_MAX, 0.15)

r_rnd   = r_inj * np.sqrt(rng.uniform(0, 1, N_PARTICLES))
phi_rnd = rng.uniform(0, 2 * math.pi, N_PARTICLES)

v_th       = math.sqrt(2.0 * T_I_EV * E_CHARGE / M_D)
v_th_small = v_th * 0.1

pos = np.vstack([
    r_rnd * np.cos(phi_rnd),
    r_rnd * np.sin(phi_rnd),
    np.full(N_PARTICLES, z_inject),
]).astype(np.float32)

vel = np.vstack([
    rng.normal(0.0, v_th_small, N_PARTICLES),
    rng.normal(0.0, v_th_small, N_PARTICLES),
    np.full(N_PARTICLES, V_INJ),
]).astype(np.float32)

t_transit = (Z_MAX - z_inject) / max(V_INJ, 1e3)
dt_f      = t_transit / max(N_STEPS, 1)
bpeak     = float(np.max(np.abs(b_axis)))
if bpeak > 1e-9:
    omega_c = (E_CHARGE / M_D) * bpeak
    dt_cyc  = 0.1 / omega_c
    if dt_cyc * N_STEPS >= t_transit:
        dt_f = dt_cyc
dt  = np.float32(dt_f)
hqm = np.float32((E_CHARGE / M_D) * float(dt) * 0.5)

n_frames_max = N_STEPS // RECORD_EVERY + 2
pos_hist  = np.full((n_frames_max, 3, N_PARTICLES), np.nan, dtype=np.float32)
active    = np.ones(N_PARTICLES, dtype=bool)
exited    = np.zeros(N_PARTICLES, dtype=bool)
frame_idx = 0

for step in range(N_STEPS):
    if step % RECORD_EVERY == 0 and frame_idx < n_frames_max:
        pos_hist[frame_idx] = pos
        frame_idx += 1
    if not np.any(active):
        break
    idx      = np.where(active)[0]
    p_new, v_new = _boris_step(pos[:, idx], vel[:, idx], gp, dt, hqm)
    pos[:, idx] = p_new
    vel[:, idx] = v_new
    out_z  = p_new[2] >= Z_MAX
    out_r  = p_new[0]**2 + p_new[1]**2 > (R_MAX * 1.1)**2
    out_zm = p_new[2] <= Z_MIN
    exited[idx[out_z]] = True
    active[idx[out_z]] = False
    active[idx[out_r]] = False
    active[idx[out_zm]] = False

pos_hist[frame_idx] = pos
N_FRAMES = frame_idx + 1
pos_hist = pos_hist[:N_FRAMES]

# Mask positions that have left the domain
for f in range(N_FRAMES):
    px, py, pz = pos_hist[f, 0], pos_hist[f, 1], pos_hist[f, 2]
    mask = (
        np.isnan(px)
        | (pz < Z_MIN) | (pz > Z_MAX * 1.05)
        | (px**2 + py**2 > (R_MAX * 1.2)**2)
    )
    pos_hist[f, :, mask] = np.nan

print(f"  Recorded {N_FRAMES} frames  ({exited.sum()} exited)")


# ---------------------------------------------------------------------------
# 4. Build animation
# ---------------------------------------------------------------------------
print("Building animation …")

fig = plt.figure(figsize=(12, 7), dpi=DPI)
fig.patch.set_facecolor("#080818")
ax  = fig.add_subplot(111, projection="3d")
ax.set_facecolor("#080818")
ax.grid(False)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#222")

ax.set_xlim(Z_MIN, Z_MAX)
ax.set_ylim(-R_MAX, R_MAX)
ax.set_zlim(-R_MAX, R_MAX)
ax.set_xlabel("z (m)", color="#aaa", fontsize=8, labelpad=8)
ax.set_ylabel("x (m)", color="#aaa", fontsize=8, labelpad=4)
ax.set_zlabel("y (m)", color="#aaa", fontsize=8, labelpad=4)
ax.tick_params(colors="#555", labelsize=6)

# Plot field lines (static)
for fl in field_lines:
    if len(fl) < 2:
        continue
    ax.plot(fl[:, 2], fl[:, 0], fl[:, 1],
            color="white", lw=0.35, alpha=0.22)

# Mark coil positions
for coil in COILS:
    phi_c = np.linspace(0, 2 * math.pi, 80)
    ax.plot(
        np.full(80, coil.z),
        coil.r * np.cos(phi_c),
        coil.r * np.sin(phi_c),
        color="#ff9900", lw=1.0, alpha=0.7,
    )

# Particle scatter — current position
scat = ax.scatter([], [], [], s=5, c="#00e5ff", alpha=0.85,
                  depthshade=False, zorder=5)

# Trail lines: one Line3D per particle (updated each frame)
trail_lines = []
for _ in range(N_PARTICLES):
    ln, = ax.plot([], [], [], lw=0.5, color="#00e5ff", alpha=0.0)
    trail_lines.append(ln)

title_text = ax.set_title(
    "Helicon  —  Sunbird 3D Magnetic Nozzle  |  frame 0",
    color="white", fontsize=10, pad=8,
)

# Camera rotation: azim sweeps slowly
AZ_START = 35.0
AZ_SWEEP = 120.0   # total degrees over full animation


def update(frame: int):
    f = min(frame, N_FRAMES - 1)

    # Camera
    az = AZ_START + AZ_SWEEP * f / max(N_FRAMES - 1, 1)
    ax.view_init(elev=22, azim=az)

    # Current positions (non-NaN particles)
    px = pos_hist[f, 0]
    py = pos_hist[f, 1]
    pz = pos_hist[f, 2]
    valid = ~np.isnan(px)

    if valid.any():
        scat._offsets3d = (pz[valid], px[valid], py[valid])
    else:
        scat._offsets3d = (np.array([]), np.array([]), np.array([]))

    # Trails
    t_start = max(0, f - TRAIL)
    for k in range(N_PARTICLES):
        trail_z = pos_hist[t_start:f+1, 2, k]
        trail_x = pos_hist[t_start:f+1, 0, k]
        trail_y = pos_hist[t_start:f+1, 1, k]
        ok = ~np.isnan(trail_z)
        if ok.any():
            trail_lines[k].set_data_3d(trail_z[ok], trail_x[ok], trail_y[ok])
            trail_lines[k].set_alpha(0.28 if ~np.isnan(pz[k]) else 0.0)
        else:
            trail_lines[k].set_alpha(0.0)

    title_text.set_text(
        f"Helicon  —  Sunbird 3D Nozzle  |  "
        f"step {f * RECORD_EVERY}/{N_STEPS}  "
        f"({valid.sum()} active, {exited.sum()} exited)"
    )
    return [scat, title_text] + trail_lines


ani = animation.FuncAnimation(
    fig, update, frames=N_FRAMES,
    interval=1000 / FPS, blit=False,
)

OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving → {OUT_VIDEO} …")
writer = animation.FFMpegWriter(
    fps=FPS, bitrate=3000, codec="h264",
    extra_args=["-pix_fmt", "yuv420p"],
)
ani.save(str(OUT_VIDEO), writer=writer, dpi=DPI)
size_kb = OUT_VIDEO.stat().st_size // 1024
print(f"Done.  {OUT_VIDEO}  ({size_kb} KB)")
plt.close(fig)
