"""
Plasma rocket exhaust — cinematic 3-D particle simulation.

A continuous particle system of 15,000 D⁺ plasma ions streams from the exit
plane of a bell nozzle.  Rendered entirely in NumPy with additive HDR blending,
multi-scale Gaussian glow, and filmic ACES tone-mapping.

Distinctive features:
  · Temperature colour gradient  white/blue core → cyan → violet → pink → red plume
  · Mach shock-diamond brightness modulation in the dense core beam
  · Multi-scale Gaussian glow (sharp core + soft halo + wide diffuse bloom)
  · Bell nozzle wireframe silhouette (longitude lines + ring hoops)
  · 400-star deep-space background that drifts as the camera orbits
  · Orbiting camera: 260 ° sweep at 18 ° elevation over 24 seconds
  · ACES filmic tone-map + γ-2.2 encode

Usage:
    uv run python scripts/make_exhaust_video.py
"""

from __future__ import annotations

import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation       # noqa: E402
import matplotlib.pyplot as plt               # noqa: E402
import numpy as np                            # noqa: E402
from scipy.ndimage import gaussian_filter     # noqa: E402
from tqdm import tqdm                         # noqa: E402

# ── Output ─────────────────────────────────────────────────────────────────────
OUT_VIDEO  = pathlib.Path("results/exhaust_animation.mp4")
W, H = 1280, 720
FPS       = 30
N_FRAMES  = 720          # 24 seconds

# ── Nozzle geometry ────────────────────────────────────────────────────────────
R_THROAT  = 0.042        # m
R_EXIT    = 0.180        # m  (expansion ratio ≈ 18)
R_CHAMBER = 0.110        # m
Z_CHAMBER = -0.38        # m  front of combustion chamber
Z_THROAT  = 0.00         # m  throat plane
Z_EXIT    = 0.42         # m  nozzle exit plane

# ── Particle system ────────────────────────────────────────────────────────────
N_PART = 15_000

# Three population layers: core / mid / outer
# All stored in a single flat array; layer boundaries
L0, L1 = 5_000, 11_000  # core [0:L0], mid [L0:L1], outer [L1:]
N_CORE, N_MID, N_OUTER = L0, L1 - L0, N_PART - L1

# Physical / visual time scale
DT = 1.0 / FPS

# ── Shock-diamond parameters ───────────────────────────────────────────────────
DIAMOND_Z   = np.array([0.13, 0.38, 0.63, 0.88, 1.13, 1.38])
DIAMOND_SIG = 0.060
DIAMOND_AMP = 3.5          # peak brightness multiplier at node centres

# ── Camera orbit ───────────────────────────────────────────────────────────────
CAM_RADIUS  = 1.50         # m from axis
CAM_ELEV    = math.radians(18.0)
CAM_TARGET  = np.array([0.0, 0.0, 0.45])   # look-at point
AZ_START    = math.radians(30.0)
AZ_SWEEP    = math.radians(260.0)
FOV         = math.radians(68.0)


# ── Temperature → RGB colour LUT ───────────────────────────────────────────────

def _make_lut(n: int = 512) -> np.ndarray:
    """
    Helicon-plasma palette:
      cold  →  dark crimson  →  red  →  pink  →  violet  →  blue  →  cyan  →  white-hot
    """
    lut = np.zeros((n, 3), dtype=np.float32)
    t   = np.linspace(0.0, 1.0, n)

    segs = [
        (0.00, 0.07,  (0.00, 0.00, 0.00),  (0.22, 0.00, 0.04)),
        (0.07, 0.18,  (0.22, 0.00, 0.04),  (0.80, 0.05, 0.05)),
        (0.18, 0.30,  (0.80, 0.05, 0.05),  (0.95, 0.20, 0.30)),
        (0.30, 0.43,  (0.95, 0.20, 0.30),  (0.85, 0.00, 0.85)),
        (0.43, 0.56,  (0.85, 0.00, 0.85),  (0.35, 0.00, 1.00)),
        (0.56, 0.68,  (0.35, 0.00, 1.00),  (0.00, 0.55, 1.00)),
        (0.68, 0.80,  (0.00, 0.55, 1.00),  (0.00, 1.00, 0.95)),
        (0.80, 0.91,  (0.00, 1.00, 0.95),  (0.65, 1.00, 1.00)),
        (0.91, 1.00,  (0.65, 1.00, 1.00),  (1.00, 1.00, 1.00)),
    ]
    for t0, t1, c0, c1 in segs:
        m = (t >= t0) & (t < t1)
        u = (t[m] - t0) / (t1 - t0)
        for ch in range(3):
            lut[m, ch] = c0[ch] + (c1[ch] - c0[ch]) * u

    lut[-1] = 1.0
    return np.clip(lut, 0.0, 1.0)


LUT = _make_lut()

def temp_rgb(tmp: np.ndarray) -> np.ndarray:
    idx = np.clip((tmp * (len(LUT) - 1)).astype(np.int32), 0, len(LUT) - 1)
    return LUT[idx]          # (N, 3) float32


# ── Nozzle profile ─────────────────────────────────────────────────────────────

def nozzle_r(z: np.ndarray) -> np.ndarray:
    r = np.empty_like(z, dtype=np.float64)
    ch   = z <= Z_CHAMBER
    conv = (z > Z_CHAMBER) & (z <= Z_THROAT)
    div  = z > Z_THROAT

    r[ch] = R_CHAMBER

    u = (z[conv] - Z_CHAMBER) / (Z_THROAT - Z_CHAMBER)
    r[conv] = R_CHAMBER + (R_THROAT - R_CHAMBER) * u ** 0.55

    w = np.clip(z[div] / Z_EXIT, 0.0, None)
    r[div] = R_THROAT + (R_EXIT - R_THROAT) * (1.0 - np.exp(-3.8 * w))

    return r


# ── Camera helpers ─────────────────────────────────────────────────────────────

def _camera_basis(az: float):
    """Returns cam_pos, fwd, right, up for the given azimuth angle."""
    cx = CAM_RADIUS * math.cos(CAM_ELEV) * math.cos(az)
    cy = CAM_RADIUS * math.cos(CAM_ELEV) * math.sin(az)
    cz = CAM_RADIUS * math.sin(CAM_ELEV) + CAM_TARGET[2]
    cam_pos = np.array([cx, cy, cz], dtype=np.float64)

    fwd = CAM_TARGET - cam_pos
    fwd /= np.linalg.norm(fwd)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return cam_pos, fwd, right, up


def project(pts: np.ndarray, cam_pos, fwd, right, up) -> tuple[np.ndarray, np.ndarray]:
    """
    Perspective-project world points (N,3) → screen pixels (N,2) + depth (N,).
    Returns (screen_xy float32, depth float32).
    """
    dv    = pts - cam_pos                # (N, 3)
    depth = dv @ fwd                     # (N,)
    x_cam = dv @ right
    y_cam = dv @ up

    aspect = W / H
    f      = 1.0 / math.tan(FOV / 2.0)
    x_ndc  = f * x_cam / (depth * aspect)
    y_ndc  = f * y_cam / depth

    sx = (x_ndc + 1.0) * (W / 2.0)
    sy = (1.0 - y_ndc) * (H / 2.0)
    return np.stack([sx, sy], axis=-1).astype(np.float32), depth.astype(np.float32)


# ── Starfield ──────────────────────────────────────────────────────────────────

_rng_s = np.random.default_rng(7777)
N_STARS = 420
_sdir   = _rng_s.standard_normal((N_STARS, 3)).astype(np.float64)
_sdir  /= np.linalg.norm(_sdir, axis=1, keepdims=True)
_sbright = _rng_s.power(0.4, N_STARS).astype(np.float32) * 0.90 + 0.10
_scol    = np.ones((N_STARS, 3), dtype=np.float32)
_b  = _rng_s.random(N_STARS) < 0.25
_y  = (~_b) & (_rng_s.random(N_STARS) < 0.20)
_scol[_b]  = [0.75, 0.85, 1.00]
_scol[_y]  = [1.00, 0.92, 0.65]

# Project stars at "infinity": they are very far away (cam_pos + 1000 * dir)
STAR_DIST = 1000.0


# ── Particle arrays ────────────────────────────────────────────────────────────

_rng = np.random.default_rng(42)

pos  = np.zeros((N_PART, 3), dtype=np.float32)
vel  = np.zeros((N_PART, 3), dtype=np.float32)
tmp  = np.zeros(N_PART, dtype=np.float32)   # temperature 0–1
age  = np.zeros(N_PART, dtype=np.float32)   # normalised age  0=new  1=dead


def _inject(idx: np.ndarray) -> None:
    """Reinitialise particles in `idx` as freshly injected at the nozzle exit."""
    N = len(idx)
    if N == 0:
        return

    is_core  = idx < L0
    is_mid   = (idx >= L0) & (idx < L1)
    is_outer = idx >= L1

    # Radial extent and injection temperature per layer
    r_max  = np.where(is_core, R_EXIT * 0.38,
             np.where(is_mid,  R_EXIT * 0.72, R_EXIT * 1.05))
    t0     = np.where(is_core, 0.93,
             np.where(is_mid,  0.62, 0.32)).astype(np.float32)
    v0     = np.where(is_core, 2.80,
             np.where(is_mid,  2.10, 1.60)).astype(np.float32)
    ang    = np.where(is_core, math.radians(4),
             np.where(is_mid,  math.radians(11), math.radians(20))).astype(np.float64)
    turb   = np.where(is_core, 0.04,
             np.where(is_mid,  0.10, 0.18)).astype(np.float32)
    life   = np.where(is_core, 1.30,
             np.where(is_mid,  1.00, 0.80)).astype(np.float32)

    r_inj  = r_max * np.sqrt(_rng.uniform(0.0, 1.0, N).astype(np.float32))
    phi    = _rng.uniform(0.0, 2.0 * math.pi, N).astype(np.float32)

    pos[idx, 0] = r_inj * np.cos(phi)
    pos[idx, 1] = r_inj * np.sin(phi)
    pos[idx, 2] = Z_EXIT + _rng.normal(0, 0.008, N).astype(np.float32)

    theta  = ang * (r_inj / np.maximum(r_max, 1e-6))
    v_z    = v0 * np.cos(theta).astype(np.float32)
    v_r    = v0 * np.sin(theta).astype(np.float32)
    tv     = (turb * v0 * _rng.standard_normal(N)).astype(np.float32)

    vel[idx, 0] = v_r * np.cos(phi) + tv * _rng.standard_normal(N).astype(np.float32)
    vel[idx, 1] = v_r * np.sin(phi) + tv * _rng.standard_normal(N).astype(np.float32)
    vel[idx, 2] = v_z + 0.3 * tv

    tmp[idx]  = np.clip(t0 + _rng.normal(0, 0.04, N).astype(np.float32), 0.0, 1.0)
    age[idx]  = _rng.uniform(0.0, 1.0, N).astype(np.float32) * life / life   # 0–1 random
    # Store effective lifetime in the age by using fractional age
    age[idx]  = _rng.uniform(0.0, 1.0, N).astype(np.float32)


# Initialise
_inject(np.arange(N_PART))


def _step_particles() -> None:
    """One time step: physics + recycling."""
    # Turbulence scaled by layer
    turb_amp = np.where(np.arange(N_PART) < L0, 0.04,
               np.where(np.arange(N_PART) < L1, 0.10, 0.18)).astype(np.float32)
    v_amp = np.where(np.arange(N_PART) < L0, 2.80,
            np.where(np.arange(N_PART) < L1, 2.10, 1.60)).astype(np.float32)

    kick = (turb_amp * v_amp)[:, None] * _rng.standard_normal((N_PART, 3)).astype(np.float32)
    kick[:, 2] *= 0.25   # less axial turbulence
    vel[:] += kick * DT * 1.8

    pos[:] += vel * DT

    # Temperature cools with expansion / age
    tmp[:] = np.clip(tmp * 0.965, 0.0, 1.0)

    # Age: different lifetimes per layer
    life = np.where(np.arange(N_PART) < L0, 1.30,
           np.where(np.arange(N_PART) < L1, 1.00, 0.80)).astype(np.float32)
    age[:] += DT / life

    dead = np.where(age >= 1.0)[0]
    _inject(dead)


# ── HDR software renderer ──────────────────────────────────────────────────────

def _aces(x: np.ndarray) -> np.ndarray:
    """ACES filmic tone-map (approximate)."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def _shock_factor(z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Brightness multiplier: Mach-diamond pattern in the dense core."""
    plume_w = R_EXIT + np.maximum(z - Z_EXIT, 0.0) * math.tan(math.radians(12))
    in_core = r < 0.55 * plume_w
    sf = np.ones(len(z), dtype=np.float32)
    for zd in DIAMOND_Z:
        contrib = DIAMOND_AMP * np.exp(-0.5 * ((z - zd) / DIAMOND_SIG) ** 2)
        sf = np.where(in_core, sf + contrib.astype(np.float32), sf)
    return sf


def render_plasma(cam_pos, fwd, right, up) -> np.ndarray:
    """
    Return (H, W, 3) uint8 frame:
      1. Black background + stars
      2. Plasma particle accumulation (additive HDR)
      3. Multi-scale Gaussian glow
      4. ACES tone-map + γ-2.2
    """
    # ── Stars ──────────────────────────────────────────────────────────────
    star_world = cam_pos + STAR_DIST * _sdir          # (N_STARS, 3)
    sxy, sdep = project(star_world, cam_pos, fwd, right, up)

    sr_buf = np.zeros((H, W), dtype=np.float32)
    sg_buf = np.zeros((H, W), dtype=np.float32)
    sb_buf = np.zeros((H, W), dtype=np.float32)

    sv = sdep > 0
    six = np.clip(sxy[sv, 0].astype(np.int32), 0, W - 1)
    siy = np.clip(sxy[sv, 1].astype(np.int32), 0, H - 1)
    sb  = _sbright[sv]
    sc  = _scol[sv]
    np.add.at(sr_buf, (siy, six), sc[:, 0] * sb * 0.55)
    np.add.at(sg_buf, (siy, six), sc[:, 1] * sb * 0.55)
    np.add.at(sb_buf, (siy, six), sc[:, 2] * sb * 0.55)

    # Blur stars slightly for a soft bokeh glow
    sr_buf = gaussian_filter(sr_buf, sigma=0.8)
    sg_buf = gaussian_filter(sg_buf, sigma=0.8)
    sb_buf = gaussian_filter(sb_buf, sigma=0.8)

    # ── Particles ──────────────────────────────────────────────────────────
    pxy, dep = project(pos, cam_pos, fwd, right, up)

    # Visibility: in front of camera and within screen bounds
    vis = (
        (dep > 0.05)
        & (pxy[:, 0] >= 0) & (pxy[:, 0] < W)
        & (pxy[:, 1] >= 0) & (pxy[:, 1] < H)
    )

    ix = pxy[vis, 0].astype(np.int32)
    iy = pxy[vis, 1].astype(np.int32)

    # Temperature → colour
    rgb = temp_rgb(tmp[vis])               # (M, 3) float32

    # Particle weight: brightness * fade * depth-based spread
    r_xy = np.sqrt(pos[vis, 0] ** 2 + pos[vis, 1] ** 2)
    sf   = _shock_factor(pos[vis, 2], r_xy)
    fade = np.clip(1.0 - age[vis], 0.0, 1.0) ** 0.6  # slower fade
    w    = (fade * sf * rgb[:, 0].clip(0.05) / np.maximum(dep[vis], 0.1) * 0.12).astype(np.float32)

    pr = rgb[:, 0] * w
    pg = rgb[:, 1] * w
    pb = rgb[:, 2] * w

    np.add.at(sr_buf, (iy, ix), pr)
    np.add.at(sg_buf, (iy, ix), pg)
    np.add.at(sb_buf, (iy, ix), pb)

    # ── Multi-scale Gaussian glow ──────────────────────────────────────────
    # Three layers: sharp core, inner halo, wide diffuse bloom
    def _glow(buf):
        sharp  = gaussian_filter(buf, sigma=1.2)
        halo   = gaussian_filter(buf, sigma=4.5)
        bloom  = gaussian_filter(buf, sigma=14.0)
        return sharp * 0.55 + halo * 0.28 + bloom * 0.17

    fr = _glow(sr_buf)
    fg = _glow(sg_buf)
    fb = _glow(sb_buf)

    # ── Throat inner-glow (very hot plasma source) ─────────────────────────
    throat_world = np.array([[0.0, 0.0, Z_THROAT]])
    txy, tdep = project(throat_world, cam_pos, fwd, right, up)
    if tdep[0] > 0.05:
        tx = int(np.clip(txy[0, 0], 0, W - 1))
        ty = int(np.clip(txy[0, 1], 0, H - 1))
        # Additive point: will be bloomed by blur
        fr[ty, tx] += 8.0
        fg[ty, tx] += 8.0
        fb[ty, tx] += 9.0

    # ── Tone-map + γ-2.2 ──────────────────────────────────────────────────
    img = np.stack([fr, fg, fb], axis=-1)
    img = _aces(img)
    img = np.clip(img ** (1.0 / 2.2), 0.0, 1.0)
    return (img * 255).astype(np.uint8)


# ── Nozzle wireframe helpers ───────────────────────────────────────────────────

_nozzle_z_long  = np.linspace(Z_CHAMBER, Z_EXIT, 60)
_nozzle_r_long  = nozzle_r(_nozzle_z_long)
_nozzle_phi_long = np.linspace(0, 2 * math.pi, 12, endpoint=False)

_nozzle_z_ring  = np.array([Z_CHAMBER, -0.22, -0.10, Z_THROAT, 0.10, 0.22, 0.34, Z_EXIT])
_nozzle_phi_ring = np.linspace(0, 2 * math.pi, 48)

# Pre-compute 3D nozzle wire-points
_nozzle_lines_3d: list[np.ndarray] = []

# Longitude lines
for phi in _nozzle_phi_long:
    pts = np.stack([
        _nozzle_r_long * math.cos(phi),
        _nozzle_r_long * math.sin(phi),
        _nozzle_z_long,
    ], axis=-1)
    _nozzle_lines_3d.append(pts)

# Rings
for zr in _nozzle_z_ring:
    r = float(nozzle_r(np.array([zr]))[0])
    pts = np.stack([
        r * np.cos(_nozzle_phi_ring),
        r * np.sin(_nozzle_phi_ring),
        np.full(len(_nozzle_phi_ring), zr),
    ], axis=-1)
    _nozzle_lines_3d.append(pts)


def project_nozzle(cam_pos, fwd, right, up) -> list[tuple[np.ndarray, np.ndarray]]:
    """Project each nozzle wire segment → list of (sx, sy) screen-coord arrays."""
    result = []
    for pts in _nozzle_lines_3d:
        sxy, dep = project(pts, cam_pos, fwd, right, up)
        vis = dep > 0.0
        if vis.any():
            result.append((sxy[vis, 0], sxy[vis, 1]))
        else:
            result.append((np.array([]), np.array([])))
    return result


# ── Figure & artists ───────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.set_xlim(0, W)
ax.set_ylim(H, 0)     # y increases downward (image convention)
ax.axis("off")

_blank = np.zeros((H, W, 3), dtype=np.uint8)
im_artist = ax.imshow(_blank, extent=[0, W, H, 0], aspect="auto", interpolation="nearest")

N_WIRE = len(_nozzle_lines_3d)
_nozzle_artists = [
    ax.plot([], [], color="#606060", lw=0.55, alpha=0.70, solid_capstyle="round")[0]
    for _ in range(N_WIRE)
]

# Exit-plane glow ring
_exit_ring_phi = np.linspace(0, 2 * math.pi, 120)
_exit_ring_pts = np.stack([
    R_EXIT * np.cos(_exit_ring_phi),
    R_EXIT * np.sin(_exit_ring_phi),
    np.full(120, Z_EXIT),
], axis=-1)
_exit_ring_art, = ax.plot([], [], color="#ff8833", lw=1.8, alpha=0.85)

title_art = ax.text(
    18, 34, "",
    color="#aaaaaa", fontsize=8.5, ha="left", va="top",
    fontfamily="monospace",
)


# ── Animation ──────────────────────────────────────────────────────────────────

def update(frame: int):
    # ── Camera ──────────────────────────────────────────────────────────────
    az = AZ_START + AZ_SWEEP * frame / max(N_FRAMES - 1, 1)
    cam_pos, fwd, right, up = _camera_basis(float(az))

    # ── Physics ──────────────────────────────────────────────────────────────
    _step_particles()

    # ── Plasma render ────────────────────────────────────────────────────────
    plasma_img = render_plasma(cam_pos, fwd, right, up)
    im_artist.set_data(plasma_img)

    # ── Nozzle wireframe ─────────────────────────────────────────────────────
    wire = project_nozzle(cam_pos, fwd, right, up)
    for k, (sx, sy) in enumerate(wire):
        _nozzle_artists[k].set_data(sx, sy)

    # Exit-plane glow ring
    er_sxy, er_dep = project(_exit_ring_pts, cam_pos, fwd, right, up)
    ev = er_dep > 0
    _exit_ring_art.set_data(er_sxy[ev, 0], er_sxy[ev, 1])

    # ── HUD ──────────────────────────────────────────────────────────────────
    active = int((age < 1.0).sum())
    az_deg = math.degrees(float(az)) % 360
    title_art.set_text(
        f"Helicon  ·  Plasma Rocket Exhaust"
        f"   frame {frame:4d}/{N_FRAMES}"
        f"   az {az_deg:5.1f}°"
        f"   {active:,} active ions"
    )

    return [im_artist, _exit_ring_art, title_art] + _nozzle_artists


ani = animation.FuncAnimation(
    fig, update,
    frames=N_FRAMES,
    blit=True,
    interval=1000 / FPS,
)

# ── Save ────────────────────────────────────────────────────────────────────────
OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
print(f"Rendering {N_FRAMES} frames @ {FPS} FPS  ({N_FRAMES / FPS:.0f} s)  →  {OUT_VIDEO}")

writer = animation.FFMpegWriter(
    fps=FPS, bitrate=6000, codec="h264",
    extra_args=["-pix_fmt", "yuv420p", "-crf", "16"],
)

pbar = tqdm(total=N_FRAMES, desc="Rendering", unit="frame")
_orig_grab = writer.grab_frame

def _grab(*a, **kw):
    _orig_grab(*a, **kw)
    pbar.update(1)

writer.grab_frame = _grab
ani.save(str(OUT_VIDEO), writer=writer, dpi=100)
pbar.close()

sz_mb = OUT_VIDEO.stat().st_size / 1024 ** 2
print(f"Done.  {OUT_VIDEO}  ({sz_mb:.1f} MB)")
plt.close(fig)
