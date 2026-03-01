"""Tests for magnozzlex.fields.biot_savart — 9 test cases.

MLX tests are skipped when MLX is not installed.
"""

from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest

from magnozzlex.fields.biot_savart import (
    HAS_MLX,
    MU_0,
    BField,
    Coil,
    Grid,
    compute_bfield,
)

if HAS_MLX:
    import mlx.core as mx

    from magnozzlex.fields.biot_savart import compute_bfield_mlx_differentiable

skip_no_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SINGLE_COIL = Coil(z=0.0, r=0.1, I=1000.0)


def on_axis_Bz_exact(coil: Coil, z: float) -> float:
    """Exact on-axis B_z for a circular loop."""
    a = coil.r
    dz = z - coil.z
    return MU_0 * coil.I * a**2 / (2.0 * (a**2 + dz**2) ** 1.5)


# ---------------------------------------------------------------------------
# Test 1: On-axis B_z vs exact formula (single coil)
# ---------------------------------------------------------------------------
class TestOnAxisBz:
    """On-axis B_z must match the exact analytic formula."""

    def _check(self, backend: str, tol: float):
        grid = Grid(z_min=-0.5, z_max=0.5, r_max=0.3, nz=201, nr=4)
        bf = compute_bfield([SINGLE_COIL], grid, backend=backend, n_phi=512)

        # r=0 is index 0
        Bz_axis = bf.Bz[0, :]
        z_pts = bf.z

        Bz_exact = np.array([on_axis_Bz_exact(SINGLE_COIL, zi) for zi in z_pts])
        rel_err = np.abs((Bz_axis - Bz_exact) / Bz_exact)
        assert np.all(rel_err < tol), f"Max rel err {rel_err.max():.6e} > {tol}"

    def test_numpy(self):
        self._check("numpy", tol=1e-4)  # < 0.01%

    @skip_no_mlx
    def test_mlx(self):
        self._check("mlx", tol=5e-3)  # < 0.5%


# ---------------------------------------------------------------------------
# Test 2: MLX vs NumPy backend consistency (off-axis grid)
# ---------------------------------------------------------------------------
@skip_no_mlx
def test_mlx_numpy_consistency():
    grid = Grid(z_min=-0.3, z_max=0.3, r_max=0.2, nz=32, nr=16)
    coils = [SINGLE_COIL]
    bf_np = compute_bfield(coils, grid, backend="numpy")
    bf_mlx = compute_bfield(coils, grid, backend="mlx", n_phi=512)

    # Exclude r=0 row for Br (both should be ~0 there, ratio is ill-defined)
    # Compare Bz everywhere
    mask = np.abs(bf_np.Bz) > 1e-12
    rel_bz = np.abs((bf_mlx.Bz[mask] - bf_np.Bz[mask]) / bf_np.Bz[mask])
    assert np.all(rel_bz < 0.01), f"Bz max rel err {rel_bz.max():.4e}"

    # Compare Br off-axis
    mask_br = (np.abs(bf_np.Br) > 1e-12) & (np.arange(bf_np.Br.shape[0])[:, None] > 0)
    if np.any(mask_br):
        rel_br = np.abs((bf_mlx.Br[mask_br] - bf_np.Br[mask_br]) / bf_np.Br[mask_br])
        assert np.all(rel_br < 0.01), f"Br max rel err {rel_br.max():.4e}"


# ---------------------------------------------------------------------------
# Test 3: Dipole limit at far field (R >> a)
# ---------------------------------------------------------------------------
def test_dipole_far_field():
    """At large distance, B_z ~ μ₀ m / (2π z³) on axis, where m = I π a²."""
    coil = Coil(z=0.0, r=0.05, I=500.0)
    m = coil.I * math.pi * coil.r**2  # magnetic moment

    grid = Grid(z_min=2.0, z_max=5.0, r_max=0.01, nz=50, nr=2)
    bf = compute_bfield([coil], grid, backend="numpy")

    Bz_axis = bf.Bz[0, :]
    z_pts = bf.z
    Bz_dipole = MU_0 * m / (2.0 * math.pi * z_pts**3)

    rel_err = np.abs((Bz_axis - Bz_dipole) / Bz_dipole)
    assert np.all(rel_err < 0.05), f"Far-field dipole err {rel_err.max():.4e}"


# ---------------------------------------------------------------------------
# Test 4: Superposition — B(coil1+coil2) == B(coil1) + B(coil2)
# ---------------------------------------------------------------------------
def test_superposition():
    coil1 = Coil(z=0.0, r=0.1, I=1000.0)
    coil2 = Coil(z=0.3, r=0.15, I=-500.0)
    grid = Grid(z_min=-0.5, z_max=0.8, r_max=0.3, nz=40, nr=20)

    bf_both = compute_bfield([coil1, coil2], grid, backend="numpy")
    bf_1 = compute_bfield([coil1], grid, backend="numpy")
    bf_2 = compute_bfield([coil2], grid, backend="numpy")

    np.testing.assert_allclose(bf_both.Bz, bf_1.Bz + bf_2.Bz, atol=1e-15)
    np.testing.assert_allclose(bf_both.Br, bf_1.Br + bf_2.Br, atol=1e-15)


# ---------------------------------------------------------------------------
# Test 5: B_r == 0 on axis (both backends)
# ---------------------------------------------------------------------------
class TestBrOnAxis:
    def _check(self, backend: str):
        grid = Grid(z_min=-0.5, z_max=0.5, r_max=0.3, nz=50, nr=8)
        bf = compute_bfield([SINGLE_COIL], grid, backend=backend, n_phi=256)
        Br_axis = bf.Br[0, :]
        assert np.all(np.abs(Br_axis) < 1e-10), f"Br on axis max = {np.abs(Br_axis).max():.2e}"

    def test_numpy(self):
        self._check("numpy")

    @skip_no_mlx
    def test_mlx(self):
        self._check("mlx")


# ---------------------------------------------------------------------------
# Test 6: Helmholtz coil — B_z at centre vs exact
# ---------------------------------------------------------------------------
def test_helmholtz_center():
    """Two coils separated by their radius: B_z at midpoint has a known exact value.

    B_center = (4/5)^(3/2) * μ₀ n I / R  for a single-turn Helmholtz pair.
    """
    R = 0.2  # coil radius
    I = 1000.0
    coils = [Coil(z=-R / 2, r=R, I=I), Coil(z=R / 2, r=R, I=I)]

    # Exact Helmholtz field at centre
    Bz_exact = (4.0 / 5.0) ** 1.5 * MU_0 * I / R

    grid = Grid(z_min=-0.01, z_max=0.01, r_max=0.01, nz=3, nr=2)
    bf = compute_bfield(coils, grid, backend="numpy")

    Bz_center = bf.Bz[0, 1]  # r=0, z≈0 (middle point)
    rel_err = abs((Bz_center - Bz_exact) / Bz_exact)
    assert rel_err < 1e-4, f"Helmholtz centre err {rel_err:.6e}"


# ---------------------------------------------------------------------------
# Test 7: MLX differentiability — mx.grad vs finite difference
# ---------------------------------------------------------------------------
@skip_no_mlx
def test_mlx_grad_vs_finite_diff():
    """Verify mx.grad of B_z at the origin w.r.t. coil current matches
    a finite-difference estimate."""

    grid_r = mx.array([0.0])
    grid_z = mx.array([0.0])

    def objective(params):
        _, Bz = compute_bfield_mlx_differentiable(params, grid_r, grid_z, n_phi=256)
        return Bz[0]

    base_params = mx.array([[0.0, 0.1, 1000.0]])
    grad_fn = mx.grad(objective)
    g = grad_fn(base_params)
    mx.eval(g)
    analytic_grad_I = np.array(g)[0, 2]  # dBz/dI

    # Finite difference
    eps = 1.0  # 1 A perturbation
    params_plus = mx.array([[0.0, 0.1, 1000.0 + eps]])
    params_minus = mx.array([[0.0, 0.1, 1000.0 - eps]])
    Bz_plus = objective(params_plus)
    Bz_minus = objective(params_minus)
    mx.eval(Bz_plus, Bz_minus)
    fd_grad = (np.array(Bz_plus) - np.array(Bz_minus)) / (2.0 * eps)

    rel_err = abs((analytic_grad_I - fd_grad) / fd_grad)
    assert rel_err < 1e-3, f"Grad vs FD rel err {rel_err:.6e}"


# ---------------------------------------------------------------------------
# Test 8: N_phi convergence — error decreases monotonically
# ---------------------------------------------------------------------------
@skip_no_mlx
def test_nphi_convergence():
    """Error of MLX vs NumPy should decrease with increasing n_phi.

    At high n_phi, float32 precision of MLX limits convergence, so we check
    that the coarsest → finest error decreases overall and that the first
    doubling (where quadrature error dominates) is monotonic.
    """
    grid = Grid(z_min=-0.2, z_max=0.2, r_max=0.15, nz=16, nr=8)
    bf_ref = compute_bfield([SINGLE_COIL], grid, backend="numpy")

    errors = []
    for n_phi in [16, 32, 64, 128]:
        bf = compute_bfield([SINGLE_COIL], grid, backend="mlx", n_phi=n_phi)
        mask = np.abs(bf_ref.Bz) > 1e-10
        err = np.max(np.abs((bf.Bz[mask] - bf_ref.Bz[mask]) / bf_ref.Bz[mask]))
        errors.append(err)

    # First two doublings must be monotonically decreasing
    for i in range(2):
        assert errors[i + 1] < errors[i], (
            f"Non-monotonic at step {i}: {errors[i]:.4e} -> {errors[i + 1]:.4e}"
        )

    # Overall: finest must be better than coarsest
    assert errors[-1] < errors[0], (
        f"No overall convergence: {errors[0]:.4e} -> {errors[-1]:.4e}"
    )


# ---------------------------------------------------------------------------
# Test 9: Singularity at coil location — no NaN/Inf
# ---------------------------------------------------------------------------
class TestSingularity:
    def _check(self, backend: str):
        coil = Coil(z=0.0, r=0.1, I=1000.0)
        # Grid includes the exact coil location (r=0.1, z=0.0)
        grid = Grid(z_min=-0.01, z_max=0.01, r_max=0.11, nz=3, nr=12)
        bf = compute_bfield([coil], grid, backend=backend, n_phi=128)
        assert np.all(np.isfinite(bf.Br)), "Br contains NaN/Inf at singularity"
        assert np.all(np.isfinite(bf.Bz)), "Bz contains NaN/Inf at singularity"

    def test_numpy(self):
        self._check("numpy")

    @skip_no_mlx
    def test_mlx(self):
        self._check("mlx")


# ---------------------------------------------------------------------------
# Bonus: HDF5 round-trip
# ---------------------------------------------------------------------------
def test_hdf5_roundtrip():
    """Save and reload a BField, verify arrays match."""
    pytest.importorskip("h5py")

    grid = Grid(z_min=-0.3, z_max=0.3, r_max=0.2, nz=16, nr=8)
    bf = compute_bfield([SINGLE_COIL], grid, backend="numpy")

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        bf.save(tmp.name)
        bf2 = BField.load(tmp.name)

    np.testing.assert_array_equal(bf.Br, bf2.Br)
    np.testing.assert_array_equal(bf.Bz, bf2.Bz)
    np.testing.assert_array_equal(bf.r, bf2.r)
    np.testing.assert_array_equal(bf.z, bf2.z)
    assert bf.backend == bf2.backend
    assert len(bf.coils) == len(bf2.coils)
    for c1, c2 in zip(bf.coils, bf2.coils, strict=True):
        assert c1.z == c2.z and c1.r == c2.r and c1.I == c2.I
