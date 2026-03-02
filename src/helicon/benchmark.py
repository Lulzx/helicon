"""Performance benchmark suite comparing NumPy vs MLX backends.

Usage::

    python -m helicon.benchmark
    helicon benchmark
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from helicon._mlx_utils import HAS_MLX


@dataclass
class BenchmarkResult:
    """Timing result for one benchmark case."""

    name: str
    numpy_ms: float
    mlx_ms: float | None  # None if MLX not available
    speedup: float | None  # numpy_ms / mlx_ms

    def __str__(self) -> str:
        if self.mlx_ms is None:
            return f"{self.name:<45}  numpy={self.numpy_ms:8.2f}ms  MLX=N/A"
        speedup = self.speedup or 0.0
        marker = " **" if speedup > 1.5 else ""
        return (
            f"{self.name:<45}  numpy={self.numpy_ms:8.2f}ms"
            f"  mlx={self.mlx_ms:8.2f}ms  {speedup:5.2f}×{marker}"
        )


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    results: list[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def summary(self) -> str:
        lines = [
            "=" * 80,
            "Helicon Benchmark Suite",
            f"MLX available: {HAS_MLX}",
            "=" * 80,
        ]
        for r in self.results:
            lines.append(str(r))
        lines.append("=" * 80)
        return "\n".join(lines)


def _time_fn(fn, n_warmup: int = 1, n_repeat: int = 3) -> float:
    """Return median wall time in milliseconds."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def _time_mlx_fn(fn, n_warmup: int = 1, n_repeat: int = 3) -> float | None:
    """Time an MLX function including mx.eval for synchronisation."""
    if not HAS_MLX:
        return None
    import mlx.core as mx

    def wrapped():
        result = fn()
        if isinstance(result, tuple):
            mx.eval(*result)
        else:
            mx.eval(result)

    for _ in range(n_warmup):
        wrapped()
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        wrapped()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------


def bench_biot_savart(suite: BenchmarkSuite) -> None:
    """Benchmark Biot-Savart at multiple grid sizes."""
    from helicon.fields.biot_savart import Coil, Grid, _compute_numpy

    coils = [
        Coil(z=0.0, r=0.1, I=50_000.0),
        Coil(z=0.3, r=0.05, I=20_000.0),
    ]
    sizes = [(64, 32), (256, 128), (512, 256)]

    for nz, nr in sizes:
        _grid = Grid(z_min=-0.5, z_max=1.5, r_max=0.3, nz=nz, nr=nr)

        def _make_np_fn(g):
            def np_fn():
                return _compute_numpy(coils, g)

            return np_fn

        t_np = _time_fn(_make_np_fn(_grid))

        if HAS_MLX:
            from helicon.fields.biot_savart import _compute_mlx

            def _make_mlx_fn(g):
                def mlx_fn():
                    Br, Bz, _r, _z = _compute_mlx(coils, g, n_phi=64)
                    return Br, Bz

                return mlx_fn

            t_mlx = _time_mlx_fn(_make_mlx_fn(_grid))
        else:
            t_mlx = None

        speedup = t_np / t_mlx if t_mlx and t_mlx > 0 else None
        suite.add(BenchmarkResult(f"Biot-Savart {nz}×{nr}", t_np, t_mlx, speedup))


def bench_thrust_integration(suite: BenchmarkSuite) -> None:
    """Benchmark thrust momentum-flux reduction on 1M synthetic particles."""
    rng = np.random.default_rng(42)
    N = 1_000_000
    wt = rng.uniform(0.5, 2.0, N).astype(np.float32)
    vz = rng.normal(40_000.0, 5_000.0, N).astype(np.float32)
    mass = 3.34e-27

    def np_fn():
        return float(np.sum(wt * mass * vz**2)), float(np.sum(wt * mass * np.abs(vz)))

    t_np = _time_fn(np_fn)

    if HAS_MLX:
        from helicon.postprocess.thrust import _thrust_reduce_mlx

        def mlx_fn():
            return _thrust_reduce_mlx(wt, mass, vz)

        t_mlx = _time_mlx_fn(mlx_fn)
    else:
        t_mlx = None

    speedup = t_np / t_mlx if t_mlx and t_mlx > 0 else None
    suite.add(BenchmarkResult("Thrust reduction (1M particles)", t_np, t_mlx, speedup))


def bench_electron_magnetization(suite: BenchmarkSuite) -> None:
    """Benchmark electron magnetization grid computation."""
    rng = np.random.default_rng(0)
    shape = (128, 256)
    Br = rng.uniform(-0.1, 0.1, shape).astype(np.float32)
    Bz = rng.uniform(0.01, 1.0, shape).astype(np.float32)
    n_e = rng.uniform(1e17, 1e19, shape).astype(np.float32)
    T_e_eV = 500.0

    from helicon.postprocess.plume import compute_electron_magnetization

    def np_fn():
        return compute_electron_magnetization(Br, Bz, n_e, T_e_eV, backend="numpy")

    t_np = _time_fn(np_fn)

    if HAS_MLX:

        def mlx_fn():
            return compute_electron_magnetization(Br, Bz, n_e, T_e_eV, backend="mlx")

        t_mlx = _time_mlx_fn(mlx_fn)
    else:
        t_mlx = None

    speedup = t_np / t_mlx if t_mlx and t_mlx > 0 else None
    suite.add(BenchmarkResult("Electron magnetization (128×256 grid)", t_np, t_mlx, speedup))


def bench_analytical_screening(suite: BenchmarkSuite) -> None:
    """Benchmark batch analytical screening over 10K mirror ratios."""
    from helicon.optimize.analytical import (
        divergence_half_angle_batch,
        thrust_coefficient_batch,
        thrust_efficiency_batch,
    )

    R_B = np.random.default_rng(7).uniform(2.0, 50.0, 10_000)

    def np_fn():
        thrust_efficiency_batch(R_B, backend="numpy")
        thrust_coefficient_batch(R_B, backend="numpy")
        divergence_half_angle_batch(R_B, backend="numpy")

    t_np = _time_fn(np_fn)

    if HAS_MLX:

        def mlx_fn():
            e = thrust_efficiency_batch(R_B, backend="mlx")
            c = thrust_coefficient_batch(R_B, backend="mlx")
            d = divergence_half_angle_batch(R_B, backend="mlx")
            return e, c, d

        t_mlx = _time_mlx_fn(mlx_fn)
    else:
        t_mlx = None

    speedup = t_np / t_mlx if t_mlx and t_mlx > 0 else None
    suite.add(BenchmarkResult("Analytical screening (10K configs)", t_np, t_mlx, speedup))


def bench_differentiable_gradient(suite: BenchmarkSuite) -> None:
    """Benchmark MLX differentiable gradient (forward + backward)."""
    if not HAS_MLX:
        suite.add(BenchmarkResult("Differentiable gradient (MLX only)", 0.0, None, None))
        return

    import mlx.core as mx

    from helicon.optimize.analytical import breizman_arefiev_ct_mlx

    coil_params = mx.array([[0.0, 0.1, 50_000.0], [0.3, 0.05, 20_000.0]], dtype=mx.float32)
    z_eval = mx.linspace(-0.5, 1.5, 64)

    def fwd_fn():
        ct = breizman_arefiev_ct_mlx(coil_params, z_eval, n_phi=32)
        mx.eval(ct)

    t_fwd = _time_mlx_fn(lambda: breizman_arefiev_ct_mlx(coil_params, z_eval, n_phi=32))

    def grad_fn(cp):
        return breizman_arefiev_ct_mlx(cp, z_eval, n_phi=32)

    grad_of = mx.grad(grad_fn)

    def bwd_fn():
        g = grad_of(coil_params)
        mx.eval(g)

    t_bwd = _time_mlx_fn(bwd_fn)

    suite.add(BenchmarkResult("Differentiable fwd (MLX)", 0.0, t_fwd, None))
    suite.add(BenchmarkResult("Differentiable fwd+bwd gradient (MLX)", 0.0, t_bwd, None))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_benchmarks() -> BenchmarkSuite:
    """Run all benchmarks and return the suite."""
    suite = BenchmarkSuite()
    bench_biot_savart(suite)
    bench_thrust_integration(suite)
    bench_electron_magnetization(suite)
    bench_analytical_screening(suite)
    bench_differentiable_gradient(suite)
    return suite


if __name__ == "__main__":
    suite = run_benchmarks()
    print(suite.summary())
