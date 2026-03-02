"""Tests for helicon.benchmark module."""

from __future__ import annotations

from helicon.benchmark import BenchmarkResult, BenchmarkSuite, run_benchmarks


def test_benchmark_result_str() -> None:
    r = BenchmarkResult("Test case", numpy_ms=10.0, mlx_ms=5.0, speedup=2.0)
    s = str(r)
    assert "Test case" in s
    assert "numpy" in s
    assert "mlx" in s
    assert "2.00" in s


def test_benchmark_result_no_mlx() -> None:
    r = BenchmarkResult("No MLX", numpy_ms=10.0, mlx_ms=None, speedup=None)
    s = str(r)
    assert "N/A" in s


def test_benchmark_suite_add() -> None:
    suite = BenchmarkSuite()
    suite.add(BenchmarkResult("A", 5.0, 2.5, 2.0))
    assert len(suite.results) == 1


def test_benchmark_suite_summary() -> None:
    suite = BenchmarkSuite()
    suite.add(BenchmarkResult("A", 5.0, None, None))
    s = suite.summary()
    assert "Helicon Benchmark Suite" in s
    assert "A" in s


def test_benchmark_runs() -> None:
    """run_benchmarks() must complete without errors."""
    suite = run_benchmarks()
    assert len(suite.results) >= 4
    for r in suite.results:
        assert isinstance(r, BenchmarkResult)
        assert r.numpy_ms >= 0.0
