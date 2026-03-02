"""Tests for helicon.perf — Apple Silicon performance profiler."""

from __future__ import annotations

import json

from helicon.perf import AppleSiliconProfiler, HardwareProfile
from helicon.perf.profiler import (
    MemoryRecommendation,
    OpenMPRecommendation,
    _cpu_cores,
    _is_apple_silicon,
    _measure_memory_bandwidth_gbs,
    _mlx_available,
    _total_memory_gb,
)

# ---------------------------------------------------------------------------
# Profiler smoke tests
# ---------------------------------------------------------------------------


def test_profile_returns_hardware_profile():
    """Profile completes without error on any platform."""
    profiler = AppleSiliconProfiler(measure_bandwidth=False)
    profile = profiler.profile()
    assert isinstance(profile, HardwareProfile)


def test_profile_chip_model_nonempty():
    profiler = AppleSiliconProfiler(measure_bandwidth=False)
    profile = profiler.profile()
    assert len(profile.chip_model) > 0


def test_profile_cores_positive():
    profiler = AppleSiliconProfiler(measure_bandwidth=False)
    profile = profiler.profile()
    assert profile.p_cores >= 1


def test_profile_memory_positive():
    profiler = AppleSiliconProfiler(measure_bandwidth=False)
    profile = profiler.profile()
    assert profile.memory_gb > 0


def test_profile_summary_contains_chip():
    profiler = AppleSiliconProfiler(measure_bandwidth=False)
    profile = profiler.profile()
    summary = profile.summary()
    assert "Chip" in summary or "chip" in summary.lower()


def test_profile_recommendations_contains_omp():
    profiler = AppleSiliconProfiler(measure_bandwidth=False)
    profile = profiler.profile()
    recs = profile.recommendations()
    assert "OMP_NUM_THREADS" in recs


def test_profile_to_dict():
    profiler = AppleSiliconProfiler(measure_bandwidth=False)
    profile = profiler.profile()
    d = profile.to_dict()
    assert "chip_model" in d
    assert "openmp" in d
    assert "mlx" in d
    assert "memory" in d


def test_profile_to_dict_json_serialisable():
    profiler = AppleSiliconProfiler(measure_bandwidth=False)
    profile = profiler.profile()
    d = profile.to_dict()
    # Should not raise
    json_str = json.dumps(d)
    assert len(json_str) > 10


# ---------------------------------------------------------------------------
# OpenMP recommendations
# ---------------------------------------------------------------------------


def test_openmp_rec_apple_silicon():
    rec = AppleSiliconProfiler._openmp_rec(p_cores=12, e_cores=4, is_as=True)
    assert isinstance(rec, OpenMPRecommendation)
    assert rec.omp_num_threads == 12  # P-cores only
    assert "P-cores" in rec.rationale or "p_cores" in rec.rationale.lower()


def test_openmp_rec_non_apple():
    rec = AppleSiliconProfiler._openmp_rec(p_cores=8, e_cores=0, is_as=False)
    assert rec.omp_num_threads >= 1
    assert "OMP_NUM_THREADS" in rec.env_snippet


def test_openmp_env_snippet_contains_all():
    rec = AppleSiliconProfiler._openmp_rec(p_cores=6, e_cores=2, is_as=True)
    assert "OMP_NUM_THREADS" in rec.env_snippet
    assert "OMP_PLACES" in rec.env_snippet
    assert "OMP_PROC_BIND" in rec.env_snippet


# ---------------------------------------------------------------------------
# MLX recommendations
# ---------------------------------------------------------------------------


def test_mlx_rec_no_mlx():
    rec = AppleSiliconProfiler._mlx_rec(p_cores=8, gpu_cores=16, mlx_avail=False, mem_gb=16.0)
    assert not rec.compile_enabled
    assert "pip install mlx" in rec.rationale.lower() or "mlx" in rec.rationale.lower()


def test_mlx_rec_with_mlx_small_gpu():
    rec = AppleSiliconProfiler._mlx_rec(p_cores=8, gpu_cores=10, mlx_avail=True, mem_gb=16.0)
    assert rec.compile_enabled
    assert rec.suggested_batch_size >= 256


def test_mlx_rec_with_mlx_large_gpu():
    rec = AppleSiliconProfiler._mlx_rec(p_cores=14, gpu_cores=40, mlx_avail=True, mem_gb=48.0)
    assert rec.suggested_batch_size >= 2048


def test_mlx_rec_rationale_nonempty():
    rec = AppleSiliconProfiler._mlx_rec(p_cores=8, gpu_cores=20, mlx_avail=True, mem_gb=32.0)
    assert len(rec.rationale) > 10


# ---------------------------------------------------------------------------
# Memory recommendations
# ---------------------------------------------------------------------------


def test_memory_rec_basic():
    rec = AppleSiliconProfiler._memory_rec(mem_gb=32.0, p_cores=8)
    assert isinstance(rec, MemoryRecommendation)
    assert rec.max_particles_m > 0
    assert rec.recommended_particles_m <= rec.max_particles_m
    assert rec.max_grid_nz_nr[0] > 0 and rec.max_grid_nz_nr[1] > 0


def test_memory_rec_large_mem():
    rec = AppleSiliconProfiler._memory_rec(mem_gb=96.0, p_cores=14)
    assert rec.max_particles_m > rec.recommended_particles_m
    assert any("3D" in n for n in rec.notes)


def test_memory_rec_small_mem():
    rec = AppleSiliconProfiler._memory_rec(mem_gb=8.0, p_cores=4)
    # Should warn about low memory
    assert any("scan" in n.lower() or "low" in n.lower() for n in rec.notes)


def test_memory_rec_notes_nonempty():
    rec = AppleSiliconProfiler._memory_rec(mem_gb=16.0, p_cores=4)
    assert len(rec.notes) > 0


# ---------------------------------------------------------------------------
# Helper function smoke tests
# ---------------------------------------------------------------------------


def test_is_apple_silicon_bool():
    result = _is_apple_silicon()
    assert isinstance(result, bool)


def test_cpu_cores_tuple():
    p, e = _cpu_cores()
    assert isinstance(p, int) and isinstance(e, int)
    assert p >= 1 and e >= 0


def test_total_memory_gb_positive():
    mem = _total_memory_gb()
    assert mem > 0


def test_mlx_available_bool():
    result = _mlx_available()
    assert isinstance(result, bool)


def test_memory_bandwidth_returns_float_or_none():
    bw = _measure_memory_bandwidth_gbs()
    assert bw is None or isinstance(bw, float)
    if bw is not None:
        assert bw > 0


# ---------------------------------------------------------------------------
# With bandwidth measurement enabled
# ---------------------------------------------------------------------------


def test_profile_with_bandwidth():
    """With bandwidth=True the profile should complete and bw may be non-None."""
    profiler = AppleSiliconProfiler(measure_bandwidth=True)
    profile = profiler.profile()
    assert profile.memory_bandwidth_gbs is None or profile.memory_bandwidth_gbs > 0
