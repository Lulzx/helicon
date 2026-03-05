# helicon.perf

Apple Silicon performance profiler and WarpX tuning guide.

Detects chip hardware (P-cores, E-cores, GPU cores, unified memory),
recommends WarpX OpenMP settings, and estimates simulation capacity.

## CLI

```bash
# Human-readable profile
helicon perf

# JSON output for scripting
helicon perf --json > perf_report.json

# Skip bandwidth measurement (faster)
helicon perf --skip-bandwidth
```

## Python API

```python
from helicon.perf import AppleSiliconProfiler

profiler = AppleSiliconProfiler()
profile = profiler.profile()

print(profile.summary())
print(profile.recommendations())

# Access specific recommendations
omp = profile.openmp
print(f"OMP_NUM_THREADS={omp.omp_num_threads}")
print(f"Rationale: {omp.rationale}")

mem = profile.memory
print(f"Max particles: {mem.max_particles_fp32:,}")
print(f"Recommended:   {mem.recommended_particles_fp32:,}")
```

## Output Example

```
============================================================
Helicon Apple Silicon Performance Profile
============================================================
  Chip:           Apple M4 Pro
  P-cores:        12
  E-cores:        4
  GPU cores:      38
  Unified memory: 48 GB
  Mem bandwidth:  98.3 GB/s
  MLX available:  True  (v0.31.0)
============================================================

WarpX OpenMP Tuning
  OMP_NUM_THREADS=12
  OMP_PLACES=cores
  OMP_PROC_BIND=close
```

::: helicon.perf.AppleSiliconProfiler
::: helicon.perf.HardwareProfile
::: helicon.perf.OpenMPRecommendation
::: helicon.perf.MLXRecommendation
::: helicon.perf.MemoryRecommendation
