"""Apple Silicon performance profiler and tuning guide.

Detects hardware capabilities, recommends WarpX OpenMP thread counts,
MLX compilation flags, and unified memory management strategies for
simulation workloads on Apple M-series chips.

Usage::

    from helicon.perf import AppleSiliconProfiler

    profiler = AppleSiliconProfiler()
    profile = profiler.profile()
    print(profile.summary())
    print(profile.recommendations())

CLI::

    helicon perf
    helicon perf --json
"""

from helicon.perf.profiler import (
    AppleSiliconProfiler,
    HardwareProfile,
    MemoryRecommendation,
    MLXRecommendation,
    OpenMPRecommendation,
)

__all__ = [
    "AppleSiliconProfiler",
    "HardwareProfile",
    "MLXRecommendation",
    "MemoryRecommendation",
    "OpenMPRecommendation",
]
