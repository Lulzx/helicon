"""Apple Silicon hardware profiler and WarpX tuning recommendations.

Implements the Apple Silicon performance guide described in spec §v2.1:

  - Detect chip model and core configuration (P-cores / E-cores / GPU)
  - Measure unified memory bandwidth
  - Recommend WarpX OpenMP thread count (P-cores only, leave E-cores idle)
  - Recommend MLX compilation flags and lazy evaluation strategy
  - Estimate simulation capacity from available unified memory

References
----------
Apple Developer documentation:
  https://developer.apple.com/documentation/apple-silicon

WarpX OpenMP documentation:
  https://warpx.readthedocs.io/en/latest/install/platforms/macos.html
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Hardware detection helpers
# ---------------------------------------------------------------------------


def _sysctl(key: str) -> str:
    """Read a sysctl key; return empty string on failure."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _chip_model() -> str:
    """Return the Apple Silicon chip model string (e.g. 'Apple M4 Pro')."""
    brand = _sysctl("machdep.cpu.brand_string")
    if brand:
        return brand
    hw = _sysctl("hw.model")
    return hw or "unknown"


def _total_memory_gb() -> float:
    """Return total unified memory in GB (cross-platform, no psutil)."""
    # macOS: hw.memsize via sysctl
    mem_bytes = _sysctl("hw.memsize")
    try:
        return int(mem_bytes) / 1e9
    except (ValueError, TypeError):
        pass

    # Linux: /proc/meminfo
    try:
        from pathlib import Path

        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1e6  # kB → GB
    except Exception:
        pass

    # POSIX fallback: sysconf
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return pages * page_size / 1e9
    except Exception:
        pass

    return 1.0  # last-resort default


def _cpu_cores() -> tuple[int, int]:
    """Return (performance_cores, efficiency_cores).

    Uses sysctl hw.perflevel0.physicalcpu for P-cores and
    hw.perflevel1.physicalcpu for E-cores (Apple Silicon).
    Falls back to total logical CPUs on non-AS hardware.
    """
    p_cores_str = _sysctl("hw.perflevel0.physicalcpu")
    e_cores_str = _sysctl("hw.perflevel1.physicalcpu")
    try:
        p = int(p_cores_str)
        e = int(e_cores_str)
        return p, e
    except (ValueError, TypeError):
        total = os.cpu_count() or 1
        return total, 0


def _gpu_cores() -> int:
    """Return GPU core count via system_profiler (best-effort)."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            line_low = line.lower()
            if "total number of cores" in line_low or "gpu core count" in line_low:
                parts = line.split(":")
                if len(parts) >= 2:
                    return int(parts[-1].strip().split()[0])
    except Exception:
        pass
    return 0


def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


def _mlx_version() -> str | None:
    try:
        import mlx

        return mlx.__version__
    except Exception:
        return None


def _numpy_version() -> str:
    try:
        import numpy as np

        return np.__version__
    except Exception:
        return "unknown"


def _python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _measure_memory_bandwidth_gbs() -> float | None:
    """Quick numpy bandwidth probe (stream-copy benchmark).

    Returns GB/s or None if measurement fails.
    """
    try:
        import time

        import numpy as np

        n = 50_000_000  # 50M float32 = 200 MB
        a = np.random.default_rng(0).random(n, dtype=np.float32)
        b = np.empty_like(a)
        # Warm-up
        np.copyto(b, a)
        # Timed copy
        t0 = time.perf_counter()
        np.copyto(b, a)
        dt = time.perf_counter() - t0
        # 2× array size (read + write)
        return 2 * a.nbytes / dt / 1e9
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Recommendation dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OpenMPRecommendation:
    """WarpX OpenMP tuning recommendation.

    Attributes
    ----------
    omp_num_threads : int
        Recommended OMP_NUM_THREADS value (P-cores only).
    omp_places : str
        Recommended OMP_PLACES value.
    omp_proc_bind : str
        Recommended OMP_PROC_BIND value.
    rationale : str
        Human-readable explanation.
    env_snippet : str
        Shell environment snippet ready to copy-paste.
    """

    omp_num_threads: int
    omp_places: str
    omp_proc_bind: str
    rationale: str
    env_snippet: str


@dataclass
class MLXRecommendation:
    """MLX compilation and tuning recommendation.

    Attributes
    ----------
    compile_enabled : bool
        Whether mlx.core.compile should be used.
    lazy_eval : bool
        Whether to batch operations and call mx.eval() periodically.
    metal_capture : bool
        Whether Metal GPU capture can be used for profiling.
    suggested_batch_size : int
        Suggested number of coils to batch in Biot-Savart vmap calls.
    rationale : str
    """

    compile_enabled: bool
    lazy_eval: bool
    metal_capture: bool
    suggested_batch_size: int
    rationale: str


@dataclass
class MemoryRecommendation:
    """Unified memory management recommendation.

    Attributes
    ----------
    max_particles_m : float
        Estimated maximum PIC particle count [millions] before memory pressure.
    recommended_particles_m : float
        Conservative recommended particle count for stable runs.
    max_grid_nz_nr : tuple[int, int]
        Estimated maximum grid dimensions.
    notes : list[str]
        Additional guidance notes.
    """

    max_particles_m: float
    recommended_particles_m: float
    max_grid_nz_nr: tuple[int, int]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HardwareProfile
# ---------------------------------------------------------------------------


@dataclass
class HardwareProfile:
    """Complete hardware profile and tuning recommendations.

    Attributes
    ----------
    is_apple_silicon : bool
    chip_model : str
    p_cores : int
    e_cores : int
    gpu_cores : int
    memory_gb : float
    memory_bandwidth_gbs : float | None
    mlx_available : bool
    mlx_version : str | None
    numpy_version : str
    python_version : str
    openmp : OpenMPRecommendation
    mlx : MLXRecommendation
    memory : MemoryRecommendation
    """

    is_apple_silicon: bool
    chip_model: str
    p_cores: int
    e_cores: int
    gpu_cores: int
    memory_gb: float
    memory_bandwidth_gbs: float | None
    mlx_available: bool
    mlx_version: str | None
    numpy_version: str
    python_version: str
    openmp: OpenMPRecommendation
    mlx: MLXRecommendation
    memory: MemoryRecommendation

    def summary(self) -> str:
        """Human-readable hardware summary."""
        bw = f"{self.memory_bandwidth_gbs:.1f} GB/s" if self.memory_bandwidth_gbs else "N/A"
        lines = [
            "=" * 60,
            "Helicon Apple Silicon Performance Profile",
            "=" * 60,
            f"  Chip:           {self.chip_model}",
            f"  P-cores:        {self.p_cores}",
            f"  E-cores:        {self.e_cores}",
            f"  GPU cores:      {self.gpu_cores or 'unknown'}",
            f"  Unified memory: {self.memory_gb:.0f} GB",
            f"  Mem bandwidth:  {bw}  (NumPy stream-copy estimate)",
            f"  MLX available:  {self.mlx_available}"
            + (f"  (v{self.mlx_version})" if self.mlx_version else ""),
            f"  Python:         {self.python_version}",
            f"  NumPy:          {self.numpy_version}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def recommendations(self) -> str:
        """Human-readable tuning recommendations."""
        lines = [
            "",
            "WarpX OpenMP Tuning",
            "-" * 40,
            f"  OMP_NUM_THREADS={self.openmp.omp_num_threads}",
            f"  OMP_PLACES={self.openmp.omp_places}",
            f"  OMP_PROC_BIND={self.openmp.omp_proc_bind}",
            f"  Rationale: {self.openmp.rationale}",
            "",
            "Shell snippet:",
            self.openmp.env_snippet,
            "",
            "MLX Recommendations",
            "-" * 40,
            f"  mlx.core.compile: {'enabled' if self.mlx.compile_enabled else 'disabled'}",
            f"  Lazy evaluation:  {'enabled' if self.mlx.lazy_eval else 'n/a'}",
            f"  Batch size:       {self.mlx.suggested_batch_size} coils per vmap call",
            f"  Rationale: {self.mlx.rationale}",
            "",
            "Unified Memory Management",
            "-" * 40,
            f"  Max particles:     {self.memory.max_particles_m:.0f}M (hard limit)",
            f"  Recommended:       {self.memory.recommended_particles_m:.0f}M"
            " (safe for full diagnostics)",
            "  Max grid (nz\u00d7nr):  "
            f"{self.memory.max_grid_nz_nr[0]}\u00d7{self.memory.max_grid_nz_nr[1]}",
        ]
        for note in self.memory.notes:
            lines.append(f"  • {note}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON export."""
        return {
            "chip_model": self.chip_model,
            "is_apple_silicon": self.is_apple_silicon,
            "p_cores": self.p_cores,
            "e_cores": self.e_cores,
            "gpu_cores": self.gpu_cores,
            "memory_gb": self.memory_gb,
            "memory_bandwidth_gbs": self.memory_bandwidth_gbs,
            "mlx_available": self.mlx_available,
            "mlx_version": self.mlx_version,
            "numpy_version": self.numpy_version,
            "python_version": self.python_version,
            "openmp": {
                "omp_num_threads": self.openmp.omp_num_threads,
                "omp_places": self.openmp.omp_places,
                "omp_proc_bind": self.openmp.omp_proc_bind,
                "rationale": self.openmp.rationale,
                "env_snippet": self.openmp.env_snippet,
            },
            "mlx": {
                "compile_enabled": self.mlx.compile_enabled,
                "lazy_eval": self.mlx.lazy_eval,
                "metal_capture": self.mlx.metal_capture,
                "suggested_batch_size": self.mlx.suggested_batch_size,
                "rationale": self.mlx.rationale,
            },
            "memory": {
                "max_particles_m": self.memory.max_particles_m,
                "recommended_particles_m": self.memory.recommended_particles_m,
                "max_grid_nz_nr": list(self.memory.max_grid_nz_nr),
                "notes": self.memory.notes,
            },
        }


# ---------------------------------------------------------------------------
# AppleSiliconProfiler
# ---------------------------------------------------------------------------


class AppleSiliconProfiler:
    """Detects hardware and generates tuning recommendations.

    Works on any platform but only produces Apple Silicon-specific
    recommendations when running on arm64 macOS.

    Parameters
    ----------
    measure_bandwidth : bool
        Whether to run the memory bandwidth probe (adds ~0.5 s).
    """

    def __init__(self, *, measure_bandwidth: bool = True) -> None:
        self._measure_bandwidth = measure_bandwidth

    def profile(self) -> HardwareProfile:
        """Probe hardware and compute recommendations.

        Returns
        -------
        HardwareProfile
        """
        is_as = _is_apple_silicon()
        chip = _chip_model()
        p_cores, e_cores = _cpu_cores()
        gpu = _gpu_cores()
        mem_gb = _total_memory_gb()
        bw = _measure_memory_bandwidth_gbs() if self._measure_bandwidth else None
        mlx_avail = _mlx_available()
        mlx_ver = _mlx_version()
        np_ver = _numpy_version()
        py_ver = _python_version()

        openmp = self._openmp_rec(p_cores, e_cores, is_as)
        mlx_rec = self._mlx_rec(p_cores, gpu, mlx_avail, mem_gb)
        mem_rec = self._memory_rec(mem_gb, p_cores)

        return HardwareProfile(
            is_apple_silicon=is_as,
            chip_model=chip,
            p_cores=p_cores,
            e_cores=e_cores,
            gpu_cores=gpu,
            memory_gb=mem_gb,
            memory_bandwidth_gbs=bw,
            mlx_available=mlx_avail,
            mlx_version=mlx_ver,
            numpy_version=np_ver,
            python_version=py_ver,
            openmp=openmp,
            mlx=mlx_rec,
            memory=mem_rec,
        )

    # ------------------------------------------------------------------
    # Recommendation generators
    # ------------------------------------------------------------------

    @staticmethod
    def _openmp_rec(p_cores: int, e_cores: int, is_as: bool) -> OpenMPRecommendation:
        """Generate WarpX OpenMP thread recommendation."""
        if is_as and p_cores > 0:
            # Use only P-cores; E-cores are too slow for PIC inner loops
            # and can degrade performance by creating thread imbalance.
            omp_threads = p_cores
            places = "cores"
            bind = "close"
            rationale = (
                f"Use only P-cores ({p_cores}) — E-cores ({e_cores}) are "
                "slower and create load imbalance in PIC loops. "
                "OMP_PLACES=cores pins threads to physical cores."
            )
        else:
            # Non-Apple-Silicon fallback
            omp_threads = max(1, (p_cores + e_cores) - 1)  # leave 1 for system
            places = "cores"
            bind = "close"
            rationale = "Non-Apple-Silicon: use all but one core to keep system responsive."

        env_snippet = (
            f"export OMP_NUM_THREADS={omp_threads}\n"
            f"export OMP_PLACES={places}\n"
            f"export OMP_PROC_BIND={bind}"
        )

        return OpenMPRecommendation(
            omp_num_threads=omp_threads,
            omp_places=places,
            omp_proc_bind=bind,
            rationale=rationale,
            env_snippet=env_snippet,
        )

    @staticmethod
    def _mlx_rec(
        p_cores: int, gpu_cores: int, mlx_avail: bool, mem_gb: float
    ) -> MLXRecommendation:
        """Generate MLX tuning recommendation."""
        if not mlx_avail:
            return MLXRecommendation(
                compile_enabled=False,
                lazy_eval=False,
                metal_capture=False,
                suggested_batch_size=64,
                rationale="MLX not installed — install with: pip install mlx",
            )

        # Larger GPU → larger batches benefit from Metal parallelism
        if gpu_cores >= 38:
            batch = 2048
        elif gpu_cores >= 20:
            batch = 1024
        elif gpu_cores >= 10:
            batch = 512
        else:
            batch = 256

        rationale = (
            f"mlx.core.compile fuses Metal kernels for 2-5× Biot-Savart speedup. "
            f"Lazy evaluation with mx.eval() every {batch} coils prevents GPU "
            f"command queue overflow. "
        )
        if mem_gb >= 64:
            rationale += "Large unified memory: can evaluate full AMR grid in one pass."
        elif mem_gb >= 32:
            rationale += "32 GB: tile large grids into 2 passes to avoid memory pressure."
        else:
            rationale += "≤24 GB: use diagnostic scan mode to reduce memory footprint."

        return MLXRecommendation(
            compile_enabled=True,
            lazy_eval=True,
            metal_capture=True,
            suggested_batch_size=batch,
            rationale=rationale,
        )

    @staticmethod
    def _memory_rec(mem_gb: float, p_cores: int) -> MemoryRecommendation:
        """Generate unified memory management recommendation."""
        # Rough model:
        # - Each particle takes ~7 floats × 4 bytes = 28 bytes
        # - Reserve 30% for OS + WarpX fields + Python heap
        usable_gb = mem_gb * 0.70
        max_particles_m = (usable_gb * 1e9) / 28 / 1e6
        recommended_particles_m = max_particles_m * 0.60  # leave headroom for diagnostics

        # Grid: 4 bytes × 7 fields × nz × nr
        # Set max at 40% of usable memory for fields
        field_bytes = usable_gb * 0.40 * 1e9
        max_grid_cells = int(field_bytes / (4 * 7))
        # Assume square-ish grid aspect ratio 2:1 (nz:nr)
        nr = int((max_grid_cells / 2) ** 0.5)
        nz = nr * 2

        notes = []
        if mem_gb < 24:
            notes.append(
                "Low memory: use scan diagnostic mode (no particle dumps); "
                "disable checkpoint saves."
            )
        if mem_gb >= 96:
            notes.append(
                "High memory (96+ GB): consider 3D simulations — "
                "full 3D with 100M particles feasible."
            )
        notes.append(
            "Set keep_checkpoints=false in config to save 50-80% disk. "
            "Restart is only needed for multi-day runs."
        )
        if p_cores >= 12:
            notes.append(
                "High P-core count: batch multiple short WarpX runs in parallel "
                "using helicon scan --batch-size for parameter sweeps."
            )

        return MemoryRecommendation(
            max_particles_m=round(max_particles_m, 1),
            recommended_particles_m=round(recommended_particles_m, 1),
            max_grid_nz_nr=(nz, nr),
            notes=notes,
        )
