"""Hardware detection and WarpX backend selection.

Detects available compute resources (Apple Silicon, NVIDIA GPU, CPU)
and recommends the appropriate WarpX backend.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""

    platform: str  # "darwin", "linux", "windows"
    arch: str  # "arm64", "x86_64"
    cpu_count: int
    is_apple_silicon: bool = False
    apple_chip: str | None = None
    has_nvidia_gpu: bool = False
    nvidia_gpu_name: str | None = None
    cuda_version: str | None = None
    has_mlx: bool = False
    mlx_version: str | None = None
    has_pywarpx: bool = False
    warpx_version: str | None = None
    recommended_backend: str = "cpu"
    omp_num_threads: int = 1

    def summary(self) -> str:
        """Human-readable hardware summary."""
        lines = [
            f"Platform: {self.platform} ({self.arch})",
            f"CPU cores: {self.cpu_count}",
        ]
        if self.is_apple_silicon:
            lines.append(f"Apple Silicon: {self.apple_chip or 'yes'}")
        if self.has_nvidia_gpu:
            lines.append(f"NVIDIA GPU: {self.nvidia_gpu_name or 'detected'}")
            if self.cuda_version:
                lines.append(f"CUDA: {self.cuda_version}")
        if self.has_mlx:
            lines.append(f"MLX: {self.mlx_version or 'available'}")
        if self.has_pywarpx:
            lines.append(f"WarpX: {self.warpx_version or 'available'}")
        lines.append(f"Recommended backend: {self.recommended_backend}")
        lines.append(f"OMP threads: {self.omp_num_threads}")
        return "\n".join(lines)


def detect_hardware() -> HardwareInfo:
    """Detect available hardware and recommend a WarpX backend."""
    info = HardwareInfo(
        platform=platform.system().lower(),
        arch=platform.machine(),
        cpu_count=os.cpu_count() or 1,
    )

    # Apple Silicon detection
    if info.platform == "darwin" and info.arch == "arm64":
        info.is_apple_silicon = True
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                info.apple_chip = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # NVIDIA GPU detection
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                info.has_nvidia_gpu = True
                info.nvidia_gpu_name = result.stdout.strip().split("\n")[0]

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                info.cuda_version = result.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # MLX detection
    try:
        import mlx.core as mx

        info.has_mlx = True
        info.mlx_version = getattr(mx, "__version__", "unknown")
    except ImportError:
        pass

    # pywarpx detection
    try:
        import pywarpx

        info.has_pywarpx = True
        info.warpx_version = getattr(pywarpx, "__version__", "unknown")
    except ImportError:
        pass

    # Recommended backend
    if info.has_nvidia_gpu:
        info.recommended_backend = "cuda"
    elif info.is_apple_silicon:
        info.recommended_backend = "omp"  # WarpX is CPU-only on macOS
    else:
        info.recommended_backend = "omp"

    # OMP threads — use all cores unless overridden
    info.omp_num_threads = int(os.environ.get("OMP_NUM_THREADS", info.cpu_count))

    return info
