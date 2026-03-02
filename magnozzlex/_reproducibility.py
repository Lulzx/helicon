"""Reproducibility metadata collection.

Every MagNozzleX run logs enough information to reproduce the result:
versions, config hash, hardware info, random seeds, and timestamps.
"""

from __future__ import annotations

import datetime
import hashlib
import platform
import socket
import sys

import magnozzlex


def collect_metadata(config: object | None = None) -> dict:
    """Collect reproducibility metadata for a simulation run.

    Parameters
    ----------
    config : SimConfig, optional
        If provided, includes config hash and contents.

    Returns
    -------
    dict
        Metadata dictionary suitable for JSON serialisation.
    """
    meta: dict = {
        "magnozzlex_version": magnozzlex.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
    }

    # Git SHA of magnozzlex installation
    meta["magnozzlex_git_sha"] = _get_git_sha()

    # WarpX version and git SHA
    meta["warpx_version"] = _get_warpx_version()
    meta["warpx_git_sha"] = None

    # Dependency versions
    for pkg in ["numpy", "scipy", "pydantic", "h5py"]:
        meta[f"{pkg}_version"] = _get_version(pkg)

    # Torch version
    meta["torch_version"] = _get_version("torch")

    # MLX info
    try:
        import mlx.core as mx

        meta["mlx_version"] = getattr(mx, "__version__", "unknown")
        meta["mlx_device"] = str(mx.default_device())
    except ImportError:
        meta["mlx_version"] = None

    # Apple Silicon info
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        import subprocess

        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                meta["apple_silicon_chip"] = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # Metal version (macOS Darwin only)
    meta["metal_version"] = _get_metal_version()

    # CUDA version
    meta["cuda_version"] = _get_cuda_version()

    # GPU model
    meta["gpu_model"] = _get_gpu_model()

    # Config hash and contents
    if config is not None:
        from magnozzlex.config.parser import SimConfig

        if isinstance(config, SimConfig):
            json_bytes = config.model_dump_json().encode()
            meta["config_hash"] = hashlib.sha256(json_bytes).hexdigest()

            import yaml

            meta["config_contents"] = yaml.safe_dump(config.model_dump())

    return meta


def _get_git_sha() -> str | None:
    """Try to get the git SHA of the magnozzlex installation."""
    import subprocess
    from pathlib import Path

    pkg_dir = Path(magnozzlex.__file__).parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(pkg_dir),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_version(package: str) -> str | None:
    """Get version string for an installed package."""
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:
        return None


def _get_warpx_version() -> str | None:
    """Try to get the WarpX version via importlib.metadata, then pywarpx.__version__."""
    try:
        from importlib.metadata import version

        return version("warpx")
    except Exception:
        pass
    try:
        import pywarpx

        return pywarpx.__version__
    except Exception:
        return None


def _get_metal_version() -> str | None:
    """Get Metal version on macOS Darwin; return None on other platforms."""
    import subprocess

    if platform.system() != "Darwin":
        return None

    # Try system_profiler for "Metal Feature Set:" line
    try:
        result = subprocess.run(
            ["system_profiler", "SPMetalDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "Metal Feature Set:" in line:
                    return line.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: sysctl hw.targettype
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.targettype"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _get_cuda_version() -> str | None:
    """Get CUDA version via torch.version.cuda if torch is available."""
    try:
        import torch

        return torch.version.cuda
    except Exception:
        return None


def _get_gpu_model() -> str | None:
    """Get GPU model name on Linux via nvidia-smi; returns None on other platforms or on failure."""
    import subprocess

    if platform.system() != "Linux":
        return None

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None
