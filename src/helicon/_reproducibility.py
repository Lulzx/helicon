"""Reproducibility metadata collection.

Every Helicon run logs enough information to reproduce the result:
versions, config hash, hardware info, random seeds, and timestamps.
"""

from __future__ import annotations

import datetime
import hashlib
import platform
import socket
import sys

import helicon


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
        "helicon_version": helicon.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
    }

    # Git SHA of helicon installation
    meta["helicon_git_sha"] = _get_git_sha()

    # WarpX version and git SHA
    meta["warpx_version"] = _get_warpx_version()
    meta["warpx_git_sha"] = _get_warpx_git_sha()

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

    # Config hash, contents, and random seed
    if config is not None:
        from helicon.config.parser import SimConfig

        if isinstance(config, SimConfig):
            json_bytes = config.model_dump_json().encode()
            cfg_hash = hashlib.sha256(json_bytes).hexdigest()
            meta["config_hash"] = cfg_hash

            import yaml

            meta["config_contents"] = yaml.safe_dump(config.model_dump())

            # §14: deterministic random seed from config_hash when not user-specified
            if config.random_seed is not None:
                meta["random_seed"] = config.random_seed
            else:
                meta["random_seed"] = int(cfg_hash[:8], 16) % (2**31)

            # §14: flag when a non-physical mass ratio is used
            mr = config.plasma.mass_ratio
            meta["mass_ratio_reduced"] = mr is not None and mr < 1836.0

    return meta


def _get_git_sha() -> str | None:
    """Try to get the git SHA of the helicon installation."""
    import subprocess
    from pathlib import Path

    pkg_dir = Path(helicon.__file__).parent
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


def _get_warpx_git_sha() -> str | None:
    """Try to get the git SHA of the WarpX installation."""
    import subprocess
    from pathlib import Path

    try:
        import pywarpx

        pkg_dir = Path(pywarpx.__file__).parent
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(pkg_dir),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
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
    """Get GPU model name on Linux via nvidia-smi.

    Returns None on other platforms or on failure.
    """
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
