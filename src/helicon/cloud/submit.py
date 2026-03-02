"""Cloud scan submission helpers.

High-level entry point for ``helicon scan --cloud``.
Serialises the scan job, selects the backend, and submits.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from helicon.cloud.backends import CloudJob, get_backend

if TYPE_CHECKING:
    from helicon.optimize.scan import ParameterRange


def submit_cloud_scan(
    config_path: str | Path,
    ranges: list[ParameterRange],
    *,
    output_dir: str | Path = "cloud_scan_results",
    backend: str = "local",
    method: str = "lhc",
    dry_run: bool = True,
    seed: int = 0,
    n_gpus: int = 1,
    instance_type: str | None = None,
    job_manifest_path: str | Path | None = None,
) -> CloudJob:
    """Submit a parameter scan job to a cloud backend.

    Parameters
    ----------
    config_path : path-like
        YAML config file for the scan.
    ranges : list of ParameterRange
        Parameter ranges to scan.
    output_dir : path-like
        Where to write results.
    backend : str
        Cloud backend name: ``"local"``, ``"lambda"``, or ``"aws"``.
    method : str
        Sampling method: ``"grid"`` or ``"lhc"``.
    dry_run : bool
        If True, generate configs without running WarpX.
    seed : int
        RNG seed for LHC sampling.
    n_gpus : int
        Number of GPUs to request (cloud backends).
    instance_type : str, optional
        Cloud instance type (backend-specific).
    job_manifest_path : path-like, optional
        Where to save the job manifest JSON.
        Defaults to ``<output_dir>/cloud_job.json``.

    Returns
    -------
    CloudJob
    """
    b = get_backend(backend)

    extra: dict[str, Any] = {
        "ranges": [{"path": r.path, "low": r.low, "high": r.high, "n": r.n} for r in ranges],
        "method": method,
        "dry_run": dry_run,
        "seed": seed,
    }

    job = b.submit(
        config_path,
        output_dir,
        n_gpus=n_gpus,
        instance_type=instance_type,
        extra=extra,
    )

    # Save job manifest
    if job_manifest_path is None:
        job_manifest_path = Path(output_dir) / "cloud_job.json"
    job.save(job_manifest_path)

    return job
