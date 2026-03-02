"""Batch run submission for parameter scans.

Supports local parallel execution, SLURM cluster submission,
and PBS cluster submission.
"""

from __future__ import annotations

import logging
import subprocess
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent

from helicon.config.parser import SimConfig

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch job submission."""

    backend: str = "local"  # "local" | "slurm" | "pbs"
    n_workers: int = 4
    slurm_partition: str | None = None
    slurm_account: str | None = None
    slurm_time: str = "12:00:00"
    slurm_ntasks: int = 1
    slurm_cpus_per_task: int = 16
    slurm_mem: str = "64G"
    pbs_queue: str | None = None
    pbs_walltime: str | None = None
    pbs_ncpus: int = 16


@dataclass
class BatchJob:
    """A single job within a batch run."""

    index: int
    config: SimConfig
    output_dir: Path
    job_id: str | None = None
    status: str = "pending"  # "pending" | "running" | "completed" | "failed"


@dataclass
class BatchResult:
    """Result of a batch run submission."""

    jobs: list[BatchJob] = field(default_factory=list)
    n_completed: int = 0
    n_failed: int = 0
    wall_time_seconds: float = 0.0


def _run_single_dry(job: BatchJob) -> BatchJob:
    """Execute a single dry-run job (creates output dir only)."""
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.status = "completed"
    return job


def _run_single(job: BatchJob) -> BatchJob:
    """Execute a single simulation job locally."""
    from helicon.runner.launch import run_simulation

    try:
        job.status = "running"
        result = run_simulation(job.config, output_dir=job.output_dir, dry_run=False)
        job.status = "completed" if result.success else "failed"
    except Exception:
        logger.exception("Job %d failed", job.index)
        job.status = "failed"
    return job


def run_local_batch(
    configs: list[SimConfig],
    output_base: str | Path,
    *,
    n_workers: int = 4,
    dry_run: bool = False,
) -> BatchResult:
    """Run a batch of simulations locally using parallel workers.

    Parameters
    ----------
    configs : list[SimConfig]
        Simulation configurations to run.
    output_base : path
        Base output directory. Each run goes to ``output_base/run_NNNN/``.
    n_workers : int
        Number of parallel workers.
    dry_run : bool
        If True, create output directories but do not launch simulations.

    Returns
    -------
    BatchResult
    """
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    jobs = [
        BatchJob(
            index=i,
            config=cfg,
            output_dir=output_base / f"run_{i:04d}",
        )
        for i, cfg in enumerate(configs)
    ]

    t0 = time.monotonic()

    executor_cls = ThreadPoolExecutor if dry_run else ProcessPoolExecutor
    worker_fn = _run_single_dry if dry_run else _run_single

    with executor_cls(max_workers=n_workers) as executor:
        completed_jobs = list(executor.map(worker_fn, jobs))

    wall = time.monotonic() - t0
    n_completed = sum(1 for j in completed_jobs if j.status == "completed")
    n_failed = sum(1 for j in completed_jobs if j.status == "failed")

    return BatchResult(
        jobs=completed_jobs,
        n_completed=n_completed,
        n_failed=n_failed,
        wall_time_seconds=wall,
    )


def generate_slurm_script(
    config: SimConfig,
    output_dir: Path,
    helicon_exe: str,
    *,
    partition: str | None = None,
    account: str | None = None,
    time: str = "12:00:00",
    ntasks: int = 1,
    cpus_per_task: int = 16,
    mem: str = "64G",
) -> str:
    """Generate a SLURM sbatch submission script.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration.
    output_dir : Path
        Directory for this run's output.
    helicon_exe : str
        Command or path for the helicon executable.
    partition, account, time, ntasks, cpus_per_task, mem
        SLURM resource directives.

    Returns
    -------
    str
        Complete SLURM script content.
    """
    config_path = output_dir / "config.yaml"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=helicon",
        f"#SBATCH --output={output_dir / 'slurm_%j.out'}",
        f"#SBATCH --error={output_dir / 'slurm_%j.err'}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --time={time}",
    ]
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    if account:
        lines.append(f"#SBATCH --account={account}")

    lines.extend([
        "",
        f"cd {output_dir}",
        f"{helicon_exe} run {config_path}",
    ])

    return "\n".join(lines) + "\n"


def generate_pbs_script(
    config: SimConfig,
    output_dir: Path,
    helicon_exe: str,
    *,
    queue: str | None = None,
    walltime: str | None = None,
    ncpus: int = 16,
) -> str:
    """Generate a PBS qsub submission script.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration.
    output_dir : Path
        Directory for this run's output.
    helicon_exe : str
        Command or path for the helicon executable.
    queue, walltime, ncpus
        PBS resource directives.

    Returns
    -------
    str
        Complete PBS script content.
    """
    config_path = output_dir / "config.yaml"
    lines = [
        "#!/bin/bash",
        f"#PBS -N helicon",
        f"#PBS -o {output_dir / 'pbs_output.log'}",
        f"#PBS -e {output_dir / 'pbs_error.log'}",
        f"#PBS -l ncpus={ncpus}",
    ]
    if queue:
        lines.append(f"#PBS -q {queue}")
    if walltime:
        lines.append(f"#PBS -l walltime={walltime}")

    lines.extend([
        "",
        f"cd {output_dir}",
        f"{helicon_exe} run {config_path}",
    ])

    return "\n".join(lines) + "\n"


def submit_batch(
    configs: list[SimConfig],
    batch_config: BatchConfig,
    output_base: str | Path,
    *,
    dry_run: bool = False,
) -> BatchResult:
    """Submit a batch of simulations using the configured backend.

    Parameters
    ----------
    configs : list[SimConfig]
        Simulation configurations to run.
    batch_config : BatchConfig
        Backend and resource configuration.
    output_base : path
        Base output directory.
    dry_run : bool
        If True, prepare but do not actually submit/run.

    Returns
    -------
    BatchResult
    """
    if batch_config.backend == "local":
        return run_local_batch(
            configs, output_base, n_workers=batch_config.n_workers, dry_run=dry_run
        )

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    jobs: list[BatchJob] = []
    t0 = time.monotonic()

    for i, cfg in enumerate(configs):
        out_dir = output_base / f"run_{i:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write config for the job
        cfg.to_yaml(out_dir / "config.yaml")

        job = BatchJob(index=i, config=cfg, output_dir=out_dir)

        if batch_config.backend == "slurm":
            script = generate_slurm_script(
                cfg,
                out_dir,
                "helicon",
                partition=batch_config.slurm_partition,
                account=batch_config.slurm_account,
                time=batch_config.slurm_time,
                ntasks=batch_config.slurm_ntasks,
                cpus_per_task=batch_config.slurm_cpus_per_task,
                mem=batch_config.slurm_mem,
            )
            script_path = out_dir / "submit.sh"
            script_path.write_text(script)

            if not dry_run:
                try:
                    result = subprocess.run(
                        ["sbatch", str(script_path)],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if result.returncode == 0:
                        # Parse job ID from "Submitted batch job 12345"
                        parts = result.stdout.strip().split()
                        job.job_id = parts[-1] if parts else None
                        job.status = "running"
                    else:
                        warnings.warn(
                            f"sbatch submission failed: {result.stderr.strip()}",
                            stacklevel=2,
                        )
                        job.status = "failed"
                except FileNotFoundError:
                    warnings.warn(
                        "sbatch command not found. Is SLURM installed?",
                        stacklevel=2,
                    )
                    job.status = "failed"
            else:
                job.status = "completed"

        elif batch_config.backend == "pbs":
            script = generate_pbs_script(
                cfg,
                out_dir,
                "helicon",
                queue=batch_config.pbs_queue,
                walltime=batch_config.pbs_walltime,
                ncpus=batch_config.pbs_ncpus,
            )
            script_path = out_dir / "submit.sh"
            script_path.write_text(script)

            if not dry_run:
                try:
                    result = subprocess.run(
                        ["qsub", str(script_path)],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if result.returncode == 0:
                        job.job_id = result.stdout.strip()
                        job.status = "running"
                    else:
                        warnings.warn(
                            f"qsub submission failed: {result.stderr.strip()}",
                            stacklevel=2,
                        )
                        job.status = "failed"
                except FileNotFoundError:
                    warnings.warn(
                        "qsub command not found. Is PBS installed?",
                        stacklevel=2,
                    )
                    job.status = "failed"
            else:
                job.status = "completed"

        else:
            msg = f"Unknown backend: {batch_config.backend!r}"
            raise ValueError(msg)

        jobs.append(job)

    wall = time.monotonic() - t0
    n_completed = sum(1 for j in jobs if j.status == "completed")
    n_failed = sum(1 for j in jobs if j.status == "failed")

    return BatchResult(
        jobs=jobs,
        n_completed=n_completed,
        n_failed=n_failed,
        wall_time_seconds=wall,
    )
