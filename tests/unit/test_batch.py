"""Tests for helicon.runner.batch module."""

from __future__ import annotations

import tempfile
from pathlib import Path

from helicon.config.parser import (
    CoilConfig,
    DomainConfig,
    NozzleConfig,
    PlasmaSourceConfig,
    SimConfig,
)
from helicon.runner.batch import (
    BatchConfig,
    BatchResult,
    generate_pbs_script,
    generate_slurm_script,
    run_local_batch,
    submit_batch,
)


def _make_config() -> SimConfig:
    return SimConfig(
        nozzle=NozzleConfig(
            coils=[CoilConfig(z=0.0, r=0.1, I=10000)],
            domain=DomainConfig(z_min=-0.3, z_max=1.0, r_max=0.5),
        ),
        plasma=PlasmaSourceConfig(n0=1e18, T_i_eV=100, T_e_eV=100, v_injection_ms=50000),
    )


def test_batch_config_defaults():
    bc = BatchConfig()
    assert bc.backend == "local"
    assert bc.n_workers == 4
    assert bc.slurm_partition is None
    assert bc.slurm_account is None
    assert bc.slurm_time == "12:00:00"
    assert bc.slurm_ntasks == 1
    assert bc.slurm_cpus_per_task == 16
    assert bc.slurm_mem == "64G"
    assert bc.pbs_queue is None
    assert bc.pbs_walltime is None
    assert bc.pbs_ncpus == 16


def test_generate_slurm_script_contains_directives():
    config = _make_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "run_0000"
        script = generate_slurm_script(
            config,
            out,
            "helicon",
            partition="gpu",
            account="myaccount",
            time="06:00:00",
            ntasks=2,
            cpus_per_task=8,
            mem="32G",
        )
        assert "#SBATCH" in script
        assert "#SBATCH --partition=gpu" in script
        assert "#SBATCH --account=myaccount" in script
        assert "#SBATCH --time=06:00:00" in script
        assert "#SBATCH --ntasks=2" in script
        assert "#SBATCH --cpus-per-task=8" in script
        assert "#SBATCH --mem=32G" in script
        assert "helicon run" in script


def test_generate_pbs_script_contains_directives():
    config = _make_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "run_0000"
        script = generate_pbs_script(
            config,
            out,
            "helicon",
            queue="batch",
            walltime="08:00:00",
            ncpus=32,
        )
        assert "#PBS" in script
        assert "#PBS -q batch" in script
        assert "#PBS -l walltime=08:00:00" in script
        assert "#PBS -l ncpus=32" in script
        assert "helicon run" in script


def test_run_local_batch_dry_run_creates_output_dirs():
    configs = [_make_config() for _ in range(3)]
    with tempfile.TemporaryDirectory() as tmpdir:
        run_local_batch(configs, tmpdir, dry_run=True)
        for i in range(3):
            assert (Path(tmpdir) / f"run_{i:04d}").is_dir()


def test_run_local_batch_dry_run_returns_correct_n_completed():
    configs = [_make_config() for _ in range(5)]
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_local_batch(configs, tmpdir, dry_run=True)
        assert result.n_completed == 5
        assert result.n_failed == 0
        assert len(result.jobs) == 5


def test_submit_batch_local_dry_run():
    configs = [_make_config() for _ in range(2)]
    bc = BatchConfig(backend="local", n_workers=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = submit_batch(configs, bc, tmpdir, dry_run=True)
        assert result.n_completed == 2
        assert result.n_failed == 0


def test_batch_result_has_correct_fields():
    br = BatchResult()
    assert hasattr(br, "jobs")
    assert hasattr(br, "n_completed")
    assert hasattr(br, "n_failed")
    assert hasattr(br, "wall_time_seconds")
    assert isinstance(br.jobs, list)
    assert br.n_completed == 0
    assert br.n_failed == 0
    assert br.wall_time_seconds == 0.0
