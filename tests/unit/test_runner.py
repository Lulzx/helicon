"""Tests for magnozzlex.runner module."""

from __future__ import annotations

import tempfile
from pathlib import Path

from magnozzlex.config.parser import (
    CoilConfig,
    DomainConfig,
    NozzleConfig,
    PlasmaSourceConfig,
    SimConfig,
)
from magnozzlex.runner.hardware_config import detect_hardware
from magnozzlex.runner.launch import run_simulation


def _make_config() -> SimConfig:
    return SimConfig(
        nozzle=NozzleConfig(
            coils=[CoilConfig(z=0.0, r=0.1, I=10000)],
            domain=DomainConfig(z_min=-0.3, z_max=1.0, r_max=0.5),
        ),
        plasma=PlasmaSourceConfig(n0=1e18, T_i_eV=100, T_e_eV=100, v_injection_ms=50000),
    )


def test_detect_hardware():
    hw = detect_hardware()
    assert hw.cpu_count >= 1
    assert hw.platform in ("darwin", "linux", "windows")
    assert hw.recommended_backend in ("omp", "cuda")


def test_dry_run_creates_files():
    config = _make_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_simulation(config, output_dir=tmpdir, dry_run=True)
        assert result.success
        assert result.input_file.exists()
        assert result.bfield_file.exists()
        assert (Path(tmpdir) / "run_metadata.json").exists()


def test_dry_run_bfield_is_valid():
    """The pre-computed B-field should be loadable."""
    from magnozzlex.fields.biot_savart import BField

    config = _make_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_simulation(config, output_dir=tmpdir, dry_run=True)
        bf = BField.load(str(result.bfield_file))
        assert bf.Bz.shape[0] > 0
