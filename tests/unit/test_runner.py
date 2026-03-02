"""Tests for helicon.runner module."""

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
from helicon.runner.hardware_config import detect_hardware
from helicon.runner.launch import run_simulation


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
    assert hw.recommended_backend in ("omp", "cuda", "metal")


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
    from helicon.fields.biot_savart import BField

    config = _make_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_simulation(config, output_dir=tmpdir, dry_run=True)
        bf = BField.load(str(result.bfield_file))
        assert bf.Bz.shape[0] > 0


def test_dry_run_metadata_has_required_fields():
    """run_metadata.json must contain all spec §14 required fields."""
    import json

    config = _make_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        run_simulation(config, output_dir=tmpdir, dry_run=True)
        meta_path = Path(tmpdir) / "run_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        for field in ("helicon_version", "config_hash", "timestamp", "python_version"):
            assert field in meta, f"Missing required metadata field: {field}"


def test_dry_run_metadata_mass_ratio_reduced_flag():
    """mass_ratio_reduced flag must be present in metadata (spec §14)."""
    import json

    config = _make_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        run_simulation(config, output_dir=tmpdir, dry_run=True)
        meta = json.loads((Path(tmpdir) / "run_metadata.json").read_text())
        assert "mass_ratio_reduced" in meta
        assert meta["mass_ratio_reduced"] is False


def test_dry_run_reduced_ratio_flagged_in_metadata():
    """mass_ratio_reduced=True when reduced mass ratio is configured."""
    import json

    config = SimConfig(
        nozzle=_make_config().nozzle,
        plasma=PlasmaSourceConfig(
            n0=1e18, T_i_eV=100, T_e_eV=100, v_injection_ms=50000, mass_ratio=100.0
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        run_simulation(config, output_dir=tmpdir, dry_run=True)
        meta = json.loads((Path(tmpdir) / "run_metadata.json").read_text())
        assert meta["mass_ratio_reduced"] is True


def test_dry_run_result_has_metadata_dict():
    """RunResult.metadata should expose the collected metadata."""
    config = _make_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_simulation(config, output_dir=tmpdir, dry_run=True)
        assert isinstance(result.metadata, dict)
        assert "helicon_version" in result.metadata
