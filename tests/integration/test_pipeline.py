"""Integration tests for the Helicon pipeline.

All tests use dry_run=True so WarpX is not required.
"""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from helicon.config.parser import (
    CoilConfig,
    DomainConfig,
    NozzleConfig,
    PlasmaSourceConfig,
    ResolutionConfig,
    SimConfig,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_simple_config() -> SimConfig:
    """Return a minimal SimConfig suitable for fast integration tests."""
    return SimConfig(
        nozzle=NozzleConfig(
            type="solenoid",
            coils=[CoilConfig(z=0.0, r=0.1, I=10000.0)],
            domain=DomainConfig(z_min=-0.3, z_max=1.0, r_max=0.5),
            resolution=ResolutionConfig(nz=16, nr=8),
        ),
        plasma=PlasmaSourceConfig(
            n0=1e18,
            T_i_eV=100.0,
            T_e_eV=100.0,
            v_injection_ms=50000.0,
        ),
    )


def _write_config_yaml(path: Path) -> Path:
    """Write a minimal YAML config to *path* and return it."""
    data = {
        "nozzle": {
            "type": "solenoid",
            "coils": [{"z": 0.0, "r": 0.1, "I": 10000}],
            "domain": {"z_min": -0.3, "z_max": 1.0, "r_max": 0.5},
            "resolution": {"nz": 16, "nr": 8},
        },
        "plasma": {
            "n0": 1e18,
            "T_i_eV": 100.0,
            "T_e_eV": 100.0,
            "v_injection_ms": 50000.0,
        },
    }
    config_file = path / "config.yaml"
    config_file.write_text(yaml.dump(data))
    return config_file


# ---------------------------------------------------------------------------
# Test 1: full dry-run pipeline — config → bfield → WarpX input file
# ---------------------------------------------------------------------------

def test_full_dry_run_pipeline(tmp_path: Path) -> None:
    """Config → B-field precomputation → WarpX input file (dry_run=True)."""
    from helicon.runner.launch import run_simulation

    config = _make_simple_config()
    output_dir = tmp_path / "pipeline_out"

    result = run_simulation(config, output_dir=output_dir, dry_run=True)

    assert result.success, "dry-run pipeline should succeed"
    assert result.output_dir == output_dir
    assert result.bfield_file.exists(), "B-field HDF5 file must be created"
    assert result.input_file.exists(), "WarpX input file must be created"
    assert (output_dir / "run_metadata.json").exists(), "metadata JSON must be created"


# ---------------------------------------------------------------------------
# Test 2: scan dry-run — 2×2 grid, verify 4 output dirs created
# ---------------------------------------------------------------------------

def test_scan_dry_run(tmp_path: Path) -> None:
    """2×2 parameter grid dry-run scan must produce 4 point directories."""
    from helicon.optimize.scan import ParameterRange, run_scan

    config = _make_simple_config()
    ranges = [
        ParameterRange(path="coils.0.I", low=8000.0, high=12000.0, n=2),
        ParameterRange(path="plasma.T_i_eV", low=50.0, high=200.0, n=2),
    ]
    output_base = tmp_path / "scan_out"

    result = run_scan(config, ranges, output_base=output_base, dry_run=True)

    assert len(result.points) == 4, "2×2 grid should produce 4 scan points"
    assert len(result.metrics) == 4, "each point must have a metrics entry"

    created_dirs = sorted(output_base.iterdir())
    assert len(created_dirs) == 4, "4 point subdirectories must be created"
    expected_names = [f"point_{i:04d}" for i in range(4)]
    assert [d.name for d in created_dirs] == expected_names


# ---------------------------------------------------------------------------
# Test 3: scan with prescreening — verify screened_out flag is set
# ---------------------------------------------------------------------------

def test_scan_prescreening(tmp_path: Path) -> None:
    """Scan with prescreening enabled; verify screened_out flag behaves correctly."""
    from helicon.optimize.scan import ParameterRange, generate_scan_points, run_scan

    # Use a very high min_mirror_ratio so that at least some (likely all)
    # points with low coil current will be screened out.
    config = _make_simple_config()
    ranges = [ParameterRange(path="coils.0.I", low=100.0, high=500.0, n=3)]

    points = generate_scan_points(
        config,
        ranges,
        prescreening=True,
        min_mirror_ratio=1e6,  # impossibly high — everything screened out
    )

    assert len(points) == 3
    # All points should be screened out with the impossibly high threshold
    assert all(p.screened_out for p in points), (
        "All points should be screened out with min_mirror_ratio=1e6"
    )

    # Now run_scan with prescreening should skip WarpX and record screened_out metrics
    output_base = tmp_path / "prescreen_scan"
    result = run_scan(
        config,
        ranges,
        output_base=output_base,
        dry_run=True,
        prescreening=True,
        min_mirror_ratio=1e6,
    )

    assert result.n_screened == 3, "all 3 points should be counted as screened"
    assert all(m.get("screened_out") for m in result.metrics), (
        "all metrics entries must carry screened_out=True"
    )
    assert all("mirror_ratio" in m for m in result.metrics), (
        "mirror_ratio must be recorded for screened-out points"
    )
    # No point directories should have been created (WarpX was skipped)
    assert not output_base.exists() or not any(output_base.iterdir()), (
        "no point subdirectories should be created for screened-out points"
    )


# ---------------------------------------------------------------------------
# Test 4: validate dry-run — free_expansion case with run_simulations=False
# ---------------------------------------------------------------------------

def test_validation_dry_run(tmp_path: Path) -> None:
    """Run the free_expansion validation case with run_simulations=False.

    With run_simulations=False the runner skips WarpX and goes straight to
    evaluate(); the case returns a result (even if passed=False because no
    output data exists yet).
    """
    from helicon.validate.runner import run_validation

    output_base = tmp_path / "validation"
    report = run_validation(
        cases=["free_expansion"],
        output_base=output_base,
        run_simulations=False,
    )

    assert report.n_total == 1, "exactly one case should have been run"
    assert len(report.results) == 1
    assert report.results[0]["case_name"] == "free_expansion"
    # The validation report JSON should have been written
    assert (output_base / "validation_report.json").exists()


# ---------------------------------------------------------------------------
# Test 5: config roundtrip — save to YAML, reload, verify equality
# ---------------------------------------------------------------------------

def test_config_roundtrip(tmp_path: Path) -> None:
    """Save a SimConfig to YAML, reload it, and verify field-by-field equality."""
    original = _make_simple_config()
    yaml_path = tmp_path / "roundtrip.yaml"

    original.to_yaml(yaml_path)
    assert yaml_path.exists(), "YAML file must be written"

    reloaded = SimConfig.from_yaml(yaml_path)

    # Top-level scalar fields
    assert reloaded.timesteps == original.timesteps
    assert reloaded.dt_multiplier == pytest.approx(original.dt_multiplier)
    assert reloaded.keep_checkpoints == original.keep_checkpoints

    # Nozzle / coils
    assert len(reloaded.nozzle.coils) == len(original.nozzle.coils)
    for rc, oc in zip(reloaded.nozzle.coils, original.nozzle.coils):
        assert rc.z == pytest.approx(oc.z)
        assert rc.r == pytest.approx(oc.r)
        assert rc.I == pytest.approx(oc.I)

    # Domain
    assert reloaded.nozzle.domain.z_min == pytest.approx(original.nozzle.domain.z_min)
    assert reloaded.nozzle.domain.z_max == pytest.approx(original.nozzle.domain.z_max)
    assert reloaded.nozzle.domain.r_max == pytest.approx(original.nozzle.domain.r_max)

    # Plasma
    assert reloaded.plasma.n0 == pytest.approx(original.plasma.n0)
    assert reloaded.plasma.T_i_eV == pytest.approx(original.plasma.T_i_eV)
    assert reloaded.plasma.T_e_eV == pytest.approx(original.plasma.T_e_eV)
    assert reloaded.plasma.v_injection_ms == pytest.approx(original.plasma.v_injection_ms)
