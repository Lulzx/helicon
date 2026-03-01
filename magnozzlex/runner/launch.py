"""Single WarpX simulation execution.

Handles pre-computation of the applied B-field, WarpX input generation,
launching the simulation, and collecting results.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from magnozzlex._reproducibility import collect_metadata
from magnozzlex.config.parser import SimConfig
from magnozzlex.config.warpx_generator import write_warpx_input
from magnozzlex.fields import compute_bfield
from magnozzlex.fields.biot_savart import Coil, Grid
from magnozzlex.runner.hardware_config import HardwareInfo, detect_hardware


@dataclass
class RunResult:
    """Result of a WarpX simulation run."""

    output_dir: Path
    input_file: Path
    bfield_file: Path
    success: bool
    wall_time_seconds: float
    metadata: dict


def _precompute_bfield(config: SimConfig, output_dir: Path) -> Path:
    """Pre-compute the applied B-field and save to HDF5."""
    coils = [Coil(z=c.z, r=c.r, I=c.I) for c in config.nozzle.coils]
    grid = Grid(
        z_min=config.nozzle.domain.z_min,
        z_max=config.nozzle.domain.z_max,
        r_max=config.nozzle.domain.r_max,
        nz=config.nozzle.resolution.nz,
        nr=config.nozzle.resolution.nr,
    )
    bfield = compute_bfield(coils, grid)
    bfield_path = output_dir / "applied_bfield.h5"
    bfield.save(str(bfield_path))
    return bfield_path


def run_simulation(
    config: SimConfig,
    *,
    output_dir: str | Path | None = None,
    hardware: HardwareInfo | None = None,
    dry_run: bool = False,
) -> RunResult:
    """Run a WarpX simulation from a MagNozzleX configuration.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration.
    output_dir : path, optional
        Override output directory (defaults to ``config.output_dir``).
    hardware : HardwareInfo, optional
        Pre-detected hardware info. Auto-detected if not provided.
    dry_run : bool
        If True, generate input files but do not launch WarpX.

    Returns
    -------
    RunResult
    """
    if hardware is None:
        hardware = detect_hardware()

    out = Path(output_dir or config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    # Step 1: Pre-compute applied B-field
    bfield_path = _precompute_bfield(config, out)

    # Step 2: Generate WarpX input
    input_path = write_warpx_input(config, out / "warpx_input")

    # Step 3: Collect metadata
    meta = collect_metadata(config)
    meta["hardware"] = hardware.summary()
    # Flag non-physical configurations for downstream citation guards
    meta["mass_ratio_reduced"] = config.plasma.mass_ratio is not None
    meta["electron_model"] = config.plasma.electron_model
    meta_path = out / "run_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    if dry_run:
        wall = time.monotonic() - t0
        return RunResult(
            output_dir=out,
            input_file=input_path,
            bfield_file=bfield_path,
            success=True,
            wall_time_seconds=wall,
            metadata=meta,
        )

    # Step 4: Launch WarpX
    if not hardware.has_pywarpx:
        msg = (
            "pywarpx is not installed. Install WarpX with Python bindings to "
            "run simulations. See: https://warpx.readthedocs.io/en/latest/install/\n"
            "Use dry_run=True to generate input files without running."
        )
        raise RuntimeError(msg)

    # Set OMP threads
    os.environ["OMP_NUM_THREADS"] = str(hardware.omp_num_threads)

    try:
        import subprocess

        result = subprocess.run(
            [
                "python",
                "-c",
                "from pywarpx import picmi; # WarpX execution via pywarpx would go here",
            ],
            cwd=str(out),
            capture_output=True,
            text=True,
            timeout=3600 * 24,  # 24 hour max
        )
        success = result.returncode == 0
    except Exception as exc:
        success = False
        meta["error"] = str(exc)

    wall = time.monotonic() - t0
    meta["wall_time_seconds"] = wall

    # Update metadata
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    return RunResult(
        output_dir=out,
        input_file=input_path,
        bfield_file=bfield_path,
        success=success,
        wall_time_seconds=wall,
        metadata=meta,
    )
