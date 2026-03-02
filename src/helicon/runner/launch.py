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

from helicon._reproducibility import collect_metadata
from helicon.config.parser import SimConfig
from helicon.config.warpx_generator import write_warpx_input
from helicon.fields import compute_bfield
from helicon.fields.biot_savart import Coil, Grid
from helicon.runner.hardware_config import HardwareInfo, detect_hardware


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
    """Run a WarpX simulation from a Helicon configuration.

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
    # Flag non-physical configurations for downstream citation guards (§14)
    mr = config.plasma.mass_ratio
    meta["mass_ratio_reduced"] = mr is not None and mr < 1836.0
    meta["electron_model"] = config.plasma.electron_model

    # Validation proximity
    try:
        from helicon.validate.proximity import config_proximity

        prox = config_proximity(config)
        meta["validation_proximity"] = {
            "nearest_case": prox.nearest_case,
            "distance": prox.distance,
            "in_validated_region": prox.in_validated_region,
            "warning": prox.warning,
        }
    except Exception:
        meta["validation_proximity"] = None

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
    # Try Metal backend first (Apple Silicon native GPU via SYCL/AdaptiveCpp).
    # The warpx-metal build is warpx.2d (2D Cartesian, single precision).
    # Skip if the inputs declare RZ geometry or openPMD diagnostics.
    _inputs_text = input_path.read_text()
    _metal_compatible = (
        "geometry.dims = RZ" not in _inputs_text
        and "format = openpmd" not in _inputs_text
    )
    if hardware.has_warpx_metal and _metal_compatible:
        from helicon.runner.metal_runner import detect_warpx_metal, run_warpx_metal

        metal_info = detect_warpx_metal()
        if metal_info.valid and metal_info.exe_2d is not None:
            os.environ["OMP_NUM_THREADS"] = str(hardware.omp_num_threads)
            metal_result = run_warpx_metal(
                metal_info=metal_info,
                output_dir=out,
                inputs_content=input_path.read_text(),
                timeout_s=3600 * 24,
            )
            wall = time.monotonic() - t0
            meta["wall_time_seconds"] = wall
            meta["backend"] = "metal"
            meta["warpx_returncode"] = metal_result.exit_code
            if not metal_result.success and metal_result.error:
                meta["error"] = metal_result.error
            meta_path.write_text(json.dumps(meta, indent=2, default=str))
            return RunResult(
                output_dir=out,
                input_file=input_path,
                bfield_file=bfield_path,
                success=metal_result.success,
                wall_time_seconds=wall,
                metadata=meta,
            )

    if not hardware.has_pywarpx:
        msg = (
            "pywarpx is not installed. Install WarpX with Python bindings to "
            "run simulations. See: https://warpx.readthedocs.io/en/latest/install/\n"
            "Use dry_run=True to generate input files without running."
        )
        raise RuntimeError(msg)

    # Set OMP threads
    os.environ["OMP_NUM_THREADS"] = str(hardware.omp_num_threads)

    import shutil
    import subprocess
    import sys

    # Build WarpX command: python -m pywarpx.WarpX <input_file>
    cmd = [sys.executable, "-m", "pywarpx.WarpX", str(input_path)]

    # On Linux with NVIDIA GPU, try mpirun for multi-rank execution
    if hardware.platform == "linux" and hardware.has_nvidia_gpu:
        mpirun = shutil.which("mpirun") or shutil.which("mpiexec")
        if mpirun:
            n_ranks = max(1, hardware.cpu_count // 4)  # 1 rank per 4 CPU threads
            cmd = [mpirun, "-n", str(n_ranks), *cmd]

    log_path = out / "warpx.log"
    success = False
    try:
        with log_path.open("w") as log_fh:
            proc = subprocess.run(
                cmd,
                cwd=str(out),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=3600 * 24,  # 24-hour max
            )
        success = proc.returncode == 0
        if not success:
            meta["warpx_returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        meta["error"] = "WarpX simulation timed out after 24 hours"
    except Exception as exc:
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
