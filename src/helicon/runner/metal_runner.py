"""WarpX Metal GPU backend — Apple Silicon M-series integration.

Discovers the warpx-metal build (github.com/lulzx/warpx-metal), generates
WarpX input files, launches the SYCL/Metal native executable, and parses
AMReX plotfile diagnostics.

Build chain (established by warpx-metal project):
    WarpX → AMReX (SYCL) → AdaptiveCpp SSCP → Metal → Apple GPU

Executables (single precision, no MPI, no FFT/PSATD):
    warpx.2d.NOMPI.SYCL.SP.PSP.EB
    warpx.3d.NOMPI.SYCL.SP.PSP.EB
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXE_2D = "warpx.2d.NOMPI.SYCL.SP.PSP.EB"
_EXE_3D = "warpx.3d.NOMPI.SYCL.SP.PSP.EB"
_BUILD_SUBPATH = "extern/warpx/build-acpp/bin"
_ACPP_BIN = "opt/adaptivecpp/bin"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass
class WarpXMetalInfo:
    """Paths and availability for a warpx-metal build."""

    root: Path
    exe_2d: Path | None
    exe_3d: Path | None
    acpp_bin: Path | None
    valid: bool

    def summary(self) -> str:
        lines = [f"warpx-metal root: {self.root}"]
        lines.append(f"  2D exe: {self.exe_2d or 'not found'}")
        lines.append(f"  3D exe: {self.exe_3d or 'not found'}")
        lines.append(f"  acpp:   {self.acpp_bin or 'not found'}")
        lines.append(f"  valid:  {self.valid}")
        return "\n".join(lines)


def find_warpx_metal_root(hint: str | Path | None = None) -> Path | None:
    """Search for the warpx-metal build directory.

    Search order:
    1. ``hint`` parameter (if provided)
    2. ``WARPX_METAL_ROOT`` environment variable
    3. Sibling directory ``../warpx-metal`` relative to this module
    4. ``~/work/warpx-metal``
    """
    candidates: list[Path] = []

    if hint is not None:
        candidates.append(Path(hint))

    env_root = os.environ.get("WARPX_METAL_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    # Sibling to Helicon source tree
    _here = Path(__file__).resolve()
    for _ in range(5):
        _here = _here.parent
        sib = _here.parent / "warpx-metal"
        candidates.append(sib)
        if _here.name in ("helicon", "Helicon", "src"):
            break

    candidates.append(Path.home() / "work" / "warpx-metal")

    for candidate in candidates:
        if _has_metal_exe(candidate):
            return candidate.resolve()
    return None


def _has_metal_exe(root: Path) -> bool:
    """Return True if at least one Metal WarpX executable exists under root."""
    for exe_name in (_EXE_2D, _EXE_3D):
        exe = root / _BUILD_SUBPATH / exe_name
        if exe.exists() and os.access(exe, os.X_OK):
            return True
    return False


def detect_warpx_metal(hint: str | Path | None = None) -> WarpXMetalInfo:
    """Detect a warpx-metal build and return its paths.

    Parameters
    ----------
    hint:
        Override search path. If None, uses :func:`find_warpx_metal_root`.

    Returns
    -------
    WarpXMetalInfo
        ``valid=True`` if at least one executable was found.
    """
    root = find_warpx_metal_root(hint)
    if root is None:
        return WarpXMetalInfo(
            root=Path("."),
            exe_2d=None, exe_3d=None, acpp_bin=None, valid=False,
        )

    def _find_exe(name: str) -> Path | None:
        p = root / _BUILD_SUBPATH / name
        return p if (p.exists() and os.access(p, os.X_OK)) else None

    exe_2d = _find_exe(_EXE_2D)
    exe_3d = _find_exe(_EXE_3D)

    acpp_dir = root / _ACPP_BIN
    acpp_bin = (acpp_dir / "acpp") if (acpp_dir / "acpp").exists() else None

    return WarpXMetalInfo(
        root=root,
        exe_2d=exe_2d,
        exe_3d=exe_3d,
        acpp_bin=acpp_bin,
        valid=(exe_2d is not None or exe_3d is not None),
    )


# ---------------------------------------------------------------------------
# AMReX plotfile parser
# ---------------------------------------------------------------------------

@dataclass
class WarpXMetalDiag:
    """Parsed AMReX/WarpX diagnostic snapshot.

    Extracted from the ``Header`` and ``WarpXHeader`` files written to
    each ``diags/diag<step>/`` directory.
    """

    diag_dir: Path
    step: int
    time_s: float
    field_vars: list[str]
    n_cells: tuple[int, ...]
    domain_lo: tuple[float, ...]
    domain_hi: tuple[float, ...]
    species: list[str] = field(default_factory=list)

    @classmethod
    def from_dir(cls, diag_dir: str | Path) -> WarpXMetalDiag:
        """Parse a ``diags/diag<step>/`` directory.

        Parameters
        ----------
        diag_dir:
            Path to the AMReX output directory (contains ``Header`` file).

        Returns
        -------
        WarpXMetalDiag
        """
        diag_dir = Path(diag_dir)
        header_path = diag_dir / "Header"
        if not header_path.exists():
            msg = f"AMReX Header not found: {header_path}"
            raise FileNotFoundError(msg)

        lines = header_path.read_text().splitlines()
        idx = 0

        # Line 0: version ("HyperCLaw-V1.1")
        idx += 1

        # Line 1: n_components
        n_comp = int(lines[idx])
        idx += 1

        # Lines 2..2+n_comp: component names
        field_vars = []
        for _ in range(n_comp):
            field_vars.append(lines[idx].strip())
            idx += 1

        # n_levels
        _n_levels = int(lines[idx])
        idx += 1

        # time
        time_s = float(lines[idx])
        idx += 1

        # finest level
        idx += 1  # skip

        # prob_lo, prob_hi
        lo_parts = lines[idx].split()
        idx += 1
        hi_parts = lines[idx].split()
        idx += 1
        domain_lo = tuple(float(x) for x in lo_parts)
        domain_hi = tuple(float(x) for x in hi_parts)

        # Skip blank line
        while idx < len(lines) and not lines[idx].strip():
            idx += 1

        # Domain box: ((0,0) (nx-1,ny-1) (0,0))
        box_line = lines[idx] if idx < len(lines) else ""
        idx += 1
        n_cells: tuple[int, ...] = ()
        if box_line:
            import re
            nums = re.findall(r"\d+", box_line)
            # AMReX box format: ((lo_x,lo_y) (hi_x,hi_y) (type_x,type_y))
            # nums has 3*ndim values: lo coords, then hi coords, then cell types
            if len(nums) >= 4 and len(nums) % 3 == 0:
                ndim = len(nums) // 3
                n_cells = tuple(
                    int(nums[ndim + i]) - int(nums[i]) + 1 for i in range(ndim)
                )

        # Infer step from directory name (diag<step>)
        stem = diag_dir.name  # e.g. "diag1000004"
        digits = "".join(c for c in stem if c.isdigit())
        step = int(digits) if digits else 0

        # Detect species from particle subdirectories (each has its own Header)
        species: list[str] = [
            d.name
            for d in sorted(diag_dir.iterdir())
            if d.is_dir() and (d / "Header").exists() and d.name != "Level_0"
        ]

        return cls(
            diag_dir=diag_dir,
            step=step,
            time_s=time_s,
            field_vars=field_vars,
            n_cells=n_cells,
            domain_lo=domain_lo,
            domain_hi=domain_hi,
            species=species,
        )

    def summary(self) -> str:
        cells = " × ".join(str(n) for n in self.n_cells)
        lo = ", ".join(f"{v:.3e}" for v in self.domain_lo)
        hi = ", ".join(f"{v:.3e}" for v in self.domain_hi)
        return (
            f"step={self.step}  t={self.time_s:.4e} s\n"
            f"  grid: {cells} cells\n"
            f"  domain: [{lo}] → [{hi}] m\n"
            f"  fields: {', '.join(self.field_vars)}\n"
            f"  species: {', '.join(self.species) or '(none parsed)'}"
        )


def find_diag_dirs(output_dir: str | Path) -> list[WarpXMetalDiag]:
    """Find and parse all AMReX diagnostic directories under *output_dir*.

    Looks for any subdirectory named ``diag*`` that contains a ``Header``
    file, parses each, and returns them sorted by step number.
    """
    output_dir = Path(output_dir)
    diags = []
    for d in sorted(output_dir.glob("diag*")):
        if (d / "Header").exists():
            with contextlib.suppress(Exception):
                diags.append(WarpXMetalDiag.from_dir(d))
    return sorted(diags, key=lambda x: x.step)


# ---------------------------------------------------------------------------
# Input file generation
# ---------------------------------------------------------------------------

def generate_metal_inputs(
    *,
    n_cell: int = 128,
    max_step: int = 4,
    n_ppc: int = 2,
    density: float = 2e24,
    domain_m: float = 20e-6,
    cfl: float = 1.0,
    diag_interval: int = 4,
    extra: dict[str, Any] | None = None,
) -> str:
    """Generate a WarpX input file for the Metal backend.

    Produces a 2D pair-plasma (electrons + positrons) FDTD simulation,
    matching the configuration validated by the warpx-metal project.

    Parameters
    ----------
    n_cell:
        Grid cells per dimension (same in x and z for 2D).
    max_step:
        Number of PIC timesteps.
    n_ppc:
        Particles per cell per dimension (n_ppc² total per species per cell).
    density:
        Plasma number density [m⁻³].
    domain_m:
        Half-width of the physical domain [m].
    cfl:
        CFL number for the FDTD timestep.
    diag_interval:
        Diagnostic output every N steps (use ``max_step + 1`` to suppress).
    extra:
        Additional key=value pairs appended verbatim.

    Returns
    -------
    str
        WarpX inputs file content.
    """
    domain = domain_m
    lines = [
        "# Helicon-generated WarpX Metal inputs",
        "# 2D pair plasma, FDTD, single precision",
        "",
        f"max_step = {max_step}",
        f"amr.n_cell = {n_cell} {n_cell}",
        "amr.max_level = 0",
        f"amr.max_grid_size = {n_cell}",
        "",
        "geometry.dims = 2",
        "geometry.coord_sys = 0",
        "geometry.is_periodic = 1 1",
        f"geometry.prob_lo = -{domain:.15e} -{domain:.15e}",
        f"geometry.prob_hi =  {domain:.15e}  {domain:.15e}",
        "",
        "boundary.field_lo = periodic periodic",
        "boundary.field_hi = periodic periodic",
        "",
        "algo.current_deposition = direct",
        "algo.field_gathering = energy-conserving",
        "algo.particle_shape = 1",
        f"warpx.cfl = {cfl}",
        "warpx.sort_intervals = 1",
        "warpx.use_filter = 0",
        "",
        "particles.species_names = electrons positrons",
        "particles.do_mem_efficient_sort = 1",
        "particles.do_tiling = 0",
        "",
        "electrons.charge = -q_e",
        "electrons.mass = m_e",
        "electrons.injection_style = NUniformPerCell",
        f"electrons.num_particles_per_cell_each_dim = {n_ppc} {n_ppc}",
        f"electrons.density = {density:.3e}",
        "electrons.profile = constant",
        f"electrons.xmin = -{domain:.15e}",
        f"electrons.xmax =  {domain:.15e}",
        f"electrons.zmin = -{domain:.15e}",
        f"electrons.zmax =  {domain:.15e}",
        "electrons.momentum_distribution_type = constant",
        "electrons.ux = 0.01",
        "electrons.uy = 0.0",
        "electrons.uz = 0.01",
        "",
        "positrons.charge = q_e",
        "positrons.mass = m_e",
        "positrons.injection_style = NUniformPerCell",
        f"positrons.num_particles_per_cell_each_dim = {n_ppc} {n_ppc}",
        f"positrons.density = {density:.3e}",
        "positrons.profile = constant",
        f"positrons.xmin = -{domain:.15e}",
        f"positrons.xmax =  {domain:.15e}",
        f"positrons.zmin = -{domain:.15e}",
        f"positrons.zmax =  {domain:.15e}",
        "positrons.momentum_distribution_type = constant",
        "positrons.ux = -0.01",
        "positrons.uy = 0.0",
        "positrons.uz = -0.01",
        "",
        "diagnostics.diags_names = diag1",
        "diag1.diag_type = Full",
        "diag1.fields_to_plot = Ex Ey Bz",
        "diag1.electrons.variables = x z w ux uy uz",
        "diag1.positrons.variables = x z w ux uy uz",
        f"diag1.intervals = {diag_interval}",
        "",
        "amrex.abort_on_out_of_gpu_memory = 1",
        "amrex.verbose = 1",
        "tiny_profiler.enabled = 1",
        "tiny_profiler.device_synchronize_around_region = 1",
        "tiny_profiler.memprof_enabled = 1",
    ]

    if extra:
        lines.append("")
        lines.append("# Extra parameters")
        for k, v in extra.items():
            lines.append(f"{k} = {v}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class MetalRunResult:
    """Result of a warpx-metal native GPU simulation."""

    success: bool
    exit_code: int
    output_dir: Path
    log_path: Path
    wall_time_s: float
    steps_completed: int
    diags: list[WarpXMetalDiag]
    error: str | None = None

    def summary(self) -> str:
        status = "OK" if self.success else "FAILED"
        lines = [
            f"[{status}] Metal WarpX run",
            f"  steps:     {self.steps_completed}",
            f"  wall time: {self.wall_time_s:.2f} s",
            f"  diags:     {len(self.diags)} snapshots",
        ]
        if self.error:
            lines.append(f"  error:     {self.error}")
        return "\n".join(lines)


def run_warpx_metal(
    *,
    metal_info: WarpXMetalInfo | None = None,
    output_dir: str | Path,
    inputs_content: str | None = None,
    n_cell: int = 128,
    max_step: int = 4,
    n_ppc: int = 2,
    density: float = 2e24,
    diag_interval: int = 4,
    timeout_s: float = 3600.0,
    extra_params: dict[str, Any] | None = None,
) -> MetalRunResult:
    """Launch WarpX on Apple Silicon GPU via the Metal SYCL backend.

    Generates a WarpX input file (or uses *inputs_content* directly),
    launches the warpx-metal 2D executable, and returns a
    :class:`MetalRunResult` with parsed diagnostics.

    Parameters
    ----------
    metal_info:
        Pre-detected :class:`WarpXMetalInfo`. Auto-detected if None.
    output_dir:
        Directory to write inputs, logs, and diagnostics.
    inputs_content:
        Raw WarpX inputs file text. If None, generated from other params.
    n_cell, max_step, n_ppc, density, diag_interval:
        Physics parameters passed to :func:`generate_metal_inputs`.
    timeout_s:
        Maximum run time before killing the process.
    extra_params:
        Extra ``key = value`` lines appended to the inputs file.

    Returns
    -------
    MetalRunResult
    """
    if metal_info is None:
        metal_info = detect_warpx_metal()

    if not metal_info.valid or metal_info.exe_2d is None:
        return MetalRunResult(
            success=False, exit_code=-1,
            output_dir=Path(output_dir),
            log_path=Path(output_dir) / "warpx_metal.log",
            wall_time_s=0.0, steps_completed=0, diags=[],
            error="warpx-metal 2D executable not found — run warpx-metal build scripts first",
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Write inputs file
    if inputs_content is None:
        inputs_content = generate_metal_inputs(
            n_cell=n_cell,
            max_step=max_step,
            n_ppc=n_ppc,
            density=density,
            diag_interval=diag_interval,
            extra=extra_params,
        )
    inputs_path = out / "inputs"
    inputs_path.write_text(inputs_content)

    # Environment: add acpp bin to PATH if present
    env = os.environ.copy()
    if metal_info.acpp_bin is not None:
        acpp_dir = str(metal_info.acpp_bin.parent)
        path_parts = env.get("PATH", "").split(":")
        if acpp_dir not in path_parts:
            env["PATH"] = acpp_dir + ":" + env.get("PATH", "")

    log_path = out / "warpx_metal.log"
    t0 = time.monotonic()
    exit_code = -1
    steps_completed = 0
    error: str | None = None

    try:
        with log_path.open("w") as log_fh:
            proc = subprocess.run(
                [str(metal_info.exe_2d), str(inputs_path)],
                cwd=str(out),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        error = f"WarpX Metal timed out after {timeout_s:.0f} s"
        exit_code = -2
    except Exception as exc:
        error = str(exc)
        exit_code = -3

    wall_time_s = time.monotonic() - t0

    # Count steps from log
    if log_path.exists():
        log_text = log_path.read_text()
        import re
        matches = re.findall(r"STEP\s+(\d+)", log_text)
        if matches:
            steps_completed = int(matches[-1])

    # Collect diagnostics
    diags = find_diag_dirs(out)
    if not diags:
        # Also check default diags/ subdir
        diags = find_diag_dirs(out / "diags")

    success = (exit_code == 0)
    return MetalRunResult(
        success=success,
        exit_code=exit_code,
        output_dir=out,
        log_path=log_path,
        wall_time_s=wall_time_s,
        steps_completed=steps_completed,
        diags=diags,
        error=error,
    )
