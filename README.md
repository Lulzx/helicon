# Helicon

GPU-Accelerated Magnetic Nozzle Simulation & Detachment Analysis Toolkit for Fusion Propulsion.

Helicon wraps [WarpX](https://github.com/ECP-WarpX/WarpX) with curated configurations,
post-processing pipelines, detachment physics, and optimization infrastructure for magnetic
nozzle research. Runs PIC simulations on Apple Silicon Metal GPU, NVIDIA CUDA, or CPU.

## Installation

```bash
# With uv (recommended)
uv sync

# With pip
pip install -e .

# With MLX support (Apple Silicon GPU field solver)
pip install -e ".[mlx]"

# Development install
uv sync --group dev
```

Requires Python 3.11+. WarpX simulation execution requires one of:
- **pywarpx** — [WarpX with Python bindings](https://warpx.readthedocs.io/en/latest/install/)
- **warpx-metal** — native Apple Silicon GPU build (see below)

## Quick Start

### Compute applied B-field from coil geometry

```python
from helicon.fields import Coil, Grid, compute_bfield

coils = [
    Coil(z=0.0, r=0.15, I=50000),
    Coil(z=0.3, r=0.25, I=30000),
]
grid = Grid(z_min=-0.5, z_max=2.0, r_max=0.8, nz=512, nr=256)

bfield = compute_bfield(coils, grid)
bfield.save("applied_bfield.h5")
```

### Run a preset simulation

```bash
# Generate WarpX input (dry run, no WarpX required)
helicon run --preset sunbird --dry-run

# Run simulation (auto-selects Metal/CUDA/OMP backend)
helicon run --preset sunbird --output results/

# Post-process output
helicon postprocess --input results/ --metrics thrust
```

### Detachment analysis

```bash
# Assess detachment state
helicon detach assess --n 1e18 --Te 30 --Ti 15 --B 0.05 --dBdz -1.0 --vz 40000

# JSON output with control recommendation
helicon detach assess --n 1e18 --Te 30 --Ti 15 --B 0.05 --dBdz -1.0 --vz 40000 \
    --control --json

# Invert thrust measurement to plasma state
helicon detach invert --F 0.05 --mdot 1e-5 --B 0.03 --area 5e-4 --json

# Calibrate detachment score weights from synthetic data
helicon detach calibrate --n-samples 200 --seed 42 --json

# Simulate Lyapunov-stable feedback controller
helicon detach simulate --n 1e18 --Te 30 --Ti 15 --B 0.05 --dBdz -1.0 \
    --vz 40000 --steps 20 --json

# Full multi-criterion detachment report (MHD, kinetic, sheath, Lyapunov)
helicon detach report --n 1e18 --Te 30 --Ti 15 --B 0.05 --dBdz -1.0 \
    --vz 40000 --json
```

### Environment check

```bash
helicon doctor
helicon doctor --json
```

## Apple Silicon Metal GPU (warpx-metal)

Helicon integrates [warpx-metal](https://github.com/lulzx/warpx-metal), a native Apple GPU
build of WarpX using the SYCL → AdaptiveCpp → Metal stack.

**Build chain:**
```
WarpX → AMReX (SYCL) → AdaptiveCpp SSCP → Metal → Apple GPU
```

Build the warpx-metal executable once (requires macOS 14+, Xcode 16, Apple Silicon):

```bash
git clone https://github.com/lulzx/warpx-metal ../warpx-metal
cd ../warpx-metal
./scripts/00-install-deps.sh
./scripts/01-build-adaptivecpp.sh
./scripts/05-build-warpx.sh
```

Helicon auto-detects the build at `../warpx-metal` (sibling directory) or `~/work/warpx-metal`.
Override with the `WARPX_METAL_ROOT` environment variable.

Run a 2D PIC simulation directly on the Metal GPU:

```python
from helicon.runner.metal_runner import detect_warpx_metal, run_warpx_metal

metal = detect_warpx_metal()
print(metal.summary())

result = run_warpx_metal(
    metal_info=metal,
    output_dir="metal_run/",
    n_cell=128,
    max_step=4,
)
print(result.summary())
for diag in result.diags:
    print(diag.summary())
```

**Limitations:** single precision (no FP64), FDTD solver only (no PSATD), 2D runs via
`warpx.2d.NOMPI.SYCL.SP.PSP.EB`. The first run is slow due to LLVM IR → Metal shader
JIT compilation; subsequent runs use `~/.acpp/apps/global/jit-cache/`.

## Architecture

| Module | Description |
|--------|-------------|
| `helicon.config` | YAML parsing, Pydantic validation, WarpX input generation |
| `helicon.fields` | Biot-Savart solver (MLX + NumPy backends), field-line tracing, FRC topology |
| `helicon.runner` | WarpX orchestration, hardware detection, Metal runner, batch/convergence |
| `helicon.postprocess` | Thrust, Isp, detachment scoring, plume, species, PPR |
| `helicon.detach` | MHD detachment model, kinetic FLR corrections, sheath coupling, BCE calibration, thrust inversion, Lyapunov controller |
| `helicon.optimize` | Parameter scan, Bayesian optimization, multi-fidelity pipeline, Pareto/Sobol |
| `helicon.surrogate` | MLX neural MLP surrogate, Monte Carlo UQ |
| `helicon.validate` | Validation cases, regression suite, proximity checking |
| `helicon.mission` | Throttle maps, trajectory analysis, pulsed profiles |
| `helicon.cloud` | Cloud HPC backends (local/Lambda/AWS) |
| `helicon.plugins` | Plugin registry, entry-point auto-discovery |
| `helicon.multithruster` | Multi-thruster arrays, plume-plume interaction |
| `helicon.valdb` | Validation database (JSON-lines), query/export |
| `helicon.widgets` | Jupyter ipywidgets for field topology and coil editing |
| `helicon.export` | STEP/IGES CAD export for coil geometry |
| `helicon.app` | Streamlit interactive design explorer |
| `helicon.provenance` | JSON-lines audit trail, lineage graph |

## Backends

| Backend | Hardware | Notes |
|---------|----------|-------|
| NumPy/SciPy | CPU | Exact elliptic integrals, always available |
| MLX | Apple Silicon GPU | Fast B-field solver, differentiable (`mx.grad`) |
| warpx-metal (SYCL/Metal) | Apple Silicon GPU | Full PIC — WarpX on Metal, single precision |
| pywarpx (OMP) | CPU | WarpX Python bindings, multi-core |
| pywarpx (CUDA) | NVIDIA GPU | WarpX Python bindings, double precision |

`helicon doctor` reports which backends are available on your machine.

## Testing

```bash
uv run pytest tests/ -q
uv run ruff check src/ tests/
```

1163 tests, 0 skipped.

## License

MIT
