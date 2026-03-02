# Helicon

GPU-Accelerated Magnetic Nozzle Simulation & Detachment Analysis Toolkit for Fusion Propulsion.

Helicon wraps [WarpX](https://github.com/ECP-WarpX/WarpX) with curated configurations, post-processing pipelines, and optimization infrastructure for magnetic nozzle research.

## Installation

```bash
# Basic install (NumPy/SciPy backend)
pip install -e .

# With MLX support (Apple Silicon GPU acceleration)
pip install -e ".[mlx]"

# Development install
pip install -e ".[dev,mlx]"
```

Requires Python 3.11+. For WarpX simulation execution, install [WarpX with pywarpx](https://warpx.readthedocs.io/en/latest/install/).

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

# Run simulation (requires WarpX)
helicon run --preset sunbird --output results/

# Post-process output
helicon postprocess --input results/ --metrics thrust
```

### Use a custom configuration

```yaml
# my_nozzle.yaml
nozzle:
  type: converging_diverging
  coils:
    - z: 0.0
      r: 0.15
      I: 50000
    - z: 0.3
      r: 0.25
      I: 30000
  domain:
    z_min: -0.5
    z_max: 2.0
    r_max: 0.8

plasma:
  species: ["D+", "e-"]
  n0: 1.0e19
  T_i_eV: 5000.0
  T_e_eV: 2000.0
  v_injection_ms: 200000.0
```

```bash
helicon run --config my_nozzle.yaml
```

## Architecture

- **helicon.config** — YAML parsing, Pydantic validation, WarpX input generation
- **helicon.fields** — Biot-Savart solver (MLX + NumPy backends)
- **helicon.runner** — WarpX orchestration, hardware detection
- **helicon.postprocess** — Thrust, Isp, moment computation from openPMD output
- **helicon.validate** — Automated validation against analytic solutions

## Backends

| Backend | Compute | Differentiable | Use Case |
|---------|---------|---------------|----------|
| NumPy/SciPy | CPU | No | Exact elliptic integrals, fallback |
| MLX | Apple Silicon GPU | Yes (`mx.grad`) | Fast B-field, coil optimization |

## Testing

```bash
uv run pytest tests/ -v
ruff check helicon/ tests/
ty check
```

## License

MIT
