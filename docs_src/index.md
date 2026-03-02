# Helicon

**GPU-Accelerated Magnetic Nozzle Simulation & Detachment Analysis Toolkit for Fusion Propulsion**

Helicon wraps and extends [WarpX](https://ecp-warpx.github.io/) with:

- Curated input configurations for magnetic nozzle geometries (solenoid, converging-diverging, FRC exhaust)
- Post-processing pipelines that extract thrust, Isp, detachment efficiency, and plume divergence from PIC output
- Parameter scan and optimization infrastructure (Bayesian, gradient-based via MLX, Sobol sensitivity)
- Validation cases against published analytical solutions and experimental data (VASIMR VX-200, Merino-Ahedo)

## Quick Install

```bash
pip install helicon
# For Apple Silicon GPU acceleration:
pip install helicon[mlx]
# For Bayesian optimization:
pip install helicon[optimize]
```

## Quick Start

```python
from helicon.fields.biot_savart import Coil, Grid, compute_bfield
from helicon.optimize.analytical import screen_geometry

coils = [Coil(z=0.0, r=0.12, I=50000.0)]
result = screen_geometry(coils, z_min=-0.3, z_max=2.0)
print(f"Mirror ratio R_B = {result.mirror_ratio:.2f}")
print(f"Thrust efficiency η_T = {result.thrust_efficiency:.3f}")
print(f"Plume half-angle θ = {result.divergence_half_angle_deg:.1f}°")
```

## Why Helicon?

**The unsolved problem:** Plasma detachment in magnetic nozzles is where fusion thrust is won or lost.
Current detachment efficiency estimates span 50–95% — a factor-of-two uncertainty that determines
whether direct-fusion-drive missions to Mars are 90 days or 9 months.

**What makes Helicon different:**

| Tool | Gap |
|---|---|
| WarpX | No propulsion-specific postprocessing |
| MN1D (Ahedo group) | 1D only, not public |
| BALOO (Merino) | Not publicly available |
| PlasmaPy | No PIC integration |

Helicon is open-source, GPU-accelerated (MLX on Apple Silicon, CUDA via WarpX), and
validation-first.

## Citing Helicon

See [CITATION.cff](https://github.com/helicon/helicon/blob/main/CITATION.cff) or cite as:

```bibtex
@software{helicon,
  title = {Helicon: GPU-Accelerated Magnetic Nozzle Simulation Toolkit},
  version = {0.4.0},
  license = {MIT},
}
```
