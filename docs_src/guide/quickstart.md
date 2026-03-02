# Quick Start

See also: [notebooks/01_quickstart.ipynb](https://github.com/lulzx/helicon/blob/main/notebooks/01_quickstart.ipynb)

## 1. Compute a magnetic field

```python
from helicon.fields.biot_savart import Coil, Grid, compute_bfield

coils = [Coil(z=0.0, r=0.12, I=50000.0)]
grid = Grid(z_min=-0.3, z_max=2.0, r_max=0.5, nz=256, nr=128)
bfield = compute_bfield(coils, grid)
bfield.save("bfield.h5")

print(f"B_z max: {bfield.Bz.max():.3f} T")
```

## 2. Analytical pre-screening

```python
from helicon.optimize.analytical import screen_geometry

result = screen_geometry(coils, z_min=-0.3, z_max=2.0)
print(f"Mirror ratio R_B = {result.mirror_ratio:.2f}")
print(f"Thrust efficiency η_T = {result.thrust_efficiency:.3f}")
print(f"Plume half-angle θ = {result.divergence_half_angle_deg:.1f}°")
```

## 3. Load a preset configuration

```python
from helicon.config.parser import SimConfig

config = SimConfig.from_preset("dfd")  # Princeton DFD reference case
print(config.model_dump_json(indent=2))
```

## 4. Run a simulation (dry run)

```python
from helicon.runner.launch import run_simulation

result = run_simulation(config, output_dir="results/dfd", dry_run=True)
print(f"Input file: {result.input_file}")
print(f"B-field file: {result.bfield_file}")
```

## 5. Run the validation suite

```bash
helicon validate --all
```

## 6. Parameter scan

```bash
helicon scan --config my_nozzle.yaml \
    --vary "coils.0.I:20000:80000:5" \
    --vary "coils.0.r:0.05:0.20:5" \
    --output scan_results/ \
    --method lhc
```
