# Configuration Reference

Helicon uses YAML configuration files validated with Pydantic.

## Full Example

```yaml
nozzle:
  type: converging_diverging   # solenoid | converging_diverging | frc_exhaust
  coils:
    - z: -0.05      # axial position [m]
      r: 0.12       # coil radius [m]
      I: 50000      # current [A-turns]
    - z: 0.05
      r: 0.12
      I: 25000
  domain:
    z_min: -0.5     # upstream boundary [m]
    z_max: 2.0      # downstream boundary [m]
    r_max: 0.8      # radial extent [m]
  resolution:
    nz: 512         # axial grid points
    nr: 256         # radial grid points

plasma:
  species: ["D+", "e-"]
  n0: 1.0e19        # number density [m^-3]
  T_i_eV: 5000.0    # ion temperature [eV]
  T_e_eV: 2000.0    # electron temperature [eV]
  v_injection_ms: 200000.0   # injection velocity [m/s]
  mass_ratio: null  # null = physical; e.g. 100 for reduced ratio
  electron_model: kinetic   # kinetic | fluid

diagnostics:
  mode: analysis    # analysis (full dumps) | scan (reduced, for parameter scans)
  field_dump_interval: 500
  particle_dump_interval: 5000

timesteps: 50000
dt_multiplier: 0.95
keep_checkpoints: false
random_seed: null   # null = derived from config hash
output_dir: results/my_run
```

## Loading from Python

```python
from helicon.config.parser import SimConfig

# From YAML file
config = SimConfig.from_yaml("my_nozzle.yaml")

# From a preset
config = SimConfig.from_preset("dfd")   # sunbird | dfd | ppr

# Programmatically
from helicon.config.parser import (
    CoilConfig, DomainConfig, NozzleConfig,
    PlasmaSourceConfig, ResolutionConfig, SimConfig,
)
config = SimConfig(
    nozzle=NozzleConfig(
        type="converging_diverging",
        coils=[CoilConfig(z=0.0, r=0.12, I=50000.0)],
        domain=DomainConfig(z_min=-0.3, z_max=2.0, r_max=0.5),
        resolution=ResolutionConfig(nz=512, nr=256),
    ),
    plasma=PlasmaSourceConfig(
        species=["D+", "e-"],
        n0=1e19, T_i_eV=5000, T_e_eV=2000,
        v_injection_ms=200000,
    ),
    timesteps=50000,
    output_dir="results/my_run",
)
```

## Preset Configurations

| Preset | Engine | Ion | T_i (eV) | B_throat (T) |
|---|---|---|---|---|
| `sunbird` | Pulsar Sunbird (DDFD) | D⁺ | 80 keV | 3.0 |
| `dfd` | Princeton DFD | D⁺/He³⁺ | 50 keV | 2.5 |
| `ppr` | Howe PPR | p⁺ | 100 keV | 4.0 |

## Physics Validation

`helicon.config.validators.validate_config(config)` checks:

- Domain extends at least 5× coil radius downstream (detachment clearance)
- Grid resolution > 10 cells per ion Larmor radius
- Plasma β < 1 at throat (confinement requirement)
- Timesteps > 10 plasma transit times (statistical convergence)
