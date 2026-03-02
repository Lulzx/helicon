# Free Expansion Validation

## Reference

Andersen, S. A., Jensen, V. O., Nielsen, P., & D'Angelo, N. (1969).
*Continuous supersonic plasma wind tunnel.*
Physics of Fluids, 12(3), 557–560.

## Physics

A collisionless plasma expanding freely through a magnetic aperture develops a
characteristic velocity distribution in the exhaust plume. This case validates
that the PIC solver correctly reproduces:

1. Ion velocity distribution (Maxwellian → drifting beam)
2. Plume divergence angle at the Mach 1 surface
3. Density falloff with distance: $n \propto r^{-2}$ in the far field

## Configuration

```python
from helicon.validate.cases.free_expansion import FreeExpansionCase
case = FreeExpansionCase()
config = case.get_config()
```

Key parameters:

| Parameter | Value |
|-----------|-------|
| $n_0$ | $10^{18}$ m⁻³ |
| $T_i$ | 10 eV |
| $T_e$ | 10 eV |
| $B_{throat}$ | 0.1 T |
| Grid | 256 × 128 |

## Acceptance Criteria

- RMS error in ion velocity distribution < 5%
- Plume half-angle within ±2° of experimental value (28°)
- Density profile slope within 10% of $r^{-2}$

## Running

```bash
helicon validate --case free_expansion
```

```python
from helicon.validate.runner import run_validation
result = run_validation("free_expansion", dry_run=True)
print(result.summary)
```

## Results

See `docs/validation_results/free_expansion/` for field plots and comparison figures
after running `helicon validate --case free_expansion`.
