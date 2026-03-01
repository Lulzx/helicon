# VASIMR VX-200 Validation

## Reference

Olsen, C. S., et al. (2015).
*Investigation of plasma detachment from a magnetic nozzle in the plume of the VX-200 magnetoplasma thruster.*
IEEE Transactions on Plasma Science, 43(1), 252–268.

## Physics

The Ad Astra VX-200 VASIMR thruster operates at 200 kW RF input power with
a helicon + ICRH heating stage, producing an argon or hydrogen plasma exhausted
through a magnetic nozzle. This case validates:

1. **Thrust**: within 15% of measured 5.7 N (at 200 kW)
2. **Specific impulse**: within 15% of measured ~5000 s
3. **Plume half-angle**: within ±5° of measured value

## Configuration

```python
from magnozzlex.validate.cases.vasimr_plume import VasimrPlumeCase
case = VasimrPlumeCase()
config = case.get_config()
```

Key parameters matching VX-200 operating point:

| Parameter | Value |
|-----------|-------|
| Propellant | Ar⁺ |
| $n_0$ | $5 \times 10^{19}$ m⁻³ |
| $T_i$ | 1000 eV |
| $T_e$ | 50 eV |
| $B_{throat}$ | 2.0 T |
| Power | 200 kW |

## Acceptance Criteria

- Thrust: $|F_{sim} - F_{exp}| / F_{exp} < 15\%$
- Isp: $|Isp_{sim} - Isp_{exp}| / Isp_{exp} < 15\%$
- Detachment efficiency: $\eta_{det} > 0.85$

## Running

```bash
magnozzlex validate --case vasimr
```

```python
from magnozzlex.validate.runner import run_validation
result = run_validation("vasimr", dry_run=True)
print(result.summary)
```

## Results

See `docs/validation_results/vasimr/` for thrust and Isp comparison.
