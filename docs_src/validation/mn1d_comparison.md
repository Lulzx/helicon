# MN1D Benchmark Comparison

## Reference

Ahedo, E. & Merino, M. (2010). Two-dimensional supersonic plasma acceleration
in a magnetic nozzle. *Physics of Plasmas*, 17(7), 073501.

## Physics

MN1D is the 1D magnetic nozzle code from the Ahedo group. It solves the
steady-state 1D ion fluid equations along the field-line coordinate. This
validation case checks that Helicon's 2D-RZ PIC results, when radially
averaged, reproduce the 1D MN1D results for:

1. Axial ion velocity profile $v_z(z)$
2. Density profile $n(z)$ along the nozzle axis
3. Thrust efficiency as a function of plasma $\beta$

Three $\beta$ values are tested to span the low-, mid-, and high-$\beta$ regimes:

| Case | $\beta$ | Physics regime |
|------|---------|----------------|
| Low-β | 0.01 | Ion-dominated, field barely perturbed |
| Mid-β | 0.1  | Coupled ion-electron dynamics |
| High-β | 0.5 | Near-detachment; field line inflation |

## Configuration

```python
from helicon.validate.cases.mn1d_comparison import MN1DComparisonCase
case = MN1DComparisonCase(beta=0.1)
config = case.get_config()
```

Key parameters (mid-β case):

| Parameter | Value |
|-----------|-------|
| $n_0$ | $10^{18}$ m⁻³ |
| $T_i$ | 100 eV |
| $T_e$ | 100 eV |
| $B_{throat}$ | 0.15 T |
| Grid | 512 × 256 |
| Timesteps | 20 000 |

## Acceptance Criteria

- Radially-averaged axial velocity $\langle v_z \rangle(z)$ matches MN1D to < 5% RMS
- Density profile $\langle n \rangle(z)$ matches MN1D to < 5% RMS
- Thrust efficiency within 10% of MN1D value at all three β values

## Running

```bash
helicon validate --case mn1d_comparison
```

```python
from helicon.validate.runner import run_validation

report = run_validation(cases=["mn1d_comparison"], run_simulations=False)
print(report.summary())
```

## Reference Data

Digitized MN1D profiles are stored under `helicon/validate/reference_data/mn1d/`.
The reference curves were digitized from Figure 4 of Ahedo & Merino (2010)
using WebPlotDigitizer.

## Notes

This case directly validates the spec requirement (§7.2):
> "Our 2D code, averaged radially, must reproduce 1D results."

A mismatch between Helicon's 2D radial average and the 1D MN1D result
indicates either a WarpX boundary condition issue, insufficient particle
statistics, or a real 2D effect (e.g., plume divergence) that is correctly
absent from the 1D code. The latter must be documented rather than "fixed."
