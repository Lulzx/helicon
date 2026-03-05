# Guiding-Center Orbit Validation

## Reference

Littlejohn, R.G. (1983). Variational principles of guiding centre motion.
*Journal of Plasma Physics*, 29(1), 111–125.

## Physics

A single charged particle in a known magnetic mirror field must follow the
guiding-center drift trajectory predicted analytically by Littlejohn (1983).
This case tests:

1. Boris pusher orbit accuracy vs. analytic guiding-center solution
2. Magnetic moment conservation (adiabatic invariant $\mu = mv_\perp^2 / 2B$)
3. Bounce oscillation frequency in the mirror field
4. Second-order convergence of orbit error with timestep: $\epsilon \propto (\Delta t)^2$

## Configuration

```python
from helicon.validate.cases.guiding_center import GuidingCenterCase
case = GuidingCenterCase()
config = case.get_config()
```

Key parameters:

| Parameter | Value |
|-----------|-------|
| Species | H⁺ (proton) |
| $B_{throat}$ | 0.1 T |
| Mirror ratio $R_B$ | 10 |
| $v_\parallel / v_\perp$ | 0.3 |
| Timesteps | 50 000 |
| `requires_warpx` | False (analytic only) |

## Acceptance Criteria

- Orbit RMS error < 1% of Larmor radius relative to analytic solution
- Magnetic moment $\mu$ conserved to < 0.1% drift over full simulation
- Bounce period matches $\omega_b = v_\parallel \sqrt{R_B - 1} / L$ to < 2%
- Timestep convergence: halving $\Delta t$ reduces orbit error by factor ≥ 3.5

## Running

```bash
helicon validate --case guiding_center
```

```python
from helicon.validate.runner import run_validation

report = run_validation(cases=["guiding_center"], run_simulations=False)
print(report.summary())
```

## Notes

This case does **not** require WarpX (`requires_warpx = False`). The
guiding-center trajectory is computed analytically in Python using the
Littlejohn equations. This makes it the fastest validation case and the
first one to run in CI.

The Boris pusher used by WarpX is 2nd-order accurate. This case
quantitatively verifies that expectation so that if it ever fails, the
failure is caught before running expensive PIC cases.
