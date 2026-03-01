# Resistive Detachment Validation

## Reference

Dimov, G. I., & Taskaev, S. Yu. (2003).
*Propagation of a plasma stream in a non-uniform magnetic field.*
In Proceedings of the 30th EPS Conference on Controlled Fusion and Plasma Physics.

## Physics

Resistive detachment occurs when electron-ion collisions are frequent enough to
break the frozen-in condition. The dimensionless Hall parameter

$$\Omega_e \tau_e = \frac{\omega_{ce}}{\nu_{ei}}$$

governs the transition:

- $\Omega_e \tau_e \gg 1$: magnetized electrons, frozen-in, good confinement
- $\Omega_e \tau_e \sim 1$: resistive regime, partial slippage
- $\Omega_e \tau_e \ll 1$: demagnetized, poor confinement

This case validates that MagNozzleX correctly identifies the transition density
where $\Omega_e \tau_e \approx 1$ and shows the expected detachment behavior.

## Configuration

```python
from magnozzlex.validate.cases.resistive_dimov import ResistiveDimovCase
case = ResistiveDimovCase()
config = case.get_config()
```

Key parameters at the $\Omega_e \tau_e \approx 1$ threshold:

| Parameter | Value | Notes |
|-----------|-------|-------|
| $n_0$ | $10^{21}$ m⁻³ | Threshold density |
| $B$ | 0.05 T | Applied field |
| $T_e$ | 10 eV | Electron temperature |
| $\Omega_e \tau_e$ | ~1 | Hall parameter at threshold |

## Threshold Derivation

At $B = 0.05$ T, $T_e = 10$ eV, the electron cyclotron frequency is

$$\omega_{ce} = \frac{eB}{m_e} \approx 8.8 \times 10^9 \text{ rad/s}$$

The Spitzer collision frequency $\nu_{ei} \propto n T_e^{-3/2}$ equals $\omega_{ce}$
at $n \approx 10^{21}$ m⁻³.

## Running

```bash
magnozzlex validate --case resistive_dimov
```

## Results

See `docs/validation_results/dimov/` for Hall parameter profiles and detachment efficiency plots.
