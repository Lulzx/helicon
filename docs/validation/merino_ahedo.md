# Merino-Ahedo Detachment Validation

## Reference

Merino, M., & Ahedo, E. (2011).
*Simulation of plasma flows in divergent magnetic nozzles.*
IEEE Transactions on Plasma Science, 39(11), 2938–2939.

## Physics

This case validates the detachment efficiency as a function of mirror ratio $R_B$.
The Merino-Ahedo hybrid fluid model gives:

$$\eta_{det}(R_B) \approx 1 - \frac{1}{\sqrt{R_B}}$$

MagNozzleX's kinetic PIC should reproduce this trend within 10% over $R_B \in [2, 10]$.

## Configuration

```python
from magnozzlex.validate.cases.merino_ahedo import MerinoAhedoCase
case = MerinoAhedoCase()
configs = case.get_scan_configs()  # 5 mirror ratios
```

## Acceptance Criteria

- Detachment efficiency within 10% of Merino-Ahedo formula for all $R_B$ tested
- Monotonic increase of $\eta_{det}$ with $R_B$

## Running

```bash
magnozzlex validate --case merino_ahedo
```

## Results

See `docs/validation_results/merino_ahedo/` for comparison plots.
