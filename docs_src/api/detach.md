# helicon.detach

Real-time reduced model for magnetic nozzle detachment onset.

Predicts detachment from local MHD invariants and sheath parameters.
Designed for embedded control loops — pure scalar arithmetic, no NumPy.

## Detachment Criteria

Three independent criteria determine detachment onset:

1. **Alfvénic**: $M_A = v_z / v_A > 1$ — momentum coupling breaks down
2. **Electron β**: $\beta_e = n k T_e / (B^2/2\mu_0) > 0.1$ — field lines pushed aside
3. **Ion demagnetization**: $\Lambda_i = r_{Li} / L_B > 1$ — ions stop following field lines

## CLI

```bash
# Assess local detachment state
helicon detach assess --n 1e19 --Te 2000 --Ti 5000 --B 0.05 --vz 200000

# Full report with all criteria
helicon detach report --n 1e19 --Te 2000 --Ti 5000 --B 0.05 --vz 200000

# Calibrate detachment model weights from synthetic data
helicon detach calibrate --n-samples 1000 --seed 42

# Invert thrust measurement → plasma state
helicon detach invert --thrust 4.8 --mdot 4.4e-5 --eta-d 0.87
```

## Python API

```python
from helicon.detach import DetachmentOnsetModel, PlasmaState

state = PlasmaState(
    n_m3=1e19, Te_eV=2000.0, Ti_eV=5000.0,
    B_T=0.05, dBdz_T_per_m=-0.1, vz_ms=200000.0,
    mass_amu=2.014,  # Deuterium
)
model = DetachmentOnsetModel()
ds = model.assess(state)
print(ds.summary())
# Get control recommendation (delta I to push toward attached regime)
rec = model.control_recommendation(state)
```

::: helicon.detach.DetachmentOnsetModel
::: helicon.detach.PlasmaState
::: helicon.detach.DetachmentState
