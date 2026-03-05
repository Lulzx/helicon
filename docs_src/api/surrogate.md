# helicon.surrogate

MLX-native neural surrogate for nozzle performance prediction.

Provides a small MLP that predicts thrust, η_d, and plume angle in
microseconds on the Metal GPU, trained on PIC scan data.

## Usage

```python
from helicon.surrogate import NozzleSurrogate, generate_training_data, train_surrogate

data = generate_training_data(n_samples=500, seed=42)
surrogate = train_surrogate(data, epochs=200)

from helicon.surrogate import SurrogateFeatures
feats = SurrogateFeatures(
    coil_r=0.15, coil_I=50000.0, nozzle_length=2.0,
    n0=1e19, T_i_eV=5000.0, T_e_eV=2000.0, v_inj_ms=200000.0,
)
pred = surrogate.predict(feats)
print(f"Thrust: {pred.thrust_N:.4f} N  η_d: {pred.eta_d:.3f}")
```

## Monte Carlo UQ

```python
from helicon.surrogate import propagate_uncertainty

result = propagate_uncertainty(
    surrogate,
    feats,
    uncertainties={"coil_I": 0.05, "n0": 0.10},
    n_samples=10000,
)
print(f"Thrust CI: {result.thrust_N_mean:.4f} ± {result.thrust_N_std:.4f} N")
```

::: helicon.surrogate.NozzleSurrogate
::: helicon.surrogate.SurrogateFeatures
::: helicon.surrogate.SurrogatePrediction
::: helicon.surrogate.UQResult
::: helicon.surrogate.generate_training_data
::: helicon.surrogate.train_surrogate
::: helicon.surrogate.propagate_uncertainty
