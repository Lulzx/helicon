"""MLX-native neural surrogate for nozzle performance prediction (v2.0).

Provides a small MLP trained on PIC scan data that predicts thrust,
detachment efficiency (η_d), and plume angle in microseconds on the
Metal GPU.

Usage::

    from helicon.surrogate import NozzleSurrogate, generate_training_data, train_surrogate

    data = generate_training_data(n_samples=500)
    surrogate = train_surrogate(data, epochs=200)
    preds = surrogate.predict(features)
"""

from __future__ import annotations

from helicon.surrogate.mlx_net import NozzleSurrogate, SurrogateFeatures, SurrogatePrediction
from helicon.surrogate.training import generate_training_data, train_surrogate
from helicon.surrogate.uncertainty import UQResult, propagate_uncertainty

__all__ = [
    "NozzleSurrogate",
    "SurrogateFeatures",
    "SurrogatePrediction",
    "UQResult",
    "generate_training_data",
    "propagate_uncertainty",
    "train_surrogate",
]
