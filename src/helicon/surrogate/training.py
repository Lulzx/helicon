"""Training data generation and surrogate training for the MLX MLP.

Generates synthetic training data from the Tier 1 analytical model
(Biot-Savart + ThrottleMap physics) and trains the NozzleSurrogate.

Usage::

    from helicon.surrogate.training import generate_training_data, train_surrogate

    data = generate_training_data(n_samples=500, seed=42)
    surrogate = train_surrogate(data, epochs=300, lr=3e-3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from helicon._mlx_utils import HAS_MLX
from helicon.surrogate.mlx_net import (
    N_FEATURES,
    N_TARGETS,
    NozzleSurrogate,
    SurrogateFeatures,
)

if HAS_MLX:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

_G0 = 9.80665  # m/s²
_MU0 = 4 * math.pi * 1e-7  # H/m


@dataclass
class TrainingData:
    """Container for surrogate training / validation data.

    Attributes
    ----------
    X : ndarray (N × 8)
        Normalised input features.
    y : ndarray (N × 3)
        Raw target values [thrust_N, eta_d, plume_angle_deg].
    input_mean : ndarray (8,)
    input_std : ndarray (8,)
    output_mean : ndarray (3,)
    output_std : ndarray (3,)
    """

    X: np.ndarray
    y: np.ndarray
    input_mean: np.ndarray
    input_std: np.ndarray
    output_mean: np.ndarray
    output_std: np.ndarray


def _biot_savart_axis(
    z_axis: np.ndarray, coil_z: float, coil_r: float, I: float
) -> np.ndarray:
    """On-axis Bz from a single circular coil (Biot-Savart)."""
    dz = z_axis - coil_z
    return _MU0 * I * coil_r**2 / (2.0 * (coil_r**2 + dz**2) ** 1.5)


def _compute_field_features(
    coil_z: float,
    coil_r: float,
    coil_I: float,
    z_max: float,
) -> dict[str, float]:
    """Compute physics-informed features from a single coil."""
    z_axis = np.linspace(coil_z - 0.5, z_max, 300)
    Bz = _biot_savart_axis(z_axis, coil_z, coil_r, coil_I)
    Bz = np.abs(Bz)

    b_peak = float(np.max(Bz))
    b_min = float(np.min(Bz[z_axis > coil_z])) if np.any(z_axis > coil_z) else b_peak * 0.01
    mirror_ratio = b_peak / max(b_min, 1e-12)

    dBdz = np.abs(np.gradient(Bz, z_axis))
    b_gradient = float(np.max(dBdz))
    nozzle_length = max(z_max - coil_z, 0.01)

    return {
        "mirror_ratio": mirror_ratio,
        "b_peak_T": b_peak,
        "b_gradient_T_m": b_gradient,
        "nozzle_length_m": nozzle_length,
    }


def _analytical_performance(
    b_peak_T: float,
    mirror_ratio: float,
    n0_m3: float,
    T_i_eV: float,
    T_e_eV: float,
    v_inj_ms: float,
    mdot_kgs: float,
    eta_thermal: float = 0.7,
) -> dict[str, float]:
    """Analytical performance from Tier 1 (ThrottleMap physics)."""
    # Detachment efficiency from mirror ratio (empirical fit)
    eta_d = float(np.clip(1.0 - 1.0 / mirror_ratio**0.5, 0.0, 0.95))

    # Divergence efficiency (half-angle model)
    plume_angle_deg = float(np.clip(15.0 + 10.0 / mirror_ratio, 5.0, 60.0))
    eta_div = math.cos(math.radians(plume_angle_deg)) ** 2

    # Effective exhaust velocity
    ke = 0.5 * mdot_kgs * v_inj_ms**2
    P_in = ke / max(eta_thermal, 1e-3)
    v_e = math.sqrt(max(2.0 * eta_thermal * P_in / max(mdot_kgs, 1e-15), 0.0))
    thrust_N = float(mdot_kgs * v_e * eta_div)

    return {
        "thrust_N": thrust_N,
        "eta_d": eta_d,
        "plume_angle_deg": plume_angle_deg,
    }


def generate_training_data(
    n_samples: int = 500,
    seed: int = 0,
    eta_thermal: float = 0.7,
) -> TrainingData:
    """Generate synthetic training data from the Tier 1 analytical model.

    Samples random nozzle configurations and plasma parameters,
    computes physics-informed features and analytical performance.

    Parameters
    ----------
    n_samples : int
        Number of training samples.
    seed : int
        Random seed for reproducibility.
    eta_thermal : float
        Assumed thermal efficiency of the RF source.

    Returns
    -------
    TrainingData
        Normalised features and raw targets ready for training.
    """
    rng = np.random.default_rng(seed)

    X_raw = np.zeros((n_samples, N_FEATURES), dtype=np.float32)
    y_raw = np.zeros((n_samples, N_TARGETS), dtype=np.float32)

    for i in range(n_samples):
        # Sample coil parameters
        coil_z = rng.uniform(-0.1, 0.1)
        coil_r = rng.uniform(0.05, 0.25)
        coil_I = rng.uniform(1000.0, 150000.0)
        z_max = rng.uniform(0.5, 3.0)

        # Sample plasma parameters
        n0 = 10 ** rng.uniform(16.0, 20.0)  # m⁻³
        T_i = rng.uniform(5.0, 500.0)  # eV
        T_e = rng.uniform(5.0, 200.0)  # eV
        v_inj = rng.uniform(5000.0, 200000.0)  # m/s

        # Estimate mass flow (proton mass × n0 × v × area)
        area = math.pi * coil_r**2
        mdot = 1.67e-27 * n0 * v_inj * area * 1e-4  # scaled area
        mdot = float(np.clip(mdot, 1e-7, 1e-2))

        # Compute field features
        ff = _compute_field_features(coil_z, coil_r, coil_I, z_max)

        # Build raw features
        feats = SurrogateFeatures(
            mirror_ratio=ff["mirror_ratio"],
            b_peak_T=ff["b_peak_T"],
            b_gradient_T_m=ff["b_gradient_T_m"],
            nozzle_length_m=ff["nozzle_length_m"],
            n0_m3=n0,
            T_i_eV=T_i,
            T_e_eV=T_e,
            v_injection_ms=v_inj,
        )
        X_raw[i] = feats.to_array()

        # Compute analytical targets
        perf = _analytical_performance(
            ff["b_peak_T"],
            ff["mirror_ratio"],
            n0,
            T_i,
            T_e,
            v_inj,
            mdot,
            eta_thermal,
        )
        y_raw[i] = [perf["thrust_N"], perf["eta_d"], perf["plume_angle_deg"]]

    # Normalise
    input_mean = X_raw.mean(axis=0)
    input_std = np.maximum(X_raw.std(axis=0), 1e-8)
    output_mean = y_raw.mean(axis=0)
    output_std = np.maximum(y_raw.std(axis=0), 1e-8)

    return TrainingData(
        X=X_raw,
        y=y_raw,
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
    )


def train_surrogate(
    data: TrainingData,
    epochs: int = 300,
    lr: float = 3e-3,
    batch_size: int = 64,
    hidden: tuple[int, ...] = (64, 64, 32),
    val_fraction: float = 0.15,
    seed: int = 0,
    verbose: bool = False,
) -> NozzleSurrogate:
    """Train the neural surrogate on generated training data.

    Parameters
    ----------
    data : TrainingData
        Output of :func:`generate_training_data`.
    epochs : int
        Training epochs.
    lr : float
        Adam learning rate.
    batch_size : int
        Mini-batch size.
    hidden : tuple
        Hidden layer widths.
    val_fraction : float
        Fraction of samples held out for validation.
    seed : int
        Shuffle seed.
    verbose : bool
        Print loss every 50 epochs.

    Returns
    -------
    NozzleSurrogate
        Trained surrogate with normalisation statistics embedded.
    """
    rng = np.random.default_rng(seed)

    X_norm = (data.X - data.input_mean) / data.input_std
    y_norm = (data.y - data.output_mean) / data.output_std

    N = len(X_norm)
    n_val = max(1, int(N * val_fraction))
    idx = rng.permutation(N)
    X_tr, y_tr = X_norm[idx[n_val:]], y_norm[idx[n_val:]]

    surrogate = NozzleSurrogate(
        hidden=hidden,
        input_mean=data.input_mean,
        input_std=data.input_std,
        output_mean=data.output_mean,
        output_std=data.output_std,
    )

    if HAS_MLX and surrogate._mlx_model is not None:
        _train_mlx(surrogate._mlx_model, X_tr, y_tr, epochs, lr, batch_size, verbose)
        # Cache numpy weights for serialisation
        surrogate._np_weights = [
            {
                "weight": np.array(layer.weight),
                "bias": np.array(layer.bias),
            }
            for layer in surrogate._mlx_model.layers
        ]
    else:
        _train_numpy(surrogate, X_tr, y_tr, epochs, lr, batch_size, verbose)

    return surrogate


def _train_mlx(
    model: Any,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    verbose: bool,
) -> None:
    """Train MLX model with Adam."""
    optimizer = optim.Adam(learning_rate=lr)

    def loss_fn(model: Any, x: mx.array, y: mx.array) -> mx.array:
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    N = len(X_tr)
    X_np = X_tr.astype(np.float32)
    y_np = y_tr.astype(np.float32)

    for epoch in range(epochs):
        # Mini-batch shuffle — index numpy arrays, then convert to MLX
        perm = np.random.permutation(N)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            xb = mx.array(X_np[idx])
            yb = mx.array(y_np[idx])
            _loss, grads = loss_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        if verbose and (epoch + 1) % 50 == 0:
            X_all = mx.array(X_np)
            y_all = mx.array(y_np)
            val_loss = float(loss_fn(model, X_all, y_all))
            print(f"  Epoch {epoch + 1}/{epochs}  loss={val_loss:.6f}")


def _train_numpy(
    surrogate: NozzleSurrogate,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    epochs: int,
    lr: float,
    batch_size: int,
    verbose: bool,
) -> None:
    """Minimal SGD-style training fallback when MLX is unavailable.

    Initialises random weights and trains with gradient descent using
    numpy automatic differentiation via finite differences.
    This is intentionally simple — for production use install MLX.
    """
    sizes = [N_FEATURES, *surrogate.hidden, N_TARGETS]
    rng = np.random.default_rng(0)
    weights = [
        {
            "weight": rng.standard_normal((sizes[i + 1], sizes[i])).astype(np.float32)
            * 0.1,
            "bias": np.zeros(sizes[i + 1], dtype=np.float32),
        }
        for i in range(len(sizes) - 1)
    ]
    surrogate._np_weights = weights

    N = len(X_tr)

    def _silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    def _silu_grad(x: np.ndarray) -> np.ndarray:
        sig = 1.0 / (1.0 + np.exp(-x))
        return sig * (1.0 + x * (1.0 - sig))

    for epoch in range(epochs):
        perm = np.random.permutation(N)
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_tr[idx].astype(np.float32)
            yb = y_tr[idx].astype(np.float32)

            # Forward pass with cache
            acts = [xb]
            pre_acts = []
            for i, w in enumerate(weights):
                z = acts[-1] @ w["weight"].T + w["bias"]
                pre_acts.append(z)
                acts.append(_silu(z) if i < len(weights) - 1 else z)

            # Backward pass
            loss_grad = 2.0 * (acts[-1] - yb) / len(idx)
            delta = loss_grad
            for i in reversed(range(len(weights))):
                gW = delta.T @ acts[i] / len(idx)
                gb = delta.mean(axis=0)
                weights[i]["weight"] -= lr * gW
                weights[i]["bias"] -= lr * gb
                if i > 0:
                    delta = (delta @ weights[i]["weight"]) * _silu_grad(pre_acts[i - 1])

        if verbose and (epoch + 1) % 50 == 0:
            pred = surrogate._numpy_forward(X_tr)
            loss = float(np.mean((pred - y_tr) ** 2))
            print(f"  Epoch {epoch + 1}/{epochs}  loss={loss:.6f}")
