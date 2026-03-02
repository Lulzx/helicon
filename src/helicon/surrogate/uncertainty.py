"""Monte Carlo uncertainty quantification for the MLX neural surrogate.

Propagates input parameter uncertainties (coil tolerances, plasma source
variability) through the surrogate to produce confidence intervals on
performance predictions.

With MLX available, 100 000 MC samples run in seconds on the Metal GPU.

Usage::

    from helicon.surrogate.uncertainty import propagate_uncertainty, UQResult

    result = propagate_uncertainty(
        surrogate,
        mean_features=feat_array,
        std_features=tol_array,
        n_samples=100_000,
    )
    print(result.thrust_ci_95)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from helicon._mlx_utils import HAS_MLX
from helicon.surrogate.mlx_net import N_FEATURES, NozzleSurrogate

if HAS_MLX:
    import mlx.core as mx


@dataclass
class UQResult:
    """Monte Carlo UQ result for one nominal operating point.

    Attributes
    ----------
    mean : ndarray (3,)
        Sample mean of [thrust_N, eta_d, plume_angle_deg].
    std : ndarray (3,)
        Sample standard deviation.
    ci_95_lo : ndarray (3,)
        2.5th percentile (lower 95 % CI bound).
    ci_95_hi : ndarray (3,)
        97.5th percentile (upper 95 % CI bound).
    n_samples : int
        Number of Monte Carlo samples used.
    """

    mean: np.ndarray
    std: np.ndarray
    ci_95_lo: np.ndarray
    ci_95_hi: np.ndarray
    n_samples: int

    @property
    def thrust_ci_95(self) -> tuple[float, float]:
        """95 % CI on thrust [N]."""
        return float(self.ci_95_lo[0]), float(self.ci_95_hi[0])

    @property
    def eta_d_ci_95(self) -> tuple[float, float]:
        """95 % CI on detachment efficiency."""
        return float(self.ci_95_lo[1]), float(self.ci_95_hi[1])

    @property
    def plume_angle_ci_95(self) -> tuple[float, float]:
        """95 % CI on plume half-angle [deg]."""
        return float(self.ci_95_lo[2]), float(self.ci_95_hi[2])

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "ci_95_lo": self.ci_95_lo.tolist(),
            "ci_95_hi": self.ci_95_hi.tolist(),
            "n_samples": self.n_samples,
        }


def propagate_uncertainty(
    surrogate: NozzleSurrogate,
    mean_features: np.ndarray,
    std_features: np.ndarray,
    n_samples: int = 100_000,
    seed: int = 0,
) -> UQResult:
    """Propagate input uncertainties through the surrogate via Monte Carlo.

    Assumes Gaussian perturbations around ``mean_features`` with
    standard deviations ``std_features``.

    Parameters
    ----------
    surrogate : NozzleSurrogate
        Trained surrogate model.
    mean_features : ndarray (8,)
        Nominal feature vector (raw, un-normalised).
    std_features : ndarray (8,)
        Per-feature standard deviations representing uncertainties.
    n_samples : int
        Number of Monte Carlo samples (100 000 recommended for 95 % CI).
    seed : int
        Random seed.

    Returns
    -------
    UQResult
        Sample statistics and 95 % confidence intervals.
    """
    if len(mean_features) != N_FEATURES:
        raise ValueError(f"mean_features must have {N_FEATURES} elements")
    if len(std_features) != N_FEATURES:
        raise ValueError(f"std_features must have {N_FEATURES} elements")

    rng = np.random.default_rng(seed)

    mean_f = mean_features.astype(np.float32)
    std_f = std_features.astype(np.float32)

    if HAS_MLX:
        samples = _mc_mlx(surrogate, mean_f, std_f, n_samples, rng)
    else:
        samples = _mc_numpy(surrogate, mean_f, std_f, n_samples, rng)

    return UQResult(
        mean=samples.mean(axis=0),
        std=samples.std(axis=0),
        ci_95_lo=np.percentile(samples, 2.5, axis=0),
        ci_95_hi=np.percentile(samples, 97.5, axis=0),
        n_samples=n_samples,
    )


def _mc_mlx(
    surrogate: NozzleSurrogate,
    mean_f: np.ndarray,
    std_f: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run MC sampling on Metal GPU in chunks to avoid OOM."""
    chunk = 10_000
    results = []
    remaining = n_samples
    while remaining > 0:
        n = min(chunk, remaining)
        noise = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
        X = mean_f + std_f * noise  # (n × 8)
        out = np.array(surrogate._mlx_model(mx.array(X)))  # (n × 3)
        # De-normalise
        out = out * surrogate.output_std.astype(np.float32) + surrogate.output_mean.astype(
            np.float32
        )
        results.append(out)
        remaining -= n
    return np.concatenate(results, axis=0)


def _mc_numpy(
    surrogate: NozzleSurrogate,
    mean_f: np.ndarray,
    std_f: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run MC sampling on CPU using the numpy fallback."""
    chunk = 5_000
    results = []
    remaining = n_samples
    while remaining > 0:
        n = min(chunk, remaining)
        noise = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
        X = mean_f + std_f * noise
        out = surrogate.predict_batch(X)
        results.append(out)
        remaining -= n
    return np.concatenate(results, axis=0)


def sensitivity_indices(
    surrogate: NozzleSurrogate,
    mean_features: np.ndarray,
    std_features: np.ndarray,
    n_samples: int = 10_000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Compute approximate first-order Sobol sensitivity indices.

    Uses the Saltelli estimator with A/B sample matrices.

    Returns
    -------
    dict
        ``"S1"`` : ndarray (n_features × n_targets) — first-order indices.
        ``"ST"`` : ndarray (n_features × n_targets) — total-effect indices.
    """
    from helicon.surrogate.mlx_net import FEATURE_NAMES, N_FEATURES, N_TARGETS

    rng = np.random.default_rng(seed)
    mean_f = mean_features.astype(np.float32)
    std_f = std_features.astype(np.float32)

    A = mean_f + std_f * rng.standard_normal((n_samples, N_FEATURES)).astype(np.float32)
    B = mean_f + std_f * rng.standard_normal((n_samples, N_FEATURES)).astype(np.float32)

    f_A = surrogate.predict_batch(A)
    f_B = surrogate.predict_batch(B)
    f0_sq = (0.5 * (f_A + f_B)).mean(axis=0) ** 2

    S1 = np.zeros((N_FEATURES, N_TARGETS))
    ST = np.zeros((N_FEATURES, N_TARGETS))

    for j in range(N_FEATURES):
        AB_j = A.copy()
        AB_j[:, j] = B[:, j]
        f_AB_j = surrogate.predict_batch(AB_j)

        var_y = ((f_A**2).mean(axis=0) - f0_sq).clip(1e-12)
        S1[j] = ((f_B * (f_AB_j - f_A)).mean(axis=0)) / var_y
        ST[j] = (0.5 * ((f_A - f_AB_j) ** 2).mean(axis=0)) / var_y

    return {
        "S1": S1,
        "ST": ST,
        "feature_names": FEATURE_NAMES,
    }
