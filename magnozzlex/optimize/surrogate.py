"""Gaussian Process surrogate model for Bayesian optimization.

Uses scikit-learn GaussianProcessRegressor as the default backend
(always available when ``scikit-learn`` is installed).  Optional
``botorch`` backend is detected at import time for advanced acquisition
functions and constrained optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import botorch  # noqa: F401

    HAS_BOTORCH = True
except ImportError:
    HAS_BOTORCH = False


@dataclass
class SurrogateResult:
    """Prediction from a GP surrogate."""

    mean: np.ndarray
    std: np.ndarray


class GPSurrogate:
    """Gaussian Process surrogate model backed by scikit-learn.

    Uses a Matérn 5/2 kernel with automatic hyperparameter optimization.
    Provides Expected Improvement acquisition function for Bayesian
    optimization of noisy, expensive-to-evaluate objectives.

    Parameters
    ----------
    normalize_y : bool
        Normalize target values before fitting (recommended).
    n_restarts : int
        Number of hyperparameter optimization restarts.
    """

    def __init__(self, normalize_y: bool = True, n_restarts: int = 5):
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for GPSurrogate. "
                "Install with: pip install scikit-learn"
            )
        # Wider bounds avoid hitting the lower limit when inputs span many
        # orders of magnitude (e.g. coil current 1e3–1e5 A).  After
        # StandardScaler normalization the effective length scale is ~O(1).
        kernel = Matern(nu=2.5, length_scale_bounds=(1e-2, 1e2))
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=normalize_y,
        )
        self._scaler: StandardScaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GP to observations.

        Inputs are normalized to zero mean and unit variance so the GP
        length-scale hyperparameter operates in a consistent numerical range
        regardless of the physical units of the parameters.

        Parameters
        ----------
        X : array, shape (n_samples, n_params)
        y : array, shape (n_samples,)
        """
        X_np = np.asarray(X, dtype=float)
        self._scaler.fit(X_np)
        self._gp.fit(self._scaler.transform(X_np), np.asarray(y, dtype=float))
        self._fitted = True

    def predict(self, X: np.ndarray) -> SurrogateResult:
        """Predict mean and std at new points.

        Parameters
        ----------
        X : array, shape (n_points, n_params)

        Returns
        -------
        SurrogateResult with ``mean`` and ``std`` arrays.
        """
        if not self._fitted:
            raise RuntimeError("GPSurrogate must be fit() before predict()")
        X_scaled = self._scaler.transform(np.asarray(X, dtype=float))
        mean, std = self._gp.predict(X_scaled, return_std=True)
        return SurrogateResult(mean=mean, std=std)

    def expected_improvement(
        self, X: np.ndarray, y_best: float, xi: float = 0.01
    ) -> np.ndarray:
        """Expected improvement acquisition function (for maximization).

        Parameters
        ----------
        X : array, shape (n_points, n_params)
        y_best : float
            Best observed objective value so far.
        xi : float
            Exploration-exploitation trade-off (larger → more exploration).

        Returns
        -------
        EI : array, shape (n_points,)
            Non-negative expected improvement at each candidate point.
        """
        from scipy.stats import norm

        result = self.predict(X)
        mu, sigma = result.mean, result.std
        sigma = np.maximum(sigma, 1e-9)
        improvement = mu - y_best - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return np.maximum(ei, 0.0)


class BayesianOptimizer:
    """Sequential model-based optimizer using a GP surrogate.

    Uses Expected Improvement as the acquisition function and maximizes
    it over a dense random candidate grid (no extra solver dependencies).

    Parameters
    ----------
    bounds : array-like, shape (n_params, 2)
        ``[[low_0, high_0], [low_1, high_1], ...]`` for each parameter.
    n_init : int
        Number of random evaluations before switching to GP-guided search.
    seed : int
        Random seed for initial sampling and acquisition maximization.
    """

    def __init__(
        self,
        bounds: np.ndarray | list,
        n_init: int = 5,
        seed: int = 0,
    ):
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_init = n_init
        self._rng = np.random.default_rng(seed)
        self._X: list[np.ndarray] = []
        self._y: list[float] = []
        self._surrogate: GPSurrogate | None = GPSurrogate() if HAS_SKLEARN else None

    @property
    def n_evaluated(self) -> int:
        """Number of observations recorded so far."""
        return len(self._y)

    def ask(self, n: int = 1) -> np.ndarray:
        """Suggest the next point(s) to evaluate.

        Returns random points until ``n_init`` observations are available,
        then maximizes Expected Improvement over a random candidate grid.

        Returns
        -------
        X : array, shape (n, n_params)
        """
        n_params = self.bounds.shape[0]
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]

        if self.n_evaluated < self.n_init or self._surrogate is None:
            return low + self._rng.uniform(size=(n, n_params)) * (high - low)

        X_train = np.array(self._X)
        y_train = np.array(self._y)
        self._surrogate.fit(X_train, y_train)

        n_candidates = max(1000, 200 * n_params)
        X_cand = low + self._rng.uniform(size=(n_candidates, n_params)) * (high - low)
        ei = self._surrogate.expected_improvement(X_cand, y_best=float(np.max(y_train)))
        top_idx = np.argsort(ei)[-n:][::-1]
        return X_cand[top_idx]

    def tell(self, X: np.ndarray, y: np.ndarray | float) -> None:
        """Record new observations.

        Parameters
        ----------
        X : array, shape (n, n_params) or (n_params,)
        y : array, shape (n,) or scalar
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        for xi, yi in zip(X, y):
            self._X.append(xi.copy())
            self._y.append(float(yi))

    def best(self) -> tuple[np.ndarray, float]:
        """Return the best observed point and its objective value.

        Returns
        -------
        x_best : array, shape (n_params,)
        y_best : float
        """
        if not self._y:
            raise RuntimeError("No observations yet. Call tell() first.")
        best_idx = int(np.argmax(self._y))
        return self._X[best_idx].copy(), self._y[best_idx]
