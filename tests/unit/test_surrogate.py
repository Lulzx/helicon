"""Tests for helicon.optimize.surrogate."""

from __future__ import annotations

import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")

from helicon.optimize.surrogate import BayesianOptimizer, GPSurrogate, SurrogateResult


# ---------------------------------------------------------------------------
# GPSurrogate
# ---------------------------------------------------------------------------
class TestGPSurrogate:
    def test_fit_predict_shapes(self):
        X = np.linspace(0, 1, 10).reshape(-1, 1)
        y = np.sin(2 * np.pi * X.ravel())
        gp = GPSurrogate()
        gp.fit(X, y)
        result = gp.predict(X)
        assert isinstance(result, SurrogateResult)
        assert result.mean.shape == (10,)
        assert result.std.shape == (10,)

    def test_predict_before_fit_raises(self):
        gp = GPSurrogate()
        with pytest.raises(RuntimeError, match="must be fit"):
            gp.predict(np.array([[0.5]]))

    def test_interpolates_training_data(self):
        X = np.linspace(0, 1, 6).reshape(-1, 1)
        y = X.ravel() ** 2
        gp = GPSurrogate()
        gp.fit(X, y)
        result = gp.predict(X)
        np.testing.assert_allclose(result.mean, y, atol=1e-4)

    def test_std_nonnegative(self):
        X = np.linspace(0, 1, 8).reshape(-1, 1)
        gp = GPSurrogate()
        gp.fit(X, X.ravel())
        result = gp.predict(np.linspace(0, 1, 20).reshape(-1, 1))
        assert np.all(result.std >= 0)

    def test_expected_improvement_nonnegative(self):
        X = np.linspace(0, 1, 15).reshape(-1, 1)
        y = -(X.ravel() - 0.5) ** 2
        gp = GPSurrogate()
        gp.fit(X, y)
        ei = gp.expected_improvement(X, y_best=float(np.max(y)))
        assert np.all(ei >= 0)

    def test_ei_positive_near_unknown_region(self):
        # Smooth sinusoidal data so the GP fits without hitting kernel bounds
        X_train = np.linspace(0, 1, 10).reshape(-1, 1)
        y_train = np.sin(np.pi * X_train.ravel())
        gp = GPSurrogate()
        gp.fit(X_train, y_train)
        # Query near the peak where EI should be negligible but non-negative
        X_test = np.linspace(0, 1, 20).reshape(-1, 1)
        ei = gp.expected_improvement(X_test, y_best=float(np.max(y_train)))
        assert np.all(ei >= 0)

    def test_predict_1d_multidim(self):
        X = np.random.default_rng(0).uniform(size=(20, 3))
        y = X[:, 0] + 2 * X[:, 1]
        gp = GPSurrogate()
        gp.fit(X, y)
        result = gp.predict(X)
        assert result.mean.shape == (20,)


# ---------------------------------------------------------------------------
# BayesianOptimizer
# ---------------------------------------------------------------------------
class TestBayesianOptimizer:
    def test_ask_shape_random_phase(self):
        bounds = np.array([[0.0, 1.0], [0.0, 2.0]])
        opt = BayesianOptimizer(bounds, n_init=5)
        X = opt.ask(n=3)
        assert X.shape == (3, 2)

    def test_ask_respects_bounds(self):
        bounds = np.array([[2.0, 5.0], [-1.0, 0.0]])
        opt = BayesianOptimizer(bounds, n_init=5)
        X = opt.ask(n=10)
        assert np.all(X[:, 0] >= 2.0) and np.all(X[:, 0] <= 5.0)
        assert np.all(X[:, 1] >= -1.0) and np.all(X[:, 1] <= 0.0)

    def test_tell_increments_count(self):
        bounds = np.array([[0.0, 1.0]])
        opt = BayesianOptimizer(bounds)
        opt.tell(np.array([[0.3]]), 0.7)
        opt.tell(np.array([[0.6]]), 0.4)
        assert opt.n_evaluated == 2

    def test_tell_batch(self):
        bounds = np.array([[0.0, 1.0]])
        opt = BayesianOptimizer(bounds)
        opt.tell(np.array([[0.1], [0.5], [0.9]]), np.array([0.3, 0.8, 0.5]))
        assert opt.n_evaluated == 3

    def test_best_returns_max(self):
        bounds = np.array([[0.0, 1.0]])
        opt = BayesianOptimizer(bounds)
        opt.tell(np.array([[0.1], [0.5], [0.9]]), np.array([0.3, 0.9, 0.4]))
        x_best, y_best = opt.best()
        assert y_best == pytest.approx(0.9)
        assert x_best[0] == pytest.approx(0.5)

    def test_best_before_observations_raises(self):
        bounds = np.array([[0.0, 1.0]])
        opt = BayesianOptimizer(bounds)
        with pytest.raises(RuntimeError, match="No observations"):
            opt.best()

    def test_ask_after_n_init_uses_surrogate(self):
        bounds = np.array([[0.0, 1.0]])
        opt = BayesianOptimizer(bounds, n_init=3, seed=0)
        for xi in [0.1, 0.3, 0.5, 0.7]:
            opt.tell(np.array([[xi]]), np.array([xi * (1.0 - xi)]))
        X_next = opt.ask(n=1)
        assert X_next.shape == (1, 1)
        assert 0.0 <= X_next[0, 0] <= 1.0

    def test_reproducible_with_seed(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        opt1 = BayesianOptimizer(bounds, seed=42)
        opt2 = BayesianOptimizer(bounds, seed=42)
        np.testing.assert_array_equal(opt1.ask(5), opt2.ask(5))
