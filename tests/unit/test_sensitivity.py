"""Tests for helicon.optimize.sensitivity."""

from __future__ import annotations

import numpy as np
import pytest

from helicon.optimize.sensitivity import SobolResult, compute_sobol, saltelli_sample


# ---------------------------------------------------------------------------
# saltelli_sample
# ---------------------------------------------------------------------------
class TestSaltelliSample:
    def test_shapes(self):
        bounds = np.array([[0.0, 1.0], [0.0, 2.0], [-1.0, 1.0]])
        A, B = saltelli_sample(100, bounds)
        assert A.shape == (100, 3)
        assert B.shape == (100, 3)

    def test_bounds_respected(self):
        bounds = np.array([[2.0, 5.0], [-1.0, 0.0]])
        A, B = saltelli_sample(200, bounds)
        assert np.all(A[:, 0] >= 2.0) and np.all(A[:, 0] <= 5.0)
        assert np.all(A[:, 1] >= -1.0) and np.all(A[:, 1] <= 0.0)
        assert np.all(B[:, 0] >= 2.0) and np.all(B[:, 0] <= 5.0)

    def test_reproducible_with_seed(self):
        bounds = np.array([[0.0, 1.0]])
        A1, B1 = saltelli_sample(50, bounds, seed=42)
        A2, B2 = saltelli_sample(50, bounds, seed=42)
        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(B1, B2)

    def test_different_seeds_differ(self):
        bounds = np.array([[0.0, 1.0]])
        A1, _ = saltelli_sample(50, bounds, seed=1)
        A2, _ = saltelli_sample(50, bounds, seed=2)
        assert not np.allclose(A1, A2)

    def test_A_B_independent(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        A, B = saltelli_sample(200, bounds)
        # A and B should not be identical
        assert not np.allclose(A, B)


# ---------------------------------------------------------------------------
# compute_sobol
# ---------------------------------------------------------------------------
class TestComputeSobol:
    @staticmethod
    def _linear(X: np.ndarray) -> np.ndarray:
        """f = 3*x0 + x1.  Var dominates x0 (9×)."""
        return 3.0 * X[:, 0] + X[:, 1]

    @staticmethod
    def _quadratic(X: np.ndarray) -> np.ndarray:
        """f = x0² (only x0 matters)."""
        return X[:, 0] ** 2

    def test_returns_sobol_result(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        result = compute_sobol(self._linear, 256, bounds, ["x0", "x1"])
        assert isinstance(result, SobolResult)
        assert result.S1.shape == (2,)
        assert result.ST.shape == (2,)

    def test_dominant_parameter_linear(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        result = compute_sobol(self._linear, 512, bounds, ["x0", "x1"], seed=0)
        assert result.S1[0] > result.S1[1], "x0 should dominate in linear f"
        assert result.ST[0] > result.ST[1]

    def test_single_parameter_quadratic(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        result = compute_sobol(self._quadratic, 512, bounds, ["x0", "x1"], seed=0)
        # x1 has zero effect
        assert result.S1[1] < 0.05, f"S1[x1]={result.S1[1]:.4f} should be ~0"
        assert result.ST[1] < 0.05, f"ST[x1]={result.ST[1]:.4f} should be ~0"

    def test_indices_in_unit_interval(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        result = compute_sobol(self._linear, 256, bounds, ["x0", "x1"])
        assert np.all(result.S1 >= 0) and np.all(result.S1 <= 1)
        assert np.all(result.ST >= 0) and np.all(result.ST <= 1)

    def test_constant_function_zero_indices(self):
        def const_f(X: np.ndarray) -> np.ndarray:
            return np.ones(X.shape[0])

        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        result = compute_sobol(const_f, 128, bounds, ["x0", "x1"])
        np.testing.assert_allclose(result.S1, [0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(result.ST, [0.0, 0.0], atol=1e-10)

    def test_param_names_preserved(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        result = compute_sobol(self._linear, 64, bounds, ["alpha", "beta"])
        assert result.param_names == ["alpha", "beta"]

    def test_param_names_length_mismatch_raises(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        with pytest.raises(ValueError):
            compute_sobol(self._linear, 64, bounds, ["only_one"])

    def test_summary_contains_names(self):
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        result = compute_sobol(self._linear, 64, bounds, ["x0", "x1"])
        s = result.summary()
        assert "x0" in s and "x1" in s
        assert "S1" in s and "ST" in s
