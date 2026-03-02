"""Tests for helicon.surrogate — MLX neural surrogate (v2.0)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from helicon.surrogate.mlx_net import (
    N_FEATURES,
    N_TARGETS,
    NozzleSurrogate,
    SurrogateFeatures,
    SurrogatePrediction,
)
from helicon.surrogate.training import (
    TrainingData,
    generate_training_data,
    train_surrogate,
)
from helicon.surrogate.uncertainty import UQResult, propagate_uncertainty

# ---------------------------------------------------------------------------
# SurrogateFeatures
# ---------------------------------------------------------------------------


class TestSurrogateFeatures:
    def _make_feats(self) -> SurrogateFeatures:
        return SurrogateFeatures(
            mirror_ratio=3.5,
            b_peak_T=0.05,
            b_gradient_T_m=1.2,
            nozzle_length_m=1.2,
            n0_m3=1e18,
            T_i_eV=100.0,
            T_e_eV=100.0,
            v_injection_ms=50000.0,
        )

    def test_to_array_shape(self):
        arr = self._make_feats().to_array()
        assert arr.shape == (N_FEATURES,)
        assert arr.dtype == np.float32

    def test_to_array_log_transforms(self):
        feats = self._make_feats()
        arr = feats.to_array()
        # log_b_peak = log10(0.05) ≈ -1.3
        assert arr[1] == pytest.approx(np.log10(0.05), rel=1e-4)


# ---------------------------------------------------------------------------
# NozzleSurrogate (untrained)
# ---------------------------------------------------------------------------


class TestNozzleSurrogateUntrained:
    def test_predict_returns_prediction(self):
        sur = NozzleSurrogate()
        feats = SurrogateFeatures(
            mirror_ratio=2.0,
            b_peak_T=0.02,
            b_gradient_T_m=0.5,
            nozzle_length_m=1.0,
            n0_m3=1e18,
            T_i_eV=50.0,
            T_e_eV=50.0,
            v_injection_ms=30000.0,
        )
        pred = sur.predict(feats)
        assert isinstance(pred, SurrogatePrediction)
        assert 0.0 <= pred.eta_d <= 1.0
        assert 0.0 <= pred.plume_angle_deg <= 90.0

    def test_predict_batch_shape(self):
        sur = NozzleSurrogate()
        X = np.random.default_rng(0).random((5, N_FEATURES)).astype(np.float32)
        out = sur.predict_batch(X)
        assert out.shape == (5, N_TARGETS)

    def test_accuracy_envelope_keys(self):
        sur = NozzleSurrogate()
        env = sur.accuracy_envelope()
        assert "feature_names" in env
        assert "target_names" in env
        assert "architecture" in env


# ---------------------------------------------------------------------------
# Training data generation
# ---------------------------------------------------------------------------


class TestGenerateTrainingData:
    def test_shape(self):
        data = generate_training_data(n_samples=20, seed=0)
        assert data.X.shape == (20, N_FEATURES)
        assert data.y.shape == (20, N_TARGETS)

    def test_eta_d_in_range(self):
        data = generate_training_data(n_samples=50, seed=1)
        assert np.all(data.y[:, 1] >= 0.0)
        assert np.all(data.y[:, 1] <= 1.0)

    def test_thrust_positive(self):
        data = generate_training_data(n_samples=30, seed=2)
        assert np.all(data.y[:, 0] >= 0.0)

    def test_normalization_stored(self):
        data = generate_training_data(n_samples=20, seed=3)
        assert data.input_mean.shape == (N_FEATURES,)
        assert data.input_std.shape == (N_FEATURES,)
        assert np.all(data.input_std > 0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTrainSurrogate:
    def _make_data(self) -> TrainingData:
        return generate_training_data(n_samples=60, seed=42)

    def test_returns_surrogate(self):
        data = self._make_data()
        sur = train_surrogate(data, epochs=5)
        assert isinstance(sur, NozzleSurrogate)

    def test_normalization_embedded(self):
        data = self._make_data()
        sur = train_surrogate(data, epochs=5)
        np.testing.assert_allclose(sur.input_mean, data.input_mean, rtol=1e-5)

    def test_trained_surrogate_predicts(self):
        data = self._make_data()
        sur = train_surrogate(data, epochs=10)
        feats = SurrogateFeatures(
            mirror_ratio=4.0,
            b_peak_T=0.08,
            b_gradient_T_m=2.0,
            nozzle_length_m=1.5,
            n0_m3=1e18,
            T_i_eV=100.0,
            T_e_eV=100.0,
            v_injection_ms=60000.0,
        )
        pred = sur.predict(feats)
        assert pred.eta_d >= 0.0


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestSurrogateSaveLoad:
    def test_roundtrip_files_exist(self):
        data = generate_training_data(n_samples=40, seed=7)
        sur = train_surrogate(data, epochs=5)
        with tempfile.TemporaryDirectory() as tmp:
            sur.save(tmp)
            assert (Path(tmp) / "weights.npz").exists()
            assert (Path(tmp) / "meta.json").exists()

    def test_predictions_consistent_after_load(self):
        data = generate_training_data(n_samples=40, seed=8)
        sur = train_surrogate(data, epochs=5)
        X = np.random.default_rng(0).random((3, N_FEATURES)).astype(np.float32)
        out1 = sur.predict_batch(X)

        with tempfile.TemporaryDirectory() as tmp:
            sur.save(tmp)
            sur2 = NozzleSurrogate.load(tmp)
        out2 = sur2.predict_batch(X)
        np.testing.assert_allclose(out1, out2, atol=1e-3)


# ---------------------------------------------------------------------------
# Uncertainty quantification
# ---------------------------------------------------------------------------


class TestUncertaintyQuantification:
    def _trained_surrogate(self) -> NozzleSurrogate:
        data = generate_training_data(n_samples=60, seed=0)
        return train_surrogate(data, epochs=10)

    def test_uq_result_shape(self):
        sur = self._trained_surrogate()
        mean_f = np.ones(N_FEATURES, dtype=np.float32)
        std_f = np.ones(N_FEATURES, dtype=np.float32) * 0.1
        uq = propagate_uncertainty(sur, mean_f, std_f, n_samples=200, seed=0)
        assert isinstance(uq, UQResult)
        assert uq.mean.shape == (N_TARGETS,)
        assert uq.std.shape == (N_TARGETS,)
        assert uq.n_samples == 200

    def test_ci_ordering(self):
        sur = self._trained_surrogate()
        mean_f = np.ones(N_FEATURES, dtype=np.float32)
        std_f = np.ones(N_FEATURES, dtype=np.float32) * 0.05
        uq = propagate_uncertainty(sur, mean_f, std_f, n_samples=500, seed=1)
        assert np.all(uq.ci_95_lo <= uq.mean)
        assert np.all(uq.mean <= uq.ci_95_hi)

    def test_thrust_ci_95_tuple(self):
        sur = self._trained_surrogate()
        mean_f = np.ones(N_FEATURES, dtype=np.float32)
        std_f = np.ones(N_FEATURES, dtype=np.float32) * 0.1
        uq = propagate_uncertainty(sur, mean_f, std_f, n_samples=200, seed=2)
        lo, hi = uq.thrust_ci_95
        assert lo <= hi

    def test_wrong_input_shape_raises(self):
        sur = self._trained_surrogate()
        with pytest.raises(ValueError):
            propagate_uncertainty(sur, np.ones(5), np.ones(5), n_samples=10)

    def test_to_dict(self):
        sur = self._trained_surrogate()
        mean_f = np.ones(N_FEATURES, dtype=np.float32)
        std_f = np.ones(N_FEATURES, dtype=np.float32) * 0.05
        uq = propagate_uncertainty(sur, mean_f, std_f, n_samples=100, seed=3)
        d = uq.to_dict()
        assert "mean" in d
        assert "ci_95_lo" in d
        assert d["n_samples"] == 100
