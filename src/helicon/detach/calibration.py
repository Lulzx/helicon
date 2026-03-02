"""Calibrate DetachmentOnsetModel weights against labelled training data.

Given a dataset of (M_A, β_e, Λᵢ, is_detached) records — from WarpX
PIC runs or validated experiments — fits the composite-score weights by
minimising binary cross-entropy loss subject to the simplex constraint
(w₁ + w₂ + w₃ = 1, wᵢ ≥ 0) plus a calibrated detection threshold τ.

The calibrated weights replace the heuristic defaults (0.45, 0.30, 0.25)
with data-driven estimates that optimally predict detachment onset.

Usage::

    from helicon.detach.calibration import DetachmentCalibrator

    records = DetachmentCalibrator.generate_synthetic_data(n_samples=500)
    cal = DetachmentCalibrator()
    result = cal.fit(records)
    model = DetachmentOnsetModel(**result.to_model_kwargs())
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationRecord:
    """Single labelled sample for model calibration.

    Attributes
    ----------
    alfven_mach : float
        Alfvén Mach number M_A = v_z / v_A.
    electron_beta : float
        Electron β_e = n k T_e / (B²/2μ₀).
    ion_magnetization : float
        Ion demagnetization Λᵢ = r_Lᵢ / L_B.
    is_detached : bool
        Ground-truth detachment label (from WarpX or experiment).
    source : str
        Provenance tag (e.g. ``"warpx"``, ``"experiment"``, ``"synthetic"``).
    """

    alfven_mach: float
    electron_beta: float
    ion_magnetization: float
    is_detached: bool
    source: str = ""


@dataclass
class CalibrationResult:
    """Result of fitting weights to labelled data.

    Attributes
    ----------
    w_alfven, w_beta, w_ion_mag : float
        Optimised criterion weights (sum to 1).
    score_threshold : float
        Optimal detection threshold τ ∈ (0, 1).
    n_samples : int
        Number of training records used.
    log_loss : float
        Final binary cross-entropy loss.
    accuracy : float
        Classification accuracy on training data.
    """

    w_alfven: float
    w_beta: float
    w_ion_mag: float
    score_threshold: float
    n_samples: int
    log_loss: float
    accuracy: float

    def to_model_kwargs(self) -> dict:
        """Return kwargs for :class:`~helicon.detach.model.DetachmentOnsetModel`."""
        return {
            "w_alfven": self.w_alfven,
            "w_beta": self.w_beta,
            "w_ion_mag": self.w_ion_mag,
            "score_detached": self.score_threshold,
        }

    def summary(self) -> str:
        return (
            f"CalibrationResult:\n"
            f"  weights:   w_A={self.w_alfven:.3f}  w_β={self.w_beta:.3f}"
            f"  w_Λ={self.w_ion_mag:.3f}\n"
            f"  threshold: τ={self.score_threshold:.3f}\n"
            f"  accuracy:  {self.accuracy:.1%}  (n={self.n_samples})\n"
            f"  log_loss:  {self.log_loss:.4f}"
        )


class DetachmentCalibrator:
    """Fit DetachmentOnsetModel weights to labelled training data.

    Minimises binary cross-entropy loss with a soft logistic decision
    boundary, subject to the simplex constraint on weights.  Uses
    ``scipy.optimize.minimize`` (SLSQP), which is in the core dep chain.

    Parameters
    ----------
    beta_crit : float
        β_e threshold used in criterion normalisation (default 0.15).
    sharpness : float
        Logistic sharpness k; higher = sharper decision boundary.
        Default 10.
    """

    def __init__(self, beta_crit: float = 0.15, sharpness: float = 10.0) -> None:
        self.beta_crit = beta_crit
        self.sharpness = sharpness

    def _features(
        self,
        M_A: np.ndarray,
        beta_e: np.ndarray,
        lambda_i: np.ndarray,
        w: np.ndarray,
    ) -> np.ndarray:
        """Compute composite score from criterion arrays and weight vector."""
        bc = self.beta_crit
        f_A = np.clip(M_A / 2.0, 0.0, 1.0)
        f_beta = np.clip(beta_e / (2.0 * bc), 0.0, 1.0)
        f_ion = np.clip(lambda_i / 2.0, 0.0, 1.0)
        return w[0] * f_A + w[1] * f_beta + w[2] * f_ion

    def _bce_loss(
        self,
        params: np.ndarray,
        M_A: np.ndarray,
        beta_e: np.ndarray,
        lambda_i: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Binary cross-entropy with logistic score-to-probability mapping."""
        w = params[:3]
        tau = params[3]
        k = self.sharpness
        scores = self._features(M_A, beta_e, lambda_i, w)
        logits = k * (scores - tau)
        # Numerically stable log-sigmoid
        log_p = -np.logaddexp(0.0, -logits)
        log_1mp = -np.logaddexp(0.0, logits)
        bce = -np.mean(y * log_p + (1.0 - y) * log_1mp)
        return float(bce)

    def fit(self, records: list[CalibrationRecord]) -> CalibrationResult:
        """Fit weights by minimising binary cross-entropy.

        Parameters
        ----------
        records : list[CalibrationRecord]
            Labelled training samples.

        Returns
        -------
        CalibrationResult
            Optimised weights, threshold, and diagnostics.
        """
        from scipy.optimize import minimize

        M_A = np.array([r.alfven_mach for r in records])
        beta_e = np.array([r.electron_beta for r in records])
        lambda_i = np.array([r.ion_magnetization for r in records])
        y = np.array([float(r.is_detached) for r in records])

        def loss(params: np.ndarray) -> float:
            return self._bce_loss(params, M_A, beta_e, lambda_i, y)

        constraints = [{"type": "eq", "fun": lambda p: p[0] + p[1] + p[2] - 1.0}]
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.01, 0.99)]
        x0 = np.array([0.45, 0.30, 0.25, 0.50])

        result = minimize(
            loss, x0, method="SLSQP",
            constraints=constraints, bounds=bounds,
            options={"ftol": 1e-10, "maxiter": 2000},
        )

        w_opt = result.x[:3]
        tau_opt = float(result.x[3])

        scores = self._features(M_A, beta_e, lambda_i, w_opt)
        preds = (scores > tau_opt).astype(float)
        accuracy = float(np.mean(preds == y))

        return CalibrationResult(
            w_alfven=float(w_opt[0]),
            w_beta=float(w_opt[1]),
            w_ion_mag=float(w_opt[2]),
            score_threshold=tau_opt,
            n_samples=len(records),
            log_loss=float(result.fun),
            accuracy=accuracy,
        )

    @staticmethod
    def generate_synthetic_data(
        n_samples: int = 500,
        seed: int = 0,
    ) -> list[CalibrationRecord]:
        """Generate physics-motivated synthetic training data.

        **Oracle label** (Merino-Ahedo 2011 criterion):
        Detached if M_A > 1  OR  (β_e > 0.15 AND Λᵢ > 0.5).

        A 5% label-noise rate is added to simulate measurement uncertainty.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        list[CalibrationRecord]
        """
        rng = np.random.default_rng(seed)

        M_A = rng.uniform(0.1, 3.0, n_samples)
        beta_e = rng.uniform(0.001, 0.50, n_samples)
        lambda_i = rng.uniform(0.0, 2.0, n_samples)

        # Oracle: Alfvénic OR (β-driven AND ion-demagnetized)
        is_detached = (M_A > 1.0) | ((beta_e > 0.15) & (lambda_i > 0.5))

        # 5% label flip (measurement uncertainty)
        noise = rng.random(n_samples) < 0.05
        is_detached = is_detached ^ noise

        return [
            CalibrationRecord(
                alfven_mach=float(M_A[i]),
                electron_beta=float(beta_e[i]),
                ion_magnetization=float(lambda_i[i]),
                is_detached=bool(is_detached[i]),
                source="synthetic",
            )
            for i in range(n_samples)
        ]
