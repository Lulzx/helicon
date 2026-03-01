"""Variance-based Sobol sensitivity analysis.

Implements the Saltelli (2010) estimator for first-order (S1) and
total-order (ST) Sobol indices using two independent sample matrices
A and B of shape (N, k).

Reference
---------
Saltelli, A. et al. (2010). Variance based sensitivity analysis of model
output. Design and estimator for the total sensitivity index.
*Computer Physics Communications*, 181(2), 259-270.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SobolResult:
    """Variance-based Sobol sensitivity indices.

    Attributes
    ----------
    param_names : list of str
    S1 : np.ndarray, shape (n_params,)
        First-order Sobol indices (main effect of each parameter alone).
    ST : np.ndarray, shape (n_params,)
        Total-order Sobol indices (main effect + all interactions).
    """

    param_names: list[str]
    S1: np.ndarray
    ST: np.ndarray

    def summary(self) -> str:
        lines = [
            "Sobol Sensitivity Indices:",
            f"  {'Parameter':<30} {'S1':>8}  {'ST':>8}",
        ]
        for name, s1, st in zip(self.param_names, self.S1, self.ST):
            lines.append(f"  {name:<30} {s1:>8.4f}  {st:>8.4f}")
        return "\n".join(lines)


def saltelli_sample(
    n_samples: int,
    bounds: np.ndarray | list,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two independent sample matrices for Sobol analysis.

    Parameters
    ----------
    n_samples : int
        Base sample size N. Total function evaluations needed: N * (k + 2).
    bounds : array-like, shape (k, 2)
        ``[[low_0, high_0], ...]`` for each parameter.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    A, B : np.ndarray, each shape (n_samples, k)
        Two independent uniform samples scaled to parameter bounds.
    """
    bounds = np.asarray(bounds, dtype=float)
    k = bounds.shape[0]
    low = bounds[:, 0]
    high = bounds[:, 1]

    rng = np.random.default_rng(seed)
    A = low + rng.uniform(size=(n_samples, k)) * (high - low)
    B = low + rng.uniform(size=(n_samples, k)) * (high - low)
    return A, B


def compute_sobol(
    f: callable,
    n_samples: int,
    bounds: np.ndarray | list,
    param_names: list[str],
    *,
    seed: int = 0,
) -> SobolResult:
    """Compute Sobol first-order and total sensitivity indices.

    The function ``f`` is called with arrays of shape ``(N, k)`` and must
    return a 1-D array of shape ``(N,)``.

    Total function evaluations: ``n_samples * (k + 2)``.

    Parameters
    ----------
    f : callable
        Vectorized objective: ``f(X: np.ndarray) -> np.ndarray``.
    n_samples : int
        Base sample count N.
    bounds : array-like, shape (k, 2)
    param_names : list of str
        Names for the ``k`` parameters (for reporting).
    seed : int

    Returns
    -------
    SobolResult
    """
    bounds = np.asarray(bounds, dtype=float)
    k = bounds.shape[0]
    if len(param_names) != k:
        raise ValueError(f"len(param_names)={len(param_names)} != k={k}")

    A, B = saltelli_sample(n_samples, bounds, seed=seed)
    yA = np.asarray(f(A), dtype=float)
    yB = np.asarray(f(B), dtype=float)

    var_total = np.var(np.concatenate([yA, yB]))
    if var_total < 1e-30:
        return SobolResult(
            param_names=param_names,
            S1=np.zeros(k),
            ST=np.zeros(k),
        )

    S1 = np.zeros(k)
    ST = np.zeros(k)

    for i in range(k):
        # AB_i: copy of A with column i replaced by B's column i
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        yAB_i = np.asarray(f(AB_i), dtype=float)

        # Saltelli (2010) estimators
        S1[i] = float(np.mean(yB * (yAB_i - yA)) / var_total)
        ST[i] = float(np.mean((yA - yAB_i) ** 2) / (2.0 * var_total))

    # Clip: finite-N estimation can produce small negatives
    S1 = np.clip(S1, 0.0, 1.0)
    ST = np.clip(ST, 0.0, 1.0)

    return SobolResult(param_names=param_names, S1=S1, ST=ST)
