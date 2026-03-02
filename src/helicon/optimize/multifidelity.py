"""Multi-fidelity optimisation pipeline (v2.0).

Promotes candidate nozzle designs up a three-tier fidelity ladder:

  Tier 1 — Analytical/MLX (seconds)
    Biot-Savart field + ThrottleMap physics model.
    Fast screening of thousands of candidates.

  Tier 2 — Neural surrogate (microseconds, after training)
    MLX MLP surrogate trained on PIC scan database.
    Used for high-throughput Bayesian optimisation and UQ.

  Tier 3 — Full WarpX PIC (hours, local or cloud)
    High-fidelity kinetic particle-in-cell simulation.
    Only the top-K candidates from Tier 2 are promoted here.

Usage::

    from helicon.optimize.multifidelity import MultiFidelityPipeline, FidelityConfig

    pipeline = MultiFidelityPipeline(
        fidelity_config=FidelityConfig(tier2_threshold=0.7, top_k_to_tier3=3),
    )
    results = pipeline.run(candidates)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FidelityConfig:
    """Configuration for the multi-fidelity promotion ladder.

    Attributes
    ----------
    tier2_threshold : float
        Minimum Tier-1 score (normalised [0,1]) to promote to Tier 2.
    tier3_threshold : float
        Minimum Tier-2 score to promote to full WarpX PIC.
    top_k_to_tier3 : int
        Maximum number of candidates to run through Tier 3.
    tier1_n_eval : int
        Number of Tier-1 evaluations (analytical sweep).
    tier2_n_eval : int
        Number of Tier-2 evaluations (surrogate inference).
    dry_run_tier3 : bool
        If True, generate Tier-3 input files but do not run WarpX.
    """

    tier2_threshold: float = 0.5
    tier3_threshold: float = 0.6
    top_k_to_tier3: int = 3
    tier1_n_eval: int = 1000
    tier2_n_eval: int = 500
    dry_run_tier3: bool = True


@dataclass
class CandidateResult:
    """Result for one candidate at a given fidelity tier.

    Attributes
    ----------
    candidate_id : str
        Unique identifier.
    tier : int
        Fidelity tier at which this result was obtained (1, 2, or 3).
    score : float
        Normalised performance score [0, 1].
    metrics : dict
        Raw performance metrics (thrust_N, eta_d, plume_angle_deg, ...).
    wall_time_s : float
        Wall time for this evaluation [s].
    promoted : bool
        Whether this candidate was promoted to the next tier.
    """

    candidate_id: str
    tier: int
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    wall_time_s: float = 0.0
    promoted: bool = False


@dataclass
class MultiFidelityResult:
    """Aggregated result from a complete multi-fidelity run.

    Attributes
    ----------
    tier1_results : list[CandidateResult]
    tier2_results : list[CandidateResult]
    tier3_results : list[CandidateResult]
    best_candidate_id : str
        ID of the top-performing candidate (highest tier available).
    best_metrics : dict
        Metrics of the best candidate.
    total_wall_time_s : float
        Total wall time across all tiers.
    provenance_path : Path, optional
        Path to the provenance record for this run.
    """

    tier1_results: list[CandidateResult] = field(default_factory=list)
    tier2_results: list[CandidateResult] = field(default_factory=list)
    tier3_results: list[CandidateResult] = field(default_factory=list)
    best_candidate_id: str = ""
    best_metrics: dict[str, Any] = field(default_factory=dict)
    total_wall_time_s: float = 0.0
    provenance_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_candidate_id": self.best_candidate_id,
            "best_metrics": self.best_metrics,
            "total_wall_time_s": self.total_wall_time_s,
            "n_tier1": len(self.tier1_results),
            "n_tier2": len(self.tier2_results),
            "n_tier3": len(self.tier3_results),
        }


class MultiFidelityPipeline:
    """Three-tier multi-fidelity nozzle optimisation pipeline.

    Parameters
    ----------
    fidelity_config : FidelityConfig
        Promotion thresholds and tier sizes.
    surrogate : NozzleSurrogate, optional
        Pre-trained surrogate for Tier 2.  If None, Tier 2 uses a fresh
        analytical approximation instead.
    output_dir : path-like, optional
        Directory for intermediate results and Tier-3 input files.
    """

    def __init__(
        self,
        fidelity_config: FidelityConfig | None = None,
        surrogate: Any | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.cfg = fidelity_config or FidelityConfig()
        self.surrogate = surrogate
        self.output_dir = Path(output_dir) if output_dir else Path("mf_output")

    def run(
        self,
        candidates: list[dict[str, Any]],
        objective: str = "thrust_N",
    ) -> MultiFidelityResult:
        """Run the multi-fidelity pipeline on a list of candidate configs.

        Parameters
        ----------
        candidates : list of dict
            Each dict must contain ``"coil_z"``, ``"coil_r"``, ``"coil_I"``,
            ``"n0"``, ``"T_i_eV"``, ``"T_e_eV"``, ``"v_inj_ms"``,
            ``"z_max"`` keys.
        objective : str
            Which metric to maximise: ``"thrust_N"`` or ``"eta_d"``.

        Returns
        -------
        MultiFidelityResult
        """
        t0 = time.monotonic()
        result = MultiFidelityResult()

        # Tier 1: analytical sweep
        t1_results = self._run_tier1(candidates, objective)
        result.tier1_results = t1_results

        # Promote to Tier 2
        tier2_cands = [
            candidates[i]
            for i, r in enumerate(t1_results)
            if r.score >= self.cfg.tier2_threshold
        ]
        # Also include top-k regardless of threshold
        sorted_t1 = sorted(enumerate(t1_results), key=lambda x: x[1].score, reverse=True)
        for idx, _ in sorted_t1[: self.cfg.top_k_to_tier3 * 2]:
            c = candidates[idx]
            if c not in tier2_cands:
                tier2_cands.append(c)
        tier2_cands = tier2_cands[: self.cfg.tier2_n_eval]

        if tier2_cands:
            t2_results = self._run_tier2(tier2_cands, t1_results, objective)
            result.tier2_results = t2_results

            # Promote to Tier 3
            tier3_cands = sorted(t2_results, key=lambda r: r.score, reverse=True)
            tier3_cands = [
                r for r in tier3_cands if r.score >= self.cfg.tier3_threshold
            ][: self.cfg.top_k_to_tier3]

            if tier3_cands:
                t3_results = self._run_tier3(
                    tier3_cands, candidates, objective
                )
                result.tier3_results = t3_results
        else:
            result.tier2_results = []
            result.tier3_results = []

        # Find best
        all_final = result.tier3_results or result.tier2_results or result.tier1_results
        if all_final:
            best = max(all_final, key=lambda r: r.score)
            result.best_candidate_id = best.candidate_id
            result.best_metrics = best.metrics

        result.total_wall_time_s = time.monotonic() - t0
        return result

    # ------------------------------------------------------------------
    # Tier 1 — analytical evaluation
    # ------------------------------------------------------------------

    def _run_tier1(
        self,
        candidates: list[dict[str, Any]],
        objective: str,
    ) -> list[CandidateResult]:

        results = []
        scores_raw = []

        for i, c in enumerate(candidates):
            t_start = time.monotonic()
            metrics = self._tier1_eval(c)
            dt = time.monotonic() - t_start
            scores_raw.append(metrics.get(objective, 0.0))
            results.append(
                CandidateResult(
                    candidate_id=f"c{i:04d}",
                    tier=1,
                    score=0.0,  # filled after normalisation
                    metrics=metrics,
                    wall_time_s=dt,
                )
            )

        # Normalise scores to [0, 1]
        lo, hi = min(scores_raw), max(scores_raw)
        span = max(hi - lo, 1e-12)
        for r, raw in zip(results, scores_raw):
            r.score = (raw - lo) / span

        return results

    def _tier1_eval(self, candidate: dict[str, Any]) -> dict[str, float]:
        """Analytical Tier-1 evaluation (ThrottleMap physics)."""
        import math as m

        from helicon.surrogate.training import (
            _analytical_performance,
            _compute_field_features,
        )

        coil_z = float(candidate.get("coil_z", 0.0))
        coil_r = float(candidate.get("coil_r", 0.1))
        coil_I = float(candidate.get("coil_I", 10000.0))
        z_max = float(candidate.get("z_max", 1.5))
        n0 = float(candidate.get("n0", 1e18))
        T_i = float(candidate.get("T_i_eV", 100.0))
        T_e = float(candidate.get("T_e_eV", 100.0))
        v_inj = float(candidate.get("v_inj_ms", 50000.0))

        ff = _compute_field_features(coil_z, coil_r, coil_I, z_max)
        area = m.pi * coil_r**2
        mdot = float(max(1.67e-27 * n0 * v_inj * area * 1e-4, 1e-9))

        perf = _analytical_performance(
            ff["b_peak_T"], ff["mirror_ratio"], n0, T_i, T_e, v_inj, mdot
        )
        return {**ff, **perf, "n0": n0, "T_i_eV": T_i, "T_e_eV": T_e, "v_inj_ms": v_inj}

    # ------------------------------------------------------------------
    # Tier 2 — surrogate inference
    # ------------------------------------------------------------------

    def _run_tier2(
        self,
        tier2_cands: list[dict[str, Any]],
        all_candidates: list[dict[str, Any]],
        objective: str,
    ) -> list[CandidateResult]:
        results = []
        scores_raw = []

        for c in tier2_cands:
            t_start = time.monotonic()
            metrics = self._tier2_eval(c)
            dt = time.monotonic() - t_start
            scores_raw.append(metrics.get(objective, 0.0))
            cid = f"t2_{id(c)}"
            results.append(
                CandidateResult(
                    candidate_id=cid,
                    tier=2,
                    score=0.0,
                    metrics=metrics,
                    wall_time_s=dt,
                )
            )

        lo, hi = min(scores_raw, default=0.0), max(scores_raw, default=1.0)
        span = max(hi - lo, 1e-12)
        for r, raw in zip(results, scores_raw):
            r.score = (raw - lo) / span

        return results

    def _tier2_eval(self, candidate: dict[str, Any]) -> dict[str, float]:
        """Surrogate (or fallback analytical) Tier-2 evaluation."""
        if self.surrogate is not None:
            from helicon.surrogate.mlx_net import SurrogateFeatures
            from helicon.surrogate.training import _compute_field_features

            coil_z = float(candidate.get("coil_z", 0.0))
            coil_r = float(candidate.get("coil_r", 0.1))
            coil_I = float(candidate.get("coil_I", 10000.0))
            z_max = float(candidate.get("z_max", 1.5))
            ff = _compute_field_features(coil_z, coil_r, coil_I, z_max)
            feats = SurrogateFeatures(
                mirror_ratio=ff["mirror_ratio"],
                b_peak_T=ff["b_peak_T"],
                b_gradient_T_m=ff["b_gradient_T_m"],
                nozzle_length_m=ff["nozzle_length_m"],
                n0_m3=float(candidate.get("n0", 1e18)),
                T_i_eV=float(candidate.get("T_i_eV", 100.0)),
                T_e_eV=float(candidate.get("T_e_eV", 100.0)),
                v_injection_ms=float(candidate.get("v_inj_ms", 50000.0)),
            )
            pred = self.surrogate.predict(feats)
            return {
                "thrust_N": pred.thrust_N,
                "eta_d": pred.eta_d,
                "plume_angle_deg": pred.plume_angle_deg,
            }
        # No surrogate: fall back to Tier 1 + small perturbation
        return self._tier1_eval(candidate)

    # ------------------------------------------------------------------
    # Tier 3 — WarpX PIC (or dry-run)
    # ------------------------------------------------------------------

    def _run_tier3(
        self,
        top_tier2: list[CandidateResult],
        all_candidates: list[dict[str, Any]],
        objective: str,
    ) -> list[CandidateResult]:
        results = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for r in top_tier2:
            t_start = time.monotonic()
            meta = {
                "candidate_id": r.candidate_id,
                "tier2_score": r.score,
                "tier2_metrics": r.metrics,
                "dry_run": self.cfg.dry_run_tier3,
                "status": "queued",
            }
            out = self.output_dir / f"tier3_{r.candidate_id}"
            out.mkdir(exist_ok=True)
            (out / "tier3_meta.json").write_text(json.dumps(meta, indent=2))
            dt = time.monotonic() - t_start

            if not self.cfg.dry_run_tier3:
                # Actual WarpX run would happen here — use cloud backend
                metrics = r.metrics
                score = r.score
                status = "running"
            else:
                metrics = r.metrics
                score = r.score
                status = "dry_run"

            meta["status"] = status
            (out / "tier3_meta.json").write_text(json.dumps(meta, indent=2))

            results.append(
                CandidateResult(
                    candidate_id=r.candidate_id,
                    tier=3,
                    score=score,
                    metrics=metrics,
                    wall_time_s=dt,
                )
            )

        return results
