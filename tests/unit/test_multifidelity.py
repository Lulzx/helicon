"""Tests for helicon.optimize.multifidelity — multi-fidelity pipeline (v2.0)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from helicon.optimize.multifidelity import (
    FidelityConfig,
    MultiFidelityPipeline,
    MultiFidelityResult,
)


def _make_candidates(n: int = 10) -> list[dict]:
    """Make n simple test candidates."""
    import numpy as np

    rng = np.random.default_rng(0)
    candidates = []
    for _ in range(n):
        candidates.append(
            {
                "coil_z": float(rng.uniform(-0.1, 0.1)),
                "coil_r": float(rng.uniform(0.05, 0.20)),
                "coil_I": float(rng.uniform(1000, 50000)),
                "z_max": float(rng.uniform(0.5, 2.0)),
                "n0": 1e18,
                "T_i_eV": 100.0,
                "T_e_eV": 100.0,
                "v_inj_ms": 50000.0,
            }
        )
    return candidates


class TestFidelityConfig:
    def test_defaults(self):
        cfg = FidelityConfig()
        assert cfg.tier2_threshold == 0.5
        assert cfg.top_k_to_tier3 == 3
        assert cfg.dry_run_tier3 is True


class TestMultiFidelityPipeline:
    def test_run_returns_result(self):
        pipeline = MultiFidelityPipeline(
            fidelity_config=FidelityConfig(
                tier2_threshold=0.0,  # promote everything
                tier3_threshold=0.0,
                top_k_to_tier3=2,
                dry_run_tier3=True,
            )
        )
        candidates = _make_candidates(5)
        result = pipeline.run(candidates)
        assert isinstance(result, MultiFidelityResult)

    def test_tier1_results_populated(self):
        pipeline = MultiFidelityPipeline()
        candidates = _make_candidates(8)
        result = pipeline.run(candidates)
        assert len(result.tier1_results) == 8

    def test_scores_normalised(self):
        pipeline = MultiFidelityPipeline()
        candidates = _make_candidates(6)
        result = pipeline.run(candidates)
        scores = [r.score for r in result.tier1_results]
        assert min(scores) >= 0.0
        assert max(scores) <= 1.0 + 1e-9

    def test_best_candidate_set(self):
        pipeline = MultiFidelityPipeline(
            fidelity_config=FidelityConfig(
                tier2_threshold=0.0,
                tier3_threshold=0.0,
                top_k_to_tier3=2,
                dry_run_tier3=True,
            )
        )
        candidates = _make_candidates(4)
        result = pipeline.run(candidates)
        assert result.best_candidate_id != ""

    def test_dry_run_tier3_files_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = MultiFidelityPipeline(
                fidelity_config=FidelityConfig(
                    tier2_threshold=0.0,
                    tier3_threshold=0.0,
                    top_k_to_tier3=2,
                    dry_run_tier3=True,
                ),
                output_dir=tmp,
            )
            candidates = _make_candidates(4)
            result = pipeline.run(candidates)
            if result.tier3_results:
                # Should have written tier3_meta.json files
                tier3_dirs = list(Path(tmp).glob("tier3_*"))
                assert len(tier3_dirs) > 0

    def test_to_dict(self):
        pipeline = MultiFidelityPipeline()
        result = pipeline.run(_make_candidates(3))
        d = result.to_dict()
        assert "best_candidate_id" in d
        assert "n_tier1" in d
        assert d["n_tier1"] == 3

    def test_eta_d_objective(self):
        pipeline = MultiFidelityPipeline(
            fidelity_config=FidelityConfig(tier2_threshold=0.0, top_k_to_tier3=1)
        )
        result = pipeline.run(_make_candidates(4), objective="eta_d")
        assert isinstance(result, MultiFidelityResult)
