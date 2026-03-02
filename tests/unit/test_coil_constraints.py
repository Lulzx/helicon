"""Tests for helicon.optimize.coil_constraints — thermal-structural limits."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import pytest

from helicon.optimize.coil_constraints import (
    CoilConstraintSet,
    CoilStructuralConstraint,
    CoilThermalConstraint,
    import_from_fea,
)

# ---------------------------------------------------------------------------
# Fake coil for testing
# ---------------------------------------------------------------------------


class FakeCoil:
    def __init__(self, I: float, r: float = 0.1):
        self.I = I
        self.r = r


# ---------------------------------------------------------------------------
# CoilThermalConstraint
# ---------------------------------------------------------------------------


class TestCoilThermalConstraint:
    def test_satisfied_below_limit(self):
        tc = CoilThermalConstraint(max_current_A=50000.0)
        assert tc.is_satisfied(FakeCoil(I=10000.0))

    def test_violated_above_limit(self):
        tc = CoilThermalConstraint(max_current_A=50000.0)
        assert not tc.is_satisfied(FakeCoil(I=100000.0))

    def test_violation_zero_when_satisfied(self):
        tc = CoilThermalConstraint(max_current_A=50000.0)
        assert tc.violation(FakeCoil(I=10000.0)) == pytest.approx(0.0)

    def test_violation_positive_when_exceeded(self):
        tc = CoilThermalConstraint(max_current_A=50000.0)
        v = tc.violation(FakeCoil(I=100000.0))
        assert v > 0  # at least one limit exceeded

    def test_current_density(self):
        tc = CoilThermalConstraint(
            max_current_A=50000.0,
            coil_cross_section_m2=1e-4,
        )
        j = tc.current_density(FakeCoil(I=10.0))
        assert j == pytest.approx(1e5, rel=1e-5)  # 10 A / 1e-4 m² = 1e5 A/m²

    def test_to_dict(self):
        tc = CoilThermalConstraint(max_current_A=50000.0)
        d = tc.to_dict()
        assert d["type"] == "thermal"
        assert d["max_current_A"] == 50000.0


# ---------------------------------------------------------------------------
# CoilStructuralConstraint
# ---------------------------------------------------------------------------


class TestCoilStructuralConstraint:
    def test_low_current_satisfied(self):
        sc = CoilStructuralConstraint(max_stress_Pa=500e6, safety_factor=2.0)
        coil = FakeCoil(I=100.0, r=0.1)
        tc = CoilThermalConstraint(max_current_A=1e6, coil_cross_section_m2=1e-4)
        j = tc.current_density(coil)
        assert sc.is_satisfied(coil, j)

    def test_hoop_stress_scales_with_j2_r2(self):
        sc = CoilStructuralConstraint()
        c1 = FakeCoil(I=1e4, r=0.1)
        c2 = FakeCoil(I=1e4, r=0.2)
        j = 1e8  # [A/m²]
        sigma1 = sc.hoop_stress(c1, j)
        sigma2 = sc.hoop_stress(c2, j)
        # σ ∝ r² → sigma2 / sigma1 = (0.2/0.1)² = 4
        assert sigma2 / sigma1 == pytest.approx(4.0, rel=1e-5)

    def test_to_dict(self):
        sc = CoilStructuralConstraint(max_stress_Pa=300e6)
        d = sc.to_dict()
        assert d["type"] == "structural"
        assert d["max_stress_Pa"] == 300e6


# ---------------------------------------------------------------------------
# CoilConstraintSet
# ---------------------------------------------------------------------------


class TestCoilConstraintSet:
    def _make_set(self):
        return CoilConstraintSet(
            thermal=[CoilThermalConstraint(max_current_A=50000.0)],
            structural=[CoilStructuralConstraint(max_stress_Pa=500e6)],
        )

    def test_all_satisfied(self):
        cs = self._make_set()
        ok, msgs = cs.check_all([FakeCoil(I=10000.0)])
        assert ok
        assert len(msgs) == 0

    def test_current_violation_message(self):
        cs = self._make_set()
        ok, msgs = cs.check_all([FakeCoil(I=100000.0)])
        assert not ok
        assert len(msgs) >= 1
        assert "thermal violation" in msgs[0]

    def test_total_violation_zero_when_ok(self):
        cs = self._make_set()
        v = cs.total_violation([FakeCoil(I=10000.0)])
        assert v == pytest.approx(0.0)

    def test_total_violation_positive_when_not_ok(self):
        cs = self._make_set()
        v = cs.total_violation([FakeCoil(I=200000.0)])
        assert v > 0

    def test_multiple_coils(self):
        cs = self._make_set()
        coils = [FakeCoil(I=10000.0), FakeCoil(I=200000.0)]
        ok, msgs = cs.check_all(coils)
        assert not ok
        # Only coil 1 violates
        assert any("Coil 1" in m for m in msgs)

    def test_save_json_roundtrip(self):
        cs = self._make_set()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "constraints.json"
            cs.save_json(p)
            cs2 = import_from_fea(p)
        assert len(cs2.thermal) == len(cs.thermal)
        assert cs2.thermal[0].max_current_A == cs.thermal[0].max_current_A


# ---------------------------------------------------------------------------
# import_from_fea
# ---------------------------------------------------------------------------


class TestImportFromFEA:
    def test_json_import(self):
        data = {
            "thermal": [
                {
                    "max_current_A": 80000,
                    "max_current_density_Am2": 5e7,
                    "max_temp_K": 20,
                    "coil_cross_section_m2": 2e-3,
                }
            ],
            "structural": [{"max_stress_Pa": 300e6, "safety_factor": 2.5}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "limits.json"
            p.write_text(json.dumps(data))
            cs = import_from_fea(p)
        assert cs.thermal[0].max_current_A == 80000.0
        assert cs.structural[0].max_stress_Pa == 300e6
        assert cs.structural[0].safety_factor == 2.5

    def test_csv_import(self):
        rows = [
            {
                "coil_id": "0",
                "max_current_A": "50000",
                "max_current_density_Am2": "1e8",
                "max_temp_K": "20",
                "max_stress_Pa": "500e6",
                "safety_factor": "2.0",
            },
            {
                "coil_id": "1",
                "max_current_A": "80000",
                "max_current_density_Am2": "8e7",
                "max_temp_K": "20",
                "max_stress_Pa": "400e6",
                "safety_factor": "2.0",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "limits.csv"
            with p.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            cs = import_from_fea(p)
        assert len(cs.thermal) == 2
        assert cs.thermal[0].max_current_A == 50000.0
        assert cs.thermal[1].max_current_A == 80000.0
        assert len(cs.structural) == 2

    def test_unsupported_extension_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "limits.xlsx"
            p.write_text("dummy")
            with pytest.raises(ValueError, match="Unsupported"):
                import_from_fea(p)
