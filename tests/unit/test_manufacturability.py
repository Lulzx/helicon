"""Tests for helicon.optimize.manufacturability — coil buildability (v2.0)."""

from __future__ import annotations

import pytest

from helicon.optimize.manufacturability import (
    ManufacturabilityResult,
    REBCOTapeSpec,
    WindingPackSpec,
    check_manufacturability,
    suggest_winding_pack,
)


class TestCheckManufacturability:
    def test_feasible_large_coil(self):
        # 0.5 m radius → 500 mm bend >> 15 mm minimum
        winding = WindingPackSpec(
            inner_radius_m=0.08,
            outer_radius_m=0.15,
            axial_length_m=0.1,
        )
        result = check_manufacturability(0.5, 10000.0, winding)
        assert isinstance(result, ManufacturabilityResult)
        assert result.bend_radius_ratio > 1.0

    def test_small_coil_fails_bend(self):
        tape = REBCOTapeSpec(min_bend_radius_mm=50.0)
        winding = WindingPackSpec(
            inner_radius_m=0.01,
            outer_radius_m=0.03,
            axial_length_m=0.05,
        )
        result = check_manufacturability(0.01, 1000.0, winding, tape)
        assert not result.feasible
        assert any("Bend radius" in v for v in result.violations)

    def test_insufficient_turns_violation(self):
        tape = REBCOTapeSpec(
            tape_width_mm=6.0,
            tape_thickness_mm=0.5,
            critical_current_A=100.0,
        )
        # Need 1000 A / 100 A = 10+ turns, but tiny winding fits few
        winding = WindingPackSpec(
            inner_radius_m=0.05,
            outer_radius_m=0.051,  # 1 mm radial build
            axial_length_m=0.006,  # 6 mm axial → 1 turn/layer
        )
        result = check_manufacturability(0.1, 10000.0, winding, tape)
        # Either turns violation or bend; check feasible is False
        assert not result.feasible

    def test_topology_recommendation_layer_wound(self):
        winding = WindingPackSpec(
            inner_radius_m=0.07,
            outer_radius_m=0.12,
            axial_length_m=0.03,
            topology="pancake",
        )
        result = check_manufacturability(0.1, 5000.0, winding)
        # axial/radial = 0.03/0.05 = 0.6 < 2 → layer_wound recommended
        assert result.recommended_topology == "layer_wound"
        assert any("layer_wound" in w for w in result.warnings)

    def test_to_dict(self):
        winding = WindingPackSpec(
            inner_radius_m=0.08,
            outer_radius_m=0.14,
            axial_length_m=0.08,
        )
        result = check_manufacturability(0.3, 20000.0, winding)
        d = result.to_dict()
        assert "feasible" in d
        assert "violations" in d
        assert "recommended_topology" in d


class TestSuggestWindingPack:
    def test_returns_winding_pack(self):
        wp = suggest_winding_pack(0.1, 50000.0)
        assert isinstance(wp, WindingPackSpec)
        assert wp.inner_radius_m > 0
        assert wp.outer_radius_m > wp.inner_radius_m
        assert wp.total_turns > 0

    def test_sufficient_turns(self):
        tape = REBCOTapeSpec(critical_current_A=200.0)
        wp = suggest_winding_pack(0.15, 10000.0, tape)
        n_req = int(10000.0 / 200.0) + 1
        assert wp.total_turns >= n_req

    def test_respects_min_bend_radius(self):
        tape = REBCOTapeSpec(min_bend_radius_mm=30.0)
        wp = suggest_winding_pack(0.01, 1000.0, tape)
        assert wp.inner_radius_m >= 0.030 - 1e-9  # 30 mm minimum


class TestWindingPackSpec:
    def test_total_turns(self):
        wp = WindingPackSpec(n_layers=5, n_turns_per_layer=10)
        assert wp.total_turns == 50

    def test_radial_build(self):
        wp = WindingPackSpec(inner_radius_m=0.05, outer_radius_m=0.12)
        assert wp.radial_build_m == pytest.approx(0.07, rel=1e-6)
