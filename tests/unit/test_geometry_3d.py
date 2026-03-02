"""Tests for 3D geometry support in config + WarpX generator (spec v1.2)."""

from __future__ import annotations

import pytest

from helicon.config.parser import ResolutionConfig, SimConfig
from helicon.config.warpx_generator import generate_warpx_input

# ---------------------------------------------------------------------------
# ResolutionConfig 3D fields
# ---------------------------------------------------------------------------


class TestResolutionConfig3D:
    def test_default_is_2d_rz(self):
        res = ResolutionConfig()
        assert res.geometry == "2d_rz"

    def test_set_3d(self):
        res = ResolutionConfig(geometry="3d", nz=256, nr=128, np_phi=32)
        assert res.geometry == "3d"
        assert res.np_phi == 32

    def test_invalid_geometry_raises(self):
        with pytest.raises(ValueError):
            ResolutionConfig(geometry="cylindrical")  # invalid

    def test_np_phi_default_1(self):
        res = ResolutionConfig()
        assert res.np_phi == 1

    def test_np_phi_positive(self):
        with pytest.raises(ValueError):
            ResolutionConfig(np_phi=0)


# ---------------------------------------------------------------------------
# WarpX generator — 2D-RZ (regression)
# ---------------------------------------------------------------------------


class TestWarpXGenerator2DRZ:
    def _make_config(self, geometry="2d_rz"):
        return SimConfig.model_validate(
            {
                "nozzle": {
                    "coils": [{"z": 0.0, "r": 0.15, "I": 50000}],
                    "domain": {"z_min": -0.5, "z_max": 2.0, "r_max": 0.8},
                    "resolution": {"nz": 128, "nr": 64, "geometry": geometry, "np_phi": 32},
                },
                "plasma": {
                    "n0": 1e18,
                    "T_i_eV": 10.0,
                    "T_e_eV": 10.0,
                    "v_injection_ms": 1e5,
                },
            }
        )

    def test_2d_rz_has_rz_dims(self):
        cfg = self._make_config("2d_rz")
        inp = generate_warpx_input(cfg)
        assert "geometry.dims = RZ" in inp

    def test_2d_rz_no_3d_dims(self):
        cfg = self._make_config("2d_rz")
        inp = generate_warpx_input(cfg)
        assert "geometry.dims = 3" not in inp

    def test_2d_rz_prob_lo_2d(self):
        cfg = self._make_config("2d_rz")
        inp = generate_warpx_input(cfg)
        assert "geometry.prob_lo = -0.5 0.0" in inp

    def test_2d_rz_n_cell_2d(self):
        cfg = self._make_config("2d_rz")
        inp = generate_warpx_input(cfg)
        assert "amr.n_cell = 128 64" in inp


# ---------------------------------------------------------------------------
# WarpX generator — 3D Cartesian
# ---------------------------------------------------------------------------


class TestWarpXGenerator3D:
    def _make_3d_config(self, np_phi=32):
        return SimConfig.model_validate(
            {
                "nozzle": {
                    "coils": [{"z": 0.0, "r": 0.15, "I": 50000}],
                    "domain": {"z_min": -0.5, "z_max": 2.0, "r_max": 0.8},
                    "resolution": {
                        "nz": 128,
                        "nr": 64,
                        "geometry": "3d",
                        "np_phi": np_phi,
                    },
                },
                "plasma": {
                    "n0": 1e18,
                    "T_i_eV": 10.0,
                    "T_e_eV": 10.0,
                    "v_injection_ms": 1e5,
                },
            }
        )

    def test_3d_has_3d_dims(self):
        cfg = self._make_3d_config()
        inp = generate_warpx_input(cfg)
        assert "geometry.dims = 3" in inp

    def test_3d_no_rz_dims(self):
        cfg = self._make_3d_config()
        inp = generate_warpx_input(cfg)
        assert "geometry.dims = RZ" not in inp

    def test_3d_prob_lo_has_three_values(self):
        cfg = self._make_3d_config()
        inp = generate_warpx_input(cfg)
        # prob_lo should have 3 numbers: z_min, -r_max, -r_max
        assert "geometry.prob_lo = -0.5 -0.8 -0.8" in inp

    def test_3d_prob_hi_has_three_values(self):
        cfg = self._make_3d_config()
        inp = generate_warpx_input(cfg)
        assert "geometry.prob_hi = 2.0 0.8 0.8" in inp

    def test_3d_n_cell_includes_phi(self):
        cfg = self._make_3d_config(np_phi=32)
        inp = generate_warpx_input(cfg)
        assert "amr.n_cell = 128 64 32" in inp

    def test_3d_boundary_has_three_fields(self):
        cfg = self._make_3d_config()
        inp = generate_warpx_input(cfg)
        assert "boundary.field_lo = none none pec" in inp

    def test_3d_includes_warning_comment(self):
        cfg = self._make_3d_config()
        inp = generate_warpx_input(cfg)
        assert "3D geometry is expensive" in inp or "Apple Silicon" in inp

    def test_3d_comment_header(self):
        cfg = self._make_3d_config()
        inp = generate_warpx_input(cfg)
        assert "3D Cartesian" in inp

    def test_np_phi_respected(self):
        cfg = self._make_3d_config(np_phi=64)
        inp = generate_warpx_input(cfg)
        assert "amr.n_cell = 128 64 64" in inp


# ---------------------------------------------------------------------------
# Config round-trip: geometry field preserved after YAML/dict parse
# ---------------------------------------------------------------------------


class TestGeometryRoundTrip:
    def test_3d_geometry_preserved_in_model_dump(self):
        res = ResolutionConfig(geometry="3d", np_phi=48)
        d = res.model_dump()
        assert d["geometry"] == "3d"
        assert d["np_phi"] == 48

    def test_2d_rz_preserved_in_model_dump(self):
        res = ResolutionConfig(geometry="2d_rz")
        d = res.model_dump()
        assert d["geometry"] == "2d_rz"

    def test_3d_config_from_preset_sunbird_defaults_2d(self):
        """Existing presets default to 2d_rz — 3D should not break them."""
        cfg = SimConfig.from_preset("sunbird")
        assert cfg.nozzle.resolution.geometry == "2d_rz"
