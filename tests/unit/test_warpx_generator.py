"""Tests for helicon.config.warpx_generator."""

from __future__ import annotations

import tempfile
from pathlib import Path

from helicon.config.parser import (
    CoilConfig,
    DomainConfig,
    NeutralsConfig,
    NozzleConfig,
    PlasmaSourceConfig,
    SimConfig,
)
from helicon.config.warpx_generator import generate_warpx_input, write_warpx_input


def _make_config() -> SimConfig:
    return SimConfig(
        nozzle=NozzleConfig(
            type="converging_diverging",
            coils=[
                CoilConfig(z=0.0, r=0.15, I=50000),
                CoilConfig(z=0.3, r=0.25, I=30000),
            ],
            domain=DomainConfig(z_min=-0.5, z_max=2.0, r_max=0.8),
        ),
        plasma=PlasmaSourceConfig(n0=1e19, T_i_eV=5000, T_e_eV=2000, v_injection_ms=200000),
    )


def test_generate_contains_geometry():
    text = generate_warpx_input(_make_config())
    assert "geometry.dims = RZ" in text
    assert "geometry.prob_lo" in text
    assert "geometry.prob_hi" in text


def test_generate_contains_species():
    text = generate_warpx_input(_make_config())
    assert "particles.species_names" in text
    assert "D_plus" in text
    assert "e_minus" in text


def test_generate_contains_diagnostics():
    text = generate_warpx_input(_make_config())
    assert "diagnostics.diags_names" in text
    assert "openpmd" in text


def test_generate_contains_seed():
    text = generate_warpx_input(_make_config())
    assert "warpx.random_seed" in text


def test_write_creates_file():
    config = _make_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_warpx_input(config, Path(tmpdir) / "warpx_input")
        assert path.exists()
        content = path.read_text()
        assert "geometry.dims = RZ" in content


def test_deterministic_seed():
    """Same config should produce the same seed."""
    config = _make_config()
    text1 = generate_warpx_input(config)
    text2 = generate_warpx_input(config)
    # Extract seed lines
    seed1 = [l for l in text1.splitlines() if "random_seed" in l]
    seed2 = [l for l in text2.splitlines() if "random_seed" in l]
    assert seed1 == seed2


class TestNeutralsInWarpXInput:
    """Tests for Monte Carlo neutrals in WarpX input generation (spec §2.1)."""

    def _config_with_neutrals(self, **kwargs) -> SimConfig:
        neutrals = NeutralsConfig(n_neutral_m3=1e17, **kwargs)
        return SimConfig(
            nozzle=NozzleConfig(
                coils=[CoilConfig(z=0.0, r=0.15, I=50000)],
                domain=DomainConfig(z_min=-0.5, z_max=2.0, r_max=0.8),
            ),
            plasma=PlasmaSourceConfig(
                n0=1e19,
                T_i_eV=5000,
                T_e_eV=2000,
                v_injection_ms=200000,
                neutrals=neutrals,
            ),
        )

    def test_no_neutrals_by_default(self) -> None:
        text = generate_warpx_input(_make_config())
        assert "mcc." not in text
        assert "neutral" not in text.lower()

    def test_neutrals_section_present(self) -> None:
        config = self._config_with_neutrals()
        text = generate_warpx_input(config)
        assert "MONTE CARLO NEUTRALS" in text
        assert "mcc.CX_cross_section" in text

    def test_neutral_density(self) -> None:
        config = self._config_with_neutrals()
        text = generate_warpx_input(config)
        assert "1.000000e+17" in text

    def test_cx_cross_section(self) -> None:
        config = self._config_with_neutrals(cx_cross_section_m2=3e-19)
        text = generate_warpx_input(config)
        assert "3.000000e-19" in text

    def test_ionization_disabled_by_default(self) -> None:
        config = self._config_with_neutrals()
        text = generate_warpx_input(config)
        assert "mcc.do_ionization" not in text

    def test_ionization_enabled_when_set(self) -> None:
        config = self._config_with_neutrals(ionization_cross_section_m2=1e-20)
        text = generate_warpx_input(config)
        assert "mcc.do_ionization = 1" in text
        assert "mcc.ionization_cross_section" in text

    def test_neutral_species_name_in_output(self) -> None:
        config = self._config_with_neutrals(species="D")
        text = generate_warpx_input(config)
        assert "D" in text  # neutral species D appears in the output


class TestAMRConfig:
    def _amr_config(self, max_level: int = 1, ref_ratio: int = 2) -> SimConfig:
        from helicon.config.parser import ResolutionConfig

        return SimConfig(
            nozzle=NozzleConfig(
                type="converging_diverging",
                coils=[CoilConfig(z=0.0, r=0.15, I=50000)],
                domain=DomainConfig(z_min=-0.5, z_max=2.0, r_max=0.8),
                resolution=ResolutionConfig(
                    nz=128,
                    nr=64,
                    amr_max_level=max_level,
                    amr_ref_ratio=ref_ratio,
                ),
            ),
            plasma=PlasmaSourceConfig(
                n0=1e18, T_i_eV=100, T_e_eV=100, v_injection_ms=50000
            ),
        )

    def test_uniform_grid_by_default(self):
        text = generate_warpx_input(_make_config())
        assert "amr.max_level = 0" in text
        assert "amr.ref_ratio" not in text

    def test_amr_max_level_written(self):
        text = generate_warpx_input(self._amr_config(max_level=1))
        assert "amr.max_level = 1" in text

    def test_amr_ref_ratio_written(self):
        text = generate_warpx_input(self._amr_config(max_level=1, ref_ratio=2))
        assert "amr.ref_ratio = 2" in text

    def test_amr_regrid_int_written(self):
        text = generate_warpx_input(self._amr_config(max_level=1))
        assert "amr.regrid_int" in text

    def test_amr_blocking_factor_written(self):
        text = generate_warpx_input(self._amr_config(max_level=1))
        assert "amr.blocking_factor" in text

    def test_refine_plasma_written(self):
        text = generate_warpx_input(self._amr_config(max_level=1))
        assert "warpx.refine_plasma = 1" in text

    def test_no_amr_block_when_disabled(self):
        text = generate_warpx_input(self._amr_config(max_level=0))
        assert "amr.ref_ratio" not in text
        assert "warpx.refine_plasma" not in text

    def test_resolution_config_defaults(self):
        from helicon.config.parser import ResolutionConfig

        r = ResolutionConfig()
        assert r.amr_max_level == 0
        assert r.amr_ref_ratio == 2
        assert r.amr_regrid_int == 10
