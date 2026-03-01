"""Tests for magnozzlex.config module."""

from __future__ import annotations

import tempfile

import pytest
import yaml

from magnozzlex.config.parser import (
    CoilConfig,
    DomainConfig,
    NozzleConfig,
    PlasmaSourceConfig,
    SimConfig,
)
from magnozzlex.config.validators import validate_config


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------
class TestSimConfig:
    def test_minimal_config(self):
        config = SimConfig(
            nozzle=NozzleConfig(
                coils=[CoilConfig(z=0.0, r=0.1, I=1000)],
                domain=DomainConfig(z_min=-0.5, z_max=2.0, r_max=0.8),
            ),
            plasma=PlasmaSourceConfig(n0=1e18, T_i_eV=100, T_e_eV=100, v_injection_ms=50000),
        )
        assert config.nozzle.type == "solenoid"
        assert config.nozzle.resolution.nz == 512
        assert config.timesteps == 50000

    def test_domain_z_order_validation(self):
        with pytest.raises(ValueError, match=r"z_max.*must be greater"):
            DomainConfig(z_min=1.0, z_max=0.0, r_max=0.5)

    def test_positive_radius(self):
        with pytest.raises(ValueError):
            CoilConfig(z=0.0, r=-0.1, I=1000)

    def test_requires_at_least_one_coil(self):
        with pytest.raises(ValueError):
            NozzleConfig(
                coils=[],
                domain=DomainConfig(z_min=-0.5, z_max=2.0, r_max=0.8),
            )

    def test_yaml_roundtrip(self):
        config = SimConfig(
            nozzle=NozzleConfig(
                type="converging_diverging",
                coils=[
                    CoilConfig(z=0.0, r=0.15, I=50000),
                    CoilConfig(z=0.3, r=0.25, I=30000),
                ],
                domain=DomainConfig(z_min=-0.5, z_max=2.0, r_max=0.8),
            ),
            plasma=PlasmaSourceConfig(
                n0=1e19, T_i_eV=5000, T_e_eV=2000, v_injection_ms=200000
            ),
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            config.to_yaml(f.name)
            loaded = SimConfig.from_yaml(f.name)

        assert loaded.nozzle.type == "converging_diverging"
        assert len(loaded.nozzle.coils) == 2
        assert loaded.plasma.n0 == 1e19

    def test_from_yaml_file(self):
        data = {
            "nozzle": {
                "type": "solenoid",
                "coils": [{"z": 0.0, "r": 0.1, "I": 5000}],
                "domain": {"z_min": -0.3, "z_max": 1.0, "r_max": 0.5},
            },
            "plasma": {
                "n0": 1e18,
                "T_i_eV": 100,
                "T_e_eV": 100,
                "v_injection_ms": 50000,
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(data, f)
            f.flush()
            config = SimConfig.from_yaml(f.name)

        assert config.nozzle.coils[0].I == 5000


class TestPresets:
    @pytest.mark.parametrize("name", ["sunbird", "dfd", "ppr"])
    def test_preset_loads(self, name: str):
        config = SimConfig.from_preset(name)
        assert len(config.nozzle.coils) >= 1
        assert config.plasma.n0 > 0

    def test_unknown_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            SimConfig.from_preset("nonexistent")


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------
class TestValidators:
    def test_valid_config_passes(self):
        config = SimConfig(
            nozzle=NozzleConfig(
                coils=[CoilConfig(z=0.0, r=0.15, I=50000)],
                domain=DomainConfig(z_min=-0.5, z_max=2.0, r_max=0.8),
            ),
            plasma=PlasmaSourceConfig(
                n0=1e19, T_i_eV=5000, T_e_eV=2000, v_injection_ms=200000
            ),
        )
        result = validate_config(config)
        assert result.passed

    def test_coil_outside_domain(self):
        config = SimConfig(
            nozzle=NozzleConfig(
                coils=[CoilConfig(z=0.0, r=1.5, I=50000)],  # r > r_max
                domain=DomainConfig(z_min=-0.5, z_max=2.0, r_max=0.8),
            ),
            plasma=PlasmaSourceConfig(
                n0=1e19, T_i_eV=5000, T_e_eV=2000, v_injection_ms=200000
            ),
        )
        result = validate_config(config)
        assert not result.passed
        assert any("exceeds domain" in e for e in result.errors)

    def test_low_mass_ratio_warning(self):
        config = SimConfig(
            nozzle=NozzleConfig(
                coils=[CoilConfig(z=0.0, r=0.15, I=50000)],
                domain=DomainConfig(z_min=-0.5, z_max=2.0, r_max=0.8),
            ),
            plasma=PlasmaSourceConfig(
                n0=1e19,
                T_i_eV=5000,
                T_e_eV=2000,
                v_injection_ms=200000,
                mass_ratio=25,
            ),
        )
        result = validate_config(config)
        assert any("mass ratio" in w.lower() for w in result.warnings)
