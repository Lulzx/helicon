"""Tests for the helicon top-level public API (spec §6.2).

Verifies that the API described in the spec is accessible and works:

    import helicon
    config = helicon.Config.from_preset("sunbird")
    bfield = helicon.fields.compute(config.nozzle)
    result = helicon.run(config, dry_run=True)
    metrics = helicon.postprocess(result.output_dir)
"""

from __future__ import annotations

import tempfile

import helicon


class TestVersion:
    def test_version_string(self):
        assert isinstance(helicon.__version__, str)
        assert helicon.__version__.count(".") == 2  # major.minor.patch


class TestConfigAlias:
    def test_config_from_preset(self):
        config = helicon.Config.from_preset("sunbird")
        assert config is not None
        assert hasattr(config, "nozzle")
        assert hasattr(config, "plasma")

    def test_config_from_preset_dfd(self):
        config = helicon.Config.from_preset("dfd")
        assert len(config.nozzle.coils) >= 1

    def test_config_from_yaml(self, tmp_path):
        import yaml

        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(
            yaml.dump(
                {
                    "nozzle": {
                        "type": "solenoid",
                        "coils": [{"z": 0.0, "r": 0.1, "I": 10000}],
                        "domain": {"z_min": -0.3, "z_max": 1.0, "r_max": 0.5},
                    },
                    "plasma": {
                        "n0": 1e18,
                        "T_i_eV": 100,
                        "T_e_eV": 100,
                        "v_injection_ms": 50000,
                    },
                }
            )
        )
        config = helicon.Config.from_yaml(cfg_file)
        assert config.plasma.n0 == 1e18


class TestFieldsNamespace:
    def test_fields_has_compute(self):
        assert callable(helicon.fields.compute)

    def test_fields_compute_returns_bfield(self):
        config = helicon.Config.from_preset("sunbird")
        bfield = helicon.fields.compute(config.nozzle)
        assert hasattr(bfield, "Bz")
        assert hasattr(bfield, "Br")
        assert bfield.Bz.ndim == 2

    def test_fields_compute_grid_shape(self):
        config = helicon.Config.from_preset("sunbird")
        bfield = helicon.fields.compute(config.nozzle)
        nz = config.nozzle.resolution.nz
        nr = config.nozzle.resolution.nr
        assert bfield.Bz.shape == (nr, nz)

    def test_fields_has_load_comsol_bfield(self):
        """load_comsol_bfield must be in helicon.fields public API."""
        from helicon.fields import load_comsol_bfield

        assert callable(load_comsol_bfield)


class TestRunFunction:
    def test_run_dry_run_returns_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = helicon.Config.from_preset("sunbird")
            result = helicon.run(config, output_dir=tmp, dry_run=True)
            assert result.success
            assert result.output_dir is not None

    def test_run_dry_run_creates_input_file(self):

        with tempfile.TemporaryDirectory() as tmp:
            config = helicon.Config.from_preset("sunbird")
            result = helicon.run(config, output_dir=tmp, dry_run=True)
            assert result.input_file.exists()
            assert result.bfield_file.exists()


class TestMetricsDataclass:
    def test_metrics_dataclass_fields(self):
        """Metrics dataclass has the expected spec §6.2 fields."""
        from helicon import Metrics

        m = Metrics(
            thrust=1e-3,
            isp=2000.0,
            exhaust_velocity_ms=20000.0,
            mass_flow_rate_kgs=5e-8,
            plume_angle_deg=15.0,
            beam_efficiency=0.85,
            radial_loss_fraction=0.05,
            detachment=None,
            config_hash="abc123",
        )
        assert m.thrust == 1e-3
        assert m.isp == 2000.0
        assert m.config_hash == "abc123"

    def test_detachment_metrics_dataclass(self):
        from helicon import DetachmentMetrics, Metrics

        det = DetachmentMetrics(momentum=0.8, particle=0.75, energy=0.7)
        m = Metrics(
            thrust=None,
            isp=None,
            exhaust_velocity_ms=None,
            mass_flow_rate_kgs=None,
            plume_angle_deg=None,
            beam_efficiency=None,
            radial_loss_fraction=None,
            detachment=det,
            config_hash=None,
        )
        assert m.detachment.momentum == 0.8
        assert m.detachment.particle == 0.75


class TestScanFunction:
    def test_scan_dry_run(self):
        from pathlib import Path

        config = helicon.Config.from_preset("sunbird")
        with tempfile.TemporaryDirectory() as tmp:
            result = helicon.scan(
                config,
                vary={"coils.0.I": (5000, 15000, 2)},
                output_base=str(Path(tmp) / "scan"),
                dry_run=True,
            )
            assert len(result.points) == 2

    def test_scan_returns_scan_result(self):
        from pathlib import Path

        from helicon.optimize.scan import ScanResult

        config = helicon.Config.from_preset("sunbird")
        with tempfile.TemporaryDirectory() as tmp:
            result = helicon.scan(
                config,
                vary={"coils.0.I": (5000, 10000, 2)},
                output_base=str(Path(tmp) / "scan"),
                dry_run=True,
            )
            assert isinstance(result, ScanResult)
