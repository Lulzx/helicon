"""Tests for helicon.runner.sim3d — 3D Boris-pusher simulation."""

from __future__ import annotations

import numpy as np

import helicon
from helicon.runner.sim3d import Sim3DConfig, Sim3DResult, _make_grid_params, run_3d_simulation


class TestSim3DConfig:
    def test_defaults(self):
        cfg = Sim3DConfig()
        assert cfg.n_particles == 2000
        assert cfg.n_steps == 5000
        assert cfg.backend == "auto"
        assert cfg.seed == 0

    def test_custom(self):
        cfg = Sim3DConfig(n_particles=50, n_steps=10, backend="numpy", seed=7)
        assert cfg.n_particles == 50
        assert cfg.n_steps == 10
        assert cfg.backend == "numpy"
        assert cfg.seed == 7


class TestSim3DResult:
    def _make_result(self):
        from helicon.fields.biot_savart_3d import Coil3D, Grid3D, compute_bfield_3d

        coil = Coil3D(z=0.0, r=0.1, I=10000.0)
        grid = Grid3D(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1,
                      z_min=-0.3, z_max=1.0, nx=4, ny=4, nz=8)
        bfield = compute_bfield_3d([coil], grid, backend="numpy")
        return Sim3DResult(
            thrust_N=1e-3,
            eta_d=0.8,
            mean_exit_angle_deg=12.0,
            mean_exit_speed_ms=50000.0,
            mirror_ratio=5.0,
            n_injected=100,
            n_exited=80,
            wall_time_s=0.5,
            bfield=bfield,
        )

    def test_fields(self):
        r = self._make_result()
        assert r.thrust_N == 1e-3
        assert r.eta_d == 0.8
        assert r.n_injected == 100
        assert r.n_exited == 80

    def test_summary_contains_key_info(self):
        r = self._make_result()
        s = r.summary()
        assert "Thrust" in s
        assert "Detachment" in s
        assert "80/100" in s

    def test_to_dict_keys(self):
        r = self._make_result()
        d = r.to_dict()
        assert "thrust_N" in d
        assert "eta_d" in d
        assert "n_injected" in d
        assert "n_exited" in d
        assert "mirror_ratio" in d


class TestMakeGridParams:
    def test_returns_tuple(self):
        from helicon.fields.biot_savart_3d import Coil3D, Grid3D, compute_bfield_3d

        coil = Coil3D(z=0.0, r=0.1, I=10000.0)
        grid = Grid3D(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1,
                      z_min=-0.3, z_max=1.0, nx=4, ny=4, nz=8)
        bfield = compute_bfield_3d([coil], grid, backend="numpy")
        gp = _make_grid_params(bfield)
        # tuple: x0, dx, y0, dy, z0, dz, nx, ny, nz, nynz, Bxf, Byf, Bzf
        assert len(gp) == 13
        _x0, _dx, _y0, _dy, _z0, _dz, nx, ny, nz, nynz, Bxf, _Byf, _Bzf = gp
        assert nx == 4
        assert nz == 8
        assert nynz == ny * nz
        assert Bxf.dtype == np.float32
        assert len(Bxf) == nx * ny * nz


class TestRun3DSimulation:
    def test_runs_and_returns_result(self):
        config = helicon.Config.from_preset("sunbird")
        cfg = Sim3DConfig(n_particles=20, n_steps=5, backend="numpy", seed=0)
        result = run_3d_simulation(config, cfg)
        assert isinstance(result, Sim3DResult)

    def test_result_fields_reasonable(self):
        config = helicon.Config.from_preset("sunbird")
        cfg = Sim3DConfig(n_particles=50, n_steps=10, backend="numpy", seed=42)
        result = run_3d_simulation(config, cfg)
        assert 0.0 <= result.eta_d <= 1.0
        assert result.n_exited <= result.n_injected
        assert result.n_injected == 50
        assert result.wall_time_s >= 0.0

    def test_mirror_ratio_positive(self):
        config = helicon.Config.from_preset("sunbird")
        cfg = Sim3DConfig(n_particles=20, n_steps=5, backend="numpy", seed=1)
        result = run_3d_simulation(config, cfg)
        assert result.mirror_ratio >= 1.0
