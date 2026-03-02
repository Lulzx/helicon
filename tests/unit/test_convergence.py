"""Tests for helicon.runner.convergence."""

import math

import numpy as np
import pytest

from helicon.runner.convergence import (
    ConvergenceLevel,
    ConvergenceResult,
    richardson_extrapolate,
    run_convergence_study,
)


class TestRichardsonExtrapolate:
    def test_second_order_convergence(self):
        # f(h) = 1 + h², refine by factor 2: h, h/2, h/4
        h = 1.0
        f_exact = 1.0
        v1 = f_exact + h**2
        v2 = f_exact + (h / 2) ** 2
        v3 = f_exact + (h / 4) ** 2
        order, extrap = richardson_extrapolate([v1, v2, v3], [2.0, 2.0])
        assert abs(order - 2.0) < 0.01
        assert abs(extrap - f_exact) < 1e-10

    def test_first_order_convergence(self):
        h = 1.0
        f_exact = 5.0
        v1 = f_exact + h
        v2 = f_exact + h / 2
        v3 = f_exact + h / 4
        order, extrap = richardson_extrapolate([v1, v2, v3], [2.0, 2.0])
        assert abs(order - 1.0) < 0.01
        assert abs(extrap - f_exact) < 1e-10

    def test_already_converged_returns_fine(self):
        # Near-zero differences → returns Q3
        values = [1.0, 1.0 + 1e-25, 1.0 + 2e-25]
        order, extrap = richardson_extrapolate(values, [2.0, 2.0])
        # Either nan order or Q3 returned; check extrap is near Q3
        assert abs(extrap - values[2]) < 1e-15

    def test_non_monotone_returns_nan_order(self):
        # Non-monotone convergence → ratio could be negative
        values = [1.0, 2.0, 1.5]  # oscillating
        order, extrap = richardson_extrapolate(values, [2.0, 2.0])
        assert math.isnan(order)

    def test_non_uniform_refinement(self):
        # Refinement ratio 3: h, h/3, h/9 with 2nd order
        h = 1.0
        f = 0.0
        v1 = f + h**2
        v2 = f + (h / 3) ** 2
        v3 = f + (h / 9) ** 2
        order, extrap = richardson_extrapolate([v1, v2, v3], [3.0, 3.0])
        assert abs(order - 2.0) < 0.05
        assert abs(extrap - f) < 1e-8

    def test_returns_tuple(self):
        result = richardson_extrapolate([3.0, 2.0, 1.5], [2.0, 2.0])
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestRunConvergenceStudy:
    """Integration tests using dry_run=True to avoid launching WarpX."""

    def _get_base_config(self):
        from helicon.config.parser import (
            CoilConfig,
            DomainConfig,
            NozzleConfig,
            PlasmaSourceConfig,
            ResolutionConfig,
            SimConfig,
        )
        return SimConfig(
            nozzle=NozzleConfig(
                type="solenoid",
                coils=[CoilConfig(z=0.0, r=0.1, I=10000.0)],
                domain=DomainConfig(z_min=-0.5, z_max=2.0, r_max=0.5),
                resolution=ResolutionConfig(nz=64, nr=32),
            ),
            plasma=PlasmaSourceConfig(
                species=["H+", "e-"],
                n0=1e18,
                T_i_eV=10.0,
                T_e_eV=5.0,
                v_injection_ms=50000.0,
            ),
            timesteps=100,
            output_dir="results/test",
        )

    def test_dry_run_returns_convergence_result(self, tmp_path):
        config = self._get_base_config()
        result = run_convergence_study(
            config,
            resolutions=[(64, 32), (128, 64), (256, 128)],
            output_base=tmp_path,
            dry_run=True,
        )
        assert isinstance(result, ConvergenceResult)

    def test_dry_run_creates_levels(self, tmp_path):
        config = self._get_base_config()
        result = run_convergence_study(
            config,
            resolutions=[(64, 32), (128, 64)],
            output_base=tmp_path,
            dry_run=True,
        )
        assert len(result.levels) == 2

    def test_level_h_values_ordered(self, tmp_path):
        config = self._get_base_config()
        result = run_convergence_study(
            config,
            resolutions=[(64, 32), (128, 64), (256, 128)],
            output_base=tmp_path,
            dry_run=True,
        )
        h_vals = [lv.h for lv in result.levels]
        # h = 1/sqrt(nz*nr): coarser grid → larger h
        assert h_vals[0] > h_vals[1] > h_vals[2]

    def test_level_output_dirs_set(self, tmp_path):
        config = self._get_base_config()
        result = run_convergence_study(
            config,
            resolutions=[(64, 32), (128, 64)],
            output_base=tmp_path,
            dry_run=True,
        )
        for lv in result.levels:
            assert lv.output_dir is not None

    def test_dry_run_no_thrust_data(self, tmp_path):
        config = self._get_base_config()
        result = run_convergence_study(
            config,
            resolutions=[(64, 32), (128, 64), (256, 128)],
            output_base=tmp_path,
            dry_run=True,
        )
        # Dry run produces no simulation output → no thrust
        for lv in result.levels:
            assert lv.thrust_N is None

    def test_dry_run_not_converged(self, tmp_path):
        config = self._get_base_config()
        result = run_convergence_study(
            config,
            resolutions=[(64, 32), (128, 64)],
            output_base=tmp_path,
            dry_run=True,
        )
        assert not result.converged

    def test_convergence_order_none_without_thrust(self, tmp_path):
        config = self._get_base_config()
        result = run_convergence_study(
            config,
            resolutions=[(64, 32), (128, 64), (256, 128)],
            output_base=tmp_path,
            dry_run=True,
        )
        assert result.convergence_order is None
        assert result.extrapolated_thrust_N is None

    def test_resolution_override(self, tmp_path):
        config = self._get_base_config()
        resolutions = [(32, 16), (64, 32)]
        result = run_convergence_study(
            config,
            resolutions=resolutions,
            output_base=tmp_path,
            dry_run=True,
        )
        assert result.levels[0].nz == 32
        assert result.levels[0].nr == 16
        assert result.levels[1].nz == 64
        assert result.levels[1].nr == 32

    def test_convergence_with_synthetic_thrust(self):
        """Unit test Richardson path using synthetic ConvergenceLevels."""
        # Build synthetic result with known 2nd-order convergence
        f_exact = 10.0
        levels = []
        for nz, nr in [(64, 32), (128, 64), (256, 128)]:
            h = 1.0 / math.sqrt(nz * nr)
            thrust = f_exact + h**2 * 1000.0  # 2nd-order error
            levels.append(ConvergenceLevel(nz=nz, nr=nr, h=h, output_dir=None, success=True, thrust_N=thrust))

        thrust_values = [lv.thrust_N for lv in levels]
        h_vals = [lv.h for lv in levels]
        h_ratios = [h_vals[i] / h_vals[i + 1] for i in range(len(h_vals) - 1)]
        order, extrap = richardson_extrapolate(thrust_values[-3:], h_ratios[-2:])
        assert abs(order - 2.0) < 0.1
        assert abs(extrap - f_exact) < 0.01
