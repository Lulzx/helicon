"""Tests for helicon.runner.diagnostics — diagnostic scheduling."""

from __future__ import annotations

from helicon.config.parser import DiagnosticsConfig
from helicon.runner.diagnostics import DiagnosticSchedule, resolve_schedule


def _make_diag(mode: str = "analysis", interval: int = 500, particle: int = 1000):
    return DiagnosticsConfig(
        mode=mode,
        field_dump_interval=interval,
        particle_dump_interval=particle,
    )


class TestResolveSchedule:
    def test_returns_diagnostic_schedule(self):
        diag = _make_diag()
        sched = resolve_schedule(diag, timesteps=5000, nz=128, nr=64, n_species=2)
        assert isinstance(sched, DiagnosticSchedule)

    def test_analysis_mode_has_particles(self):
        diag = _make_diag(mode="analysis")
        sched = resolve_schedule(diag, timesteps=5000, nz=128, nr=64, n_species=2)
        assert sched.particle_intervals is not None

    def test_scan_mode_no_particles(self):
        diag = _make_diag(mode="scan")
        sched = resolve_schedule(diag, timesteps=5000, nz=128, nr=64, n_species=2)
        assert sched.particle_intervals is None

    def test_scan_mode_minimal_fields(self):
        diag = _make_diag(mode="scan")
        sched = resolve_schedule(diag, timesteps=5000, nz=128, nr=64, n_species=1)
        assert "Bz" in sched.fields_to_plot
        assert "rho" in sched.fields_to_plot
        # scan mode should have fewer fields than analysis
        assert len(sched.fields_to_plot) < 7

    def test_analysis_mode_full_fields(self):
        diag = _make_diag(mode="analysis")
        sched = resolve_schedule(diag, timesteps=5000, nz=128, nr=64, n_species=2)
        assert len(sched.fields_to_plot) >= 5

    def test_species_to_dump_count(self):
        diag = _make_diag()
        sched = resolve_schedule(diag, timesteps=5000, nz=128, nr=64, n_species=3)
        assert len(sched.species_to_dump) == 3

    def test_estimated_size_positive(self):
        diag = _make_diag()
        sched = resolve_schedule(diag, timesteps=5000, nz=128, nr=64, n_species=2)
        assert sched.estimated_size_gb >= 0.0

    def test_scan_smaller_than_analysis(self):
        """Scan mode should produce a smaller output estimate than analysis mode."""
        diag_scan = _make_diag(mode="scan")
        diag_full = _make_diag(mode="analysis")
        sched_scan = resolve_schedule(diag_scan, timesteps=5000, nz=128, nr=64, n_species=2)
        sched_full = resolve_schedule(diag_full, timesteps=5000, nz=128, nr=64, n_species=2)
        assert sched_scan.estimated_size_gb < sched_full.estimated_size_gb

    def test_field_interval_matches_config(self):
        diag = _make_diag(interval=250)
        sched = resolve_schedule(diag, timesteps=5000, nz=64, nr=32, n_species=1)
        assert sched.field_intervals == 250
