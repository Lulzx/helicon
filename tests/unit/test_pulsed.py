"""Tests for magnozzlex.postprocess.pulsed module."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from magnozzlex.postprocess.pulsed import PulsedResult, compute_pulsed_metrics


def _write_pulsed_snapshot(
    path: Path,
    *,
    species_name: str = "D_plus",
    time: float = 0.0,
    n_particles: int = 500,
    pz_mean: float = 1e-20,
    seed: int = 42,
) -> None:
    """Write a single time-snapshot for pulsed mode analysis."""
    rng = np.random.default_rng(seed)
    pz = rng.normal(pz_mean, abs(pz_mean) * 0.1, size=n_particles)
    w = np.ones(n_particles)

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        base = f.create_group("data/0")
        base.attrs["time"] = time
        sp = base.create_group(f"particles/{species_name}")
        mom = sp.create_group("momentum")
        mom.create_dataset("z", data=pz)
        sp.create_dataset("weighting", data=w)


@pytest.fixture
def pulsed_output(tmp_path: Path) -> Path:
    """Create multiple time snapshots simulating a pulsed signal."""
    out = tmp_path / "pulsed"
    out.mkdir()

    # Create a pulse-like time series: momentum rises and falls
    n_snapshots = 20
    times = np.linspace(0.0, 1e-6, n_snapshots)

    for i, t in enumerate(times):
        # Two pulses in the time window
        pulse_signal = np.sin(2 * np.pi * 2e6 * t)  # 2 MHz
        pz_mean = max(1e-21, 1e-20 * max(0, pulse_signal))

        _write_pulsed_snapshot(
            out / f"snap_{i:04d}.h5",
            time=float(t),
            pz_mean=pz_mean,
            seed=100 + i,
        )

    return out


class TestPulsedMetrics:
    """Tests for compute_pulsed_metrics."""

    def test_returns_result(self, pulsed_output: Path) -> None:
        result = compute_pulsed_metrics(pulsed_output)
        assert isinstance(result, PulsedResult)

    def test_detects_pulses(self, pulsed_output: Path) -> None:
        result = compute_pulsed_metrics(pulsed_output)
        assert result.n_pulses >= 1

    def test_total_impulse_nonnegative(self, pulsed_output: Path) -> None:
        result = compute_pulsed_metrics(pulsed_output)
        assert result.total_impulse_Ns >= 0

    def test_mean_impulse_bit(self, pulsed_output: Path) -> None:
        result = compute_pulsed_metrics(pulsed_output)
        if result.n_pulses > 0:
            assert result.mean_impulse_bit_Ns >= 0
            # Mean should equal total / n_pulses
            expected = result.total_impulse_Ns / result.n_pulses
            assert abs(result.mean_impulse_bit_Ns - expected) < 1e-30

    def test_pulse_metrics_structure(self, pulsed_output: Path) -> None:
        result = compute_pulsed_metrics(pulsed_output)
        for pulse in result.pulses:
            assert pulse.pulse_index >= 0
            assert pulse.impulse_bit_Ns >= 0
            assert pulse.pulse_duration_s > 0
            assert pulse.particle_count >= 0

    def test_repetition_rate(self, pulsed_output: Path) -> None:
        result = compute_pulsed_metrics(pulsed_output)
        if result.repetition_rate_Hz is not None:
            assert result.repetition_rate_Hz > 0

    def test_specified_n_pulses(self, pulsed_output: Path) -> None:
        result = compute_pulsed_metrics(pulsed_output, n_pulses=3)
        assert result.n_pulses == 3

    def test_insufficient_snapshots(self, tmp_path: Path) -> None:
        """Should raise with < 2 snapshots."""
        out = tmp_path / "single"
        out.mkdir()
        _write_pulsed_snapshot(out / "snap.h5")
        with pytest.raises(FileNotFoundError, match="multiple time snapshots"):
            compute_pulsed_metrics(out)

    def test_empty_result_when_no_species(self, tmp_path: Path) -> None:
        """Should return empty result if species not found."""
        out = tmp_path / "empty_species"
        out.mkdir()
        # Write snapshots with a different species
        for i in range(3):
            _write_pulsed_snapshot(
                out / f"snap_{i:04d}.h5",
                species_name="He_plus",
                time=float(i) * 1e-7,
                seed=200 + i,
            )
        result = compute_pulsed_metrics(out, species_name="D_plus")
        assert result.n_pulses == 0
        assert result.total_impulse_Ns == 0.0
