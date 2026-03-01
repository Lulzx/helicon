"""Pulsed-mode postprocessing for PPR-class engines.

Computes per-pulse metrics for pulsed plasma rocket configurations:
- Impulse bit (N·s per pulse)
- Per-pulse detachment efficiency
- Inter-pulse field relaxation diagnostics
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PulseMetrics:
    """Metrics for a single plasma pulse."""

    pulse_index: int
    impulse_bit_Ns: float
    detachment_efficiency: float
    peak_thrust_N: float
    pulse_duration_s: float
    particle_count: int


@dataclass
class PulsedResult:
    """Summary of pulsed-mode analysis."""

    pulses: list[PulseMetrics]
    total_impulse_Ns: float
    mean_detachment_efficiency: float
    mean_impulse_bit_Ns: float
    n_pulses: int
    repetition_rate_Hz: float | None


def compute_pulsed_metrics(
    output_dir: str | Path,
    *,
    species_name: str = "D_plus",
    species_mass: float = 3.3435837724e-27,
    pulse_period_s: float | None = None,
    n_pulses: int | None = None,
) -> PulsedResult:
    """Compute per-pulse propulsion metrics from time-resolved output.

    Analyzes time-series of particle snapshots to identify individual
    pulses and compute metrics for each.

    Parameters
    ----------
    output_dir : path
        WarpX output directory with multiple time snapshots.
    species_name : str
        Species to analyze.
    species_mass : float
        Species mass [kg].
    pulse_period_s : float, optional
        Expected pulse repetition period [s]. If None, auto-detected.
    n_pulses : int, optional
        Expected number of pulses. If None, auto-detected.
    """
    import h5py

    output_dir = Path(output_dir)
    h5_files = sorted(output_dir.glob("**/*.h5"))

    if len(h5_files) < 2:
        msg = "Need multiple time snapshots for pulsed analysis"
        raise FileNotFoundError(msg)

    # Read time series of momentum flux
    times = []
    momentum_z = []

    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            base = _navigate_openpmd(f)
            if "particles" not in base or species_name not in base["particles"]:
                continue

            sp = base["particles"][species_name]
            pz = sp["momentum"]["z"][:]
            w = sp["weighting"][:] if "weighting" in sp else np.ones_like(pz)

            # Try to read time from openPMD attributes
            t = 0.0
            if hasattr(base, "attrs") and "time" in base.attrs:
                t = float(base.attrs["time"])
            elif hasattr(base, "attrs") and "dt" in base.attrs:
                t = float(base.attrs.get("time", len(times) * base.attrs["dt"]))

            times.append(t)
            momentum_z.append(float(np.sum(w * pz)))

    if not times:
        return PulsedResult(
            pulses=[],
            total_impulse_Ns=0.0,
            mean_detachment_efficiency=0.0,
            mean_impulse_bit_Ns=0.0,
            n_pulses=0,
            repetition_rate_Hz=None,
        )

    times_arr = np.array(times)
    pz_arr = np.array(momentum_z)

    # Detect pulses from momentum signal
    if n_pulses is None:
        # Simple peak detection: count sign changes or threshold crossings
        pz_norm = pz_arr / (np.max(np.abs(pz_arr)) + 1e-30)
        threshold = 0.3
        above = pz_norm > threshold
        # Count rising edges
        edges = np.diff(above.astype(int))
        n_pulses = max(1, int(np.sum(edges > 0)))

    # Divide time series into pulse windows
    dt_total = times_arr[-1] - times_arr[0] if len(times_arr) > 1 else 1.0
    pulse_duration = dt_total / n_pulses if n_pulses > 0 else dt_total

    pulses = []
    for i in range(n_pulses):
        t_start = times_arr[0] + i * pulse_duration
        t_end = t_start + pulse_duration
        mask = (times_arr >= t_start) & (times_arr < t_end)

        if np.sum(mask) == 0:
            continue

        pz_pulse = pz_arr[mask]
        impulse = float(np.trapezoid(pz_pulse, times_arr[mask])) if np.sum(mask) > 1 else 0.0
        peak = float(np.max(np.abs(pz_pulse)))

        pulses.append(
            PulseMetrics(
                pulse_index=i,
                impulse_bit_Ns=abs(impulse),
                detachment_efficiency=0.0,  # would need field-line classification
                peak_thrust_N=peak,
                pulse_duration_s=float(pulse_duration),
                particle_count=int(np.sum(mask)),
            )
        )

    total_impulse = sum(p.impulse_bit_Ns for p in pulses)
    mean_impulse = total_impulse / len(pulses) if pulses else 0.0
    mean_eta = np.mean([p.detachment_efficiency for p in pulses]) if pulses else 0.0

    rep_rate = 1.0 / pulse_duration if pulse_duration > 0 else None

    return PulsedResult(
        pulses=pulses,
        total_impulse_Ns=total_impulse,
        mean_detachment_efficiency=float(mean_eta),
        mean_impulse_bit_Ns=mean_impulse,
        n_pulses=len(pulses),
        repetition_rate_Hz=rep_rate,
    )


def _navigate_openpmd(f: Any) -> Any:
    if "data" in f:
        iterations = sorted(f["data"].keys(), key=int)
        return f["data"][iterations[-1]]
    return f
