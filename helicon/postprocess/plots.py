"""Auto-generated postprocessing plots (spec §6.3).

Generates publication-quality figures saved alongside simulation output:
- B-field topology (field lines + |B| colour map)
- Thrust convergence time-series
- Detachment map (η_d vs downstream position)

All functions gracefully skip when matplotlib or required data is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def generate_all_plots(
    output_dir: str | Path,
    *,
    bfield_file: str | Path | None = None,
    fmt: str = "png",
    dpi: int = 150,
) -> list[Path]:
    """Generate all standard auto-plots for a completed simulation run.

    Called automatically by the postprocessing pipeline after each run.
    Failures are non-fatal — each plot is attempted independently.

    Parameters
    ----------
    output_dir:
        Simulation output directory.
    bfield_file:
        Path to the pre-computed B-field HDF5 file.  Inferred from
        ``output_dir / "applied_bfield.h5"`` when not supplied.
    fmt:
        Figure file format (``"png"``, ``"pdf"``, ``"svg"``).
    dpi:
        Figure resolution in dots-per-inch.

    Returns
    -------
    list of Path
        Paths of figures that were successfully saved.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    # 1. B-field topology
    bf_path = Path(bfield_file) if bfield_file else output_dir / "applied_bfield.h5"
    p = _plot_bfield_topology(bf_path, plots_dir, fmt=fmt, dpi=dpi)
    if p:
        saved.append(p)

    # 2. Thrust convergence (requires time-series HDF5 snapshots)
    p = _plot_thrust_convergence(output_dir, plots_dir, fmt=fmt, dpi=dpi)
    if p:
        saved.append(p)

    # 3. Detachment map (requires particle/field snapshots)
    p = _plot_detachment_map(output_dir, plots_dir, fmt=fmt, dpi=dpi)
    if p:
        saved.append(p)

    return saved


# ---------------------------------------------------------------------------
# Individual plot generators
# ---------------------------------------------------------------------------


def _plot_bfield_topology(
    bfield_file: Path,
    plots_dir: Path,
    *,
    fmt: str = "png",
    dpi: int = 150,
) -> Path | None:
    """Plot B-field topology: |B| colour map + field lines.

    Saves to ``plots_dir/bfield_topology.<fmt>``.
    Returns the saved path, or None on failure.
    """
    try:
        import matplotlib.pyplot as plt
        from helicon.fields.biot_savart import BField

        if not bfield_file.exists():
            return None

        bf = BField.load(str(bfield_file))
        fig, ax = bf.plot(component="Bmag", field_lines=True, figsize=(10, 5))
        ax.set_title("Applied magnetic field topology")
        ax.set_xlabel("z (m)")
        ax.set_ylabel("r (m)")
        out = plots_dir / f"bfield_topology.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out
    except Exception:
        return None


def _plot_thrust_convergence(
    output_dir: Path,
    plots_dir: Path,
    *,
    fmt: str = "png",
    dpi: int = 150,
) -> Path | None:
    """Plot thrust (proxy: exit momentum) vs timestep.

    Saves to ``plots_dir/thrust_convergence.<fmt>``.
    Returns the saved path, or None on failure / insufficient data.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import h5py

        h5_files = sorted(output_dir.glob("**/*.h5"))
        # Exclude the bfield file
        h5_files = [f for f in h5_files if "bfield" not in f.name]
        if len(h5_files) < 2:
            return None

        steps: list[int] = []
        pz_vals: list[float] = []

        for i, h5_path in enumerate(h5_files):
            try:
                with h5py.File(h5_path, "r") as f:
                    base = (
                        f["data"][sorted(f["data"].keys(), key=int)[-1]]
                        if "data" in f
                        else f
                    )
                    if "particles" not in base:
                        continue
                    total_pz = 0.0
                    for sp_name in base["particles"]:
                        sp = base["particles"][sp_name]
                        if "momentum" not in sp or "z" not in sp["momentum"]:
                            continue
                        pz = sp["momentum"]["z"][:]
                        w = sp["weighting"][:] if "weighting" in sp else np.ones_like(pz)
                        total_pz += float(np.sum(w * pz))
                steps.append(i)
                pz_vals.append(total_pz)
            except Exception:
                continue

        if len(pz_vals) < 2:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, pz_vals, linewidth=1.5, color="steelblue")
        ax.set_xlabel("Snapshot index")
        ax.set_ylabel("Exit momentum proxy (kg m/s)")
        ax.set_title("Thrust convergence monitor")
        ax.grid(True, alpha=0.3)

        # Mark last-10% region
        n_last = max(1, len(steps) // 10)
        ax.axvspan(steps[-n_last], steps[-1], alpha=0.15, color="orange",
                   label="Last 10%")
        ax.legend(fontsize=9)

        out = plots_dir / f"thrust_convergence.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out
    except Exception:
        return None


def _plot_detachment_map(
    output_dir: Path,
    plots_dir: Path,
    *,
    fmt: str = "png",
    dpi: int = 150,
) -> Path | None:
    """Plot detachment efficiency as a function of downstream position.

    Uses the last available HDF5 snapshot.
    Saves to ``plots_dir/detachment_map.<fmt>``.
    Returns the saved path, or None on failure / insufficient data.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import h5py

        h5_files = sorted(output_dir.glob("**/*.h5"))
        h5_files = [f for f in h5_files if "bfield" not in f.name]
        if not h5_files:
            return None

        # Use last snapshot
        h5_path = h5_files[-1]
        z_bins_list: list[float] = []
        detach_frac_list: list[float] = []

        with h5py.File(h5_path, "r") as f:
            base = (
                f["data"][sorted(f["data"].keys(), key=int)[-1]]
                if "data" in f
                else f
            )
            if "particles" not in base:
                return None

            for sp_name in base["particles"]:
                sp = base["particles"][sp_name]
                if "position" not in sp or "z" not in sp["position"]:
                    continue
                z_arr = sp["position"]["z"][:]
                if "momentum" not in sp or "z" not in sp["momentum"]:
                    continue
                pz_arr = sp["momentum"]["z"][:]
                w = sp["weighting"][:] if "weighting" in sp else np.ones_like(z_arr)

                # Bin particles by z-position; fraction with pz > 0 = detached proxy
                z_min, z_max = z_arr.min(), z_arr.max()
                if z_max <= z_min:
                    continue
                n_bins = 20
                edges = np.linspace(z_min, z_max, n_bins + 1)
                centres = 0.5 * (edges[:-1] + edges[1:])

                for j in range(n_bins):
                    mask = (z_arr >= edges[j]) & (z_arr < edges[j + 1])
                    if mask.sum() < 10:
                        continue
                    frac = float(np.sum(w[mask] * (pz_arr[mask] > 0)) /
                                 np.sum(w[mask]))
                    z_bins_list.append(float(centres[j]))
                    detach_frac_list.append(frac)
                break  # use first ion species only

        if not z_bins_list:
            return None

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(z_bins_list, detach_frac_list, "o-", linewidth=1.5,
                color="darkorange", markersize=4)
        ax.set_xlabel("z (m)")
        ax.set_ylabel("Downstream momentum fraction (detachment proxy)")
        ax.set_title("Detachment map — last snapshot")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        out = plots_dir / f"detachment_map.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return out
    except Exception:
        return None
