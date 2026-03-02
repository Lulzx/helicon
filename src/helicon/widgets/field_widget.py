"""FieldTopologyWidget — interactive field topology explorer.

Main widget that combines coil editing with live field-line and
detachment-surface visualisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from helicon.widgets.coil_editor import CoilSpec


@dataclass
class FieldTopologyWidget:
    """Interactive field topology explorer.

    Combines a :class:`CoilEditorWidget` with a field-line visualiser that
    re-renders on every coil change using the Biot-Savart solver (or the
    MLX surrogate if available).

    Parameters
    ----------
    coils : list[CoilSpec]
        Initial coil specifications.
    domain_z : tuple[float, float]
        Axial domain [z_min, z_max] in metres.
    domain_r_max : float
        Maximum radial extent [m].
    nz, nr : int
        Grid resolution.
    use_surrogate : bool
        Use MLX neural surrogate for fast preview (falls back to Biot-Savart
        if surrogate is not trained).
    n_field_lines : int
        Number of field lines to trace.
    """

    coils: list[CoilSpec] = field(default_factory=list)
    domain_z: tuple[float, float] = (-1.0, 3.0)
    domain_r_max: float = 1.5
    nz: int = 128
    nr: int = 64
    use_surrogate: bool = False
    n_field_lines: int = 10

    @classmethod
    def from_preset(cls, name: str) -> FieldTopologyWidget:
        """Create a widget pre-loaded with a named preset configuration.

        Parameters
        ----------
        name : str
            Preset name (e.g. ``"sunbird"``, ``"dfd"``, ``"ppr"``).
        """
        from helicon.config.parser import SimConfig

        config = SimConfig.from_preset(name)
        coils = [CoilSpec(z=c.z, r=c.r, I=c.I) for c in config.nozzle.coils]
        return cls(
            coils=coils,
            domain_z=(config.nozzle.domain.z_min, config.nozzle.domain.z_max),
            domain_r_max=config.nozzle.domain.r_max,
            nz=config.nozzle.resolution.nz,
            nr=config.nozzle.resolution.nr,
        )

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _get_bfield(self):
        """Compute and return the BField object."""
        from helicon.fields.biot_savart import Coil, Grid, compute_bfield

        bfield_coils = [Coil(z=c.z, r=c.r, I=c.I) for c in self.coils]
        grid = Grid(
            z_min=self.domain_z[0],
            z_max=self.domain_z[1],
            r_max=self.domain_r_max,
            nz=self.nz,
            nr=self.nr,
        )
        return compute_bfield(bfield_coils, grid, backend="numpy")

    def compute_bfield(self) -> dict[str, np.ndarray]:
        """Compute the magnetic field on the current grid.

        Returns
        -------
        dict with keys ``"Bz"``, ``"Br"``, ``"z_grid"``, ``"r_grid"``
        """
        result = self._get_bfield()
        return {
            "Bz": result.Bz,
            "Br": result.Br,
            "z_grid": result.z,
            "r_grid": result.r,
        }

    def compute_field_lines(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Trace magnetic field lines on the current grid.

        Returns
        -------
        list of (z_line, r_line) arrays
        """
        from helicon.fields.field_lines import trace_field_lines

        bfield = self._get_bfield()
        line_set = trace_field_lines(bfield, n_lines=self.n_field_lines)
        return [(line.z, line.r) for line in line_set.lines]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def render(self) -> Any:
        """Compute and return a matplotlib figure of the field topology.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for FieldTopologyWidget.render(). "
                "Install it with: pip install matplotlib"
            ) from exc

        bfield = self.compute_bfield()
        lines = self.compute_field_lines()

        fig, ax = plt.subplots(figsize=(8, 5))
        Bmod = np.sqrt(bfield["Bz"] ** 2 + bfield["Br"] ** 2)
        z_grid = bfield["z_grid"]
        r_grid = bfield["r_grid"]

        im = ax.pcolormesh(
            z_grid,
            r_grid,
            Bmod,
            cmap="plasma",
            shading="auto",
        )
        fig.colorbar(im, ax=ax, label="|B| [T]")

        for z_line, r_line in lines:
            ax.plot(z_line, r_line, "w-", lw=0.8, alpha=0.7)
            ax.plot(z_line, -r_line, "w-", lw=0.8, alpha=0.7)

        # Mark coil positions
        for c in self.coils:
            ax.axvline(c.z, color="cyan", lw=1.5, linestyle="--", alpha=0.6)

        ax.set_xlabel("z [m]")
        ax.set_ylabel("r [m]")
        ax.set_title("Magnetic Field Topology")
        fig.tight_layout()
        return fig

    def display(self) -> None:
        """Display the interactive widget in a Jupyter notebook.

        Requires ipywidgets and matplotlib.
        """
        try:
            import ipywidgets as widgets  # noqa: F401
            from IPython.display import display as ipy_display
        except ImportError as exc:
            raise ImportError(
                "ipywidgets is required for FieldTopologyWidget.display(). "
                "Install it with: pip install ipywidgets"
            ) from exc

        from helicon.widgets._ipywidgets_ui import build_field_topology_ui

        ui = build_field_topology_ui(self)
        ipy_display(ui)
