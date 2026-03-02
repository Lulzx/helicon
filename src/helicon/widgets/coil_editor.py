"""CoilEditorWidget — interactive coil parameter editor.

Provides sliders for coil current and position with live B-field preview.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CoilSpec:
    """Mutable coil specification used by the editor."""

    z: float  # axial position [m]
    r: float  # radius [m]
    I: float  # current [A]


@dataclass
class CoilEditorWidget:
    """Widget for editing coil parameters interactively.

    This is the data model / controller layer.  The actual ipywidgets UI is
    created in :meth:`display` which imports ipywidgets lazily.

    Parameters
    ----------
    coils : list[CoilSpec]
        Initial coil specifications.
    on_change : callable, optional
        Callback invoked with the updated coil list whenever a parameter
        changes.
    """

    coils: list[CoilSpec] = field(default_factory=list)
    on_change: Callable[[list[CoilSpec]], None] | None = None

    def update_coil(self, index: int, **kwargs: Any) -> None:
        """Update coil parameters by index.

        Parameters
        ----------
        index : int
            Coil index to update.
        **kwargs
            Parameter names and new values (``z``, ``r``, ``I``).
        """
        coil = self.coils[index]
        for attr, val in kwargs.items():
            if not hasattr(coil, attr):
                raise AttributeError(f"CoilSpec has no attribute '{attr}'")
            setattr(coil, attr, float(val))
        if self.on_change is not None:
            self.on_change(self.coils)

    def display(self) -> None:
        """Render the ipywidgets UI.

        Raises
        ------
        ImportError
            If ipywidgets is not installed.
        """
        try:
            import ipywidgets as widgets  # noqa: F401
            from IPython.display import display as ipy_display
        except ImportError as exc:
            raise ImportError(
                "ipywidgets is required for CoilEditorWidget.display(). "
                "Install it with: pip install ipywidgets"
            ) from exc

        from helicon.widgets._ipywidgets_ui import build_coil_editor_ui

        ui = build_coil_editor_ui(self)
        ipy_display(ui)
