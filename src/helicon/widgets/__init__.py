"""Jupyter widgets for interactive Helicon field topology exploration.

Provides an ipywidgets-based interface to drag coils and see field lines
and the detachment surface update in real time via the MLX surrogate.

Usage (in a Jupyter notebook)::

    from helicon.widgets import FieldTopologyWidget

    widget = FieldTopologyWidget.from_preset("sunbird")
    widget.display()

Requires ``ipywidgets`` and ``matplotlib``.  Both are soft dependencies;
if not installed the widget raises an informative ImportError.
"""

from helicon.widgets.coil_editor import CoilEditorWidget
from helicon.widgets.field_widget import FieldTopologyWidget

__all__ = ["CoilEditorWidget", "FieldTopologyWidget"]
