"""ipywidgets UI builders.

Separate module so that ipywidgets is imported only when display() is
called, keeping the rest of the widgets package importable without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helicon.widgets.coil_editor import CoilEditorWidget
    from helicon.widgets.field_widget import FieldTopologyWidget


def _coil_editor_handlers(editor, idx):
    """Return (on_z, on_r, on_I) handlers bound to *idx* for CoilEditorWidget."""

    def on_z(change):
        editor.update_coil(idx, z=change["new"])

    def on_r(change):
        editor.update_coil(idx, r=change["new"])

    def on_I(change):
        editor.update_coil(idx, I=change["new"])

    return on_z, on_r, on_I


def _field_topology_handlers(widget_ref, idx, refresh_fn):
    """Return (on_z, on_r, on_I) handlers bound to *idx* for FieldTopologyWidget."""

    def on_z(change):
        widget_ref.coils[idx].z = change["new"]
        refresh_fn()

    def on_r(change):
        widget_ref.coils[idx].r = change["new"]
        refresh_fn()

    def on_I(change):
        widget_ref.coils[idx].I = change["new"]
        refresh_fn()

    return on_z, on_r, on_I


def build_coil_editor_ui(editor: CoilEditorWidget):
    """Build an ipywidgets UI for the CoilEditorWidget."""
    import ipywidgets as widgets

    children = []
    for i, coil in enumerate(editor.coils):
        on_z, on_r, on_I = _coil_editor_handlers(editor, i)

        z_slider = widgets.FloatSlider(
            value=coil.z,
            min=-5.0,
            max=5.0,
            step=0.05,
            description=f"Coil {i} z [m]",
        )
        r_slider = widgets.FloatSlider(
            value=coil.r,
            min=0.01,
            max=2.0,
            step=0.01,
            description=f"Coil {i} r [m]",
        )
        I_slider = widgets.FloatSlider(
            value=coil.I,
            min=-1e6,
            max=1e6,
            step=1000.0,
            description=f"Coil {i} I [A]",
        )
        z_slider.observe(on_z, names="value")
        r_slider.observe(on_r, names="value")
        I_slider.observe(on_I, names="value")

        children.extend(
            [
                widgets.HTML(f"<b>Coil {i}</b>"),
                z_slider,
                r_slider,
                I_slider,
                widgets.HTML("<hr/>"),
            ]
        )

    return widgets.VBox(children)


def build_field_topology_ui(widget: FieldTopologyWidget):
    """Build an ipywidgets UI for the FieldTopologyWidget."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display as ipy_display

    output = widgets.Output()

    def refresh(_=None):
        with output:
            output.clear_output(wait=True)
            fig = widget.render()
            ipy_display(fig)
            plt.close(fig)

    # Build coil sliders
    coil_controls = []
    for i, coil in enumerate(widget.coils):
        on_z, on_r, on_I = _field_topology_handlers(widget, i, refresh)

        z_slider = widgets.FloatSlider(
            value=coil.z,
            min=-5.0,
            max=5.0,
            step=0.05,
            description=f"z{i} [m]",
            layout=widgets.Layout(width="350px"),
        )
        r_slider = widgets.FloatSlider(
            value=coil.r,
            min=0.01,
            max=2.0,
            step=0.01,
            description=f"r{i} [m]",
            layout=widgets.Layout(width="350px"),
        )
        I_slider = widgets.FloatSlider(
            value=coil.I,
            min=-1e6,
            max=1e6,
            step=1000.0,
            description=f"I{i} [A]",
            layout=widgets.Layout(width="350px"),
        )
        z_slider.observe(on_z, names="value")
        r_slider.observe(on_r, names="value")
        I_slider.observe(on_I, names="value")

        coil_controls.extend(
            [
                widgets.HTML(f"<b>Coil {i}</b>"),
                z_slider,
                r_slider,
                I_slider,
            ]
        )

    refresh_btn = widgets.Button(description="Refresh", button_style="info")
    refresh_btn.on_click(refresh)

    controls = widgets.VBox(
        [*coil_controls, widgets.HTML("<br/>"), refresh_btn],
        layout=widgets.Layout(width="380px", overflow_y="scroll", max_height="600px"),
    )
    main = widgets.HBox([controls, output])

    # Initial render
    refresh()

    return main
