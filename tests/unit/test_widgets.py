"""Tests for helicon.widgets — Jupyter widget data layer."""

from __future__ import annotations

import pytest

from helicon.widgets.coil_editor import CoilEditorWidget, CoilSpec
from helicon.widgets.field_widget import FieldTopologyWidget

# ---------------------------------------------------------------------------
# CoilSpec
# ---------------------------------------------------------------------------


def test_coil_spec_fields():
    c = CoilSpec(z=0.5, r=0.3, I=10000.0)
    assert c.z == 0.5
    assert c.r == 0.3
    assert c.I == 10000.0


# ---------------------------------------------------------------------------
# CoilEditorWidget
# ---------------------------------------------------------------------------


def test_coil_editor_update():
    coils = [CoilSpec(z=0.0, r=0.3, I=1000.0)]
    editor = CoilEditorWidget(coils=coils)
    editor.update_coil(0, z=0.5)
    assert coils[0].z == pytest.approx(0.5)


def test_coil_editor_update_all_params():
    coils = [CoilSpec(z=0.0, r=0.3, I=1000.0)]
    editor = CoilEditorWidget(coils=coils)
    editor.update_coil(0, z=0.1, r=0.4, I=2000.0)
    assert coils[0].z == pytest.approx(0.1)
    assert coils[0].r == pytest.approx(0.4)
    assert pytest.approx(2000.0) == coils[0].I


def test_coil_editor_callback():
    coils = [CoilSpec(z=0.0, r=0.3, I=1000.0)]
    called_with = []

    def cb(c):
        called_with.append([spec.z for spec in c])

    editor = CoilEditorWidget(coils=coils, on_change=cb)
    editor.update_coil(0, z=1.0)
    assert called_with == [[1.0]]


def test_coil_editor_bad_attribute():
    coils = [CoilSpec(z=0.0, r=0.3, I=1000.0)]
    editor = CoilEditorWidget(coils=coils)
    with pytest.raises(AttributeError):
        editor.update_coil(0, nonexistent=99)


def test_coil_editor_no_callback():
    coils = [CoilSpec(z=0.0, r=0.3, I=1000.0)]
    editor = CoilEditorWidget(coils=coils)
    # Should not raise even without a callback
    editor.update_coil(0, z=2.0)


def test_coil_editor_display_no_ipywidgets():
    import unittest.mock as mock

    with mock.patch.dict("sys.modules", {"ipywidgets": None}):
        import importlib

        import helicon.widgets.coil_editor as _mod

        importlib.reload(_mod)
        editor2 = _mod.CoilEditorWidget(coils=[_mod.CoilSpec(z=0.0, r=0.3, I=1e3)])
        with pytest.raises(ImportError, match="ipywidgets"):
            editor2.display()


# ---------------------------------------------------------------------------
# FieldTopologyWidget
# ---------------------------------------------------------------------------


def make_simple_widget() -> FieldTopologyWidget:
    coils = [
        CoilSpec(z=0.0, r=0.2, I=10000.0),
        CoilSpec(z=1.0, r=0.2, I=5000.0),
    ]
    return FieldTopologyWidget(
        coils=coils,
        domain_z=(-0.5, 2.0),
        domain_r_max=0.5,
        nz=32,
        nr=16,
        n_field_lines=3,
    )


def test_field_widget_compute_bfield():
    widget = make_simple_widget()
    bfield = widget.compute_bfield()
    assert "Bz" in bfield
    assert "Br" in bfield
    # BField shape is (nr, nz) — widget has nz=32, nr=16
    assert bfield["Bz"].shape == (16, 32)


def test_field_widget_compute_field_lines():
    widget = make_simple_widget()
    lines = widget.compute_field_lines()
    assert isinstance(lines, list)
    # Should have some lines traced
    assert len(lines) > 0


def test_field_widget_render_returns_figure():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    widget = make_simple_widget()
    fig = widget.render()
    assert hasattr(fig, "savefig")
    plt.close(fig)


def test_field_widget_from_preset():
    widget = FieldTopologyWidget.from_preset("sunbird")
    assert len(widget.coils) > 0
    assert widget.domain_z[0] < widget.domain_z[1]


def test_field_widget_display_no_ipywidgets():
    import unittest.mock as mock

    with mock.patch.dict(
        "sys.modules",
        {"ipywidgets": None, "IPython": None, "IPython.display": None},
    ):
        import importlib

        import helicon.widgets.field_widget as _mod

        importlib.reload(_mod)
        widget2 = _mod.FieldTopologyWidget(
            coils=[_mod.CoilSpec(z=0.0, r=0.2, I=1e4)],  # type: ignore[attr-defined]
        )
        with pytest.raises((ImportError, AttributeError)):
            widget2.display()
