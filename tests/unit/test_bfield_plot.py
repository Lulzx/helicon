"""Tests for BField.plot() and ParetoResult.plot()."""

from __future__ import annotations

import unittest.mock as mock

import numpy as np
import pytest

from magnozzlex.fields.biot_savart import BField, Coil, Grid, compute_bfield
from magnozzlex.optimize.pareto import ParetoResult, pareto_front


@pytest.fixture
def simple_bfield():
    coils = [Coil(z=0.0, r=0.1, I=10000.0)]
    grid = Grid(z_min=-0.2, z_max=0.5, r_max=0.3, nz=32, nr=16)
    return compute_bfield(coils, grid, backend="numpy")


def test_bfield_plot_requires_matplotlib(simple_bfield):
    """BField.plot() raises ImportError when matplotlib unavailable."""
    import sys
    with mock.patch.dict(sys.modules, {"matplotlib.pyplot": None}):
        with pytest.raises((ImportError, TypeError)):
            simple_bfield.plot()


def test_bfield_plot_returns_fig_ax(simple_bfield):
    """BField.plot() returns (fig, ax) tuple."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    fig, ax = simple_bfield.plot()
    assert fig is not None
    assert ax is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_bfield_plot_component_bz(simple_bfield):
    """BField.plot() works with Bz component."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = simple_bfield.plot(component="Bz")
    assert ax.get_title() == "Magnetic Field — Bz"
    plt.close(fig)


def test_bfield_plot_component_br(simple_bfield):
    """BField.plot() works with Br component."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = simple_bfield.plot(component="Br")
    assert ax.get_title() == "Magnetic Field — Br"
    plt.close(fig)


def test_bfield_plot_component_bmag(simple_bfield):
    """BField.plot() works with |B| component."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = simple_bfield.plot(component="Bmag")
    assert "Bmag" in ax.get_title() or "|B|" in ax.get_title() or "Bmag" in ax.get_title()
    plt.close(fig)


def test_bfield_plot_invalid_component(simple_bfield):
    """BField.plot() raises ValueError for unknown component."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")

    with pytest.raises(ValueError, match="Unknown component"):
        simple_bfield.plot(component="invalid")


def test_bfield_plot_no_field_lines(simple_bfield):
    """BField.plot() works with field_lines=False."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = simple_bfield.plot(field_lines=False)
    assert fig is not None
    plt.close(fig)


def test_bfield_plot_custom_axes(simple_bfield):
    """BField.plot() accepts an existing axes object."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_pre, ax_pre = plt.subplots()
    fig_out, ax_out = simple_bfield.plot(ax=ax_pre)
    assert ax_out is ax_pre
    plt.close(fig_pre)


# --- ParetoResult.plot() ---

@pytest.fixture
def simple_pareto():
    """2-objective Pareto front."""
    costs = np.array([
        [1.0, 4.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [4.0, 1.0],
        [3.0, 3.5],  # dominated
    ])
    return pareto_front(costs)


def test_pareto_plot_returns_fig_ax(simple_pareto):
    """ParetoResult.plot() returns (fig, ax)."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = simple_pareto.plot()
    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_pareto_plot_with_labels(simple_pareto):
    """ParetoResult.plot() sets axis labels."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = simple_pareto.plot(labels=("-Thrust (N)", "-η_det"))
    assert ax.get_xlabel() == "-Thrust (N)"
    assert ax.get_ylabel() == "-η_det"
    plt.close(fig)


def test_pareto_plot_rejects_3d():
    """ParetoResult.plot() raises ValueError for 3-objective problems."""
    pytest.importorskip("matplotlib")
    costs = np.random.rand(10, 3)
    result = pareto_front(costs)
    with pytest.raises(ValueError, match="2-objective"):
        result.plot()


def test_pareto_plot_no_dominated_points():
    """ParetoResult.plot() handles case where all points are Pareto-optimal."""
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # All 4 points non-dominated
    costs = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    result = pareto_front(costs)
    fig, ax = result.plot()
    assert fig is not None
    plt.close(fig)
