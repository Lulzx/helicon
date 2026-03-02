"""Multi-objective Pareto front computation.

Identifies non-dominated (Pareto-optimal) solutions from a set of
objective vectors.  Used to build Pareto fronts from parameter scan
results where multiple objectives (e.g. thrust, η_d, coil mass) are
traded off against each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParetoResult:
    """Pareto front from a multi-objective evaluation.

    Attributes
    ----------
    front_mask : np.ndarray of bool, shape (n_points,)
        True for Pareto-optimal (non-dominated) points.
    front_indices : np.ndarray of int
        Indices of Pareto-optimal points in the original array.
    costs : np.ndarray, shape (n_points, n_objectives)
        Original cost matrix (minimization convention).
    """

    front_mask: np.ndarray
    front_indices: np.ndarray
    costs: np.ndarray

    @property
    def front_costs(self) -> np.ndarray:
        """Cost values for Pareto-optimal points only."""
        return self.costs[self.front_indices]

    def plot(
        self,
        *,
        labels: tuple[str, str] | None = None,
        ax=None,
        figsize: tuple = (6, 5),
        dominated_color: str = "lightgray",
        front_color: str = "steelblue",
    ):
        """Plot the 2-objective Pareto front (minimization convention).

        Parameters
        ----------
        labels : tuple of str, optional
            Axis labels ``(x_label, y_label)``.
        ax : matplotlib Axes, optional
            Axes to draw on. Creates new figure if None.
        figsize : tuple
            Figure size when creating a new figure.
        dominated_color : str
            Color for dominated (non-Pareto) points.
        front_color : str
            Color for Pareto-optimal points.

        Returns
        -------
        fig, ax
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for ParetoResult.plot()") from exc

        if self.costs.shape[1] != 2:
            raise ValueError("plot() is only supported for 2-objective problems")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        dominated_mask = self.front_mask == False  # noqa: E712
        if np.any(dominated_mask):
            ax.scatter(
                self.costs[dominated_mask, 0],
                self.costs[dominated_mask, 1],
                c=dominated_color,
                s=30,
                label="Dominated",
                zorder=2,
            )

        front = self.front_costs
        # Sort by first objective for line connection
        order = np.argsort(front[:, 0])
        front_sorted = front[order]
        ax.plot(
            front_sorted[:, 0],
            front_sorted[:, 1],
            "o-",
            color=front_color,
            ms=7,
            lw=1.5,
            label="Pareto front",
            zorder=3,
        )

        if labels:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
        ax.legend()
        ax.set_title("Pareto Front")
        fig.tight_layout()

        return fig, ax


def is_dominated(costs: np.ndarray) -> np.ndarray:
    """Return a boolean mask: True if a point is dominated by any other.

    A point *i* is dominated by point *j* if *j* is at least as good in
    all objectives and strictly better in at least one.

    Parameters
    ----------
    costs : array, shape (n_points, n_objectives)
        Objective values. **Minimization** convention.

    Returns
    -------
    dominated : bool array, shape (n_points,)
    """
    costs = np.asarray(costs, dtype=float)
    n = costs.shape[0]
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # j dominates i: all j ≤ i AND at least one j < i
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                dominated[i] = True
                break

    return dominated


def pareto_front(costs: np.ndarray) -> ParetoResult:
    """Compute the Pareto front from a set of objective vectors.

    Parameters
    ----------
    costs : array, shape (n_points, n_objectives)
        Objective values in **minimization** convention.
        To maximize an objective, pass its negation.

    Returns
    -------
    ParetoResult
        Contains ``front_mask`` (bool array) and ``front_indices``.

    Examples
    --------
    Maximize thrust (F) and detachment efficiency (η_d) simultaneously::

        costs = np.column_stack([-thrust_values, -eta_d_values])
        result = pareto_front(costs)
        best_configs = scan_points[result.front_indices]
    """
    costs = np.atleast_2d(np.asarray(costs, dtype=float))
    mask = ~is_dominated(costs)
    indices = np.where(mask)[0]
    return ParetoResult(front_mask=mask, front_indices=indices, costs=costs)


def hypervolume_indicator(front_costs: np.ndarray, reference: np.ndarray) -> float:
    """Compute the hypervolume indicator for a 2-objective Pareto front.

    The hypervolume is the area of the objective space dominated by the
    front and bounded by a reference point.  Higher is better.

    Only supported for 2 objectives (exact algorithm).

    Parameters
    ----------
    front_costs : array, shape (n_front, 2)
        Pareto-optimal objective values (minimization convention).
    reference : array, shape (2,)
        Reference (worst) point; must dominate all front points.

    Returns
    -------
    float
        Hypervolume.
    """
    costs = np.asarray(front_costs, dtype=float)
    ref = np.asarray(reference, dtype=float)
    if costs.shape[1] != 2:
        raise ValueError("hypervolume_indicator only supports 2 objectives")

    # Sort by first objective ascending
    order = np.argsort(costs[:, 0])
    sorted_costs = costs[order]

    hv = 0.0
    prev_y = ref[1]
    for point in sorted_costs:
        width = ref[0] - point[0]
        height = prev_y - point[1]
        if width > 0 and height > 0:
            hv += width * height
        prev_y = min(prev_y, point[1])

    return hv
