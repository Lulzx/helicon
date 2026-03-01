"""Parameter scan infrastructure for MagNozzleX.

Supports grid sweeps and Latin Hypercube sampling over coil/plasma parameters.
Each scan point modifies a base SimConfig via dot-notation parameter paths
(e.g. ``"coils.0.I"``, ``"plasma.T_i_eV"``).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from magnozzlex.config.parser import SimConfig


@dataclass
class ParameterRange:
    """One parameter axis in a scan.

    Parameters
    ----------
    path : str
        Dot-notation path into the config dict, e.g. ``"coils.0.I"`` or
        ``"plasma.T_i_eV"``.
    low, high : float
        Inclusive range boundaries.
    n : int
        Number of points along this axis.
    """

    path: str
    low: float
    high: float
    n: int

    def values(self) -> np.ndarray:
        """Return ``n`` linearly spaced values from ``low`` to ``high``."""
        return np.linspace(self.low, self.high, self.n)

    @classmethod
    def from_string(cls, s: str) -> ParameterRange:
        """Parse ``"path:low:high:n"`` format used by the CLI."""
        parts = s.split(":")
        if len(parts) != 4:
            raise ValueError(f"Expected 'path:low:high:n', got {s!r}")
        return cls(path=parts[0], low=float(parts[1]), high=float(parts[2]), n=int(parts[3]))


@dataclass
class ScanPoint:
    """One point in a parameter scan."""

    index: int
    params: dict[str, float]
    config: SimConfig


@dataclass
class ScanResult:
    """Results from a completed parameter scan."""

    points: list[ScanPoint]
    metrics: list[dict[str, Any]]
    param_names: list[str]
    base_config: SimConfig


def _set_nested(data: dict, path: str, value: float) -> dict:
    """Set a value in a nested structure using dot-notation path.

    Handles list indices (``"coils.0.I"``) and searches recursively when
    a key is not found at the current level — so shorthand paths like
    ``"coils.0.I"`` resolve correctly even when ``coils`` is nested under
    ``nozzle`` in the full config dict.
    """
    keys = path.split(".")
    if not _set_nested_keys(data, keys, value):
        raise KeyError(f"Path {path!r} not found in config dict")
    return data


def _set_nested_keys(obj: Any, keys: list[str], value: float) -> bool:
    """Recursively navigate ``obj`` by ``keys`` and set the leaf to ``value``.

    Returns True if the path was found and the value set.
    """
    if not keys:
        return False
    key = keys[0]
    rest = keys[1:]

    if isinstance(obj, list):
        try:
            idx = int(key)
        except ValueError:
            return False
        if not rest:
            obj[idx] = value
            return True
        return _set_nested_keys(obj[idx], rest, value)

    if isinstance(obj, dict):
        if key in obj:
            if not rest:
                obj[key] = value
                return True
            return _set_nested_keys(obj[key], rest, value)
        # Key not at this level — search one level deeper
        for v in obj.values():
            if isinstance(v, (dict, list)) and _set_nested_keys(v, keys, value):
                return True
        return False

    return False


def _apply_params(base_config: SimConfig, params: dict[str, float]) -> SimConfig:
    """Return a new SimConfig with the given parameters applied."""
    data = base_config.model_dump(mode="python")
    for path, value in params.items():
        _set_nested(data, path, value)
    return SimConfig.model_validate(data)


def generate_scan_points(
    base_config: SimConfig,
    ranges: list[ParameterRange],
    *,
    method: str = "grid",
    seed: int = 0,
) -> list[ScanPoint]:
    """Generate scan points from parameter ranges.

    Parameters
    ----------
    base_config : SimConfig
        Base configuration to modify.
    ranges : list of ParameterRange
        Parameter axes to vary.
    method : ``"grid"`` | ``"lhc"``
        ``"grid"`` — full Cartesian product.
        ``"lhc"`` — Latin Hypercube sampling with total n = product of all n values.
    seed : int
        Random seed for LHC sampling.

    Returns
    -------
    list of ScanPoint
    """
    if method == "grid":
        value_lists = [r.values() for r in ranges]
        combos = list(itertools.product(*value_lists))
        points = []
        for i, combo in enumerate(combos):
            params = {r.path: float(v) for r, v in zip(ranges, combo)}
            config = _apply_params(base_config, params)
            points.append(ScanPoint(index=i, params=params, config=config))
        return points

    elif method == "lhc":
        n_total = math.prod(r.n for r in ranges)
        rng = np.random.default_rng(seed)
        n_params = len(ranges)
        lhc = np.zeros((n_total, n_params))
        for j in range(n_params):
            perm = rng.permutation(n_total)
            lhc[:, j] = (perm + rng.uniform(size=n_total)) / n_total
        points = []
        for i in range(n_total):
            params: dict[str, float] = {}
            for j, r in enumerate(ranges):
                params[r.path] = float(r.low + lhc[i, j] * (r.high - r.low))
            config = _apply_params(base_config, params)
            points.append(ScanPoint(index=i, params=params, config=config))
        return points

    else:
        raise ValueError(f"Unknown scan method {method!r}. Use 'grid' or 'lhc'.")


def run_scan(
    base_config: SimConfig,
    ranges: list[ParameterRange],
    *,
    output_base: str | Path = "scan_results",
    method: str = "grid",
    dry_run: bool = False,
    seed: int = 0,
) -> ScanResult:
    """Run a full parameter scan.

    Each scan point gets its own subdirectory ``output_base/point_NNNN/``.

    Parameters
    ----------
    base_config : SimConfig
        Base configuration.
    ranges : list of ParameterRange
        Parameters to vary.
    output_base : path
        Root output directory.
    method : ``"grid"`` | ``"lhc"``
        Sampling strategy.
    dry_run : bool
        Generate configs and B-fields without launching WarpX.
    seed : int
        RNG seed for LHC sampling.

    Returns
    -------
    ScanResult
    """
    from magnozzlex.runner.launch import run_simulation

    output_base = Path(output_base)
    points = generate_scan_points(base_config, ranges, method=method, seed=seed)
    metrics: list[dict[str, Any]] = []

    for point in points:
        point_dir = output_base / f"point_{point.index:04d}"
        result = run_simulation(point.config, output_dir=point_dir, dry_run=dry_run)
        metrics.append(
            {
                "output_dir": str(result.output_dir),
                "success": result.success,
                "wall_time_seconds": result.wall_time_seconds,
                **point.params,
            }
        )

    return ScanResult(
        points=points,
        metrics=metrics,
        param_names=[r.path for r in ranges],
        base_config=base_config,
    )
