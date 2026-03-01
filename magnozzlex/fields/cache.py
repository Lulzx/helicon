"""HDF5 field map cache for Biot-Savart computations.

Caches BField results keyed by a SHA-256 hash of coil + grid parameters,
avoiding redundant recomputation for the same geometry.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections.abc import Sequence

from magnozzlex.fields.biot_savart import BField, Coil, Grid, compute_bfield


class FieldCache:
    """On-disk HDF5 cache for BField results."""

    def __init__(self, cache_dir: str | pathlib.Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = pathlib.Path.home() / ".cache" / "magnozzlex" / "bfield"
        self._dir = pathlib.Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def key(self, coils: Sequence[Coil], grid: Grid) -> str:
        """SHA-256 cache key (first 16 hex chars) from coil + grid params."""
        coil_data = sorted(
            [{"z": c.z, "r": c.r, "I": c.I} for c in coils],
            key=lambda d: (d["z"], d["r"], d["I"]),
        )
        grid_data = {
            "z_min": grid.z_min,
            "z_max": grid.z_max,
            "r_max": grid.r_max,
            "nz": grid.nz,
            "nr": grid.nr,
        }
        blob = json.dumps({"coils": coil_data, "grid": grid_data}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def get(self, coils: Sequence[Coil], grid: Grid) -> BField | None:
        """Load a cached BField, or return None on miss / corrupt file."""
        path = self._dir / f"{self.key(coils, grid)}.h5"
        if not path.exists():
            return None
        try:
            return BField.load(str(path))
        except Exception:
            return None

    def put(self, coils: Sequence[Coil], grid: Grid, bfield: BField) -> None:
        """Save a BField to the cache."""
        path = self._dir / f"{self.key(coils, grid)}.h5"
        bfield.save(str(path))

    def clear(self) -> None:
        """Delete all .h5 files in the cache directory."""
        for f in self._dir.glob("*.h5"):
            f.unlink()

    def size(self) -> int:
        """Number of cached .h5 files."""
        return len(list(self._dir.glob("*.h5")))


_default_cache: FieldCache | None = None


def get_default_cache() -> FieldCache:
    """Return (lazily-initialised) module-level default cache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = FieldCache()
    return _default_cache


def compute_bfield_cached(
    coils: Sequence[Coil],
    grid: Grid,
    *,
    cache: FieldCache | None = None,
    backend: str = "auto",
    n_phi: int = 128,
) -> BField:
    """Like ``compute_bfield`` but with transparent HDF5 caching."""
    if cache is None:
        cache = get_default_cache()

    hit = cache.get(coils, grid)
    if hit is not None:
        return hit

    bf = compute_bfield(coils, grid, backend=backend, n_phi=n_phi)
    cache.put(coils, grid, bf)
    return bf
