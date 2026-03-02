"""Tests for helicon.fields.cache — HDF5 field map cache."""

from __future__ import annotations

import tempfile

import numpy as np
import pytest

from helicon.fields.biot_savart import BField, Coil, Grid, compute_bfield
from helicon.fields.cache import FieldCache, compute_bfield_cached

pytest.importorskip("h5py")

COILS = [Coil(z=0.0, r=0.1, I=1000.0)]
GRID = Grid(z_min=-0.3, z_max=0.3, r_max=0.2, nz=16, nr=8)


@pytest.fixture()
def cache(tmp_path):
    return FieldCache(cache_dir=tmp_path)


# --- key tests ---


def test_key_deterministic(cache):
    """Same coils + grid produce the same cache key."""
    k1 = cache.key(COILS, GRID)
    k2 = cache.key(COILS, GRID)
    assert k1 == k2


def test_key_changes_with_coil(cache):
    """Different coil parameters produce a different key."""
    k1 = cache.key(COILS, GRID)
    k2 = cache.key([Coil(z=0.0, r=0.1, I=2000.0)], GRID)
    assert k1 != k2


# --- put / get ---


def test_get_returns_none_on_miss(cache):
    assert cache.get(COILS, GRID) is None


def test_put_get_roundtrip(cache):
    bf = compute_bfield(COILS, GRID, backend="numpy")
    cache.put(COILS, GRID, bf)
    loaded = cache.get(COILS, GRID)
    assert loaded is not None
    np.testing.assert_array_equal(bf.Bz, loaded.Bz)


# --- size / clear ---


def test_size(cache):
    assert cache.size() == 0
    bf = compute_bfield(COILS, GRID, backend="numpy")
    cache.put(COILS, GRID, bf)
    assert cache.size() == 1


def test_clear(cache):
    bf = compute_bfield(COILS, GRID, backend="numpy")
    cache.put(COILS, GRID, bf)
    assert cache.size() == 1
    cache.clear()
    assert cache.size() == 0


# --- compute_bfield_cached ---


def test_compute_bfield_cached_returns_bfield(tmp_path):
    c = FieldCache(cache_dir=tmp_path)
    bf = compute_bfield_cached(COILS, GRID, cache=c, backend="numpy")
    assert isinstance(bf, BField)
    assert bf.Bz.shape == (GRID.nr, GRID.nz)


def test_compute_bfield_cached_uses_cache(tmp_path):
    c = FieldCache(cache_dir=tmp_path)
    assert c.size() == 0
    compute_bfield_cached(COILS, GRID, cache=c, backend="numpy")
    assert c.size() == 1
    # Second call should not increase size
    compute_bfield_cached(COILS, GRID, cache=c, backend="numpy")
    assert c.size() == 1
