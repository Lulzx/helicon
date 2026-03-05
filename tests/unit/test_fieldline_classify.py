"""Tests for helicon.postprocess.fieldline_classify — particle topology classification."""

from __future__ import annotations

import numpy as np
import pytest

from helicon.postprocess.fieldline_classify import ParticleClassification, _navigate_openpmd


class TestParticleClassificationDataclass:
    def test_fields_accessible(self):
        n = 100
        rng = np.random.default_rng(7)
        labels = rng.integers(0, 3, size=n).astype(np.int32)
        pc = ParticleClassification(
            species="D_plus",
            n_open=int(np.sum(labels == 0)),
            n_closed=int(np.sum(labels == 1)),
            n_separatrix=int(np.sum(labels == 2)),
            n_total=n,
            labels=labels,
            positions_r=rng.uniform(0, 0.1, n),
            positions_z=rng.uniform(-0.3, 1.0, n),
        )
        assert pc.n_total == n
        assert pc.n_open + pc.n_closed + pc.n_separatrix == n
        assert pc.labels.dtype == np.int32
        assert pc.positions_r.shape == (n,)
        assert pc.positions_z.shape == (n,)

    def test_all_open(self):
        n = 50
        pc = ParticleClassification(
            species="He_2plus",
            n_open=50,
            n_closed=0,
            n_separatrix=0,
            n_total=50,
            labels=np.zeros(n, dtype=np.int32),
            positions_r=np.ones(n) * 0.05,
            positions_z=np.linspace(0, 1, n),
        )
        assert pc.n_open == n
        assert np.all(pc.labels == 0)

    def test_species_name_preserved(self):
        pc = ParticleClassification(
            species="alpha",
            n_open=10, n_closed=5, n_separatrix=1, n_total=16,
            labels=np.zeros(16, dtype=np.int32),
            positions_r=np.zeros(16),
            positions_z=np.zeros(16),
        )
        assert pc.species == "alpha"


class TestNavigateOpenPMD:
    def test_flat_dict_returned_as_is(self):
        """If no 'data' key, return the dict unchanged."""
        d = {"particles": "mock"}
        result = _navigate_openpmd(d)
        assert result is d

    def test_navigates_data_key(self):
        """With 'data' key, returns the last iteration's subtree."""
        d = {"data": {"100": {"particles": "step100"}, "200": {"particles": "step200"}}}
        result = _navigate_openpmd(d)
        assert result == {"particles": "step200"}

    def test_navigates_single_iteration(self):
        d = {"data": {"0": {"particles": "step0"}}}
        result = _navigate_openpmd(d)
        assert result == {"particles": "step0"}

    def test_sorts_iterations_numerically(self):
        """Iterations must be sorted as integers, not lexicographically."""
        d = {"data": {"9": {"step": 9}, "10": {"step": 10}, "2": {"step": 2}}}
        result = _navigate_openpmd(d)
        assert result == {"step": 10}


class TestClassifyParticlesErrors:
    def test_raises_on_missing_h5_files(self, tmp_path):
        """classify_particles raises FileNotFoundError if no HDF5 files present."""
        import helicon
        from helicon.postprocess.fieldline_classify import classify_particles

        config = helicon.Config.from_preset("sunbird")
        bfield = helicon.fields.compute(config.nozzle)
        with pytest.raises(FileNotFoundError, match="No HDF5 files"):
            classify_particles(tmp_path, bfield)
