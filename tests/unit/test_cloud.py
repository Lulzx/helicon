"""Tests for helicon.cloud — cloud HPC backend abstraction (spec v1.3)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from helicon.cloud.backends import (
    CloudJob,
    LocalBackend,
    get_backend,
)

# ---------------------------------------------------------------------------
# CloudJob serialisation
# ---------------------------------------------------------------------------


class TestCloudJob:
    def _make_job(self):
        return CloudJob(
            job_id="test-123",
            backend="local",
            status="pending",
            config_path="/tmp/config.yaml",
            output_dir="/tmp/out",
            submitted_at="2026-03-01T00:00:00+00:00",
        )

    def test_to_dict_keys(self):
        job = self._make_job()
        d = job.to_dict()
        assert "job_id" in d
        assert "backend" in d
        assert "status" in d

    def test_save_load_roundtrip(self):
        job = self._make_job()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "job.json"
            job.save(p)
            job2 = CloudJob.load(p)
        assert job2.job_id == job.job_id
        assert job2.status == job.status

    def test_load_creates_parent_dirs(self):
        job = self._make_job()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "sub" / "dir" / "job.json"
            job.save(p)
            assert p.exists()


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestGetBackend:
    def test_local(self):
        b = get_backend("local")
        assert isinstance(b, LocalBackend)
        assert b.name == "local"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("gcp_unknown_backend")


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------


class TestLocalBackend:
    def _make_config(self, tmpdir: str) -> str:
        import yaml

        data = {
            "nozzle": {
                "type": "solenoid",
                "coils": [{"z": 0.0, "r": 0.1, "I": 10000}],
                "domain": {"z_min": -0.3, "z_max": 1.0, "r_max": 0.5},
            },
            "plasma": {
                "n0": 1e18,
                "T_i_eV": 100,
                "T_e_eV": 100,
                "v_injection_ms": 50000,
            },
        }
        p = Path(tmpdir) / "cfg.yaml"
        p.write_text(yaml.dump(data))
        return str(p)

    def test_submit_returns_done_job(self):
        backend = LocalBackend()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_config(tmp)
            job = backend.submit(
                cfg,
                Path(tmp) / "out",
                extra={
                    "ranges": [
                        {"path": "nozzle.coils.0.I", "low": 5000, "high": 15000, "n": 2}
                    ],
                    "method": "grid",
                    "dry_run": True,
                    "seed": 0,
                },
            )
        assert job.status == "done"
        assert job.backend == "local"

    def test_status_is_done(self):
        assert LocalBackend().status("any-id") == "done"

    def test_retrieve_true(self):
        job = CloudJob(
            job_id="x",
            backend="local",
            status="done",
            config_path="",
            output_dir="",
            submitted_at="",
        )
        assert LocalBackend().retrieve(job) is True


# ---------------------------------------------------------------------------
# submit_cloud_scan
# ---------------------------------------------------------------------------


class TestSubmitCloudScan:
    def _make_config(self, tmp: str) -> str:
        import yaml

        data = {
            "nozzle": {
                "type": "solenoid",
                "coils": [{"z": 0.0, "r": 0.1, "I": 10000}],
                "domain": {"z_min": -0.3, "z_max": 1.0, "r_max": 0.5},
            },
            "plasma": {
                "n0": 1e18,
                "T_i_eV": 100,
                "T_e_eV": 100,
                "v_injection_ms": 50000,
            },
        }
        p = Path(tmp) / "cfg.yaml"
        p.write_text(yaml.dump(data))
        return str(p)

    def test_local_submit_and_manifest(self):
        from helicon.cloud.submit import submit_cloud_scan
        from helicon.optimize.scan import ParameterRange

        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_config(tmp)
            ranges = [ParameterRange(path="nozzle.coils.0.I", low=5000, high=15000, n=2)]
            output = Path(tmp) / "scan_out"
            job = submit_cloud_scan(
                cfg,
                ranges,
                output_dir=output,
                backend="local",
                dry_run=True,
            )
            manifest = output / "cloud_job.json"
            assert manifest.exists()
            data = json.loads(manifest.read_text())
            assert data["job_id"] == job.job_id
