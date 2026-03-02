"""Tests for helicon.provenance — audit trail (v2.0)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from helicon.provenance import ProvenanceDB, ProvenanceRecord


class TestProvenanceRecord:
    def test_roundtrip_dict(self):
        rec = ProvenanceRecord(
            record_id="abc-123",
            timestamp="2026-01-01T00:00:00+00:00",
            source="tier1_analytical",
            config={"coil_r": 0.1},
            metrics={"thrust_N": 0.5},
            fidelity_tier=1,
        )
        d = rec.to_dict()
        rec2 = ProvenanceRecord.from_dict(d)
        assert rec2.record_id == "abc-123"
        assert rec2.fidelity_tier == 1
        assert rec2.metrics["thrust_N"] == 0.5


class TestProvenanceDB:
    def test_record_returns_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = ProvenanceDB(Path(tmp) / "prov.jsonl")
            rid = db.record(
                config={"coil_r": 0.1},
                source="test",
                metrics={"thrust_N": 0.4},
            )
        assert isinstance(rid, str)
        assert len(rid) > 0

    def test_get_returns_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = ProvenanceDB(Path(tmp) / "prov.jsonl")
            rid = db.record(
                config={"coil_r": 0.1},
                source="test",
                metrics={"thrust_N": 0.4},
            )
            rec = db.get(rid)
        assert rec is not None
        assert rec.record_id == rid
        assert rec.source == "test"

    def test_get_unknown_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = ProvenanceDB(Path(tmp) / "prov.jsonl")
            assert db.get("nonexistent-id") is None

    def test_lineage_single_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = ProvenanceDB(Path(tmp) / "prov.jsonl")
            rid = db.record(config={}, source="root", metrics={})
            chain = db.lineage(rid)
        assert len(chain) == 1
        assert chain[0].record_id == rid

    def test_lineage_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = ProvenanceDB(Path(tmp) / "prov.jsonl")
            r1 = db.record(config={}, source="tier1")
            r2 = db.record(config={}, source="tier2", parent_ids=[r1])
            r3 = db.record(config={}, source="tier3", parent_ids=[r2])
            chain = db.lineage(r3)
        # Chain: r1 → r2 → r3
        assert len(chain) == 3
        assert chain[0].record_id == r1
        assert chain[-1].record_id == r3

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "prov.jsonl"
            db = ProvenanceDB(p)
            rid = db.record(config={"x": 1}, source="test", metrics={"y": 2.0})
            # Reload from disk
            db2 = ProvenanceDB(p)
            rec = db2.get(rid)
        assert rec is not None
        assert rec.metrics["y"] == 2.0

    def test_all_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = ProvenanceDB(Path(tmp) / "prov.jsonl")
            for i in range(5):
                db.record(config={"i": i}, source=f"step{i}", metrics={})
            recs = db.all_records()
        assert len(recs) == 5

    def test_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = ProvenanceDB(Path(tmp) / "prov.jsonl")
            db.record(config={}, source="tier1", fidelity_tier=1, metrics={})
            db.record(config={}, source="tier2", fidelity_tier=2, metrics={})
            summary = db.summary()
        assert summary["n_records"] == 2
        assert summary["tier_counts"][1] == 1
        assert summary["tier_counts"][2] == 1
        assert "tier1" in summary["sources"]

    def test_parent_directory_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            deep = Path(tmp) / "a" / "b" / "prov.jsonl"
            _db = ProvenanceDB(deep)
            assert deep.exists()
