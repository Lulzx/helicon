"""Provenance and audit trail for nozzle design decisions (v2.0).

Every design decision is traceable from final coil geometry back through
optimisation history, PIC validation, and input assumptions.

The :class:`ProvenanceDB` stores a chain of :class:`ProvenanceRecord` objects
in a JSON-lines file.  Each record links:
  - a design configuration (coil geometry + plasma params)
  - the tool / module that produced it (``source``)
  - the parent record(s) it was derived from
  - performance metrics and fidelity tier

Usage::

    from helicon.provenance import ProvenanceDB

    db = ProvenanceDB(path="results/provenance.jsonl")
    rec_id = db.record(
        config=config_dict,
        source="tier1_analytical",
        metrics={"thrust_N": 0.42, "eta_d": 0.85},
    )
    db.record(
        config=refined_config,
        source="tier2_surrogate",
        parent_ids=[rec_id],
        metrics={"thrust_N": 0.45, "eta_d": 0.87},
        fidelity_tier=2,
    )
    lineage = db.lineage(rec_id)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ProvenanceRecord:
    """Single node in the design provenance graph.

    Attributes
    ----------
    record_id : str
        UUID for this record.
    timestamp : str
        ISO 8601 UTC timestamp.
    source : str
        Module / tool that created this record
        (e.g. ``"tier1_analytical"``, ``"tier2_surrogate"``, ``"tier3_warpx"``).
    config : dict
        Snapshot of the design configuration at this node.
    metrics : dict
        Performance metrics at this fidelity level.
    fidelity_tier : int
        1 = analytical, 2 = surrogate, 3 = WarpX PIC.
    parent_ids : list[str]
        Record IDs of the parent nodes.
    notes : str
        Free-text annotation.
    """

    record_id: str
    timestamp: str
    source: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    fidelity_tier: int = 1
    parent_ids: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "config": self.config,
            "metrics": self.metrics,
            "fidelity_tier": self.fidelity_tier,
            "parent_ids": self.parent_ids,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProvenanceRecord:
        return cls(
            record_id=d["record_id"],
            timestamp=d["timestamp"],
            source=d["source"],
            config=d.get("config", {}),
            metrics=d.get("metrics", {}),
            fidelity_tier=d.get("fidelity_tier", 1),
            parent_ids=d.get("parent_ids", []),
            notes=d.get("notes", ""),
        )


class ProvenanceDB:
    """JSON-lines provenance database.

    Each line in the file is a JSON-serialised :class:`ProvenanceRecord`.

    Parameters
    ----------
    path : path-like
        Path to the ``.jsonl`` provenance file.  Created if absent.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")
        self._cache: dict[str, ProvenanceRecord] = {}
        self._load()

    def _load(self) -> None:
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = ProvenanceRecord.from_dict(json.loads(line))
                self._cache[rec.record_id] = rec
            except (json.JSONDecodeError, KeyError):
                pass

    def record(
        self,
        config: dict[str, Any],
        source: str,
        metrics: dict[str, Any] | None = None,
        parent_ids: list[str] | None = None,
        fidelity_tier: int = 1,
        notes: str = "",
    ) -> str:
        """Append a new provenance record and return its ID.

        Parameters
        ----------
        config : dict
            Design configuration snapshot.
        source : str
            Originating module / tool.
        metrics : dict, optional
            Performance metrics.
        parent_ids : list of str, optional
            IDs of parent records (e.g. Tier-1 result that was promoted).
        fidelity_tier : int
            1 / 2 / 3.
        notes : str
            Free-text annotation.

        Returns
        -------
        str
            The new record's UUID.
        """
        rec = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            timestamp=datetime.now(tz=UTC).isoformat(),
            source=source,
            config=config,
            metrics=metrics or {},
            fidelity_tier=fidelity_tier,
            parent_ids=parent_ids or [],
            notes=notes,
        )
        with self.path.open("a") as f:
            f.write(json.dumps(rec.to_dict()) + "\n")
        self._cache[rec.record_id] = rec
        return rec.record_id

    def get(self, record_id: str) -> ProvenanceRecord | None:
        """Retrieve a record by ID."""
        return self._cache.get(record_id)

    def lineage(self, record_id: str) -> list[ProvenanceRecord]:
        """Return the full ancestor chain for a record (depth-first).

        The returned list starts with the root ancestor(s) and ends
        with the record itself.
        """
        visited: list[ProvenanceRecord] = []
        seen: set[str] = set()

        def _walk(rid: str) -> None:
            if rid in seen:
                return
            seen.add(rid)
            rec = self._cache.get(rid)
            if rec is None:
                return
            for pid in rec.parent_ids:
                _walk(pid)
            visited.append(rec)

        _walk(record_id)
        return visited

    def all_records(self) -> list[ProvenanceRecord]:
        """Return all stored records in insertion order."""
        return list(self._cache.values())

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary of the provenance database."""
        records = self.all_records()
        tier_counts: dict[int, int] = {}
        sources: set[str] = set()
        for r in records:
            tier_counts[r.fidelity_tier] = tier_counts.get(r.fidelity_tier, 0) + 1
            sources.add(r.source)
        return {
            "n_records": len(records),
            "tier_counts": tier_counts,
            "sources": sorted(sources),
            "path": str(self.path),
        }
