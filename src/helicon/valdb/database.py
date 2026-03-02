"""JSON-lines storage backend for the validation database."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from helicon.valdb.schema import ValidationRecord


class ValidationDatabase:
    """Persistent validation database backed by a JSON-lines file.

    Each line in the database file is a JSON object representing one
    :class:`ValidationRecord`.  The database supports add, query (by
    case_id, contributor, data_type, and/or tags), delete, and export.

    Parameters
    ----------
    path : str | Path
        Path to the database file or directory.  If a directory is given,
        the file ``valdb.jsonl`` inside it is used.
    """

    def __init__(self, path: str | Path) -> None:
        p = Path(path).expanduser()
        if p.is_dir() or (not p.suffix):
            p.mkdir(parents=True, exist_ok=True)
            self._path = p / "valdb.jsonl"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            self._path = p

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_all(self) -> list[dict]:
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def _write_all(self, records: list[dict]) -> None:
        self._path.write_text(
            "\n".join(json.dumps(r, default=str) for r in records) + "\n" if records else "",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, record: ValidationRecord) -> None:
        """Append a record to the database.

        Parameters
        ----------
        record : ValidationRecord
        """
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump(mode="json"), default=str) + "\n")

    def count(self) -> int:
        """Return the total number of records."""
        return len(self._read_all())

    def get(self, record_id: str) -> ValidationRecord | None:
        """Fetch a record by its unique ID.

        Returns ``None`` if not found.
        """
        for raw in self._read_all():
            if raw.get("record_id") == record_id:
                return ValidationRecord.model_validate(raw)
        return None

    def query(
        self,
        *,
        case_id: str | None = None,
        contributor: str | None = None,
        data_type: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ValidationRecord]:
        """Return records matching all provided filters.

        Parameters
        ----------
        case_id : str, optional
            Exact match on ``case_id``.
        contributor : str, optional
            Exact match on ``contributor``.
        data_type : str, optional
            One of ``"experimental"``, ``"simulation"``, ``"analytical"``.
        tags : list[str], optional
            Records must contain ALL listed tags.

        Returns
        -------
        list[ValidationRecord]
        """
        results = []
        for raw in self._read_all():
            if case_id is not None and raw.get("case_id") != case_id:
                continue
            if contributor is not None and raw.get("contributor") != contributor:
                continue
            if data_type is not None and raw.get("data_type") != data_type:
                continue
            if tags is not None:
                record_tags = set(raw.get("tags", []))
                if not all(t in record_tags for t in tags):
                    continue
            results.append(ValidationRecord.model_validate(raw))
        return results

    def delete(self, record_id: str) -> bool:
        """Delete a record by ID.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
        records = self._read_all()
        before = len(records)
        records = [r for r in records if r.get("record_id") != record_id]
        if len(records) == before:
            return False
        self._write_all(records)
        return True

    def export_json(self, path: str | Path) -> None:
        """Export all records to a JSON array file.

        Parameters
        ----------
        path : str | Path
            Destination file path.
        """
        records = self._read_all()
        Path(path).write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")

    def export_csv(self, path: str | Path) -> None:
        """Export all records to CSV (flat format, metrics columns expanded).

        Parameters
        ----------
        path : str | Path
            Destination CSV file path.
        """
        import csv
        import io

        records = self._read_all()
        if not records:
            Path(path).write_text("", encoding="utf-8")
            return

        # Collect all top-level keys (excluding nested dicts) plus expanded metrics/params
        rows = []
        fieldnames_set: dict[str, None] = {}  # ordered set

        for raw in records:
            row: dict[str, Any] = {}
            for k, v in raw.items():
                if k in ("parameters", "metrics"):
                    for sub_k, sub_v in v.items():
                        col = f"{k}.{sub_k}"
                        row[col] = sub_v
                        fieldnames_set[col] = None
                elif k == "tags":
                    row["tags"] = ",".join(v)
                    fieldnames_set["tags"] = None
                else:
                    row[k] = v
                    fieldnames_set[k] = None
            rows.append(row)

        fieldnames = list(fieldnames_set.keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        Path(path).write_text(buf.getvalue(), encoding="utf-8")

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"ValidationDatabase(path={self._path}, n={self.count()})"
