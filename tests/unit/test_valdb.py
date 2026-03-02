"""Tests for helicon.valdb — validation database."""

from __future__ import annotations

import json

import pytest

from helicon.valdb import ValidationDatabase, ValidationRecord


def _make_record(**kwargs) -> ValidationRecord:
    defaults = dict(
        case_id="test_case",
        source="Test source",
        contributor="tester",
        data_type="simulation",
        parameters={"beta": 0.1},
        metrics={"eta_d": 0.72},
        tags=["test"],
    )
    defaults.update(kwargs)
    return ValidationRecord(**defaults)


@pytest.fixture
def tmp_db(tmp_path):
    return ValidationDatabase(tmp_path / "valdb.jsonl")


def test_add_and_count(tmp_db):
    tmp_db.add(_make_record())
    assert tmp_db.count() == 1


def test_add_multiple(tmp_db):
    for i in range(5):
        tmp_db.add(_make_record(case_id=f"case_{i}"))
    assert tmp_db.count() == 5


def test_get_by_id(tmp_db):
    rec = _make_record()
    tmp_db.add(rec)
    fetched = tmp_db.get(rec.record_id)
    assert fetched is not None
    assert fetched.case_id == rec.case_id


def test_get_missing_returns_none(tmp_db):
    assert tmp_db.get("nonexistent-id") is None


def test_query_by_case_id(tmp_db):
    tmp_db.add(_make_record(case_id="alpha"))
    tmp_db.add(_make_record(case_id="beta"))
    results = tmp_db.query(case_id="alpha")
    assert len(results) == 1
    assert results[0].case_id == "alpha"


def test_query_by_contributor(tmp_db):
    tmp_db.add(_make_record(contributor="alice"))
    tmp_db.add(_make_record(contributor="bob"))
    results = tmp_db.query(contributor="alice")
    assert len(results) == 1


def test_query_by_data_type(tmp_db):
    tmp_db.add(_make_record(data_type="experimental"))
    tmp_db.add(_make_record(data_type="simulation"))
    results = tmp_db.query(data_type="experimental")
    assert len(results) == 1
    assert results[0].data_type == "experimental"


def test_query_by_tags_single(tmp_db):
    tmp_db.add(_make_record(tags=["merino_ahedo", "collisionless"]))
    tmp_db.add(_make_record(tags=["vasimr"]))
    results = tmp_db.query(tags=["merino_ahedo"])
    assert len(results) == 1


def test_query_by_tags_multiple_required(tmp_db):
    tmp_db.add(_make_record(tags=["a", "b", "c"]))
    tmp_db.add(_make_record(tags=["a", "b"]))
    results = tmp_db.query(tags=["a", "b", "c"])
    assert len(results) == 1


def test_query_combined_filters(tmp_db):
    tmp_db.add(_make_record(case_id="x", contributor="alice", tags=["t1"]))
    tmp_db.add(_make_record(case_id="x", contributor="bob", tags=["t1"]))
    results = tmp_db.query(case_id="x", contributor="alice")
    assert len(results) == 1


def test_query_no_match_returns_empty(tmp_db):
    tmp_db.add(_make_record())
    results = tmp_db.query(case_id="nonexistent")
    assert results == []


def test_delete(tmp_db):
    rec = _make_record()
    tmp_db.add(rec)
    assert tmp_db.count() == 1
    deleted = tmp_db.delete(rec.record_id)
    assert deleted is True
    assert tmp_db.count() == 0


def test_delete_missing_returns_false(tmp_db):
    assert tmp_db.delete("nonexistent-id") is False


def test_delete_specific_record(tmp_db):
    rec_a = _make_record(case_id="a")
    rec_b = _make_record(case_id="b")
    tmp_db.add(rec_a)
    tmp_db.add(rec_b)
    tmp_db.delete(rec_a.record_id)
    assert tmp_db.count() == 1
    assert tmp_db.query(case_id="b")[0].case_id == "b"


def test_export_json(tmp_db, tmp_path):
    tmp_db.add(_make_record(case_id="case1"))
    out = tmp_path / "export.json"
    tmp_db.export_json(out)
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["case_id"] == "case1"


def test_export_csv(tmp_db, tmp_path):
    tmp_db.add(_make_record(metrics={"eta_d": 0.72, "thrust_N": 0.1}))
    out = tmp_path / "export.csv"
    tmp_db.export_csv(out)
    content = out.read_text()
    assert "metrics.eta_d" in content
    assert "metrics.thrust_N" in content


def test_export_csv_empty(tmp_db, tmp_path):
    out = tmp_path / "empty.csv"
    tmp_db.export_csv(out)
    assert out.read_text() == ""


def test_len_dunder(tmp_db):
    assert len(tmp_db) == 0
    tmp_db.add(_make_record())
    assert len(tmp_db) == 1


def test_repr(tmp_db):
    r = repr(tmp_db)
    assert "ValidationDatabase" in r


def test_record_auto_id():
    rec = _make_record()
    assert rec.record_id != ""
    assert len(rec.record_id) == 36  # UUID4 format


def test_record_created_at_utc():
    rec = _make_record()
    assert rec.created_at.tzinfo is not None


def test_record_data_type_validation():
    with pytest.raises((ValueError, Exception)):
        ValidationRecord(
            case_id="x",
            source="src",
            contributor="me",
            data_type="invalid_type",  # type: ignore[arg-type]
        )


def test_db_directory_input(tmp_path):
    """Passing a directory creates valdb.jsonl inside it."""
    db = ValidationDatabase(tmp_path)
    db.add(_make_record())
    assert (tmp_path / "valdb.jsonl").exists()


def test_db_persistence(tmp_path):
    """Records persist between database instances."""
    db1 = ValidationDatabase(tmp_path / "valdb.jsonl")
    rec = _make_record(case_id="persistent")
    db1.add(rec)

    db2 = ValidationDatabase(tmp_path / "valdb.jsonl")
    results = db2.query(case_id="persistent")
    assert len(results) == 1


def test_record_with_notes():
    rec = _make_record(notes="Some important context.")
    assert rec.notes == "Some important context."


def test_record_tags_default_empty():
    rec = ValidationRecord(case_id="x", source="s", contributor="me", data_type="analytical")
    assert rec.tags == []
