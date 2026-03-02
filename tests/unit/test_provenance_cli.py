"""Tests for helicon provenance CLI commands."""

from __future__ import annotations

from click.testing import CliRunner

from helicon.cli import main
from helicon.provenance import ProvenanceDB


def _make_db(tmp_path):
    """Create a small populated provenance DB and return (db_path, rec_ids)."""
    db_path = str(tmp_path / "prov.jsonl")
    db = ProvenanceDB(db_path)
    r1 = db.record(
        config={"r": 0.1, "I": 20000},
        source="tier1_analytical",
        metrics={"thrust_efficiency": 0.82, "mirror_ratio": 5.1},
        fidelity_tier=1,
    )
    r2 = db.record(
        config={"r": 0.11, "I": 22000},
        source="tier2_surrogate",
        metrics={"thrust_efficiency": 0.85, "mirror_ratio": 6.0},
        fidelity_tier=2,
        parent_ids=[r1],
    )
    return db_path, (r1, r2)


# ---------------------------------------------------------------------------
# provenance list
# ---------------------------------------------------------------------------


def test_provenance_list_empty(tmp_path):
    runner = CliRunner()
    db_path = str(tmp_path / "empty.jsonl")
    result = runner.invoke(main, ["provenance", "--db", db_path, "list"])
    assert result.exit_code == 0, result.output
    assert "0" in result.output


def test_provenance_list_shows_records(tmp_path):
    runner = CliRunner()
    db_path, _ = _make_db(tmp_path)
    result = runner.invoke(main, ["provenance", "--db", db_path, "list"])
    assert result.exit_code == 0, result.output
    assert "tier1" in result.output or "tier2" in result.output


def test_provenance_list_tail(tmp_path):
    runner = CliRunner()
    db_path, _ = _make_db(tmp_path)
    result = runner.invoke(main, ["provenance", "--db", db_path, "list", "--tail", "1"])
    assert result.exit_code == 0, result.output
    assert "1" in result.output


# ---------------------------------------------------------------------------
# provenance show
# ---------------------------------------------------------------------------


def test_provenance_show_full_id(tmp_path):
    runner = CliRunner()
    db_path, (r1, _) = _make_db(tmp_path)
    result = runner.invoke(main, ["provenance", "--db", db_path, "show", r1])
    assert result.exit_code == 0, result.output
    assert "tier1_analytical" in result.output


def test_provenance_show_prefix(tmp_path):
    runner = CliRunner()
    db_path, (r1, _) = _make_db(tmp_path)
    prefix = r1[:6]
    result = runner.invoke(main, ["provenance", "--db", db_path, "show", prefix])
    assert result.exit_code == 0, result.output
    assert "tier1_analytical" in result.output


def test_provenance_show_metrics(tmp_path):
    runner = CliRunner()
    db_path, (r1, _) = _make_db(tmp_path)
    result = runner.invoke(main, ["provenance", "--db", db_path, "show", r1])
    assert "thrust_efficiency" in result.output
    assert "mirror_ratio" in result.output


def test_provenance_show_not_found(tmp_path):
    runner = CliRunner()
    db_path, _ = _make_db(tmp_path)
    result = runner.invoke(main, ["provenance", "--db", db_path, "show", "aaaaaaaa"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# provenance lineage
# ---------------------------------------------------------------------------


def test_provenance_lineage_two_nodes(tmp_path):
    runner = CliRunner()
    db_path, (_r1, r2) = _make_db(tmp_path)
    result = runner.invoke(main, ["provenance", "--db", db_path, "lineage", r2])
    assert result.exit_code == 0, result.output
    assert "2" in result.output  # two nodes in chain


def test_provenance_lineage_root(tmp_path):
    runner = CliRunner()
    db_path, (r1, _) = _make_db(tmp_path)
    result = runner.invoke(main, ["provenance", "--db", db_path, "lineage", r1])
    assert result.exit_code == 0, result.output
    assert "tier1_analytical" in result.output


def test_provenance_lineage_not_found(tmp_path):
    runner = CliRunner()
    db_path, _ = _make_db(tmp_path)
    result = runner.invoke(main, ["provenance", "--db", db_path, "lineage", "00000000"])
    assert result.exit_code != 0
