"""Tests for magnozzlex.runner.checkpoints module."""

from __future__ import annotations

import tempfile
from pathlib import Path

from magnozzlex.runner.checkpoints import (
    checkpoint_exists,
    cleanup_checkpoints,
    find_checkpoints,
    find_latest_checkpoint,
    get_restart_flag,
)


def test_find_checkpoints_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = find_checkpoints(tmpdir)
        assert result == []


def test_checkpoint_exists_false_when_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert checkpoint_exists(tmpdir) is False


def test_find_latest_checkpoint_none_when_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert find_latest_checkpoint(tmpdir) is None


def test_find_checkpoints_finds_created_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        chk_dir = Path(tmpdir) / "chk00100"
        chk_dir.mkdir()
        # Create a small file so the checkpoint has content
        (chk_dir / "Header").write_text("checkpoint data")

        result = find_checkpoints(tmpdir)
        assert len(result) == 1
        assert result[0].step == 100
        assert result[0].path == chk_dir


def test_find_latest_checkpoint_returns_highest_step():
    with tempfile.TemporaryDirectory() as tmpdir:
        for step in [100, 300, 200]:
            chk = Path(tmpdir) / f"chk{step:05d}"
            chk.mkdir()
            (chk / "Header").write_text("data")

        latest = find_latest_checkpoint(tmpdir)
        assert latest is not None
        assert latest.step == 300


def test_cleanup_checkpoints_keeps_latest():
    with tempfile.TemporaryDirectory() as tmpdir:
        for step in [100, 200, 300]:
            chk = Path(tmpdir) / f"chk{step:05d}"
            chk.mkdir()
            (chk / "Header").write_text("data")

        cleanup_checkpoints(tmpdir, keep_latest=True)

        remaining = find_checkpoints(tmpdir)
        assert len(remaining) == 1
        assert remaining[0].step == 300


def test_get_restart_flag_none_when_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert get_restart_flag(tmpdir) is None


def test_get_restart_flag_returns_path_when_checkpoint_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        chk = Path(tmpdir) / "chk00500"
        chk.mkdir()
        (chk / "Header").write_text("data")

        flag = get_restart_flag(tmpdir)
        assert flag is not None
        assert "--restart" in flag
        assert "chk00500" in flag
