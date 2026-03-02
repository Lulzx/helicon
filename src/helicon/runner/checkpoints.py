"""WarpX checkpoint management.

Provides utilities for finding, cleaning up, and restarting from
WarpX checkpoint directories.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class CheckpointInfo:
    """Information about a single WarpX checkpoint."""

    path: Path
    step: int
    timestamp: datetime
    size_bytes: int


def _parse_step(name: str) -> int | None:
    """Extract step number from a checkpoint directory name like 'chk00100'."""
    m = re.search(r"chk(\d+)", name)
    return int(m.group(1)) if m else None


def _dir_size(path: Path) -> int:
    """Compute total size of a directory in bytes."""
    total = 0
    if path.is_file():
        return path.stat().st_size
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def find_checkpoints(output_dir: str | Path) -> list[CheckpointInfo]:
    """Search for WarpX checkpoint directories.

    Looks for directories matching ``chk*`` or ``diags/chk*`` patterns.

    Parameters
    ----------
    output_dir : path
        Simulation output directory to search.

    Returns
    -------
    list[CheckpointInfo]
        Checkpoints sorted by step number ascending. Empty list if none found.
    """
    output_dir = Path(output_dir)
    checkpoints: list[CheckpointInfo] = []

    # Search patterns: chk* in output_dir and in output_dir/diags/
    search_dirs = [output_dir, output_dir / "diags"]
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        for entry in search_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("chk"):
                step = _parse_step(entry.name)
                if step is not None:
                    stat = entry.stat()
                    checkpoints.append(
                        CheckpointInfo(
                            path=entry,
                            step=step,
                            timestamp=datetime.fromtimestamp(stat.st_mtime),
                            size_bytes=_dir_size(entry),
                        )
                    )

    checkpoints.sort(key=lambda c: c.step)
    return checkpoints


def find_latest_checkpoint(output_dir: str | Path) -> CheckpointInfo | None:
    """Return the checkpoint with the highest step number.

    Parameters
    ----------
    output_dir : path
        Simulation output directory to search.

    Returns
    -------
    CheckpointInfo or None
        The latest checkpoint, or None if no checkpoints exist.
    """
    chks = find_checkpoints(output_dir)
    return chks[-1] if chks else None


def cleanup_checkpoints(
    output_dir: str | Path,
    *,
    keep_latest: bool = True,
    keep_steps: list[int] | None = None,
) -> None:
    """Delete checkpoint directories.

    Parameters
    ----------
    output_dir : path
        Simulation output directory.
    keep_latest : bool
        If True, keep the most recent checkpoint.
    keep_steps : list[int], optional
        Specific step numbers to keep.
    """
    chks = find_checkpoints(output_dir)
    if not chks:
        return

    keep_set: set[int] = set()
    if keep_steps:
        keep_set.update(keep_steps)
    if keep_latest:
        keep_set.add(chks[-1].step)

    for chk in chks:
        if chk.step not in keep_set:
            shutil.rmtree(chk.path)


def checkpoint_exists(output_dir: str | Path) -> bool:
    """Check if any WarpX checkpoint exists.

    Parameters
    ----------
    output_dir : path
        Simulation output directory.

    Returns
    -------
    bool
    """
    return len(find_checkpoints(output_dir)) > 0


def get_restart_flag(output_dir: str | Path) -> str | None:
    """Get WarpX restart flag if a checkpoint exists.

    Parameters
    ----------
    output_dir : path
        Simulation output directory.

    Returns
    -------
    str or None
        WarpX ``--restart <path>`` flag string, or None if no checkpoint.
    """
    latest = find_latest_checkpoint(output_dir)
    if latest is None:
        return None
    return f"--restart {latest.path}"
