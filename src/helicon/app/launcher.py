"""Launcher helper for the Helicon design app."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch_app(port: int = 8501, browser: bool = True) -> None:
    """Launch the Streamlit design app.

    Parameters
    ----------
    port : int
        Port to bind the Streamlit server to.
    browser : bool
        Whether to automatically open a browser tab.
    """
    app_file = Path(__file__).parent / "design_app.py"
    if not app_file.exists():
        raise FileNotFoundError(f"App file not found: {app_file}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_file),
        f"--server.port={port}",
        f"--server.headless={'false' if browser else 'true'}",
        "--theme.base=dark",
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise ImportError(
            "Streamlit is not installed. Install with: pip install streamlit"
        ) from None
