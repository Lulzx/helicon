"""Interactive local-first design app for Helicon (v2.0).

Launches a Streamlit-based web app for real-time nozzle design exploration
using the MLX surrogate on Metal GPU.

Usage::

    helicon app          # launch the design explorer
    helicon app --port 8502
"""

from __future__ import annotations

from helicon.app.launcher import launch_app

__all__ = ["launch_app"]
