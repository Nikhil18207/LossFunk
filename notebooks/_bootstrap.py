"""Notebook cwd bootstrap. Importable from any notebook in this folder.

Usage at the top of a notebook (or script run from notebooks/):

    from _bootstrap import PROJECT_ROOT     # noqa
    # cwd is now PROJECT_ROOT; data/, models/, results/ all resolve.

Notebooks 01–08 originally assumed the user launched Jupyter from the project
root. After the layout reorganization (notebooks/ subfolder), launching Jupyter
from notebooks/ would break all the './results/...' paths. This module makes
that case work too.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Idempotent cwd shift: only chdir if we're not already at the root.
_marker = PROJECT_ROOT / "PAPER.md"
if _marker.exists() and Path.cwd().resolve() != PROJECT_ROOT:
    os.chdir(PROJECT_ROOT)

# Make `lib.common` importable from notebooks too.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"[bootstrap] cwd = {Path.cwd()}")
