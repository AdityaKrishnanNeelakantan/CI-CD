"""Streamlit launcher.

Run from the repository root with:
    python -m streamlit run streamlit_app.py --server.port 8502
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synth_platform.interfaces.streamlit.app import run

run()
