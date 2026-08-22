"""Streamlit launcher.

Run from the repository root with:
    python -m streamlit run streamlit_app.py --server.port 8502
"""
from synth_platform.interfaces.streamlit.app import run

run()
