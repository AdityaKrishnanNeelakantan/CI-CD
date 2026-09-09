"""Unified Streamlit product shell for the three supported workflows."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def run() -> None:
    st.set_page_config(
        page_title="Synthetic Data Twin Platform",
        page_icon=":material/data_object:",
        layout="wide",
    )

    pages_dir = Path(__file__).resolve().parent / "pages"
    chat = st.Page(
        str(pages_dir / "chat.py"),
        title="Chat",
        icon=":material/chat:",
        default=True,
    )
    home = st.Page(str(pages_dir / "home.py"), title="About", icon=":material/info:")
    schema = st.Page(
        str(pages_dir / "schema_twin.py"),
        title="Schema Twin",
        icon=":material/schema:",
    )
    database = st.Page(
        str(pages_dir / "database_twin.py"),
        title="Database Twin",
        icon=":material/database:",
    )
    pdf = st.Page(
        str(pages_dir / "pdf_twin.py"),
        title="Document Twin",
        icon=":material/description:",
    )

    page = st.navigation(
        {
            "Platform": [chat, home],
            "Specialized workflows": [schema, database, pdf],
        },
        position="top",
    )
    page.run()
