"""Unified Streamlit product shell for the supported workflows."""
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
    home = st.Page(str(pages_dir / "home.py"), title="Home", icon=":material/home:", default=True)
    my_projects = st.Page(str(pages_dir / "my_projects.py"), title="My Twin", icon=":material/folder:")
    schema = st.Page(str(pages_dir / "schema_twin.py"), title="Schema Mode", icon=":material/schema:")
    database = st.Page(str(pages_dir / "database_twin.py"), title="Database Twin", icon=":material/database:")
    pdf = st.Page(str(pages_dir / "pdf_twin.py"), title="PDF Twin", icon=":material/description:")
    interaction = st.Page(
        str(pages_dir / "interaction_twin.py"),
        title="Customer Interaction Twin",
        icon=":material/support_agent:",
    )

    page = st.navigation(
        {
            "Platform": [home, my_projects],
            "Create": [schema, database, pdf, interaction],
        },
        position="sidebar",
    )
    page.run()
