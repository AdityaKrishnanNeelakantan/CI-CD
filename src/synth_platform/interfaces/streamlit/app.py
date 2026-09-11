"""Unified Streamlit product shell for the four supported workflows."""

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
        title="Home / Chat",
        icon=":material/home:",
        default=True,
    )
    projects = st.Page(
        str(pages_dir / "projects.py"), title="My Projects", icon=":material/folder:"
    )
    results = st.Page(
        str(pages_dir / "results.py"), title="Results", icon=":material/analytics:"
    )
    templates = st.Page(
        str(pages_dir / "templates.py"),
        title="Templates",
        icon=":material/dashboard_customize:",
    )
    help_page = st.Page(
        str(pages_dir / "help.py"), title="Help & Guides", icon=":material/help:"
    )
    settings = st.Page(
        str(pages_dir / "settings_page.py"),
        title="Settings",
        icon=":material/settings:",
    )
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
    interaction = st.Page(
        str(pages_dir / "interaction_twin.py"),
        title="Interaction Twin",
        icon=":material/record_voice_over:",
    )

    page = st.navigation(
        {
            "Workspace": [chat, projects, results, templates],
            "Specialized workflows": [schema, database, pdf, interaction],
            "Support": [help_page, settings],
        },
        position="sidebar",
    )
    st.caption(
        "Local workspace mode · bound to 127.0.0.1 · authentication is required "
        "before any shared deployment"
    )
    page.run()
