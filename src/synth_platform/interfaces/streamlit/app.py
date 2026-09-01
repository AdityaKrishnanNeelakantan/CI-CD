"""Unified Streamlit product shell for multiple SSOT input workflows."""
from __future__ import annotations

from pathlib import Path
import runpy

import streamlit as st

from synth_platform.interfaces.streamlit.ui_config import load_ui_config


def _run_legacy_navigation(app_config: dict, pages_dir: Path) -> None:
    entries = [
        {"section": section, **entry}
        for section, section_entries in app_config["nav"].items()
        for entry in section_entries
    ]
    default_index = next((idx for idx, entry in enumerate(entries) if entry.get("default")), 0)
    labels = [f"{entry['section']} / {entry['title']}" for entry in entries]
    selected_label = st.sidebar.selectbox("Page", labels, index=default_index)
    selected = entries[labels.index(selected_label)]
    runpy.run_path(str(pages_dir / selected["file"]), run_name="__main__")


def run() -> None:
    ui_config = load_ui_config()
    app_config = ui_config["app"]
    st.set_page_config(
        page_title=app_config["page_title"],
        page_icon=app_config["page_icon"],
        layout=app_config["layout"],
    )

    pages_dir = Path(__file__).resolve().parent / "pages"
    if not hasattr(st, "Page") or not hasattr(st, "navigation"):
        _run_legacy_navigation(app_config, pages_dir)
        return

    nav = {}
    for section, entries in app_config["nav"].items():
        nav[section] = [
            st.Page(
                str(pages_dir / entry["file"]),
                title=entry["title"],
                icon=entry.get("icon"),
                default=bool(entry.get("default", False)),
            )
            for entry in entries
        ]

    page = st.navigation(
        nav,
        position="top",
    )
    page.run()
