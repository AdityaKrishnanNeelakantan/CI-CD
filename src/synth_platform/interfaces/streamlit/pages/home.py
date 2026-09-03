"""Landing page - choose an operating mode of the Synthetic Data Twin Platform."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from synth_platform.interfaces.streamlit.components.common.ux import platform_intro
from synth_platform.interfaces.streamlit.ui_config import ui_value

HOME_CONFIG = ui_value("home", default={})

st.title(HOME_CONFIG.get("title", "Synthetic Data Twin Platform"))
platform_intro()


def _safe_page_link(filename: str, *, label: str, icon: str) -> None:
    page_path = str(Path(__file__).resolve().parent / filename)
    page_icon = None if str(icon).startswith(":material/") else icon
    try:
        st.page_link(page_path, label=label, icon=page_icon)
    except Exception:
        st.caption(label)


st.markdown(f"### {HOME_CONFIG.get('chooser_heading', 'What would you like to create?')}")
st.write(HOME_CONFIG.get("chooser_copy", "One platform. Multiple SSOT input paths. Pick the source type you have."))

workflows = HOME_CONFIG.get("workflows", [])
for column, workflow in zip(st.columns(max(1, len(workflows))), workflows):
    with column:
        st.markdown(f"#### {workflow['title']}")
        st.write(workflow["body"])
        st.caption(workflow["caption"])
        _safe_page_link(workflow["page"], label=workflow["link_label"], icon=workflow["icon"])

