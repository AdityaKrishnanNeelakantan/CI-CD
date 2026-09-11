"""Architecture and capability overview for the Synthetic Data Twin Platform."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from synth_platform.interfaces.streamlit.components.common.ux import platform_intro

st.title("Synthetic Data Twin Platform")
platform_intro()

st.markdown("### One entry point, specialized workflow capabilities")
st.write(
    "Start in Chat to route a request. The coordinator selects a capability but does not "
    "generate data, train models, validate outputs, or package artifacts."
)
st.page_link(
    str(Path(__file__).resolve().parent / "chat.py"),
    label="Open unified Chat",
    icon=":material/chat:",
)

st.markdown("### Capabilities")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("#### Schema Twin")
    st.write("Create synthetic data from structure alone.")
    st.caption("Schema → configure → generate → validate → download")

with c2:
    st.markdown("#### Database Twin")
    st.write("Learn from an existing database and create a portable synthetic twin.")
    st.caption("Discover → train → disconnect → generate → validate → export")

with c3:
    st.markdown("#### Document Twin")
    st.write("Create synthetic PDFs while preserving useful document structure.")
    st.caption("Extract → bind → generate → render → validate")

with c4:
    st.markdown("#### Interaction Twin")
    st.write(
        "Create one structured, privacy-validated SSOT from a customer interaction."
    )
    st.caption("Sanitize → structure → validate privacy/replay → package")

st.divider()
with st.expander("Architecture boundary", expanded=False):
    st.caption(
        "Chat coordination, workflow orchestration, and model-backed generation are separate "
        "responsibilities. Existing workflow stages remain authoritative. MCP, a live-agent "
        "plane, and unified evaluation are later phases, not current runtime components."
    )
