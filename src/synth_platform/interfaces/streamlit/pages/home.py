"""Landing page — choose an operating mode of the Synthetic Data Twin Platform."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from synth_platform.interfaces.streamlit.components.common.ux import platform_intro

st.title("Synthetic Data Twin Platform")
platform_intro()

st.markdown("### What would you like to create?")
st.write("One platform. Four modes. Pick the path that matches what you have.")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("#### Schema Mode")
    st.write("Create synthetic data from structure alone.")
    st.caption("Provide schema → review → configure → generate → validate → download")
    st.page_link(str(Path(__file__).resolve().parent / "schema_twin.py"), label="Open Schema Mode", icon=":material/schema:")

with c2:
    st.markdown("#### Database Twin")
    st.write("Learn from an existing database and generate a portable synthetic twin.")
    st.caption("Connect → understand → train → disconnect → generate → validate → export")
    st.page_link(str(Path(__file__).resolve().parent / "database_twin.py"), label="Open Database Twin", icon=":material/database:")

with c3:
    st.markdown("#### PDF Twin")
    st.write("Create synthetic documents while preserving useful document structure.")
    st.caption("Upload → understand → generate → validate → download")
    st.page_link(str(Path(__file__).resolve().parent / "pdf_twin.py"), label="Open PDF Twin", icon=":material/description:")

with c4:
    st.markdown("#### Customer Interaction Twin")
    st.write("Create synthetic customer-service conversations from transcript files.")
    st.caption("Upload → configure → generate → validate → download")
    st.page_link(
        str(Path(__file__).resolve().parent / "interaction_twin.py"),
        label="Open Interaction Twin",
        icon=":material/support_agent:",
    )

st.divider()
with st.expander("Technical details", expanded=False):
    st.caption(
        "Shared stages behind the modes: discovery → understanding → twin creation → "
        "validation → export. Schema Mode uses metadata only; Database Twin trains a "
        "portable artifact; PDF Twin preserves layout structure with synthetic values; "
        "Customer Interaction Twin preserves conversation shape with synthetic turns."
    )
