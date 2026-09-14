"""Settings — persisted local product defaults backed by platform.db."""

from __future__ import annotations

import streamlit as st

from synth_platform.infrastructure.persistence.platform_db import get_platform_db
from synth_platform.interfaces.streamlit.project_ui import read_product_settings, write_product_settings

GENERATION_MODES = ["schema_driven", "source_driven"]
PRIVACY_LEVELS = ["standard", "restricted", "strict"]
OUTPUT_FORMATS = ["csv", "parquet", "sqlite", "pdf"]

st.title("Settings")
st.caption("Local product defaults used as starting values by generation workflows when a run has no override.")

db = get_platform_db()
settings = read_product_settings(db)

with st.form("product_settings"):
    generation_mode = st.selectbox(
        "Generation mode",
        options=GENERATION_MODES,
        index=GENERATION_MODES.index(settings["generation_mode"])
        if settings["generation_mode"] in GENERATION_MODES
        else 0,
    )
    default_record_count = st.number_input(
        "Default record count",
        min_value=1,
        max_value=1_000_000,
        value=int(settings["default_record_count"]),
        step=10,
    )
    privacy_level = st.selectbox(
        "Privacy level",
        options=PRIVACY_LEVELS,
        index=PRIVACY_LEVELS.index(settings["privacy_level"])
        if settings["privacy_level"] in PRIVACY_LEVELS
        else 0,
    )
    default_output_format = st.selectbox(
        "Default output format",
        options=OUTPUT_FORMATS,
        index=OUTPUT_FORMATS.index(settings["default_output_format"])
        if settings["default_output_format"] in OUTPUT_FORMATS
        else 0,
    )
    submitted = st.form_submit_button("Save settings", icon=":material/save:")

if submitted:
    write_product_settings(
        db,
        {
            "generation_mode": generation_mode,
            "default_record_count": int(default_record_count),
            "privacy_level": privacy_level,
            "default_output_format": default_output_format,
        },
    )
    st.success("Settings saved.")

st.info("Run-level controls still take precedence when you change them inside a workflow.")
