"""Built-in reusable workflow templates."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from synth_platform.application.dto.tool_commands import ListTemplatesCommand
from synth_platform.application.dto.workspace import WorkflowKind
from synth_platform.interfaces.streamlit.workspace import get_workspace_tools

st.title("Templates")
st.caption(
    "Start from a versioned built-in asset. Only real Schema templates are available today."
)
templates = get_workspace_tools().list_templates(
    ListTemplatesCommand(workflow=WorkflowKind.SCHEMA)
)
schema_page = str(Path(__file__).resolve().parent / "schema_twin.py")

for template in templates:
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        c1.markdown(f"### {template.name}")
        c1.write(template.description)
        c1.caption(f"Schema Twin · v{template.version} · {' · '.join(template.tags)}")
        table_count = len(template.definition.get("tables") or [])
        c2.metric("Tables", table_count)
        if st.button(
            "Use template", key=f"use_{template.template_id}", use_container_width=True
        ):
            st.session_state.schema_template_definition = template.definition
            st.session_state.schema_template_id = template.template_id
            st.session_state.schema_file_id = "template:" + template.template_id
            st.session_state.schema_config = None
            st.session_state.schema_summary = None
            st.session_state.schema_result = None
            st.session_state.schema_zip_bytes = None
            st.switch_page(schema_page)
        with st.expander("Template definition", expanded=False):
            st.code(json.dumps(template.definition, indent=2), language="json")
