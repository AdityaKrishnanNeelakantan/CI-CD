"""Product help and workflow guides."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.title("Help & Guides")
st.caption(
    "Choose the workflow that matches the source you have and the artifact you need."
)
pages = Path(__file__).resolve().parent

_guides = [
    (
        "Schema Twin",
        "You have SQL DDL, JSON, or YAML structure and do not need source records.",
        "schema_twin.py",
    ),
    (
        "Database Twin",
        "You have an approved SQLite source and need a portable trained twin plus relational output.",
        "database_twin.py",
    ),
    (
        "Document Twin",
        "You have a PDF and need a layout-preserving synthetic document.",
        "pdf_twin.py",
    ),
    (
        "Customer Interaction Twin",
        "You have TXT/LOG dialogue and need a sanitized structured SSOT package.",
        "interaction_twin.py",
    ),
]
for title, description, filename in _guides:
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.write(description)
        st.page_link(
            str(pages / filename),
            label=f"Open {title}",
            icon=":material/arrow_forward:",
        )

with st.expander("Privacy and architecture boundaries", expanded=True):
    st.write(
        "The Chat coordinator only routes. Each specialized workflow owns its stages, validation, "
        "and artifacts. Workspace persistence stores safe fingerprints and references, not database "
        "credentials or raw transcript text. Chat input is masked/checked before it is "
        "stored for routing. Optional model calls pass through input and output guardrails, "
        "use loopback-only Ollama, and fall back deterministically when blocked. Shared "
        "workspace reads and writes use strict command schemas plus a local permission "
        "policy; deletion is denied by default."
    )

with st.expander("Guardrail limitations", expanded=False):
    st.write(
        "The deterministic injection and content checks are inspectable safeguards, not "
        "a comprehensive classifier. Tool permissions protect local operation scope but "
        "are not multi-user RBAC because authentication is not implemented."
    )
