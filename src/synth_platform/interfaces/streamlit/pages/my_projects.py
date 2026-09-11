"""My Projects — local project and run history backed by platform.db."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from synth_platform.infrastructure.persistence.platform_db import get_platform_db
from synth_platform.interfaces.streamlit.project_ui import (
    WORKFLOW_LABELS,
    load_project_summaries,
    project_run_rows,
    project_table_rows,
)

st.title("My Projects")
st.caption("Saved local project history from generation and validation runs on this machine.")

db = get_platform_db()

tabs = st.tabs(["All Projects", "Schema", "Database", "Document", "Interaction"])
tab_specs = [
    (tabs[0], None),
    (tabs[1], "schema"),
    (tabs[2], "database"),
    (tabs[3], "pdf"),
    (tabs[4], "interaction"),
]

for tab, workflow_type in tab_specs:
    with tab:
        summaries = load_project_summaries(db, workflow_type=workflow_type)
        if not summaries:
            label = "projects" if workflow_type is None else f"{WORKFLOW_LABELS[workflow_type].lower()} projects"
            st.info(f"No {label} have been recorded yet.")
            continue

        st.dataframe(
            pd.DataFrame(project_table_rows(summaries)),
            hide_index=True,
            width="stretch",
        )

        st.markdown("### Project Details")
        for summary in summaries:
            with st.expander(f"{summary.name} · {summary.workflow_label}", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Status", summary.status.replace("_", " ").title())
                c2.metric("Validation", summary.validation)
                c3.metric("Transfer", summary.transfer)
                c4.metric("Records", summary.records)

                with st.form(f"rename_{summary.id}"):
                    new_name = st.text_input("Project name", value=summary.name)
                    submitted = st.form_submit_button("Rename project", icon=":material/edit:")
                    if submitted:
                        clean_name = new_name.strip()
                        if not clean_name:
                            st.error("Project name cannot be blank.")
                        elif clean_name == summary.name:
                            st.info("Project name is unchanged.")
                        else:
                            db.update_project(summary.id, name=clean_name)
                            st.success("Project renamed.")
                            st.rerun()

                rows = project_run_rows(db, summary.id)
                if rows:
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                else:
                    st.caption("No runs recorded for this project yet.")
