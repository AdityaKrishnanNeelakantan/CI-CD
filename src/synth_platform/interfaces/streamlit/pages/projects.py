"""My Projects: durable local references to workflow runs and artifacts."""

from __future__ import annotations

import streamlit as st

from synth_platform.application.dto.tool_commands import (
    CreateProjectCommand,
    ListProjectsCommand,
    ListSessionsCommand,
)
from synth_platform.interfaces.streamlit.workspace import get_workspace_tools

st.title("My Projects")
st.caption("Organize local workflow sessions and their result/artifact references.")
tools = get_workspace_tools()

with st.form("create_project", border=True):
    st.markdown("### New project")
    name = st.text_input("Project name")
    description = st.text_area("Description", height=80)
    submitted = st.form_submit_button("Create project", type="primary")
    if submitted:
        if not name.strip():
            st.warning("Enter a project name.")
        else:
            tools.create_project(
                CreateProjectCommand(name=name, description=description)
            )
            st.rerun()

projects = tools.list_projects(ListProjectsCommand())
if not projects:
    st.info(
        "No projects yet. Complete a workflow and save its result, or create one above."
    )
else:
    for project in projects:
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.markdown(f"### {project.name}")
            c1.caption(project.description or "No description")
            c2.metric("Runs", len(project.session_ids))
            c3.metric("Artifacts", len(project.artifact_ids))
            sessions = tools.list_sessions(
                ListSessionsCommand(project_id=project.project_id)
            )
            if sessions:
                st.dataframe(
                    [
                        {
                            "workflow": session.workflow.value,
                            "title": session.title,
                            "state": session.execution_state.value,
                            "step": session.current_step,
                            "updated": session.updated_at.isoformat(timespec="seconds"),
                            "session_id": session.session_id,
                        }
                        for session in sessions
                    ],
                    hide_index=True,
                    width="stretch",
                )
