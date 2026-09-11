"""Shared result viewer for completed workflow sessions."""

from __future__ import annotations

import streamlit as st

from synth_platform.application.dto.tool_commands import (
    GetResultCommand,
    ListSessionsCommand,
)
from synth_platform.interfaces.streamlit.workspace import (
    get_workspace_tools,
    render_project_save,
    render_result_summary,
)

st.title("Results")
st.caption(
    "Review execution, validation, release status, previews, lineage, and downloads separately."
)
tools = get_workspace_tools()
sessions = [
    session
    for session in tools.list_sessions(ListSessionsCommand())
    if session.result_id
]
if not sessions:
    st.info(
        "No persisted results yet. Complete one of the specialized workflows first."
    )
    st.stop()

labels = {
    f"{session.title} · {session.workflow.value} · {session.updated_at.isoformat(timespec='seconds')}": session
    for session in sessions
}
selected_label = st.selectbox("Result", list(labels))
session = labels[selected_label]
result = (
    tools.get_result(
        GetResultCommand(
            session_id=session.session_id,
            result_id=session.result_id,
        )
    )
    if session.result_id
    else None
)
if result is None:
    st.error("The result metadata is unavailable.")
    st.stop()
render_result_summary(result)
if result.lineage:
    with st.expander("Lineage", expanded=False):
        st.json(result.lineage.model_dump(mode="json"))
render_project_save(session.session_id)
