"""Streamlit bindings for the shared local workspace backend."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import streamlit as st

from synth_platform.application.dto.tool_commands import (
    CreateProjectCommand,
    CreateSessionCommand,
    GetProjectCommand,
    GetSessionCommand,
    ListProjectsCommand,
    ReadArtifactCommand,
    RecordProgressCommand,
    SaveSessionToProjectCommand,
    UpdatePreferencesCommand,
)
from synth_platform.application.dto.workspace import (
    ProgressStatus,
    UserPreferences,
    WorkflowKind,
    WorkflowResultView,
    WorkflowSession,
)
from synth_platform.application.services.tool_gateway import (
    GuardedWorkspaceTools,
    LocalToolPermissionPolicy,
    ToolPrincipal,
)
from synth_platform.bootstrap import (
    build_workspace_service,
    default_settings,
    workspace_root_path,
)


@st.cache_resource
def get_workspace_tools() -> GuardedWorkspaceTools:
    """Expose the guarded facade as the sole shared-workspace UI boundary."""

    return GuardedWorkspaceTools(
        build_workspace_service(),
        ToolPrincipal.local_ui(),
        LocalToolPermissionPolicy(version=default_settings().guardrail_policy_version),
    )


def workflow_run_dir(session_id: str, workflow: WorkflowKind) -> Path:
    path = workspace_root_path() / "runs" / workflow.value / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def fingerprint_configuration(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def ensure_workflow_session(
    *,
    state_key: str,
    workflow: WorkflowKind,
    title: str,
    source_fingerprint: str,
    configuration_fingerprint: str | None = None,
    current_step: str = "input",
) -> WorkflowSession:
    tools = get_workspace_tools()
    existing_id = st.session_state.get(state_key)
    if existing_id:
        existing = tools.get_session(GetSessionCommand(session_id=existing_id))
        if (
            existing is not None
            and existing.workflow == workflow
            and existing.source_fingerprint == source_fingerprint
            and existing.configuration_fingerprint == configuration_fingerprint
        ):
            return existing
    session = tools.create_session(
        CreateSessionCommand(
            workflow=workflow,
            title=title,
            source_fingerprint=source_fingerprint,
            configuration_fingerprint=configuration_fingerprint,
            current_step=current_step,
        )
    )
    st.session_state[state_key] = session.session_id
    return session


def record_progress_once(
    session_id: str,
    *,
    macro_step: str,
    stage_id: str,
    stage_label: str,
    status: ProgressStatus,
    message: str = "",
    counts: dict[str, int] | None = None,
) -> WorkflowSession:
    tools = get_workspace_tools()
    session = tools.get_session(GetSessionCommand(session_id=session_id))
    if session is None:
        raise KeyError(f"session not found: {session_id}")
    latest = next(
        (event for event in reversed(session.progress) if event.stage_id == stage_id),
        None,
    )
    if latest is not None and latest.status == status and latest.message == message:
        return session
    return tools.record_progress(
        RecordProgressCommand(
            session_id=session_id,
            macro_step=macro_step,
            stage_id=stage_id,
            stage_label=stage_label,
            status=status,
            message=message,
            counts=counts or {},
        )
    )


def record_failure(
    session_id: str,
    *,
    macro_step: str,
    stage_id: str,
    stage_label: str,
) -> WorkflowSession:
    return record_progress_once(
        session_id,
        macro_step=macro_step,
        stage_id=stage_id,
        stage_label=stage_label,
        status=ProgressStatus.FAILED,
        message="The specialized workflow stage failed. See its in-session details.",
    )


def render_backend_progress(session_id: str) -> None:
    session = get_workspace_tools().get_session(GetSessionCommand(session_id=session_id))
    if session is None or not session.progress:
        return
    latest_by_stage = {}
    for event in session.progress:
        latest_by_stage[event.stage_id] = event
    with st.container(border=True):
        st.markdown("#### Generation progress")
        for event in latest_by_stage.values():
            icon = {
                ProgressStatus.PENDING: "○",
                ProgressStatus.RUNNING: "◌",
                ProgressStatus.SUCCEEDED: "✓",
                ProgressStatus.WARN: "!",
                ProgressStatus.FAILED: "✕",
                ProgressStatus.BLOCKED: "⊘",
            }[event.status]
            detail = f" — {event.message}" if event.message else ""
            st.write(f"{icon} **{event.stage_label}** · {event.status.value}{detail}")
            if event.counts:
                st.caption(
                    " · ".join(
                        f"{key}: {value:,}" for key, value in event.counts.items()
                    )
                )


def render_result_summary(result: WorkflowResultView) -> None:
    with st.container(border=True):
        st.markdown("### Shared result summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Execution", result.execution_status.value)
        c2.metric("Validation", result.validation_status.value)
        c3.metric(
            "Release",
            result.release_verdict.value if result.release_verdict else "not evaluated",
        )
        if result.blockers:
            st.error("Blockers: " + "; ".join(result.blockers))
        if result.warnings:
            st.warning("Warnings: " + "; ".join(result.warnings))
        if result.metrics:
            with st.expander("Run metrics", expanded=False):
                st.json(result.metrics)
        if result.previews:
            st.markdown("**Previews**")
            st.dataframe(
                [
                    {
                        "name": item.label,
                        "type": item.kind,
                        "rows": item.row_count,
                        "columns": item.column_count,
                        "description": item.description,
                    }
                    for item in result.previews
                ],
                hide_index=True,
                width="stretch",
            )
        if result.artifacts:
            st.markdown("**Downloads**")
            for artifact in result.artifacts:
                if artifact.downloadable:
                    try:
                        content = get_workspace_tools().read_artifact(
                            ReadArtifactCommand(
                                session_id=artifact.session_id,
                                artifact_id=artifact.artifact_id,
                            )
                        )
                    except (KeyError, OSError, ValueError):
                        st.error(f"{artifact.label} · integrity verification failed")
                        continue
                    st.download_button(
                        artifact.label,
                        data=content,
                        file_name=Path(artifact.local_path).name,
                        mime=artifact.media_type,
                        key=f"artifact_download_{artifact.artifact_id}",
                        use_container_width=True,
                    )
                else:
                    st.caption(f"{artifact.label} · unavailable")
        workflow_pages = {
            WorkflowKind.SCHEMA: "schema_twin.py",
            WorkflowKind.DATABASE: "database_twin.py",
            WorkflowKind.DOCUMENT: "pdf_twin.py",
            WorkflowKind.INTERACTION: "interaction_twin.py",
        }
        page_name = workflow_pages[result.workflow]
        st.page_link(
            str(Path(__file__).resolve().parent / "pages" / page_name),
            label="Regenerate in specialized workflow",
            icon=":material/refresh:",
            use_container_width=True,
        )


def render_project_save(session_id: str) -> None:
    tools = get_workspace_tools()
    session = tools.get_session(GetSessionCommand(session_id=session_id))
    if session is None:
        return
    projects = tools.list_projects(ListProjectsCommand())
    with st.container(border=True):
        st.markdown("#### Save to My Projects")
        if session.project_id:
            project = tools.get_project(
                GetProjectCommand(project_id=session.project_id)
            )
            st.success(f"Saved to {project.name if project else session.project_id}.")
            return
        project_names = {project.name: project.project_id for project in projects}
        selected = st.selectbox(
            "Project",
            ["Create a new project", *project_names],
            key=f"project_select_{session_id}",
        )
        new_name = ""
        if selected == "Create a new project":
            new_name = st.text_input("Project name", key=f"project_name_{session_id}")
        if st.button(
            "Save run", key=f"save_project_{session_id}", use_container_width=True
        ):
            if selected == "Create a new project":
                if not new_name.strip():
                    st.warning("Enter a project name.")
                    return
                project_id = tools.create_project(
                    CreateProjectCommand(name=new_name)
                ).project_id
            else:
                project_id = project_names[selected]
            tools.save_session_to_project(
                SaveSessionToProjectCommand(
                    session_id=session_id,
                    project_id=project_id,
                )
            )
            st.rerun()


def save_preferences_from_form(preferences: UserPreferences) -> UserPreferences:
    return get_workspace_tools().update_preferences(
        UpdatePreferencesCommand(preferences=preferences)
    )
