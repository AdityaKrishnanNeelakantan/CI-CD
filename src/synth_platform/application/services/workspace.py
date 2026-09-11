"""Application service for sessions, projects, results, templates, and settings.

The service manages product workspace state around specialized workflows. It
never invokes generation stages and is deliberately separate from the chat
capability coordinator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from synth_platform.application.dto.workspace import (
    ArtifactRef,
    ArtifactRegistration,
    ExecutionState,
    ProgressEvent,
    ProgressStatus,
    Project,
    RuntimeSettingsView,
    Template,
    UserPreferences,
    WorkflowKind,
    WorkflowResultView,
    WorkflowSession,
)
from synth_platform.application.ports.workspace_repositories import (
    ArtifactCatalog,
    ProjectRepository,
    ResultRepository,
    SessionRepository,
    SettingsRepository,
    TemplateRepository,
    WorkspaceCommitter,
)
from synth_platform.settings import Settings


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class WorkspaceRepositories:
    sessions: SessionRepository
    results: ResultRepository
    projects: ProjectRepository
    templates: TemplateRepository
    settings: SettingsRepository
    artifacts: ArtifactCatalog
    committer: WorkspaceCommitter


class WorkspaceService:
    """Persist safe workflow metadata and presentation views."""

    def __init__(self, repositories: WorkspaceRepositories, runtime: Settings):
        self._repositories = repositories
        self._runtime = runtime

    def create_session(
        self,
        workflow: WorkflowKind,
        title: str,
        *,
        project_id: str | None = None,
        source_fingerprint: str | None = None,
        configuration_fingerprint: str | None = None,
        current_step: str = "input",
    ) -> WorkflowSession:
        if (
            project_id is not None
            and self._repositories.projects.get(project_id) is None
        ):
            raise KeyError(f"project not found: {project_id}")
        session = WorkflowSession(
            workflow=workflow,
            title=title.strip(),
            project_id=project_id,
            source_fingerprint=source_fingerprint,
            configuration_fingerprint=configuration_fingerprint,
            current_step=current_step,
        )
        self._repositories.sessions.save(session)
        if project_id is not None:
            self._attach_session(project_id, session.session_id)
        return session.model_copy(deep=True)

    def get_session(self, session_id: str) -> WorkflowSession | None:
        return self._repositories.sessions.get(session_id)

    def list_sessions(self, *, project_id: str | None = None) -> list[WorkflowSession]:
        sessions = self._repositories.sessions.list()
        if project_id is not None:
            sessions = [item for item in sessions if item.project_id == project_id]
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def set_step(self, session_id: str, step: str) -> WorkflowSession:
        session = self._require_session(session_id)
        session.current_step = step
        session.updated_at = _now()
        self._repositories.sessions.save(session)
        return session.model_copy(deep=True)

    def set_configuration_fingerprint(
        self, session_id: str, configuration_fingerprint: str
    ) -> WorkflowSession:
        session = self._require_session(session_id)
        session.configuration_fingerprint = configuration_fingerprint
        session.updated_at = _now()
        self._repositories.sessions.save(session)
        return session.model_copy(deep=True)

    def start_session(
        self, session_id: str, *, run_id: str | None = None
    ) -> WorkflowSession:
        session = self._require_session(session_id)
        session.execution_state = ExecutionState.RUNNING
        session.run_id = run_id or session.run_id
        session.started_at = session.started_at or _now()
        session.completed_at = None
        session.updated_at = _now()
        self._repositories.sessions.save(session)
        return session.model_copy(deep=True)

    def record_progress(
        self,
        session_id: str,
        *,
        macro_step: str,
        stage_id: str,
        stage_label: str,
        status: ProgressStatus,
        message: str = "",
        counts: dict[str, int] | None = None,
    ) -> WorkflowSession:
        session = self._require_session(session_id)
        now = _now()
        event = ProgressEvent(
            macro_step=macro_step,
            stage_id=stage_id,
            stage_label=stage_label,
            status=status,
            message=message,
            counts=counts or {},
            started_at=now if status == ProgressStatus.RUNNING else None,
            completed_at=(
                now
                if status
                in {
                    ProgressStatus.SUCCEEDED,
                    ProgressStatus.WARN,
                    ProgressStatus.FAILED,
                    ProgressStatus.BLOCKED,
                }
                else None
            ),
        )
        session.current_step = event.macro_step
        session.progress.append(event)
        if status == ProgressStatus.RUNNING:
            session.execution_state = ExecutionState.RUNNING
            session.started_at = session.started_at or now
        elif status == ProgressStatus.FAILED:
            session.execution_state = ExecutionState.FAILED
            session.completed_at = now
        session.updated_at = now
        self._repositories.sessions.save(session)
        return session.model_copy(deep=True)

    def register_artifact(
        self,
        session_id: str,
        registration: ArtifactRegistration,
    ) -> ArtifactRef:
        self._require_session(session_id)
        return self._repositories.artifacts.register(session_id, registration)

    def get_artifact(self, artifact_id: str) -> ArtifactRef | None:
        return self._repositories.artifacts.get(artifact_id)

    def read_artifact(self, artifact_id: str) -> bytes:
        return self._repositories.artifacts.read_bytes(artifact_id)

    def complete_session(self, result: WorkflowResultView) -> WorkflowSession:
        session = self._require_session(result.session_id)
        if session.workflow != result.workflow:
            raise ValueError("result workflow does not match session workflow")
        for artifact in result.artifacts:
            if artifact.session_id != session.session_id:
                raise ValueError("result contains an artifact from another session")
        now = _now()
        session.result_id = result.result_id
        session.execution_state = result.execution_status
        session.current_step = "results"
        session.completed_at = now
        session.updated_at = now
        project = None
        if session.project_id:
            project = self._require_project(session.project_id)
            if result.result_id not in project.result_ids:
                project.result_ids.append(result.result_id)
            known_artifacts = set(project.artifact_ids)
            project.artifact_ids.extend(
                artifact.artifact_id
                for artifact in result.artifacts
                if artifact.artifact_id not in known_artifacts
            )
            project.updated_at = now
        self._repositories.committer.commit_completion(result, session, project)
        return session.model_copy(deep=True)

    def get_result(self, result_id: str) -> WorkflowResultView | None:
        return self._repositories.results.get(result_id)

    def get_session_result(self, session_id: str) -> WorkflowResultView | None:
        session = self._require_session(session_id)
        return (
            self._repositories.results.get(session.result_id)
            if session.result_id
            else None
        )

    def create_project(self, name: str, description: str = "") -> Project:
        project = Project(name=name.strip(), description=description.strip())
        self._repositories.projects.save(project)
        return project.model_copy(deep=True)

    def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Project:
        project = self._require_project(project_id)
        if name is not None:
            project.name = name.strip()
        if description is not None:
            project.description = description.strip()
        project.updated_at = _now()
        self._repositories.projects.save(project)
        return project.model_copy(deep=True)

    def get_project(self, project_id: str) -> Project | None:
        return self._repositories.projects.get(project_id)

    def list_projects(self) -> list[Project]:
        return sorted(
            self._repositories.projects.list(),
            key=lambda item: item.updated_at,
            reverse=True,
        )

    def delete_project(self, project_id: str) -> None:
        project = self._require_project(project_id)
        for session_id in project.session_ids:
            session = self._repositories.sessions.get(session_id)
            if session is not None:
                session.project_id = None
                session.updated_at = _now()
                self._repositories.sessions.save(session)
        self._repositories.projects.delete(project_id)

    def save_session_to_project(self, session_id: str, project_id: str) -> Project:
        session = self._require_session(session_id)
        self._require_project(project_id)
        if session.project_id and session.project_id != project_id:
            self._detach_session(session.project_id, session_id)
        session.project_id = project_id
        session.updated_at = _now()
        self._repositories.sessions.save(session)
        self._attach_session(project_id, session_id)
        if session.result_id:
            result = self._repositories.results.get(session.result_id)
            if result is not None:
                self._attach_result(project_id, result)
        return self._require_project(project_id).model_copy(deep=True)

    def list_templates(self, *, workflow: WorkflowKind | None = None) -> list[Template]:
        templates = self._repositories.templates.list()
        if workflow is not None:
            templates = [item for item in templates if item.workflow == workflow]
        return sorted(templates, key=lambda item: item.name.lower())

    def get_template(self, template_id: str) -> Template | None:
        return self._repositories.templates.get(template_id)

    def get_preferences(self) -> UserPreferences:
        return self._repositories.settings.get()

    def update_preferences(self, preferences: UserPreferences) -> UserPreferences:
        updated = preferences.model_copy(update={"updated_at": _now()})
        self._repositories.settings.save(updated)
        return updated.model_copy(deep=True)

    def runtime_settings(self) -> RuntimeSettingsView:
        return RuntimeSettingsView(
            approved_output_root=self._runtime.approved_output_root,
            staging_root=self._runtime.staging_root,
            allowed_backends=list(self._runtime.allowed_backends),
            max_rows_per_table=self._runtime.max_rows_per_table,
            minimum_fidelity_score=self._runtime.minimum_fidelity_score,
            guardrail_policy_version=self._runtime.guardrail_policy_version,
            guardrail_max_input_characters=(
                self._runtime.guardrail_max_input_characters
            ),
            guardrail_max_output_characters=(
                self._runtime.guardrail_max_output_characters
            ),
            local_model_host=self._runtime.local_model_host,
        )

    def _require_session(self, session_id: str) -> WorkflowSession:
        session = self._repositories.sessions.get(session_id)
        if session is None:
            raise KeyError(f"session not found: {session_id}")
        return session

    def _require_project(self, project_id: str) -> Project:
        project = self._repositories.projects.get(project_id)
        if project is None:
            raise KeyError(f"project not found: {project_id}")
        return project

    def _attach_session(self, project_id: str, session_id: str) -> None:
        project = self._require_project(project_id)
        if session_id not in project.session_ids:
            project.session_ids.append(session_id)
            project.updated_at = _now()
            self._repositories.projects.save(project)

    def _detach_session(self, project_id: str, session_id: str) -> None:
        project = self._repositories.projects.get(project_id)
        if project is None:
            return
        project.session_ids = [
            item for item in project.session_ids if item != session_id
        ]
        project.updated_at = _now()
        self._repositories.projects.save(project)

    def _attach_result(self, project_id: str, result: WorkflowResultView) -> None:
        project = self._require_project(project_id)
        if result.result_id not in project.result_ids:
            project.result_ids.append(result.result_id)
        known_artifacts = set(project.artifact_ids)
        project.artifact_ids.extend(
            artifact.artifact_id
            for artifact in result.artifacts
            if artifact.artifact_id not in known_artifacts
        )
        project.updated_at = _now()
        self._repositories.projects.save(project)
