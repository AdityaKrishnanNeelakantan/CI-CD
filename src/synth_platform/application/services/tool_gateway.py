"""Schema-validated, permission-checked facade over local workspace operations.

This is the tool-guardrail boundary. It does not route prompts, invoke models,
or execute specialized workflows. A future MCP adapter may deserialize into
these commands, but it must not bypass this service.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from synth_platform.application.dto.tool_commands import (
    CompleteSessionCommand,
    CreateProjectCommand,
    CreateSessionCommand,
    DeleteProjectCommand,
    GetProjectCommand,
    GetResultCommand,
    GetSessionCommand,
    GetTemplateCommand,
    ListProjectsCommand,
    ListSessionsCommand,
    ListTemplatesCommand,
    ReadArtifactCommand,
    ReadPreferencesCommand,
    ReadRuntimeSettingsCommand,
    RecordProgressCommand,
    RegisterArtifactCommand,
    SaveSessionToProjectCommand,
    SetConfigurationFingerprintCommand,
    StartSessionCommand,
    UpdatePreferencesCommand,
)
from synth_platform.application.dto.workspace import (
    ArtifactRef,
    Project,
    RuntimeSettingsView,
    Template,
    UserPreferences,
    WorkflowResultView,
    WorkflowSession,
)
from synth_platform.application.services.workspace import WorkspaceService
from synth_platform.domain.guardrails.models import (
    GuardrailAction,
    GuardrailFinding,
    GuardrailReport,
    GuardrailStage,
)


class ToolOperation(StrEnum):
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"


class ToolPrincipal(BaseModel):
    """Local operation scope; this is not a substitute for authentication/RBAC."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    principal_id: str
    local_request: bool = True
    workspace_read: bool = False
    workspace_write: bool = False
    can_delete: bool = False
    session_ids: tuple[str, ...] = ()
    project_ids: tuple[str, ...] = ()

    @classmethod
    def local_ui(cls) -> ToolPrincipal:
        return cls(
            principal_id="local-streamlit-ui",
            local_request=True,
            workspace_read=True,
            workspace_write=True,
            can_delete=False,
        )


class ToolGuardrailViolation(PermissionError):
    """An operation was denied without exposing resource details."""


class LocalToolPermissionPolicy:
    def __init__(self, *, version: str = "1.0") -> None:
        self.version = version

    def evaluate(
        self,
        principal: ToolPrincipal,
        operation: ToolOperation,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        explicit_confirmation: bool = False,
    ) -> GuardrailReport:
        allowed = principal.local_request
        if operation == ToolOperation.READ:
            allowed = allowed and (
                principal.workspace_read
                or (session_id is not None and session_id in principal.session_ids)
                or (project_id is not None and project_id in principal.project_ids)
                or (
                    session_id is None
                    and project_id is None
                    and bool(principal.session_ids or principal.project_ids)
                )
            )
        elif operation in {ToolOperation.WRITE, ToolOperation.UPDATE}:
            allowed = allowed and principal.workspace_write
            if session_id is not None and not principal.workspace_read:
                allowed = allowed and session_id in principal.session_ids
            if project_id is not None and not principal.workspace_read:
                allowed = allowed and project_id in principal.project_ids
        elif operation == ToolOperation.DELETE:
            allowed = (
                allowed
                and principal.workspace_write
                and principal.can_delete
                and explicit_confirmation
                and (
                    principal.workspace_read
                    or (project_id is not None and project_id in principal.project_ids)
                )
            )

        findings: tuple[GuardrailFinding, ...] = ()
        if not allowed:
            findings = (
                GuardrailFinding(
                    code="tool_permission_denied",
                    category="permission_enforcement",
                    action=GuardrailAction.BLOCK,
                    message="The local principal is not permitted to perform this operation.",
                ),
            )
        return GuardrailReport(
            policy_name="local_workspace_tools",
            policy_version=self.version,
            stage=GuardrailStage.TOOL,
            action=GuardrailAction.ALLOW if allowed else GuardrailAction.BLOCK,
            findings=findings,
            checks=("schema_validation", "permission_enforcement"),
        )


class GuardedWorkspaceTools:
    """Explicit workspace read/write/update/delete operations behind policy."""

    def __init__(
        self,
        workspace: WorkspaceService,
        principal: ToolPrincipal,
        policy: LocalToolPermissionPolicy,
    ) -> None:
        self._workspace = workspace
        self._principal = principal
        self._policy = policy
        self.last_report: GuardrailReport | None = None

    def _authorize(
        self,
        operation: ToolOperation,
        *,
        session_id: str | None = None,
        project_id: str | None = None,
        explicit_confirmation: bool = False,
    ) -> None:
        report = self._policy.evaluate(
            self._principal,
            operation,
            session_id=session_id,
            project_id=project_id,
            explicit_confirmation=explicit_confirmation,
        )
        self.last_report = report
        if report.blocked:
            raise ToolGuardrailViolation("Workspace tool operation was denied.")

    def create_session(
        self, command: CreateSessionCommand | dict
    ) -> WorkflowSession:
        parsed = CreateSessionCommand.model_validate(command)
        self._authorize(ToolOperation.WRITE, project_id=parsed.project_id)
        return self._workspace.create_session(
            parsed.workflow,
            parsed.title,
            project_id=parsed.project_id,
            source_fingerprint=parsed.source_fingerprint,
            configuration_fingerprint=parsed.configuration_fingerprint,
            current_step=parsed.current_step,
        )

    def start_session(self, command: StartSessionCommand | dict) -> WorkflowSession:
        parsed = StartSessionCommand.model_validate(command)
        self._authorize(ToolOperation.UPDATE, session_id=parsed.session_id)
        return self._workspace.start_session(parsed.session_id, run_id=parsed.run_id)

    def set_configuration_fingerprint(
        self, command: SetConfigurationFingerprintCommand | dict
    ) -> WorkflowSession:
        parsed = SetConfigurationFingerprintCommand.model_validate(command)
        self._authorize(ToolOperation.UPDATE, session_id=parsed.session_id)
        return self._workspace.set_configuration_fingerprint(
            parsed.session_id, parsed.configuration_fingerprint
        )

    def record_progress(
        self, command: RecordProgressCommand | dict
    ) -> WorkflowSession:
        parsed = RecordProgressCommand.model_validate(command)
        self._authorize(ToolOperation.UPDATE, session_id=parsed.session_id)
        return self._workspace.record_progress(
            parsed.session_id,
            macro_step=parsed.macro_step,
            stage_id=parsed.stage_id,
            stage_label=parsed.stage_label,
            status=parsed.status,
            message=parsed.message,
            counts=parsed.counts,
        )

    def register_artifact(
        self, command: RegisterArtifactCommand | dict
    ) -> ArtifactRef:
        parsed = RegisterArtifactCommand.model_validate(command)
        self._authorize(ToolOperation.WRITE, session_id=parsed.session_id)
        return self._workspace.register_artifact(
            parsed.session_id, parsed.registration
        )

    def complete_session(
        self, command: CompleteSessionCommand | dict
    ) -> WorkflowSession:
        parsed = CompleteSessionCommand.model_validate(command)
        self._authorize(ToolOperation.UPDATE, session_id=parsed.result.session_id)
        return self._workspace.complete_session(parsed.result)

    def list_projects(
        self, command: ListProjectsCommand | dict
    ) -> list[Project]:
        ListProjectsCommand.model_validate(command)
        self._authorize(ToolOperation.READ)
        projects = self._workspace.list_projects()
        if self._principal.workspace_read:
            return projects
        allowed = set(self._principal.project_ids)
        return [project for project in projects if project.project_id in allowed]

    def list_sessions(
        self, command: ListSessionsCommand | dict
    ) -> list[WorkflowSession]:
        parsed = ListSessionsCommand.model_validate(command)
        self._authorize(ToolOperation.READ, project_id=parsed.project_id)
        sessions = self._workspace.list_sessions(project_id=parsed.project_id)
        if self._principal.workspace_read:
            return sessions
        allowed = set(self._principal.session_ids)
        return [session for session in sessions if session.session_id in allowed]

    def get_session(
        self, command: GetSessionCommand | dict
    ) -> WorkflowSession | None:
        parsed = GetSessionCommand.model_validate(command)
        self._authorize(ToolOperation.READ, session_id=parsed.session_id)
        return self._workspace.get_session(parsed.session_id)

    def get_project(self, command: GetProjectCommand | dict) -> Project | None:
        parsed = GetProjectCommand.model_validate(command)
        self._authorize(ToolOperation.READ, project_id=parsed.project_id)
        return self._workspace.get_project(parsed.project_id)

    def get_result(
        self, command: GetResultCommand | dict
    ) -> WorkflowResultView | None:
        parsed = GetResultCommand.model_validate(command)
        self._authorize(ToolOperation.READ, session_id=parsed.session_id)
        result = self._workspace.get_result(parsed.result_id)
        if result is not None and result.session_id != parsed.session_id:
            raise ToolGuardrailViolation("Result does not belong to the permitted session.")
        return result

    def read_artifact(self, command: ReadArtifactCommand | dict) -> bytes:
        parsed = ReadArtifactCommand.model_validate(command)
        self._authorize(ToolOperation.READ, session_id=parsed.session_id)
        artifact = self._workspace.get_artifact(parsed.artifact_id)
        if artifact is None:
            raise KeyError("artifact not found")
        if artifact.session_id != parsed.session_id:
            raise ToolGuardrailViolation("Artifact does not belong to the permitted session.")
        return self._workspace.read_artifact(parsed.artifact_id)

    def create_project(self, command: CreateProjectCommand | dict) -> Project:
        parsed = CreateProjectCommand.model_validate(command)
        self._authorize(ToolOperation.WRITE)
        return self._workspace.create_project(parsed.name, parsed.description)

    def save_session_to_project(
        self, command: SaveSessionToProjectCommand | dict
    ) -> Project:
        parsed = SaveSessionToProjectCommand.model_validate(command)
        self._authorize(
            ToolOperation.UPDATE,
            session_id=parsed.session_id,
            project_id=parsed.project_id,
        )
        return self._workspace.save_session_to_project(
            parsed.session_id, parsed.project_id
        )

    def list_templates(
        self, command: ListTemplatesCommand | dict
    ) -> list[Template]:
        parsed = ListTemplatesCommand.model_validate(command)
        self._authorize(ToolOperation.READ)
        return self._workspace.list_templates(workflow=parsed.workflow)

    def get_template(self, command: GetTemplateCommand | dict) -> Template | None:
        parsed = GetTemplateCommand.model_validate(command)
        self._authorize(ToolOperation.READ)
        return self._workspace.get_template(parsed.template_id)

    def runtime_settings(
        self, command: ReadRuntimeSettingsCommand | dict
    ) -> RuntimeSettingsView:
        ReadRuntimeSettingsCommand.model_validate(command)
        self._authorize(ToolOperation.READ)
        return self._workspace.runtime_settings()

    def get_preferences(
        self, command: ReadPreferencesCommand | dict
    ) -> UserPreferences:
        ReadPreferencesCommand.model_validate(command)
        self._authorize(ToolOperation.READ)
        return self._workspace.get_preferences()

    def update_preferences(
        self, command: UpdatePreferencesCommand | dict
    ) -> UserPreferences:
        parsed = UpdatePreferencesCommand.model_validate(command)
        self._authorize(ToolOperation.UPDATE)
        return self._workspace.update_preferences(parsed.preferences)

    def delete_project(self, command: DeleteProjectCommand | dict) -> None:
        parsed = DeleteProjectCommand.model_validate(command)
        self._authorize(
            ToolOperation.DELETE,
            project_id=parsed.project_id,
            explicit_confirmation=parsed.confirm,
        )
        self._workspace.delete_project(parsed.project_id)
