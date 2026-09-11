"""Strict commands accepted by the guarded local workspace tool facade."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from synth_platform.application.dto.workspace import (
    ArtifactRegistration,
    ProgressStatus,
    UserPreferences,
    WorkflowKind,
    WorkflowResultView,
)


class _ToolCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ListProjectsCommand(_ToolCommand):
    pass


class ReadPreferencesCommand(_ToolCommand):
    pass


class ListSessionsCommand(_ToolCommand):
    project_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")


class GetSessionCommand(_ToolCommand):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")


class GetProjectCommand(_ToolCommand):
    project_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")


class GetResultCommand(_ToolCommand):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    result_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")


class ReadArtifactCommand(_ToolCommand):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    artifact_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")


class CreateProjectCommand(_ToolCommand):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=1000)


class SaveSessionToProjectCommand(_ToolCommand):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    project_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")


class UpdatePreferencesCommand(_ToolCommand):
    preferences: UserPreferences


class DeleteProjectCommand(_ToolCommand):
    project_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    confirm: bool = False



class CreateSessionCommand(_ToolCommand):
    workflow: WorkflowKind
    title: str = Field(min_length=1, max_length=160)
    project_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")
    source_fingerprint: str | None = Field(default=None, max_length=128)
    configuration_fingerprint: str | None = Field(default=None, max_length=128)
    current_step: str = Field(default="input", min_length=1, max_length=64)


class StartSessionCommand(_ToolCommand):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    run_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_.-]+$")


class SetConfigurationFingerprintCommand(_ToolCommand):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    configuration_fingerprint: str = Field(min_length=1, max_length=128)


class RecordProgressCommand(_ToolCommand):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    macro_step: str = Field(min_length=1, max_length=64)
    stage_id: str = Field(min_length=1, max_length=96)
    stage_label: str = Field(min_length=1, max_length=160)
    status: ProgressStatus
    message: str = Field(default="", max_length=500)
    counts: dict[str, int] = Field(default_factory=dict)


class RegisterArtifactCommand(_ToolCommand):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    registration: ArtifactRegistration


class CompleteSessionCommand(_ToolCommand):
    result: WorkflowResultView


class ListTemplatesCommand(_ToolCommand):
    workflow: WorkflowKind | None = None


class GetTemplateCommand(_ToolCommand):
    template_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")


class ReadRuntimeSettingsCommand(_ToolCommand):
    pass
