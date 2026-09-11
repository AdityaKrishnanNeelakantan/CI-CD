"""Persistence ports for shared workflow workspace metadata."""

from __future__ import annotations

from typing import Protocol

from synth_platform.application.dto.workspace import (
    ArtifactRef,
    ArtifactRegistration,
    Project,
    Template,
    UserPreferences,
    WorkflowResultView,
    WorkflowSession,
)


class SessionRepository(Protocol):
    def save(self, session: WorkflowSession) -> None: ...

    def get(self, session_id: str) -> WorkflowSession | None: ...

    def list(self) -> list[WorkflowSession]: ...

    def delete(self, session_id: str) -> None: ...


class ResultRepository(Protocol):
    def save(self, result: WorkflowResultView) -> None: ...

    def get(self, result_id: str) -> WorkflowResultView | None: ...

    def list(self) -> list[WorkflowResultView]: ...

    def delete(self, result_id: str) -> None: ...


class ProjectRepository(Protocol):
    def save(self, project: Project) -> None: ...

    def get(self, project_id: str) -> Project | None: ...

    def list(self) -> list[Project]: ...

    def delete(self, project_id: str) -> None: ...


class TemplateRepository(Protocol):
    def get(self, template_id: str) -> Template | None: ...

    def list(self) -> list[Template]: ...


class SettingsRepository(Protocol):
    def get(self) -> UserPreferences: ...

    def save(self, preferences: UserPreferences) -> None: ...


class WorkspaceCommitter(Protocol):
    def commit_completion(
        self,
        result: WorkflowResultView,
        session: WorkflowSession,
        project: Project | None,
    ) -> None: ...


class ArtifactCatalog(Protocol):
    def register(
        self, session_id: str, registration: ArtifactRegistration
    ) -> ArtifactRef: ...

    def get(self, artifact_id: str) -> ArtifactRef | None: ...

    def read_bytes(self, artifact_id: str) -> bytes: ...

    def list(self, *, session_id: str | None = None) -> list[ArtifactRef]: ...

    def delete(self, artifact_id: str) -> None: ...
