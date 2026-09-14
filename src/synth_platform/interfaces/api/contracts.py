"""Shared API contracts for workflow sessions, jobs, results, and files."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from synth_platform.application.dto.intent import (
    DetectedWorkflowInput,
    IntentNextAction,
    IntentWorkflowType,
    WorkflowIntentAlternative,
    WorkflowIntentAttachment,
    WorkflowIntentPrefill,
    WorkflowIntentRequest,
    WorkflowIntentResponse,
)

WorkflowType = Literal["schema_twin", "database_twin", "pdf_twin", "document_twin", "interaction_twin"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class WorkflowSession(BaseModel):
    id: str
    workflow: WorkflowType | str
    workflow_type: WorkflowType | str
    state: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class JobError(BaseModel):
    code: str
    message: str


class GenerationJob(BaseModel):
    id: str
    job_id: str
    session_id: str | None = None
    workflow_type: WorkflowType | str | None = None
    kind: str | None = None
    status: JobStatus
    stage: str
    percent: float
    message: str
    progress: list[str] = Field(default_factory=list)
    error: JobError | dict[str, Any] | None = None
    result_id: str | None = None
    result: dict[str, Any] | None = None
    created_at: str
    updated_at: str


class ArtifactFileMetadata(BaseModel):
    id: str
    artifact_id: str | None = None
    name: str
    filename: str | None = None
    path: str
    media_type: str = "application/octet-stream"
    content_type: str | None = None
    size: int | None = None
    size_bytes: int | None = None
    role: str | None = None
    kind: str | None = None
    downloadable: bool = False
    download_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResultBundle(BaseModel):
    id: str
    result_id: str
    session_id: str
    workflow_type: WorkflowType | str
    status: str = "available"
    preview: dict[str, Any] | None = None
    quality_report: dict[str, Any] | None = None
    summary: dict[str, Any] | None = None
    artifacts: list[ArtifactFileMetadata] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

