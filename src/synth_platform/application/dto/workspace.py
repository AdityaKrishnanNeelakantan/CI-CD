"""Shared application contracts for the canonical product workspace.

These models describe presentation and persistence state only. Specialized
workflow facades continue to own generation, validation, and packaging.
Source payloads and connection credentials are intentionally absent.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from synth_platform.domain.validation.models import Status


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class WorkflowKind(StrEnum):
    SCHEMA = "schema"
    DATABASE = "database"
    DOCUMENT = "document"
    INTERACTION = "interaction"


class ExecutionState(StrEnum):
    DRAFT = "draft"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class ProgressStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    WARN = "warn"
    FAILED = "failed"
    BLOCKED = "blocked"


class ProgressEvent(_Contract):
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    macro_step: str = Field(min_length=1, max_length=64)
    stage_id: str = Field(min_length=1, max_length=96)
    stage_label: str = Field(min_length=1, max_length=160)
    status: ProgressStatus
    message: str = Field(default="", max_length=500)
    recorded_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("macro_step", "stage_id")
    @classmethod
    def _validate_key(cls, value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_")
        if not normalized or any(
            not (char.isalnum() or char in "_-") for char in normalized
        ):
            raise ValueError("step and stage identifiers must be path-safe")
        return normalized

    @field_validator("counts")
    @classmethod
    def _non_negative_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if any(count < 0 for count in value.values()):
            raise ValueError("progress counts cannot be negative")
        return value


class WorkflowSession(_Contract):
    session_id: str = Field(default_factory=lambda: new_id("session"))
    workflow: WorkflowKind
    title: str = Field(min_length=1, max_length=160)
    project_id: str | None = None
    source_fingerprint: str | None = Field(default=None, max_length=128)
    configuration_fingerprint: str | None = Field(default=None, max_length=128)
    current_step: str = Field(default="input", min_length=1, max_length=64)
    execution_state: ExecutionState = ExecutionState.DRAFT
    progress: list[ProgressEvent] = Field(default_factory=list)
    result_id: str | None = None
    run_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @field_validator("session_id", "project_id", "result_id", "run_id")
    @classmethod
    def _validate_optional_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or any(not (char.isalnum() or char in "_.-") for char in value):
            raise ValueError("identifiers must be path-safe")
        return value

    @field_validator("source_fingerprint", "configuration_fingerprint")
    @classmethod
    def _validate_fingerprint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower().removeprefix("sha256:")
        if len(normalized) != 64 or any(
            char not in "0123456789abcdef" for char in normalized
        ):
            raise ValueError("fingerprints must be SHA-256 hex digests")
        return normalized

    @field_validator("current_step")
    @classmethod
    def _validate_step(cls, value: str) -> str:
        return ProgressEvent._validate_key(value)


class ArtifactRef(_Contract):
    artifact_id: str = Field(default_factory=lambda: new_id("artifact"))
    session_id: str
    label: str = Field(min_length=1, max_length=160)
    media_type: str = Field(min_length=1, max_length=128)
    format: str = Field(min_length=1, max_length=32)
    local_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)
    downloadable: bool = True
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        normalized = value.lower()
        if any(char not in "0123456789abcdef" for char in normalized):
            raise ValueError("artifact checksum must be SHA-256 hex")
        return normalized


class PreviewDescriptor(_Contract):
    preview_id: str = Field(default_factory=lambda: new_id("preview"))
    label: str = Field(min_length=1, max_length=160)
    kind: str = Field(min_length=1, max_length=64)
    row_count: int | None = Field(default=None, ge=0)
    column_count: int | None = Field(default=None, ge=0)
    artifact_id: str | None = None
    description: str = Field(default="", max_length=500)


class LineageRef(_Contract):
    run_id: str | None = None
    manifest_path: str | None = None
    code_version: str | None = None
    source_fingerprint: str | None = None
    stage_ids: list[str] = Field(default_factory=list)


class WorkflowResultView(_Contract):
    result_id: str = Field(default_factory=lambda: new_id("result"))
    session_id: str
    workflow: WorkflowKind
    execution_status: ExecutionState
    validation_status: Status = Status.NOT_RUN
    release_verdict: Status | None = None
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    previews: list[PreviewDescriptor] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    lineage: LineageRef | None = None
    created_at: datetime = Field(default_factory=utc_now)


class Project(_Contract):
    project_id: str = Field(default_factory=lambda: new_id("project"))
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=1000)
    session_ids: list[str] = Field(default_factory=list)
    result_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Template(_Contract):
    template_id: str
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=1000)
    workflow: WorkflowKind
    version: str = Field(min_length=1, max_length=32)
    tags: list[str] = Field(default_factory=list)
    definition: dict[str, Any]
    built_in: bool = True


class UserPreferences(_Contract):
    locale: str = Field(default="en_US", min_length=2, max_length=32)
    default_row_count: int = Field(default=100, ge=1, le=1_000_000)
    default_export_format: str = Field(default="csv", pattern="^(csv|parquet)$")
    updated_at: datetime = Field(default_factory=utc_now)


class RuntimeSettingsView(_Contract):
    approved_output_root: str
    staging_root: str
    allowed_backends: list[str]
    max_rows_per_table: int
    minimum_fidelity_score: float
    guardrails_enforced: bool = True
    guardrail_policy_version: str
    guardrail_max_input_characters: int
    guardrail_max_output_characters: int
    local_model_host: str
    local_model_network_scope: str = "loopback_only"


class ArtifactRegistration(_Contract):
    path: str
    label: str = Field(min_length=1, max_length=160)
    media_type: str = Field(min_length=1, max_length=128)
    format: str = Field(min_length=1, max_length=32)
    downloadable: bool = True
