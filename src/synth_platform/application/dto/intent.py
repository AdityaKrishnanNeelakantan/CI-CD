"""Workflow intent DTOs shared across API and classifiers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

IntentWorkflowType = Literal["schema_twin", "database_twin", "document_twin", "interaction_twin", "unknown"]
IntentNextAction = Literal["upload_required", "configure_required", "ready_to_generate", "choose_workflow", "unsupported"]


class WorkflowIntentAttachment(BaseModel):
    filename: str | None = None
    content_type: str | None = None
    size_bytes: int | None = None
    extension: str | None = None
    upload_id: str | None = None
    path: str | None = None
    content_preview: str | None = None


class WorkflowIntentRequest(BaseModel):
    message: str = ""
    attachments: list[WorkflowIntentAttachment] = Field(default_factory=list)
    allow_auto_start: bool = False


class DetectedWorkflowInput(BaseModel):
    kind: str = "unknown"
    file_type: str | None = None
    source: str = "prompt"


class WorkflowIntentPrefill(BaseModel):
    record_count: int | None = None
    privacy_level: str | None = None
    output_format: str | None = None
    interaction_type: str | None = None
    document_type: str | None = None
    other: dict[str, Any] = Field(default_factory=dict)


class WorkflowIntentAlternative(BaseModel):
    workflow_type: IntentWorkflowType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class WorkflowIntentResponse(BaseModel):
    workflow_type: IntentWorkflowType
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    suggested_route: str
    can_auto_start: bool
    next_action: IntentNextAction
    detected_input: DetectedWorkflowInput
    prefill: WorkflowIntentPrefill = Field(default_factory=WorkflowIntentPrefill)
    alternatives: list[WorkflowIntentAlternative] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
