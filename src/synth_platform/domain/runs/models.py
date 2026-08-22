"""Source/run identity models shared by both isolated source tracks."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict


class SourceKind(str, Enum):
    DATABASE = "database"
    PDF = "pdf"


class RunStatus(str, Enum):
    CREATED = "created"
    CONNECTED = "connected"
    EXTRACTED = "extracted"
    QUALITY_PASSED = "quality_passed"
    TRAINED = "trained"
    EXPORTED = "exported"
    SOURCE_DISCONNECTED = "source_disconnected"
    GENERATED = "generated"
    VALIDATED = "validated"
    PUBLISHED = "published"
    FAILED = "failed"


class SourceRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    source_kind: SourceKind
    source_fingerprint: str
    status: RunStatus
    artifact_id: str | None = None
    dataset_id: str | None = None
    target_schema: str | None = None
