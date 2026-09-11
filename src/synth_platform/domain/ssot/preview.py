"""Common live-preview shape for any SSOT input type."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SSOTPreviewItem(_Base):
    name: str
    kind: str
    record_count: int | None = None
    field_count: int | None = None
    relationship_count: int | None = None
    sample: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SSOTPreview(_Base):
    source_type: str
    source_name: str = ""
    record_count: int | None = None
    field_count: int | None = None
    entity_count: int | None = None
    relationship_count: int | None = None
    items: list[SSOTPreviewItem] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    privacy_warnings: list[str] = Field(default_factory=list)
