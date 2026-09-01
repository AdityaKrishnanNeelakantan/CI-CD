"""Synthetic document rendering result DTOs."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DocumentRenderResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    document_id: str
    sha256: str
    page_count: int
    valid: bool
    bound_entity_ids: dict[str, str] = Field(default_factory=dict)
    checks: dict[str, bool] = Field(default_factory=dict)
