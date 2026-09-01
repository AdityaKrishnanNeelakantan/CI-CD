"""Extraction quality report contracts."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ExtractionCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    passed: bool
    critical: bool = True
    metric: float | None = None
    threshold: float | None = None
    detail: str = ""


class ExtractionQualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checks: list[ExtractionCheck] = Field(default_factory=list)
    passed: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)
