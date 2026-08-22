"""Dependency evidence separated from enforced business rules."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DependencyStatus(str, Enum):
    INFERRED = "inferred"       # evidence only, not enforced
    CONFIRMED = "confirmed"     # approved for enforcement
    PERMITTED = "permitted"     # policy allows enforcement


class DependencyEvidence(_Base):
    parent_column: str
    child_column: str
    method: str = "heuristic"
    effect_size: float = 0.0
    confidence: float = 0.0
    stability: float = 0.0
    status: DependencyStatus = DependencyStatus.INFERRED


class SemanticProfile(_Base):
    dependencies: list[DependencyEvidence] = Field(default_factory=list)
