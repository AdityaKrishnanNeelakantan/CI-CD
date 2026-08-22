"""Typed constraint IR enforced during generation and validation."""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ConstraintKind(str, Enum):
    RANGE = "range"
    MAX_CHILDREN = "max_children"
    ENUM = "enum"
    UNIQUE = "unique"
    CONDITIONAL = "conditional"
    TEMPORAL = "temporal"
    ARITHMETIC = "arithmetic"


class ConstraintDefinition(_Base):
    kind: ConstraintKind
    table: str
    column: str | None = None
    columns: list[str] = Field(default_factory=list)
    minimum: float | None = None
    maximum: float | None = None
    limit: int | None = None
    allowed_values: list[Any] = Field(default_factory=list)
    when_column: str | None = None
    when_values: list[Any] = Field(default_factory=list)
    required_column: str | None = None
    earlier_column: str | None = None
    later_column: str | None = None
    expression: str | None = None
    tolerance: float = 1e-6
    inferred: bool = False
    confidence: float = 1.0
    source: str = "declared"


class ConstraintSet(_Base):
    constraints: list[ConstraintDefinition] = Field(default_factory=list)
