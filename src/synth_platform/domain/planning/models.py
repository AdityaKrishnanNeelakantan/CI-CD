"""Learning plan domain models (pure)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LearningTask(_Base):
    table: str
    column: str
    backend: str
    strategy: str
    reason: str
    decision_basis: Literal["heuristic", "benchmark", "policy_forced", "user_forced"] = "heuristic"


class LearningPlan(_Base):
    tasks: list[LearningTask] = Field(default_factory=list)
