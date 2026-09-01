"""Compiled relational graph and cardinality models."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from synth_platform.domain.generation.conditional import ConditionalModel


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GraphEdge(_Base):
    parent_table: str
    parent_column: str
    child_table: str
    child_column: str
    confirmed: bool = True


class CardinalityModel(_Base):
    values: list[int] = Field(default_factory=lambda: [0])
    probabilities: list[float] = Field(default_factory=lambda: [1.0])
    minimum: int = 0
    maximum: int | None = None
    mean: float = 0.0
    parent_context_column: str | None = None
    conditional_values: dict[str, list[int]] = Field(default_factory=dict)
    conditional_probabilities: dict[str, list[float]] = Field(default_factory=dict)


class CrossTableConditional(_Base):
    parent_table: str
    parent_key_column: str
    parent_attribute: str
    child_table: str
    child_fk_column: str
    child_attribute: str
    context_name: str
    strength: float = 0.0
    model: ConditionalModel


class RelationalPlan(_Base):
    levels: list[list[str]] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    self_references: list[str] = Field(default_factory=list)
    child_per_parent: dict[str, float] = Field(default_factory=dict)
    cardinality_models: dict[str, CardinalityModel] = Field(default_factory=dict)
    cross_table_conditionals: list[CrossTableConditional] = Field(default_factory=list)

    @property
    def order(self) -> list[str]:
        return [t for level in self.levels for t in level]

    def edges_into(self, child_table: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.child_table == child_table]
