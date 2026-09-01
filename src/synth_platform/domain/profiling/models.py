"""Profiling domain models (pure typed contracts)."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from synth_platform.domain.generation.conditional import ConditionalModel
from synth_platform.domain.semantics.dependencies import DependencyGraph
from synth_platform.domain.relational.models import CardinalityModel


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SensitivityClass(str, Enum):
    PUBLIC = "public"
    QUASI_IDENTIFIER = "quasi_identifier"
    DIRECT_IDENTIFIER = "direct_identifier"
    FREE_TEXT = "free_text"


class NumericStatistics(_Base):
    minimum: float
    maximum: float
    mean: float
    quantiles: list[float]
    representation: str = "numeric"


class CategoricalStatistics(_Base):
    values: list[str]
    probabilities: list[float]


class ColumnProfile(_Base):
    table_name: str
    column_name: str
    physical_type: str
    semantic_type: str | None
    nullable: bool
    cardinality: int
    missing_rate: float
    sensitivity: SensitivityClass
    numeric: NumericStatistics | None = None
    categorical: CategoricalStatistics | None = None


class TableProfile(_Base):
    table_name: str
    row_count: int
    columns: dict[str, ColumnProfile] = Field(default_factory=dict)
    # W2 (RC-1/A-01): the within-table column dependency DAG and the compiled
    # conditional model per column. Generation draws columns in the DAG's
    # topological order, each conditioned on its parents' drawn values.
    dependencies: "DependencyGraph" = Field(default_factory=lambda: DependencyGraph())
    conditionals: dict[str, "ConditionalModel"] = Field(default_factory=dict)


class SamplingRecord(_Base):
    strategy: str = "deterministic"
    seed: int = 0
    population_count: int = 0
    sample_count: int = 0
    bias_warning: str = ""


class DataProfile(_Base):
    tables: dict[str, TableProfile] = Field(default_factory=dict)
    sampling: dict[str, SamplingRecord] = Field(default_factory=dict)
    child_per_parent: dict[str, float] = Field(default_factory=dict)
    cardinality_models: dict[str, CardinalityModel] = Field(default_factory=dict)
