"""Column dependency DAG (pure).

Within a table, an attribute may depend on other attributes: `balance` on
`account_type`, `city` on `state`. Generation must draw a child AFTER its
parents and CONDITIONED on their drawn values. This module models that DAG and
its topological order. It is domain-generic: nodes are column names, edges are
"child depends on parent" — no business meaning is encoded here.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DependencyEdge(_Base):
    parent: str          # conditioning column
    child: str           # conditioned column
    strength: float = 0.0  # normalized mutual information in [0,1]


class DependencyKind(str, Enum):
    CORRELATION = "correlation"
    MUTUAL_INFORMATION = "mutual_information"
    FUNCTIONAL = "functional"
    CONDITIONAL = "conditional"
    NULL_PATTERN = "null_pattern"
    CROSS_TABLE = "cross_table"
    TEMPORAL = "temporal"
    CARDINALITY = "cardinality"
    FK_EVIDENCE = "fk_evidence"


class DependencyStatus(str, Enum):
    EVIDENCE = "evidence"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    REJECTED = "rejected"


class FieldRef(_Base):
    table: str
    column: str

    @property
    def field_id(self) -> str:
        return f"field:{self.table}.{self.column}"


class DependencyEvidenceItem(_Base):
    method: str
    score: float = 0.0
    sample_size: int = 0
    description: str = ""


class LearnedDependency(_Base):
    dependency_id: str
    kind: DependencyKind
    source_fields: list[FieldRef] = Field(default_factory=list)
    target_fields: list[FieldRef] = Field(default_factory=list)
    confidence: float = 0.0
    status: DependencyStatus = DependencyStatus.EVIDENCE
    evidence: list[DependencyEvidenceItem] = Field(default_factory=list)
    provenance: list[str] = Field(default_factory=list)


class DependencyProfile(_Base):
    """Portable dependency evidence consumed by planning, artifacts, and validation."""
    format_version: str = "1.0.0"
    dependencies: list[LearnedDependency] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def approved(self) -> list[LearnedDependency]:
        return [dep for dep in self.dependencies if dep.status == DependencyStatus.APPROVED]

    def for_table(self, table: str) -> list[LearnedDependency]:
        return [
            dep for dep in self.dependencies
            if any(ref.table == table for ref in [*dep.source_fields, *dep.target_fields])
        ]


class DependencyGraph(_Base):
    """Per-table column dependency DAG. `roots` (no parents) are drawn first
    from their marginal; every other column is drawn conditioned on its parents."""
    edges: list[DependencyEdge] = Field(default_factory=list)

    def parents_of(self, column: str) -> list[str]:
        return [e.parent for e in self.edges if e.child == column]

    def topological_order(self, columns: list[str]) -> list[str]:
        """Columns ordered so every parent precedes its children. Cycles cannot
        occur (inference only adds an edge to an already-ordered predecessor),
        but a stable fallback guards against malformed input."""
        parents = {c: set(self.parents_of(c)) for c in columns}
        ordered: list[str] = []
        placed: set[str] = set()
        remaining = list(columns)
        while remaining:
            progressed = False
            for c in list(remaining):
                if parents[c] <= placed:
                    ordered.append(c)
                    placed.add(c)
                    remaining.remove(c)
                    progressed = True
            if not progressed:  # malformed (cycle) — emit rest stably, drop edges
                ordered.extend(remaining)
                break
        return ordered
