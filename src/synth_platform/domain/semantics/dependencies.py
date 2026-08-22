"""Column dependency DAG (pure).

Within a table, an attribute may depend on other attributes: `balance` on
`account_type`, `city` on `state`. Generation must draw a child AFTER its
parents and CONDITIONED on their drawn values. This module models that DAG and
its topological order. It is domain-generic: nodes are column names, edges are
"child depends on parent" — no business meaning is encoded here.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DependencyEdge(_Base):
    parent: str          # conditioning column
    child: str           # conditioned column
    strength: float = 0.0  # normalized mutual information in [0,1]


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
