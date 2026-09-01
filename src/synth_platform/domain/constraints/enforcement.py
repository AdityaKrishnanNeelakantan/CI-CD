"""Pure dataframe-agnostic helpers for generation-time constraint decisions."""
from __future__ import annotations

from synth_platform.domain.constraints.models import ConstraintKind, ConstraintSet


def max_children_for(constraints: ConstraintSet, child_table: str) -> int | None:
    limits = [c.limit for c in constraints.constraints
              if c.kind == ConstraintKind.MAX_CHILDREN
              and c.table == child_table and c.limit is not None]
    return min(limits) if limits else None
